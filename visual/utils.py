import torch
from torch.utils.data import DataLoader
import numpy as np
import imageio
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from helpers.utils import is_main_process, get_rank, get_world_size


def _get_image_write_workers(H):
    workers = int(getattr(H, 'image_write_workers', 0) or 0)
    if workers > 0:
        return workers
    cpu_count = os.cpu_count() or 4
    return max(4, min(16, cpu_count))


def _write_png(path_and_img):
    path, img = path_and_img
    imageio.imwrite(path, img)


def _parallel_write_pngs(path_and_imgs, max_workers):
    if not path_and_imgs:
        return
    if max_workers <= 1:
        for item in path_and_imgs:
            _write_png(item)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Materialize to propagate worker exceptions before proceeding.
        list(executor.map(_write_png, path_and_imgs, chunksize=16))

def delete_content_of_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    for x in DataLoader(data, batch_size=num):
        break
    orig_image = (x[0]).to(torch.uint8).permute(0, 2, 3, 1) if dataset == 'lsun' else x[0]
    preprocessed = preprocess_fn(x)[0]
    preprocessed = preprocessed.cpu().numpy()
    return orig_image, preprocessed



def generate_for_NN(sampler, orig, initial, shape, ema_imle, fname, logprint, condition=None):
    mb = shape[0]
    initial = initial[:mb].to(ema_imle.device)
    cond = condition[:mb].to(ema_imle.device) if condition is not None else None
    nns = sampler.sample(initial, ema_imle, None, condition=cond)
    batches = [orig[:mb], nns]
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def generate_visualization(H, sampler, orig, initial, last_latents, latent_for_visualization, shape, imle, fname, logprint, experiment=None, latent_labels=None):
    mb = shape[0]
    initial = initial[:mb]
    last_latents = last_latents[:mb]
    latent_rows = [initial, last_latents] + [latent_for_visualization[t] for t in range(H.num_rows_visualize)]

    # Rows can arrive from mixed sources (CPU tensors, CUDA tensors, or NumPy arrays).
    # Normalize to a single device before concatenation to avoid device mismatch errors.
    cat_device = getattr(sampler, 'device', imle.device)
    normalized_rows = []
    for row in latent_rows:
        if not torch.is_tensor(row):
            row = torch.as_tensor(row)
        normalized_rows.append(row[:mb].detach().to(cat_device))
    batches = [orig[:mb]]
    for idx, row in enumerate(normalized_rows):
        cond = latent_labels[idx][:mb].to(cat_device) if latent_labels is not None else None
        batches.append(sampler.sample(row, imle, None, condition=cond))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
    if(experiment):
        experiment.log_image(fname, overwrite=True)
        # experiment.log_image(image_data=im, name="latest.png")


def generate_and_save(H, imle, sampler, n_samp, subdir='fid'):
    # Get the current process rank and world size.
    
    rank = get_rank()
    world_size = get_world_size()

    save_dir = os.path.join(H.save_dir, subdir)

    if is_main_process():
        delete_content_of_dir(save_dir)
    
    torch.distributed.barrier()

    indices = list(range(rank, n_samp, world_size))
    n_local = len(indices)
    write_workers = _get_image_write_workers(H)

    imle.eval()

    num_classes = getattr(H, 'num_classes', 0)
    ae_batch = getattr(H, 'ae_batch', H.imle_batch)
    with torch.inference_mode():
        # Process images in batches
        for i in range(0, n_local, ae_batch):
            current_batch_size = min(ae_batch, n_local - i)
            # Generate random latent vectors for the current batch
            latent_batch = torch.randn([current_batch_size, H.latent_dim], dtype=torch.float32,
                                       device=imle.device,
                                       generator=sampler.generator_seed)
            # latent_batch.normal_()  # Reinitialize latent_batch from normal distribution
            # For conditional models, assign class labels round-robin across the batch
            if num_classes > 0:
                condition = torch.tensor(
                    [indices[i + j] % num_classes for j in range(current_batch_size)],
                    dtype=torch.long, device=imle.device)
            else:
                condition = None
            # Generate samples using the provided sampler
            samp = sampler.sample(latent_batch, imle, None, condition=condition)
            path_and_imgs = []
            for j in range(current_batch_size):
                global_index = indices[i + j]
                path_and_imgs.append((os.path.join(save_dir, f'{global_index}.png'), samp[j]))
            _parallel_write_pngs(path_and_imgs, write_workers)
    
    imle.train()

def generate_and_save2(H, imle, sampler, n_samp, subdir='fid'):
    # Get the current process rank and world size.
    
    rank = get_rank()
    world_size = get_world_size()

    save_dir = os.path.join(H.save_dir, subdir)

    if is_main_process():
        delete_content_of_dir(save_dir)
    
    torch.distributed.barrier()

    indices = list(range(rank, n_samp, world_size))
    n_local = len(indices)
    write_workers = _get_image_write_workers(H)

    imle.eval()

    ae_batch = getattr(H, 'ae_batch', H.imle_batch)
    with torch.inference_mode():
        # Process images in batches
        for i in range(0, n_local, ae_batch):
            current_batch_size = min(ae_batch, n_local - i)
            # Generate random latent vectors for the current batch
            latent_batch = torch.randn([current_batch_size, H.latent_dim], dtype=torch.float32,
                                       device=imle.device,
                                       generator=sampler.generator_seed)
            # latent_batch.normal_()  # Reinitialize latent_batch from normal distribution
            # Generate samples using the provided sampler
            samp = sampler.sample_multi(latent_batch, imle, None)
            path_and_imgs = []
            for j in range(current_batch_size):
                global_index = indices[i + j]
                for k in range(2, len(samp)):
                    path_and_imgs.append((os.path.join(save_dir, f'{global_index}_{1 << k}.png'), samp[k][j]))
            _parallel_write_pngs(path_and_imgs, write_workers)

    imle.train()


def generate_activations(H, imle, sampler, n_samp):
    """Generate `n_samp` samples and return their Inception activations in memory.

    Unlike `generate_and_save`, this writes no PNGs to disk. Each rank generates
    its shard, runs it through Inception immediately, and the 2048-dim features
    are gathered onto rank 0. Returns an [n_samp, 2048] float64 array on rank 0
    and None on other ranks.
    """
    from evaluate import load_inception_detector, compute_activations_from_images

    rank = get_rank()
    world_size = get_world_size()
    device = imle.device

    detector_net = load_inception_detector(device)

    indices = list(range(rank, n_samp, world_size))
    n_local = len(indices)

    imle.eval()
    ae_batch = getattr(H, 'ae_batch', H.imle_batch)
    local_feats = []
    with torch.inference_mode():
        for i in range(0, n_local, ae_batch):
            current_batch_size = min(ae_batch, n_local - i)
            latent_batch = torch.randn([current_batch_size, H.latent_dim], dtype=torch.float32,
                                       device=device, generator=sampler.generator_seed)
            # sampler.sample returns uint8 NHWC; Inception wants uint8 NCHW.
            samp = sampler.sample(latent_batch, imle, None, condition=None)
            samp = torch.from_numpy(samp).permute(0, 3, 1, 2).contiguous()
            local_feats.append(compute_activations_from_images(samp, detector_net, device))
    imle.train()

    local_feats = np.concatenate(local_feats, axis=0) if local_feats else np.zeros((0, 2048), dtype=np.float64)

    if world_size == 1:
        return local_feats

    # Gather variable-length per-rank feature arrays onto rank 0.
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_feats)
    if not is_main_process():
        return None
    return np.concatenate(gathered, axis=0)


def compute_reference_activations(H, real_image_path, device, cache_path=None):
    """Compute (and cache) Inception activations for the real ImageFolder dataset.

    Runs on the calling (rank-0) process only. If `cache_path` exists it is
    loaded; otherwise the real images are pushed through Inception once and the
    resulting [N, 2048] activations are saved to a single .npz file for reuse.
    """
    from evaluate import load_inception_detector, compute_activations_from_images
    from training import dataset as _ds_module

    if cache_path is not None and os.path.exists(cache_path):
        with np.load(cache_path) as data:
            return data['feat']

    detector_net = load_inception_detector(device)
    dataset_obj = _ds_module.ImageFolderDataset(path=real_image_path)
    ref_batch = getattr(H, 'ae_batch', H.imle_batch)
    data_loader = DataLoader(dataset_obj, batch_size=ref_batch, num_workers=4)

    feats = []
    with torch.inference_mode():
        for images, _labels in data_loader:
            feats.append(compute_activations_from_images(images, detector_net, device))
    feats = np.concatenate(feats, axis=0)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.savez(cache_path, feat=feats)
    return feats
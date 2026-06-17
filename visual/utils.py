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

def _patch_max_sharpness_batch(imgs_uint8_rgb, patch_size):
    """
    Vectorized reproduction of filter_blurry_samples_patch.py:patch_sharpness_score.

    Args:
        imgs_uint8_rgb: ndarray of shape (B, H, W, 3), dtype uint8.
        patch_size: non-overlapping patch side length.

    Returns:
        ndarray of shape (B,), float32. Per-image: max over non-overlapping patches
        of the variance of the Laplacian (kernel [[0,1,0],[1,-4,1],[0,1,0]], reflect
        boundary) of the grayscale image. Matches PIL convert('L') + scipy.ndimage.laplace.
    """
    from PIL import Image
    from scipy.ndimage import laplace

    B = imgs_uint8_rgb.shape[0]
    out = np.empty(B, dtype=np.float32)
    for i in range(B):
        gray = np.array(Image.fromarray(imgs_uint8_rgb[i]).convert('L'), dtype=np.float32)
        lap = laplace(gray)
        h, w = lap.shape
        nh, nw = h // patch_size, w // patch_size
        if nh == 0 or nw == 0:
            out[i] = float(lap.var())
            continue
        # Crop to multiple of patch_size, then reshape into (nh, nw, p, p) and take per-patch var.
        lap = lap[:nh * patch_size, :nw * patch_size]
        patches = lap.reshape(nh, patch_size, nw, patch_size).transpose(0, 2, 1, 3)
        patch_vars = patches.reshape(nh * nw, patch_size * patch_size).var(axis=1)
        out[i] = float(patch_vars.max())
    return out


def generate_and_save_smart(H, imle, sampler, n_samp, subdir='fid'):
    """
    Like generate_and_save, but rejects samples whose patch-max grayscale Laplacian
    variance score falls below H.reject_threshold and re-samples (new latent, same
    class) up to H.reject_max_attempts times. After exhausting attempts, writes the
    highest-scoring attempt seen.
    """
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
    threshold = float(H.reject_threshold)
    patch_size = int(H.reject_patch_size)
    max_attempts = int(H.reject_max_attempts)

    # pending[k] = (global_index, attempts_so_far, best_score, best_img_or_None)
    pending = [(idx, 0, -np.inf, None) for idx in indices]
    accepted_count = 0
    exhausted_count = 0
    total_attempts = 0
    log_every = max(1, n_local // 20)
    next_log_at = log_every

    with torch.inference_mode():
        while pending:
            cur = pending[:ae_batch]
            pending = pending[ae_batch:]
            cur_bs = len(cur)

            latent_batch = torch.randn(
                [cur_bs, H.latent_dim], dtype=torch.float32,
                device=imle.device, generator=sampler.generator_seed,
            )
            if num_classes > 0:
                condition = torch.tensor(
                    [g_idx % num_classes for (g_idx, _, _, _) in cur],
                    dtype=torch.long, device=imle.device,
                )
            else:
                condition = None

            samp = sampler.sample(latent_batch, imle, None, condition=condition)
            scores = _patch_max_sharpness_batch(samp, patch_size)
            total_attempts += cur_bs

            path_and_imgs = []
            for j in range(cur_bs):
                g_idx, attempts, best_score, best_img = cur[j]
                attempts += 1
                score = float(scores[j])
                if score > best_score:
                    best_score = score
                    best_img = samp[j]

                if score >= threshold:
                    path_and_imgs.append((os.path.join(save_dir, f'{g_idx}.png'), samp[j]))
                    accepted_count += 1
                elif attempts >= max_attempts:
                    path_and_imgs.append((os.path.join(save_dir, f'{g_idx}.png'), best_img))
                    accepted_count += 1
                    exhausted_count += 1
                else:
                    pending.append((g_idx, attempts, best_score, best_img))

            _parallel_write_pngs(path_and_imgs, write_workers)

            if accepted_count >= next_log_at and is_main_process():
                acc_rate = accepted_count / total_attempts if total_attempts else 0.0
                print(f'[eval_fid_smart] rank0 progress: {accepted_count}/{n_local} accepted '
                      f'({total_attempts} attempts, accept_rate={acc_rate:.3f}, '
                      f'exhausted={exhausted_count})')
                next_log_at += log_every

    if is_main_process():
        acc_rate = accepted_count / total_attempts if total_attempts else 0.0
        print(f'[eval_fid_smart] rank0 done: {accepted_count}/{n_local} written '
              f'({total_attempts} total attempts, accept_rate={acc_rate:.3f}, '
              f'{exhausted_count} written from best-of-attempts after exhaustion)')

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
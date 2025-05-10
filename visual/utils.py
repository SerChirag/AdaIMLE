import torch
from torch.utils.data import DataLoader
import numpy as np
import imageio
import os
import shutil
from helpers.utils import is_main_process, get_rank, get_world_size

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
    orig_image = (x[0] * 255.0).to(torch.uint8).permute(0, 2, 3, 1) if dataset == 'ffhq_1024' else x[0]
    preprocessed = preprocess_fn(x)[0]
    return orig_image, preprocessed



def generate_for_NN(sampler, orig, initial, shape, ema_imle, fname, logprint):
    mb = shape[0]
    initial = initial[:mb].to(ema_imle.device)
    nns = sampler.sample(initial, ema_imle, None)
    batches = [orig[:mb], nns]
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def generate_visualization(H, sampler, orig, initial, last_latents, latent_for_visualization, shape, imle, fname, logprint, experiment=None):
    mb = shape[0]
    initial = initial[:mb]
    last_latents = last_latents[:mb]
    batches = [orig[:mb], sampler.sample(initial, imle, None), sampler.sample(last_latents, imle, None)]

    for t in range(H.num_rows_visualize):
        batches.append(sampler.sample(latent_for_visualization[t], imle, None))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
    if(experiment):
        experiment.log_image(fname, overwrite=True)

def generate_and_save(H, imle, sampler, n_samp, subdir='fid'):
    # Get the current process rank and world size.
    
    rank = get_rank()
    world_size = get_world_size()

    if is_main_process():
        delete_content_of_dir(f'{H.save_dir}/{subdir}')
    
    torch.distributed.barrier()

    indices = list(range(rank, n_samp, world_size))
    n_local = len(indices)

    imle.eval()

    with torch.no_grad():
        # Process images in batches
        for i in range(0, n_local, H.imle_batch):
            current_batch_size = min(H.imle_batch, n_local - i)
            # Generate random latent vectors for the current batch
            latent_batch = torch.randn([current_batch_size, H.latent_dim], dtype=torch.float32, 
                                       device=imle.device, 
                                       generator=sampler.generator_seed)
            # latent_batch.normal_()  # Reinitialize latent_batch from normal distribution
            # Generate samples using the provided sampler
            samp = sampler.sample(latent_batch, imle, None)
            # Save each sample with its corresponding global index
            for j in range(current_batch_size):
                global_index = indices[i + j]
                imageio.imwrite(f'{H.save_dir}/{subdir}/{global_index}.png', samp[j])
    
    imle.train()
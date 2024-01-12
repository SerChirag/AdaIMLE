import torch
from torch.utils.data import DataLoader
import numpy as np
import imageio
import os
import shutil


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



def generate_for_NN(sampler, orig, initial, snoise, shape, ema_imle, fname, logprint):
    mb = shape[0]
    initial = initial[:mb].cuda()
    nns = sampler.sample(initial, ema_imle, snoise)
    batches = [orig[:mb], nns]
    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def generate_images_initial(H, sampler, orig, initial, snoise, shape, imle, ema_imle, fname, logprint, experiment=None):
    mb = shape[0]
    initial = initial[:mb]
    batches = [orig[:mb], sampler.sample(initial, imle, snoise)]

    temp_latent_rnds = torch.randn([mb, H.latent_dim], dtype=torch.float32).cuda()
    for t in range(H.num_rows_visualize):
        temp_latent_rnds.normal_()
        if(H.use_snoise == True):
            tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
        else:
            tmp_snoise = [s[:mb] for s in sampler.neutral_snoise]
        batches.append(sampler.sample(temp_latent_rnds, imle, tmp_snoise))

    if(H.use_snoise == True):
        tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
    else:
        tmp_snoise = [s[:mb] for s in sampler.neutral_snoise]
    batches.append(sampler.sample(temp_latent_rnds, imle, tmp_snoise))

    if(H.use_snoise == True):
        tmp_snoise = [s[:mb].normal_() for s in sampler.snoise_tmp]
    else:
        tmp_snoise = [s[:mb] for s in sampler.neutral_snoise]
    batches.append(sampler.sample(temp_latent_rnds, imle, tmp_snoise))

    tmp_snoise = [s[:mb] for s in sampler.neutral_snoise]
    batches.append(sampler.sample(temp_latent_rnds, imle, tmp_snoise))

    tmp_snoise = [s[:mb] for s in sampler.neutral_snoise]
    temp_latent_rnds.normal_()
    batches.append(sampler.sample(temp_latent_rnds, imle, tmp_snoise))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
    if(experiment):
        experiment.log_image(fname, overwrite=True)

def generate_and_save(H, imle, sampler, n_samp, subdir='fid'):

    delete_content_of_dir(f'{H.save_dir}/{subdir}')

    phi = 0.9
    with torch.no_grad():
        temp_latent_rnds = torch.randn([H.imle_batch, H.latent_dim], dtype=torch.float32).cuda()
        for i in range(0, n_samp, H.imle_batch):

            if(i + H.imle_batch > n_samp):
                batch_size = n_samp - i
                print("batch size: ", batch_size)
            else:
                batch_size = H.imle_batch

            temp_latent_rnds.normal_()

            if(H.use_snoise == True):
                tmp_snoise = [s[:H.imle_batch].normal_() for s in sampler.snoise_tmp]
            else:
                tmp_snoise = [s[:H.imle_batch] for s in sampler.neutral_snoise]

            # tmp_snoise = [s[:H.imle_batch].normal_() for s in sampler.snoise_tmp]
            # z_mean = torch.mean(temp_latent_rnds,axis=0)
            # snoise_mean = []
            # for j in tmp_snoise:
            #     snoise_mean.append(torch.unsqueeze(torch.mean(j,axis=0),0))

            # trunc_latents_og = temp_latent_rnds * phi + z_mean * (1 - phi)
            # snoise_trunc_og = []
            # for j in range(len(tmp_snoise)):
            #     snoise_trunc_og.append(tmp_snoise[j] * phi + snoise_mean[j] * (1 - phi))

            # samp = sampler.sample(trunc_latents_og, imle, snoise_trunc_og)
            samp = sampler.sample(temp_latent_rnds, imle, tmp_snoise)

            for j in range(batch_size):
                imageio.imwrite(f'{H.save_dir}/{subdir}/{i + j}.png', samp[j])


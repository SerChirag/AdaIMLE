import os
import time

from comet_ml import Experiment, ExistingExperiment
import imageio
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from cleanfid import fid
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from data import set_up_data
from helpers.imle_helpers import backtrack, reconstruct
from helpers.train_helpers import (load_imle, load_opt, save_latents,
                                   save_latents_latest, save_model,
                                   save_snoise, set_up_hyperparams, update_ema)
from helpers.utils import ZippedDataset, get_cpu_stats_over_ranks
from metrics.ppl import calc_ppl
from metrics.ppl_uniform import calc_ppl_uniform
from sampler import Sampler
from visual.generate_rnd import generate_rnd
from visual.generate_rnd_nn import generate_rnd_nn
from visual.generate_sample_nn import generate_sample_nn
from visual.generate_video import generate_video
from visual.interpolate import random_interp
from visual.nn_interplate import nn_interp
from visual.spatial_visual import spatial_vissual
from visual.utils import (generate_and_save, generate_for_NN,
                          generate_images_initial,
                          get_sample_for_visualization)
from helpers.improved_precision_recall import compute_prec_recall
from torch.cuda.amp import autocast
import torch.distributed as dist

import os
import torch.distributed as dist

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        print(f"Initializing process group: rank {rank}/{world_size} on GPU {local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    else:
        print("Not running in distributed mode.")


def training_step_imle(H, n, targets, latents, imle, ema_imle, optimizer, loss_fn, scaler):
    t0 = time.time()
    imle.zero_grad()

    cur_batch_latents = latents
    
    # torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    with torch.amp.autocast('cuda', dtype=torch.float16):

        px_z = imle(cur_batch_latents)
        loss_256 = loss_fn(px_z, targets.permute(0, 3, 1, 2))
        loss = loss_256
        num_resolutions = 1

        if(H.use_multi_res):
            px_z_16 = F.interpolate(px_z, scale_factor = 0.0625, antialias=True, mode='bicubic')
            px_z_32 = F.interpolate(px_z, scale_factor = 0.125, antialias=True, mode='bicubic')
            px_z_64 = F.interpolate(px_z, scale_factor = 0.25, antialias=True, mode='bicubic')
            px_z_128 = F.interpolate(px_z, scale_factor = 0.5, antialias=True, mode='bicubic')

            targets_16 = F.interpolate(targets.permute(0, 3, 1, 2), scale_factor = 0.0625, antialias=True, mode='bicubic')
            targets_32 = F.interpolate(targets.permute(0, 3, 1, 2), scale_factor = 0.125, antialias=True, mode='bicubic')
            targets_64 = F.interpolate(targets.permute(0, 3, 1, 2), scale_factor = 0.25, antialias=True, mode='bicubic')
            targets_128 = F.interpolate(targets.permute(0, 3, 1, 2), scale_factor = 0.5, antialias=True, mode='bicubic')

            loss_16 = loss_fn(px_z_16, targets_16, only_l2 = True)
            loss_32 = loss_fn(px_z_32, targets_32)
            loss_64 = loss_fn(px_z_64, targets_64)
            loss_128 = loss_fn(px_z_128, targets_128)
            loss += loss_16 + loss_32 + loss_64 + loss_128
            num_resolutions = 5

            for scale in H['multi_res_scales']:
                px_z_scale = F.interpolate(px_z, scale_factor = scale, antialias=True, mode='bicubic')
                targets_scale = F.interpolate(targets.permute(0, 3, 1, 2), scale_factor = scale, antialias=True, mode='bicubic')
                if(px_z_scale.shape[2] < 32):
                    loss_scale = loss_fn(px_z_scale, targets_scale, only_l2 = True)
                else:
                    loss_scale = loss_fn(px_z_scale, targets_scale)
                loss += loss_scale
                num_resolutions += 1

    loss = loss / num_resolutions
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()  
    if ema_imle is not None:
        update_ema(imle, ema_imle, H.ema_rate)



def train_loop_imle(H, data_train, data_valid, preprocess_fn, imle, ema_imle, logprint, experiment=None):
    subset_len = len(data_train)
    if H.subset_len != -1:
        subset_len = H.subset_len
    for data_train in DataLoader(data_train, batch_size=subset_len):
        data_train = TensorDataset(data_train[0])
        break

    optimizer, scheduler, _, iterate, starting_epoch = load_opt(H, imle, logprint)
    print("Starting epoch: ", starting_epoch)
    print("Starting iteration: ", iterate)

    stats = []
    H.ema_rate = torch.as_tensor(H.ema_rate)

    subset_len = H.subset_len if H.subset_len != -1 else len(data_train)

    sampler = Sampler(H, subset_len, preprocess_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    last_updated = torch.zeros(subset_len, dtype=torch.int16)
    times_updated = torch.zeros(subset_len, dtype=torch.int8)
    change_thresholds = torch.empty(subset_len)
    change_thresholds[:] = H.change_threshold
    best_fid = 100000
    epoch = starting_epoch - 1

    split_ind = 0
    split_x_tensor = data_train.tensors[0].pin_memory()
    split_x = TensorDataset(split_x_tensor)
    sampler.init_projection(split_x_tensor)
    viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn, H.num_images_visualize, H.dataset)

    while (epoch < H.num_epochs):
        epoch += 1

        # Update the IMLE force resampling every imle_force_resample epochs.
        if epoch % H.imle_force_resample == 0:
            sampler.imle_sample_force(split_x_tensor, imle)

        if (epoch % 5 == 0):
            latents = sampler.selected_latents[:H.num_images_visualize]
            with torch.no_grad():
                generate_for_NN(sampler, split_x_tensor[:H.num_images_visualize], latents,
                                viz_batch_original.shape, imle,
                                f'{H.save_dir}/NN-samples_{epoch}-imle.png', logprint)

        # Create a dataset that pairs images with their current latents.
        comb_dataset = ZippedDataset(split_x, TensorDataset(sampler.selected_latents))

        # Use a DistributedSampler if in distributed training.
        if torch.distributed.is_initialized():
            train_sampler = DistributedSampler(comb_dataset, shuffle=True)
            data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, sampler=train_sampler,
                                     pin_memory=True, num_workers=4, persistent_workers=True)
        else:
            data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, shuffle=True,
                                     pin_memory=True, num_workers=4, persistent_workers=True)

        # If using distributed sampler, set the epoch for shuffling
        if torch.distributed.is_initialized():
            train_sampler.set_epoch(epoch)

        start_time = time.time()

        # Main training loop.
        for cur, indices in data_loader:
            x = cur[0]
            latents = cur[1][0]
            _, target = preprocess_fn(x)
            target = target.to(device)
            latents = latents.to(device)

            training_step_imle(H, target.shape[0], target, latents, imle, ema_imle,
                               optimizer, sampler.calc_loss, sampler.scaler)
            scheduler.step()

            # Save or log only on rank 0.
            # if iterate % H.iters_per_images == 0:
            #     with torch.no_grad():
            #         generate_images_initial(H, sampler, viz_batch_original,
            #                                 sampler.selected_latents[0: H.num_images_visualize],
            #                                 sampler.last_selected_latents[0: H.num_images_visualize],
            #                                 viz_batch_original.shape, imle, ema_imle,
            #                                 f'{H.save_dir}/samples-{iterate}.png', logprint, experiment)
            iterate += 1
            if iterate % H.iters_per_save == 0 and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                fp = os.path.join(H.save_dir, 'latest')
                logprint(f'Saving model@ {iterate} to {fp}')
                save_model(fp, imle, ema_imle, optimizer, scheduler, H)
            if iterate % H.iters_per_ckpt == 0 and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), imle, ema_imle, optimizer, scheduler, H)

        print(f'Epoch {epoch} took {time.time() - start_time} seconds')

        if epoch % 5 == 0:
            cur_dists = torch.empty([subset_len], dtype=torch.float32, device='cuda')
            cur_dists_lpips = torch.empty([subset_len], dtype=torch.float32, device='cuda')
            cur_dists_l2 = torch.empty([subset_len], dtype=torch.float32, device='cuda')
            cur_dists[:], cur_dists_lpips[:], cur_dists_l2[:] = sampler.calc_dists_existing(
                split_x_tensor, imle, dists=cur_dists, dists_lpips=cur_dists_lpips, dists_l2=cur_dists_l2, logging=True)
            metrics = {
                'mean_loss': torch.mean(cur_dists).item(),
                'std_loss': torch.std(cur_dists).item(),
                'max_loss': torch.max(cur_dists).item(),
                'min_loss': torch.min(cur_dists).item(),
                'mean_loss_lpips': torch.mean(cur_dists_lpips).item(),
                'std_loss_lpips': torch.std(cur_dists_lpips).item(),
                'max_loss_lpips': torch.max(cur_dists_lpips).item(),
                'min_loss_lpips': torch.min(cur_dists_lpips).item(),
                'mean_loss_l2': torch.mean(cur_dists_l2).item(),
                'std_loss_l2': torch.std(cur_dists_l2).item(),
                'max_loss_l2': torch.max(cur_dists_l2).item(),
                'min_loss_l2': torch.min(cur_dists_l2).item(),
                'total_excluded': sampler.total_excluded,
                'total_excluded_percentage': sampler.total_excluded_percentage,
            }
            logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)

        # Periodically compute FID and update model checkpoints (only from rank 0).
        # if (epoch > 0 and epoch % H.fid_freq == 0):
        #     print("Learning rate: ", optimizer.param_groups[0]['lr'])
        #     generate_and_save(H, imle, sampler, min(5000, subset_len * H.fid_factor))
        #     cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False)
        #     if cur_fid < best_fid and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        #         best_fid = cur_fid
        #         fp = os.path.join(H.save_dir, 'best_fid')
        #         logprint(f'Saving model best fid {best_fid} @ {iterate} to {fp}')
        #         save_model(fp, imle, ema_imle, optimizer, scheduler, H)

        #     precision, recall = compute_prec_recall(f'{H.data_root}/img', f'{H.save_dir}/fid/')
        #     metrics.update({'fid': cur_fid, 'best_fid': best_fid, 'precision': precision, 'recall': recall})

        if epoch % 50 == 0 and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            with torch.no_grad():
                generate_images_initial(H, sampler, viz_batch_original,
                                        sampler.selected_latents[0: H.num_images_visualize],
                                        sampler.last_selected_latents[0: H.num_images_visualize],
                                        viz_batch_original.shape, imle, ema_imle,
                                        f'{H.save_dir}/latest.png', logprint, experiment)

        if epoch % 5 == 0 and experiment is not None and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            experiment.log_metrics(metrics, epoch=epoch, step=iterate)



def main(H=None):
    H_cur, logprint = set_up_hyperparams()
    if not H:
        H = H_cur
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    imle, ema_imle = load_imle(H, logprint)

    if H.use_comet and H.comet_api_key:
        if(H.comet_experiment_key):
            print("Resuming experiment")
            experiment = ExistingExperiment(
                api_key=H.comet_api_key,
                previous_experiment=H.comet_experiment_key
            )
            experiment.log_parameters(H)

        else:
            experiment = Experiment(
                api_key=H.comet_api_key,
                project_name="adaptiveimle-ablation",
                workspace="serchirag",
            )
            experiment.set_name(H.comet_name)
            experiment.log_parameters(H)
    else:
        experiment = None

    os.makedirs(f'{H.save_dir}/fid', exist_ok=True)
    

    if H.mode == 'eval':
        
        os.makedirs(f'{H.save_dir}/eval', exist_ok=True)
        print(H)

        with torch.no_grad():
            # Generating
            sampler = Sampler(H, len(data_train), preprocess_fn)
            n_samp = H.n_batch
            temp_latent_rnds = torch.randn([n_samp, H.latent_dim], dtype=torch.float32).cuda()
            for i in range(0, H.num_images_to_generate // n_samp):
                if (i % 10 == 0):
                    print(i * n_samp)
                temp_latent_rnds.normal_()
                tmp_snoise = [s[:n_samp].normal_() for s in sampler.snoise_tmp]
                torch.save(temp_latent_rnds, f'{H.save_dir}/eval/temp_latent_rnds_{i}.pt')
                torch.save(tmp_snoise, f'{H.save_dir}/eval/tmp_snoise_{i}.pt')
                samp = sampler.sample(temp_latent_rnds, imle, tmp_snoise)
                for j in range(n_samp):
                    imageio.imwrite(f'{H.save_dir}/eval/{i * n_samp + j}.png', samp[j])

    elif H.mode == 'eval_fid':
        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        sampler = Sampler(H, len(data_train), preprocess_fn)
        # generate_and_save(H, imle, sampler, 5000)

        generate_and_save(H, imle, sampler, 5000)
        print(f'{H.data_root}/img', f'{H.save_dir}/fid/')
        cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False)
        print("FID: ", cur_fid)


    elif H.mode == 'reconstruct':

        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        ind = 0
        for split_ind, split_x_tensor in enumerate(DataLoader(data_train, batch_size=H.subset_len, pin_memory=True)):
            if (ind == 14):
                break
            split_x = TensorDataset(split_x_tensor[0])
            ind += 1
            
        for param in imle.parameters():
            param.requires_grad = False
        viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                H.num_images_visualize, H.dataset)
        if os.path.isfile(str(H.restore_latent_path)):
            latents = torch.tensor(torch.load(H.restore_latent_path), requires_grad=True)
        else:
            latents = torch.randn([viz_batch_original.shape[0], H.latent_dim], requires_grad=True)
        sampler = Sampler(H, subset_len, preprocess_fn)
        reconstruct(H, sampler, imle, preprocess_fn, viz_batch_original, latents, 'reconstruct', logprint, training_step_imle)

    elif H.mode == 'backtrack':
        for param in imle.parameters():
            param.requires_grad = False
        for split_x in DataLoader(data_train, batch_size=H.subset_len):
            split_x = split_x[0]
            pass
        print(f'split shape is {split_x.shape}')
        sampler = Sampler(H, H.subset_len, preprocess_fn)
        backtrack(H, sampler, imle, preprocess_fn, split_x, logprint, training_step_imle)


    elif H.mode == 'train':
        print(H)
        init_distributed()  # Initialize the process group early.
        if dist.is_initialized():
            imle = torch.nn.parallel.DistributedDataParallel(
                imle, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device()
            )
        train_loop_imle(H, data_train, data_valid_or_test, preprocess_fn, imle, ema_imle, logprint, experiment)

    elif H.mode == 'interpolate':
        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, subset_len, preprocess_fn)
            for i in range(H.num_images_to_generate):
                random_interp(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/interp-{i}.png', logprint)
    
    elif H.mode == 'generate_video':
        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, subset_len, preprocess_fn)
            generate_video(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/slerp.mp4', logprint)

        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        with torch.no_grad():
            for split_x in DataLoader(data_train, batch_size=subset_len):
                split_x = split_x[0]
            viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn,
                                                                    H.num_images_visualize, H.dataset)
            sampler = Sampler(H, subset_len, preprocess_fn)
            latents = torch.tensor(torch.load(f'{H.restore_latent_path}'), requires_grad=True, dtype=torch.float32, device='cuda')
            for i in range(latents.shape[0] - 1):
                lat0 = latents[i:i+1]
                lat1 = latents[i+1:i+2]
                sn1 = None
                sn2 = None
                random_interp(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/back-interp-{i}.png', logprint, lat0, lat1, sn1, sn2)

    elif H.mode == 'prec_rec':
        
        os.makedirs(f'{H.save_dir}/prec_rec', exist_ok=True)

        subset_len = H.subset_len
        if subset_len == -1:
            subset_len = len(data_train)
        sampler = Sampler(H, len(data_train), preprocess_fn)
        # generate_and_save(H, imle, sampler, 5000)

        print("Generating images")
        generate_and_save(H, imle, sampler, 1000, subdir='prec_rec')
        print(f'{H.data_root}/img', f'{H.save_dir}/prec_rec/')
        precision, recall = compute_prec_recall(f'{H.data_root}/img', f'{H.save_dir}/prec_rec/')
        print("Precision: ", precision)
        print("Recall: ", recall)


if __name__ == "__main__":
    main()

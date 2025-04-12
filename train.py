import os
import time

from comet_ml import Experiment, ExistingExperiment
import imageio
import torch
import wandb
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

def training_step_imle(H, n, targets, latents, imle, ema_imle, optimizer, loss_fn, accelerator):
    t0 = time.time()
    imle.zero_grad()

    cur_batch_latents = latents
    
    with accelerator.autocast():

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
    
    accelerator.backward(loss)
    optimizer.step()
    # if ema_imle is not None:
    #     update_ema(imle, ema_imle, H.ema_rate)


def train_loop_imle(H, data_train, data_valid, preprocess_fn, imle, ema_imle, logprint, experiment = None):
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

    subset_len = H.subset_len
    if subset_len == -1:
        subset_len = len(data_train)

    sampler = Sampler(H, subset_len, preprocess_fn)

    optimizer = sampler.accelerator.prepare_optimizer(optimizer)
    scheduler = sampler.accelerator.prepare_scheduler(scheduler)
    imle = sampler.accelerator.prepare_model(imle)

    best_fid = 100000
    epoch = starting_epoch - 1

    split_x_tensor = data_train.tensors[0].pin_memory()
    split_x = TensorDataset(split_x_tensor)
    sampler.init_projection(split_x_tensor)
    viz_batch_original, _ = get_sample_for_visualization(split_x, preprocess_fn, H.num_images_visualize, H.dataset)

    while (epoch < H.num_epochs):
        
        epoch += 1
       
        if epoch % H.imle_force_resample == 0:
            sampler.imle_sample_force(split_x_tensor, imle)

        if (epoch % 5 == 0):
            latents = sampler.selected_latents[:H.num_images_visualize]
            with torch.no_grad():
                generate_for_NN(sampler, split_x_tensor[:H.num_images_visualize], latents,
                                viz_batch_original.shape, imle,
                                f'{H.save_dir}/NN-samples_{epoch}-imle.png', logprint)

    
        comb_dataset = ZippedDataset(split_x, TensorDataset(sampler.selected_latents))
        data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, pin_memory=True, shuffle=True, num_workers=4, persistent_workers=True)

        data_loader = sampler.accelerator.prepare_data_loader(data_loader)
        sampler.accelerator.wait_for_everyone()
        
        start_time = time.time()

        for cur, indices in data_loader:
            x = cur[0]
            latents = cur[1][0]
            _, target = preprocess_fn(x)

            target = target.to(sampler.accelerator.device)
            latents = latents.to(sampler.accelerator.device)
            
            training_step_imle(H, target.shape[0], target, latents, imle, ema_imle, optimizer, sampler.calc_loss, sampler.scaler)

            scheduler.step()

            if iterate % H.iters_per_images == 0:
                with torch.no_grad():
                    generate_images_initial(H, sampler, viz_batch_original,
                                            sampler.selected_latents[0: H.num_images_visualize],
                                            sampler.last_selected_latents[0: H.num_images_visualize],
                                            viz_batch_original.shape, imle, ema_imle,
                                            f'{H.save_dir}/samples-{iterate}.png', logprint, experiment)

            iterate += 1
            if iterate % H.iters_per_save == 0:
                fp = os.path.join(H.save_dir, 'latest')
                logprint(f'Saving model@ {iterate} to {fp}')
                save_model(fp, imle, ema_imle, optimizer, scheduler, H)

            if iterate % H.iters_per_ckpt == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), imle, ema_imle, optimizer, scheduler, H)

        print(f'Epoch {epoch} took {time.time() - start_time} seconds')

        # if epoch % 5 == 0:
            
        #     cur_dists = torch.empty([subset_len], dtype=torch.float32, device='cuda')
        #     cur_dists_lpips = torch.empty([subset_len], dtype=torch.float32, device='cuda')
        #     cur_dists_l2 = torch.empty([subset_len], dtype=torch.float32, device='cuda')


        #     cur_dists[:], cur_dists_lpips[:], cur_dists_l2[:] = sampler.calc_dists_existing(split_x_tensor, imle, 
        #                                                                                     dists=cur_dists,  
        #                                                                                     dists_lpips=cur_dists_lpips,
        #                                                                                     dists_l2=cur_dists_l2, 
        #                                                                                     logging=True)
                    
        #     metrics = {
        #         'mean_loss': torch.mean(cur_dists).item(),
        #         'std_loss': torch.std(cur_dists).item(),
        #         'max_loss': torch.max(cur_dists).item(),
        #         'min_loss': torch.min(cur_dists).item(),
        #         'mean_loss_lpips': torch.mean(cur_dists_lpips).item(),
        #         'std_loss_lpips': torch.std(cur_dists_lpips).item(),
        #         'max_loss_lpips': torch.max(cur_dists_lpips).item(),
        #         'min_loss_lpips': torch.min(cur_dists_lpips).item(),
        #         'mean_loss_l2': torch.mean(cur_dists_l2).item(),
        #         'std_loss_l2': torch.std(cur_dists_l2).item(),
        #         'max_loss_l2': torch.max(cur_dists_l2).item(),
        #         'min_loss_l2': torch.min(cur_dists_l2).item(),
        #         'total_excluded': sampler.total_excluded,
        #         'total_excluded_percentage': sampler.total_excluded_percentage,
        #     }
            
        #     logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)

        # if (epoch > 0 and epoch % H.fid_freq == 0):
        #     print("Learning rate: ", optimizer.param_groups[0]['lr'])
        #     generate_and_save(H, imle, sampler, min(5000,subset_len * H.fid_factor))
        #     print(f'{H.data_root}/img', f'{H.save_dir}/fid/')
        #     cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False)
        #     if cur_fid < best_fid:
        #         best_fid = cur_fid
        #         # save models
        #         fp = os.path.join(H.save_dir, 'best_fid')
        #         logprint(f'Saving model best fid {best_fid} @ {iterate} to {fp}')
        #         save_model(fp, imle, ema_imle, optimizer, scheduler, H)
            
        #     precision, recall = compute_prec_recall(f'{H.data_root}/img', f'{H.save_dir}/fid/')

        #     metrics['fid'] = cur_fid
        #     metrics['best_fid'] = best_fid
        #     metrics['precision'] = precision
        #     metrics['recall'] = recall
            

        # if epoch % 50 == 0:
        #     with torch.no_grad():
        #         generate_images_initial(H, sampler, viz_batch_original,
        #                                 sampler.selected_latents[0: H.num_images_visualize],
        #                                 sampler.last_selected_latents[0: H.num_images_visualize],
        #                                 viz_batch_original.shape, imle, ema_imle,
        #                                 f'{H.save_dir}/latest.png', logprint, experiment)


        # if H.use_wandb:
        #     wandb.log(metrics, step=iterate)
        
        # if epoch % 5 == 0 and experiment is not None:
        #     experiment.log_metrics(metrics, epoch=epoch, step=iterate)

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

    if H.mode == 'train':
        print(H)
        train_loop_imle(H, data_train, data_valid_or_test, preprocess_fn, imle, ema_imle, logprint, experiment)

if __name__ == "__main__":
    main()

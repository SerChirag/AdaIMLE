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
from models import IMLE
import numpy as np
from data import set_up_data
from helpers.train_helpers import (load_imle, load_opt, load_teacher, save_model, set_up_hyperparams, update_ema, set_seed)
from helpers.utils import ZippedDataset, init_distributed_mode, is_main_process, get_world_size, get_rank, safe_barrier
from sampler import Sampler
from torch import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import datetime
import os
import torch.distributed as dist

def isValid(num):
    return not num != num

def cleanup():
    dist.destroy_process_group()

def print_seed(device):
    cpu_seed = torch.initial_seed()
    cuda_seed = torch.cuda.initial_seed()
    print(f"Device {device} CPU seed = {cpu_seed}, GPU seed = {cuda_seed} \n")

def training_step_imle(H, latents, imle, teacher_imle, optimizer, scaler):
    
    loss = 0.0
    with autocast(device_type='cuda'):

        loss
        px_z = imle(latents)
        num_layers = len(px_z)
        with torch.no_grad():
            px_z_teacher = teacher_imle(latents)
        
        for i in range(len(px_z)):
            loss += F.l1_loss(px_z[i], px_z_teacher[i], reduction='mean')
    

    loss = loss / num_layers
    loss = loss / (H.accumulation_steps)
    
    scaler.scale(loss).backward()
    return loss.detach()

def generate_visualization(H, sampler, latent_for_visualization, shape, imle, fname, logprint, experiment=None):
    mb = shape[0] + 1
    latent_for_visualization = latent_for_visualization[:mb]
    batches = []

    for t in range(H.num_rows_visualize):
        batches.append(sampler.sample(latent_for_visualization[t], imle, None))

    n_rows = len(batches)
    img_height, img_width = batches[0].shape[1], batches[0].shape[2]
    
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, img_height, img_width, 3)).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * img_height, mb * img_width, 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)
    if(experiment):
        experiment.log_image(fname, overwrite=True)

def train_loop_imle(H, data_len, imle, ema_imle, teacher_imle, logprint, experiment):

    optimizer, scheduler, scaler, best_fid, iterate, starting_epoch = load_opt(H, imle, logprint)

    H.ema_rate = torch.as_tensor(H.ema_rate)

    device = torch.device("cuda", torch.cuda.current_device())

    epoch = starting_epoch     
    safe_barrier()

    latent_for_visualization = []

    sampler = Sampler()

    rand_z = torch.randn(data_len, H.latent_dim, device='cpu')

    if(is_main_process()):
        latent_for_visualization = torch.randn(H.num_rows_visualize, H.num_images_visualize, H.latent_dim).to(device)
    
    mean_loss = float('inf')
    metrics = {
        'mean_loss': mean_loss
    }
        
    while (epoch < H.num_epochs):

        safe_barrier()        
        rand_z.normal_()
        comb_dataset = TensorDataset(rand_z)

        # Use a DistributedSampler if in distributed training.
        train_sampler = DistributedSampler(comb_dataset, 
                                           shuffle=True, 
                                           num_replicas=H.world_size,
                                           rank=H.local_rank,
                                           seed=H.seed)
        
        data_loader = DataLoader(comb_dataset, batch_size=H.n_batch, sampler=train_sampler,
                                    pin_memory=True, num_workers=4, 
                                    persistent_workers=True, 
                                    multiprocessing_context="spawn",
                                    shuffle=False)

        # If using distributed sampler, set the epoch for shuffling
        train_sampler.set_epoch(epoch)

        if(is_main_process()):
            start_time = time.time()

        safe_barrier()        # Main training loop.

        epoch_loss_sum = 0.0  # We'll accumulate loss from each batch.
        epoch_iter_count = 0
        accum_counter = 0
        imle.zero_grad(set_to_none=True)


        for cur in data_loader:
            x = cur[0]
            x = x.to(device)

            loss = training_step_imle(H, x, imle, teacher_imle,
                               optimizer, scaler)
            
            epoch_loss_sum += loss.item()
            epoch_iter_count += 1

            accum_counter += 1

            # When we have accumulated enough mini-batches, perform the step.
            if accum_counter % H.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                imle.zero_grad(set_to_none=True)
                update_ema(imle.module, ema_imle, H.ema_rate)
            
            if iterate % H.iters_per_images == 0:
                if(is_main_process()):
                    imle.eval()
                    with torch.no_grad():
                        generate_visualization(H, sampler,
                                                latent_for_visualization,
                                                latent_for_visualization.shape, 
                                                imle,
                                                f'{H.save_dir}/samples-{iterate}.png', 
                                                logprint, 
                                                experiment)
                        generate_visualization(H, sampler,
                                                latent_for_visualization,
                                                latent_for_visualization.shape, 
                                                teacher_imle,
                                                f'{H.save_dir}/samples_teacher-{iterate}.png', 
                                                logprint, 
                                                experiment)
                    imle.train()
            iterate += 1
            
            

            
            if iterate % H.iters_per_ckpt == 0 and is_main_process():
                fp = os.path.join(H.save_dir, f'iter-{iterate}')
                logprint(f'Saving model@ {iterate} to {fp}')
                save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H)
            safe_barrier()
        
        if accum_counter % H.accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            imle.zero_grad(set_to_none=True)
            update_ema(imle.module, ema_imle, H.ema_rate)
        
        epoch_loss_tensor = torch.tensor(epoch_loss_sum, device=device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        total_batches_tensor = torch.tensor(epoch_iter_count, device=device)
        dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)

        mean_loss = epoch_loss_tensor.item() / total_batches_tensor.item()

        metrics = {
            'mean_loss': mean_loss,
            'curr_lr': optimizer.param_groups[0]['lr'],
        }

        if(is_main_process()):
            print(f'Epoch {epoch} took {time.time() - start_time} seconds')

            if epoch % 5 == 0:
                logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)


        if (epoch % 5 == 0 and is_main_process()):
            imle.eval()
            with torch.no_grad():
                generate_visualization(H, sampler,
                                        latent_for_visualization,
                                        latent_for_visualization.shape, imle,
                                        f'{H.save_dir}/latest.png', logprint, experiment)
            imle.train()

        if (epoch % 5 == 0 and experiment is not None and is_main_process()):
            experiment.log_metrics(metrics, epoch=epoch, step=iterate)
        
        if epoch % H.epoch_per_save == 0 and is_main_process() and isValid(mean_loss):
            fp = os.path.join(H.save_dir, 'latest')
            logprint(f'Saving latest model@ {iterate} to {fp}')
            save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H)
        safe_barrier()
        epoch += 1
    
    if is_main_process():
        print("Training complete. Saving final model.")
        fp = os.path.join(H.save_dir, 'final')
        logprint(f'Saving final model@ {iterate} to {fp}')
        save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H)
    safe_barrier()

def main():
    init_distributed_mode()
    
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)

    H.world_size = get_world_size()
    H.local_rank = get_rank()
    # imle, ema_imle = load_imle(H, logprint)

    experiment = None
    if(is_main_process()):
        print(H)
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

    safe_barrier()
    if(is_main_process()):
        logprint('training model', H.desc, 'on', H.dataset)

    imle, ema_imle = load_imle(H, logprint)
    teacher_imle = load_teacher(H)

    if(is_main_process()):
        num_params = sum(p.numel() for p in imle.parameters())
        print("Number of parameters in IMLE: ", num_params)
        logprint("Number of parameters in IMLE: ", num_params)
        H.num_params = num_params
        if(experiment is not None):
            experiment.log_parameter("num_params", num_params)

    if(H.mode == 'train'):
        train_loop_imle(H, len(data_train), imle, ema_imle, teacher_imle, logprint, experiment)

    cleanup()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

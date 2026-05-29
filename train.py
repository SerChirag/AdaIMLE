import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import time
from contextlib import nullcontext

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
from helpers.train_helpers import (configure_runtime_performance, load_imle, load_opt, load_sampler_state, save_model, set_up_hyperparams, update_ema, set_seed)
from helpers.utils import ZippedDataset, init_distributed_mode, is_main_process, get_world_size, get_rank, safe_barrier
from sampler import Sampler
from visual.interpolate import random_interp
from visual.utils import (generate_and_save, generate_for_NN,
                          generate_visualization,
                          get_sample_for_visualization)
from helpers.improved_precision_recall import compute_prec_recall
from torch import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import datetime
import os
import torch.distributed as dist
import resize_right
import resize_right.interp_methods as interp_methods

def isValid(num):
    return not num != num

def cleanup():
    dist.destroy_process_group()

def print_seed(device):
    cpu_seed = torch.initial_seed()
    cuda_seed = torch.cuda.initial_seed()
    print(f"Device {device} CPU seed = {cpu_seed}, GPU seed = {cuda_seed} \n")

def training_step_imle(H, targets_bchw, latents, class_labels, direction, imle, loss_fn, scaler):
    """Single training step. Per-sample reconstruction loss across the output resolution and
    each multi-res scale, weighted by `direction` (1.0 -> forward, 0.0 -> reverse). Forward
    rows use weight 1; reverse rows use weight H.reverse_loss_strength. When
    H.use_reverse_loss is off, every row in the batch has direction==1 and the math reduces
    exactly to the forward-only loss.
    """
    # torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    with autocast(device_type='cuda', dtype=H.amp_dtype_torch):
        px_z = imle(latents, class_labels, train=True)
        # Per-sample loss at output resolution.
        per_sample = loss_fn(px_z[-1], targets_bchw, use_mean=False)
        if H.use_multi_res:
            for i in range(2, len(px_z) - 1):
                px_z_scale = px_z[i]
                if H.use_resize_right:
                    targets_scale = resize_right.resize(
                        targets_bchw,
                        out_shape=(px_z_scale.shape[2], px_z_scale.shape[3]),
                        interp_method=interp_methods.cubic, antialiasing=True,
                    )
                else:
                    targets_scale = F.interpolate(
                        targets_bchw, size=(px_z_scale.shape[2], px_z_scale.shape[3]),
                        antialias=True, mode='bicubic', align_corners=H.align_corners,
                    )
                per_sample = per_sample + loss_fn(px_z_scale, targets_scale, use_mean=False)
        # direction: 1.0 (forward) or 0.0 (reverse) -> weight 1 or reverse_loss_strength.
        w = direction + (1.0 - direction) * H.reverse_loss_strength
        loss = (w * per_sample).mean()
        loss_measure = loss.detach().clone()

    loss = loss / (H.accumulation_steps)
    scaler.scale(loss).backward()
    return loss_measure.detach()

def train_loop_imle(H, data_train, data_valid, preprocess_fn, imle, ema_imle, logprint, experiment=None, autoencoder=None):
    # Recompute total_iters to include reverse samples so the scheduler covers the full
    # combined dataset. data.py sets H.total_iters using only N; with reverse loss the
    # DataLoader iterates over N+K rows per epoch.
    _N = len(data_train)
    _K = int(H.reverse_factor * _N) if H.use_reverse_loss else 0
    if _K > 0:
        H.total_iters = H.num_epochs * ((_N + _K + H.global_batch_size - 1) // H.global_batch_size)

    optimizer, scheduler, scaler, best_fid, iterate, starting_epoch = load_opt(H, imle, logprint)

    H.ema_rate = torch.as_tensor(H.ema_rate)

    sampler = Sampler(H, len(data_train), preprocess_fn, autoencoder=autoencoder)
    safe_barrier()
    device = torch.device("cuda", torch.cuda.current_device())

    load_sampler_state(H, sampler, logprint)

    epoch = starting_epoch
    sampler.init_projection(data_train)

    safe_barrier()
    if H.num_classes > 0:
        # One image per class, evenly strided so all classes are represented.
        # Dataset is sorted by class, so striding by (N // num_images_visualize) gives class diversity.
        n_total = len(data_train)
        stride = max(1, n_total // H.num_images_visualize)
        viz_indices = list(range(0, stride * H.num_images_visualize, stride))[:H.num_images_visualize]
        viz_indices_tensor = torch.tensor(viz_indices, dtype=torch.long)
        viz_subset = torch.utils.data.Subset(data_train, viz_indices)
        viz_batch_original, _ = get_sample_for_visualization(viz_subset, preprocess_fn, H.num_images_visualize, H.dataset)
    else:
        viz_indices_tensor = torch.arange(H.num_images_visualize, dtype=torch.long)
        viz_batch_original, _ = get_sample_for_visualization(data_train, preprocess_fn, H.num_images_visualize, H.dataset)

    latent_for_visualization = []

    if(is_main_process()):
        # Fixed seed so the same noise vectors are used across epochs and restarts.
        _viz_gen = torch.Generator().manual_seed(42)
        latent_for_visualization = torch.randn(H.num_rows_visualize, H.num_images_visualize, H.latent_dim,
                                               generator=_viz_gen).to(device)
    
    mean_loss = float('inf')
    metrics = {
        'mean_loss': mean_loss
    }

    # Single combined dataset for forward + (optional) reverse pairs.
    #   rows [0, N)    -> forward pairs   (direction=1.0, target_idx=arange(N))
    #   rows [N, N+K)  -> reverse pairs   (direction=0.0, target_idx from sampler.reverse_target_indices)
    # When H.use_reverse_loss=False, K=0 and the dataset is bit-equivalent to the old
    # forward-only setup. All three columns are CPU shared-memory tensors updated in-place
    # after each resample so persistent DataLoader workers see the new content.
    N = len(data_train)
    K = int(H.reverse_factor * N) if H.use_reverse_loss else 0
    total = N + K

    combined_latents     = torch.empty((total, H.latent_dim), dtype=torch.float32)
    combined_target_idx  = torch.empty((total,),              dtype=torch.long)
    combined_direction   = torch.empty((total,),              dtype=torch.float32)
    combined_latents.share_memory_()
    combined_target_idx.share_memory_()
    combined_direction.share_memory_()

    # Forward segment [0:N] — direction stays 1.0; target_idx stays arange(N); latents copied
    # from sampler.selected_latents now and after every forward resample.
    combined_target_idx[:N].copy_(torch.arange(N))
    combined_direction[:N].fill_(1.0)
    combined_latents[:N].copy_(sampler.selected_latents)
    # Reverse segment [N:N+K] — direction stays 0.0; target_idx and latents updated each resample.
    if K > 0:
        combined_direction[N:].fill_(0.0)

    comb_dataset = TensorDataset(combined_target_idx, combined_latents, combined_direction)
    train_sampler = DistributedSampler(
        comb_dataset,
        shuffle=True,
        num_replicas=H.world_size,
        rank=H.local_rank,
        seed=H.seed,
    )

    train_num_workers = getattr(H, 'num_workers', None)
    if train_num_workers is None:
        train_num_workers = 4
    data_loader = DataLoader(
        comb_dataset,
        batch_size=H.n_batch,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=train_num_workers,
        persistent_workers=train_num_workers > 0,
        multiprocessing_context="spawn" if train_num_workers > 0 else None,
        prefetch_factor=getattr(H, 'prefetch_factor', 4) if train_num_workers > 0 else None,
        shuffle=False,
    )

    if H.use_reverse_loss and is_main_process():
        print(f"[reverse-loss] enabled: N={N} forward + K={K} reverse pairs, "
              f"reverse_factor={H.reverse_factor}, reverse_loss_strength={H.reverse_loss_strength}")

    force_initial_resample = True  # Track the last epoch when resampling was done.
    reverse_loss_strength_final = H.reverse_loss_strength   # target value; ramp from 0 if requested

    while (epoch < H.num_epochs):
        # Linearly ramp reverse_loss_strength from 0 → target over reverse_rampup_epochs.
        # At epoch 0 the weight is 0 (pure forward IMLE); at epoch reverse_rampup_epochs it
        # reaches the configured target and stays there. Works correctly on resume because
        # `epoch` is restored from the checkpoint via starting_epoch.
        if H.use_reverse_loss and H.reverse_rampup_epochs > 0:
            H.reverse_loss_strength = min(1.0, epoch / H.reverse_rampup_epochs) * reverse_loss_strength_final
        # Update the IMLE force resampling every imle_force_resample epochs.
        if (epoch % H.imle_force_resample == 0) or (force_initial_resample):
            sampler.imle_sample_force(imle)
            combined_latents[:N].copy_(sampler.selected_latents)
            if H.use_reverse_loss:
                # Reverse direction reuses the same dispatcher (resample_pool + NN search
                # with swapped args + broadcast). Updates sampler.reverse_latents and
                # sampler.reverse_target_indices on all ranks; copy them into the reverse
                # segment of the combined table in-place.
                sampler.imle_sample_force(imle, reverse=True)
                combined_latents[N:].copy_(sampler.reverse_latents)
                combined_target_idx[N:].copy_(sampler.reverse_target_indices)
            force_initial_resample = False


        if (epoch % H.viz_freq == 0 and is_main_process()):
            latents = sampler.selected_latents[viz_indices_tensor]
            with torch.inference_mode():
                imle.eval()
                vis_labels = H.labels[viz_indices_tensor].to(device) if H.num_classes > 0 else None
                generate_for_NN(sampler, viz_batch_original, latents,
                                viz_batch_original.shape, imle,
                                f'{H.save_dir}/NN-samples_{epoch}-imle.png', logprint,
                                condition=vis_labels)
                imle.train()
        # If using distributed sampler, set the epoch for shuffling
        train_sampler.set_epoch(epoch)

        if(is_main_process()):
            start_time = time.time()

        epoch_loss_sum = torch.zeros((), device=device)  # Accumulate on device to avoid per-step host syncs.
        epoch_iter_count = 0
        accum_counter = 0
        imle.zero_grad(set_to_none=True)


        for target_idx_batch, latent_batch, direction_batch in data_loader:
            target_idx_batch = target_idx_batch.to(device, non_blocking=True)
            latents          = latent_batch.to(device, non_blocking=True)
            direction        = direction_batch.to(device, non_blocking=True)
            _proj = sampler._dataset_proj_gpu if sampler._dataset_proj_gpu is not None else sampler.dataset_proj_torch.to(device, non_blocking=True)
            flat_target = _proj.index_select(0, target_idx_batch)
            target_bchw = flat_target.view(
                flat_target.shape[0],
                H.image_channels,
                H.latent_spatial_size,
                H.latent_spatial_size,
            ).contiguous(memory_format=torch.channels_last)

            # Conditional case: derive class labels from the target data index. H.labels
            # is (N,) int64 — for forward rows target_idx==data index; for reverse rows
            # target_idx is the matched data point, whose class is still the right target.
            class_labels = H.labels[target_idx_batch.cpu()].to(device, non_blocking=True) if H.num_classes > 0 else None

            should_sync_grads = ((accum_counter + 1) % H.accumulation_steps == 0)
            grad_sync_context = nullcontext() if should_sync_grads or not hasattr(imle, 'no_sync') else imle.no_sync()
            with grad_sync_context:
                loss = training_step_imle(
                    H, target_bchw, latents, class_labels, direction,
                    imle, sampler.calc_loss, scaler,
                )
            
            epoch_loss_sum.add_(loss)
            epoch_iter_count += 1

            accum_counter += 1

            # When we have accumulated enough mini-batches, perform the step.
            if accum_counter % H.accumulation_steps == 0:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(imle.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                imle.zero_grad(set_to_none=True)
                update_ema(imle.module, ema_imle, H.ema_rate)
            
            if iterate % H.iters_per_images == 0:
                if(is_main_process()):
                    imle.eval()
                    with torch.inference_mode():
                        if H.num_classes > 0:
                            vis_row_labels = [H.labels[viz_indices_tensor].to(device)] * 2 + [
                                torch.full((H.num_images_visualize,), i % H.num_classes, dtype=torch.long, device=device)
                                for i in range(H.num_rows_visualize)
                            ]
                        else:
                            vis_row_labels = None
                        generate_visualization(H, sampler, viz_batch_original,
                                                sampler.selected_latents[viz_indices_tensor],
                                                sampler.last_selected_latents[viz_indices_tensor],
                                                latent_for_visualization,
                                                viz_batch_original.shape, imle,
                                                f'{H.save_dir}/samples-{iterate}.png', logprint, experiment,
                                                latent_labels=vis_row_labels)
                    imle.train()
            iterate += 1
            
            

            
            if iterate % H.iters_per_ckpt == 0:
                safe_barrier()
                if is_main_process():
                    fp = os.path.join(H.save_dir, f'iter-{iterate}')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H, sampler=sampler)
                safe_barrier()
        
        if accum_counter % H.accumulation_steps != 0:
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(imle.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            imle.zero_grad(set_to_none=True)
            update_ema(imle.module, ema_imle, H.ema_rate)
        
        epoch_loss_tensor = epoch_loss_sum
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        total_batches_tensor = torch.tensor(epoch_iter_count, device=device)
        dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)

        mean_loss = epoch_loss_tensor.item() / total_batches_tensor.item()
        
        base_model = imle.module if hasattr(imle, 'module') else imle
        class_emb_norm = base_model.decoder.class_embedding.weight.norm(dim=1).mean().item() if hasattr(base_model.decoder, 'class_embedding') else 0.0
        metrics = {
            'mean_loss': mean_loss,
            'curr_lr': optimizer.param_groups[0]['lr'],
            'unique_indices': sampler.unique_indices,
            'class_emb_norm': class_emb_norm,
        }

        if (epoch > 0 and epoch % H.fid_freq == 0):
            generate_and_save(H, imle, sampler, min(5000, len(data_train) * H.fid_factor))
            safe_barrier()            
            if(is_main_process()):
                if not H.autoencoder_decode_for_metrics:
                    metrics.update({'fid': float('nan'), 'best_fid': best_fid, 'precision': float('nan'), 'recall': float('nan')})
                else:
                    cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False, use_dataparallel=False, num_workers=0, device=device)
                    precision, recall = compute_prec_recall(f'{H.data_root}/img', f'{H.save_dir}/fid/')
                    if cur_fid < best_fid:
                        best_fid = cur_fid

                    metrics.update({'fid': cur_fid, 'best_fid': best_fid, 'precision': precision, 'recall': recall})

                    if cur_fid == best_fid:
                        fp = os.path.join(H.save_dir, 'best_fid')
                        logprint(f'Saving model best fid {best_fid} @ {iterate} to {fp}')
                        logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)
                        save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H, sampler=sampler)

            safe_barrier()

        if(is_main_process()):
            print(f'Epoch {epoch} took {time.time() - start_time} seconds')

            if epoch % 5 == 0:
                logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)


        if (epoch % H.viz_freq == 0 and is_main_process()):
            imle.eval()
            with torch.inference_mode():
                if H.num_classes > 0:
                    vis_row_labels = [H.labels[viz_indices_tensor].to(device)] * 2 + [
                        torch.full((H.num_images_visualize,), i % H.num_classes, dtype=torch.long, device=device)
                        for i in range(H.num_rows_visualize)
                    ]
                else:
                    vis_row_labels = None
                generate_visualization(H, sampler, viz_batch_original,
                                        sampler.selected_latents[viz_indices_tensor],
                                        sampler.last_selected_latents[viz_indices_tensor],
                                        latent_for_visualization,
                                        viz_batch_original.shape, imle,
                                        f'{H.save_dir}/latest.png', logprint, experiment,
                                        latent_labels=vis_row_labels)
            imle.train()

        if (epoch % 5 == 0 and experiment is not None and is_main_process()):
            experiment.log_metrics(metrics, epoch=epoch, step=iterate)
        
        if epoch % H.epoch_per_save == 0 and isValid(mean_loss):
            safe_barrier()
            if is_main_process():
                fp = os.path.join(H.save_dir, 'latest')
                logprint(f'Saving latest model@ {iterate} to {fp}')
                save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H, sampler=sampler)
            safe_barrier()
        epoch += 1
    
    if is_main_process():
        print("Training complete. Saving final model.")
        fp = os.path.join(H.save_dir, 'final')
        logprint(f'Saving final model@ {iterate} to {fp}')
        save_model(fp, imle, ema_imle, optimizer, scheduler, scaler, H, sampler=sampler)
    safe_barrier()

def main():
    init_distributed_mode()
    
    H, logprint = set_up_hyperparams()
    configure_runtime_performance(H, logprint)
    H.search_type = 'l2'
    H.lpips_coef = 0.0
    H.dino_coef = 0.0
    if H.l2_coef == 0.0:
        H.l2_coef = 1.0
    H, data_train, data_valid_or_test, preprocess_fn, autoencoder = set_up_data(H)

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

    if(is_main_process()):
        num_params = sum(p.numel() for p in imle.parameters())
        print("Number of parameters in IMLE: ", num_params)
        logprint("Number of parameters in IMLE: ", num_params)
        H.num_params = num_params
        if(experiment is not None):
            experiment.log_parameter("num_params", num_params)

    if(H.mode == 'train'):
        train_loop_imle(H, data_train, data_valid_or_test, preprocess_fn, imle, ema_imle, logprint, experiment, autoencoder=autoencoder)

    elif H.mode == 'eval_fid':
        sampler = Sampler(H, len(data_train), preprocess_fn, autoencoder=autoencoder)
        # generate_and_save(H, imle, sampler, 5000)
        safe_barrier()        
        if(is_main_process()):
            print("Generating samples for FID")

        imle.eval()
        generate_and_save(H, imle, sampler, 50000)
        safe_barrier()        # if(is_main_process()):
            
        #     cur_fid = fid.compute_fid(f'{H.data_root}/img', f'{H.save_dir}/fid/', verbose=False)
        #     print("FID: ", cur_fid)

    elif H.mode == 'interpolate':
        if(is_main_process()):
            print("Generating interpolations")
            os.makedirs(f'{H.save_dir}/interp', exist_ok=True)

        imle.eval()
        with torch.inference_mode():
            sampler = Sampler(H, len(data_train), preprocess_fn)
            safe_barrier()
            rank = get_rank()
            world_size = get_world_size()
            for i in range(rank,H.num_images_to_generate, world_size):
                random_interp(H, sampler, (0, 256, 256, 3), imle, f'{H.save_dir}/interp/{i}.png', logprint)
                
    cleanup()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

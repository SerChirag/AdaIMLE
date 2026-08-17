import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import time
import math
from contextlib import nullcontext

import comet_ml  # noqa: F401 -- must be imported before torch; set_up_comet does the real work
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
from helpers.train_helpers import (configure_runtime_performance, load_imle, load_opt, load_sampler_state, save_model, set_up_comet, set_up_hyperparams, update_ema, set_seed)
from helpers.utils import ZippedDataset, init_distributed_mode, is_main_process, get_world_size, get_rank, safe_barrier
from helpers.sam import SAM
from sampler import Sampler
from visual.interpolate import random_interp
from visual.utils import (generate_and_save, generate_and_save_smart,
                          compute_fid_smart, compute_fid_smart_roundtrip,
                          generate_and_save_smart_roundtrip,
                          generate_and_save_per_class_roundtrip,
                          generate_for_NN, generate_visualization,
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

def training_step_imle(H, targets_bchw, latents, labels, imle, loss_fn, scaler,
                       ema_imle=None, tr_latents=None, tr_labels=None, lambda_tr=0.0):

    # torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    with autocast(device_type='cuda', dtype=H.amp_dtype_torch):
        px_z = imle(latents, labels, train=True)
        loss = loss_fn(px_z[-1], targets_bchw)
        loss_measure = loss.detach().clone()
        num_resolutions = 1

        if(H.use_multi_res):
            
            for i in range(2,len(px_z)-1):
                px_z_scale = px_z[i]

                if(H.use_resize_right):
                    targets_scale = resize_right.resize(targets_bchw, out_shape=(px_z_scale.shape[2], px_z_scale.shape[3]), 
                                                        interp_method=interp_methods.cubic, antialiasing =True)
                else:
                    targets_scale = F.interpolate(targets_bchw, size=(px_z_scale.shape[2], px_z_scale.shape[3]), 
                                                  antialias=True, mode='bicubic', align_corners=H.align_corners)
                

                loss_scale = loss_fn(px_z_scale, targets_scale)
                
                loss.add_(loss_scale)
                num_resolutions += 1

        # Trust region. Drift is scored with the same sampler.calc_loss the IMLE term uses, and
        # under tr_multi_res over the same resolutions, so both sides of the total are on one
        # scale and lambda_tr reads directly as a fraction of the main loss. No target resizing
        # is needed here -- the two nets emit matching resolutions by construction.
        tr_loss_measure = None
        if tr_latents is not None:
            tr_train = bool(H.tr_multi_res)
            with torch.no_grad():
                ref_out = ema_imle(tr_latents, tr_labels, train=tr_train)
            live_out = imle(tr_latents, tr_labels, train=tr_train)
            if tr_train:
                tr_loss = loss_fn(live_out[-1], ref_out[-1].detach())
                for i in range(2, len(live_out) - 1):
                    tr_loss = tr_loss + loss_fn(live_out[i], ref_out[i].detach())
            else:
                tr_loss = loss_fn(live_out, ref_out.detach())
            tr_loss_measure = tr_loss.detach().clone()
            loss = loss + lambda_tr * tr_loss

    loss = loss / (H.accumulation_steps)

    scaler.scale(loss).backward()
    return loss_measure.detach(), tr_loss_measure

def train_loop_imle(H, data_train, data_valid, preprocess_fn, imle, ema_imle, logprint, experiment=None, autoencoder=None):
    optimizer, scheduler, scaler, best_fid, iterate, starting_epoch = load_opt(H, imle, logprint)

    H.ema_rate = torch.as_tensor(H.ema_rate)

    sampler = Sampler(H, len(data_train), preprocess_fn, autoencoder=autoencoder)
    safe_barrier()
    device = torch.device("cuda", torch.cuda.current_device())

    # SAM is built after load_opt so it inherits any requires_grad_(False) done there.
    sam = SAM(imle.parameters(), rho=H.sam_rho, adaptive=H.sam_adaptive) if H.sam else None
    if sam is not None and is_main_process():
        logprint(f'SAM enabled: rho={H.sam_rho}, adaptive={H.sam_adaptive}, '
                 f'start_epoch={H.sam_start_epoch}, freq={H.sam_freq}')

    # Trust region. set_seed gives every rank the same global RNG, so a dedicated per-rank
    # stream is what makes the ranks draw different latents -- otherwise all of them anchor
    # the same k points each step and the extra GPUs buy no coverage.
    trust_region = bool(getattr(H, 'use_trust_region', False))
    tr_k = max(1, int(round(H.n_batch * H.tr_k_frac))) if trust_region else 0
    tr_generator = None
    if trust_region:
        tr_generator = torch.Generator(device=device)
        tr_generator.manual_seed(H.seed + get_rank())
        if is_main_process():
            logprint(f'Trust region enabled: lambda_max={H.tr_lambda_max}, k={tr_k}/step/gpu, '
                     f'loss={H.loss_type}, multi_res={H.tr_multi_res}, '
                     f'ramped linearly to lambda_max over {H.num_epochs} epochs')

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

    # Keep a single mutable latent table for the lifetime of DataLoader workers.
    # Workers read this CPU shared-memory tensor, and we update it in-place after resampling.
    latent_table = torch.empty((len(data_train), H.latent_dim), dtype=torch.float32)
    latent_table.share_memory_()
    latent_table.copy_(sampler.selected_latents)

    comb_dataset = ZippedDataset(data_train, TensorDataset(latent_table))
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

    force_initial_resample = True  # Track the last epoch when resampling was done.
    opt_step_count = 0

    def optimizer_step(micro_batches, sam_active, lambda_tr=0.0):
        """Steps the optimizer, optionally re-evaluating the gradient at w + e_w first."""
        nonlocal opt_step_count
        # First-pass grads are already accumulated (and all-reduced) at this point.
        if sam_active and sam.ascent_step():
            imle.zero_grad(set_to_none=True)
            last = len(micro_batches) - 1
            # The replayed latents are the ones buffered on the first pass, not fresh draws:
            # SAM's second pass has to see the identical minibatch, trust-region latents included.
            for j, (mb_target, mb_latents, mb_labels, mb_tr_latents, mb_tr_labels) in enumerate(micro_batches):
                sync = (j == last) or not hasattr(imle, 'no_sync')
                with nullcontext() if sync else imle.no_sync():
                    training_step_imle(H, mb_target, mb_latents, mb_labels, imle,
                                       sampler.calc_loss, scaler,
                                       ema_imle=ema_imle, tr_latents=mb_tr_latents,
                                       tr_labels=mb_tr_labels, lambda_tr=lambda_tr)
            sam.restore()  # step from w, using grad L(w + e_w)

        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(imle.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        imle.zero_grad(set_to_none=True)
        update_ema(imle.module, ema_imle, H.ema_rate)
        opt_step_count += 1


    while (epoch < H.num_epochs):
        # Update the IMLE force resampling every imle_force_resample epochs.
        if (epoch % H.imle_force_resample == 0) or (force_initial_resample):
            sampler.imle_sample_force(imle)
            latent_table.copy_(sampler.selected_latents)
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

        # lambda_tr ramps linearly across the whole run: near zero while theta_ref is still
        # near-random and the network should be moving fast, at lambda_max by the final epoch.
        lambda_tr = H.tr_lambda_max * min(1.0, epoch / max(1, H.num_epochs)) if trust_region else 0.0

        epoch_loss_sum = torch.zeros((), device=device)  # Accumulate on device to avoid per-step host syncs.
        epoch_tr_loss_sum = torch.zeros((), device=device)
        epoch_iter_count = 0
        accum_counter = 0
        imle.zero_grad(set_to_none=True)
        sam_window_active = False
        micro_batches = []


        for cur, indices in data_loader:
            if accum_counter % H.accumulation_steps == 0:
                # Start of an accumulation window: decide once, so buffering is consistent.
                sam_window_active = (sam is not None
                                     and epoch >= H.sam_start_epoch
                                     and opt_step_count % max(1, H.sam_freq) == 0)
                micro_batches = []
            latents = cur[1][0]
            # cur[0] is (image_tensor, label_tensor) when num_classes > 0, else (image_tensor,)
            labels = cur[0][1].to(device, non_blocking=True) if H.num_classes > 0 and len(cur[0]) > 1 else None
            _proj = sampler._dataset_proj_gpu if sampler._dataset_proj_gpu is not None else sampler.dataset_proj_torch.to(device, non_blocking=True)
            flat_target = _proj.index_select(0, indices.to(device, non_blocking=True))
            target_bchw = flat_target.view(
                flat_target.shape[0],
                H.image_channels,
                H.latent_spatial_size,
                H.latent_spatial_size,
            )
            target_bchw = target_bchw.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            latents = latents.to(device, non_blocking=True)

            # Drawn every step even while lambda_tr is still 0, so the autograd graph keeps the
            # same shape across iterations -- DDP is built with static_graph=True.
            if trust_region:
                tr_latents = torch.randn(tr_k, H.latent_dim, device=device, generator=tr_generator)
                tr_labels = torch.randint(0, H.num_classes, (tr_k,), device=device,
                                          generator=tr_generator) if H.num_classes > 0 else None
            else:
                tr_latents, tr_labels = None, None

            should_sync_grads = ((accum_counter + 1) % H.accumulation_steps == 0)
            grad_sync_context = nullcontext() if should_sync_grads or not hasattr(imle, 'no_sync') else imle.no_sync()
            with grad_sync_context:
                loss, tr_loss = training_step_imle(H, target_bchw, latents, labels, imle,
                                                   sampler.calc_loss, scaler,
                                                   ema_imle=ema_imle, tr_latents=tr_latents,
                                                   tr_labels=tr_labels, lambda_tr=lambda_tr)

            if sam_window_active:
                micro_batches.append((target_bchw, latents, labels, tr_latents, tr_labels))

            epoch_loss_sum.add_(loss)
            if tr_loss is not None:
                epoch_tr_loss_sum.add_(tr_loss)
            epoch_iter_count += 1

            accum_counter += 1

            # When we have accumulated enough mini-batches, perform the step.
            if accum_counter % H.accumulation_steps == 0:
                optimizer_step(micro_batches, sam_window_active, lambda_tr)

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
            optimizer_step(micro_batches, sam_window_active, lambda_tr)

        epoch_loss_tensor = epoch_loss_sum
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        total_batches_tensor = torch.tensor(epoch_iter_count, device=device)
        dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)

        mean_loss = epoch_loss_tensor.item() / total_batches_tensor.item()

        mean_tr_loss = epoch_tr_loss_sum.item() / max(1, epoch_iter_count)  # this rank's average; logging only

        base_model = imle.module if hasattr(imle, 'module') else imle
        class_emb_norm = base_model.decoder.class_embedding.weight.norm(dim=1).mean().item() if hasattr(base_model.decoder, 'class_embedding') else 0.0
        metrics = {
            'mean_loss': mean_loss,
            'curr_lr': optimizer.param_groups[0]['lr'],
            'unique_indices': sampler.unique_indices,
            'class_emb_norm': class_emb_norm,
        }
        if sam is not None:
            metrics['sam_grad_norm'] = sam.last_grad_norm
        if trust_region:
            metrics['tr_loss'] = mean_tr_loss
            metrics['lambda_tr'] = lambda_tr

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
        experiment = set_up_comet(H, logprint)

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

    elif H.mode == 'eval_fid_smart':
        sampler = Sampler(H, len(data_train), preprocess_fn, autoencoder=autoencoder)
        safe_barrier()
        if(is_main_process()):
            print(f"Generating samples for FID with rejection (threshold={H.reject_threshold}, "
                  f"patch_size={H.reject_patch_size}, max_attempts={H.reject_max_attempts})")

        imle.eval()
        fid = compute_fid_smart(H, imle, sampler, 50000)
        safe_barrier()
        if(is_main_process()):
            print(f"FID: {fid}")

    elif H.mode == 'eval_fid_smart_roundtrip':
        sampler = Sampler(H, len(data_train), preprocess_fn, autoencoder=autoencoder)
        safe_barrier()
        if(is_main_process()):
            n_over = math.ceil(H.roundtrip_target / H.roundtrip_keep_frac)
            print(f"Generating {n_over} samples, keeping best {H.roundtrip_target} "
                  f"by round-trip {H.roundtrip_metric} (keep_frac={H.roundtrip_keep_frac})")

        imle.eval()
        fid = compute_fid_smart_roundtrip(H, imle, sampler, H.roundtrip_target)
        safe_barrier()
        if(is_main_process()):
            print(f"FID: {fid}")

    elif H.mode == 'generate_smart_roundtrip':
        sampler = Sampler(H, len(data_train), preprocess_fn, autoencoder=autoencoder)
        safe_barrier()
        if(is_main_process()):
            print(f"Generating and saving {H.roundtrip_target} samples, rejecting round-trip "
                  f"{H.roundtrip_metric} > {H.roundtrip_reject_threshold} "
                  f"(max_oversample={H.roundtrip_max_oversample})")

        imle.eval()
        generate_and_save_smart_roundtrip(H, imle, sampler, H.roundtrip_target,
                                          subdir=H.roundtrip_save_subdir)
        safe_barrier()

    elif H.mode == 'generate_per_class_roundtrip':
        sampler = Sampler(H, len(data_train), preprocess_fn, autoencoder=autoencoder)
        safe_barrier()
        if(is_main_process()):
            n_cls = len(H.per_class_classes) if H.per_class_classes else H.num_classes
            print(f"Generating {H.per_class_count} samples for each of {n_cls} classes, "
                  f"rejecting round-trip {H.roundtrip_metric} > {H.roundtrip_reject_threshold}")

        imle.eval()
        generate_and_save_per_class_roundtrip(H, imle, sampler,
                                              subdir=H.roundtrip_save_subdir)
        safe_barrier()

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

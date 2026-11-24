import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import time
import math
from contextlib import nullcontext

import comet_ml  # noqa: F401 -- must be imported before torch; set_up_comet does the real work
import imageio
import torch
import torch.nn as nn
from cleanfid import fid
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from models import IMLE
import numpy as np
from data import set_up_data
from helpers.train_helpers import (configure_runtime_performance, load_imle, load_opt, load_sampler_state, save_model, set_up_comet, set_up_hyperparams, update_ema, set_seed)
from helpers.utils import init_distributed_mode, is_main_process, get_world_size, get_rank, safe_barrier
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

def isValid(num):
    return not num != num

def cleanup():
    dist.destroy_process_group()

def print_seed(device):
    cpu_seed = torch.initial_seed()
    cuda_seed = torch.cuda.initial_seed()
    print(f"Device {device} CPU seed = {cpu_seed}, GPU seed = {cuda_seed} \n")

def build_image_weights(H, n, device):
    """--old_weight: per-dataset-index loss weight (old images = files under --old_images_root, after
    resolving symlinks; everything else 1). Index order = ImageFolder order, which is what set_up_data
    uses for imagenet_folder (class-sorted; its stable sort by label is a no-op). None when off."""
    w_old = float(getattr(H, 'old_weight', 1.0))
    if w_old == 1.0:
        return None
    if H.dataset != 'imagenet_folder':
        raise ValueError('--old_weight is only implemented for --dataset imagenet_folder.')
    from torchvision.datasets import ImageFolder
    samples = ImageFolder(H.data_root).samples                      # lists files only, loads nothing
    if len(samples) != n:
        raise ValueError(f'--old_weight: ImageFolder lists {len(samples)} files, dataset has {n}.')
    if H.labels is not None and not torch.equal(torch.tensor([c for _, c in samples]), H.labels.cpu().long()):
        raise ValueError('--old_weight: ImageFolder order does not match the dataset labels.')
    root = os.path.realpath(H.old_images_root) + os.sep
    is_old = torch.tensor([os.path.realpath(p).startswith(root) for p, _ in samples])
    if is_old.all() or not is_old.any():
        raise ValueError(f'--old_weight: {int(is_old.sum())}/{n} images are under {root}; need a mix.')
    if is_main_process():
        print(f'[old_weight] {int(is_old.sum())} old images (weight {w_old}), {int((~is_old).sum())} new (weight 1.0)')
    return torch.where(is_old, w_old, 1.0).float().to(device)


def training_step_imle(H, targets_bchw, latents, labels, imle, sampler, scaler, img_weights=None, perc_weights=None):
    if img_weights is None:
        loss_fn = sampler.calc_loss
    else:
        loss_fn = lambda inp, tar: sampler.calc_loss(inp, tar, weights=img_weights)

    # torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    with autocast(device_type='cuda', dtype=H.amp_dtype_torch):
        px_z = imle(latents, labels, train=True)
        loss = loss_fn(px_z[-1], targets_bchw)
        if img_weights is None:
            loss_measure = loss.detach().clone()
        else:   # log the unweighted loss so mean_loss stays comparable across --old_weight settings
            loss_measure = sampler.calc_loss(px_z[-1].detach(), targets_bchw)
        num_resolutions = 1

        if(H.use_multi_res):

            for i in range(len(px_z)-1):
                px_z_scale = px_z[i]
                # Skip unprojected low-res taps (Identity, res < 8): their channel
                # count is `width`, not image_channels, so they can't be scored
                # against the target. Previously this loop hardcoded range(2, ...)
                # assuming exactly two sub-8 stages; the channel guard makes it
                # correct for any pyramid, including ones that drop res 1 / 4.
                if px_z_scale.shape[1] != targets_bchw.shape[1]:
                    continue

                targets_scale = F.interpolate(targets_bchw, size=(px_z_scale.shape[2], px_z_scale.shape[3]),
                                              antialias=True, mode='bicubic', align_corners=H.align_corners)
                

                loss_scale = loss_fn(px_z_scale, targets_scale)

                loss.add_(loss_scale)
                num_resolutions += 1

    # 'sum' (default): every resolution contributes at full weight.
    # 'mean': average over resolutions, so the objective is invariant to how
    # many pyramid levels the decoder exposes.
    if getattr(H, 'multi_res_reduce', 'mean') == 'mean':
        loss = loss / num_resolutions

    # Weight on the latent-space term itself, so it can be traded off against the
    # perceptual term below. `loss_measure` is captured unweighted above, which keeps
    # the logged mean_loss comparable across different --latent_loss_coef settings.
    latent_coef = getattr(H, 'latent_loss_coef', 1.0)
    if latent_coef != 1.0:
        loss = loss * latent_coef

    # Perceptual term on the full-resolution output only, in decoder space: LPIPS-VGG or
    # spatial DINOv2 (one or the other). Kept outside the autocast block: it decodes through
    # the frozen VAE and runs the perceptual net, both in fp32.
    perc_measure = torch.zeros((), device=loss.device)
    lpips2_measure = torch.zeros((), device=loss.device)   # LPIPS when it runs next to DINO (--lpips_with_dino)
    if perc_weights is not None and not (H.lpips_coef > 0.0 and H.dino_coef > 0.0):
        raise NotImplementedError('per-sample perceptual weights (--balanced_moved_weight) are only wired for '
                                  '--lpips_with_dino (LPIPS + DINO together)')
    if H.lpips_coef > 0.0 and H.dino_coef > 0.0:            # only reachable with --lpips_with_dino (set_up_hyperparams)
        dino_loss, lpips_loss = sampler.calc_dino_lpips_loss(px_z[-1], targets_bchw, weights=perc_weights)
        perc_measure = dino_loss.detach().clone()
        lpips2_measure = lpips_loss.detach().clone()
        loss = loss.float() + H.dino_coef * dino_loss + H.lpips_coef * lpips_loss
    elif H.lpips_coef > 0.0:
        lpips_loss = sampler.calc_lpips_loss(px_z[-1], targets_bchw)
        perc_measure = lpips_loss.detach().clone()
        loss = loss.float() + H.lpips_coef * lpips_loss
    elif H.dino_coef > 0.0:
        dino_loss = sampler.calc_dino_loss(px_z[-1], targets_bchw)
        perc_measure = dino_loss.detach().clone()
        loss = loss.float() + H.dino_coef * dino_loss

    loss = loss / (H.accumulation_steps)

    scaler.scale(loss).backward()
    return loss_measure.detach(), perc_measure.detach(), lpips2_measure.detach()

def train_loop_imle(H, data_train, data_valid, preprocess_fn, imle, ema_imle, logprint, experiment=None, autoencoder=None):
    optimizer, scheduler, scaler, best_fid, iterate, starting_epoch = load_opt(H, imle, logprint)

    H.ema_rate = torch.as_tensor(H.ema_rate)

    train_len = len(data_train)

    sampler = Sampler(H, train_len, preprocess_fn, autoencoder=autoencoder)
    safe_barrier()
    device = torch.device("cuda", torch.cuda.current_device())

    load_sampler_state(H, sampler, logprint)

    epoch = starting_epoch
    sampler.init_projection(data_train)

    safe_barrier()
    if H.num_classes > 0:
        # One image per class, evenly strided so all classes are represented.
        # Dataset is sorted by class, so striding by (N // num_images_visualize) gives class diversity.
        n_total = train_len
        stride = max(1, n_total // H.num_images_visualize)
        viz_indices = list(range(0, stride * H.num_images_visualize, stride))[:H.num_images_visualize]
        viz_indices_tensor = torch.tensor(viz_indices, dtype=torch.long)
        viz_subset = torch.utils.data.Subset(data_train, viz_indices)
        viz_batch_original, _ = get_sample_for_visualization(viz_subset, preprocess_fn, H.num_images_visualize, H.dataset)
    else:
        viz_indices_tensor = torch.arange(H.num_images_visualize, dtype=torch.long)
        viz_batch_original, _ = get_sample_for_visualization(data_train, preprocess_fn, H.num_images_visualize, H.dataset)

    # Pixels are needed only by init_projection (which encodes them to VAE latents) and by
    # the fixed visualization subset built above. Training targets come from
    # sampler._dataset_proj_gpu and nothing reads pixels again, so drop the image tensor --
    # for FFHQ-256 that is ~14 GB of host RAM held for the whole run. A zero-width stand-in
    # keeps len(data_train) and the tensor-tuple shape valid; H.labels holds its own
    # reference to the label tensor, so labels are unaffected.
    if isinstance(data_train, TensorDataset) and len(data_train.tensors) > 0:
        placeholder = torch.empty(train_len, 0, dtype=data_train.tensors[0].dtype)
        data_train.tensors = (placeholder,) + tuple(data_train.tensors[1:])

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

    # Every per-step input is an index lookup: targets from sampler._dataset_proj_gpu,
    # latents from sampler.selected_latents, labels from H.labels. The only thing the
    # DataLoader ever contributed was the index stream -- it also collated, pinned and
    # shipped a batch of full-resolution pixels through shared memory every step, which
    # the training loop discarded. Iterate the index permutation directly instead.
    #
    # This reproduces DistributedSampler(shuffle=True, seed=H.seed).set_epoch(e) exactly
    # (same permutation, same wrap-around padding, same rank stride) so batch composition
    # and step count per rank are unchanged -- the latter matters for DDP, where a rank
    # running a different number of steps would hang at the gradient all-reduce.

    labels_gpu = H.labels.to(device) if (H.num_classes > 0 and H.labels is not None) else None
    img_weight_table = build_image_weights(H, train_len, device)

    def epoch_index_batches(ep, n_extra=0, active_idx=None):
        # n_extra > 0 (--balanced_extra_moved): indices train_len .. train_len + n_extra - 1 are the extra
        # moved-latent examples. active_idx (--balanced_rs_radius): train only these image indices this epoch.
        # n_extra == 0 and active_idx None reproduce the original stream exactly.
        n_items = active_idx.numel() if active_idx is not None else train_len + n_extra
        padded_len = math.ceil(n_items / H.world_size) * H.world_size
        g = torch.Generator()
        g.manual_seed(H.seed + ep)
        order = torch.randperm(n_items, generator=g)
        if active_idx is not None:
            order = active_idx[order]
        pad = padded_len - n_items
        if pad > 0:
            order = torch.cat([order, order.repeat(math.ceil(pad / n_items))[:pad]])
        rank_order = order[H.local_rank:padded_len:H.world_size]
        for start in range(0, rank_order.numel(), H.n_batch):
            yield rank_order[start:start + H.n_batch]

    force_initial_resample = True  # Track the last epoch when resampling was done.

    def optimizer_step():
        """Unscales, clips, steps the optimizer and scheduler, then updates the EMA weights."""
        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(imle.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        imle.zero_grad(set_to_none=True)
        update_ema(imle.module, ema_imle, H.ema_rate)


    while (epoch < H.num_epochs):
        # Update the IMLE force resampling every imle_force_resample epochs.
        if (epoch % H.imle_force_resample == 0) or (force_initial_resample):
            sampler.imle_sample_force(imle)
            force_initial_resample = False
        if sampler.match_balanced:
            sampler.set_balanced_slot(epoch)


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
        # Resolve the projection table once per epoch. Ranks where the sampler keeps no
        # GPU copy (unconditional, non-main) otherwise re-uploaded the entire table on
        # every step. imle_sample_force may reassign it, so re-read after the resample.
        proj_table = sampler._dataset_proj_gpu
        if proj_table is None:
            proj_table = sampler.dataset_proj_torch.to(device, non_blocking=True)

        if(is_main_process()):
            start_time = time.time()

        epoch_loss_sum = torch.zeros((), device=device)  # Accumulate on device to avoid per-step host syncs.
        epoch_perc_sum = torch.zeros((), device=device)
        epoch_lpips2_sum = torch.zeros((), device=device)
        epoch_iter_count = 0
        accum_counter = 0
        imle.zero_grad(set_to_none=True)
        # per-epoch own / moved split of the per-sample DINO and LPIPS (match_balanced + --lpips_with_dino):
        # [dino_own, lpips_own, n_own, dino_moved, lpips_moved, n_moved]
        split_sums = torch.zeros(6, dtype=torch.float64, device=device)
        track_split = (sampler.match_balanced and sampler.balanced_slot_moved is not None
                       and H.lpips_coef > 0.0 and H.dino_coef > 0.0)

        active_idx = None
        if getattr(sampler, 'balanced_rs_radius', 0.0) > 0 and sampler.balanced_active is not None:
            active_idx = torch.nonzero(sampler.balanced_active).flatten()
            if active_idx.numel() < H.world_size * H.n_batch:
                active_idx = None   # (nearly) every slot inside the radius: train everyone on their slot this epoch
        n_extra = 0
        if getattr(sampler, 'balanced_extra_moved', False) and sampler.balanced_extra_idx is not None:
            n_extra = sampler.balanced_extra_idx.numel()
        use_moved_w = getattr(sampler, 'balanced_moved_weight', 1.0) != 1.0 and sampler.balanced_slot_moved is not None
        for indices in epoch_index_batches(epoch, n_extra, active_idx):
            if n_extra > 0:
                is_extra = indices >= train_len
                j = (indices - train_len).clamp(min=0, max=n_extra - 1)
                img_idx = torch.where(is_extra, sampler.balanced_extra_idx[j], indices)
                latents = sampler.selected_latents.index_select(0, img_idx)
                latents[is_extra] = sampler.balanced_extra_latents[j[is_extra]]
            else:
                is_extra = None
                img_idx = indices
                latents = sampler.selected_latents.index_select(0, indices)
            idx_gpu = img_idx.to(device, non_blocking=True)
            labels = labels_gpu.index_select(0, idx_gpu) if labels_gpu is not None else None
            latents = latents.to(device, non_blocking=True)
            moved_w = None
            if use_moved_w:
                mv = sampler.balanced_slot_moved.index_select(0, img_idx)
                if is_extra is not None:
                    mv = mv | is_extra
                moved_w = torch.where(mv, sampler.balanced_moved_weight, 1.0).float().to(device, non_blocking=True)
            flat_target = proj_table.index_select(0, idx_gpu)
            target_bchw = flat_target.view(
                flat_target.shape[0],
                H.image_channels,
                H.latent_spatial_size,
                H.latent_spatial_size,
            ).contiguous(memory_format=torch.channels_last)

            should_sync_grads = ((accum_counter + 1) % H.accumulation_steps == 0)
            grad_sync_context = nullcontext() if should_sync_grads or not hasattr(imle, 'no_sync') else imle.no_sync()
            with grad_sync_context:
                img_w = img_weight_table.index_select(0, idx_gpu) if img_weight_table is not None else None
                if moved_w is not None:
                    img_w = moved_w if img_w is None else img_w * moved_w
                loss, perc_loss, lpips2_loss = training_step_imle(H, target_bchw, latents, labels,
                                                     imle, sampler, scaler, img_weights=img_w, perc_weights=moved_w)

            epoch_loss_sum.add_(loss)
            epoch_perc_sum.add_(perc_loss)
            epoch_lpips2_sum.add_(lpips2_loss)
            epoch_iter_count += 1
            if track_split and getattr(sampler, '_last_per_sample', None) is not None:
                ps_dino, ps_lp = sampler._last_per_sample
                mvb = sampler.balanced_slot_moved.index_select(0, img_idx)
                if is_extra is not None:
                    mvb = mvb | is_extra
                mvb = mvb.to(device, non_blocking=True)
                ow = (~mvb).double()
                mw = mvb.double()
                split_sums += torch.stack([(ps_dino.double() * ow).sum(), (ps_lp.double() * ow).sum(), ow.sum(),
                                           (ps_dino.double() * mw).sum(), (ps_lp.double() * mw).sum(), mw.sum()])

            accum_counter += 1

            # When we have accumulated enough mini-batches, perform the step.
            if accum_counter % H.accumulation_steps == 0:
                optimizer_step()

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
            optimizer_step()

        epoch_loss_tensor = epoch_loss_sum
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_perc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_lpips2_sum, op=dist.ReduceOp.SUM)
        total_batches_tensor = torch.tensor(epoch_iter_count, device=device)
        dist.all_reduce(total_batches_tensor, op=dist.ReduceOp.SUM)

        total_batches = total_batches_tensor.item()
        mean_loss = epoch_loss_tensor.item() / total_batches

        base_model = imle.module if hasattr(imle, 'module') else imle
        class_emb_norm = base_model.decoder.class_embedding.weight.norm(dim=1).mean().item() if hasattr(base_model.decoder, 'class_embedding') else 0.0
        metrics = {
            'mean_loss': mean_loss,
            'curr_lr': optimizer.param_groups[0]['lr'],
            'unique_indices': sampler.unique_indices,
            'class_emb_norm': class_emb_norm,
        }
        if H.lpips_coef > 0.0 and H.dino_coef > 0.0:
            metrics['mean_dino'] = epoch_perc_sum.item() / total_batches
            metrics['mean_lpips'] = epoch_lpips2_sum.item() / total_batches
        elif H.lpips_coef > 0.0:
            metrics['mean_lpips'] = epoch_perc_sum.item() / total_batches
        elif H.dino_coef > 0.0:
            metrics['mean_dino'] = epoch_perc_sum.item() / total_batches
        if sampler.use_rs_imle:
            metrics['rs_reject_pct'] = sampler.rs_reject_pct
        if sampler.match_balanced:
            metrics['balanced_moved_pct'] = sampler.balanced_moved_pct
            for k, v in getattr(sampler, 'balanced_zone_stats', {}).items():
                if isinstance(v, list):
                    for name, x in zip(('p10', 'p50', 'p90'), v):
                        metrics[f'{k.split("_d")[0]}_dist_{name}'] = x
                else:
                    metrics[k] = v
            if track_split:
                dist.all_reduce(split_sums, op=dist.ReduceOp.SUM)
                so = split_sums.tolist()
                if so[2] > 0:
                    metrics['dino_own'], metrics['lpips_own'] = so[0] / so[2], so[1] / so[2]
                if so[5] > 0:
                    metrics['dino_moved'], metrics['lpips_moved'] = so[3] / so[5], so[4] / so[5]
                metrics['trained_moved_frac'] = so[5] / max(so[2] + so[5], 1.0)
                metrics['trained_examples'] = so[2] + so[5]

        if (epoch > 0 and epoch % H.fid_freq == 0):
            generate_and_save(H, imle, sampler, min(5000, train_len * H.fid_factor))
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

            logprint(model=H.desc, type='train_loss', epoch=epoch, step=iterate, **metrics)   # every epoch


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

        if (experiment is not None and is_main_process()):
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

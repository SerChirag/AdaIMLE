from pathlib import Path

import torch
import numpy as np
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_imle_arguments
from helpers.utils import (is_dist_avail_and_initialized, logger, maybe_download)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
# from apex.optimizers import FusedAdam as AdamW
from torch.optim import AdamW
from models import IMLE
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import random
from helpers.utils import is_main_process, get_world_size, get_rank
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn


def resolve_amp_dtype(H):
    requested = str(getattr(H, 'amp_dtype', 'auto')).lower()
    if requested == 'auto':
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            requested = 'bf16'
        else:
            requested = 'fp16'
    if requested == 'bf16':
        return requested, torch.bfloat16
    return 'fp16', torch.float16


def configure_runtime_performance(H, logprint=None):
    torch.backends.cudnn.benchmark = bool(getattr(H, 'cudnn_benchmark', True))

    allow_tf32 = bool(getattr(H, 'allow_tf32', True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    matmul_precision = getattr(H, 'float32_matmul_precision', 'high')
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision(matmul_precision)

    amp_name, amp_dtype = resolve_amp_dtype(H)
    H.amp_dtype = amp_name
    H.amp_dtype_torch = amp_dtype

    if logprint is not None and is_main_process():
        logprint(
            f"runtime config: amp_dtype={H.amp_dtype} "
            f"cudnn_benchmark={torch.backends.cudnn.benchmark} "
            f"allow_tf32={allow_tf32} "
            f"float32_matmul_precision={matmul_precision} "
            f"channels_last={bool(getattr(H, 'use_channels_last', True))}"
        )


def maybe_to_channels_last(module, enabled):
    if enabled and torch.cuda.is_available():
        module.to(memory_format=torch.channels_last)
        for param in module.parameters():
            if param.ndim == 4 and param.shape[0] == 1 and param.shape[2] == 1 and param.shape[3] == 1:
                param.data = param.data.contiguous()
    return module

def update_ema(imle, ema_imle, ema_rate):
    ema_rate = float(ema_rate)
    src_params = [p.detach() for p in imle.parameters()]
    ema_params = [p.detach() for p in ema_imle.parameters()]
    one_minus_ema = 1 - ema_rate
    torch._foreach_mul_(ema_params, ema_rate)
    torch._foreach_add_(ema_params, src_params, alpha=one_minus_ema)


def as_plain_nn(model):
    """Returns the model without optimization wrappers."""
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        return as_plain_nn(model._orig_mod)
    elif isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        return as_plain_nn(model.module)
    elif isinstance(model, nn.DataParallel):
        return model.module
    else:
        return model

def map_saved_by_type(x):
    if isinstance(x, nn.Module):
        return as_plain_nn(x).state_dict()
    elif hasattr(x, "state_dict"):
        return x.state_dict()
    else:
        return x

def _save_model_worker(path, model_state, ema_state, optim_state, sched_state, scaler_state, sampler_state, from_log, to_log):
    import shutil
    torch.save(model_state,  f"{path}-model.th")
    torch.save(ema_state,    f"{path}-model-ema.th")
    torch.save(optim_state,  f"{path}-opt.th")
    torch.save(sched_state,  f"{path}-sched.th")
    torch.save(scaler_state, f"{path}-scaler.th")
    if sampler_state is not None:
        torch.save(sampler_state, f"{path}-sampler.th")
    if os.path.exists(from_log):
        shutil.copy2(from_log, to_log)


def save_model(path, imle, ema_imle, optimizer, scheduler, scaler, H, sampler=None):
    import threading
    model_state  = map_saved_by_type(imle)
    ema_state    = map_saved_by_type(ema_imle)
    optim_state  = map_saved_by_type(optimizer)
    sched_state  = map_saved_by_type(scheduler)
    scaler_state = map_saved_by_type(scaler)
    sampler_state = sampler.state_dict() if sampler is not None and hasattr(sampler, 'state_dict') else None
    from_log = os.path.join(H.save_dir, 'log.jsonl')
    to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    t = threading.Thread(
        target=_save_model_worker,
        args=(path, model_state, ema_state, optim_state, sched_state, scaler_state, sampler_state, from_log, to_log),
        daemon=True,
    )
    t.start()


def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if k in ['distortion_nans', 'rate_nans', 'skipped_updates', 'gcskip', 'loss_nans']:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            if len(finites) == 0:
                z[k] = 0.0
            else:
                z[k] = np.max(finites)
        elif k == 'loss':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['loss'] = np.mean(vals)
            z['loss_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = stats[-1][k] if len(stats) < frequency else np.mean([a[k] for a in stats[-frequency:]])
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:]])
    return z


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f



def distributed_maybe_download(path, local_rank, mpi_size):
    if not path.startswith('gs://'):
        return path
    filename = path[5:].replace('/', '-')
    with first_rank_first(local_rank, mpi_size):
        fp = maybe_download(path, filename)
    return fp


@contextmanager
def first_rank_first(local_rank, mpi_size):
    if mpi_size > 1 and local_rank > 0:
        dist.barrier()

    try:
        yield
    finally:
        if mpi_size > 1 and local_rank == 0:
            dist.barrier()


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(H.save_dir)
    H.logdir = os.path.join(H.save_dir, 'log')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    

def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_imle_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_save_dirs(H)
    set_seed(H.seed)
    logprint = logger(H.logdir)
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    random.seed(H.seed)
    return H, logprint


def restore_params(model, path, local_rank, mpi_size, map_ddp=True, map_cpu=False, strict=True):
    state_dict = torch.load(distributed_maybe_download(path, local_rank, mpi_size), map_location='cpu')
    if map_ddp:
        new_state_dict = {}
        l = len('module.')
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=strict)


def restore_log(path, local_rank, mpi_size):
    loaded = [json.loads(l) for l in open(distributed_maybe_download(path, local_rank, mpi_size))]

    try:
        cur_eval_loss = float('inf')
        for z in loaded:
            if 'type' in z and z['type'] == 'train_loss' and 'best_fid' in z:
                cur_eval_loss = min(cur_eval_loss, z['best_fid'])
    except:
        cur_eval_loss = float('inf')
    starting_epoch = max([z['epoch'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    iterate = max([z['step'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    return cur_eval_loss, iterate, starting_epoch


def load_imle(H, logprint):
    local_rank = get_rank()
    device = torch.device("cuda")

    imle = IMLE(H)
    imle.to(device)
    maybe_to_channels_last(imle, getattr(H, 'use_channels_last', True))
    
    if H.restore_path:
        if(is_main_process()):
            logprint(f'Restoring imle from {H.restore_path}')
        restore_params(imle, H.restore_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size, strict=H.load_strict)
        maybe_to_channels_last(imle, getattr(H, 'use_channels_last', True))

    ema_imle = IMLE(H)
    ema_imle = ema_imle.to(device)  # Move to the correct device.
    maybe_to_channels_last(ema_imle, getattr(H, 'use_channels_last', True))

    if H.restore_ema_path:
        if(is_main_process()):
            logprint(f'Restoring ema imle from {H.restore_ema_path}')
        restore_params(ema_imle, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size, strict=H.load_strict)
        maybe_to_channels_last(ema_imle, getattr(H, 'use_channels_last', True))
    else:
        ema_imle.load_state_dict(imle.state_dict())

    ema_imle.requires_grad_(False)
    ema_imle.eval()

    ddp_dev = torch.cuda.current_device()

    if(is_dist_avail_and_initialized()):
        imle = DDP(imle, device_ids=[ddp_dev], 
                    output_device=ddp_dev,
                    gradient_as_bucket_view=True,
                    static_graph=True
                    )
    
    if(H.compile):
        imle = torch.compile(imle)
        # ema_imle is frozen eval-only — compiling it doubles inductor overhead with no training benefit.
    
    return imle, ema_imle


def load_opt(H, imle, logprint):
    optimizer_kwargs = dict(
        weight_decay=H.wd,
        lr=H.lr,
        betas=(H.adam_beta1, H.adam_beta2),
        eps=H.adam_eps,
    )
    use_fused_adamw = bool(getattr(H, 'use_fused_adamw', True))
    fused_requested = use_fused_adamw and torch.cuda.is_available()
    if fused_requested:
        optimizer_kwargs['fused'] = True

    try:
        optimizer = AdamW(imle.parameters(), **optimizer_kwargs)
        if is_main_process():
            logprint(f'AdamW fused={bool(optimizer_kwargs.get("fused", False))}')
    except (TypeError, RuntimeError) as exc:
        if 'fused' in optimizer_kwargs:
            optimizer_kwargs.pop('fused')
            if is_main_process():
                logprint(f'AdamW fused fallback: {exc}')
            optimizer = AdamW(imle.parameters(), **optimizer_kwargs)
        else:
            raise

    scheduler1 = LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))
    cosine_iters = H.total_iters - H.warmup_iters
    scheduler2 = CosineAnnealingLR(optimizer, T_max=cosine_iters, eta_min=0.1 * H.lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[H.warmup_iters])
    scaler = torch.GradScaler(device="cuda", enabled=(getattr(H, 'amp_dtype', 'fp16') == 'fp16'))
    
    if H.restore_optimizer_path:
        if(is_main_process()):
            logprint(f'Restoring optimizer from {H.restore_optimizer_path}')
        optimizer.load_state_dict(
            torch.load(H.restore_optimizer_path, map_location='cpu'))
        
    if H.restore_scheduler_path:
        if(is_main_process()):
            logprint(f'Restoring scheduler from {H.restore_scheduler_path}')
        scheduler.load_state_dict(
            torch.load(H.restore_scheduler_path, map_location='cpu', weights_only=False))
        
    if H.restore_scaler_path:
        if(is_main_process()):
            logprint(f'Restoring scaler from {H.restore_scaler_path}')
        scaler.load_state_dict(
            torch.load(H.restore_scaler_path, map_location='cpu'))
        
    if H.restore_log_path:
        cur_eval_loss, iterate, starting_epoch = restore_log(H.restore_log_path, H.local_rank, H.mpi_size)
    else:
        cur_eval_loss, iterate, starting_epoch = float('inf'), 0, 0

    logprint('starting at epoch', starting_epoch, 'iterate', iterate, 'eval loss', cur_eval_loss)
    return optimizer, scheduler, scaler, cur_eval_loss, iterate, starting_epoch


def _resolve_sampler_restore_path(H):
    explicit = getattr(H, 'restore_sampler_path', None)
    if explicit:
        return explicit

    candidates = [
        getattr(H, 'restore_path', None),
        getattr(H, 'restore_ema_path', None),
        getattr(H, 'restore_optimizer_path', None),
        getattr(H, 'restore_scheduler_path', None),
        getattr(H, 'restore_scaler_path', None),
    ]

    for p in candidates:
        if not p:
            continue
        if p.endswith('-model.th'):
            return p[:-len('-model.th')] + '-sampler.th'
        if p.endswith('-model-ema.th'):
            return p[:-len('-model-ema.th')] + '-sampler.th'
        if p.endswith('-opt.th'):
            return p[:-len('-opt.th')] + '-sampler.th'
        if p.endswith('-sched.th'):
            return p[:-len('-sched.th')] + '-sampler.th'
        if p.endswith('-scaler.th'):
            return p[:-len('-scaler.th')] + '-sampler.th'

    return None


def load_sampler_state(H, sampler, logprint):
    if not bool(getattr(H, 'use_rs_imle', False)):
        return

    sampler_path = _resolve_sampler_restore_path(H)
    if not sampler_path:
        return

    try:
        state = torch.load(distributed_maybe_download(sampler_path, H.local_rank, H.mpi_size), map_location='cpu')
    except Exception as e:
        if is_main_process():
            logprint(f'Could not restore sampler state from {sampler_path} ({e})')
        return

    if hasattr(sampler, 'load_state_dict'):
        sampler.load_state_dict(state)
        if is_main_process():
            logprint(f"Restored sampler RS state from {sampler_path}")


def save_latents(H, outer, split_ind, latents, name='latents'):
    Path("{}/latent/".format(H.save_dir)).mkdir(parents=True, exist_ok=True)
    # for ind, z in enumerate(latents):
    torch.save(latents, '{}/latent/{}-{}-{}.npy'.format(H.save_dir, outer, split_ind, name))


def save_snoise(H, outer, snoise):
    Path("{}/latent/".format(H.save_dir)).mkdir(parents=True, exist_ok=True)
    for sn in snoise:
        torch.save(sn, '{}/latent/snoise-{}-{}.npy'.format(H.save_dir, outer, sn.shape[2]))


def save_latents_latest(H, split_ind, latents, name='latest'):
    Path("{}/latent/".format(H.save_dir)).mkdir(parents=True, exist_ok=True)
    # for ind, z in enumerate(latents):
    torch.save(latents, '{}/latent/{}-{}.npy'.format(H.save_dir, split_ind, name))

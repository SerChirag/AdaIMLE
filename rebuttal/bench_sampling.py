"""Sampling throughput of the dino30k-architecture IMLE generator (+ eq-vae-ema decoder), single GPU.
Stages: G = generator z,c -> 4x32x32 latent; VAE = latent -> 256x256 RGB; E2E = both + uint8 conversion.
Variants: eager fp32 / eager bf16 / compiled bf16 (inference_mode, channels_last, TF32 on)."""
import argparse, os, sys, time, json, copy
import torch
from torch import autocast
ROOT = '/localscratch/cva19/AdaIMLE'; sys.path.insert(0, ROOT); os.chdir(ROOT)
from helpers.train_helpers import set_up_hyperparams, restore_params, maybe_to_channels_last
from helpers.autoencoder import load_autoencoder, decode_latents_to_images
from models import IMLE

p = argparse.ArgumentParser()
p.add_argument('--ckpt', default='./new-vanilla-results/fd-virtual-incep-lr3e6/iter-60000-model.th')
p.add_argument('--g_batches', default='64,256,1024')
p.add_argument('--v_batches', default='16,64')
p.add_argument('--iters', type=int, default=20)
p.add_argument('--compile_mode', default='max-autotune-no-cudagraphs')
p.add_argument('--out', default='')
p.add_argument('--variants', default='eager-fp32,eager-bf16,compiled-bf16')
p.add_argument('--e2e_batch', type=int, default=0)   # 0 = largest --v_batches
args = p.parse_args()

dev = torch.device('cuda', 0); torch.cuda.set_device(dev)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
SP = os.path.dirname(os.path.abspath(__file__))
H, _ = set_up_hyperparams(s=['--hps', 'imagenet_folder', '--num_classes', '1000',
        '--data_root', './datasets/imagenet_dino_40_30k/', '--save_dir', f'{SP}/bench_h', '--use_comet', 'False',
        '--latent_dim', '512', '--width', '1200', '--autoencoder_type', 'eq-vae-ema', '--lpips_coef', '0',
        '--dec_blocks', '1x1,4m1,4x2,8m4,8x3,16m8,16x4,32m16,32x10', '--use_multi_res', 'True'])
H.image_size, H.image_channels, H.latent_spatial_size, H.mpi_size, H.local_rank = 256, 4, 32, 1, 0

model = IMLE(H).to(dev)
restore_params(model, args.ckpt, local_rank=0, mpi_size=1, map_cpu=True)
model = maybe_to_channels_last(model.eval().requires_grad_(False), True)
n_par = sum(p.numel() for p in model.parameters())
ae = load_autoencoder(H, dev).eval()
sf = getattr(ae, '_cached_scaling_factor', 1.0)
dec16 = copy.deepcopy(ae.decoder).to(torch.bfloat16).to(memory_format=torch.channels_last).eval()
use_pq = getattr(getattr(ae, 'config', None), 'use_post_quant_conv', True) and getattr(ae, 'post_quant_conv', None) is not None
pq16 = copy.deepcopy(ae.post_quant_conv).to(torch.bfloat16).eval() if use_pq else None
print(json.dumps(dict(gen_params_M=round(n_par / 1e6, 1), vae_dec_params_M=round(sum(p.numel() for p in ae.decoder.parameters()) / 1e6, 1),
                      gpu=torch.cuda.get_device_name(0), torch=torch.__version__)), flush=True)

def G_fp32(z, c):
    return model(z, c).float().reshape(len(z), 4, 32, 32)
def G_bf16(z, c):
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        return model(z, c).float().reshape(len(z), 4, 32, 32)
def V_fp32(lat):                                     # = what our eval path does (fp32 VAE)
    return decode_latents_to_images(ae, lat, (32, 32))
def V_bf16_raw(lat):
    z16 = (lat / sf).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
    if pq16 is not None:
        z16 = pq16(z16)
    return dec16(z16).float().clamp_(-1, 1)
G_c = torch.compile(G_bf16, mode=args.compile_mode, dynamic=False)
V_c = torch.compile(V_bf16_raw, mode=args.compile_mode, dynamic=False)

def bench(fn, make_inp, n_img, iters=args.iters, warm=3):
    with torch.inference_mode():
        inp = make_inp()
        t0 = time.time()
        for _ in range(warm):
            fn(*inp)
        torch.cuda.synchronize(); t_warm = time.time() - t0
        torch.cuda.reset_peak_memory_stats()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn(*inp)
        e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return dict(img_s=round(n_img / (ms / 1000), 1), ms_batch=round(ms, 2), warm_s=round(t_warm, 1),
                peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))

res = []
def run(stage, var, fn, mk, b):
    try:
        r = bench(fn, mk, b)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); r = dict(img_s='OOM')
    r.update(stage=stage, variant=var, batch=b); res.append(r); print(json.dumps(r), flush=True)
    torch.cuda.empty_cache()

zc = lambda b: (lambda: (torch.randn(b, 512, device=dev), torch.randint(0, 1000, (b,), device=dev)))
lat = lambda b: (lambda: (torch.randn(b, 4, 32, 32, device=dev),))
for b in map(int, args.g_batches.split(',')):
    for var, fn in (('eager-fp32', G_fp32), ('eager-bf16', G_bf16), ('compiled-bf16', G_c)):
        if var in args.variants: run('G', var, fn, zc(b), b)
for b in map(int, args.v_batches.split(',')):
    for var, fn in (('eager-fp32', V_fp32), ('eager-bf16', V_bf16_raw), ('compiled-bf16', V_c)):
        if var in args.variants: run('VAE', var, fn, lat(b), b)

def E2E(z, c):                                       # fastest path: compiled G + compiled bf16 VAE + uint8
    x = V_c(G_c(z, c))
    return ((x + 1) * 127.5).round_().to(torch.uint8)
b = args.e2e_batch or max(map(int, args.v_batches.split(',')))
run('E2E', 'compiled-bf16', E2E, zc(b), b)
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)

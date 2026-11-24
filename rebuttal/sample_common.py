"""Shared setup for the sampling benchmarks / VAE-precision FID check: dino30k-architecture generator + eq-vae-ema."""
import os, sys, copy, json
import torch
from torch import autocast
ROOT = '/localscratch/cva19/AdaIMLE'; sys.path.insert(0, ROOT); os.chdir(ROOT)
from helpers.train_helpers import set_up_hyperparams, restore_params, maybe_to_channels_last
from helpers.autoencoder import load_autoencoder
from models import IMLE

SP = os.path.dirname(os.path.abspath(__file__))
CKPT = './new-vanilla-results/fd-virtual-incep-lr3e6/iter-60000-model.th'


def setup(ckpt=CKPT, compile_mode='max-autotune-no-cudagraphs'):
    dev = torch.device('cuda', 0); torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    H, _ = set_up_hyperparams(s=['--hps', 'imagenet_folder', '--num_classes', '1000',
            '--data_root', './datasets/imagenet_dino_40_30k/', '--save_dir', f'{SP}/bench_h', '--use_comet', 'False',
            '--latent_dim', '512', '--width', '1200', '--autoencoder_type', 'eq-vae-ema', '--lpips_coef', '0',
            '--dec_blocks', '1x1,4m1,4x2,8m4,8x3,16m8,16x4,32m16,32x10', '--use_multi_res', 'True'])
    H.image_size, H.image_channels, H.latent_spatial_size, H.mpi_size, H.local_rank = 256, 4, 32, 1, 0
    model = IMLE(H).to(dev)
    restore_params(model, ckpt, local_rank=0, mpi_size=1, map_cpu=True)
    model = maybe_to_channels_last(model.eval().requires_grad_(False), True)
    ae = load_autoencoder(H, dev).eval()
    sf = getattr(ae, '_cached_scaling_factor', 1.0)
    use_pq = getattr(getattr(ae, 'config', None), 'use_post_quant_conv', True) and getattr(ae, 'post_quant_conv', None) is not None
    dec32, pq32 = ae.decoder.to(memory_format=torch.channels_last), (ae.post_quant_conv if use_pq else None)
    dec16 = copy.deepcopy(ae.decoder).to(torch.bfloat16).to(memory_format=torch.channels_last).eval()
    pq16 = copy.deepcopy(ae.post_quant_conv).to(torch.bfloat16).eval() if use_pq else None

    def G(z, c):                                   # bf16 autocast generator -> fp32 latent B x 4 x 32 x 32
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            return model(z, c).float().reshape(len(z), 4, 32, 32)

    def V32(lat):                                  # fp32 VAE decode (= ae.decode, what the FID eval path uses)
        x = (lat / sf).float().contiguous(memory_format=torch.channels_last)
        if pq32 is not None:
            x = pq32(x)
        return dec32(x).float().clamp(-1, 1)

    def V16(lat):                                  # bf16 VAE decode
        x = (lat / sf).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
        if pq16 is not None:
            x = pq16(x)
        return dec16(x).float().clamp(-1, 1)

    to_u8 = lambda x: ((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)   # B x 3 x H x W
    C = lambda f: torch.compile(f, mode=compile_mode, dynamic=False)
    return dict(dev=dev, G=G, V32=V32, V16=V16, Gc=C(G), V32c=C(V32), V16c=C(V16), to_u8=to_u8,
                gpu=torch.cuda.get_device_name(0))

"""Does decoding with the bf16 VAE (fastest sampling path) change FID? Generate 50k samples (class i % 1000) with the
compiled bf16 generator, decode the SAME latents with the fp32 and the bf16 VAE, save both as ADM-evaluator npz
(arr_0, N x 256 x 256 x 3 uint8) and report the pixel difference between the two decodes."""
import argparse, json, os, time
import numpy as np, torch
from sample_common import setup

p = argparse.ArgumentParser()
p.add_argument('--out_dir', required=True)
p.add_argument('--n', type=int, default=50000)
p.add_argument('--batch', type=int, default=250)       # 50000 / 250: every batch has the same shape (no recompiles)
p.add_argument('--vae_chunk', type=int, default=50)
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
S = setup(compile_mode='default'); dev = S['dev']
a32 = np.empty((args.n, 256, 256, 3), np.uint8); a16 = np.empty_like(a32)
g = torch.Generator(device=dev).manual_seed(args.seed)
diff_sum, diff_max, t0 = 0.0, 0, time.time()
with torch.inference_mode():
    for s in range(0, args.n, args.batch):
        z = torch.randn(args.batch, 512, device=dev, generator=g)
        c = torch.arange(s, s + args.batch, device=dev) % 1000
        lat = S['Gc'](z, c)
        for i in range(0, args.batch, args.vae_chunk):
            l = lat[i:i + args.vae_chunk]
            x32 = S['to_u8'](S['V32c'](l)); x16 = S['to_u8'](S['V16c'](l))
            d = (x32.int() - x16.int()).abs()
            diff_sum += d.float().mean().item() * len(l); diff_max = max(diff_max, int(d.max()))
            a32[s + i:s + i + len(l)] = x32.permute(0, 2, 3, 1).cpu().numpy()
            a16[s + i:s + i + len(l)] = x16.permute(0, 2, 3, 1).cpu().numpy()
        if s % 10000 == 0:
            print(json.dumps(dict(done=s + args.batch, sec=round(time.time() - t0, 1))), flush=True)
print(json.dumps(dict(event='generated', n=args.n, sec=round(time.time() - t0, 1),
                      mean_abs_diff_uint8=round(diff_sum / args.n, 4), max_abs_diff_uint8=diff_max)), flush=True)
np.savez(f'{args.out_dir}/samples_vae32.npz', arr_0=a32)
np.savez(f'{args.out_dir}/samples_vae16.npz', arr_0=a16)
print('saved', flush=True)

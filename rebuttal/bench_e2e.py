"""End-to-end sampling throughput (noise + class -> 256x256 uint8 image on GPU, no disk I/O), single GPU.
Generator always compiled bf16 at --batch; VAE decode in chunks of --vae_chunk with the chosen precision/mode."""
import argparse, json, time
import torch
from sample_common import setup

p = argparse.ArgumentParser()
p.add_argument('--batch', type=int, default=256)
p.add_argument('--vae_chunk', type=int, default=64)
p.add_argument('--modes', default='vae32-eager,vae32-compiled')
p.add_argument('--iters', type=int, default=10)
p.add_argument('--out', default='')
args = p.parse_args()
S = setup(); dev = S['dev']
V = {'vae32-eager': S['V32'], 'vae32-compiled': S['V32c'], 'vae16-eager': S['V16'], 'vae16-compiled': S['V16c']}

def e2e(z, c, vf):
    lat = S['Gc'](z, c)
    return torch.cat([S['to_u8'](vf(lat[i:i + args.vae_chunk])) for i in range(0, len(lat), args.vae_chunk)])

res = []
for m in args.modes.split(','):
    with torch.inference_mode():
        z = torch.randn(args.batch, 512, device=dev); c = torch.randint(0, 1000, (args.batch,), device=dev)
        t0 = time.time()
        for _ in range(3):
            e2e(z, c, V[m])
        torch.cuda.synchronize(); warm = time.time() - t0
        torch.cuda.reset_peak_memory_stats()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.iters):
            e2e(z, c, V[m])
        e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / args.iters
    r = dict(img_s=round(args.batch / (ms / 1000), 1), ms_batch=round(ms, 1), warm_s=round(warm, 1),
             peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), stage='E2E', gen='compiled-bf16',
             vae=m, batch=args.batch, vae_chunk=args.vae_chunk, gpu=S['gpu'])
    res.append(r); print(json.dumps(r), flush=True); torch.cuda.empty_cache()
if args.out:
    json.dump(res, open(args.out, 'w'), indent=1)

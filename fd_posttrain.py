"""FD-loss post-training of a dino30k IMLE generator, after Yang et al., "Representation Frechet Loss for Visual
Generation" (arXiv 2604.28190).

Loss = Frechet distance between Inception-v3 pool3 features (pytorch-fid FID weights) of generated images and fixed
reference statistics (mu_r, Sigma_r):
    FD = |mu_g - mu_r|^2 + tr(Sigma_g) + tr(Sigma_r) - 2 tr((S Sigma_g S)^1/2),   S = Sigma_r^1/2
Generated statistics are an EMA of the first and second feature moments (decay --beta). The current batch, gathered
over all GPUs, enters with weight (1 - beta) and is the only part that carries gradients; the trace-sqrt term is the
sum of sqrt(eigvalsh(S Sigma_g S)). The moments are warm-started from --warm_n samples of the starting model (class
i % 1000, uint8-quantised like the eval PNGs), so the first log line is that model's FID under pytorch-fid Inception.
No IMLE loss and no training images: the generator only sees the FD gradient.

!! The default --ref is the ADM VIRTUAL ImageNet-256 reference = the FID evaluation reference (full ImageNet train
   set statistics). A run with it is a feasibility probe / lower bound, NOT a fair 30-images-per-class result.

Outputs in --save_dir: iter-<k>-model.th (state_dict, loadable by train.py --restore_path / eval_dino.sh),
state.pt (resume: model, optimizer, moments, step), log.jsonl.
Launch: torchrun --nproc_per_node=8 fd_posttrain.py --restore <ckpt.th> --save_dir <dir> [--lr 3e-6 ...]"""
import argparse, json, math, os, sys, time
import numpy as np, torch, torch.distributed as dist
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT); os.chdir(ROOT)
from helpers.train_helpers import set_up_hyperparams, restore_params
from helpers.autoencoder import (load_autoencoder, decode_latents_to_images,
                                 decode_latents_to_images_differentiable)
from models import IMLE
from pytorch_fid.inception import InceptionV3

p = argparse.ArgumentParser()
p.add_argument('--restore', required=True)
p.add_argument('--save_dir', required=True)
p.add_argument('--ref', default='./evaluation/VIRTUAL_imagenet256_labeled.npz')
p.add_argument('--lr', type=float, default=3e-6)
p.add_argument('--warmup', type=int, default=200)          # linear lr warmup steps, then constant (or cosine, below)
p.add_argument('--sched', default='const', choices=['const', 'cosine'])
p.add_argument('--decay_start', type=int, default=0)       # cosine: lr stays at --lr until this step, then cosine to
p.add_argument('--lr_min', type=float, default=0.0)        #   --lr_min at --iters (lets a resumed constant-lr run anneal)
p.add_argument('--n_batch', type=int, default=32)          # per GPU
p.add_argument('--iters', type=int, default=20000)
p.add_argument('--ckpt_every', type=int, default=1000)
p.add_argument('--log_every', type=int, default=25)
p.add_argument('--beta', type=float, default=0.999)        # EMA decay of the generated feature moments
p.add_argument('--warm_n', type=int, default=50000)
p.add_argument('--warm_batch', type=int, default=64)
p.add_argument('--grad_ckpt', type=int, default=1)         # gradient checkpointing in the VAE decoder
p.add_argument('--clip', type=float, default=0.0)          # grad-norm clip (0 = off; the norm is logged either way)
p.add_argument('--eig_eps', type=float, default=1e-10)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--stop_after', type=int, default=0)        # exit after this many steps in this launch (tests)
p.add_argument('--dec_blocks', default='1x1,4m1,4x2,8m4,8x3,16m8,16x4,32m16,32x10')   # dino30k decoder (default)
p.add_argument('--data_root', default='./datasets/imagenet_dino_40_30k/')            # only fills H; no images are read
args = p.parse_args()

dist.init_process_group('nccl')
rank, W, lrank = dist.get_rank(), dist.get_world_size(), int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(lrank); dev = torch.device('cuda', lrank)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
main = rank == 0
os.makedirs(args.save_dir, exist_ok=True)

def log(**kw):
    if main:
        kw['time'] = time.strftime('%m-%d %H:%M:%S')
        print(json.dumps(kw), flush=True)
        with open(f'{args.save_dir}/log.jsonl', 'a') as f:
            f.write(json.dumps(kw) + '\n')

H, _ = set_up_hyperparams(s=['--hps', 'imagenet_folder', '--num_classes', '1000',
        '--data_root', args.data_root, '--save_dir', f'{args.save_dir}/h', '--use_comet', 'False',
        '--latent_dim', '512', '--width', '1200', '--autoencoder_type', 'eq-vae-ema', '--lpips_coef', '0',
        '--dec_blocks', args.dec_blocks, '--use_multi_res', 'True'])
H.image_size, H.image_channels, H.latent_spatial_size, H.mpi_size, H.local_rank = 256, 4, 32, W, lrank

ae = load_autoencoder(H, dev)
if args.grad_ckpt:
    ae.enable_gradient_checkpointing()
inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=False)
inc = inc.to(dev).eval().requires_grad_(False)

def feats(x):                      # x: B x 3 x 256 x 256 in [-1, 1] -> B x 2048 (fp32, 299 bilinear inside)
    return inc(x)[0].flatten(1)

with np.load(args.ref) as d:
    mu_r = torch.from_numpy(d['mu']).double().to(dev)
    sig_r = torch.from_numpy(d['sigma']).double().to(dev)
ev, V = torch.linalg.eigh(sig_r)
S = (V * ev.clamp_min(0).sqrt()) @ V.T
tr_r = torch.trace(sig_r)

def frechet(mu_g, sig_g):
    A = S @ sig_g @ S
    lam = torch.linalg.eigvalsh((A + A.T) / 2)
    mean_term = (mu_g - mu_r).pow(2).sum()
    return mean_term + torch.trace(sig_g) + tr_r - 2 * lam.clamp_min(args.eig_eps).sqrt().sum(), mean_term

model = IMLE(H).to(dev)
restore_params(model, args.restore, local_rank=lrank, mpi_size=W, map_cpu=True)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
step, m1, M2 = 0, None, None
state_path = f'{args.save_dir}/state.pt'
if os.path.exists(state_path):
    st = torch.load(state_path, map_location='cpu')
    model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
    step, m1, M2 = st['step'], st['m1'].to(dev), st['M2'].to(dev)
    log(event='resumed', step=step)
model = DDP(model, device_ids=[lrank], static_graph=True, gradient_as_bucket_view=True)

def gen_latents(z, c):
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        return model(z, c).float().reshape(len(z), 4, 32, 32)

if m1 is None:                     # warm start from the starting model, eval protocol (class i % 1000)
    t0 = time.time()
    s1 = torch.zeros(2048, device=dev, dtype=torch.float64); s2 = torch.zeros(2048, 2048, device=dev, dtype=torch.float64)
    n_loc = args.warm_n // W; g = torch.Generator(device=dev).manual_seed(1000 + rank)
    with torch.no_grad():
        for s in range(rank * n_loc, (rank + 1) * n_loc, args.warm_batch):
            b = min(args.warm_batch, (rank + 1) * n_loc - s)
            z = torch.randn(b, 512, device=dev, generator=g)
            c = torch.arange(s, s + b, device=dev) % 1000
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                lat = model.module(z, c).float().reshape(b, 4, 32, 32)
            x = decode_latents_to_images(ae, lat, (32, 32)).clone()
            x = ((x + 1) * 127.5).round() / 127.5 - 1
            f = feats(x).double(); s1 += f.sum(0); s2 += f.T @ f
    cnt = torch.tensor(float(n_loc * W), device=dev, dtype=torch.float64)
    dist.all_reduce(s1); dist.all_reduce(s2)
    m1, M2 = s1 / cnt, s2 / cnt
    fd0, mt0 = frechet(m1, M2 - torch.outer(m1, m1))
    log(event='warm_start', n=int(cnt.item()), fd=round(fd0.item(), 4), mean_term=round(mt0.item(), 4),
        sec=round(time.time() - t0, 1))

torch.manual_seed(args.seed + 1000 * rank + step)
model.train()
t_last, start_step, gn_sum, n_sum = time.time(), step, 0.0, 0
while step < args.iters:
    lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
    if args.sched == 'cosine' and step >= args.decay_start:
        t = min(1.0, (step - args.decay_start) / max(1, args.iters - args.decay_start))
        lr = args.lr_min + (lr - args.lr_min) * 0.5 * (1 + math.cos(math.pi * t))
    for gp in opt.param_groups:
        gp['lr'] = lr
    z = torch.randn(args.n_batch, 512, device=dev)
    c = torch.randint(0, 1000, (args.n_batch,), device=dev)
    x = decode_latents_to_images_differentiable(ae, gen_latents(z, c), (32, 32)).clamp(-1, 1)
    f = feats(x).double()
    parts = [torch.empty_like(f) for _ in range(W)]
    dist.all_gather(parts, f.detach())
    parts[rank] = f                                      # own slice keeps its graph
    Fb = torch.cat(parts)
    mu_g = args.beta * m1 + (1 - args.beta) * Fb.mean(0)
    M_g = args.beta * M2 + (1 - args.beta) * (Fb.T @ Fb) / len(Fb)
    loss, mean_term = frechet(mu_g, M_g - torch.outer(mu_g, mu_g))
    # x W: each rank only backprops through its own slice and DDP averages, so this restores the full gradient;
    # / (1 - beta): batch moments enter with weight (1 - beta), rescale so gradients are O(1) for Adam's eps.
    (loss * W / (1 - args.beta)).backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip if args.clip > 0 else float('inf'))
    if torch.isfinite(gn) and torch.isfinite(loss):
        opt.step()
        m1, M2 = mu_g.detach(), M_g.detach()
    else:
        log(event='skip_nonfinite', step=step, loss=float(loss), gn=float(gn))
    opt.zero_grad(set_to_none=True)
    step += 1; gn_sum += float(gn); n_sum += 1
    if step % args.log_every == 0:
        torch.cuda.synchronize(); dt = (time.time() - t_last) / n_sum
        log(step=step, fd_live=round(loss.item(), 4), mean_term=round(mean_term.item(), 4), gn=round(gn_sum / n_sum, 4),
            lr=lr, sec_it=round(dt, 3), mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
        t_last, gn_sum, n_sum = time.time(), 0.0, 0
    if step % args.ckpt_every == 0 or step == args.iters:
        if main:
            torch.save(model.module.state_dict(), f'{args.save_dir}/iter-{step}-model.th')
            torch.save({'model': model.module.state_dict(), 'opt': opt.state_dict(), 'step': step,
                        'm1': m1.cpu(), 'M2': M2.cpu()}, state_path + '.tmp')
            os.replace(state_path + '.tmp', state_path)
        dist.barrier()
    if args.stop_after and step - start_step >= args.stop_after:
        break
if main and step >= args.iters:
    open(f'{args.save_dir}/DONE', 'w').close()
dist.destroy_process_group()

"""DINOv2-L features (FD-Loss paper protocol) for a folder of {i}.png samples, sharded over torchrun ranks (no comms).
Uses FD-Loss's own TimmReprModel: [0,1] -> bicubic-antialias to 256 (no-op for 256 px) -> ImageNet norm ->
timm vit_large_patch14_dinov2.lvd142m (dynamic img size/pad) under bf16 autocast -> CLS token and mean patch token.
Each rank writes <out>.rank{r}.npz with idx / cls / avg (float32)."""
import argparse, os, sys, numpy as np, torch
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--src', required=True); p.add_argument('--out', required=True)
p.add_argument('--n', type=int, default=50000); p.add_argument('--batch', type=int, default=250)
p.add_argument('--fmt', default='{}.png')   # iMF saves {:05d}.png
a = p.parse_args()
sys.path.insert(0, a.repo)
from frechet_distance.repr_models import load_repr_model
r, W = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
idx = np.array_split(np.arange(a.n), W)[r]

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(idx)
    def __getitem__(self, k):
        return torch.from_numpy(np.asarray(Image.open(os.path.join(a.src, a.fmt.format(int(idx[k])))).convert('RGB'))).permute(2, 0, 1)

net, dim, _, ts = load_repr_model('vit_large_patch14_dinov2.lvd142m', device='cuda', target_size=256)
assert ts == 256, ts
cls, avg = [], []
dl = torch.utils.data.DataLoader(DS(), batch_size=a.batch, num_workers=8, pin_memory=True)
with torch.inference_mode():
    for x in dl:
        x = x.cuda(non_blocking=True).float() / 255.0
        with torch.autocast('cuda', dtype=torch.bfloat16):
            c, m = net(x)
        cls.append(c.float().cpu()); avg.append(m.float().cpu())
np.savez(f'{a.out}.rank{r}.npz', idx=idx, cls=torch.cat(cls).numpy(), avg=torch.cat(avg).numpy())
print(f'rank {r}: {len(idx)} images done', flush=True)

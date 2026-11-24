"""DINOv2-L (FD-Loss protocol) features of the 10k ADM/VIRTUAL reference images, saved; then improved precision/recall
(Kynkaanniemi k=3, FD-Loss metrics.compute_precision_recall) of the 50k sample features vs these 10k reals."""
import sys, json, numpy as np, torch
REPO = '/localscratch/cva19/AdaIMLE-tmp/fdloss/repo'; sys.path.insert(0, REPO)
from frechet_distance.repr_models import load_repr_model
from frechet_distance.metrics import compute_precision_recall
D = '/localscratch/cva19/AdaIMLE/fid_refs/dinov2_feats'
REF = f'{D}/virtual-ref10k-dinov2L-t256.npz'
try:
    r = np.load(REF); rc, ra = r['cls'], r['avg']
except FileNotFoundError:
    imgs = np.load('/localscratch/cva19/AdaIMLE/evaluation/VIRTUAL_imagenet256_labeled.npz')['arr_0']
    net, _, _, ts = load_repr_model('vit_large_patch14_dinov2.lvd142m', device='cuda', target_size=256); assert ts == 256
    c, m = [], []
    with torch.inference_mode():
        for s in range(0, len(imgs), 250):
            x = torch.from_numpy(imgs[s:s + 250]).cuda().permute(0, 3, 1, 2).float() / 255.0
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a, b = net(x)
            c.append(a.float().cpu()); m.append(b.float().cpu())
    rc, ra = torch.cat(c).numpy(), torch.cat(m).numpy()
    np.savez(REF, cls=rc, avg=ra)
g = np.load(f'{D}/fd-iter-60000-dinov2L-t256.npz')
res = dict(n_real=len(rc), n_gen=len(g['cls']), k=3)
for name, R, G in (('cls', rc, g['cls']), ('avg', ra, g['avg'])):
    p, rec = compute_precision_recall(torch.from_numpy(R).cuda(), torch.from_numpy(G).cuda(), k=3, batch_size=5000)
    res[f'precision_{name}'], res[f'recall_{name}'] = float(p), float(rec)
print(json.dumps(res))

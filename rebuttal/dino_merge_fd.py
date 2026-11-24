"""Merge rank shards -> one features npz (index order, + class label i % 1000); FD vs FD-Loss DINOv2 ref stats."""
import argparse, glob, json, os, sys, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--out', required=True); p.add_argument('--ref', required=True)
p.add_argument('--label_div', type=int, default=0)   # 0: class = idx % 1000 (ours); 50: class = idx // 50 (iMF)
a = p.parse_args()
sys.path.insert(0, a.repo)
from frechet_distance.metrics import compute_fid
sh = [np.load(f) for f in sorted(glob.glob(f'{a.out}.rank*.npz'))]
idx = np.concatenate([s['idx'] for s in sh]); o = np.argsort(idx)
cls = np.concatenate([s['cls'] for s in sh])[o]; avg = np.concatenate([s['avg'] for s in sh])[o]; idx = idx[o]
assert (idx == np.arange(len(idx))).all(), 'missing/duplicate indices'
np.savez(f'{a.out}.npz', idx=idx, label=(idx // a.label_div) if a.label_div else (idx % 1000), cls=cls, avg=avg)
ref = np.load(a.ref); res = dict(n=len(idx))
for name, f, mk, sk in (('dinov2_cls', cls, 'mu', 'sigma'), ('dinov2_avg', avg, 'avg_mu', 'avg_sigma')):
    f = f.astype(np.float64); mu, sig = f.mean(0), np.cov(f, rowvar=False)
    res[name] = float(compute_fid(mu, sig, ref[mk], ref[sk]))
res['fdr_dinov2_cls'] = res['dinov2_cls'] / 14.19      # FD-Loss README valFD normaliser for DINOv2
print(json.dumps(res))
for f in glob.glob(f'{a.out}.rank*.npz'): os.remove(f)

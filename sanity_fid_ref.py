"""
Sanity check for helpers/fid_score.py.

Runs the 10k reference images (arr_0) stored in the FID reference npz through the
pytorch-fid Inception extractor and computes FID against the stored mu/sigma in that same
npz. If the PyTorch extractor matches the network that produced the stored stats, this FID
should be close to 0 (a small positive value from the 10k-sample subset is expected).

Usage (in the project's torch env):
    python sanity_fid_ref.py [path/to/VIRTUAL_imagenet256_labeled.npz]
"""

import sys
import numpy as np
import torch

from helpers.fid_score import (
    get_inception_model, inception_features, compute_fid_from_features,
)

ref = sys.argv[1] if len(sys.argv) > 1 else 'evaluation/VIRTUAL_imagenet256_labeled.npz'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Loading reference images from {ref} ...')
with np.load(ref) as obj:
    imgs = obj['arr_0']  # (N, 256, 256, 3) uint8
print(f'arr_0: {imgs.shape} {imgs.dtype}')

model = get_inception_model(device)
feats = inception_features(imgs, model, device, batch_size=50)
print(f'features: {tuple(feats.shape)}')

fid = compute_fid_from_features(feats, ref)
print(f'Self-FID (arr_0 vs stored mu/sigma): {fid:.4f}')
print('Expect a small value (< ~5 for the 10k subset). A large value flags a '
      'preprocessing/weights mismatch with the original stats.')

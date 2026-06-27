"""
Pure-PyTorch FID utilities.

This module computes FID against the *precomputed* reference statistics stored in
`evaluation/VIRTUAL_imagenet256_labeled.npz` (keys ``mu`` -> (2048,), ``sigma`` ->
(2048, 2048)). Those reference stats are read as-is and never recomputed.

The only thing computed on the fly is the statistics of the *generated* samples. To make
the two sets of statistics comparable, the generated samples are passed through the
pytorch-fid ported InceptionV3 (mseitzer, ``pt_inception-2015-12-05-6726825d.pth``), which
reproduces the ``pool_3`` (2048-d) activations of the original TF Inception graph that
produced the stored stats.

Sanity check: run the 10k reference images (``arr_0`` in the npz) through
``inception_features`` and compute FID against the stored ``mu``/``sigma``; the result
should be close to 0.
"""

import numpy as np
import torch
from scipy import linalg

try:
    from pytorch_fid.inception import InceptionV3
except Exception as _e:  # pragma: no cover
    try:
        # Optional vendored copy (drop-in of mseitzer/pytorch-fid inception.py).
        from helpers._inception import InceptionV3
    except Exception:
        raise ImportError(
            "FID requires the pytorch-fid InceptionV3 extractor. Install it with "
            "`pip install pytorch-fid` (it is listed in requirements.txt), or vendor "
            "mseitzer/pytorch-fid's inception.py at helpers/_inception.py."
        ) from _e


# Block index 3 -> final average pooling, 2048-d (pool_3 in the TF graph).
_POOL3_BLOCK_IDX = InceptionV3.BLOCK_INDEX_BY_DIM[2048]


def get_inception_model(device, weights_path=None):
    """
    Build the pytorch-fid InceptionV3 feature extractor (2048-d pool_3 output).

    If ``weights_path`` is given, those weights are loaded from disk instead of
    auto-downloading ``pt_inception-2015-12-05-6726825d.pth``.
    """
    if weights_path:
        try:
            model = InceptionV3([_POOL3_BLOCK_IDX], weights_path=weights_path)
        except TypeError:
            # Older pytorch-fid: no weights_path kwarg. Build default, then load weights.
            model = InceptionV3([_POOL3_BLOCK_IDX])
            state = torch.load(weights_path, map_location='cpu')
            model.load_state_dict(state)
    else:
        model = InceptionV3([_POOL3_BLOCK_IDX])
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def inception_features(images_uint8_nhwc, model, device, batch_size=50):
    """
    Compute 2048-d Inception pool_3 features for a batch of images.

    Args:
        images_uint8_nhwc: array/tensor of shape (N, H, W, 3), dtype uint8, range [0, 255]
            (exactly what ``sampler.sample`` returns).
        model: an InceptionV3 from ``get_inception_model``.
        device: torch device to run the network on.
        batch_size: forward-pass batch size.

    Returns:
        torch.FloatTensor of shape (N, 2048) on CPU.
    """
    if not torch.is_tensor(images_uint8_nhwc):
        images_uint8_nhwc = torch.from_numpy(np.ascontiguousarray(images_uint8_nhwc))

    n = images_uint8_nhwc.shape[0]
    feats = []
    for i in range(0, n, batch_size):
        batch = images_uint8_nhwc[i:i + batch_size].to(device, non_blocking=True)
        # NHWC uint8 [0,255] -> NCHW float [0,1]; InceptionV3 resizes to 299 internally.
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        pred = model(batch)[0]
        # InceptionV3 returns (N, 2048, 1, 1) for the pool_3 block.
        pred = pred.squeeze(3).squeeze(2)
        feats.append(pred.cpu())
    return torch.cat(feats, dim=0)


def compute_stats(features):
    """features: (N, D) tensor or array -> (mu (D,), sigma (D, D)) numpy float64."""
    if torch.is_tensor(features):
        features = features.cpu().numpy()
    features = features.astype(np.float64)
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def load_reference_stats(npz_path):
    """Read the precomputed reference ``mu``/``sigma`` from the npz (used as-is)."""
    with np.load(npz_path) as obj:
        if "mu" not in obj or "sigma" not in obj:
            raise KeyError(
                f"{npz_path} has no 'mu'/'sigma' keys; available: {list(obj.keys())}"
            )
        return obj["mu"].astype(np.float64), obj["sigma"].astype(np.float64)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Frechet distance between two Gaussians. Mirrors the well-tested
    evaluation/evaluator.py implementation (scipy.linalg.sqrtm with eps fallback).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        f"mean vectors have different lengths: {mu1.shape}, {mu2.shape}"
    )
    assert sigma1.shape == sigma2.shape, (
        f"covariances have different dimensions: {sigma1.shape}, {sigma2.shape}"
    )

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_fid_from_features(features, ref_npz_path):
    """Compute FID of generated ``features`` (N, 2048) against the reference npz stats."""
    mu_gen, sigma_gen = compute_stats(features)
    mu_ref, sigma_ref = load_reference_stats(ref_npz_path)
    return frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)

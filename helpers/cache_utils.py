"""
Utilities for on-disk caching of heavy pre-processing artefacts:
  - image tensor materialisation   (data.py / fewshot / stl10)
  - latent projection tensors      (sampler.py / init_projection)
"""
import hashlib
import json
import os

import torch


def _stable_hash(d: dict, length: int = 16) -> str:
    """Return a short hex digest for a dict of scalar values."""
    canonical = json.dumps(d, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def _atomic_save(tensor: torch.Tensor, final_path: str):
    tmp_path = final_path + '.tmp'
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, final_path)  # atomic on POSIX


def _canonical_ae_path(autoencoder_type: str, autoencoder_name_or_path: str) -> str:
    """Mirror the defaulting logic in helpers/autoencoder.py:load_autoencoder."""
    if autoencoder_name_or_path:
        return autoencoder_name_or_path
    defaults = {
        'tiny':       'madebyollin/taesd',
        'eq-vae-ema': 'zelaki/eq-vae-ema',
        'vr-eq':      'Anzhc/MS-LC-EQ-D-VR_VAE',
        'eq-sdxl':    'KBlueLeaf/EQ-SDXL-VAE',
    }
    return defaults.get(autoencoder_type, 'zelaki/eq-vae')


# ---------------------------------------------------------------------------
# Image-cache helpers  (Problem: slow startup for large fewshot/stl10 datasets)
# ---------------------------------------------------------------------------

def image_cache_key(data_root: str, image_size: int, dataset_type: str,
                    sorted_by_class: bool = False,
                    cache_dataset_id: str = '') -> str:
    d = {
        'data_root':       cache_dataset_id if cache_dataset_id else os.path.abspath(data_root),
        'image_size':      int(image_size),
        'dataset_type':    str(dataset_type),
        'sorted_by_class': bool(sorted_by_class),
    }
    h = _stable_hash(d)
    return f"images_{dataset_type}_sz{image_size}_{h}"


def _image_cache_paths(cache_dir: str, key: str):
    return (
        os.path.join(cache_dir, f"{key}.pt"),
        os.path.join(cache_dir, f"{key}.meta"),
    )


def load_image_cache(cache_dir: str, key: str, expected_size: int | None = None):
    """Return the cached image tensor (or dict with 'images'/'labels'), or None on any miss."""
    pt_path, meta_path = _image_cache_paths(cache_dir, key)
    if not os.path.isfile(pt_path) or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        obj = torch.load(pt_path, map_location='cpu', weights_only=True)
        # obj can be a plain tensor (legacy) or a dict {'images': ..., 'labels': ...}
        size = obj['images'].shape[0] if isinstance(obj, dict) else obj.shape[0]
        cached_size = meta.get('dataset_size')
        if cached_size is not None and size != int(cached_size):
            print(f"[cache] Image meta/tensor size disagreement. Rebuilding.")
            return None
        if expected_size is not None and size != expected_size:
            print(f"[cache] Image cache size mismatch "
                  f"(cached {size} != expected {expected_size}). Rebuilding.")
            return None
        return obj
    except Exception as e:
        print(f"[cache] Could not load image cache ({e}). Rebuilding.")
        return None


def save_image_cache(cache_dir: str, key: str, obj):
    """Save image cache. obj is either a Tensor or a dict {'images': Tensor, 'labels': Tensor}."""
    os.makedirs(cache_dir, exist_ok=True)
    pt_path, meta_path = _image_cache_paths(cache_dir, key)
    try:
        size = obj['images'].shape[0] if isinstance(obj, dict) else obj.shape[0]
        _atomic_save(obj, pt_path)
        with open(meta_path, 'w') as f:
            json.dump({'dataset_size': size}, f)
        print(f"[cache] Saved image cache -> {pt_path}")
    except Exception as e:
        print(f"[cache] Failed to save image cache ({e}). Continuing without cache.")


# ---------------------------------------------------------------------------
# Latent-projection-cache helpers  (Problem: slow init_projection every run)
# ---------------------------------------------------------------------------

def latent_cache_key(
    data_root: str,
    dataset_type: str,
    image_size: int,
    latent_spatial_size: int,
    autoencoder_type: str,
    autoencoder_name_or_path: str,
    image_channels: int,
    num_classes: int = 0,
    sorted_by_class: bool = False,
    cache_dataset_id: str = '',
) -> str:
    d = {
        'data_root':          cache_dataset_id if cache_dataset_id else os.path.abspath(data_root),
        'dataset_type':       str(dataset_type),
        'image_size':         int(image_size),
        'latent_spatial_size': int(latent_spatial_size),
        'ae_path':            _canonical_ae_path(autoencoder_type, autoencoder_name_or_path),
        'image_channels':     int(image_channels),
        'num_classes':        int(num_classes),
        'sorted_by_class':    bool(sorted_by_class),
    }
    h = _stable_hash(d)
    return (
        f"latents_{dataset_type}_sz{image_size}"
        f"_lss{latent_spatial_size}_ae{autoencoder_type}_{h}"
    )


def _latent_cache_paths(cache_dir: str, key: str):
    return (
        os.path.join(cache_dir, f"{key}.pt"),
        os.path.join(cache_dir, f"{key}.meta"),
    )


def load_latent_cache(cache_dir: str, key: str, expected_size: int | None = None):
    """Return the cached latent projection tensor, or None on any miss / mismatch."""
    pt_path, meta_path = _latent_cache_paths(cache_dir, key)
    if not os.path.isfile(pt_path) or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        tensor = torch.load(pt_path, map_location='cpu', weights_only=True)
        cached_size = meta.get('dataset_size')
        if cached_size is not None and tensor.shape[0] != int(cached_size):
            print(f"[cache] Latent meta/tensor size disagreement. Rebuilding.")
            return None
        if expected_size is not None and tensor.shape[0] != expected_size:
            print(f"[cache] Latent cache size mismatch "
                  f"(cached {tensor.shape[0]} != expected {expected_size}). Rebuilding.")
            return None
        return tensor
    except Exception as e:
        print(f"[cache] Could not load latent cache ({e}). Rebuilding.")
        return None


def save_latent_cache(cache_dir: str, key: str, tensor: torch.Tensor):
    os.makedirs(cache_dir, exist_ok=True)
    pt_path, meta_path = _latent_cache_paths(cache_dir, key)
    try:
        _atomic_save(tensor, pt_path)
        with open(meta_path, 'w') as f:
            json.dump({'dataset_size': tensor.shape[0]}, f)
        print(f"[cache] Saved latent cache -> {pt_path}")
    except Exception as e:
        print(f"[cache] Failed to save latent cache ({e}). Continuing without cache.")

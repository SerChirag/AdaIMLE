"""On-disk caching for dataset latent projections."""
import hashlib
import json
import os

import torch


def latent_cache_key(
    data_root,
    dataset_type,
    image_size,
    latent_spatial_size,
    autoencoder_type,
    autoencoder_name_or_path,
    image_channels,
    num_classes=0,
    sorted_by_class=False,
    cache_dataset_id='',
    search_type='l2',
    proj_dim=0,
    lpips_net='',
    proj_proportion=0,
    l2_search_downsample=1.0,
):
    root_token = cache_dataset_id if cache_dataset_id else data_root
    payload = json.dumps({
        'root': root_token,
        'dataset': dataset_type,
        'image_size': image_size,
        'latent_spatial_size': latent_spatial_size,
        'ae_type': autoencoder_type,
        'ae_path': autoencoder_name_or_path,
        'channels': image_channels,
        'num_classes': num_classes,
        'sorted_by_class': sorted_by_class,
        'search_type': search_type,
        'proj_dim': proj_dim if search_type != 'l2' else 0,
        'lpips_net': lpips_net if search_type == 'lpips' else '',
        'proj_proportion': proj_proportion if search_type == 'lpips' else 0,
        'l2_search_downsample': l2_search_downsample if search_type != 'l2' else 1.0,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_latent_cache(cache_dir, key, expected_size):
    path = os.path.join(cache_dir, f'proj_{key}.pt')
    if not os.path.exists(path):
        return None
    try:
        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            meta = json.loads(open(meta_path).read())
            if meta.get('size') != expected_size:
                return None
        tensor = torch.load(path, map_location='cpu', weights_only=True)
        if tensor.shape[0] != expected_size:
            return None
        return tensor
    except Exception:
        return None


def save_latent_cache(cache_dir, key, tensor):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'proj_{key}.pt')
    tmp_path = path + '.tmp'
    torch.save(tensor.cpu(), tmp_path)
    os.replace(tmp_path, path)
    meta = json.dumps({'size': tensor.shape[0]})
    open(path + '.meta.json', 'w').write(meta)

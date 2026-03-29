import torch
import torch.nn.functional as F
from contextlib import nullcontext

from diffusers import AutoencoderKL, AutoencoderTiny
from huggingface_hub import hf_hub_download


def _fp32_vae_context(tensor):
    if torch.is_tensor(tensor) and tensor.is_cuda:
        return torch.autocast(device_type='cuda', enabled=False)
    return nullcontext()


def load_autoencoder(H, device):
    model_type = getattr(H, 'autoencoder_type', 'kl')
    model_path = getattr(H, 'autoencoder_name_or_path', '')
    subfolder = getattr(H, 'autoencoder_subfolder', '')

    # Default model paths: tiny AE, standard EQ-VAE variants, VR-EQ, or EQ-SDXL.
    if not model_path:
        if model_type == 'tiny':
            model_path = 'madebyollin/taesd'
        elif model_type == 'eq-vae-ema':
            model_path = 'zelaki/eq-vae-ema'
        elif model_type == 'vr-eq':
            model_path = 'Anzhc/MS-LC-EQ-D-VR_VAE'
        elif model_type == 'eq-sdxl':
            model_path = 'KBlueLeaf/EQ-SDXL-VAE'
        else:  # kl, eqvae, eq-vae
            model_path = 'zelaki/eq-vae'

    if model_type == 'tiny':
        ae = AutoencoderTiny.from_pretrained(model_path)
    elif model_type == 'vr-eq':
        # This repo ships standalone safetensors weights (no root config.json),
        # so we must load it as a single-file VAE.
        if model_path.endswith('.safetensors'):
            single_file_path = model_path
        else:
            single_file_name = 'MS-LC-EQ-D-VR VAE.safetensors'
            single_file_path = hf_hub_download(model_path, single_file_name)
        ae = AutoencoderKL.from_single_file(single_file_path)
    elif model_type in ('kl', 'eqvae', 'eq-vae', 'eq-vae-ema', 'eq-sdxl'):
        kwargs = {}
        if subfolder:
            kwargs['subfolder'] = subfolder
        ae = AutoencoderKL.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(f'Unsupported autoencoder_type: {model_type}')

    ae = ae.to(device)
    if bool(getattr(H, 'use_channels_last', True)) and torch.cuda.is_available():
        ae = ae.to(memory_format=torch.channels_last)
    ae.eval()
    ae.requires_grad_(False)
    ae._cached_scaling_factor = float(getattr(getattr(ae, 'config', None), 'scaling_factor', 1.0))
    return ae


def _extract_latents(encoded):
    if hasattr(encoded, 'latent_dist') and encoded.latent_dist is not None:
        return encoded.latent_dist.sample()
    if hasattr(encoded, 'latents'):
        return encoded.latents
    if isinstance(encoded, (tuple, list)):
        return encoded[0]
    return encoded


def _extract_sample(decoded):
    if hasattr(decoded, 'sample'):
        return decoded.sample
    if isinstance(decoded, (tuple, list)):
        return decoded[0]
    return decoded


def encode_images_to_latents(autoencoder, images_chw, target_spatial=None):
    if autoencoder is None:
        return images_chw
    if images_chw.is_cuda and images_chw.ndim == 4:
        images_chw = images_chw.contiguous(memory_format=torch.channels_last)

    with torch.inference_mode():
        with _fp32_vae_context(images_chw):
            encoded = autoencoder.encode(images_chw.float())
        latents = _extract_latents(encoded)
    scaling_factor = getattr(autoencoder, '_cached_scaling_factor', 1.0)
    latents = latents * scaling_factor

    if target_spatial is not None and (latents.shape[-2], latents.shape[-1]) != tuple(target_spatial):
        latents = F.interpolate(latents, size=target_spatial, mode='bicubic', align_corners=False)

    return latents


def decode_latents_to_images(autoencoder, latents_chw, latent_spatial=None):
    if autoencoder is None:
        return latents_chw

    latents = latents_chw
    if latent_spatial is not None and (latents.shape[-2], latents.shape[-1]) != tuple(latent_spatial):
        latents = F.interpolate(latents, size=latent_spatial, mode='bicubic', align_corners=False)
    if latents.is_cuda and latents.ndim == 4:
        latents = latents.contiguous(memory_format=torch.channels_last)

    scaling_factor = getattr(autoencoder, '_cached_scaling_factor', 1.0)
    with torch.inference_mode():
        with _fp32_vae_context(latents):
            decoded = autoencoder.decode((latents / scaling_factor).float())
        images = _extract_sample(decoded)

    return torch.clamp(images, -1.0, 1.0)

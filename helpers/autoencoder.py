import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL, AutoencoderTiny


def load_autoencoder(H, device):
    model_type = getattr(H, 'autoencoder_type', 'tiny')
    model_path = getattr(H, 'autoencoder_name_or_path', 'madebyollin/taesd')
    subfolder = getattr(H, 'autoencoder_subfolder', '')

    if model_type == 'tiny':
        ae = AutoencoderTiny.from_pretrained(model_path)
    elif model_type == 'kl':
        kwargs = {}
        if subfolder:
            kwargs['subfolder'] = subfolder
        ae = AutoencoderKL.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(f'Unsupported autoencoder_type: {model_type}')

    ae = ae.to(device)
    ae.eval()
    ae.requires_grad_(False)
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

    with torch.no_grad():
        encoded = autoencoder.encode(images_chw)
        latents = _extract_latents(encoded)

    scaling_factor = getattr(getattr(autoencoder, 'config', None), 'scaling_factor', 1.0)
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

    scaling_factor = getattr(getattr(autoencoder, 'config', None), 'scaling_factor', 1.0)
    with torch.no_grad():
        decoded = autoencoder.decode(latents / scaling_factor)
        images = _extract_sample(decoded)

    return images

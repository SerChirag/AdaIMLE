import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DinoSpatial(nn.Module):
    """Spatial DINO perceptual distance (frozen backbone, no learned weights).

    Both images are resized to ``size`` (a multiple of the model's patch size), ImageNet-
    normalized and run through the backbone. The distance is 1 - cos between the matching
    last-layer patch tokens, averaged over positions, so it compares layout as well as
    content -- unlike a CLS / mean-pooled embedding. Each token is unit-normalized first.

    Any DINO checkpoint AutoModel can load works; the CLS and register tokens are skipped
    using the checkpoint's own config:
      facebook/dinov2-base                  patch 14, no registers (size 224)
      facebook/dinov2-with-registers-base   patch 14, 4 registers  (size 224; cleaner patch maps)
      DINOv3 ViT (patch 16, registers)      size 256 = native for 256px images; needs a newer
                                            transformers than 4.50
    Without registers, DINOv2 has a few high-norm artifact tokens carrying global rather than
    local information, which is why the registers variant is preferred for this loss.

    Inputs to ``forward`` are NCHW images in [-1, 1]; the return is a per-image distance of
    shape (B,) in [0, 2]. Run in fp32 (no autocast).
    """

    def __init__(self, path='facebook/dinov2-base', size=224):
        super().__init__()
        self.model = AutoModel.from_pretrained(path).eval()
        cfg = self.model.config
        self.patch_size = int(cfg.patch_size)
        self.num_skip = 1 + int(getattr(cfg, 'num_register_tokens', 0) or 0)  # CLS + registers
        if size % self.patch_size != 0:
            raise ValueError(f'--dino_size {size} is not a multiple of the patch size {self.patch_size} of {path}')
        self.size = size
        self.num_patches = (size // self.patch_size) ** 2
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        for p in self.parameters():
            p.requires_grad_(False)

    def _tokens(self, x):
        x = (x.clamp(-1, 1) + 1) / 2
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bicubic', antialias=True, align_corners=False)
        x = (x - self.mean) / self.std
        tok = self.model(pixel_values=x).last_hidden_state[:, self.num_skip:]   # (B, num_patches, C)
        if tok.shape[1] != self.num_patches:
            raise RuntimeError(f'expected {self.num_patches} patch tokens after skipping {self.num_skip} '
                               f'CLS/register tokens, got {tok.shape[1]}')
        return F.normalize(tok, dim=-1)

    def forward(self, a, b):   # a, b in [-1, 1], NCHW
        return 1 - (self._tokens(a) * self._tokens(b)).sum(-1).mean(1)   # (B,)


def load_dino_spatial(device, path='facebook/dinov2-base', size=224):
    """Build the spatial DINO distance module on ``device`` in eval mode."""
    return DinoSpatial(path=path, size=size).to(device).eval()

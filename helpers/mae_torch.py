# Vendored verbatim from ../drifting/mae_torch.py (PyTorch port of the Drifting MAE-ResNet, verified
# against the official JAX model there by check_mae_jax_vs_torch.py). Keep in sync by copying.
"""PyTorch port of the Drifting MAE-ResNet (official JAX: official/models/mae_model.py).

Tensors are NCHW. Module names mirror the Flax param tree so JAX checkpoints map 1:1
(see convert_mae_jax_to_torch.py). Flax GroupNorm uses eps=1e-6, which is kept here.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

GN_EPS = 1e-6  # flax.linen.GroupNorm default


def _choose_gn_groups(num_channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g != 0):
        g -= 1
    return max(g, 1)


def _gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(_choose_gn_groups(num_channels, max_groups), num_channels, eps=GN_EPS)


class _BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs with GroupNorm, ReLU, and residual skip."""

    def __init__(self, in_channels: int, filters: int, stride: int = 1, dropout_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.gn2 = _gn(filters)
        self.drop = nn.Dropout(dropout_prob)
        # Flax only materializes proj_* when the residual shape changes.
        self.has_proj = stride != 1 or in_channels != filters
        if self.has_proj:
            self.proj_conv = nn.Conv2d(in_channels, filters, 1, stride=stride, bias=False)
            self.proj_gn = _gn(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = F.relu(self.gn1(self.conv1(x)))
        y = self.drop(y)
        y = self.gn2(self.conv2(y))
        if self.has_proj:
            residual = self.proj_gn(self.proj_conv(residual))
        return F.relu(residual + y)


class _ResNetEncoder(nn.Module):
    """4-stage ResNet encoder producing multi-scale feature maps {conv1, layer1..layer4}."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        layers: Sequence[int] = (2, 2, 2, 2),
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, stride=1, padding=1, bias=False)
        self.gn1 = _gn(base_channels)

        ch = base_channels
        stages = []
        prev_ch = ch
        for stage_idx, num_blocks in enumerate(layers):
            stride = 2 if stage_idx > 0 else 1
            out_ch = ch * (2 ** stage_idx)
            blocks = [_BasicBlock(prev_ch, out_ch, stride=stride, dropout_prob=dropout_prob)]
            blocks += [_BasicBlock(out_ch, out_ch, stride=1, dropout_prob=dropout_prob) for _ in range(1, num_blocks)]
            stages.append(nn.Sequential(*blocks))
            setattr(self, f"layer{stage_idx + 1}_norm", _gn(out_ch))
            prev_ch = out_ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor, return_block_outputs: bool = False):
        feats: Dict[str, torch.Tensor] = {}
        block_outputs: Dict[str, List[torch.Tensor]] = {}
        x = F.relu(self.gn1(self.conv1(x)))
        feats["conv1"] = x

        for i, blocks in enumerate(self.stages):
            layer_name = f"layer{i + 1}"
            outs: List[torch.Tensor] = []
            for block in blocks:
                x = block(x)
                outs.append(x)
            block_outputs[layer_name] = outs
            x = getattr(self, f"{layer_name}_norm")(x)
            feats[layer_name] = x
        if return_block_outputs:
            return feats, block_outputs
        return feats


class _ConvGNReLU(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel, padding=kernel // 2, bias=False)
        self.gn = _gn(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.concat_norm_fn = nn.GroupNorm(32, in_channels + skip_channels, eps=GN_EPS)
        self.proj = _ConvGNReLU(in_channels + skip_channels, out_channels)
        self.refine = _ConvGNReLU(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # jax.image.resize(..., "bilinear") == half-pixel bilinear for upsampling.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_norm_fn(x)
        x = self.proj(x)
        x = self.refine(x)
        return x


class _UNetDecoder(nn.Module):
    def __init__(self, base_channels: int, out_channels: int):
        super().__init__()
        c1 = base_channels
        c2 = base_channels
        c3 = base_channels * 2
        c4 = base_channels * 4
        c5 = base_channels * 8
        self.bridge = _ConvGNReLU(c5, c5)
        self.up43 = _UpBlock(c5, c4, c4)
        self.up32 = _UpBlock(c4, c3, c3)
        self.up21 = _UpBlock(c3, c2, c2)
        self.up10 = _UpBlock(c2, c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 1)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.bridge(feats["layer4"])
        x = self.up43(x, feats["layer3"])
        x = self.up32(x, feats["layer2"])
        x = self.up21(x, feats["layer1"])
        x = self.up10(x, feats["conv1"])
        return self.head(x)


def patch_input(x: torch.Tensor, input_patch_size: int) -> torch.Tensor:
    # Channel order (h2 w2 c) matches the JAX NHWC "b (h1 h2) (w1 w2) c -> b h1 w1 (h2 w2 c)".
    return rearrange(x, "b c (h1 h2) (w1 w2) -> b (h2 w2 c) h1 w1", h2=input_patch_size, w2=input_patch_size)


def unpatch_output(x: torch.Tensor, input_patch_size: int) -> torch.Tensor:
    return rearrange(x, "b (h2 w2 c) h1 w1 -> b c (h1 h2) (w1 w2)", h2=input_patch_size, w2=input_patch_size)


def make_patch_mask(x: torch.Tensor, mask_ratio: torch.Tensor, patch_size: int = 4, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Returns a (B, 1, H, W) mask with 1 = masked, sampled per ``patch_size`` x ``patch_size`` patch."""
    b, _, h, w = x.shape
    noise = torch.rand((b, h // patch_size, w // patch_size), device=x.device, generator=generator)
    mask = (noise < mask_ratio[:, None, None]).to(x.dtype)
    mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return mask[:, None]


def safe_std(x: torch.Tensor, dim, eps: float = 1e-6, keepdim: bool = False) -> torch.Tensor:
    x32 = x.float()
    mean = x32.mean(dim=dim, keepdim=True)
    var = ((x32 - mean) ** 2).mean(dim=dim, keepdim=keepdim)
    return torch.sqrt(var.clamp_min(0.0) + eps)


class MAEResNet(nn.Module):
    """PyTorch equivalent of ``MAEResNetJAX``. Inputs are ``(B, C, H, W)``."""

    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        base_channels: int = 64,
        patch_size: int = 4,
        dropout_prob: float = 0.0,
        layers: Sequence[int] = (2, 2, 2, 2),
        use_bf16: bool = False,  # accepted for config compatibility; use torch.autocast for bf16
        input_patch_size: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.input_patch_size = input_patch_size
        enc_in = in_channels * input_patch_size * input_patch_size
        self.encoder = _ResNetEncoder(enc_in, base_channels, tuple(layers), dropout_prob)
        self.decoder = _UNetDecoder(base_channels, enc_in)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        *,
        lambda_cls: float = 0.0,
        mask_ratio_min: float = 0.75,
        mask_ratio_max: float = 0.75,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = patch_input(x, self.input_patch_size)
        b = x.shape[0]
        mask_ratio = torch.rand((b,), device=x.device, generator=generator) * (mask_ratio_max - mask_ratio_min) + mask_ratio_min
        mask = make_patch_mask(x, mask_ratio, self.patch_size, generator=generator)
        x_in = x * (1.0 - mask)

        feats = self.encoder(x_in)
        logits = self.fc(feats["layer4"].mean(dim=(2, 3)))
        recon = self.decoder(feats)

        cls_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        mse = (recon - x) ** 2
        # Denominator counts masked pixels (not pixels x channels), as in the JAX loss.
        recon_loss = (mse * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-8)
        loss = lambda_cls * cls_loss + (1.0 - lambda_cls) * recon_loss
        metrics = {
            "loss": loss,
            "cls_loss": cls_loss,
            "recon_loss": recon_loss,
            "accuracy": (logits.argmax(dim=-1) == labels).to(x.dtype),
            "mask_ratio": mask.mean(dim=(1, 2, 3)),
        }
        return loss, metrics

    def encode_decode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic (recon, logits) for an already-masked input; recon is unpatchified."""
        xp = patch_input(x, self.input_patch_size)
        feats = self.encoder(xp)
        logits = self.fc(feats["layer4"].mean(dim=(2, 3)))
        recon = unpatch_output(self.decoder(feats), self.input_patch_size)
        return recon, logits

    def get_activations(
        self,
        x: torch.Tensor,
        *,
        patch_mean_size: Optional[List[int]] = [2, 4],
        patch_std_size: Optional[List[int]] = [2, 4],
        use_std: bool = True,
        use_mean: bool = True,
        every_k_block: float = 2,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features for the drift loss.

        Returns a dict of ``(B, T, D)`` tensors with the same keys and token order as
        ``MAEResNetJAX.get_activations``.
        """
        patch_mean_size = patch_mean_size or []
        patch_std_size = patch_std_size or []

        x = patch_input(x, self.input_patch_size)
        need_blocks = isinstance(every_k_block, (int, float)) and not math.isinf(float(every_k_block)) and every_k_block >= 1
        if need_blocks:
            feats, block_outputs = self.encoder(x, return_block_outputs=True)
        else:
            feats = self.encoder(x)
            block_outputs = {}

        out: Dict[str, torch.Tensor] = {}
        out["norm_x"] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[:, None, :]

        def process_feat(name: str, feat: torch.Tensor) -> None:
            _, _, h, w = feat.shape
            out[name] = rearrange(feat, "b c h w -> b (h w) c")
            if use_mean:
                out[f"{name}_mean"] = feat.mean(dim=(2, 3))[:, None, :]
            if use_std:
                out[f"{name}_std"] = safe_std(feat, dim=(2, 3))[:, None, :]

            for size in patch_mean_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(feat, "b c (h s1) (w s2) -> b (h w) (s1 s2) c", s1=size, s2=size)
                    out[f"{name}_mean_{size}"] = reshaped.mean(dim=2)

            for size in patch_std_size:
                if h % size == 0 and w % size == 0:
                    reshaped = rearrange(feat, "b c (h s1) (w s2) -> b (h w) (s1 s2) c", s1=size, s2=size)
                    out[f"{name}_std_{size}"] = safe_std(reshaped, dim=2)

        for name, feat in feats.items():
            process_feat(name, feat)

        if need_blocks:
            k = int(every_k_block)
            for i in range(1, 5):
                lname = f"layer{i}"
                for blk_idx, feat_i in enumerate(block_outputs.get(lname, []), start=1):
                    if blk_idx % k == 0:
                        process_feat(f"{lname}_blk{blk_idx}", feat_i)

        return out


def drift_features(mae: Optional[MAEResNet], x: torch.Tensor, has_scale: bool = False, **kwargs) -> Dict[str, torch.Tensor]:
    """Port of ``activation_fn`` from ``build_activation_function`` (MAE branch only).

    ``global`` flattens in NHWC order so it matches the JAX feature exactly.
    """
    feats: Dict[str, torch.Tensor] = {}
    feats["global"] = x.permute(0, 2, 3, 1).reshape(x.shape[0], 1, -1)
    if has_scale:
        feats["norm_x"] = torch.sqrt((x ** 2).mean(dim=(2, 3)) + 1e-6)[:, None, :]
    if mae is not None:
        feats = {**feats, **mae.get_activations(x, **kwargs)}
    return feats


def load_mae_torch(path: str, map_location: Any = "cpu", encoder_only: bool = False) -> MAEResNet:
    """Load a checkpoint written by convert_mae_jax_to_torch.py."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = MAEResNet(**ckpt["model_config"])
    sd = ckpt["state_dict"]
    if encoder_only or not any(k.startswith("decoder.") for k in sd):
        sd = {k: v for k, v in sd.items() if k.startswith("encoder.")}
        model.decoder = None
        model.fc = None
    model.load_state_dict(sd, strict=True)
    return model.eval()

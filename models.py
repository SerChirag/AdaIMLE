import torch
from torch import nn
from torch.nn import functional as F

from mapping_network import MappingNetowrk, AdaptiveInstanceNorm, NoiseInjection
from helpers.imle_helpers import get_1x1, get_3x3, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools


class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Reduce dimensionality for queries and keys
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # Learnable scaling factor initialized as zero
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        # Compute query, key and value maps
        proj_query = self.query_conv(x).view(B, -1, H * W)  # [B, C//8, N]
        proj_key   = self.key_conv(x).view(B, -1, H * W)      # [B, C//8, N]
        proj_value = self.value_conv(x).view(B, -1, H * W)      # [B, C, N]
        
        # Compute attention map using matrix multiplication and softmax
        attention = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # [B, N, N]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to the value maps
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, N]
        out = out.view(B, C, H, W)
        
        # Apply scaling and residual connection
        out = self.gamma * out + x
        return out


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion=2, kernel_size=7):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw_conv1 = nn.Conv2d(dim, expansion * dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.pw_conv2 = nn.Conv2d(expansion * dim, dim, kernel_size=1)
    
    def forward(self, x):
        residual = x
        # Depthwise convolution with larger kernel
        x = self.dw_conv(x)
        # Permute to channels-last for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Permute back to channels-first
        x = x.permute(0, 3, 1, 2)
        # Pointwise conv to expand channels
        x = self.pw_conv1(x)
        x = self.gelu(x)
        # Pointwise conv to compress channels back
        x = self.pw_conv2(x)
        return x + residual  # Residual connection


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        if res <= H.max_hierarchy:
            self.noise = NoiseInjection(width)
        self.adaIN = AdaptiveInstanceNorm(width, H.latent_dim)
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.resnet = ConvNeXtBlock(width, kernel_size=7)
        self.use_attention = None
        if (res >= 16 and res <= 64):
            self.use_attention = True
            self.attention = SelfAttention(width)
        else:
            self.attention = None
        # self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

    def forward(self, x, w, spatial_noise):
        if self.mixin is not None:
            x = F.interpolate(x, scale_factor=self.base // self.mixin, mode='bicubic')
        if self.base <= self.H.max_hierarchy:
            x = self.noise(x, spatial_noise)
        x = self.adaIN(x, w)
        x = self.resnet(x)
        if self.use_attention:
            x = self.attention(x)
        return x




class Decoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.mapping_network = MappingNetowrk(code_dim=H.latent_dim, n_mlp=H.n_mpl)
        resos = set()
        cond_width = int(H.width * H.bottleneck_multiple)
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        first_res = self.resolutions[0]
        self.constant = nn.Parameter(torch.randn(1, self.widths[first_res], first_res, first_res))
        self.resnet = get_1x1(H.width, H.image_channels)
        self.gain = nn.Parameter(torch.ones(1, H.image_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.image_channels, 1, 1))

    def forward(self, latent_code, spatial_noise, input_is_w=False):
        if not input_is_w:
            w = self.mapping_network(latent_code)[0]
        else:
            w = latent_code
        
        x = self.constant.repeat(latent_code.shape[0], 1, 1, 1)

        for idx, block in enumerate(self.dec_blocks):
            noise = None
            x = block(x, w, noise)
        x = self.resnet(x)
        x = self.gain * x + self.bias
        return x


class IMLE(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.dci_db = None
        self.decoder = Decoder(H)

    def forward(self, latents, spatial_noise=None, input_is_w=False):
        return self.decoder.forward(latents, spatial_noise, input_is_w)


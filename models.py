import torch
from torch import nn
from torch.nn import functional as F

from mapping_network import MappingNetowrk, AdaptiveInstanceNorm, NoiseInjection
from helpers.imle_helpers import get_1x1
from collections import defaultdict
import numpy as np
import itertools
from torchvision.ops import StochasticDepth

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

def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, H, expansion=4, kernel_size=7, use_se=True, reduction=16, dropout=0.0):
        super().__init__()
        self.H = H
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-3)
        self.pw_conv1 = nn.Conv2d(dim, expansion * dim, kernel_size=1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.pw_conv2 = nn.Conv2d(expansion * dim, dim, kernel_size=1)

        ## single parameter for residual ratio
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(dim, reduction=reduction)  
        else:
            # Indentity layer if SE is not used
            self.se = nn.Identity()
        self.residual_ratio = nn.Parameter(torch.zeros(1))
    

    
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
        x = self.se(x)

        x = x * self.sigmoid(self.residual_ratio)

        return x + residual


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks, drop_path=0.0):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        self.adaIN = AdaptiveInstanceNorm(width, H.latent_dim)
        self.resnet = ConvNeXtBlock(width, H, kernel_size=7, 
                                    expansion=H.convnext_expansion, 
                                    use_se=H.use_se,
                                    reduction=H.se_reduction,
                                    dropout=drop_path)
        if mixin is not None:
            in_width = self.widths[mixin]
            out_width = self.widths[res]
            self.resnet_1x1 = get_1x1(in_width, out_width)



    def forward(self, x, w):
        if self.mixin is not None:
            x = F.interpolate(x, scale_factor=self.base / self.mixin, mode='bicubic')
            x = self.resnet_1x1(x)
            
        x = self.adaIN(x, w)
        x = self.resnet(x)
        return x

class Decoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.mapping_network = MappingNetowrk(code_dim=H.latent_dim, n_mlp=H.n_mpl, lr_multiplier=H.mapping_lr_multiplier)
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)

        total = 0
        for idx, (res, mixin) in enumerate(blocks):
            if mixin is None:
                total += 1

        dp_index = 0
        for idx, (res, mixin) in enumerate(blocks):
            if mixin is not None:
                dp_rate = 0.0
            else:
                dp_rate = H.drop_path_rate * (dp_index / total)
                dp_index += 1
            # print(f"Decoder block {idx}: res={res}, mixin={mixin}, drop_path_rate={dp_rate:.4f}")
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks), drop_path=dp_rate))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        first_res = self.resolutions[0]
        last_res = self.resolutions[-1]
        self.constant = nn.Parameter(torch.randn(1, self.widths[first_res], first_res, first_res))
        self.resnet = get_1x1(self.widths[last_res], H.image_channels)
        self.gain = nn.Parameter(torch.ones(1, H.image_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.image_channels, 1, 1))

    def forward(self, latent_code, input_is_w=False):
        if not input_is_w:
            w = self.mapping_network(latent_code)
        else:
            w = latent_code
        
        x = self.constant.repeat(latent_code.shape[0], 1, 1, 1)

        for idx, block in enumerate(self.dec_blocks):
            x = block(x, w)
        x = self.resnet(x)
        x = self.gain * x + self.bias
        return x


class IMLE(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.decoder = Decoder(H)

    def forward(self, latents, input_is_w=False):
        return self.decoder.forward(latents, input_is_w)


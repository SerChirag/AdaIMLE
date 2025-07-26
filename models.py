import torch
from torch import nn
from torch.nn import functional as F

from mapping_network import MappingNetowrk, AdaptiveInstanceNorm, NoiseInjection
from helpers.imle_helpers import get_1x1
from collections import defaultdict
import numpy as np
import itertools
from diffusers import UNet2DModel


class IMLE(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.decoder = UNet2DModel(
            sample_size=256,             # Height and width of the input image
            in_channels=3,               # RGB
            out_channels=3,              # Usually same as in_channels (for predicting noise)
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),  # Controls depth and size
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "UpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D", "UpBlock2D",
            )
        )

    def forward(self, latents, input_is_w=False):
        return self.decoder.forward(latents)

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


class IMLE(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.decoder = UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            layers_per_block=3,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D"
            )
        )

    def forward(self, latents, input_is_w=False):
        latent_reshaped = latents.view(latents.shape[0], 3, 32, 32)
        output = self.decoder.forward(latent_reshaped, timestep=0)
        return output.sample
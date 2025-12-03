import torch
from torch import nn
from torch.nn import functional as F

from mapping_network import MappingNetwork, AdaptiveInstanceNorm, NoiseInjection
from helpers.imle_helpers import get_1x1
from collections import defaultdict
import numpy as np
import itertools

from unet.unet import UNetModelWrapper


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
        self.dropout = nn.Dropout2d(p=dropout)  # <- NEW LINE

    
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

        # Apply dropout
        # x = self.dropout(x)
        # Pointwise conv to compress channels back
        x = self.pw_conv2(x)
        x = self.se(x)
        return x * self.sigmoid(self.residual_ratio) + residual


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
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
                                    dropout=H.dropout_p)

    def forward(self, x, w):
        if self.mixin is not None:
            x = F.interpolate(x, scale_factor=self.base / self.mixin, mode='bicubic')
        x = self.adaIN(x, w)
        x = self.resnet(x)
        return x

class Decoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.mapping_network = MappingNetwork(code_dim=H.latent_dim, n_mlp=H.n_mpl, lr_multiplier=H.mapping_lr_multiplier)
        resos = set()
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
        self.embedding = nn.Embedding(100, H.latent_dim)

    def forward(self, latent_code, condition):
        
        class_emb = self.embedding(condition)
        latent_code_2 = latent_code + class_emb
        w = self.mapping_network(latent_code_2)
        x = self.constant.repeat(latent_code_2.shape[0], 1, 1, 1)

        for idx, block in enumerate(self.dec_blocks):
            x = block(x, w)
        x = self.resnet(x)
        x = self.gain * x + self.bias
        return x


class IMLE(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        if(H.model_type == 'convnext'):
            self.decoder = Decoder(H)
        elif(H.model_type == 'unet'):
            # self.decoder = SongUNet(
            #     in_channels=3,
            #     out_channels=3,
            #     img_resolution=H.image_size,
            #     attn_resolutions = [8,16,32],
            #     model_channels = 192,
            #     label_dim = H.num_classes
            # )
            self.decoder = UNetModelWrapper(dim=(3, H.image_size, H.image_size), 
                num_channels=H.width, 
                num_res_blocks=3,
                attention_resolutions="8,16,32",
                num_classes=H.num_classes,
                class_cond = True
            )


    def forward(self, latents, condition):

        if(self.H.model_type == 'convnext'):
            return self.decoder.forward(latents, condition)
        
        elif(self.H.model_type == 'unet'):
            latents = latents.reshape(-1, 3, self.H.image_size, self.H.image_size)
            t = torch.randint(0, 1000, (latents.shape[0],)).to(latents.device)
            # class_label = torch.nn.functional.one_hot(condition, num_classes=self.H.num_classes).float()
            return self.decoder(t, latents, y = condition)



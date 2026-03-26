from math import sqrt

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-6)

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        # Keep scalar gains as plain Python floats to avoid Dynamo creating from_numpy tensors.
        self.weight_gain = float(lr_multiplier) / float(sqrt(in_features))
        self.bias_gain = float(lr_multiplier)

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        x = torch.addmm(b.unsqueeze(0), x, w.t())
        return x

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        # linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = linear

    def forward(self, input):
        return self.linear(input)

def normalize_2nd_moment(x, dim=1, eps=1e-6):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class MappingNetwork(nn.Module):
    def __init__(self, H):
        super().__init__()

        layers = [PixelNorm()]
        for i in range(H.n_mpl):
            layers.append(FullyConnectedLayer(H.latent_dim, H.latent_dim, lr_multiplier=H.mapping_lr_multiplier))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, input, **kwargs):
        
        x = self.style(input)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(in_channel, eps=1e-3)
        self.style = EqualLinear(style_dim, in_channel * 2)

        # if(H.zero_init):
        nn.init.zeros_(self.style.linear.bias)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        if input.shape[-1] > 1:
            out = self.norm(input)
        else:
            out = input

        out = (1 + gamma) * out + beta
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(1, channel, 1, 1), requires_grad=False)

    def forward(self, image, spatial_noise):
        return image 

from math import sqrt

import torch
from torch import nn
import numpy as np


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
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

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
    def __init__(self, code_dim=512, n_mlp=8, lr_multiplier=0.01):
        super().__init__()
        self.code_dim = code_dim

        assert n_mlp % 2 == 0, "n_mlp must be even (2 FCs per residual block)."

        layers = []
        for i in range(n_mlp):
            layers.append(FullyConnectedLayer(code_dim, code_dim, lr_multiplier=lr_multiplier))
            layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.ModuleList(layers)
        self.norm = PixelNorm()
        self.sigmoid = nn.Sigmoid()

        # One residual scale per block (each block = 2 FC layers)
        n_blocks = n_mlp // 2
        self.res_scale = nn.Parameter(torch.zeros(n_blocks))

    def forward(self, x):
        x = self.norm(x)

        block_idx = 0
        i = 0
        # Each block uses 4 entries in self.layers: (fc1, act1, fc2, act2)
        while i < len(self.layers):
            residual = x  # identity skip

            fc1 = self.layers[i]
            act1 = self.layers[i + 1]
            fc2 = self.layers[i + 2]
            act2 = self.layers[i + 3]

            out = fc1(x)
            out = act1(out)

            out = fc2(out)

            # ResNet-style: x + α * F(x)
            out = residual * self.sigmoid(self.res_scale[block_idx]) + out

            out = act2(out)

            x = out
            i += 4
            block_idx += 1

        return x



class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel, eps=1e-3)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = input
        if input.shape[3] > 1:
            out = self.norm(input)
        out = gamma * out + beta
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(1, channel, 1, 1), requires_grad=False)

    def forward(self, image, spatial_noise):
        return image 

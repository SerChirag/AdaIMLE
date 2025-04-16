import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import imageio
from visual.utils import get_sample_for_visualization, generate_for_NN, generate_visualization
from torch.utils.data import DataLoader, TensorDataset
from helpers.utils import ZippedDataset, get_cpu_stats_over_ranks


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma, eps):
    return torch.exp(logsigma) * eps + mu


def get_conv(in_dim, out_dim, kernel_size, stride, padding, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def reconstruct(H, sampler, imle, preprocess_fn, images, latents, snoise, name, logprint, training_step_imle):
    latent_optimizer = AdamW([latents], lr=H.latent_lr)
    generate_for_NN(sampler, images, latents.detach(), snoise, images.shape, imle,
                    f'{H.save_dir}/{name}-initial.png', logprint)
    for i in range(H.latent_epoch):
        for iter in range(H.reconstruct_iter_num):
            _, target = preprocess_fn([images])
            stat = training_step_imle(H, target.shape[0], target, latents, snoise, imle, None, latent_optimizer, sampler.calc_loss)

            latents.grad.zero_()
            if iter % 50 == 0:
                print('loss is: ', stat['loss'])
                generate_for_NN(sampler, images, latents.detach(), snoise, images.shape, imle,
                                f'{H.save_dir}/{name}-{iter}.png', logprint)

                torch.save(latents.detach(), '{}/reconstruct-latest.npy'.format(H.save_dir))
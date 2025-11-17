import torch
from torch import nn


class PseudoHuberLoss(nn.Module):
    """The Pseudo-Huber loss."""

    reductions = {'mean': torch.mean, 'sum': torch.sum, 'none': lambda x: x}
    
    def __init__(self, beta=1, reduction='mean'):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def extra_repr(self):
        return f'beta={self.beta:g}, reduction={self.reduction!r}'

    def forward(self, input, target):
        output = self.beta**2 * input.sub(target).div(self.beta).pow(2).add(1).sqrt().sub(1)
        return self.reductions[self.reduction](output)
    
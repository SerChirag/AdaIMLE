import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class LPIPSvgg(nn.Module):
    """Self-contained LPIPS(net='vgg').

    Reproduces the standard LPIPS-VGG perceptual distance using a torchvision VGG16
    backbone plus the learned 1x1 linear layers shipped with the repo
    (``lpips/weights/v0.1/vgg.pth``). No pip ``lpips`` and no downloads are required
    (the VGG16 ImageNet weights come from the local torch hub cache).

    Inputs to ``forward`` are NCHW images in [-1, 1]; the return is a per-image
    distance of shape (B,). Run in fp32 (no autocast).
    """

    def __init__(self, lin_path='lpips/weights/v0.1/vgg.pth'):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        # feature stacks ending at relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
        self.slices = nn.ModuleList()
        for a, b in [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(a, b)]))
        chns = [64, 128, 256, 512, 512]
        lin = torch.load(lin_path, map_location='cpu')
        self.lins = nn.ModuleList([nn.Conv2d(c, 1, 1, bias=False) for c in chns])
        for i, l in enumerate(self.lins):
            l.weight.data = lin[f'lin{i}.model.1.weight']
        # LPIPS input normalization (ImageNet-derived shift/scale).
        self.register_buffer('shift', torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.tensor([.458, .448, .450])[None, :, None, None])
        for p in self.parameters():
            p.requires_grad_(False)

    def _feat(self, x):
        x = (x - self.shift) / self.scale
        outs = []
        for s in self.slices:
            x = s(x)
            outs.append(x)
        return outs

    @staticmethod
    def _norm(t):
        return t / (t.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)

    def forward(self, a, b):   # a, b in [-1, 1], NCHW
        d = 0
        for fa, fb, lin in zip(self._feat(a), self._feat(b), self.lins):
            d = d + lin((self._norm(fa) - self._norm(fb)) ** 2).mean([2, 3])
        return d.squeeze(1)   # (B,)


def load_lpips_vgg(device, lin_path='lpips/weights/v0.1/vgg.pth'):
    """Build the LPIPS-VGG module on ``device`` in eval mode."""
    return LPIPSvgg(lin_path=lin_path).to(device).eval()

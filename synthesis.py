import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class SimpleSynthesisInput(nn.Module):
    def __init__(
        self,
        w_dim,
        channels,
        size,   
    ):
        super().__init__()
        self.channels = channels
        self.size = size
        freq_scale = size / 10


        # Fixed Fourier frequencies and phases
        self.register_buffer(
            "freqs",
            torch.randn(channels, 2) * freq_scale
        )
        self.register_buffer(
            "base_phase",
            torch.rand(channels) * 2 * math.pi
        )


        # Latent → phase modulation
        self.affine = nn.Linear(w_dim, channels, bias=True)

        # Learnable channel mixing
        self.weight = nn.Parameter(
            torch.randn(channels, channels) / math.sqrt(channels)
        )

        self.norm = nn.RMSNorm(channels, elementwise_affine=True)
        # self.norm = nn.Identity()

        # nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size),
            torch.linspace(-1, 1, size),
            indexing='ij'
        )
        coords = torch.stack([x, y], dim=-1)  # [H, W, 2]

        x = coords @ self.freqs.t()           # [H, W, C]
        x = 2 * math.pi * x

        self.register_buffer("x", x)


    def forward(self, w):
        """
        w: [B, w_dim]
        returns: [B, C, H, W]
        """

        # Coordinate grid in [-1, 1]
        
        # Phase modulation from latent
        phase = self.base_phase + self.norm(self.affine(w))    # [B, C]
        x = self.x.unsqueeze(0) + phase[:, None, None, :]

        x = torch.sin(x)

        # Channel mixing
        x = x @ self.weight.t()

        # [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x


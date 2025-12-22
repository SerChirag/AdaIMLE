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
        freq_scale=10.0,
    ):
        super().__init__()
        self.channels = channels
        self.size = size

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

        self.layernorm = nn.LayerNorm(channels)

        # nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)


    def forward(self, w):
        """
        w: [B, w_dim]
        returns: [B, C, H, W]
        """
        B = w.shape[0]
        H, W = self.size
        device = w.device

        # Coordinate grid in [-1, 1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        coords = torch.stack([x, y], dim=-1)  # [H, W, 2]

        x = coords @ self.freqs.t()           # [H, W, C]
        x = 2 * math.pi * x

        # Phase modulation from latent
        phase = self.base_phase + self.layernorm(self.affine(w))   # [B, C]
        x = x.unsqueeze(0) + phase[:, None, None, :]

        x = torch.sin(x)

        # Channel mixing
        x = x @ self.weight.t()

        # [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        return x




class ResolutionDrivenSynthesisInput(nn.Module):
    def __init__(
        self,
        w_dim,
        channels,
        resolution,          # single int, e.g. 8, 16, 32, 64
        bandwidth_factor=4,  # bandwidth = resolution / factor
    ):
        super().__init__()

        self.channels = channels
        self.resolution = resolution
        self.sampling_rate = resolution
        self.bandwidth = resolution / bandwidth_factor

        # ------------------------------------------------------------
        # Sample band-limited frequencies from uniform 2D disk
        # ------------------------------------------------------------
        freqs = torch.randn(channels, 2)
        freqs = freqs / freqs.norm(dim=1, keepdim=True)

        u = torch.rand(channels, 1)
        freqs = freqs * torch.sqrt(u)              # uniform disk
        freqs = freqs * self.bandwidth

        phases = torch.rand(channels) * 2 * math.pi

        self.register_buffer("freqs", freqs)
        self.register_buffer("phases", phases)

        # ------------------------------------------------------------
        # Latent → rotation + translation
        # ------------------------------------------------------------
        self.affine = nn.Linear(w_dim, 4, bias=True)

        nn.init.zeros_(self.affine.weight)
        self.affine.bias.data[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # ------------------------------------------------------------
        # Channel mixing
        # ------------------------------------------------------------
        self.weight = nn.Parameter(
            torch.randn(channels, channels) / math.sqrt(channels)
        )

    def forward(self, w):
        B = w.shape[0]
        H = W = self.resolution
        device = w.device

        # ------------------------------------------------------------
        # Coordinate grid
        # ------------------------------------------------------------
        y, x = torch.meshgrid(
            torch.linspace(-H / 2, H / 2, H, device=device),
            torch.linspace(-W / 2, W / 2, W, device=device),
            indexing="ij"
        )
        coords = torch.stack([x, y], dim=-1) / self.sampling_rate

        # ------------------------------------------------------------
        # Latent-conditioned transform
        # ------------------------------------------------------------
        t = self.affine(w)
        t = t / t[:, :2].norm(dim=1, keepdim=True)

        cos, sin = t[:, 0], t[:, 1]
        tx, ty = t[:, 2], t[:, 3]

        R = torch.stack([
            torch.stack([cos, -sin], dim=1),
            torch.stack([sin,  cos], dim=1),
        ], dim=1)

        freqs = self.freqs[None] @ R
        phases = self.phases[None] + (
            freqs @ torch.stack([tx, ty], dim=1).unsqueeze(2)
        ).squeeze(2)

        # ------------------------------------------------------------
        # Nyquist-aware damping
        # ------------------------------------------------------------
        freq_norm = freqs.norm(dim=2)
        nyquist = self.sampling_rate / 2
        amplitudes = (1 - (freq_norm - self.bandwidth) /
                      (nyquist - self.bandwidth)).clamp(0, 1)

        # ------------------------------------------------------------
        # Evaluate Fourier features
        # ------------------------------------------------------------
        # print(coords.shape, freqs.shape, phases.shape)
        x = torch.einsum("hwk,bck->bhwc", coords, freqs)
        # x = coords @ freqs.permute(0, 2, 1)
        x = 2 * math.pi * x + phases[:, None, None, :]
        x = torch.sin(x)
        x = x * amplitudes[:, None, None, :]

        # ------------------------------------------------------------
        # Channel mixing
        # ------------------------------------------------------------
        x = x @ self.weight.t()
        return x.permute(0, 3, 1, 2)

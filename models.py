import torch
import torch.nn as nn

from dit import DiT


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


class IMLEDiT(nn.Module):
    """DiT for IMLE: no timestep. The IMLE latent is reshaped to (B, C, H, W)
    and fed as x into the patch embedder. y is the class label.
    """
    def __init__(self, H):
        super().__init__()
        self.H = H
        num_classes = getattr(H, 'num_classes', 0)
        self.num_classes = num_classes

        # latent_dim must equal latent_channels * latent_spatial_size^2
        # e.g. latent_dim=128, latent_channels=4, latent_spatial_size=... user sets these
        self.latent_channels = H.image_channels
        self.latent_spatial = H.latent_spatial_size if H.latent_spatial_size > 0 else H.image_size

        self.dit = DiT(
            img_resolution=self.latent_spatial,
            patch_size=H.dit_patch_size,
            in_channels=self.latent_channels,
            hidden_size=H.dit_hidden_size,
            depth=H.dit_depth,
            num_heads=H.dit_num_heads,
            mlp_ratio=H.dit_mlp_ratio,
            class_dropout_prob=0.0,
            num_classes=max(num_classes, 1),
        )

    def forward(self, latents, condition=None, train=False):
        B = latents.shape[0]
        device = latents.device

        # Reshape flat latent to spatial: (B, C, H, W)
        x = latents.view(B, self.latent_channels, self.latent_spatial, self.latent_spatial)

        # Patch embed + positional encoding
        x = self.dit.x_embedder(x) + self.dit.pos_embed  # (B, T, hidden_size)

        # Class conditioning only
        if self.num_classes > 0 and condition is not None:
            c = self.dit.y_embedder(condition, self.training)
        else:
            dummy = torch.zeros(B, dtype=torch.long, device=device)
            c = self.dit.y_embedder(dummy, self.training)

        # Transformer blocks
        for block in self.dit.blocks:
            x = block(x, c)

        # Final layer + unpatchify -> (B, C, H, W)
        x = self.dit.final_layer(x, c)
        x = self.dit.unpatchify(x)

        if train:
            return [x]
        return x


class IMLE(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.decoder = IMLEDiT(H)

    def forward(self, latents, condition=None, train=False):
        return self.decoder(latents, condition, train)

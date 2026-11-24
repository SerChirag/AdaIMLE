import torch
import torch.nn as nn

from helpers.mae_torch import load_mae_torch


class MAEFeatureLoss(nn.Module):
    """Paired feature distance through Drifting's latent MAE-ResNet (frozen, no decode).

    The MAE was trained on ImageNet sd-vae-ft-mse latents (B, 4, 32, 32) scaled by 0.18215, so
    inputs must be latents of that VAE at that scale. Features are the 85 keys of
    ``MAEResNet.get_activations`` with the official drift-loss kwargs; each is a (B, T, D) set
    of per-position tokens (spatial maps, their means/stds and 2x2 / 4x4 patch statistics).

    Mirrors the per-feature normalization of the official drift_loss, applied to an IMLE
    (generated, matched target) pair instead of the drift field: per key, both sides are divided
    by a stop-gradient scale s = mean all-pairs gen-vs-target token distance / sqrt(D), so
    coordinates are O(1), and the key's loss is the mean squared matched-pair difference in that
    space (~1 for an unrelated pair, 0 at a match). The scale uses all B x B pairs, not only the
    matched ones: those shrink as pairs converge and would keep renormalizing the loss to ~1.
    Keys are averaged with equal weight (the official code sums them; only the coef differs).

    The official "global" key (the raw latent) is left out: that is the --latent_loss_coef term.
    Returns a scalar; gradients flow to ``gen`` only.

    For NN search (--mae_search), ``set_targets`` caches the data's features once and
    ``pairwise_dist`` scores pool samples against them with the same per-pair quantity, but with
    fixed per-key scales from all target-target pairs, so distances don't drift with the generator.
    """

    def __init__(self, path):
        super().__init__()
        self.mae = load_mae_torch(path, encoder_only=True)
        self.mae.requires_grad_(False)
        self.act_kwargs = dict(patch_mean_size=[2, 4], patch_std_size=[2, 4],
                               use_std=True, use_mean=True, every_k_block=2)
        self.tar_feats = None
        self.scales = None

    @torch.no_grad()
    def set_targets(self, tar, bs=50):
        """Cache the targets' features (fp32) and per-key s = mean target-target token distance
        over all pairs i != j, / sqrt(D)."""
        feats = [self.mae.get_activations(tar[i:i + bs], **self.act_kwargs) for i in range(0, len(tar), bs)]
        self.tar_feats = {k: torch.cat([f[k] for f in feats]).float() for k in feats[0]}
        n = len(tar)
        self.scales = {}
        for k, t in self.tar_feats.items():
            tt = t.transpose(0, 1)                                  # (T, N, D)
            dist = torch.cdist(tt, tt).sum() / (tt.shape[0] * n * (n - 1))
            self.scales[k] = (dist / t.shape[-1] ** 0.5).clamp_min(1e-3)

    @torch.no_grad()
    def pairwise_dist(self, x, bs=50):
        """(N, M) loss between each latent in ``x`` and each of the M cached targets: mean over keys
        of ||f(x) - f(t)||^2 / (T * D * s^2), i.e. forward's per-pair value at the fixed scales."""
        out = torch.zeros(len(x), len(next(iter(self.tar_feats.values()))), device=x.device)
        for i in range(0, len(x), bs):
            f = self.mae.get_activations(x[i:i + bs], **self.act_kwargs)
            for k, g in f.items():
                t = self.tar_feats[k].flatten(1)
                mu = t.mean(0)                                      # centre: less cancellation below
                g, t = g.float().flatten(1) - mu, t - mu
                d2 = (g.pow(2).sum(1, keepdim=True) + t.pow(2).sum(1)[None] - 2 * g @ t.T).clamp_min(0)
                out[i:i + bs] += d2 / (t.shape[1] * self.scales[k] ** 2)
        return out / len(self.tar_feats)

    def forward(self, gen, tar):
        with torch.no_grad():
            f_tar = self.mae.get_activations(tar, **self.act_kwargs)
        f_gen = self.mae.get_activations(gen, **self.act_kwargs)
        losses = []
        for k, g in f_gen.items():
            g, t = g.float(), f_tar[k].float()
            with torch.no_grad():
                # (T, B, D) x (T, B, D) -> (T, B, B): all pairs at the same token position.
                dist = torch.cdist(g.transpose(0, 1), t.transpose(0, 1)).mean()
                s = (dist / g.shape[-1] ** 0.5).clamp_min(1e-3)
            losses.append((((g - t) / s) ** 2).mean())
        return torch.stack(losses).mean()


def load_mae_feature_loss(device, path):
    """Build the latent-MAE feature loss on ``device`` in eval mode."""
    return MAEFeatureLoss(path).to(device).eval()

# Classic s27: the one-step IMLE DiT generator (IMLEDiT-S/2 for s27), with the s27 design only.
#
# Derived from facebookresearch/DiT (see LICENSE_DiT.txt), with block internals from improved MeanFlow
# (Lyy-iiis/imeanflow). This is the s27 code path of the experimental dit_models.py (commit 8cf3445, branch
# imle-latent-dit-new) with every other option removed. Parameter names and the order of initialization are
# unchanged, so s27 checkpoints load as they are and the same seed gives the same weights.
#
# z -> w (conditioning):
#   z (N, 128), RMS-normalized, split into 16 tokens of 8 -> linear to D + learned token embedding
#   [+ class embedding, added to every token, init std class_emb_init_std ~ the z tokens' scale]
#   -> 2 mixer blocks (self-attention, SwiGLU MLP; additive residuals with per-channel gates at 1)
#   -> mean over tokens -> RMS-normalize = w (N, D). The class enters before the mixer, so w is a nonlinear
#   function of (z, class), as with the conv generator's class embedding added to z before its mapping MLP.
# Image tokens:
#   a learned canvas, one token per 2x2 latent patch, init std 0.1 (small), the same for every z.
# Each of the `depth` blocks: self-attention (1-D RoPE, QK RMSNorm) -> SwiGLU MLP, each one gated branch:
#   h = RMSNorm(x) * (1 + gamma(w)) + beta(w);  f = RMSNorm_noscale(sublayer(h));  x = (1 - r) x + r f
#   r_j = 2 * sigmoid(logit_j + alpha_j(w)) / (j + 4) for branch j = 1..2*depth; logit 0 and alpha 0 at init,
#   so every branch starts with the same share of the output and none can exceed ~2x its share.
# Head: RMSNorm -> adaLN(w) -> linear (small, nonzero init) -> unpatchify to (N, C, H, W).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM = 128            # size of s27's z: 16 conditioning tokens x 8 numbers


def rms_normalize(x, eps=1e-6):
    """Scale each vector (last dim) to RMS 1, without learnable parameters."""
    return x * (x.square().mean(dim=-1, keepdim=True) + eps).rsqrt()


class LabelEmbedder(nn.Module):
    """Class labels -> vectors added to the conditioning tokens before the mixer. No label dropout: IMLE has no
    classifier-free guidance."""
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)


def precompute_rope_freqs(dim, seq_len, theta=10000.0):
    """1D RoPE over the raster-ordered image tokens (as in iMF). Returns cos, sin of shape (seq_len, dim // 2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)
    return angles.cos(), angles.sin()


def apply_rotary_pos_emb(x, cos, sin):
    """x: (N, T, heads, head_dim). Rotates consecutive channel pairs. Computed in fp32 and cast back."""
    x_ = x.float().unflatten(-1, (-1, 2))
    x0, x1 = x_[..., 0], x_[..., 1]
    cos = cos.float()[None, :, None, :]
    sin = sin.float()[None, :, None, :]
    out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
    return out.flatten(-2).to(x.dtype)


class Attention(nn.Module):
    """Multi-head self-attention with QK RMSNorm (iMF); RoPE when rope is given (image tokens only)."""
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x, rope=None):
        N, T, D = x.shape
        q = self.q_norm(self.q_proj(x).view(N, T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(N, T, self.num_heads, self.head_dim))
        v = self.v_proj(x).view(N, T, self.num_heads, self.head_dim)
        if rope is not None:
            q = apply_rotary_pos_emb(q, *rope)
            k = apply_rotary_pos_emb(k, *rope)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out_proj(x.transpose(1, 2).reshape(N, T, D))


class SwiGLUMlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ResidualGate(nn.Module):
    """
    Mixes the residual stream x with a normalized branch output f, with a per-channel gate.
      additive (mixer): x + g * f,             g initialized to gate_init (1)
      convex (trunk):   (1 - r) * x + r * f,   r = cap * sigmoid(logit + offset(w)); cap set by IMLEDiT
    """
    def __init__(self, hidden_size, residual_type, gate_init):
        super().__init__()
        assert residual_type in ('additive', 'convex')
        self.residual_type = residual_type
        self.cap = None
        self.gate = nn.Parameter(torch.full((hidden_size,), float(gate_init)))

    def forward(self, x, f, offset=None):
        """offset: optional per-sample (N, D) addition to the gate logit."""
        g = self.gate if offset is None else self.gate + offset.unsqueeze(1)
        if self.residual_type == 'additive':
            return x + f * g
        r = self.cap * g.sigmoid()
        return x * (1 - r) + f * r


class Branch(nn.Module):
    """RMSNorm [-> adaLN] -> sublayer -> RMSNorm (no learnable scale, so the gate is the true update size) -> gate."""
    def __init__(self, hidden_size, sublayer, residual_type, gate_init):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.sublayer = sublayer
        self.out_norm = nn.RMSNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.gate = ResidualGate(hidden_size, residual_type, gate_init)

    def forward(self, x, mod=None, **kwargs):
        """mod: optional (scale, shift, gate_offset), each (N, D), from w."""
        h = self.norm(x)
        offset = None
        if mod is not None:
            scale, shift, offset = mod
            h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.gate(x, self.out_norm(self.sublayer(h, **kwargs)), offset)


class IMLEDiTBlock(nn.Module):
    """
    Self-attention -> SwiGLU MLP, each a gated Branch. mod_dim > 0 (trunk blocks): a linear map of w gives each
    branch (scale, shift, gate offset). mod_dim = 0 (mixer blocks): no modulation.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio, residual_type, gate_init, mod_dim=0):
        super().__init__()
        self.self_attn = Branch(hidden_size, Attention(hidden_size, num_heads), residual_type, gate_init)
        self.mlp = Branch(hidden_size, SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio)), residual_type, gate_init)
        self.modulation = nn.Linear(mod_dim, 6 * hidden_size) if mod_dim > 0 else None

    def forward(self, x, rope=None, w=None):
        if self.modulation is None:
            return self.mlp(self.self_attn(x, rope=rope))
        m = self.modulation(w).chunk(6, dim=1)            # (scale, shift, offset) for attention, then MLP
        x = self.self_attn(x, mod=m[0:3], rope=rope)
        return self.mlp(x, mod=m[3:6])


class FinalLayer(nn.Module):
    """RMSNorm -> adaLN from w -> linear projection to patches."""
    def __init__(self, hidden_size, patch_size, out_channels, mod_dim):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.modulation = nn.Linear(mod_dim, 2 * hidden_size)

    def forward(self, x, w):
        scale, shift = self.modulation(w).chunk(2, dim=1)
        return self.linear(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class IMLEDiT(nn.Module):
    """One-step transformer generator for IMLE: z (N, 128) [+ labels] -> latent image (N, C, H, W)."""
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=8 / 3,
        num_classes=0,
        num_cond_tokens=16,
        mixer_depth=2,
        canvas_std=0.1,
        mod_init=0.5,
        head_mod_init=0.1,
        gate_k0=4.0,
        gate_cap_scale=2.0,
        gate_cap_logit=0.0,
        weight_init_constant=0.32,
        embedding_init_constant=1.0,
        token_init_constant=1.0,
        class_emb_init_std=1.0,
    ):
        super().__init__()
        assert Z_DIM % num_cond_tokens == 0, 'z must split evenly into num_cond_tokens'
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.latent_dim = Z_DIM
        self.num_cond_tokens = num_cond_tokens
        self.canvas_std = canvas_std
        self.mod_init = mod_init
        self.head_mod_init = head_mod_init
        self.weight_init_constant = weight_init_constant
        self.embedding_init_constant = embedding_init_constant
        self.token_init_constant = token_init_constant
        self.class_emb_init_std = class_emb_init_std
        self.head_init_std = 1 / math.sqrt(hidden_size)    # unit-RMS tokens in -> output std ~1 at init
        num_patches = (input_size // patch_size) ** 2

        # z -> w: 16 tokens, mixed by a small transformer, then pooled.
        self.cond_embed = nn.Parameter(torch.zeros(num_cond_tokens, hidden_size))
        self.z_proj = nn.Linear(Z_DIM // num_cond_tokens, hidden_size)
        self.mixer = nn.ModuleList([
            IMLEDiTBlock(hidden_size, num_heads, mlp_ratio, 'additive', 1.0) for _ in range(mixer_depth)
        ])
        if num_classes > 0:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # Image tokens: learned constant canvas.
        self.canvas = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        rope_cos, rope_sin = precompute_rope_freqs(hidden_size // num_heads, num_patches)
        self.register_buffer('rope_cos', rope_cos, persistent=False)
        self.register_buffer('rope_sin', rope_sin, persistent=False)

        self.blocks = nn.ModuleList([
            IMLEDiTBlock(hidden_size, num_heads, mlp_ratio, 'convex', gate_cap_logit, mod_dim=hidden_size)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, mod_dim=hidden_size)

        # Gate cap: branch j = 1..2*depth (attention, MLP, attention, ...) gets cap 2 / (j + 4).
        gates = [br.gate for blk in self.blocks for br in (blk.self_attn, blk.mlp)]
        for j, gt in enumerate(gates, start=1):
            gt.cap = gate_cap_scale / (j + gate_k0)
            assert gt.cap < 1, f'cap {gt.cap:.2f} >= 1 for branch {j}: raise gate_k0'
        self.initialize_weights()

    def initialize_weights(self):
        hidden_size = self.canvas.shape[-1]
        # Transformer linears: std = c / sqrt(fan_in) (iMF scaled variance).
        for block in list(self.blocks) + list(self.mixer):
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.weight_init_constant / math.sqrt(module.in_features))

        # adaLN: w has RMS 1, so gamma, beta start with std ~ mod_init (nonzero: w is z's only path).
        # Gate offsets start at zero, so every z starts with the same gates.
        for block in self.blocks:
            nn.init.normal_(block.modulation.weight, std=self.mod_init / math.sqrt(block.modulation.in_features))
            nn.init.constant_(block.modulation.bias, 0)
            D = block.modulation.out_features // 6
            for i in range(2):
                rows = slice((3 * i + 2) * D, (3 * i + 3) * D)
                nn.init.zeros_(block.modulation.weight[rows])
                nn.init.zeros_(block.modulation.bias[rows])
        head_mod = self.final_layer.modulation
        nn.init.normal_(head_mod.weight, std=self.head_mod_init / math.sqrt(head_mod.in_features))
        nn.init.constant_(head_mod.bias, 0)

        nn.init.normal_(self.z_proj.weight, std=self.embedding_init_constant / math.sqrt(self.z_proj.in_features))
        nn.init.constant_(self.z_proj.bias, 0)
        nn.init.normal_(self.cond_embed, std=self.token_init_constant / math.sqrt(hidden_size))
        if self.num_classes > 0:
            # Per-element std ~1, the scale of the z tokens (z_proj of an RMS-1 z), so the class starts as
            # strong as z in the mixer input (the conv generator uses 0.9 against z ~ N(0, 1)).
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=self.class_emb_init_std)

        # Canvas: small, so its pass-through to the output is small and Adam's steps change it relatively more.
        nn.init.normal_(self.canvas, std=self.canvas_std)

        # Output head: small and nonzero (a zero head would map every z to the same image).
        nn.init.normal_(self.final_layer.linear.weight, std=self.head_init_std)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """x: (N, T, patch_size**2 * C) -> imgs: (N, C, H, W)"""
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def conditioning(self, z, y=None):
        """z (N, 128) [+ labels y] -> w (N, D) with RMS 1. The class embedding is added to every conditioning
        token before the mixer, so the mixer combines z and class nonlinearly."""
        N = z.shape[0]
        z = rms_normalize(z)
        ctx = self.z_proj(z.view(N, self.num_cond_tokens, -1)) + self.cond_embed      # (N, K, D)
        if self.num_classes > 0:
            assert y is not None, 'class-conditional model needs labels'
            ctx = ctx + self.y_embedder(y).unsqueeze(1)                                # same class vector on all K tokens
        for block in self.mixer:
            ctx = block(ctx)
        return rms_normalize(ctx.mean(dim=1))                                          # (N, D)

    def forward(self, z, y=None, train=False):
        """z: (N, 128) Gaussian latent. y: (N,) labels if num_classes > 0. train=True returns [x] (AdaIMLE's
        multi-resolution interface: a list whose last entry is the output)."""
        w = self.conditioning(z, y)
        x = self.canvas.expand(z.shape[0], -1, -1)                                     # (N, T, D)
        rope = (self.rope_cos, self.rope_sin)
        for block in self.blocks:
            x = block(x, rope=rope, w=w)
        x = self.unpatchify(self.final_layer(x, w))                                    # (N, C, H, W)
        return [x] if train else x


#################################################################################
#                                   Configs                                     #
#################################################################################

def IMLEDiT_S_2(**kwargs):      # s27
    return IMLEDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def IMLEDiT_B_2(**kwargs):
    return IMLEDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def IMLEDiT_L_2(**kwargs):
    return IMLEDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def IMLEDiT_XL_2(**kwargs):
    return IMLEDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


IMLEDiT_models = {
    'IMLEDiT-S/2': IMLEDiT_S_2, 'IMLEDiT-B/2': IMLEDiT_B_2, 'IMLEDiT-L/2': IMLEDiT_L_2, 'IMLEDiT-XL/2': IMLEDiT_XL_2,
}

"""Sharpness-Aware Minimization (SAM) for the IMLE training loop.

Two-pass update: perturb the weights towards the local gradient-ascent direction,
re-evaluate the gradient there, then step the base optimizer with that gradient
from the original weights.

    w_adv = w + rho * g / ||g||          (ASAM: w + rho * |w|^2 g / || |w| g ||)
    w <- w - lr * grad L(w_adv)

Notes on this particular integration:
  * The perturbation direction is scale invariant, so AMP-scaled gradients can be
    used directly -- no need to unscale before the ascent step (which would clash
    with the single `scaler.unscale_` the training loop already does before
    clipping). A non-finite gradient norm means the fp16 pass overflowed, in which
    case we skip the perturbation and fall back to a plain step.
  * Under DDP the ascent step must run after grads have been all-reduced, so every
    rank perturbs by the same e_w and the replicas stay bitwise in sync.
"""

import torch


class SAM:
    """Applies/undoes the SAM weight perturbation. Wraps params, not the optimizer."""

    def __init__(self, params, rho=0.05, adaptive=False):
        self.params = [p for p in params if p.requires_grad]
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self._backup = None
        self.last_grad_norm = float('nan')

    @torch.no_grad()
    def _grad_norm(self):
        norms = []
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.adaptive:
                g = g * p.abs()
            norms.append(torch.linalg.vector_norm(g))
        if not norms:
            return None
        return torch.linalg.vector_norm(torch.stack(norms))

    @torch.no_grad()
    def ascent_step(self):
        """Perturbs weights to w + e_w. Returns True if the perturbation was applied.

        Gradients must already be populated (and all-reduced under DDP). They may be
        AMP-scaled; the scale cancels in g / ||g||.
        """
        assert self._backup is None, 'ascent_step() called twice without restore()'

        grad_norm = self._grad_norm()
        if grad_norm is None:
            return False
        self.last_grad_norm = float(grad_norm)
        if not torch.isfinite(grad_norm) or float(grad_norm) == 0.0:
            # fp16 overflow or a dead batch -- the scaler will handle/skip this step.
            return False

        scale = self.rho / (grad_norm + 1e-12)
        # Keep the pre-perturbation weights rather than undoing the add later: a
        # sub_ round-trip is not exact in fp32 and the residual would accumulate
        # over every step of training.
        backup = []
        for p in self.params:
            if p.grad is None:
                backup.append(None)
                continue
            backup.append(p.detach().clone())
            e = p.grad * scale
            if self.adaptive:
                e = e * p.pow(2)
            p.add_(e)
        self._backup = backup
        return True

    @torch.no_grad()
    def restore(self):
        """Moves the weights back from w + e_w to w, exactly."""
        if self._backup is None:
            return
        for p, w in zip(self.params, self._backup):
            if w is not None:
                p.copy_(w)
        self._backup = None

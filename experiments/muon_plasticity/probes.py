"""Direction 005 — read-only plasticity diagnostics.

The hypothesis under causal test (arXiv:2509.22335) is that loss of plasticity
is driven by SPECTRAL COLLAPSE: as a network is trained on a long task sequence
its feature representation and weight matrices lose effective rank, units die,
and the optimization landscape becomes harder to move in. This module measures
exactly those quantities so we can ask whether Muon (which orthogonalizes its
updates) preserves them where AdamW / SGDM do not.

All probes are @no_grad and side-effect-free EXCEPT `optimization_readiness`,
which intentionally runs TWO extra backward passes (documented below) to read
the gradient-signal strength and cross-minibatch gradient cosine. None mutate
optimizer state or model parameters.

Diagnostics
-----------
1. effective_rank(M)        — entropy-based effective rank exp(H(p)), p_i =
                              sigma_i / sum(sigma) (Roy & Vetterli 2007).
2. stable_rank(M)           — ||M||_F^2 / ||M||_2^2 (sum sigma^2 / max sigma^2).
3. feature_rank_stats       — eff_rank + stable_rank of penultimate activations
                              on a probe batch (the "feature" collapse signal).
4. dead_unit_fraction       — fraction of penultimate ReLU units that are zero
                              for EVERY example in the probe batch (dead units).
5. weight_effective_ranks   — eff_rank of each hidden weight matrix.
6. optimization_readiness   — gradient-norm strength + cosine reliability of the
                              gradient across two independent minibatches
                              (1 extra backward each => 2 total). A collapsing,
                              plasticity-losing net shows shrinking grad norm
                              and/or noisy (low-cosine) gradients.
7. lambda_max               — top Hessian eigenvalue, REUSED by import from
                              experiments/eos_tiny/hessian.py:top_eigenvalue.

`plasticity_probe(...)` runs the whole suite and returns a flat dict.

Run `python probes.py` for a self-test (effective_rank on a known low-rank
matrix returns ~ its true rank).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

# --- import discipline (root README): LOCAL dir FIRST, other dirs APPENDED. --
# We only need top_eigenvalue from eos_tiny/hessian.py; eos_tiny has no module
# names that collide with ours, but we still append it (never insert) so this
# directory's own modules always win.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_EOS_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "eos_tiny"))
if _EOS_DIR not in sys.path:
    sys.path.append(_EOS_DIR)

from hessian import top_eigenvalue  # noqa: E402  (eos_tiny infra, by import)


# --------------------------------------------------------------------------- #
# Pure linear-algebra rank measures
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _singular_values(M: torch.Tensor) -> torch.Tensor:
    """Singular values of a 2-D matrix, descending, on CPU float32."""
    return torch.linalg.svdvals(M.detach().float().cpu())


@torch.no_grad()
def effective_rank(M: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank: exp(-sum p_i log p_i), p_i = s_i/sum(s).

    Ranges in [1, min(rows, cols)]; equals the true rank for a matrix with k
    equal nonzero singular values, and degrades smoothly under spectral decay.
    """
    s = _singular_values(M)
    total = s.sum()
    if total < eps:
        return 0.0
    p = s / total
    p = p[p > eps]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy))


@torch.no_grad()
def stable_rank(M: torch.Tensor, eps: float = 1e-12) -> float:
    """Stable rank ||M||_F^2 / ||M||_2^2 = sum(s^2) / max(s)^2."""
    s = _singular_values(M)
    smax2 = float(s.max() ** 2) if s.numel() else 0.0
    if smax2 < eps:
        return 0.0
    return float((s ** 2).sum() / smax2)


# --------------------------------------------------------------------------- #
# Feature-space diagnostics (penultimate activations)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def feature_rank_stats(model, X: torch.Tensor) -> dict:
    """eff_rank + stable_rank of the penultimate activations on probe batch X.

    We center the [N, width] activation matrix (rank of the covariance-bearing
    part) before taking singular values, so a constant-offset feature does not
    inflate the rank.
    """
    feats = model.features(X)                       # [N, width]
    centered = feats - feats.mean(dim=0, keepdim=True)
    return {
        "feat_eff_rank": effective_rank(centered),
        "feat_stable_rank": stable_rank(centered),
    }


@torch.no_grad()
def dead_unit_fraction(model, X: torch.Tensor) -> float:
    """Fraction of penultimate ReLU units that are 0 for ALL probe examples."""
    feats = model.features(X)                       # [N, width], post-ReLU >= 0
    alive = (feats > 0).any(dim=0)                  # [width] bool
    return float((~alive).float().mean())


@torch.no_grad()
def weight_effective_ranks(model) -> dict:
    """eff_rank of every hidden weight matrix, keyed by layer index."""
    out = {}
    for i, lin in enumerate(model.hidden):
        out[f"w_eff_rank_L{i}"] = effective_rank(lin.weight)
    return out


# --------------------------------------------------------------------------- #
# Optimization-readiness proxy (the ONLY probe that runs backward passes)
# --------------------------------------------------------------------------- #
def _flat_grad(model, loss) -> torch.Tensor:
    """Flattened gradient of `loss` wrt model params (no graph retained)."""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
    return torch.cat([g.reshape(-1) for g in grads])


def optimization_readiness(model, batch_a, batch_b, loss_fn) -> dict:
    """Gradient-signal strength + cross-minibatch gradient cosine reliability.

    DOCUMENTED EXTRA WORK: this runs TWO backward passes — one on `batch_a`,
    one on `batch_b` — to obtain two independent stochastic gradients g_a, g_b.
    We report:
      grad_norm    : ||g_a||  (signal strength; collapses toward 0 when a net
                     loses the ability to move).
      grad_cosine  : cos(g_a, g_b)  (gradient reliability across minibatches; a
                     low/noisy cosine means the descent direction is unreliable).
    Gradients are taken with autograd.grad (no .backward()), so NO optimizer
    state and NO .grad buffers on the model are touched/mutated.
    """
    Xa, Ya = batch_a
    Xb, Yb = batch_b
    g_a = _flat_grad(model, loss_fn(model(Xa), Ya))
    g_b = _flat_grad(model, loss_fn(model(Xb), Yb))
    grad_norm = float(g_a.norm())
    denom = (g_a.norm() * g_b.norm())
    cosine = float(torch.dot(g_a, g_b) / denom) if denom > 1e-12 else 0.0
    return {"grad_norm": grad_norm, "grad_cosine": cosine}


# --------------------------------------------------------------------------- #
# Hessian sharpness (REUSED from eos_tiny by import)
# --------------------------------------------------------------------------- #
def lambda_max(model, loss_fn, data, iters: int = 20, seed: int = 0) -> float:
    """Top Hessian eigenvalue via eos_tiny.hessian.top_eigenvalue (by import).

    loss_fn(model, data) -> scalar loss tensor with a grad graph (the signature
    top_eigenvalue expects).
    """
    return float(top_eigenvalue(model, loss_fn, data, iters=iters, seed=seed))


# --------------------------------------------------------------------------- #
# Full suite
# --------------------------------------------------------------------------- #
def plasticity_probe(model, probe_batch, second_batch, *, lam_iters: int = 20,
                     lam_seed: int = 0) -> dict:
    """Run every diagnostic on `model` and return a flat metrics dict.

    probe_batch  : (X, Y) used for feature/dead-unit/grad-A/hessian probes.
    second_batch : (X, Y) used as the SECOND minibatch for the grad cosine.
    """
    X, Y = probe_batch

    def ce_logits(logits, target):
        return F.cross_entropy(logits, target)

    def ce_loss_fn(m, d):
        Xd, Yd = d
        return F.cross_entropy(m(Xd), Yd)

    metrics: dict = {}
    metrics.update(feature_rank_stats(model, X))
    metrics["dead_frac"] = dead_unit_fraction(model, X)
    metrics.update(weight_effective_ranks(model))
    metrics.update(optimization_readiness(model, probe_batch, second_batch, ce_logits))
    metrics["lambda_max"] = lambda_max(model, ce_loss_fn, (X, Y),
                                       iters=lam_iters, seed=lam_seed)
    return metrics


# --------------------------------------------------------------------------- #
# Self-test: effective_rank / stable_rank on a KNOWN low-rank matrix must
# recover its true rank. Run with `python probes.py`.
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    torch.manual_seed(0)
    ok = True

    # Build a matrix of exact rank r with r EQUAL nonzero singular values, for
    # which the entropy-effective-rank and the stable-rank both equal r exactly.
    rows, cols, r = 64, 48, 5
    U, _ = torch.linalg.qr(torch.randn(rows, rows))
    V, _ = torch.linalg.qr(torch.randn(cols, cols))
    s = torch.zeros(min(rows, cols))
    s[:r] = 3.0                                  # r equal nonzero singular values
    S = torch.zeros(rows, cols)
    S[:min(rows, cols), :min(rows, cols)] = torch.diag(s)
    M = U @ S @ V.t()

    er = effective_rank(M)
    sr = stable_rank(M)
    print(f"SELF-TEST low-rank matrix (true rank={r}, equal sing. vals):")
    print(f"  effective_rank = {er:.4f}  (expect ~{r})")
    print(f"  stable_rank    = {sr:.4f}  (expect ~{r})")
    if abs(er - r) > 1e-3:
        ok = False
        print(f"  FAIL: effective_rank {er:.6f} != {r}")
    if abs(sr - r) > 1e-3:
        ok = False
        print(f"  FAIL: stable_rank {sr:.6f} != {r}")

    # A full-rank orthonormal-columns matrix has effective rank ~ its dimension.
    Q, _ = torch.linalg.qr(torch.randn(40, 16))   # [40, 16] orthonormal cols
    er_full = effective_rank(Q)
    print(f"SELF-TEST full-rank (16 orthonormal cols): effective_rank "
          f"= {er_full:.4f}  (expect ~16)")
    if abs(er_full - 16) > 1e-2:
        ok = False
        print(f"  FAIL: full-rank effective_rank {er_full:.6f} != 16")

    # Dead-unit + feature-rank sanity on a tiny MLP with a known-dead unit.
    sys.path.insert(0, _THIS_DIR)
    from model import MLP
    net = MLP(d_in=8, width=12, n_layers=2, n_classes=3)
    with torch.no_grad():
        # Force the last hidden unit dead: zero its incoming weights+bias and
        # make its bias strongly negative so ReLU clamps it to 0 always.
        net.hidden[-1].weight[-1].zero_()
        net.hidden[-1].bias[-1] = -1e3
    Xp = torch.randn(32, 8)
    dead = dead_unit_fraction(net, Xp)
    print(f"SELF-TEST dead_unit_fraction with 1/12 forced-dead: {dead:.4f} "
          f"(expect >= {1/12:.4f})")
    if dead < 1 / 12 - 1e-9:
        ok = False
        print(f"  FAIL: dead fraction {dead:.6f} < {1/12:.6f}")

    print("PROBE SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())

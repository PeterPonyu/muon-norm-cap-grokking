"""Single-GPU Muon optimizer (Keller Jordan, 2024).

Muon = SGD-momentum whose 2-D parameter update is orthogonalized via a
quintic Newton-Schulz iteration before being applied. This controls the
spectral geometry of every step. Use ONLY on 2-D hidden matrices; embeddings,
the unembed head, biases, and norm params should be handled by AdamW.

This is a faithful, dependency-free reimplementation of the public reference,
minus the distributed (DDP) sharding machinery (we run on one GPU).
"""
from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Compute an orthogonalization of G via 5 coefficients of a quintic NS iteration.

    Returns a matrix whose singular values are pushed toward 1 (semi-orthogonal),
    sharing G's singular vectors. Operates in bfloat16 for speed/stability.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X


class Muon(Optimizer):
    """Muon for 2-D matrices. Non-2-D params must go to a different optimizer.

    Args:
        lr: learning rate.
        momentum: SGD momentum coefficient.
        nesterov: use Nesterov-style lookahead on the momentum buffer.
        ns_steps: Newton-Schulz iterations (5 is standard).
        weight_decay: decoupled (AdamW-style) weight decay.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                assert g.ndim == 2, "Muon params must be 2-D matrices"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                # scale so the RMS of the update matches across shapes
                scale = max(1.0, g.size(0) / g.size(1)) ** 0.5
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr * scale)
        return loss


def split_params_for_muon(model):
    """Partition model params into (muon_2d, adamw_other).

    2-D weight matrices inside transformer blocks go to Muon; everything else
    (embeddings, unembed head, layernorm weights/biases, any bias) goes to AdamW.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_matrix = (
            p.ndim == 2
            and "emb" not in name           # tok_emb / pos_emb
            and "unembed" not in name       # output head
        )
        (muon_params if is_hidden_matrix else adamw_params).append(p)
    return muon_params, adamw_params

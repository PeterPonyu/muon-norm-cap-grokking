"""Modular-arithmetic dataset for grokking experiments.

Format (Nanda-style): each example is a length-3 token sequence [a, b, EQ];
the model predicts (a op b) mod p at the final position.

Vocabulary: tokens 0..p-1 are the operands/results, token p is the "=" marker.
So vocab_size = p + 1. The target is an integer in [0, p).
"""
from __future__ import annotations

import torch


def make_modular_dataset(p: int = 97, op: str = "add", device: str = "cpu"):
    """Return (X, Y) covering all p*p operand pairs.

    X: LongTensor [p*p, 3] = [a, b, EQ_TOKEN]
    Y: LongTensor [p*p]    = (a op b) mod p
    """
    eq_token = p
    a = torch.arange(p, device=device).repeat_interleave(p)  # [p*p]
    b = torch.arange(p, device=device).repeat(p)             # [p*p]
    if op == "add":
        y = (a + b) % p
    elif op == "sub":
        y = (a - b) % p
    elif op == "mul":
        y = (a * b) % p
    else:
        raise ValueError(f"unknown op {op!r}")
    eq = torch.full_like(a, eq_token)
    X = torch.stack([a, b, eq], dim=1)  # [p*p, 3]
    return X, y


def make_s5_dataset(device: str = "cpu"):
    """Composition in the symmetric group S5 (non-abelian; Power et al. 2022).

    120 permutations as tokens 0..119, EQ token = 120 → vocab_size 121.
    X: [120*120, 3] = [a, b, EQ]; Y = index of (perm_a ∘ perm_b).
    """
    import itertools
    perms = list(itertools.permutations(range(5)))
    idx = {p: i for i, p in enumerate(perms)}
    n = len(perms)
    A, B, Y = [], [], []
    for i, pa in enumerate(perms):
        for j, pb in enumerate(perms):
            c = tuple(pa[pb[k]] for k in range(5))
            A.append(i)
            B.append(j)
            Y.append(idx[c])
    X = torch.stack([torch.tensor(A), torch.tensor(B),
                     torch.full((n * n,), n)], dim=1).to(device)
    return X, torch.tensor(Y, device=device)


def train_test_split(X, Y, train_frac: float, seed: int):
    """Deterministic split of all pairs into train/test by a seeded permutation."""
    g = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    perm = torch.randperm(n, generator=g)
    n_train = int(round(train_frac * n))
    tr, te = perm[:n_train], perm[n_train:]
    return (X[tr], Y[tr]), (X[te], Y[te])

"""Direction 005 — small MLP for the continual-plasticity study.

A deliberately plain feed-forward classifier: d_in -> [hidden]*n_layers -> C,
ReLU nonlinearities. Plasticity-loss phenomenology (dead units, feature-rank
collapse, spectral collapse) is classically studied on exactly this kind of
small MLP (Dohare et al. 2024), so we keep it standard and self-contained
rather than reusing the grokking transformer.

We expose `features(x)` returning the PENULTIMATE (pre-logit, post-ReLU)
activations so the probe suite (probes.py) can measure feature effective rank
and dead-unit fraction on a probe batch.

`split_params_for_muon(model)` is LOCAL to this direction. The grokking
split_params_for_muon is GrokTransformer-specific (it filters on "emb" /
"unembed" name substrings that do not exist here), so we write our own:
hidden 2-D weight matrices -> Muon group; first & last layer weights, all
biases, and any 1-D params -> AdamW group. Keeping the first (input) and last
(readout) layers in AdamW follows the Muon reference recipe (orthogonalize only
the interior matrices).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """d_in -> (hidden, ReLU) x n_layers -> n_classes.

    Default width 512 with 3 hidden layers gives ~0.6M params at d_in=128,
    inside the 0.4-2M target band; widen via `width` / `n_layers`.
    """

    def __init__(self, d_in: int = 128, width: int = 512, n_layers: int = 3,
                 n_classes: int = 10):
        super().__init__()
        assert n_layers >= 1
        dims = [d_in] + [width] * n_layers
        self.hidden = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.readout = nn.Linear(width, n_classes)
        self.d_in = d_in
        self.width = width
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate post-ReLU activations [N, width] (pre-readout)."""
        h = x
        for lin in self.hidden:
            h = F.relu(lin(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.features(x))


def split_params_for_muon(model: MLP):
    """Partition params into (muon_2d, adamw_other) — LOCAL to direction 005.

    Muon group : interior hidden 2-D weight matrices (everything except the
                 first/input layer and the readout layer).
    AdamW group: first hidden layer weight, readout weight, ALL biases, and any
                 non-2-D params.

    Rationale: Muon orthogonalizes interior matrices; the input embedding and
    output head are kept in AdamW (the reference recipe), and the first hidden
    layer is treated like an embedding (it sees the raw input).
    """
    first = model.hidden[0]
    last = model.readout
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_interior_matrix = (
            p.ndim == 2
            and p is not first.weight
            and p is not last.weight
        )
        (muon_params if is_interior_matrix else adamw_params).append(p)
    return muon_params, adamw_params

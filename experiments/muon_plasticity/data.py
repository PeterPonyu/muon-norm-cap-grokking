"""Direction 005 — fully-synthetic continual-learning benchmark (zero downloads).

We construct a long sequence of related-but-shifting classification tasks so
that we can MEASURE loss of plasticity: the (empirically observed) decay in a
network's ability to fit *new* tasks as it is trained on more and more of them
(Dohare et al. 2024; "spectral collapse drives loss of plasticity", arXiv
:2509.22335). Direction 005 asks the causal question at the OPTIMIZER level:
does Muon's orthogonalized (spectrum-preserving) update keep the network
trainable where AdamW / SGD-momentum lose plasticity?

Two PRIMARY synthetic arms (no external data, deterministic per seed+task idx):

  (a) "proj_shift"  — shifting random-projection classification.
      A single fixed pool of inputs x in R^d (standard normal). For task t we
      draw a FRESH random projection P_t : R^d -> R^m and a fresh random linear
      teacher W_t : R^m -> R^C, then label  y = argmax( W_t @ relu(P_t @ x) ).
      Inputs are shared across tasks; only the input->feature->label map shifts.
      This is the classic "the representation that mattered last task is wrong
      now" continual setting — a network that over-commits its features loses
      the plasticity to re-fit.

  (b) "label_refit" — random-label memorization-capacity probe.
      Same shared inputs x, but task t assigns FRESH uniformly-random labels
      y in {0..C-1} per example (no teacher structure at all). Pure memorization
      capacity: how fast can the optimizer drive a *new* random labelling to
      high train accuracy? A plasticity-losing optimizer slows down here even
      though every task is information-theoretically identical.

Optional MNIST arm (permuted-MNIST), behind a flag and a guarded torchvision
import. NEVER downloads in smoke; raises a clear message if torchvision/the
dataset is unavailable.

Everything in arms (a)/(b) is produced in memory and is deterministic given
(seed, task_index): task t's projection/teacher/labels come from a generator
seeded by hash(seed, t), and each sampled minibatch is seeded by
(seed, t, step). No files, no downloads.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
@dataclass
class ContinualSpec:
    """Fully describes a continual benchmark (deterministic given seed)."""
    arm: str = "proj_shift"      # "proj_shift" | "label_refit" | "perm_mnist"
    d_in: int = 128              # input dimensionality
    m_feat: int = 128            # teacher hidden width (proj_shift only)
    n_classes: int = 10          # number of label classes
    n_examples: int = 2048       # size of the shared fixed input pool
    seed: int = 0

    @property
    def input_dim(self) -> int:
        # perm_mnist overrides d_in to 784 at build time; for the synthetic
        # arms input_dim is just d_in.
        return 784 if self.arm == "perm_mnist" else self.d_in


# --------------------------------------------------------------------------- #
# Deterministic per-(seed, task) generators
# --------------------------------------------------------------------------- #
def _task_generator(seed: int, task_idx: int, salt: int = 0) -> torch.Generator:
    """A CPU generator deterministic in (seed, task_idx, salt).

    We fold the three integers into one 63-bit seed so different tasks / roles
    (projection vs teacher vs labels) never share a stream.
    """
    s = (int(seed) * 1_000_003 + int(task_idx) * 9_176 + int(salt) * 31) & 0x7FFF_FFFF_FFFF_FFFF
    return torch.Generator().manual_seed(s)


def _shared_inputs(spec: ContinualSpec, device: str = "cpu") -> torch.Tensor:
    """The fixed input pool x in R^{n_examples x d_in}, shared across all tasks.

    Deterministic in `seed` only (task-independent). For perm_mnist this is
    unused (inputs come from torchvision); kept here for the synthetic arms.
    """
    g = _task_generator(spec.seed, task_idx=-1, salt=777)
    x = torch.randn(spec.n_examples, spec.d_in, generator=g)
    return x.to(device)


# --------------------------------------------------------------------------- #
# Arm builders -> return a (X, Y) tensor pair for a given task index
# --------------------------------------------------------------------------- #
def _proj_shift_task(spec: ContinualSpec, task_idx: int, device: str):
    """Task t: y = argmax( W_t @ relu(P_t @ x) ) over the shared input pool.

    Returns (X [N, d_in] float, Y [N] long). Deterministic in (seed, task_idx).
    """
    x = _shared_inputs(spec, device=device)              # [N, d_in]
    gp = _task_generator(spec.seed, task_idx, salt=1)
    gw = _task_generator(spec.seed, task_idx, salt=2)
    # Fresh random projection and linear teacher, normalized for stable scale.
    P = torch.randn(spec.d_in, spec.m_feat, generator=gp) / (spec.d_in ** 0.5)
    W = torch.randn(spec.m_feat, spec.n_classes, generator=gw) / (spec.m_feat ** 0.5)
    P = P.to(device)
    W = W.to(device)
    feats = torch.relu(x @ P)                            # [N, m_feat]
    logits = feats @ W                                   # [N, n_classes]
    y = logits.argmax(dim=1)                             # [N] teacher labels
    return x, y


def _label_refit_task(spec: ContinualSpec, task_idx: int, device: str):
    """Task t: shared inputs, FRESH uniform-random labels (memorization probe).

    Returns (X [N, d_in] float, Y [N] long). Deterministic in (seed, task_idx).
    """
    x = _shared_inputs(spec, device=device)
    gl = _task_generator(spec.seed, task_idx, salt=3)
    y = torch.randint(0, spec.n_classes, (spec.n_examples,), generator=gl).to(device)
    return x, y


# --------------------------------------------------------------------------- #
# Optional permuted-MNIST arm (NEVER downloads in smoke; guarded import)
# --------------------------------------------------------------------------- #
# Module-level cache: MNIST is loaded from a prebuilt tensor file exactly once
# per process (the torchvision yann.lecun.com mirror is dead; the pool is built
# out-of-band via HuggingFace `datasets` into mnist_pool.pt next to this file).
_MNIST_CACHE: dict | None = None


def _load_mnist_pool():
    """Load the cached MNIST tensor pool (images [60000,784] in [0,1], labels).

    Returns (images_cpu, labels_cpu). Cached at module level so the continual
    loop (100s of tasks) never reloads. Raises a CLEAR message if the prebuilt
    file is missing (build it once with the HF-datasets snippet in README).
    """
    global _MNIST_CACHE
    if _MNIST_CACHE is None:
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mnist_pool.pt")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"perm_mnist arm needs the prebuilt pool at {path!r} "
                "(the torchvision MNIST mirror is dead). Build it once via "
                "HuggingFace datasets: load_dataset('mnist', split='train') -> "
                "save {'images':[N,784] float in [0,1], 'labels':[N] long}."
            )
        d = torch.load(path)
        _MNIST_CACHE = {"images": d["images"].float(), "labels": d["labels"].long()}
    return _MNIST_CACHE["images"], _MNIST_CACHE["labels"]


def _perm_mnist_task(spec: ContinualSpec, task_idx: int, device: str,
                     data_root: str):
    """Task t: a fixed random PERMUTATION of the 784 pixels + true MNIST labels.

    The canonical loss-of-plasticity benchmark (Dohare et al. 2024): inputs and
    labels are fixed; each task applies a new fixed pixel permutation, so the
    semantic content is constant while the input representation is scrambled.
    Returns (X [N, 784] float, Y [N] long). Deterministic in (seed, task_idx).
    """
    images, labels = _load_mnist_pool()                         # CPU tensors
    n = min(spec.n_examples, images.shape[0])
    pool = images[:n].to(device)                                # [n, 784]
    y = labels[:n].to(device)                                   # [n]
    gp = _task_generator(spec.seed, task_idx, salt=4)
    perm = torch.randperm(784, generator=gp).to(device)
    return pool[:, perm], y


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_task(spec: ContinualSpec, task_idx: int, device: str = "cpu",
               data_root: str = "./_mnist_data"):
    """Return (X, Y) for `task_idx` of the continual sequence (full task pool).

    X : FloatTensor [N, input_dim]
    Y : LongTensor  [N]
    Deterministic given (spec.seed, task_idx). In-memory for synthetic arms.
    """
    if spec.arm == "proj_shift":
        return _proj_shift_task(spec, task_idx, device)
    if spec.arm == "label_refit":
        return _label_refit_task(spec, task_idx, device)
    if spec.arm == "perm_mnist":
        return _perm_mnist_task(spec, task_idx, device, data_root)
    raise ValueError(f"unknown arm {spec.arm!r}")


def sample_minibatch(X: torch.Tensor, Y: torch.Tensor, batch_size: int,
                     seed: int):
    """Deterministic minibatch (indices) from a built task's (X, Y) pool.

    If batch_size >= pool size, returns the whole pool (full-batch step).
    """
    n = X.shape[0]
    if batch_size >= n:
        return X, Y
    g = torch.Generator().manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)
    idx = torch.randint(0, n, (batch_size,), generator=g).to(X.device)
    return X[idx], Y[idx]


def probe_batch(X: torch.Tensor, Y: torch.Tensor, n: int, seed: int):
    """A small fixed probe batch (for the diagnostics suite). Deterministic."""
    return sample_minibatch(X, Y, n, seed)

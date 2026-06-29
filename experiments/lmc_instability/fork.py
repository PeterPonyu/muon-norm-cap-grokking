"""Direction 012 — fork machinery for the linear-mode-connectivity (LMC) study.

This is the NEW load-bearing module. It implements the three primitives the
spawn-fork protocol (Frankle et al., "Linear Mode Connectivity and the Lottery
Ticket Hypothesis", arXiv:1912.05671) needs on top of the grokking testbed:

  (i)   spawn_child(state_dict, mode, scale, seed)
        -> a perturbed/diverged COPY of a parent checkpoint. Two divergence
           mechanisms are provided behind `mode` because the grokking testbed is
           full-batch deterministic (same-seed children would be bit-identical):
             - "perturb"   : add tiny Gaussian noise, per-tensor scaled by the
                             tensor's own L2 norm (sigma = scale * ||W||).
             - "minibatch" : NO weight change here; divergence comes from the
                             trainer using large-minibatch sampling with a
                             child-specific shuffle seed. spawn_child returns a
                             clean copy and records the seed for the trainer.
           Only float, >=1-D tensors are perturbed; integer buffers / 0-D scalars
           are copied untouched (perturbing them is meaningless and would corrupt
           e.g. step counters).

  (ii)  linear_barrier(model_template, sd_a, sd_b, data, n_points=11)
        -> evaluate loss & accuracy along the linear interpolation
           theta(alpha) = (1 - alpha) * a + alpha * b for alpha in [0, 1]
           (forward passes only, @no_grad). Returns the loss-based barrier
             barrier = max_alpha loss(alpha) - max(loss(0), loss(1))
           and the accuracy-based variant
             acc_barrier = max(acc(0), acc(1)) - min_alpha acc(alpha)
           plus the raw per-alpha curves. Only float tensors are interpolated;
           non-float buffers are taken from sd_a (they must match structurally).

  (iii) k_star(barrier_by_spawn, threshold)
        -> the earliest spawn step after which children stay linearly connected,
           i.e. the smallest spawn step k such that barrier(k') < threshold for
           every spawn step k' >= k. Returns None if no such k exists (children
           never lock into a shared basin within the probed range).

Self-test (run `python fork.py`): barrier between a state_dict and ITSELF is 0
exactly; barrier between two independently-initialized models is >> 0; the
perturb mode moves weights by the expected relative norm; k_star recovers a
known monotone-onset pattern.

Import discipline (root README): this directory is inserted at sys.path[0] and
`experiments/grokking/` is APPENDED at the back, so our local modules win and we
reuse ONLY GrokTransformer / make_modular_dataset / train_test_split from
grokking. grokking files are never modified.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Optional

# --- import discipline: LOCAL dir FIRST, grokking APPENDED. ----------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_GROKKING_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.append(_GROKKING_DIR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


# --------------------------------------------------------------------------- #
# (i) spawn_child
# --------------------------------------------------------------------------- #
def _is_perturbable(t: torch.Tensor) -> bool:
    """Only perturb real-valued tensors with at least one element to perturb.

    Integer/bool buffers (step counters, masks) and 0-D scalars are left alone:
    Gaussian noise on them is meaningless and can corrupt bookkeeping state.
    """
    return torch.is_floating_point(t) and t.ndim >= 1 and t.numel() > 0


def spawn_child(state_dict, mode: str = "perturb", scale: float = 1e-3,
                seed: int = 0):
    """Return a diverged COPY of a parent `state_dict`.

    Args:
        state_dict: parent checkpoint (mapping name -> tensor). NOT mutated.
        mode: "perturb" (Gaussian noise, default primary) or "minibatch"
              (clean copy; divergence is delegated to the trainer's shuffle).
        scale: perturb_scale. For "perturb", per-ELEMENT noise std is
               scale * rms(tensor) = scale * ||tensor||_2 / sqrt(numel). With
               this choice E[||noise||_2] ~= scale * ||tensor||_2, so the
               RELATIVE L2 perturbation ||noise|| / ||tensor|| ~= `scale` for
               every tensor regardless of its native norm or size. (Scaling std
               by the raw norm instead would inflate the relative change by a
               factor sqrt(numel), which is why we divide it out here. The spec's
               "sigma = perturb_scale * per-tensor weight norm" is realized as a
               relative-norm target of `scale`.)
        seed: RNG seed for the perturbation (and the child's minibatch shuffle).

    Returns:
        A new state_dict (deep-copied, detached, on the same devices/dtypes).
    """
    if mode not in ("perturb", "minibatch"):
        raise ValueError(f"unknown child-noise mode {mode!r}")

    # Deep copy so the parent checkpoint is never aliased or mutated.
    child = copy.deepcopy(state_dict)

    if mode == "minibatch":
        # No weight change: the trainer diverges children via a child-specific
        # minibatch shuffle seed. We still return a clean, independent copy.
        return child

    # mode == "perturb": per-element Gaussian noise with std = scale * rms(t), so
    # the per-tensor RELATIVE L2 change ||noise||/||t|| ~= scale (see docstring).
    gen = torch.Generator().manual_seed(int(seed))
    for t in child.values():
        if not _is_perturbable(t):
            continue
        # Sample on CPU (deterministic w/ a CPU generator), then move to t.device.
        noise = torch.randn(t.shape, generator=gen, dtype=torch.float32)
        noise = noise.to(device=t.device, dtype=t.dtype)
        rms = t.detach().float().norm().item() / (t.numel() ** 0.5)
        std = scale * rms
        t.add_(noise * std)
    return child


@torch.no_grad()
def relative_perturbation(sd_a, sd_b) -> float:
    """Global relative L2 distance ||b - a|| / ||a|| across perturbable tensors.

    Used by the smoke probe / self-test to confirm that `perturb` mode moved the
    weights by ~perturb_scale. Only float tensors are compared (the same set
    spawn_child perturbs), so integer buffers don't dilute the ratio.
    """
    num_sq = 0.0
    den_sq = 0.0
    for name, a in sd_a.items():
        b = sd_b[name]
        if not _is_perturbable(a):
            continue
        af = a.detach().float()
        bf = b.detach().float()
        num_sq += (bf - af).pow(2).sum().item()
        den_sq += af.pow(2).sum().item()
    return (num_sq ** 0.5) / (den_sq ** 0.5 + 1e-12)


# --------------------------------------------------------------------------- #
# (ii) linear_barrier
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _interpolate(sd_a, sd_b, alpha: float):
    """theta(alpha) = a + alpha*(b - a) for float tensors; copy a otherwise.

    This is mathematically (1-alpha)*a + alpha*b, but the `a + alpha*(b - a)`
    form is the one that makes the SELF-barrier exactly 0: when a == b, (b - a)
    is exactly 0, so the result is bit-identical to `a` for every alpha (the
    naive (1-alpha)*a + alpha*a form rounds to a != a by ~1e-8). It also pins the
    endpoints exactly (alpha=0 -> a, alpha=1 -> b up to one rounding of b-a+a).

    Non-float buffers (if any) are taken from sd_a; they must match sd_b
    structurally (same keys/shapes), which holds for forked siblings.
    """
    out = {}
    for name, a in sd_a.items():
        b = sd_b[name]
        if torch.is_floating_point(a):
            out[name] = a + alpha * (b - a)
        else:
            out[name] = a.clone()
    return out


@torch.no_grad()
def _eval_state(model, state_dict, X, Y):
    """Load `state_dict` into `model` and return (loss, acc) on (X, Y)."""
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logits = model(X)
    loss = F.cross_entropy(logits, Y).item()
    acc = (logits.argmax(-1) == Y).float().mean().item()
    return loss, acc


@torch.no_grad()
def linear_barrier(model_template, sd_a, sd_b, data, n_points: int = 11):
    """Loss/acc barrier along the linear interpolation between sd_a and sd_b.

    Args:
        model_template: a GrokTransformer instance to load interpolated weights
            into (its own weights are overwritten; structure must match the
            state dicts). Forward passes only — never trained.
        sd_a, sd_b: the two endpoint state dicts (e.g. the two children's final
            weights).
        data: (X, Y) tensors to evaluate the interpolated models on.
        n_points: number of alpha grid points in [0, 1] inclusive (>= 2).

    Returns:
        dict with:
          alphas        : list of alpha grid points
          losses, accs  : per-alpha loss / accuracy curves
          barrier       : max_alpha loss(alpha) - max(loss(0), loss(1))
          acc_barrier   : max(acc(0), acc(1)) - min_alpha acc(alpha)
          loss_endpoints, acc_endpoints : (value@0, value@1)
    """
    if n_points < 2:
        raise ValueError("n_points must be >= 2 (need both endpoints)")
    X, Y = data
    alphas = [i / (n_points - 1) for i in range(n_points)]

    losses, accs = [], []
    for alpha in alphas:
        sd = _interpolate(sd_a, sd_b, alpha)
        loss, acc = _eval_state(model_template, sd, X, Y)
        losses.append(loss)
        accs.append(acc)

    loss0, loss1 = losses[0], losses[-1]
    acc0, acc1 = accs[0], accs[-1]
    barrier = max(losses) - max(loss0, loss1)
    acc_barrier = max(acc0, acc1) - min(accs)
    return {
        "alphas": alphas,
        "losses": losses,
        "accs": accs,
        "barrier": barrier,
        "acc_barrier": acc_barrier,
        "loss_endpoints": (loss0, loss1),
        "acc_endpoints": (acc0, acc1),
    }


# --------------------------------------------------------------------------- #
# (iii) k_star
# --------------------------------------------------------------------------- #
def k_star(barrier_by_spawn, threshold: float) -> Optional[int]:
    """Earliest spawn step after which children stay linearly connected.

    Definition (Frankle et al.): k* is the smallest spawn step k such that the
    barrier is below `threshold` at k AND at every later probed spawn step. This
    "stays-below" rule (rather than first-crossing) is robust to a noisy barrier
    that dips below threshold once and then rises again.

    Args:
        barrier_by_spawn: mapping {spawn_step: barrier_value}. NaN/None barriers
            are treated as "not connected" (>= threshold).
        threshold: connectivity threshold; barrier < threshold == connected.

    Returns:
        The spawn step k* (an int key from the mapping), or None if no spawn
        step has all-later barriers below threshold.
    """
    steps = sorted(barrier_by_spawn.keys())
    if not steps:
        return None

    def connected(step) -> bool:
        v = barrier_by_spawn[step]
        if v is None:
            return False
        try:
            if v != v:  # NaN
                return False
        except TypeError:
            return False
        return v < threshold

    # Scan from the latest spawn step backward: k* is the start of the maximal
    # all-connected suffix. The moment we hit a disconnected step, anything at or
    # before it cannot be k*.
    k = None
    for step in reversed(steps):
        if connected(step):
            k = step
        else:
            break
    return k


# --------------------------------------------------------------------------- #
# Self-test (python fork.py)
# --------------------------------------------------------------------------- #
def _build_template(vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                    seed: int, device: str):
    """Build a GrokTransformer (imported from grokking) for self-tests/probes."""
    from model import GrokTransformer  # grokking infra (appended path)
    torch.manual_seed(seed)
    return GrokTransformer(
        vocab_size=vocab_size, seq_len=3, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, mlp_ratio=4, init_scale=1.0,
    ).to(device)


def _train_steps(model, X, Y, steps: int, lr: float = 1e-2):
    """Full-batch AdamW for a handful of steps (self-test helper only).

    Drives a fresh model toward a non-trivial solution so that two independently
    initialized+trained models occupy DISTINCT basins (and thus exhibit a real
    interpolation barrier). Untrained tiny transformers sit at chance everywhere,
    where any barrier is ~0 — correct behavior, but useless for asserting the
    "independent solutions are NOT linearly connected" property.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def _self_test() -> bool:
    from data import make_modular_dataset, train_test_split  # grokking infra

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = 23  # tiny prime: keep the self-test fast and CPU-friendly
    vocab_size = p + 1
    d_model, n_heads, n_layers = 32, 4, 2

    X, Y = make_modular_dataset(p, "add", device=device)
    (Xtr, Ytr), _ = train_test_split(X, Y, train_frac=0.5, seed=0)
    data = (Xtr, Ytr)

    template = _build_template(vocab_size, d_model, n_heads, n_layers, 0, device)

    # Endpoint A: an independently-initialized model trained to a non-trivial
    # (non-chance) solution, so barriers below are meaningful.
    model_a = _build_template(vocab_size, d_model, n_heads, n_layers, 1, device)
    _train_steps(model_a, Xtr, Ytr, steps=80)
    sd_a = {k: v.detach().clone() for k, v in model_a.state_dict().items()}

    ok = True

    # --- Test 1: barrier(sd, sd) == 0 exactly. ----------------------------- #
    res_self = linear_barrier(template, sd_a, sd_a, data, n_points=11)
    self_barrier = res_self["barrier"]
    self_acc_barrier = res_self["acc_barrier"]
    t1 = (self_barrier == 0.0) and (self_acc_barrier == 0.0)
    print(f"[self-test 1] self-barrier loss={self_barrier:.8f} "
          f"acc={self_acc_barrier:.8f} -> {'PASS' if t1 else 'FAIL'} "
          f"(must be exactly 0)")
    ok = ok and t1

    # --- Test 2: barrier between two INDEPENDENT solutions is large. ------- #
    # Endpoint B: a different init+seed, trained the same way. Two independent
    # SGD solutions on modular addition land in distinct basins -> a pronounced
    # loss bump at the interpolation midpoint (Frankle et al.'s "not connected").
    model_b = _build_template(vocab_size, d_model, n_heads, n_layers, 999, device)
    _train_steps(model_b, Xtr, Ytr, steps=80)
    sd_b = {k: v.detach().clone() for k, v in model_b.state_dict().items()}
    res_indep = linear_barrier(template, sd_a, sd_b, data, n_points=11)
    indep_barrier = res_indep["barrier"]
    # "large" == well above any plausible numerical-noise floor and well above the
    # tiny perturbed-pair barrier the smoke probe reports.
    t2 = indep_barrier > 0.1
    print(f"[self-test 2] independent-solution barrier={indep_barrier:.6f} "
          f"-> {'PASS' if t2 else 'FAIL'} (must be >> 0)")
    ok = ok and t2

    # --- Test 3: perturb mode moves weights by ~perturb_scale (rel norm). -- #
    scale = 1e-2
    child = spawn_child(sd_a, mode="perturb", scale=scale, seed=7)
    rel = relative_perturbation(sd_a, child)
    # Per-element std = scale * rms(W) => E[||noise_i||] ~= scale * ||W_i||, so the
    # global relative norm sqrt(sum||noise_i||^2)/sqrt(sum||W_i||^2) ~= scale (up
    # to RNG spread). Tight 0.5x..2x band catches the prior sqrt(numel) bug.
    t3 = 0.5 * scale < rel < 2.0 * scale
    print(f"[self-test 3] perturb rel-norm={rel:.6f} target~{scale:.6f} "
          f"-> {'PASS' if t3 else 'FAIL'} (within 0.5x..2x)")
    ok = ok and t3

    # --- Test 3b: minibatch mode is a clean (zero-perturbation) copy. ------ #
    clean = spawn_child(sd_a, mode="minibatch", scale=scale, seed=7)
    rel_clean = relative_perturbation(sd_a, clean)
    t3b = rel_clean == 0.0
    print(f"[self-test 3b] minibatch-mode rel-norm={rel_clean:.8f} "
          f"-> {'PASS' if t3b else 'FAIL'} (must be exactly 0)")
    ok = ok and t3b

    # --- Test 4: k_star recovers a known monotone-onset pattern. ----------- #
    thr = 0.1
    # Barrier high early, drops below threshold from spawn step 250 onward.
    bbs = {0: 1.0, 50: 0.8, 100: 0.5, 250: 0.05, 500: 0.02, 1000: 0.01}
    ks = k_star(bbs, thr)
    t4a = ks == 250
    # A late re-spike must NOT count as connected: the stays-below rule should
    # push k* past the spike.
    bbs2 = {0: 1.0, 100: 0.05, 250: 0.5, 500: 0.02, 1000: 0.01}
    ks2 = k_star(bbs2, thr)
    t4b = ks2 == 500
    # Never-connected -> None.
    bbs3 = {0: 1.0, 100: 0.9, 250: 0.8}
    ks3 = k_star(bbs3, thr)
    t4c = ks3 is None
    t4 = t4a and t4b and t4c
    print(f"[self-test 4] k_star monotone={ks}(exp 250) "
          f"respike={ks2}(exp 500) never={ks3}(exp None) "
          f"-> {'PASS' if t4 else 'FAIL'}")
    ok = ok and t4

    print(f"\nfork.py self-test: {'ALL PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    success = _self_test()
    sys.exit(0 if success else 1)

"""Direction 012 — spawn-fork trainer for the LMC-onset (k*) study.

Implements the Frankle et al. (arXiv:1912.05671) spawn-fork instrument on the
full-batch grokking testbed:

  PARENT run    : train a GrokTransformer with one optimizer family
                  ({muon, adamw, sgdm}) on modular addition, full-batch, saving a
                  checkpoint (state_dict) at each of ~9 log-spaced spawn steps.
  CHILD forks   : at each spawn step, fork TWO children from the parent
                  checkpoint, diverge them via an explicit noise source
                  (perturb | minibatch), and train each to the SAME fixed end
                  step. The two children's final weights feed fork.linear_barrier.
  k*            : fork.k_star over {spawn_step -> barrier} gives the earliest
                  spawn step after which the two children stay linearly connected.

Child divergence (the grokking testbed is full-batch DETERMINISTIC, so same-seed
children would be bit-identical) — BOTH mechanisms behind `child_noise`:
  - "perturb"   (default primary): add a tiny Gaussian perturbation to the spawn
                checkpoint, per-tensor std = perturb_scale * rms(W) (relative
                L2 ~= perturb_scale). The two children get DIFFERENT perturbation
                seeds; training is otherwise identical full-batch.
  - "minibatch" : children train with large-minibatch sampling (batch
                `minibatch_size` of the train pool) under DIFFERENT shuffle seeds;
                the spawn checkpoint itself is copied unperturbed. Divergence
                comes from the SGD noise of distinct minibatch orders.

Checkpoint policy: smoke / in-process runs keep spawn checkpoints in MEMORY only;
real runs (out_dir set) additionally write them under a temp dir and append a
per-run jsonl. Smoke writes NO files.

Import discipline (root README): this dir at sys.path[0], grokking APPENDED. We
reuse ONLY GrokTransformer / make_modular_dataset / train_test_split / Muon /
split_params_for_muon from grokking, plus the LOCAL fork.py. grokking is never
modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field

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

# grokking infra (appended path) — reuse only, never modify.
from model import GrokTransformer  # noqa: E402
from data import make_modular_dataset, train_test_split  # noqa: E402
from muon import Muon, split_params_for_muon  # noqa: E402

# local load-bearing module (wins via sys.path[0]).
from fork import spawn_child, linear_barrier, k_star, relative_perturbation  # noqa: E402


DEFAULT_SPAWN_STEPS = [0, 50, 100, 250, 500, 1000, 2000, 4000, 8000]


@dataclass
class Config:
    # task / data
    p: int = 97
    op: str = "add"
    train_frac: float = 0.4
    # model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_scale: float = 1.0
    # optimization
    optimizer: str = "adamw"        # "adamw" | "muon" | "sgdm"
    lr: float = 1e-3                # AdamW lr (and AdamW side of hybrids)
    muon_lr: float = 0.02           # Muon / SGDM lr for hidden matrices
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    # spawn-fork protocol
    spawn_steps: list = field(default_factory=lambda: list(DEFAULT_SPAWN_STEPS))
    child_end_step: int = 12000     # all children train to THIS step (fixed end)
    child_noise: str = "perturb"    # "perturb" | "minibatch"
    perturb_scale: float = 1e-3     # relative-L2 perturbation for "perturb"
    minibatch_size: int = 2048      # large-minibatch size for "minibatch"
    n_children: int = 2             # children per spawn point (barrier needs 2)
    barrier_points: int = 11        # alpha grid points for linear_barrier
    barrier_threshold: float = 0.1  # loss-barrier connectivity threshold for k*
    # bookkeeping
    seed: int = 0
    device: str = "cuda"


# --------------------------------------------------------------------------- #
# building blocks (mirror grokking's recipe; local copies, no grokking edits)
# --------------------------------------------------------------------------- #
def build_model(cfg: Config, vocab_size: int, device: str) -> GrokTransformer:
    return GrokTransformer(
        vocab_size=vocab_size, seq_len=3, d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, mlp_ratio=cfg.mlp_ratio,
        init_scale=cfg.init_scale,
    ).to(device)


def build_optimizer(model, cfg: Config):
    """muon/adamw/sgdm via the imported grokking split (hidden 2-D -> Muon/SGD)."""
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)]
    if cfg.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model)
        opt_muon = Muon(muon_p, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                        ns_steps=5, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_muon, opt_adamw]
    if cfg.optimizer == "sgdm":
        sgd_p, adamw_p = split_params_for_muon(model)
        opt_sgd = torch.optim.SGD(sgd_p, lr=cfg.muon_lr, momentum=0.95,
                                  nesterov=True, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_sgd, opt_adamw]
    raise ValueError(cfg.optimizer)


@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    logits = model(X)
    loss = F.cross_entropy(logits, Y).item()
    acc = (logits.argmax(-1) == Y).float().mean().item()
    return loss, acc


def _cpu_state(model):
    """Detached CPU clone of the model's state dict (a portable spawn checkpoint)."""
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def _minibatch(X, Y, batch_size: int, gen: torch.Generator):
    """Sample a minibatch WITHOUT replacement using a child-specific generator."""
    n = X.shape[0]
    if batch_size >= n:
        return X, Y
    idx = torch.randperm(n, generator=gen, device="cpu")[:batch_size].to(X.device)
    return X[idx], Y[idx]


def _train_to(model, optimizers, Xtr, Ytr, start_step: int, end_step: int,
              cfg: Config, mb_gen: "torch.Generator | None"):
    """Train `model` from start_step to end_step (full-batch or minibatch).

    If `mb_gen` is None: full-batch deterministic steps (parent + perturb child).
    Else: large-minibatch steps drawn with the child-specific generator (the
    minibatch divergence mechanism).
    """
    model.train()
    for _ in range(start_step, end_step):
        if mb_gen is None:
            Xb, Yb = Xtr, Ytr
        else:
            Xb, Yb = _minibatch(Xtr, Ytr, cfg.minibatch_size, mb_gen)
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
    return model


# --------------------------------------------------------------------------- #
# parent run: train and snapshot at each spawn step
# --------------------------------------------------------------------------- #
def run_parent(cfg: Config, X, Y, vocab_size: int, device: str):
    """Train the parent to the last spawn step, returning {spawn_step: state_dict}.

    Checkpoints are kept in MEMORY (CPU tensors). Callers that want them on disk
    write them; this function never touches the filesystem.
    """
    (Xtr, Ytr), (Xte, Yte) = train_test_split(X, Y, cfg.train_frac, seed=cfg.seed)
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, vocab_size, device)
    optimizers = build_optimizer(model, cfg)

    spawn_steps = sorted(set(int(s) for s in cfg.spawn_steps))
    checkpoints = {}
    step = 0
    for target in spawn_steps:
        _train_to(model, optimizers, Xtr, Ytr, step, target, cfg, mb_gen=None)
        step = target
        checkpoints[target] = _cpu_state(model)
    return checkpoints, (Xtr, Ytr), (Xte, Yte)


# --------------------------------------------------------------------------- #
# child continuation: fork from a spawn checkpoint, diverge, train to end
# --------------------------------------------------------------------------- #
def run_child(cfg: Config, parent_state, spawn_step: int, child_idx: int,
              vocab_size: int, Xtr, Ytr, device: str):
    """Fork ONE child from `parent_state`, diverge it, train to child_end_step.

    Returns the child's final CPU state dict. The two divergence mechanisms:
      - perturb  : Gaussian perturbation w/ a child-specific seed; full-batch.
      - minibatch: unperturbed copy; large-minibatch SGD w/ child-specific seed.
    """
    # child-specific seed: distinct per (seed, optimizer-run, spawn, child).
    child_seed = (cfg.seed * 1_000_003 + spawn_step) * 17 + child_idx

    child_init = spawn_child(parent_state, mode=cfg.child_noise,
                             scale=cfg.perturb_scale, seed=child_seed)
    child_init = {k: v.to(device) for k, v in child_init.items()}

    torch.manual_seed(child_seed)
    model = build_model(cfg, vocab_size, device)
    model.load_state_dict(child_init, strict=True)
    optimizers = build_optimizer(model, cfg)

    mb_gen = None
    if cfg.child_noise == "minibatch":
        mb_gen = torch.Generator().manual_seed(child_seed)

    end = max(cfg.child_end_step, spawn_step)
    _train_to(model, optimizers, Xtr, Ytr, spawn_step, end, cfg, mb_gen=mb_gen)
    return _cpu_state(model)


# --------------------------------------------------------------------------- #
# full single-run driver: parent + children + barriers + k*
# --------------------------------------------------------------------------- #
def run(cfg: Config, out_path: str | None = None, ckpt_dir: str | None = None):
    """One full spawn-fork run for one optimizer family.

    Trains the parent, forks `n_children` children at each spawn step, measures
    the pairwise linear barrier (child 0 vs child 1) at each spawn step, and
    extracts k*. Real runs (out_path set) append a per-run jsonl and may persist
    checkpoints to `ckpt_dir`; in-process/smoke runs keep everything in memory.
    """
    device = cfg.device if torch.cuda.is_available() else "cpu"
    X, Y = make_modular_dataset(cfg.p, cfg.op, device=device)
    vocab_size = cfg.p + 1

    t0 = time.time()
    # The barrier is measured on the TRAIN pool (the loss the optimizer shaped);
    # run_parent also returns the held-out split for callers who want test-set
    # barriers, which this headline run does not use.
    checkpoints, (Xtr, Ytr), _test_split = run_parent(cfg, X, Y, vocab_size, device)

    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": asdict(cfg)}) + "\n")
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    template = build_model(cfg, vocab_size, device)  # reused for barrier evals
    barrier_by_spawn = {}
    history = []
    for spawn_step in sorted(checkpoints.keys()):
        parent_state = checkpoints[spawn_step]
        if ckpt_dir:
            torch.save(parent_state,
                       os.path.join(ckpt_dir, f"spawn_{spawn_step}.pt"))
        # fork n_children; barrier uses the first two (the protocol's pair).
        child_states = [
            run_child(cfg, parent_state, spawn_step, ci, vocab_size,
                      Xtr, Ytr, device)
            for ci in range(cfg.n_children)
        ]
        sd_a = {k: v.to(device) for k, v in child_states[0].items()}
        sd_b = {k: v.to(device) for k, v in child_states[1].items()}
        res = linear_barrier(template, sd_a, sd_b, (Xtr, Ytr),
                             n_points=cfg.barrier_points)
        barrier_by_spawn[spawn_step] = res["barrier"]
        rec = {
            "spawn_step": spawn_step,
            "barrier": res["barrier"],
            "acc_barrier": res["acc_barrier"],
            "loss_endpoints": res["loss_endpoints"],
            "acc_endpoints": res["acc_endpoints"],
        }
        history.append(rec)
        if f:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    ks = k_star(barrier_by_spawn, cfg.barrier_threshold)
    elapsed = time.time() - t0
    summary = {
        **asdict(cfg),
        "vocab_size": vocab_size,
        "n_params": sum(p.numel() for p in template.parameters()),
        "barrier_by_spawn": barrier_by_spawn,
        "k_star": ks,
        "elapsed_sec": elapsed,
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


# --------------------------------------------------------------------------- #
# Smoke: labeled lines, <=1-2 steps total, in-memory only, NO files, exit 0, <60s
# --------------------------------------------------------------------------- #
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    # tiny config: small prime, tiny model, a single forward/step.
    cfg = Config(p=23, d_model=32, n_heads=4, n_layers=2, optimizer="muon",
                 train_frac=0.5, perturb_scale=1e-3, seed=0, device=device)
    X, Y = make_modular_dataset(cfg.p, cfg.op, device=device)
    vocab_size = cfg.p + 1
    (Xtr, Ytr), _ = train_test_split(X, Y, cfg.train_frac, seed=cfg.seed)

    # 1. dataset shape (train split of the modular-add pairs)
    print(f"SMOKE DATASET SHAPE: X={tuple(Xtr.shape)}, Y={tuple(Ytr.shape)}")

    # 2. param count
    model = build_model(cfg, vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. forward + loss
    logits = model(Xtr)
    loss = F.cross_entropy(logits, Ytr)
    print(f"SMOKE FORWARD LOSS: {loss.item():.6f}")

    # 4. one Muon-hybrid optimizer step
    optimizers = build_optimizer(model, cfg)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers:
        opt.step()
    print("SMOKE OPTIMIZER STEP: OK")

    # 5. bonus: fork machinery end-to-end (in-memory, no files). We use the
    #    1-step-trained `model` as the spawn checkpoint, then fork TWO children
    #    via distinct Gaussian perturbations and give EACH exactly one more step.
    #    The barrier between the two diverged children is a small POSITIVE number;
    #    the self-barrier (a child vs itself) is exactly 0.
    #
    #    Note: a pure perturbation of a single point cannot raise the
    #    interpolation midpoint above the worse endpoint (the path is monotone),
    #    so a positive barrier requires two endpoints that have DIVERGED via
    #    training — exactly what the real protocol does. We therefore use a
    #    visible probe scale (`probe_scale`, larger than the protocol default
    #    `perturb_scale`) so one extra step opens a measurable gap; this keeps the
    #    whole smoke at 1 parent step + 1 step per child (<= 2 steps per lineage).
    probe_scale = 0.5
    sd = _cpu_state(model)  # the (1-step-trained) parent / spawn snapshot

    def _perturb_then_step(seed: int):
        child = spawn_child(sd, mode="perturb", scale=probe_scale, seed=seed)
        child = {k: v.to(device) for k, v in child.items()}
        cm = build_model(cfg, vocab_size, device)
        cm.load_state_dict(child, strict=True)
        opts = build_optimizer(cm, cfg)
        cm.train()
        lg = cm(Xtr)
        ls = F.cross_entropy(lg, Ytr)
        for o in opts:
            o.zero_grad(set_to_none=True)
        ls.backward()
        for o in opts:
            o.step()
        return _cpu_state(cm)

    sd_c0 = _perturb_then_step(seed=1)
    sd_c1 = _perturb_then_step(seed=2)
    dev0 = {k: v.to(device) for k, v in sd_c0.items()}
    dev1 = {k: v.to(device) for k, v in sd_c1.items()}
    template = build_model(cfg, vocab_size, device)

    self_res = linear_barrier(template, dev0, dev0, (Xtr, Ytr), n_points=5)
    pair_res = linear_barrier(template, dev0, dev1, (Xtr, Ytr), n_points=5)
    # rel_perturb: relative L2 distance of one child's INITIAL perturbation, to
    # confirm spawn_child moved the weights by ~probe_scale.
    probe_child = spawn_child(sd, mode="perturb", scale=probe_scale, seed=1)
    rel = relative_perturbation(sd, probe_child)
    print(f"SMOKE FORK PROBE: self_barrier={self_res['barrier']:.6f} "
          f"perturbed_barrier={pair_res['barrier']:.6f} rel_perturb={rel:.6f}")


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser(description="spawn-fork LMC trainer (dir 012)")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    defaults = asdict(Config())
    for k, v in defaults.items():
        if k == "spawn_steps":
            ap.add_argument("--spawn_steps", type=int, nargs="+", default=v,
                            help="log-spaced spawn steps (space-separated ints)")
        elif isinstance(v, bool):
            ap.add_argument(f"--{k}", action="store_true", default=v)
            ap.add_argument(f"--no-{k}", dest=k, action="store_false")
        else:
            ap.add_argument(f"--{k}", type=type(v) if v is not None else str,
                            default=v)
    a = vars(ap.parse_args())
    smoke = a.pop("smoke")
    cfg = Config(**a)
    return cfg, smoke


if __name__ == "__main__":
    cfg, smoke = parse_args()
    if smoke:
        run_smoke()
        sys.exit(0)
    summary, _ = run(cfg, out_path=None)
    print(json.dumps(summary, indent=2))

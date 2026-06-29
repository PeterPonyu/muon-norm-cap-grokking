"""Direction 005 — continual-plasticity trainer.

One run = one (optimizer, arm, seed) configuration. We train a small MLP on a
LONG sequence of T tasks (default 100), K steps per task, and measure loss of
plasticity = the decay of per-task fit speed as more tasks are seen. After each
task we run the full read-only diagnostic suite (probes.py) so we can correlate
fit-speed decay with feature/weight spectral collapse, dead units, gradient
signal, and Hessian sharpness — the quantities the spectral-collapse hypothesis
(arXiv:2509.22335) says should drive plasticity loss.

Optimizer families compared: {muon, adamw, sgdm}. muon/sgdm use the local
hybrid split (interior 2-D matrices -> Muon/SGD-momentum; first/last/biases ->
AdamW), reusing the grokking Muon class by import. AdamW is the all-AdamW
baseline.

Per-task metrics
----------------
- steps_to_threshold : first step at which train accuracy on THIS task crosses
                       `acc_threshold` (fit speed; rises => plasticity loss).
                       None/censored if never reached within K steps.
- final_acc          : train accuracy on this task at the end of its K steps.
- final_loss         : train loss at the end of this task.
- probes             : the full plasticity_probe() suite, on a fixed probe batch
                       of THIS task.

Flags
-----
--smoke : print the labeled smoke lines, run <=2 tiny tasks x 1 step, write NO
          files, exit 0 in <60s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

# --- import discipline (root README): LOCAL dir FIRST, other dirs APPENDED. --
# We reuse ONLY the Muon optimizer class from grokking and top_eigenvalue from
# eos_tiny (the latter via probes.py). Our local model/data/probes must win, so
# this directory is inserted at the front and grokking is appended at the back.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_GROKKING_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.append(_GROKKING_DIR)

import torch
import torch.nn.functional as F

from muon import Muon  # noqa: E402  (grokking infra: reuse the optimizer class)

from data import ContinualSpec, build_task, sample_minibatch, probe_batch  # noqa: E402
from model import MLP, split_params_for_muon                               # noqa: E402
from probes import plasticity_probe                                        # noqa: E402


@dataclass
class Config:
    # task / data
    arm: str = "proj_shift"      # "proj_shift" | "label_refit" | "perm_mnist"
    d_in: int = 128
    m_feat: int = 128            # teacher hidden width (proj_shift only)
    n_classes: int = 10
    n_examples: int = 2048       # shared input-pool size
    n_tasks: int = 100           # length of the continual sequence (T)
    steps_per_task: int = 200    # K optimization steps per task
    batch_size: int = 256        # minibatch per step
    probe_n: int = 512           # probe-batch size for diagnostics
    acc_threshold: float = 0.80  # "fit" = train acc crosses this
    lam_iters: int = 20          # power-iteration steps for lambda_max
    # model
    width: int = 512
    n_layers: int = 3
    # optimization
    optimizer: str = "adamw"     # "adamw" | "muon" | "sgdm"
    lr: float = 1e-3             # AdamW lr (and AdamW side of hybrids)
    muon_lr: float = 0.02        # Muon / SGDM lr for interior matrices
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.98
    warm_start: bool = True      # carry weights/optimizer across tasks (the
                                 # realistic continual setting where plasticity
                                 # loss appears). --no-warm_start = cold-start
                                 # control (fresh model per task; the runner's
                                 # warm-start contrast arm).
    seed: int = 0
    device: str = "cuda"


def make_spec(cfg: Config) -> ContinualSpec:
    return ContinualSpec(arm=cfg.arm, d_in=cfg.d_in, m_feat=cfg.m_feat,
                         n_classes=cfg.n_classes, n_examples=cfg.n_examples,
                         seed=cfg.seed)


def build_model(cfg: Config, spec: ContinualSpec, device: str) -> MLP:
    return MLP(d_in=spec.input_dim, width=cfg.width, n_layers=cfg.n_layers,
               n_classes=cfg.n_classes).to(device)


def build_optimizer(model: MLP, cfg: Config):
    """muon/adamw/sgdm hybrid split (mirrors grokking's recipe, local split)."""
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
def _accuracy(model, X, Y) -> float:
    return float((model(X).argmax(dim=1) == Y).float().mean())


def train_one_task(model, optimizers, X, Y, cfg: Config, task_idx: int,
                   device: str):
    """Train `model` on one task for K steps; return per-task fit metrics.

    Tracks steps-to-threshold-accuracy (fit speed) using full-pool train acc.
    """
    threshold_step = None
    final_loss = None
    for step in range(cfg.steps_per_task):
        Xb, Yb = sample_minibatch(
            X, Y, cfg.batch_size,
            seed=(cfg.seed * 911 + task_idx) * 100003 + step)
        model.train()
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        final_loss = float(loss.detach())
        if threshold_step is None:
            if _accuracy(model, X, Y) >= cfg.acc_threshold:
                threshold_step = step
    final_acc = _accuracy(model, X, Y)
    return {
        "steps_to_threshold": threshold_step,   # None => never fit within K
        "final_acc": final_acc,
        "final_loss": final_loss,
    }


def run(cfg: Config, out_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    spec = make_spec(cfg)

    model = build_model(cfg, spec, device)
    # warm_start=True keeps a single model/optimizer across the whole sequence
    # (the realistic continual setting). The cold-start contrast (fresh model
    # per task) is the warm_start=False arm stub the runner exposes.
    optimizers = build_optimizer(model, cfg)

    history: list = []
    t0 = time.time()

    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": asdict(cfg)}) + "\n")

    for task_idx in range(cfg.n_tasks):
        if not cfg.warm_start:
            # cold-start arm: fresh model + optimizer each task (control).
            model = build_model(cfg, spec, device)
            optimizers = build_optimizer(model, cfg)

        X, Y = build_task(spec, task_idx, device=device)
        fit = train_one_task(model, optimizers, X, Y, cfg, task_idx, device)

        pa = probe_batch(X, Y, cfg.probe_n, seed=cfg.seed * 7 + task_idx)
        pb = probe_batch(X, Y, cfg.probe_n, seed=cfg.seed * 7 + task_idx + 50000)
        probes = plasticity_probe(model, pa, pb, lam_iters=cfg.lam_iters,
                                  lam_seed=cfg.seed)

        rec = {"task": task_idx, **fit, "probes": probes}
        history.append(rec)
        if f:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    elapsed = time.time() - t0
    fit_speeds = [r["steps_to_threshold"] for r in history]
    reached = [s for s in fit_speeds if s is not None]
    summary = {
        **asdict(cfg),
        "n_params": sum(p.numel() for p in model.parameters()),
        "mean_steps_to_threshold": (sum(reached) / len(reached)) if reached else None,
        "n_tasks_fit": len(reached),
        "first_task_steps": fit_speeds[0] if fit_speeds else None,
        "last_task_steps": fit_speeds[-1] if fit_speeds else None,
        "final_task_acc": history[-1]["final_acc"] if history else None,
        "final_dead_frac": history[-1]["probes"]["dead_frac"] if history else None,
        "final_feat_eff_rank": history[-1]["probes"]["feat_eff_rank"] if history else None,
        "elapsed_sec": elapsed,
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


# --------------------------------------------------------------------------- #
# Smoke: labeled lines, <=2 tiny tasks x 1 step, NO files, exit 0, <60s.
# --------------------------------------------------------------------------- #
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    # tiny config: 2 tasks, 1 step each, small pool/model/probe, few lam iters.
    cfg = Config(arm="proj_shift", d_in=128, n_classes=10, n_examples=128,
                 n_tasks=2, steps_per_task=1, batch_size=64, probe_n=64,
                 width=256, n_layers=3, optimizer="muon", lam_iters=5,
                 warm_start=True, seed=0)
    spec = make_spec(cfg)

    # 1. dataset shape (task-0 batch shapes)
    X, Y = build_task(spec, 0, device=device)
    Xb, Yb = sample_minibatch(X, Y, cfg.batch_size, seed=0)
    print(f"SMOKE DATASET SHAPE: X={tuple(Xb.shape)}, Y={tuple(Yb.shape)}")

    # 2. param count
    model = build_model(cfg, spec, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. forward + loss
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    print(f"SMOKE FORWARD LOSS: {loss.item():.6f}")

    # 4. one Muon-hybrid optimizer step
    optimizers = build_optimizer(model, cfg)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers:
        opt.step()
    print("SMOKE OPTIMIZER STEP: OK")

    # 5. bonus: full plasticity probe suite on the (barely-trained) model,
    #    including the imported hessian.py lambda_max with ~5 iters.
    pa = probe_batch(X, Y, cfg.probe_n, seed=0)
    pb = probe_batch(X, Y, cfg.probe_n, seed=12345)
    probes = plasticity_probe(model, pa, pb, lam_iters=cfg.lam_iters, lam_seed=0)
    print(f"SMOKE PLASTICITY PROBE: eff_rank={probes['feat_eff_rank']:.3f} "
          f"dead_frac={probes['dead_frac']:.3f} "
          f"lambda_max={probes['lambda_max']:.3f}")


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    defaults = asdict(Config())
    for k, v in defaults.items():
        if isinstance(v, bool):
            # argparse type=bool is a footgun; expose --flag / --no-flag.
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

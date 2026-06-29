"""Direction 008 — BB-task trainer with triad logging (online fresh-batch).

One run = one (optimizer, norm_position, n_layers, seed) configuration. Trains a
tiny causal transformer on the Bigram-Backcopy task and logs the extreme-token
triad (attention sink, massive activation / spike, value drain, residual peak)
plus the sink ablation cost over training — the optimizer × architecture
decomposition at the heart of Direction 008: does Muon (orthogonalized update,
OFF the Adam-SGD adaptivity line) form the same triad as AdamW, the same as
SGDM, and does sandwich norm (2603.05498) suppress it as it does under Adam?

Design (mirrors induction_emergence/train_induction.py):
- ONLINE fresh BB batch every step (no memorization phase); eval on a FIXED
  held-out stream so curves are comparable across steps/runs.
- Same muon/adamw/sgdm hybrid split as grokking: Muon or SGDM on 2-D hidden
  matrices, AdamW always on embeddings/unembed/norms — so the manipulated
  variable is the HIDDEN-MATRIX update geometry only (sgdm = Muon's momentum and
  lr WITHOUT Newton-Schulz; isolates orthogonalization).
- norm_position ∈ {pre, sandwich} is the second manipulated variable.
- Per-eval jsonl line: role accuracies/losses + per-layer triad metrics +
  scalar ablation_cost. Summary: final triad values + formation steps.

Flags
-----
--smoke : labeled smoke lines, <=1 training step, NO files written, exit 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict

# --- import grokking infra without modifying it (LOCAL first, grokking back) ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_GROKKING_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.append(_GROKKING_DIR)

import torch
import torch.nn.functional as F

from muon import Muon  # noqa: E402  (grokking infra)

from data import BBSpec, sample_batch, batch_seed  # noqa: E402
from model import SinkTransformer, split_params_for_muon, NORM_POSITIONS  # noqa: E402
from probes import (  # noqa: E402
    triad_metrics, role_accuracies, ablation_cost,
    detect_formation, detect_formation_down,
)


@dataclass
class Config:
    # task / data
    vocab_size: int = 64
    seq_len: int = 128
    n_triggers: int = 3
    trigger_boost: float = 4.0
    chain_seed: int = 1234        # task identity (FIXED across the grid)
    batch_size: int = 64
    eval_batch: int = 256
    n_eval_batches: int = 1
    # model (grokking architecture family; depth is a grid axis)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    init_scale: float = 1.0
    norm_position: str = "pre"    # "pre" | "sandwich"
    # optimization
    optimizer: str = "adamw"      # "adamw" | "muon" | "sgdm"
    lr: float = 1e-3              # AdamW lr (and AdamW side of hybrids)
    muon_lr: float = 0.02         # Muon/SGDM lr for hidden matrices
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.98
    steps: int = 10000
    eval_every: int = 100
    # triad probe / formation thresholds
    query_offset: int = 8
    sink_threshold: float = 0.5    # sink_ratio up-crossing = sink formed
    drain_threshold: float = 0.5   # value_drain down-crossing = drain formed
    peak_threshold: float = 3.0    # residual_peak up-crossing = peak formed
    seed: int = 0
    device: str = "cuda"


def make_spec(cfg: Config) -> BBSpec:
    return BBSpec(vocab_size=cfg.vocab_size, seq_len=cfg.seq_len,
                  n_triggers=cfg.n_triggers, trigger_boost=cfg.trigger_boost,
                  chain_seed=cfg.chain_seed)


def build_model(cfg: Config, spec: BBSpec, device: str) -> SinkTransformer:
    assert cfg.norm_position in NORM_POSITIONS, cfg.norm_position
    return SinkTransformer(
        vocab_size=spec.vocab_size,
        seq_len=spec.seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio,
        init_scale=cfg.init_scale,
        norm_position=cfg.norm_position,
    ).to(device)


def build_optimizer(model, cfg: Config):
    """Same muon/adamw/sgdm hybrid split as grokking's train.py (and 007)."""
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)]
    elif cfg.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model)
        opt_muon = Muon(muon_p, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                        ns_steps=5, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_muon, opt_adamw]
    elif cfg.optimizer == "sgdm":
        sgd_p, adamw_p = split_params_for_muon(model)
        opt_sgd = torch.optim.SGD(sgd_p, lr=cfg.muon_lr, momentum=0.95,
                                  nesterov=True, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_sgd, opt_adamw]
    else:
        raise ValueError(cfg.optimizer)


def make_eval_batches(spec: BBSpec, cfg: Config, device: str):
    """Fixed held-out eval stream (disjoint seed range from training)."""
    return [sample_batch(spec, cfg.eval_batch,
                         seed=batch_seed(50_000_000 + cfg.seed, j),
                         device=device)
            for j in range(cfg.n_eval_batches)]


@torch.no_grad()
def evaluate(model, eval_batches, cfg: Config):
    """Average role metrics + triad metrics + ablation cost over the eval stream."""
    agg: dict = {}
    for batch in eval_batches:
        ra = role_accuracies(model, batch)
        tm = triad_metrics(model, batch[0], query_offset=cfg.query_offset)
        ac = {"ablation_cost": ablation_cost(model, batch)}
        for k, v in {**ra, **tm, **ac}.items():
            agg[k] = agg.get(k, 0.0) + v
    nb = max(1, len(eval_batches))
    return {k: v / nb for k, v in agg.items()}


def run(cfg: Config, out_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    spec = make_spec(cfg)

    model = build_model(cfg, spec, device)
    optimizers = build_optimizer(model, cfg)
    eval_batches = make_eval_batches(spec, cfg, device)

    history: list = []
    t0 = time.time()
    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": asdict(cfg)}) + "\n")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            model.eval()
            rec = {"step": step, **evaluate(model, eval_batches, cfg)}
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

        model.train()
        Xb, Yb, bc, bg = sample_batch(
            spec, cfg.batch_size, seed=batch_seed(cfg.seed, step), device=device)
        logits = model(Xb)
        V = logits.shape[-1]
        ce = F.cross_entropy(logits.reshape(-1, V), Yb.reshape(-1),
                             reduction="none").reshape(Yb.shape)
        tmask = (bc | bg).float()
        loss = (ce * tmask).sum() / tmask.sum().clamp(min=1)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    elapsed = time.time() - t0
    steps_seq = [r["step"] for r in history]
    summary = {
        **asdict(cfg),
        "final_backcopy_acc": history[-1]["backcopy_acc"],
        "final_bigram_loss": history[-1]["bigram_loss"],
        "final_sink_ratio": history[-1]["sink_ratio"],
        "final_value_drain": history[-1]["value_drain"],
        "final_residual_peak": history[-1]["residual_peak"],
        "final_spike_magnitude": history[-1]["spike_magnitude"],
        "final_ablation_cost": history[-1]["ablation_cost"],
        "sink_formation_step": detect_formation(
            steps_seq, [r["sink_ratio"] for r in history], cfg.sink_threshold),
        "drain_formation_step": detect_formation_down(
            steps_seq, [r["value_drain"] for r in history], cfg.drain_threshold),
        "peak_formation_step": detect_formation(
            steps_seq, [r["residual_peak"] for r in history], cfg.peak_threshold),
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": elapsed,
        "stopped_step": history[-1]["step"],
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


# ---------------------------------------------------------------------------
# Smoke: labeled lines, <=1 training step, NO files, exit 0.
# ---------------------------------------------------------------------------
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = Config(optimizer="muon", seed=0)
    spec = make_spec(cfg)

    # 1. one online batch: shapes + role-mask sanity
    X, Y, bc, bg = sample_batch(spec, cfg.batch_size, seed=0, device=device)
    print(f"SMOKE DATASET SHAPE: X={tuple(X.shape)} Y={tuple(Y.shape)} "
          f"backcopy_frac={float(bc.float().mean()):.3f}")

    # 2. param count
    model = build_model(cfg, spec, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. forward + masked loss
    logits = model(X)
    V = logits.shape[-1]
    ce = F.cross_entropy(logits.reshape(-1, V), Y.reshape(-1),
                         reduction="none").reshape(Y.shape)
    tmask = (bc | bg).float()
    loss = (ce * tmask).sum() / tmask.sum().clamp(min=1)
    print(f"SMOKE FORWARD LOSS: {loss.item():.6f}")

    # 4. one Muon-hybrid optimizer step
    optimizers = build_optimizer(model, cfg)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers:
        opt.step()
    print("SMOKE OPTIMIZER STEP: OK")

    # 5. bonus: full triad probe suite end-to-end on the (untrained) model
    model.eval()
    batch = (X, Y, bc, bg)
    tm = triad_metrics(model, X, query_offset=cfg.query_offset)
    ac = ablation_cost(model, batch)
    print(f"SMOKE SINK PROBE: sink_ratio={tm['sink_ratio']:.4f} "
          f"spike={tm['spike_magnitude']:.4f} ablation_cost={ac:.4f}")


def parse_args() -> tuple[Config, bool]:
    ap = argparse.ArgumentParser(description="BB-task triad trainer (008)")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    defaults = asdict(Config())
    for k, v in defaults.items():
        ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
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

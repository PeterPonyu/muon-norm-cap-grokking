"""Direction 002 — S5 mechanism runs.

Grid: optimizer ∈ {muon, adamw, sgdm} × init_scale ∈ {1.0, 3.0}
      × wd=0.01 × seeds 0-4, op="s5", mech=True
      steps = 20000 (sc1.0) / 12000 (sc3.0), eval_every=50
Output: ../../experiments/results/s5_mech/<name>.jsonl

Flags
-----
--smoke   : run smoke checks and exit 0 (NO files written, runtime <60s)
--dry-run : print planned cells and exit 0 (NO training, NO files)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Allow import from the sibling grokking package without modifying any file
# there and without installing anything.
# ---------------------------------------------------------------------------
_GROKKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.insert(0, _GROKKING_DIR)

import torch
import torch.nn.functional as F

from data import make_s5_dataset
from model import GrokTransformer
from muon import Muon, split_params_for_muon
from train import Config, run


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
OPTIMIZERS   = ["muon", "adamw", "sgdm"]
INIT_SCALES  = [1.0, 3.0]
WD           = 0.01
SEEDS        = list(range(5))          # 0-4
OUT          = os.path.join(os.path.dirname(__file__),
                            "..", "..", "experiments", "results", "s5_mech")

S5_VOCAB_SIZE = 121   # 120 perms + EQ token
S5_SEQ_LEN    = 3


def steps_for(sc: float) -> int:
    return 20000 if sc == 1.0 else 12000


def _build_cells():
    return [
        (opt, sc, WD, seed)
        for opt in OPTIMIZERS
        for sc in INIT_SCALES
        for seed in SEEDS
    ]


def already_done(path: str) -> bool:
    """True iff the jsonl exists and ends with a _summary line."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size == 0:
            return False
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", errors="replace")
    return '"_summary"' in tail


# ---------------------------------------------------------------------------
# Smoke check — no files written, exits 0
# ---------------------------------------------------------------------------
def run_smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # 1. Dataset shape
    X, Y = make_s5_dataset(device=device)
    print(f"SMOKE DATASET SHAPE: X={tuple(X.shape)}, y={tuple(Y.shape)}")

    # 2. Param count — build model exactly as the real run would
    cfg = Config(op="s5", init_scale=1.0, optimizer="muon",
                 weight_decay=WD, seed=0)
    model = GrokTransformer(
        vocab_size=S5_VOCAB_SIZE,
        seq_len=S5_SEQ_LEN,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio,
        init_scale=cfg.init_scale,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SMOKE PARAM COUNT: {n_params}")

    # 3. Forward pass + loss (use a small slice to stay fast)
    model.eval()
    with torch.no_grad():
        slice_X = X[:256]
        slice_Y = Y[:256]
        logits = model(slice_X)
        fwd_loss = F.cross_entropy(logits, slice_Y).item()
    print(f"SMOKE FORWARD LOSS: {fwd_loss:.6f}")

    # 4. One Muon-hybrid optimizer step
    muon_p, adamw_p = split_params_for_muon(model)
    opt_muon  = Muon(muon_p, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                     ns_steps=5, weight_decay=WD)
    opt_adamw = torch.optim.AdamW(
        adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=WD)
    optimizers = [opt_muon, opt_adamw]

    model.train()
    logits = model(slice_X)
    loss = F.cross_entropy(logits, slice_Y)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss.backward()
    for opt in optimizers:
        opt.step()
    print("SMOKE OPTIMIZER STEP: OK")


# ---------------------------------------------------------------------------
# Real grid runner
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="S5 mechanism grid runner")
    ap.add_argument("--smoke",   action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells and exit (no training)")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        sys.exit(0)

    cells = _build_cells()

    if args.dry_run:
        print(f"[s5_mech] dry-run: {len(cells)} cells planned")
        for i, (opt, sc, wd, seed) in enumerate(cells):
            st = steps_for(sc)
            name = f"{opt}_sc{sc}_wd{wd}_s{seed}"
            print(f"  [{i+1:02d}/{len(cells)}] {name}  steps={st}")
        sys.exit(0)

    # --- Real training path (only reached when neither flag is set) ---
    os.makedirs(OUT, exist_ok=True)
    print(f"[s5_mech] {len(cells)} cells -> {OUT}", flush=True)

    for i, (opt, sc, wd, seed) in enumerate(cells):
        name = f"{opt}_sc{sc}_wd{wd}_s{seed}"
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        st = steps_for(sc)
        cfg = Config(
            op="s5",
            optimizer=opt,
            init_scale=sc,
            weight_decay=wd,
            seed=seed,
            steps=st,
            eval_every=50,
            mech=True,
        )
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"mem={s['memorize_step']} grok={s['grok_step']} "
            f"test={s['final_test_acc']:.3f} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[s5_mech] DONE", flush=True)


if __name__ == "__main__":
    main()

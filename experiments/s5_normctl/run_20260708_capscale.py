"""Scale ladder for the norm-controlled-Muon cap (findings-021 follow-up).

Two arms, both reusing run_s5_normctl.py's cap-injection mechanism UNCHANGED
(NormControlledMuon + make_builder monkeypatch of train.build_optimizer):

ARM "capscale" — mod-add SCALE ladder. Does the growth-cap dose-response
survive moving off the p=97/d=128 operating point? Rungs (p, d_model):
  (97,128), (251,256), (337,384)
k in {inf (vanilla Muon growth control), 2, 1} x optimizer muon, plus a plain
AdamW-uncapped reference cell per rung (no Muon, no cap concept at all).
Seeds: 5 for p97/p251, 3 for p337 (compute budget; p337*d384 full-batch is the
biggest cell here, ~7GB on a 24GB card).

ARM "capwd" — S5 width/depth ladder. Does the cap dose-response (k=inf vs
k=1) hold as capacity grows? d_model in {128,256,512} x n_layers in {2,4} x
k in {inf,1} x optimizer muon x 5 seeds.

Same operating point as run_s5_normctl.py otherwise: weight_decay=0.01,
init_scale=1.0, mech=True, steps=20000, eval_every=50.

  python run_20260708_capscale.py --smoke              # CPU: tiny integration cells
  python run_20260708_capscale.py --dry-run
  python run_20260708_capscale.py [--num-shards N --shard-id I]

Output: ../../experiments/results/s5_normctl_scale/<name>.jsonl  (own namespace;
does NOT write into results/s5_normctl/). Resume-aware via `_summary` marker.
"""
from __future__ import annotations

import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
_GROKKING = os.path.abspath(os.path.join(_THIS, "..", "grokking"))
if _GROKKING not in sys.path:
    sys.path.insert(0, _GROKKING)
_EXP = os.path.dirname(_THIS)
if _EXP not in sys.path:
    sys.path.append(_EXP)

import train as TRAIN  # noqa: E402  (grokking trainer; we monkeypatch build_optimizer)
from run_s5_normctl import (  # noqa: E402  (reuse cap-injection mechanism unchanged)
    make_builder, already_done, _ktag, STEPS, EVAL_EVERY, WD, INIT_SCALE,
)

# Captured BEFORE any monkeypatching, so AdamW-uncapped reference cells can
# restore the real train.build_optimizer (which branches on cfg.optimizer).
_ORIG_BUILD_OPTIMIZER = TRAIN.build_optimizer

OUT = os.path.join(_THIS, "..", "..", "experiments", "results", "s5_normctl_scale")

CAPSCALE_RUNGS = [(97, 128, 5), (251, 256, 5), (337, 384, 3)]  # (p, d_model, n_seeds)
CAPSCALE_KS = [None, 2.0, 1.0]                                 # inf, 2, 1

CAPWD_DS = [128, 256, 512]
CAPWD_LAYERS = [2, 4]
CAPWD_KS = [None, 1.0]                                         # inf, 1
CAPWD_SEEDS = 5


def build_cells():
    """Each cell: (arm, p, d_model, n_layers, k, opt, seed). p=None for capwd (op=s5, unused)."""
    cells = []
    for p, d, n_seeds in CAPSCALE_RUNGS:
        for k in CAPSCALE_KS:
            for s in range(n_seeds):
                cells.append(("capscale", p, d, 2, k, "muon", s))
        for s in range(n_seeds):
            cells.append(("capscale", p, d, 2, None, "adamw", s))
    for d in CAPWD_DS:
        for n_layers in CAPWD_LAYERS:
            for k in CAPWD_KS:
                for s in range(CAPWD_SEEDS):
                    cells.append(("capwd", None, d, n_layers, k, "muon", s))
    return cells


def cell_name(cell):
    arm, p, d, n_layers, k, opt, seed = cell
    ptag = f"p{p}_" if p is not None else ""
    return f"{arm}_{ptag}d{d}_L{n_layers}_{_ktag(k)}_{opt}_s{seed}"


def run_smoke():
    # Tiny capscale muon-capped cell (k=1) — exercises the reused cap mechanism.
    TRAIN.build_optimizer = make_builder(1.0)
    cfg = TRAIN.Config(op="add", p=7, d_model=32, optimizer="muon",
                       init_scale=INIT_SCALE, weight_decay=WD, seed=0,
                       steps=20, eval_every=10, mech=True, device="cpu")
    s, _ = TRAIN.run(cfg, out_path=None)
    assert "grok_step" in s and "final_wn_hidden" in s
    print(f"SMOKE capscale muon k=1 (p=7,d=32): "
          f"final_test_acc={s['final_test_acc']:.3f} OK")

    # Tiny capscale AdamW-uncapped reference cell — restores real build_optimizer.
    TRAIN.build_optimizer = _ORIG_BUILD_OPTIMIZER
    cfg = TRAIN.Config(op="add", p=7, d_model=32, optimizer="adamw",
                       init_scale=INIT_SCALE, weight_decay=WD, seed=0,
                       steps=20, eval_every=10, mech=True, device="cpu")
    s, _ = TRAIN.run(cfg, out_path=None)
    assert "grok_step" in s
    print(f"SMOKE capscale adamw-uncapped ref (p=7,d=32): "
          f"final_test_acc={s['final_test_acc']:.3f} OK")

    # Tiny capwd muon-capped s5 cell (k=1, n_layers=2).
    TRAIN.build_optimizer = make_builder(1.0)
    cfg = TRAIN.Config(op="s5", d_model=32, n_layers=2, optimizer="muon",
                       init_scale=INIT_SCALE, weight_decay=WD, seed=0,
                       steps=20, eval_every=10, mech=True, device="cpu")
    s, _ = TRAIN.run(cfg, out_path=None)
    assert "grok_step" in s
    print(f"SMOKE capwd muon k=1 s5 (d=32,L=2): "
          f"final_test_acc={s['final_test_acc']:.3f} OK")
    print("RUN_20260708_CAPSCALE SMOKE PASS: cap-injection reused unchanged from "
          "run_s5_normctl; capscale (add p/d ladder) + capwd (s5 width/depth) "
          "both wired; zero writes")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    try:
        from runner_utils import add_shard_args, shard_cells, validate_shard_args
        add_shard_args(ap); _shard = True
    except Exception:
        _shard = False
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    cells = build_cells()
    if _shard:
        validate_shard_args(args)
        cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for cell in cells:
            print(cell_name(cell))
        print(f"{len(cells)} cells")
        return

    os.makedirs(OUT, exist_ok=True)
    print(f"[s5_normctl_scale] {len(cells)} cells -> {OUT}", flush=True)
    for i, cell in enumerate(cells):
        arm, p, d, n_layers, k, opt, seed = cell
        name = cell_name(cell)
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        if opt == "muon":
            TRAIN.build_optimizer = make_builder(k)
        else:
            TRAIN.build_optimizer = _ORIG_BUILD_OPTIMIZER
        cfg_kwargs = dict(d_model=d, n_layers=n_layers, optimizer=opt,
                          init_scale=INIT_SCALE, weight_decay=WD, seed=seed,
                          steps=STEPS, eval_every=EVAL_EVERY, mech=True)
        if arm == "capscale":
            cfg_kwargs.update(op="add", p=p)
        else:
            cfg_kwargs.update(op="s5")
        cfg = TRAIN.Config(**cfg_kwargs)
        t0 = time.time()
        s, _ = TRAIN.run(cfg, out_path=path)
        print(f"[{i+1}/{len(cells)}] {name}: grok={s['grok_step']} "
              f"test={s['final_test_acc']:.3f} wn_hidden={s['final_wn_hidden']:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[s5_normctl_scale] DONE", flush=True)


if __name__ == "__main__":
    main()

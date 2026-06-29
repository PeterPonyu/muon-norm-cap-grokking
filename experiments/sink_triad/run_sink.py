"""Direction 008 — attention-sink triad grid runner (NOT executed yet).

Core grid (30 cells):
    optimizer     ∈ {muon, adamw, sgdm}
    norm_position ∈ {pre, sandwich}
    seed          ∈ 0..4                  (5 seeds for cross-seed variance)
3 x 2 x 5 = 30 cells, all at the default depth (n_layers=2). Each cell trains
online (fresh BB batch) and logs the triad curves (sink_ratio, spike_magnitude,
value_drain, residual_peak) + ablation_cost; the summary records the formation
steps. The two axes adjudicate the triad's cause:
  - optimizer family (Muon vs AdamW vs SGDM, with Muon OFF the Adam-SGD
    adaptivity line) tests the OPTIMIZATION-artifact hypothesis
    (coordinate-adaptivity / gradient-sink, 2410.13835 / 2603.17771);
  - norm_position (pre vs sandwich) tests the PRE-NORM-architecture hypothesis
    (2603.05498's decoupling intervention);
  - ablation_cost across all cells tests FUNCTIONAL NECESSITY.

Depth dose-response arm (flagged, +18 cells):
    --depth-arm adds the secondary depth sweep n_layers ∈ {1, 3} (the core grid
    already covers depth 2) at norm=pre over seeds 0..2:
        3 opt x 2 depths x 3 seeds = 18 cells.
    Together with the core depth-2 cells this gives the 1/2/3-layer
    dose-response described in the direction brief.

Output: ../../experiments/results/sink_triad/<name>.jsonl  (per convention).
Resume-aware: a cell whose jsonl already ends with a _summary line is skipped.

Flags
-----
--smoke      : delegate to train_sink smoke (no files, <60s) and exit 0.
--dry-run    : print planned cells and exit 0 (launches NOTHING).
--depth-arm  : include the depth dose-response arm in the (dry-run or real) plan.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_EXPERIMENTS_DIR = os.path.dirname(_THIS_DIR)
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.append(_EXPERIMENTS_DIR)

from train_sink import Config, run, run_smoke  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args,
    shard_cells,
    shard_suffix,
    validate_shard_args,
)


OPTIMIZERS = ["muon", "adamw", "sgdm"]
NORM_POSITIONS = ["pre", "sandwich"]
SEEDS = list(range(5))            # 0-4
STEPS = 12000
EVAL_EVERY = 150

# depth dose-response arm (secondary): depths {1, 3} at norm=pre, seeds 0..2
# (core grid already covers depth 2 -> together a 1/2/3-layer dose-response).
DEPTH_ARM_DEPTHS = [1, 3]
DEPTH_ARM_NORM = "pre"
DEPTH_ARM_SEEDS = list(range(3))  # 0-2

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results", "sink_triad")


def _core_cells():
    """Core grid: 3 opt x 2 norm x 5 seeds = 30 cells (n_layers=2)."""
    cells = []
    for opt in OPTIMIZERS:
        for norm in NORM_POSITIONS:
            for seed in SEEDS:
                name = f"{opt}_{norm}_s{seed}"
                cells.append((name, dict(optimizer=opt, norm_position=norm,
                                         seed=seed, n_layers=2)))
    return cells


def _depth_arm_cells():
    """Depth dose-response: 3 opt x {1,3} depth x 3 seeds = 18 cells (norm=pre)."""
    cells = []
    for opt in OPTIMIZERS:
        for depth in DEPTH_ARM_DEPTHS:
            for seed in DEPTH_ARM_SEEDS:
                name = f"depth_{opt}_{DEPTH_ARM_NORM}_L{depth}_s{seed}"
                cells.append((name, dict(optimizer=opt,
                                         norm_position=DEPTH_ARM_NORM,
                                         n_layers=depth, seed=seed)))
    return cells


def already_done(path: str) -> bool:
    """True iff the jsonl exists and ends with a _summary line."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if size == 0:
            return False
        fh.seek(max(0, size - 4096))
        tail = fh.read().decode("utf-8", errors="replace")
    return '"_summary"' in tail


def main():
    ap = argparse.ArgumentParser(description="sink-triad grid runner (008)")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells and exit (no training)")
    ap.add_argument("--depth-arm", action="store_true",
                    help="Include the depth dose-response arm in the plan")
    add_shard_args(ap)
    args = ap.parse_args()
    validate_shard_args(args)

    if args.smoke:
        run_smoke()
        sys.exit(0)

    all_cells = _core_cells()
    if args.depth_arm:
        all_cells = all_cells + _depth_arm_cells()
    cells = shard_cells(all_cells, args.num_shards, args.shard_id)

    if args.dry_run:
        n_core, n_depth = len(_core_cells()), len(_depth_arm_cells())
        print(f"[sink_triad] dry-run: {len(cells)} cells planned "
              f"(core={n_core}" + (f" + depth_arm={n_depth}" if args.depth_arm else "")
              + ")"
              + shard_suffix(args.num_shards, args.shard_id,
                             len(all_cells), len(cells)))
        print(f"  core grid : {len(OPTIMIZERS)} opt x {len(NORM_POSITIONS)} norm "
              f"x {len(SEEDS)} seeds = {n_core}")
        if args.depth_arm:
            print(f"  depth arm : {len(OPTIMIZERS)} opt x {len(DEPTH_ARM_DEPTHS)} "
                  f"depths x {len(DEPTH_ARM_SEEDS)} seeds (norm={DEPTH_ARM_NORM}) "
                  f"= {n_depth}")
        for i, (name, ov) in enumerate(cells):
            print(f"  [{i+1:02d}/{len(cells)}] {name}  steps={STEPS}  {ov}")
        sys.exit(0)

    # --- real training path (only when neither smoke nor dry-run set) ---
    os.makedirs(OUT, exist_ok=True)
    print(f"[sink_triad] {len(cells)} cells -> {OUT}"
          + shard_suffix(args.num_shards, args.shard_id,
                         len(all_cells), len(cells)),
          flush=True)

    for i, (name, ov) in enumerate(cells):
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = Config(steps=STEPS, eval_every=EVAL_EVERY, **ov)
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"sink={s['final_sink_ratio']:.3f} "
            f"drain={s['final_value_drain']:.3f} "
            f"peak={s['final_residual_peak']:.2f} "
            f"abl={s['final_ablation_cost']:.3f} "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[sink_triad] DONE", flush=True)


if __name__ == "__main__":
    main()

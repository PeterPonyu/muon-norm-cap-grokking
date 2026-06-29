"""Direction 005 — Muon-plasticity grid runner (NOT executed yet).

Grid (the causal-test design):
    optimizer ∈ {muon, adamw, sgdm}
    arm       ∈ {proj_shift, label_refit}
    seed      ∈ 0..4
3 x 2 x 5 = 30 cells. Each cell trains a small MLP on a 100-task continual
sequence (warm-start: weights carried across tasks) and logs per-task fit speed
+ the full plasticity-probe suite. We then ask whether Muon's orthogonalized
updates preserve fit speed / feature rank where AdamW / SGDM lose plasticity.

Warm-start contrast arm (FLAGGED, stub): --include-coldstart adds a cold-start
control for each cell (fresh model per task, --no-warm_start) so the per-task
fit-speed decay can be separated from intrinsic per-task difficulty. Off by
default to keep the headline grid at 30 cells.

Output: ../../experiments/results/muon_plasticity/<name>.jsonl
Resume-aware: skips a cell whose jsonl already ends with a _summary line.

Flags
-----
--smoke           : delegate to train_plasticity smoke (no files, <60s), exit 0.
--dry-run         : print planned cells and exit 0 (launches NOTHING).
--include-coldstart : add the cold-start (--no-warm_start) contrast cells.
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

from train_plasticity import Config, run, run_smoke  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args,
    shard_cells,
    shard_suffix,
    validate_shard_args,
)


OPTIMIZERS = ["muon", "adamw", "sgdm"]
ARMS = ["proj_shift", "label_refit"]
SEEDS = list(range(5))            # 0-4

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results", "muon_plasticity")


def _build_cells(include_coldstart: bool):
    """Return list of (optimizer, arm, seed, warm_start) cells.

    Headline grid is warm-start only (30 cells). The flagged cold-start contrast
    arm doubles the grid (adds 30 cold-start cells) when requested.
    """
    warm_flags = [True] + ([False] if include_coldstart else [])
    return [
        (opt, arm, seed, warm)
        for warm in warm_flags
        for opt in OPTIMIZERS
        for arm in ARMS
        for seed in SEEDS
    ]


def _cell_name(opt: str, arm: str, seed: int, warm: bool) -> str:
    tag = "warm" if warm else "cold"
    return f"{opt}_{arm}_{tag}_s{seed}"


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
    ap = argparse.ArgumentParser(description="muon-plasticity grid runner")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print planned cells and exit (no training)")
    ap.add_argument("--include-coldstart", action="store_true",
                    help="Add the cold-start (no warm_start) contrast cells")
    add_shard_args(ap)
    args = ap.parse_args()
    validate_shard_args(args)

    if args.smoke:
        run_smoke()
        sys.exit(0)

    all_cells = _build_cells(args.include_coldstart)
    cells = shard_cells(all_cells, args.num_shards, args.shard_id)

    if args.dry_run:
        print(f"[muon_plasticity] dry-run: {len(cells)} cells planned"
              + shard_suffix(args.num_shards, args.shard_id,
                             len(all_cells), len(cells)))
        for i, (opt, arm, seed, warm) in enumerate(cells):
            name = _cell_name(opt, arm, seed, warm)
            print(f"  [{i+1:02d}/{len(cells)}] {name}")
        sys.exit(0)

    # --- real training path (only when neither flag set) ---
    os.makedirs(OUT, exist_ok=True)
    print(f"[muon_plasticity] {len(cells)} cells -> {OUT}"
          + shard_suffix(args.num_shards, args.shard_id,
                         len(all_cells), len(cells)),
          flush=True)

    for i, (opt, arm, seed, warm) in enumerate(cells):
        name = _cell_name(opt, arm, seed, warm)
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = Config(optimizer=opt, arm=arm, seed=seed, warm_start=warm)
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"mean_fit={s['mean_steps_to_threshold']} "
            f"first={s['first_task_steps']} last={s['last_task_steps']} "
            f"dead={s['final_dead_frac']:.3f} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[muon_plasticity] DONE", flush=True)


if __name__ == "__main__":
    main()

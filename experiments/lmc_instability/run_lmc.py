"""Direction 012 — spawn-fork LMC grid runner (NOT executed yet).

Grid (the k*-onset design, Frankle et al. arXiv:1912.05671 on the grokking
testbed):
    optimizer  ∈ {muon, adamw, sgdm}                          (3)
    spawn step ∈ {0,50,100,250,500,1000,2000,4000,8000}       (9, log-spaced)
    child      ∈ {0, 1}                                        (2 per spawn point)
    seed       ∈ {0, 1, 2}                                     (3)

Counting (what a real launch would do):
    PARENTS  = optimizers × seeds                = 3 × 3       = 9  parent RUNS,
               but each parent snapshots all 9 spawn points, so the number of
               (optimizer × spawn × seed) PARENT CELLS = 3 × 9 × 3 = 81.
               The task's "~27 parents" counts (optimizer × spawn) parent
               checkpoints at the headline seed; with 3 seeds that is 81 parent
               checkpoints, produced by 9 actual parent training runs.
    CHILDREN = optimizer × spawn × child × seed  = 3 × 9 × 2 × 3 = 162 child runs.

We report BOTH framings in --dry-run so the counts are unambiguous: 9 parent
training runs (re-using each run's 9 spawn checkpoints), 81 parent checkpoint
cells, and 162 child continuation runs. Children are SHORT: each trains from its
spawn step to the shared `child_end_step`.

One grid CELL here = one (optimizer, seed) PARENT run that internally forks all
spawn × child children and emits one jsonl with the per-spawn barriers + k*.
That keeps resume granularity at the parent-run level (9 cells) while the heavy
work (162 children) lives inside run().

Output : ../../experiments/results/lmc_instability/<name>.jsonl  (real runs only)
Checkpoints (real runs): ../../experiments/results/lmc_instability/ckpt/<name>/
Resume-aware: a cell whose jsonl already ends with a _summary line is skipped.

Flags
-----
--smoke   : delegate to train_fork smoke (no files, <60s), exit 0.
--dry-run : print the parent + child cell plan and exit 0 (launches NOTHING).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# --- import discipline: LOCAL dir FIRST, experiments dir APPENDED. ---------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_EXPERIMENTS_DIR = os.path.dirname(_THIS_DIR)
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.append(_EXPERIMENTS_DIR)

from train_fork import Config, run, run_smoke, DEFAULT_SPAWN_STEPS  # noqa: E402
from runner_utils import (  # noqa: E402
    add_shard_args,
    shard_cells,
    shard_suffix,
    validate_shard_args,
)


OPTIMIZERS = ["muon", "adamw", "sgdm"]
SPAWN_STEPS = list(DEFAULT_SPAWN_STEPS)   # 0,50,100,250,500,1000,2000,4000,8000
SEEDS = list(range(3))                    # 0-2
N_CHILDREN = 2

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results",
                   "lmc_instability")


def _build_cells():
    """Return the list of (optimizer, seed) PARENT-run cells.

    Each cell's run() trains one parent (snapshotting all spawn steps) and forks
    all spawn × child children internally. 3 optimizers × 3 seeds = 9 cells.
    """
    return [(opt, seed) for opt in OPTIMIZERS for seed in SEEDS]


def _cell_name(opt: str, seed: int) -> str:
    return f"{opt}_s{seed}"


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


def _print_plan(cells, num_shards, shard_id, all_cells):
    n_opt = len(OPTIMIZERS)
    n_spawn = len(SPAWN_STEPS)
    n_seed = len(SEEDS)
    parent_checkpoints = n_opt * n_spawn * n_seed          # 81
    parent_runs = n_opt * n_seed                            # 9
    children = n_opt * n_spawn * N_CHILDREN * n_seed        # 162

    print("[lmc_instability] dry-run cell plan")
    print(f"  optimizers : {OPTIMIZERS}")
    print(f"  spawn steps: {SPAWN_STEPS}  ({n_spawn} log-spaced)")
    print(f"  children   : {N_CHILDREN} per spawn point")
    print(f"  seeds      : {SEEDS}")
    print("  ---- counts ----")
    print(f"  PARENT runs (optimizer × seed)              = {parent_runs}")
    print(f"  PARENT checkpoints (optimizer × spawn × seed) = {parent_checkpoints}")
    print(f"  CHILD runs (optimizer × spawn × child × seed) = {children}")
    print(f"  -> grid CELLS (one parent-run each)           = {len(all_cells)}"
          + shard_suffix(num_shards, shard_id, len(all_cells), len(cells)))
    print("  ---- cells ----")
    for i, (opt, seed) in enumerate(cells):
        name = _cell_name(opt, seed)
        # each cell internally produces n_spawn parent checkpoints and
        # n_spawn * N_CHILDREN child runs.
        n_child_here = n_spawn * N_CHILDREN
        print(f"  [{i+1:02d}/{len(cells)}] {name}: "
              f"{n_spawn} spawn ckpts, {n_child_here} children")


def main():
    ap = argparse.ArgumentParser(description="lmc_instability spawn-fork grid")
    ap.add_argument("--smoke", action="store_true",
                    help="Run smoke checks and exit (no files written)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the parent+child cell plan and exit (no training)")
    add_shard_args(ap)
    args = ap.parse_args()
    validate_shard_args(args)

    if args.smoke:
        run_smoke()
        sys.exit(0)

    all_cells = _build_cells()
    cells = shard_cells(all_cells, args.num_shards, args.shard_id)

    if args.dry_run:
        _print_plan(cells, args.num_shards, args.shard_id, all_cells)
        sys.exit(0)

    # --- real training path (only when neither flag set) ---
    os.makedirs(OUT, exist_ok=True)
    print(f"[lmc_instability] {len(cells)} parent-cells -> {OUT}"
          + shard_suffix(args.num_shards, args.shard_id,
                         len(all_cells), len(cells)),
          flush=True)

    for i, (opt, seed) in enumerate(cells):
        name = _cell_name(opt, seed)
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        ckpt_dir = os.path.join(OUT, "ckpt", name)
        cfg = Config(optimizer=opt, seed=seed,
                     spawn_steps=list(SPAWN_STEPS), n_children=N_CHILDREN)
        t0 = time.time()
        s, _ = run(cfg, out_path=path, ckpt_dir=ckpt_dir)
        print(
            f"[{i+1}/{len(cells)}] {name}: "
            f"k_star={s['k_star']} "
            f"barriers={ {k: round(v, 4) for k, v in s['barrier_by_spawn'].items()} } "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )

    print("[lmc_instability] DONE", flush=True)


if __name__ == "__main__":
    main()

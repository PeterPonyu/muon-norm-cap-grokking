"""Group-complexity ladder under plain Adam (weight_decay=0.0) — no-WD control.

Stage B (run_group.py) sweeps {adamw, muon, sgdm} at weight_decay=1.0 across the
comparable-|G|^2 ladders. This runner adds one more arm at the SAME operating
point (d_model=256, steps=30000, eval_every=100, train_frac=default, same
groups) but with optimizer=adamw and weight_decay=0.0 (i.e. plain Adam, no
decoupled decay) — isolating whether Stage B's necessity-threshold result
depends on the weight-decay-driven norm-growth route rather than the optimizer
family itself.

Reuses run_group.run_cell UNCHANGED (weight_decay is now a plain parameter on
that function; existing Stage A/B callers are unaffected since they don't pass
it and keep the old default of 1.0).

Groups: Z60, D30, A5 (order-60 rung) and Z120, D60, S5 (order-120 rung) x 5 seeds.

  python run_20260708_adam_nowd.py --smoke      # CPU: tiny cell
  python run_20260708_adam_nowd.py --dry-run
  python run_20260708_adam_nowd.py [--num-shards N --shard-id I]

Output: ../../experiments/results/group_complexity_nowd/<name>.jsonl. Resume-safe.
"""
from __future__ import annotations

import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_THIS)                  # experiments/ — holds runner_utils
_GK = os.path.join(_EXP, "grokking")
for p in (_THIS, _GK, _EXP):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_group import run_cell, already_done  # noqa: E402  (reuse Stage B loop unchanged)

OUT = os.path.join(_THIS, "..", "..", "experiments", "results", "group_complexity_nowd")

GROUPS = ["Z60", "D30", "A5", "Z120", "D60", "S5"]
D_MODEL = 256
N_SEEDS = 5


def build_cells():
    return [(g, s) for g in GROUPS for s in range(N_SEEDS)]


def cell_name(group, seed):
    return f"nowd_{group}_adamw_d{D_MODEL}_s{seed}"


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
        s, h = run_cell("Z60", "adamw", 0, d_model=32, steps=50,
                        weight_decay=0.0)
        assert s["optimizer"] == "adamw" and s["weight_decay"] == 0.0
        print(f"SMOKE Z60/adamw wd=0 50-step: params={s['n_params']} "
              f"final_train_acc={s['final_train_acc']:.3f} OK")
        print("RUN_20260708_ADAM_NOWD SMOKE PASS")
        sys.exit(0)

    cells = build_cells()
    if _shard:
        validate_shard_args(args)
        cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for g, s in cells:
            print(cell_name(g, s))
        print(f"{len(cells)} cells")
        sys.exit(0)

    os.makedirs(OUT, exist_ok=True)
    print(f"[group_complexity_nowd] {len(cells)} cells -> {OUT}", flush=True)
    for i, (g, s) in enumerate(cells):
        name = cell_name(g, s)
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        t0 = time.time()
        summ, _ = run_cell(g, "adamw", s, d_model=D_MODEL, weight_decay=0.0,
                           out_path=path)
        print(f"[{i+1}/{len(cells)}] {name}: grokked={summ['grokked']} "
              f"grok_step={summ['grok_step']} test={summ['final_test_acc']:.3f} "
              f"norm_growth={summ['norm_growth_ratio']:.2f} ({time.time()-t0:.0f}s)",
              flush=True)
    print("[group_complexity_nowd] DONE", flush=True)


if __name__ == "__main__":
    main()

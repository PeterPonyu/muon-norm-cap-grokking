"""Run the grokking × optimizer grid and collect summaries.

Usage:
    python run_grid.py --out_dir ../../experiments/results/grid_main

Writes one <name>.jsonl per run (full history) plus summaries.csv / summaries.json
aggregating all runs. Resumable: skips runs whose .jsonl already has a _summary line.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time

from train import Config, run


def already_done(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            last = line
    if last:
        try:
            obj = json.loads(last)
            if "_summary" in obj:
                return obj["_summary"]
        except json.JSONDecodeError:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="../../experiments/results/grid_main")
    ap.add_argument("--op", default="add", choices=["add", "sub", "mul", "s5"])
    ap.add_argument("--optimizers", nargs="+", default=["adamw", "muon", "sgdm"])
    ap.add_argument("--wds", nargs="+", type=float, default=[0.01, 1.0])
    ap.add_argument("--init_scales", nargs="+", type=float, default=[1.0, 3.0])
    ap.add_argument("--train_fracs", nargs="+", type=float, default=[0.4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval_every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    grid = list(itertools.product(
        args.optimizers, args.wds, args.init_scales, args.train_fracs, args.seeds))
    print(f"[grid] {len(grid)} runs -> {args.out_dir}")

    for i, (opt, wd, sc, tf, seed) in enumerate(grid):
        name = f"{opt}_wd{wd}_sc{sc}_tf{tf}_s{seed}"
        path = os.path.join(args.out_dir, name + ".jsonl")
        done = already_done(path)
        if done is not None:
            print(f"[{i+1}/{len(grid)}] skip {name} (done)")
            summaries.append(done)
            continue
        cfg = Config(optimizer=opt, weight_decay=wd, init_scale=sc, train_frac=tf,
                     seed=seed, steps=args.steps, eval_every=args.eval_every,
                     op=args.op)
        t0 = time.time()
        summ, _ = run(cfg, out_path=path)
        dt = time.time() - t0
        summaries.append(summ)
        print(f"[{i+1}/{len(grid)}] {name}: memorize={summ['memorize_step']} "
              f"grok={summ['grok_step']} delay={summ['delay_ratio']} "
              f"test={summ['final_test_acc']:.3f} ({dt:.0f}s)", flush=True)

    # aggregate
    with open(os.path.join(args.out_dir, "summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    if summaries:
        keys = list(summaries[0].keys())
        with open(os.path.join(args.out_dir, "summaries.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in summaries:
                w.writerow(s)
    print(f"[grid] done. wrote summaries.json / summaries.csv")


if __name__ == "__main__":
    main()

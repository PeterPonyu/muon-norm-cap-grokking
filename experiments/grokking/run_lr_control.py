"""LR-confound control: is the Muon delay-collapse just a higher effective LR?

For each optimizer family we sweep its hidden-matrix learning rate over a wide
range (at the strong-grokking init_scale) and measure the grok delay. If Muon
shows ~no delay across its whole LR range while AdamW and SGD-momentum show a
large delay across theirs, the effect is attributable to orthogonalization, not
to learning rate.

Writes one jsonl per run + summaries.{json,csv} into --out_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

from train import Config, run
from run_grid import already_done


# (optimizer, lr_field, lr_value). For adamw the swept lr is the global `lr`;
# for muon/sgdm it is `muon_lr` (hidden-matrix lr), with the AdamW side fixed.
SWEEPS = {
    "adamw": ("lr", [3e-4, 1e-3, 3e-3, 1e-2]),
    "muon": ("muon_lr", [0.005, 0.01, 0.02, 0.04]),
    "sgdm": ("muon_lr", [0.005, 0.01, 0.02, 0.04]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="../../experiments/results/lr_control")
    ap.add_argument("--init_scale", type=float, default=3.0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--eval_every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = []
    for opt, (field, lrs) in SWEEPS.items():
        for lr in lrs:
            for seed in args.seeds:
                runs.append((opt, field, lr, seed))
    print(f"[lr_control] {len(runs)} runs -> {args.out_dir}")

    summaries = []
    for i, (opt, field, lr, seed) in enumerate(runs):
        name = f"{opt}_{field}{lr}_s{seed}"
        path = os.path.join(args.out_dir, name + ".jsonl")
        done = already_done(path)
        if done is not None:
            print(f"[{i+1}/{len(runs)}] skip {name} (done)")
            summaries.append(done)
            continue
        kw = dict(optimizer=opt, weight_decay=args.weight_decay,
                  init_scale=args.init_scale, seed=seed,
                  steps=args.steps, eval_every=args.eval_every)
        kw[field] = lr
        cfg = Config(**kw)
        t0 = time.time()
        summ, _ = run(cfg, out_path=path)
        dt = time.time() - t0
        summ["swept_lr"] = lr  # the lr we varied, for plotting
        summaries.append(summ)
        print(f"[{i+1}/{len(runs)}] {name}: memorize={summ['memorize_step']} "
              f"grok={summ['grok_step']} delay={summ['delay_ratio']} "
              f"test={summ['final_test_acc']:.3f} ({dt:.0f}s)", flush=True)

    with open(os.path.join(args.out_dir, "summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    if summaries:
        keys = sorted({k for s in summaries for k in s.keys()})
        with open(os.path.join(args.out_dir, "summaries.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in summaries:
                w.writerow({k: s.get(k) for k in keys})
    print("[lr_control] done")


if __name__ == "__main__":
    main()

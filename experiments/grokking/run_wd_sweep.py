"""Weight-decay sweep to test the Norm-Separation Delay Law (arXiv:2603.13331).

Law: T_grok - T_mem = Theta(gamma_eff^-1 * log(||theta_mem||^2 / ||theta_post||^2)),
with gamma_eff = eta*lambda for SGD. So at fixed eta the realized delay should scale
~ 1/lambda for SGD-momentum (and similarly for AdamW), and -> infinity as lambda->0.

If Muon's delay is instead ~flat in lambda (and it still groks at lambda=0), its
contraction is geometric (orthogonalization), not weight-decay driven.

Run at init_scale=1 (where all three optimizers can reach grokking, giving finite
T_grok for the fit). Fixed lr per family; sweep weight_decay; multiple seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

from train import Config, run
from run_grid import already_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="../../experiments/results/wd_sweep")
    ap.add_argument("--init_scale", type=float, default=1.0)
    ap.add_argument("--optimizers", nargs="+", default=["adamw", "muon", "sgdm"])
    ap.add_argument("--wds", nargs="+", type=float,
                    default=[0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = [(o, wd, s) for o in args.optimizers for wd in args.wds for s in args.seeds]
    print(f"[wd_sweep] {len(runs)} runs -> {args.out_dir}", flush=True)

    summaries = []
    for i, (opt, wd, seed) in enumerate(runs):
        name = f"{opt}_wd{wd}_s{seed}"
        path = os.path.join(args.out_dir, name + ".jsonl")
        done = already_done(path)
        if done is not None:
            summaries.append(done)
            continue
        cfg = Config(optimizer=opt, weight_decay=wd, init_scale=args.init_scale,
                     seed=seed, steps=args.steps, eval_every=args.eval_every)
        t0 = time.time()
        summ, _ = run(cfg, out_path=path)
        dt = time.time() - t0
        summaries.append(summ)
        print(f"[{i+1}/{len(runs)}] {name}: mem={summ['memorize_step']} "
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
    print("[wd_sweep] done", flush=True)


if __name__ == "__main__":
    main()

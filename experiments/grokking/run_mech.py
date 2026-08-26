"""Mechanism runs: directional/spectral probes on the route-discriminating cells.

Cells chosen to contrast the two routes (3 seeds each, eval_every=25):
  - muon  : (sc1, wd0) (sc1, wd1) (sc3, wd0.01) (sc3, wd1)   — growth route
  - adamw : (sc1, wd0) (sc1, wd1) (sc3, wd0.01) (sc3, wd1)   — plateau & contraction
  - sgdm  : (sc1, wd0) (sc3, wd0.01)                          — contraction w/o NS
"""
from __future__ import annotations

import json
import os
import time

from train import Config, run
from run_grid import already_done

CELLS = [
    ("muon", 1.0, 0.0), ("muon", 1.0, 1.0), ("muon", 3.0, 0.01), ("muon", 3.0, 1.0),
    ("adamw", 1.0, 0.0), ("adamw", 1.0, 1.0), ("adamw", 3.0, 0.01), ("adamw", 3.0, 1.0),
    ("sgdm", 1.0, 0.0), ("sgdm", 3.0, 0.01),
]
SEEDS = [0, 1, 2]
OUT = "../../experiments/results/mech"


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = [(o, sc, wd, s) for (o, sc, wd) in CELLS for s in SEEDS]
    print(f"[mech] {len(runs)} runs -> {OUT}", flush=True)
    for i, (opt, sc, wd, seed) in enumerate(runs):
        name = f"{opt}_sc{sc}_wd{wd}_s{seed}"
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(runs)}] skip {name}", flush=True)
            continue
        steps = 20000 if sc == 1.0 else 12000
        cfg = Config(optimizer=opt, weight_decay=wd, init_scale=sc, seed=seed,
                     steps=steps, eval_every=25, mech=True)
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        print(f"[{i+1}/{len(runs)}] {name}: mem={s['memorize_step']} "
              f"grok={s['grok_step']} test={s['final_test_acc']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[mech] DONE", flush=True)


if __name__ == "__main__":
    main()

"""Direction 012 — Muon spawn-ladder EXTENSION arm.

The headline grid (run_lmc.py, spawn ≤ 8000, child_end 12000) right-censors
Muon's k*: every Muon seed still shows a high inter-child barrier (2.3-4.4) at
the last spawn point, while AdamW locks (barrier -> ~0, k* ≈ 8000) and SGDM is
trivially connected throughout. This arm pins (or bounds) Muon's lock time by
extending the spawn ladder well past 8000.

Muon-only, 3 seeds, child_noise="perturb" (the headline protocol). Spawn ladder
[8000, 16000, 24000, 32000]; child_end_step 40000 so EVERY spawn point still
has substantial post-fork training (the smallest, spawn 32000, trains 8000
more — matching the headline's spawn-8000/child-end-12000 = 4000-step minimum
in spirit, here widened). Parent trains to 32000.

Output: ../../experiments/results/lmc_instability_ext/<name>.jsonl  (own slug,
isolated from the headline namespace per the repo convention). Resume-safe.
"""
from __future__ import annotations

import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_EXP = os.path.dirname(_THIS_DIR)
if _EXP not in sys.path:
    sys.path.append(_EXP)

from train_fork import Config, run  # noqa: E402
from run_lmc import already_done  # noqa: E402

EXT_SPAWN_STEPS = [8000, 16000, 24000, 32000]
CHILD_END_STEP = 40000
SEEDS = [0, 1, 2]

OUT = os.path.join(_THIS_DIR, "..", "..", "experiments", "results",
                   "lmc_instability_ext")


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"[lmc_instability_ext] muon x {len(SEEDS)} seeds -> {OUT} | "
          f"spawn={EXT_SPAWN_STEPS} child_end={CHILD_END_STEP}", flush=True)
    for seed in SEEDS:
        name = f"muon_ext_s{seed}"
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"skip {name}", flush=True)
            continue
        cfg = Config(optimizer="muon", seed=seed,
                     spawn_steps=list(EXT_SPAWN_STEPS),
                     child_end_step=CHILD_END_STEP,
                     child_noise="perturb")
        t0 = time.time()
        s, _ = run(cfg, out_path=path)
        bs = s["barrier_by_spawn"]
        print(f"{name}: k_star={s['k_star']} "
              f"barriers={ {k: round(v, 3) for k, v in bs.items()} } "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[lmc_instability_ext] DONE", flush=True)


if __name__ == "__main__":
    main()

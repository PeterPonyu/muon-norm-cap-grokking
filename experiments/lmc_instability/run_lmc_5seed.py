"""Direction 012 — 5-seed hardening (red-team audit 2026-06-13).

The headline grid ran 3 seeds; the audit flagged Muon's k* as right-censored at
n=3 (the "≥4× later than AdamW" claim needs more seeds to bound k* and the
seed-variance). This adds seeds {3,4} for all 3 optimizers (6 new parent cells)
to the SAME results/lmc_instability/ namespace (names opt_s3/opt_s4), resume-safe.
Headline config reused verbatim (default SPAWN_STEPS, N_CHILDREN).
"""
from __future__ import annotations

import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_THIS)
for p in (_THIS, _EXP):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_fork import Config, run  # noqa: E402
from run_lmc import (OUT, SPAWN_STEPS, N_CHILDREN, OPTIMIZERS,  # noqa: E402
                     _cell_name, already_done)

EXTRA_SEEDS = [3, 4]


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = [(opt, s) for opt in OPTIMIZERS for s in EXTRA_SEEDS]
    print(f"[lmc_5seed] {len(cells)} new cells (seeds {EXTRA_SEEDS}) -> {OUT}",
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
        print(f"[{i+1}/{len(cells)}] {name}: k_star={s['k_star']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[lmc_5seed] DONE", flush=True)


if __name__ == "__main__":
    main()

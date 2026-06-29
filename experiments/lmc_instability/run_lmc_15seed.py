"""Direction 012 / A §3.3 — densify LMC seeds 5..14 (n=5 -> n=15).

paper-A §3.3 (linear-mode-connectivity / basin-lock: Muon's k* ~4x later than AdamW
& seed-variable) rests on n=5. The red-team flagged the k* bound + seed-variance as
thin. This adds seeds 5..14 (10 new parent cells per optimizer) into the SAME
results/lmc_instability/ namespace (names opt_s{seed}), reusing the headline config
verbatim (default SPAWN_STEPS, N_CHILDREN), resume-safe. Mirrors run_lmc_5seed.py.
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

def main():
    import argparse
    ap = argparse.ArgumentParser()
    # TRIMMED default (2026-06-15): each LMC fork cell costs ~25-44 min, so the full
    # 3-opt x 10-seed sweep would be ~15 h for an OPTIONAL §3.3 strengthening. SGDM is
    # excluded (findings-012: never learns -> connectivity vacuous), and +5 seeds for
    # {muon,adamw} (n=5 -> n=10) already powers the distributional "basin stays open
    # longer" claim (e.g. muon 0/10 vs adamw ~4/10 -> Fisher p~0.04).
    ap.add_argument("--optimizers", default="muon,adamw")
    ap.add_argument("--seeds", default="5-9", help="extra seeds, e.g. 5-9 or 5,6,7")
    args = ap.parse_args()
    opts = [o.strip() for o in args.optimizers.split(",") if o.strip()]
    if "-" in args.seeds:
        a, b = args.seeds.split("-"); extra = list(range(int(a), int(b) + 1))
    else:
        extra = [int(x) for x in args.seeds.split(",") if x]

    os.makedirs(OUT, exist_ok=True)
    cells = [(opt, s) for opt in opts for s in extra]
    print(f"[lmc_15seed] {len(cells)} new cells (opts={opts} seeds={extra}) "
          f"-> {OUT}", flush=True)
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
    print("[lmc_15seed] DONE", flush=True)


if __name__ == "__main__":
    main()

"""Aggregate the LMC basin-lock metric k* across seeds (012 / paper-A §3.3).

k* = earliest spawn step after which the two forked children stay linearly connected
(loss barrier < threshold). The §3.3 claim: Muon basin-locks LATER than AdamW and is
seed-variable. At n=3 Muon's k* was right-censored; this re-aggregates over the
densified n=15 (run_lmc_15seed.py) and reports per-optimizer median k*, spread, the
muon/adamw ratio, and the censoring rate (k*=None ⇒ not connected by the last spawn
step, 8000). SGDM is reported but flagged: if it never learns, its k* is vacuous.

Reads results/lmc_instability/*.jsonl (summaries). Writes
results/figures-012/lmc_kstar_verdict.json. Read-only on data.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "experiments", "results")
DIR = os.path.join(RES, "lmc_instability")
FIG = os.path.join(RES, "figures-012")
CENSOR = 8000   # last spawn step; k*=None is right-censored beyond this


def last_summary(path):
    summ = None
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            s = o.get("_summary") if isinstance(o.get("_summary"), dict) else (
                o if "_summary" in o else None)
            if s:
                summ = s
    return summ


def main():
    by_opt = defaultdict(list)   # opt -> [(seed, k_star)]
    for p in sorted(glob.glob(os.path.join(DIR, "*.jsonl"))):
        s = last_summary(p)
        if not s or "k_star" not in s:
            continue
        by_opt[s["optimizer"]].append((s.get("seed"), s["k_star"]))

    out = {"censor_spawn_step": CENSOR, "by_optimizer": {}}
    print(f"{'opt':6s}  n  n_censored  median_k*  k*_values")
    for opt in ("adamw", "muon", "sgdm"):
        rows = sorted(by_opt.get(opt, []))
        if not rows:
            continue
        ks = [k for _, k in rows]
        present = [k for k in ks if k is not None]
        # for the median, treat censored (None) as CENSOR (lower bound on k*)
        imputed = [k if k is not None else CENSOR for k in ks]
        rec = {
            "n": len(ks),
            "n_censored": sum(1 for k in ks if k is None),
            # the DISTRIBUTIONAL readout (paper-A §3.3: "distributional, NOT 4x"):
            # fraction whose basin LOCKS by the last spawn step. Lower => basin stays
            # open longer. Most base-grid k* are censored at 8000, so this rate (not an
            # imputed median) is the clean comparison.
            "locked_by_8000_rate": float(len(present) / len(ks)) if ks else None,
            "median_k_star_imputed": float(np.median(imputed)) if imputed else None,
            "median_k_star_uncensored_only": float(np.median(present)) if present else None,
            "min": int(min(present)) if present else None,
            "max": int(max(present)) if present else None,
            "k_star_per_seed": {str(sd): k for sd, k in rows},
        }
        out["by_optimizer"][opt] = rec
        print(f"{opt:6s} {rec['n']:2d}  {rec['n_censored']:^10d}  "
              f"lock_rate={rec['locked_by_8000_rate']:.2f}  {ks}")

    # distributional muon-vs-adamw: does Muon lock by 8000 LESS often (basin open longer)?
    aw_r = out["by_optimizer"].get("adamw")
    mu_r = out["by_optimizer"].get("muon")
    if aw_r and mu_r:
        aw_lock = aw_r["n"] - aw_r["n_censored"]
        mu_lock = mu_r["n"] - mu_r["n_censored"]
        p = None
        try:
            from scipy.stats import fisher_exact
            # 2x2: rows [adamw, muon], cols [locked, censored]; alt: adamw locks more
            res = fisher_exact([[aw_lock, aw_r["n_censored"]],
                                [mu_lock, mu_r["n_censored"]]], alternative="greater")
            p = float(res[1])
        except Exception:
            p = None
        out["distributional_test"] = {
            "adamw_locked": aw_lock, "adamw_n": aw_r["n"],
            "muon_locked": mu_lock, "muon_n": mu_r["n"],
            "fisher_p_adamw_locks_more": (float(p) if p is not None else None),
            "claim": "basin stays open longer under Muon (lower lock-by-8000 rate)",
        }
        print(f"\nLock-by-8000:  adamw {aw_lock}/{aw_r['n']}  muon {mu_lock}/{mu_r['n']}  "
              f"Fisher p(adamw locks more)={p if p is None else round(p,4)} "
              f"-> {'Muon basin stays open longer' if mu_lock < aw_lock else 'no distributional gap'}")

    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, "lmc_kstar_verdict.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(FIG, 'lmc_kstar_verdict.json')}")


if __name__ == "__main__":
    main()

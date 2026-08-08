#!/usr/bin/env python3
"""A-T0-2 (revision plan 2026-07-16): refit the Muon equilibrium-hidden-norm vs
weight-decay exponent from the RAW normlaw jsonls and adjudicate the paper's
Prop-1 prediction (||W*|| prop 1/lambda, slope -1) against the rotational-
equilibrium prediction (||W*|| prop (eta/lambda)^{1/2}, slope -1/2 at fixed eta;
Kosson et al. 2024, already cited as kosson2023rotational).

Data: results/ieee_gap_20260705/A/normlaw/{muon,adamw}_lam*_s*.jsonl
      (S5, 6 lambdas x 5 seeds x 2 optimizers, steps=10000, eval_every=200).
Statistic: OLS of log10(final_wn_hidden) on log10(lambda), per-run points;
slope SE, R2; t-tests of slope against -1 and -0.5; sensitivity fits dropping
the saturating endpoints (lambda=0.001 where the equilibrium is not yet reached
within the 10k-step budget, and lambda=0.3 where the norm floors at ~57).
Also a late-window median (last 25% of eval rows) as an equilibrium-robustness
check against using the final row only.

CPU-only. Output: refit_normlaw.json here.
"""
from __future__ import annotations
from pathlib import Path
import glob, json, math, os, re
import numpy as np

RES = str(Path(__file__).resolve().parents[3] / 'experiments' / 'results' / 'ieee_gap_20260705' / 'A' / 'normlaw')
OUT = os.path.dirname(os.path.abspath(__file__))
LAM = {"0p001": 0.001, "0p003": 0.003, "0p01": 0.01, "0p03": 0.03, "0p1": 0.1, "0p3": 0.3}


def load_run(path):
    rows = []
    with open(path) as f:
        for l in f:
            if l.strip():
                rows.append(json.loads(l))
    meta = rows[0].get("_meta", {})
    summ = rows[-1].get("_summary", {})
    trail = [r for r in rows[1:] if "wn_hidden" in r]
    lastq = trail[int(0.75 * len(trail)):]
    return {
        "final_wn_hidden": summ.get("final_wn_hidden"),
        "late_wn_hidden_med": float(np.median([r["wn_hidden"] for r in lastq])),
        "eta_muon": meta.get("muon_lr"), "eta_adamw": meta.get("lr"),
    }


def ols_loglog(pairs):
    x = np.log10([p[0] for p in pairs])
    y = np.log10([p[1] for p in pairs])
    n = len(x)
    sx, sy = x - x.mean(), y - y.mean()
    slope = float((sx * sy).sum() / (sx * sx).sum())
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)
    se = float(math.sqrt((resid ** 2).sum() / (n - 2) / (sx * sx).sum()))
    r2 = float(1 - (resid ** 2).sum() / (sy * sy).sum())
    return {"n": n, "slope": slope, "se": se, "intercept": intercept, "r2": r2}


def ttests(fit):
    from scipy import stats
    out = {}
    for name, target in [("vs_minus1_prop1", -1.0), ("vs_minus0p5_roteq", -0.5)]:
        t = (fit["slope"] - target) / fit["se"]
        p = 2 * stats.t.sf(abs(t), fit["n"] - 2)
        out[name] = {"target": target, "t": float(t), "p": float(p),
                     "abs_sigma_away": float(abs(t))}
    return out


def main():
    data = {"muon": {}, "adamw": {}}
    for path in sorted(glob.glob(os.path.join(RES, "*.jsonl"))):
        m = re.match(r"(muon|adamw)_lam(\w+)_s(\d+)\.jsonl", os.path.basename(path))
        if not m:
            continue
        opt, lam, seed = m.group(1), LAM[m.group(2)], int(m.group(3))
        data[opt].setdefault(lam, {})[seed] = load_run(path)

    results = {}
    for opt in ("muon", "adamw"):
        pairs_final, pairs_late = [], []
        per_seed = {}
        for lam in sorted(data[opt]):
            per_seed[str(lam)] = {str(s): data[opt][lam][s]["final_wn_hidden"]
                                  for s in sorted(data[opt][lam])}
            for s, r in data[opt][lam].items():
                pairs_final.append((lam, r["final_wn_hidden"]))
                pairs_late.append((lam, r["late_wn_hidden_med"]))
        fits = {"full_final": ols_loglog(pairs_final),
                "full_latewindow": ols_loglog(pairs_late)}
        # sensitivity: drop saturating endpoints
        fits["drop_lam0p001"] = ols_loglog([p for p in pairs_final if p[0] > 0.001])
        fits["drop_lam0p3"] = ols_loglog([p for p in pairs_final if p[0] < 0.3])
        fits["interior_0p003_to_0p1"] = ols_loglog(
            [p for p in pairs_final if 0.001 < p[0] < 0.3])
        for k in fits:
            fits[k]["tests"] = ttests(fits[k])
        results[opt] = {"per_seed_final_wn_by_lambda": per_seed, "fits": fits}

    results["eta"] = {"muon_hidden_lr": 0.02, "adamw_lr": 0.001}
    results["published"] = {"muon_slope": -0.43, "muon_se": 0.03, "muon_r2": 0.87}
    with open(os.path.join(OUT, "refit_normlaw.json"), "w") as f:
        json.dump(results, f, indent=1)

    for opt in ("muon", "adamw"):
        print(f"== {opt}")
        for k, fit in results[opt]["fits"].items():
            t = fit["tests"]
            print(f"  {k:24s} slope {fit['slope']:+.3f} ± {fit['se']:.3f}  R²={fit['r2']:.3f}  "
                  f"| vs -1: {t['vs_minus1_prop1']['abs_sigma_away']:.1f}σ (p={t['vs_minus1_prop1']['p']:.2g})"
                  f"  vs -0.5: {t['vs_minus0p5_roteq']['abs_sigma_away']:.1f}σ (p={t['vs_minus0p5_roteq']['p']:.2g})")


if __name__ == "__main__":
    main()

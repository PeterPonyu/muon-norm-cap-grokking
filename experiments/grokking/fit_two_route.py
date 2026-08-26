"""Two-route model fit for AdamW's delay-vs-λ curve (resolves the slope anomaly).

Single-power-law reading of the delay law predicts delay ∝ λ^{-1}; we measured a
pooled slope of only −0.34 with a plateau at small λ. Hypothesis: grokking can
proceed via two parallel routes, and the realized delay is set by the faster one:

    T(λ) = 1 / ( 1/D0  +  λ/C )        ("parallel resistors")
      - D0   : λ-independent growth/directional route time
      - C/λ  : contraction route time, with the law's exponent FIXED at −1

Model comparison on the wd_sweep AdamW data (8 λ × 3 seeds, λ>0 plus λ=0 which
only the two-route model can even represent):
  M1 single power law  T = A·λ^{-α}          (2 params, cannot fit λ=0)
  M2 two-route          T = 1/(1/D0 + λ/C)    (2 params, includes λ=0 naturally)
  M3 two-route free exp T = 1/(1/D0 + λ^β/C)  (3 params; check β ≈ 1)

Outputs fit params, R² (on log-delay), AIC, and a figure.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "../../experiments/results/wd_sweep"
FIG_DIR = "../../experiments/results/figures"


def load_delays():
    pts = []
    for p in sorted(glob.glob(os.path.join(DIR, "adamw_*.jsonl"))):
        rows = [json.loads(l) for l in open(p)]
        s = next((r["_summary"] for r in rows if "_summary" in r), None)
        if s and s["grok_step"] is not None and s["memorize_step"]:
            pts.append((s["weight_decay"], s["grok_step"] - s["memorize_step"]))
    return np.array([x for x, _ in pts]), np.array([y for _, y in pts], dtype=float)


def r2_log(y, yhat):
    ly, lyh = np.log(y), np.log(yhat)
    return 1 - np.sum((ly - lyh) ** 2) / np.sum((ly - np.mean(ly)) ** 2)


def aic(y, yhat, k):
    n = len(y)
    rss = np.sum((np.log(y) - np.log(yhat)) ** 2)
    return n * np.log(rss / n) + 2 * k


def main():
    lam, T = load_delays()
    print(f"n={len(T)} AdamW runs; λ values: {sorted(set(lam))}")

    pos = lam > 0
    results = {}

    # M1: single power law (λ>0 only — cannot represent λ=0)
    def m1(l, A, a):
        return A * l ** (-a)
    p1, _ = curve_fit(m1, lam[pos], T[pos], p0=[100, 0.5], maxfev=20000)
    yh1 = m1(lam[pos], *p1)
    results["M1_power_law"] = {
        "params": {"A": p1[0], "alpha": p1[1]},
        "r2_log": r2_log(T[pos], yh1), "aic": aic(T[pos], yh1, 2),
        "n_fit": int(pos.sum()), "note": "cannot include lambda=0 points",
    }

    # M2: two-route, contraction exponent fixed at the law's -1 (all points)
    def m2(l, D0, C):
        return 1.0 / (1.0 / D0 + l / C)
    p2, _ = curve_fit(m2, lam, T, p0=[5000, 500], maxfev=20000)
    yh2 = m2(lam, *p2)
    results["M2_two_route_beta1"] = {
        "params": {"D0": p2[0], "C": p2[1]},
        "r2_log": r2_log(T, yh2), "aic": aic(T, yh2, 2), "n_fit": len(T),
    }

    # M3: two-route with free contraction exponent β (does β recover ≈1?)
    def m3(l, D0, C, b):
        return 1.0 / (1.0 / D0 + np.power(l, b) / C)
    p3, _ = curve_fit(m3, lam, T, p0=[5000, 500, 1.0], maxfev=50000)
    yh3 = m3(lam, *p3)
    results["M3_two_route_free_beta"] = {
        "params": {"D0": p3[0], "C": p3[1], "beta": p3[2]},
        "r2_log": r2_log(T, yh3), "aic": aic(T, yh3, 3), "n_fit": len(T),
    }

    for k, v in results.items():
        ps = ", ".join(f"{n}={x:.3g}" for n, x in v["params"].items())
        print(f"{k:24s} {ps:42s} R2(log)={v['r2_log']:.3f} AIC={v['aic']:.1f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "two_route_fit.json"), "w") as f:
        json.dump(results, f, indent=2)

    # figure
    import sys as _sys
    import shutil as _shutil
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    lam_plot = np.where(lam == 0, 2e-4, lam)  # show λ=0 at left edge
    ax.scatter(lam_plot, T, color=figstyle.CB["blue"], alpha=0.6,
               label="AdamW runs ($\\lambda$=0 at edge)")
    lg = np.logspace(np.log10(2e-4), 0, 200)
    ax.plot(lg, m1(lg, *p1), ":", color="0.3", label=f"M1 power law ($\\alpha$={p1[1]:.2f})")
    ax.plot(lg, m2(lg, *p2), "-", color=figstyle.CB["vermillion"],
            label=f"M2 two-route ($D_0$={p2[0]:.0f}, $C$={p2[1]:.0f})")
    ax.plot(lg, m3(lg, *p3), "--", color=figstyle.CB["green"],
            label=f"M3 free exp ($\\beta$={p3[2]:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("weight decay $\\lambda$")
    ax.set_ylabel(r"realized delay $T_{\mathrm{grok}}-T_{\mathrm{mem}}$ (steps)")
    # title removed: caption carries it
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fn = os.path.join(FIG_DIR, "fig_two_route_fit.png")
    fig.savefig(fn)
    _shutil.copy(fn, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "papers", "figs", "A_two_route.png"))
    print("wrote", fn, "+ copied to papers/figs/A_two_route.png")


if __name__ == "__main__":
    main()

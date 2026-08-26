"""Test the Norm-Separation Delay Law against Muon (the novel analysis).

Law (Truong et al. 2026, arXiv:2603.13331):
    T_grok - T_mem = Theta( gamma_eff^-1 * log(||theta_mem||^2 / ||theta_post||^2) )
with gamma_eff = eta*lambda (SGD).  We extract per run:
    T_mem, T_grok, ||theta||@mem, ||theta||@grok
and produce:
  1. fig_delay_vs_wd.png   : realized delay vs weight decay (log-log), per optimizer
                             -> law predicts slope ~ -1; flat = lambda-independent.
  2. fig_law_fit.png       : realized delay vs gamma_eff^-1 * log-norm-ratio
                             -> law-obeying families collapse to a line; outliers don't.
  3. wd0_table.json        : at wd=0, which optimizers grok at all (the discriminator).
Reads <dir>/*.jsonl (default wd_sweep).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"adamw": "tab:blue", "muon": "tab:red", "sgdm": "tab:green"}


def load(path):
    meta = summ = None
    hist = []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            if "_meta" in o:
                meta = o["_meta"]
            elif "_summary" in o:
                summ = o["_summary"]
            else:
                hist.append(o)
    return meta, summ, hist


def norm_at(hist, step, key="wn_total"):
    """Weight norm at the eval record closest to `step`."""
    if not hist:
        return None
    best = min(hist, key=lambda h: abs(h["step"] - step))
    return best[key]


def eta_of(meta):
    return meta["lr"] if meta["optimizer"] == "adamw" else meta["muon_lr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../../experiments/results/grid_main")
    ap.add_argument("--fig_dir", default="../../experiments/results/figures")
    ap.add_argument("--init_scale", type=float, default=1.0,
                    help="restrict the law fit to one init scale (where runs grok)")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    rows = []
    for p in sorted(glob.glob(os.path.join(args.dir, "*.jsonl"))):
        meta, summ, hist = load(p)
        if not (meta and summ):
            continue
        if args.init_scale is not None and meta.get("init_scale", 1.0) != args.init_scale:
            continue
        T_mem = summ["memorize_step"]
        T_grok = summ["grok_step"]
        grokked = T_grok is not None
        theta_mem = norm_at(hist, T_mem) if T_mem is not None else None
        theta_post = norm_at(hist, T_grok) if grokked else norm_at(hist, summ["stopped_step"])
        log_ratio = (math.log((theta_mem**2) / (theta_post**2))
                     if theta_mem and theta_post and theta_post > 0 else None)
        rows.append({
            "opt": meta["optimizer"], "wd": meta["weight_decay"], "eta": eta_of(meta),
            "seed": meta["seed"], "grokked": grokked,
            "T_mem": T_mem, "T_grok": T_grok,
            "delay": (T_grok - T_mem) if grokked and T_mem is not None else None,
            "theta_mem": theta_mem, "theta_post": theta_post, "log_ratio": log_ratio,
        })

    # ---- 1) delay vs weight decay (log-log) ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["grokked"] and r["delay"] is not None and r["wd"] > 0:
            by[r["opt"]][r["wd"]].append(r["delay"])
    for opt in sorted(by):
        wds = sorted(by[opt])
        means = [np.mean(by[opt][w]) for w in wds]
        errs = [np.std(by[opt][w]) for w in wds]
        ax.errorbar(wds, means, yerr=errs, fmt="-o", color=COLORS.get(opt, "gray"),
                    capsize=4, label=opt)
    # reference slope -1 (law prediction for gamma_eff = eta*lambda)
    if by:
        wref = np.array([1e-3, 1e0])
        anchor = max((np.mean(v[min(v)]) for v in by.values()), default=1e3)
        ax.plot(wref, anchor * (wref[0] / wref) ** 1.0 * 0.3, "k--", alpha=0.5,
                label="slope -1 (law)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("weight decay λ"); ax.set_ylabel("realized delay  T_grok - T_mem (steps)")
    ax.set_title("Delay vs weight decay: law predicts ∝ 1/λ; flat ⇒ λ-independent")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fn = os.path.join(args.fig_dir, "fig_delay_vs_wd.png")
    fig.savefig(fn, dpi=130); plt.close(fig); print("wrote", fn)

    # ---- 2) law-collapse scatter: delay vs (1/gamma_eff) * log_ratio ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for opt in sorted({r["opt"] for r in rows}):
        xs, ys = [], []
        for r in rows:
            if (r["opt"] == opt and r["grokked"] and r["delay"] is not None
                    and r["log_ratio"] is not None and r["wd"] > 0):
                gamma_eff = r["eta"] * r["wd"]
                if gamma_eff > 0 and r["log_ratio"] > 0:
                    xs.append(r["log_ratio"] / gamma_eff)
                    ys.append(r["delay"])
        if xs:
            ax.scatter(xs, ys, color=COLORS.get(opt, "gray"), label=opt, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("γ_eff⁻¹ · log(‖θ_mem‖²/‖θ_post‖²)   (law RHS, η·λ)")
    ax.set_ylabel("realized delay (steps)")
    ax.set_title("Law collapse: obeying families fall on a line; outliers don't")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fn = os.path.join(args.fig_dir, "fig_law_fit.png")
    fig.savefig(fn, dpi=130); plt.close(fig); print("wrote", fn)

    # ---- 3) wd=0 discriminator table ----
    wd0 = defaultdict(lambda: {"n": 0, "grokked": 0, "grok_steps": []})
    for r in rows:
        if r["wd"] == 0.0:
            d = wd0[r["opt"]]
            d["n"] += 1
            if r["grokked"]:
                d["grokked"] += 1
                d["grok_steps"].append(r["T_grok"])
    table = {opt: {"n_seeds": d["n"], "n_grokked": d["grokked"],
                   "mean_grok_step": (float(np.mean(d["grok_steps"]))
                                      if d["grok_steps"] else None)}
             for opt, d in wd0.items()}
    with open(os.path.join(args.fig_dir, "wd0_table.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("wd=0 discriminator:", json.dumps(table))


if __name__ == "__main__":
    main()

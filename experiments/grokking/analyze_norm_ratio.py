"""Mechanism analysis: does grokking coincide with weight-norm contraction?

The delay law's driving quantity is log(||theta_mem||^2 / ||theta_post||^2) — the
norm must SHRINK between memorization and grokking. We test this directly from the
logged histories: for each run, take wn_hidden (the matrices Muon touches) at
T_mem and at T_grok and form the contraction ratio.

Outputs:
  - norm_ratio_table.json : per (opt, wd, sc): mean±std of wn(T_grok)/wn(T_mem),
    plus wn at init and the law log-ratio.
  - fig_norm_trajectories.png : wn_hidden vs step for key cells, with T_mem/T_grok
    marked — the visual mechanism story.
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


def at(hist, step, key):
    return min(hist, key=lambda h: abs(h["step"] - step))[key] if hist else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../../experiments/results/grid_main")
    ap.add_argument("--fig_dir", default="../../experiments/results/figures")
    ap.add_argument("--norm_key", default="wn_hidden")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    K = args.norm_key

    cells = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.dir, "*.jsonl"))):
        meta, summ, hist = load(p)
        if not (meta and summ and hist):
            continue
        cells[(meta["optimizer"], meta["weight_decay"],
               meta.get("init_scale", 1.0))].append((meta, summ, hist))

    table = {}
    for (opt, wd, sc), runs in sorted(cells.items()):
        ratios, logr, init_norms, mem_norms = [], [], [], []
        n_grok = 0
        for meta, summ, hist in runs:
            T_mem, T_grok = summ["memorize_step"], summ["grok_step"]
            init_norms.append(hist[0][K])
            if T_mem is None:
                continue
            w_mem = at(hist, T_mem, K)
            mem_norms.append(w_mem)
            if T_grok is None:
                continue
            n_grok += 1
            w_grok = at(hist, T_grok, K)
            if w_mem and w_grok:
                ratios.append(w_grok / w_mem)
                logr.append(math.log((w_mem ** 2) / (w_grok ** 2)))
        table[f"{opt}_wd{wd}_sc{sc}"] = {
            "optimizer": opt, "weight_decay": wd, "init_scale": sc,
            "n_runs": len(runs), "n_grokked": n_grok,
            "init_norm_mean": float(np.mean(init_norms)) if init_norms else None,
            "mem_norm_mean": float(np.mean(mem_norms)) if mem_norms else None,
            "contraction_ratio_mean": float(np.mean(ratios)) if ratios else None,
            "contraction_ratio_std": float(np.std(ratios)) if ratios else None,
            "law_log_ratio_mean": float(np.mean(logr)) if logr else None,
        }
    out = os.path.join(args.fig_dir, "norm_ratio_table.json")
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print("wrote", out)
    for k, v in table.items():
        if v["contraction_ratio_mean"] is not None:
            print(f"  {k:28s} wn(T_grok)/wn(T_mem) = {v['contraction_ratio_mean']:.3f} "
                  f"± {v['contraction_ratio_std']:.3f}  (grok {v['n_grokked']}/{v['n_runs']})")

    # ---- trajectory figure for key cells ----
    key_cells = [
        ("muon", 0.0, 1.0), ("adamw", 0.0, 1.0),
        ("muon", 0.01, 3.0), ("adamw", 1.0, 3.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, cell in zip(axes.flat, key_cells):
        runs = cells.get(cell, [])
        if not runs:
            ax.set_visible(False)
            continue
        opt, wd, sc = cell
        for meta, summ, hist in runs:
            steps = [h["step"] for h in hist]
            ax.plot(steps, [h[K] for h in hist], color=COLORS[opt], alpha=0.5)
            if summ["memorize_step"] is not None:
                ax.axvline(summ["memorize_step"], color="gray", ls=":", alpha=0.4)
            if summ["grok_step"] is not None:
                ax.axvline(summ["grok_step"], color="black", ls="--", alpha=0.4)
        ax.set_xscale("log")
        ax.set_title(f"{opt}  wd={wd}  sc={sc}  ({K})")
        ax.set_xlabel("step (log)")
        ax.set_ylabel(K)
        ax.grid(alpha=0.3)
    fig.suptitle("Hidden weight-norm trajectories (dotted=memorize, dashed=grok)")
    fig.tight_layout()
    fn = os.path.join(args.fig_dir, "fig_norm_trajectories.png")
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    print("wrote", fn)


if __name__ == "__main__":
    main()

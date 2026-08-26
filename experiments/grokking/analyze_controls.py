"""Figures for the control experiments.

1. init-scale sweep (from calib_init or any dir): grok step vs init_scale per
   optimizer, with non-grokking runs marked as censored at the step budget.
2. LR-confound control (from lr_control dir): effective delay vs swept lr per
   optimizer family — shows whether the Muon effect is just a higher LR.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"adamw": "tab:blue", "muon": "tab:red", "sgdm": "tab:green"}


def load_run(path):
    meta, summary = None, None
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if "_meta" in obj:
                meta = obj["_meta"]
            elif "_summary" in obj:
                summary = obj["_summary"]
    return meta, summary


def load_summaries(out_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        meta, summary = load_run(path)
        if meta and summary:
            out.append((meta, summary))
    return out


def fig_init_sweep(in_dir, fig_path):
    data = load_summaries(in_dir)
    by_opt = defaultdict(list)
    for meta, s in data:
        by_opt[meta["optimizer"]].append(
            (meta.get("init_scale", 1.0), s.get("grok_step"), s["stopped_step"],
             s["final_test_acc"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt, rows in sorted(by_opt.items()):
        rows.sort()
        xs = [r[0] for r in rows]
        ys = [r[1] if r[1] is not None else r[2] for r in rows]  # censor at stop
        grokked = [r[1] is not None for r in rows]
        ax.plot(xs, ys, "-o", color=COLORS.get(opt, "gray"), label=opt)
        for x, y, gk in zip(xs, ys, grokked):
            if not gk:  # mark censored (no grok within budget)
                ax.scatter([x], [y], marker="x", s=90, color=COLORS.get(opt, "gray"),
                           zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("init_scale (Omnigrok)")
    ax.set_ylabel("grok step (log); x = no grok within budget")
    ax.set_title("Grok step vs init scale (single seed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print("wrote", fig_path)


def fig_lr_control(in_dir, fig_path):
    data = load_summaries(in_dir)
    # group by (optimizer, swept_lr) -> aggregate effective delay over seeds
    by = defaultdict(list)
    for meta, s in data:
        lr = s.get("swept_lr")
        if lr is None:
            lr = meta["lr"] if meta["optimizer"] == "adamw" else meta["muon_lr"]
        mem = s.get("memorize_step") or 1
        grok = s.get("grok_step")
        eff = (grok if grok is not None else s["stopped_step"]) / mem
        by[(meta["optimizer"], lr)].append((eff, grok is not None))

    fig, ax = plt.subplots(figsize=(8, 5))
    opts = sorted({k[0] for k in by})
    for opt in opts:
        lrs = sorted({k[1] for k in by if k[0] == opt})
        means, errs, frac_grok = [], [], []
        for lr in lrs:
            vals = by[(opt, lr)]
            effs = [v[0] for v in vals]
            means.append(np.mean(effs))
            errs.append(np.std(effs))
            frac_grok.append(np.mean([v[1] for v in vals]))
        ax.errorbar(lrs, means, yerr=errs, fmt="-o", color=COLORS.get(opt, "gray"),
                    capsize=4, label=opt)
        for lr, m, fg in zip(lrs, means, frac_grok):
            if fg < 1.0:
                ax.annotate(f"{fg:.0%} grok", (lr, m), fontsize=7,
                            textcoords="offset points", xytext=(0, 6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("hidden-matrix learning rate (log)")
    ax.set_ylabel("effective delay grok/memorize (log; censored at stop)")
    ax.set_title("LR-confound control: delay vs LR per optimizer family")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print("wrote", fig_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_dir", default="../../experiments/results/calib_init")
    ap.add_argument("--lr_dir", default="../../experiments/results/lr_control")
    ap.add_argument("--fig_dir", default="../../experiments/results/figures")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    if os.path.isdir(args.init_dir) and glob.glob(os.path.join(args.init_dir, "*.jsonl")):
        fig_init_sweep(args.init_dir, os.path.join(args.fig_dir, "fig_init_sweep.png"))
    if os.path.isdir(args.lr_dir) and glob.glob(os.path.join(args.lr_dir, "*.jsonl")):
        fig_lr_control(args.lr_dir, os.path.join(args.fig_dir, "fig_lr_control.png"))


if __name__ == "__main__":
    main()

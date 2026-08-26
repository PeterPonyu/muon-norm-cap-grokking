"""Analyze grokking grid results: aggregate over seeds and make figures.

Reads <out_dir>/*.jsonl, produces:
  - fig_curves_<wd>.png : train/test acc vs step (adamw vs muon), seed-averaged
  - fig_weightnorm_<wd>.png : hidden weight-norm trajectory (adamw vs muon)
  - fig_delay_summary.png : delay_ratio bar chart across wd, by optimizer
  - stats.json : aggregated mean/std of grok_step, memorize_step, delay_ratio
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


def load_run(path):
    meta, summary, hist = None, None, []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            if "_meta" in obj:
                meta = obj["_meta"]
            elif "_summary" in obj:
                summary = obj["_summary"]
            else:
                hist.append(obj)
    return meta, summary, hist


def load_all(out_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        meta, summary, hist = load_run(path)
        if meta is None or summary is None:
            continue
        runs.append({"meta": meta, "summary": summary, "hist": hist,
                     "name": os.path.basename(path)[:-6]})
    return runs


def seed_average_curve(runs, key, max_step):
    """Average a per-step metric across seeds onto a common step grid."""
    grid = np.arange(0, max_step + 1, 200)
    stacked = []
    for r in runs:
        steps = np.array([h["step"] for h in r["hist"]])
        vals = np.array([h[key] for h in r["hist"]])
        if len(steps) < 2:
            continue
        # forward-fill last value after early stop
        interp = np.interp(grid, steps, vals, right=vals[-1])
        stacked.append(interp)
    if not stacked:
        return grid, None, None
    arr = np.stack(stacked)
    return grid, arr.mean(0), arr.std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="../../experiments/results/grid_main")
    ap.add_argument("--fig_dir", default="../../experiments/results/figures")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    runs = load_all(args.out_dir)
    print(f"loaded {len(runs)} runs")

    # group by (optimizer, wd, init_scale)
    groups = defaultdict(list)
    for r in runs:
        m = r["meta"]
        groups[(m["optimizer"], m["weight_decay"], m.get("init_scale", 1.0))].append(r)

    wds = sorted({k[1] for k in groups})
    scs = sorted({k[2] for k in groups})
    opts_present = [o for o in ["adamw", "sgdm", "muon"]
                   if any(k[0] == o for k in groups)]
    colors = {"adamw": "tab:blue", "muon": "tab:red", "sgdm": "tab:green"}

    # ---- accuracy curves & weight-norm, one figure per (wd, init_scale) ----
    for sc in scs:
        for wd in wds:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            present = False
            for opt in opts_present:
                rs = groups.get((opt, wd, sc), [])
                if not rs:
                    continue
                present = True
                max_step = max(r["hist"][-1]["step"] for r in rs)
                g, tr_m, _ = seed_average_curve(rs, "train_acc", max_step)
                _, te_m, te_s = seed_average_curve(rs, "test_acc", max_step)
                axes[0].plot(g, tr_m, "--", color=colors[opt], alpha=0.6,
                             label=f"{opt} train")
                axes[0].plot(g, te_m, "-", color=colors[opt],
                             label=f"{opt} test")
                if te_s is not None:
                    axes[0].fill_between(g, te_m - te_s, te_m + te_s,
                                         color=colors[opt], alpha=0.12)
                _, wn_m, _ = seed_average_curve(rs, "wn_hidden", max_step)
                axes[1].plot(g, wn_m, "-", color=colors[opt], label=f"{opt}")
            if not present:
                plt.close(fig)
                continue
            axes[0].set_xscale("log")
            axes[0].set_xlabel("step (log)")
            axes[0].set_ylabel("accuracy")
            axes[0].set_title(f"Accuracy  (wd={wd}, init_scale={sc})")
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.3)
            axes[1].set_xscale("log")
            axes[1].set_xlabel("step (log)")
            axes[1].set_ylabel("hidden weight L2 norm")
            axes[1].set_title(f"Hidden weight norm  (wd={wd}, init_scale={sc})")
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            fn = os.path.join(args.fig_dir, f"fig_curves_wd{wd}_sc{sc}.png")
            fig.savefig(fn, dpi=130)
            plt.close(fig)
            print("wrote", fn)

    # ---- delay-ratio / grok-step summary stats ----
    # "effective delay" right-censors non-grokking runs at their stop step:
    # eff_grok = grok_step if grokked else stopped_step (a lower bound on the
    # true grok step). This lets AdamW's "never groks in budget" show up honestly.
    stats = {}
    for (opt, wd, sc), rs in sorted(groups.items()):
        def agg(field):
            vals = [r["summary"][field] for r in rs if r["summary"].get(field) is not None]
            if not vals:
                return None, None, 0
            return float(np.mean(vals)), float(np.std(vals)), len(vals)
        gm, gs, gn = agg("grok_step")
        mm, ms, _ = agg("memorize_step")
        dm, ds, _ = agg("delay_ratio")
        eff_delays = []
        for r in rs:
            s = r["summary"]
            mem = s.get("memorize_step") or 1
            grok = s.get("grok_step")
            eff = (grok if grok is not None else s["stopped_step"]) / mem
            eff_delays.append(eff)
        fta = float(np.mean([r["summary"]["final_test_acc"] for r in rs]))
        stats[f"{opt}_wd{wd}_sc{sc}"] = {
            "optimizer": opt, "weight_decay": wd, "init_scale": sc,
            "n_seeds": len(rs), "n_grokked": gn,
            "grok_step_mean": gm, "grok_step_std": gs,
            "memorize_step_mean": mm, "memorize_step_std": ms,
            "delay_ratio_mean": dm, "delay_ratio_std": ds,
            "eff_delay_mean": float(np.mean(eff_delays)),
            "eff_delay_std": float(np.std(eff_delays)),
            "final_test_acc_mean": fta,
        }
    with open(os.path.join(args.fig_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote stats.json")

    # ---- effective-delay bar chart: conditions (sc,wd) x optimizers ----
    conds = [(sc, wd) for sc in scs for wd in wds]
    fig, ax = plt.subplots(figsize=(11, 5))
    n_opt = len(opts_present)
    width = 0.8 / n_opt
    x = np.arange(len(conds))
    for j, opt in enumerate(opts_present):
        means, errs, n_grok, n_tot = [], [], [], []
        for (sc, wd) in conds:
            s = stats.get(f"{opt}_wd{wd}_sc{sc}", {})
            means.append(s.get("eff_delay_mean") or 0)
            errs.append(s.get("eff_delay_std") or 0)
            n_grok.append(s.get("n_grokked") or 0)
            n_tot.append(s.get("n_seeds") or 0)
        bars = ax.bar(x + (j - (n_opt - 1) / 2) * width, means, width,
                      yerr=errs, capsize=3, color=colors[opt], alpha=0.85, label=opt)
        for b, ng, nt in zip(bars, n_grok, n_tot):
            if nt and ng < nt:  # mark censored (didn't always grok)
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{ng}/{nt}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sc={sc}\nwd={wd}" for (sc, wd) in conds])
    ax.set_ylabel("effective delay  (grok/memorize, log; censored at stop)")
    ax.set_title("Grokking delay by optimizer  (n/N = #seeds grokked when <full)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fn = os.path.join(args.fig_dir, "fig_delay_summary.png")
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    print("wrote", fn)


if __name__ == "__main__":
    main()

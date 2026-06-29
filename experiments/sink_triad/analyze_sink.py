"""Direction 008 analysis — P1–P5 verdicts for the sink-triad factorial.

Reads results/sink_triad/*.jsonl (main: opt × norm_position × 5 seeds;
depth arm: opt × L{1,3} × 3 seeds). Writes to results/figures-008/:
  sink_verdicts.json   per-cell aggregates + P4 correlations
  fig_triad_factorial.png  sink/spike/drain/peak by optimizer × norm position
  fig_ablation.png         P5 ablation cost by cell
  fig_depth.png            triad quantities vs depth (pre-norm)
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle
figstyle.apply()

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "experiments", "results")
DIR = os.path.join(RES, "sink_triad")
FIG = os.path.join(RES, "figures-008")
PAPERS_FIGS = os.path.join(RES, "..", "..", "papers", "figs")
COLORS = {k: figstyle.OPT[k] for k in ("adamw", "muon", "sgdm")}
OPT_LABEL = {"adamw": "AdamW", "muon": "Muon", "sgdm": "SGDM"}
FIELDS = ["final_sink_ratio", "final_spike_magnitude", "final_value_drain",
          "final_residual_peak", "final_ablation_cost", "final_backcopy_acc",
          "sink_formation_step"]


def load_summaries():
    main, depth = defaultdict(list), defaultdict(list)
    for p in sorted(glob.glob(os.path.join(DIR, "*.jsonl"))):
        last = None
        with open(p) as f:
            for line in f:
                last = line
        try:
            s = json.loads(last).get("_summary")
        except Exception:
            s = None
        if not s:
            continue
        name = os.path.basename(p)
        if name.startswith("depth_"):
            depth[(s["optimizer"], s["n_layers"])].append(s)
        else:
            main[(s["optimizer"], s["norm_position"])].append(s)
    return main, depth


def agg(cells, key):
    out = {}
    for cell, ss in sorted(cells.items()):
        vals = [s[key] for s in ss if s.get(key) is not None]
        out[cell] = (float(np.mean(vals)), float(np.std(vals)), len(vals)) \
            if vals else (None, None, 0)
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    main_cells, depth_cells = load_summaries()
    n_main = sum(len(v) for v in main_cells.values())
    n_depth = sum(len(v) for v in depth_cells.values())
    print(f"main runs: {n_main}; depth runs: {n_depth}")

    table = {"main": {}, "depth": {}}
    for f in FIELDS:
        table["main"][f] = {f"{o}_{n}": v for (o, n), v in agg(main_cells, f).items()}
        table["depth"][f] = {f"{o}_L{l}": v for (o, l), v in agg(depth_cells, f).items()}

    # readable console table (main arm)
    print(f"{'cell':18s} {'sink':>7s} {'spike':>9s} {'drain':>7s} {'peak':>7s} {'abl':>8s} {'backcopy':>9s}")
    for (o, n), ss in sorted(main_cells.items()):
        def m(k):
            return float(np.mean([s[k] for s in ss]))
        print(f"{o+'_'+n:18s} {m('final_sink_ratio'):7.3f} {m('final_spike_magnitude'):9.1f} "
              f"{m('final_value_drain'):7.3f} {m('final_residual_peak'):7.2f} "
              f"{m('final_ablation_cost'):8.4f} {m('final_backcopy_acc'):9.3f}")

    # ---- P4: does drain track peak or sink? (across ALL runs) ----
    allruns = [s for ss in list(main_cells.values()) + list(depth_cells.values())
               for s in ss]
    drain = np.array([s["final_value_drain"] for s in allruns])
    peak = np.array([s["final_residual_peak"] for s in allruns])
    sink = np.array([s["final_sink_ratio"] for s in allruns])
    r_dp = float(np.corrcoef(drain, peak)[0, 1])
    r_ds = float(np.corrcoef(drain, sink)[0, 1])
    table["P4"] = {"corr_drain_peak": r_dp, "corr_drain_sink": r_ds, "n": len(allruns)}
    print(f"P4: corr(drain, peak) = {r_dp:.3f}; corr(drain, sink) = {r_ds:.3f} (n={len(allruns)})")

    with open(os.path.join(FIG, "sink_verdicts.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("wrote sink_verdicts.json")

    # ---- factorial figure ----
    metrics = [("final_sink_ratio", "sink ratio", False),
               ("final_spike_magnitude", "spike magnitude", True),
               ("final_value_drain", "value drain", False),
               ("final_residual_peak", "residual peak", True)]
    # 4 panels -> figure* (full 2-column ~7in); render near final width
    fig, axes = plt.subplots(1, 4, figsize=(figstyle.WIDTH_IN["col2_full"], 2.6))
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]
    for ax, (key, lab, logy), tag in zip(axes, metrics, panel_tags):
        x = np.arange(2)  # pre, sandwich
        for j, opt in enumerate(["adamw", "sgdm", "muon"]):
            means, errs = [], []
            for npos in ["pre", "sandwich"]:
                m, sd, _ = agg(main_cells, key)[(opt, npos)]
                means.append(m)
                errs.append(sd)
            ax.bar(x + (j - 1) * 0.26, means, 0.26, yerr=errs, capsize=3,
                   color=COLORS[opt], alpha=0.9, label=OPT_LABEL[opt])
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(["pre", "sandwich"])
        ax.set_title(f"{tag} {lab}")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend()
    # suptitle removed: the caption carries the description
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_triad_factorial.png"))
    shutil.copy(os.path.join(FIG, "fig_triad_factorial.png"),
                os.path.join(PAPERS_FIGS, "A_sink.png"))
    plt.close(fig)
    print("wrote fig_triad_factorial.png + copied to papers/figs/A_sink.png")

    # ---- ablation cost (P5) ----
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(2)
    for j, opt in enumerate(["adamw", "sgdm", "muon"]):
        means, errs = [], []
        for npos in ["pre", "sandwich"]:
            m, sd, _ = agg(main_cells, "final_ablation_cost")[(opt, npos)]
            means.append(m)
            errs.append(sd)
        ax.bar(x + (j - 1) * 0.26, means, 0.26, yerr=errs, capsize=3,
               color=COLORS[opt], alpha=0.85, label=opt)
    ax.set_xticks(x)
    ax.set_xticklabels(["pre", "sandwich"])
    ax.set_ylabel("Δloss when sink column ablated")
    ax.set_title("P5: functional necessity of the sink, by cell")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_ablation.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_ablation.png")

    # ---- depth arm ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (key, lab, logy) in zip(axes, [
            ("final_sink_ratio", "sink ratio", False),
            ("final_spike_magnitude", "spike magnitude", True),
            ("final_residual_peak", "residual peak", True)]):
        for opt in ["adamw", "sgdm", "muon"]:
            xs, ys, es = [], [], []
            for L in [1, 2, 3]:
                if L == 2:  # L2 lives in the main pre-norm arm
                    m, sd, k = agg(main_cells, key)[(opt, "pre")]
                else:
                    m, sd, k = agg(depth_cells, key).get((opt, L), (None, None, 0))
                if m is not None:
                    xs.append(L)
                    ys.append(m)
                    es.append(sd)
            ax.errorbar(xs, ys, yerr=es, fmt="-o", color=COLORS[opt], capsize=3,
                        label=opt)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("n_layers")
        ax.set_title(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Depth arm (pre-norm): triad vs depth")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_depth.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_depth.png")


if __name__ == "__main__":
    main()

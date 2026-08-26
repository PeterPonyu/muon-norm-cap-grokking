"""Route-map analysis of the mechanism runs (direct two-route evidence).

For each run that groks, extract at T_grok relative to T_mem:
  - dlog_norm = log( wn_hidden(T_grok) / wn_hidden(T_mem) )   (x-axis)
  - angle_deg = arccos( cos_mem at T_grok )                    (y-axis)
  - rot_per_100 = angular distance per 100 steps in the mem->grok window
Predictions:
  contraction route (AdamW wd=1, SGDM): dlog_norm < 0, slow rotation
  directional route (Muon):             dlog_norm > 0, fast rotation
Also plots stable-rank trajectories (spectral flatness over training).
"""
from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "../../experiments/results/mech"
FIG_DIR = "../../experiments/results/figures"
COLORS = {"adamw": "tab:blue", "muon": "tab:red", "sgdm": "tab:green"}
MARKERS = {(1.0, 0.0): "o", (1.0, 1.0): "s", (3.0, 0.01): "^", (3.0, 1.0): "D"}


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


def rec_at(hist, step):
    return min(hist, key=lambda h: abs(h["step"] - step))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []
    trajs = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(DIR, "*.jsonl"))):
        meta, summ, hist = load(p)
        if not (meta and summ and hist):
            continue
        opt, sc, wd = meta["optimizer"], meta.get("init_scale", 1.0), meta["weight_decay"]
        trajs[(opt, sc, wd)].append(hist)
        T_mem, T_grok = summ["memorize_step"], summ["grok_step"]
        if T_mem is None or T_grok is None:
            continue
        h_mem, h_grok = rec_at(hist, T_mem), rec_at(hist, T_grok)
        cosm = h_grok.get("cos_mem")
        if cosm is None or h_mem["wn_hidden"] <= 0:
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosm))))
        dlog = math.log(h_grok["wn_hidden"] / h_mem["wn_hidden"])
        span = max(T_grok - T_mem, 1)
        rows.append({
            "opt": opt, "sc": sc, "wd": wd, "seed": meta["seed"],
            "T_mem": T_mem, "T_grok": T_grok,
            "dlog_norm": dlog, "angle_deg": angle,
            "rot_per_100": angle / span * 100.0,
        })

    with open(os.path.join(FIG_DIR, "mech_route_table.json"), "w") as f:
        json.dump(rows, f, indent=2)
    # per-cell means to stdout
    cells = defaultdict(list)
    for r in rows:
        cells[(r["opt"], r["sc"], r["wd"])].append(r)
    print(f"{'cell':26s} {'dlogN':>7s} {'angle°':>7s} {'°/100st':>8s}")
    for k in sorted(cells):
        rs = cells[k]
        print(f"{k[0]}_sc{k[1]}_wd{k[2]:<6} "
              f"{np.mean([r['dlog_norm'] for r in rs]):7.3f} "
              f"{np.mean([r['angle_deg'] for r in rs]):7.1f} "
              f"{np.mean([r['rot_per_100'] for r in rs]):8.2f}")

    # ---- route map ----
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for r in rows:
        ax.scatter(r["dlog_norm"], r["angle_deg"], s=70,
                   color=COLORS[r["opt"]], marker=MARKERS.get((r["sc"], r["wd"]), "o"),
                   alpha=0.75, edgecolors="k", linewidths=0.4)
    ax.axvline(0, color="gray", lw=1)
    ax.annotate("contraction route\n(norm shrinks)", xy=(0.02, 0.96),
                xycoords="axes fraction", fontsize=9, va="top", color="gray")
    ax.annotate("growth route\n(norm grows)", xy=(0.98, 0.96),
                xycoords="axes fraction", fontsize=9, va="top", ha="right", color="gray")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=c, label=o)
               for o, c in COLORS.items()]
    handles += [Line2D([0], [0], marker=m, ls="", color="gray",
                       label=f"sc={sc}, wd={wd}")
                for (sc, wd), m in MARKERS.items()]
    ax.legend(handles=handles, fontsize=8, loc="center left")
    ax.set_xlabel("Δlog hidden norm  (T_mem → T_grok)")
    ax.set_ylabel("angular distance traveled (degrees, T_mem → T_grok)")
    ax.set_title("Route map: every grokking event, by norm change vs direction change")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fn = os.path.join(FIG_DIR, "fig_route_map.png")
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    print("wrote", fn)

    # ---- stable-rank trajectories ----
    key_cells = [("muon", 1.0, 0.0), ("adamw", 1.0, 0.0),
                 ("adamw", 3.0, 1.0), ("sgdm", 3.0, 0.01)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for (opt, sc, wd) in key_cells:
        for i, hist in enumerate(trajs.get((opt, sc, wd), [])):
            steps = [h["step"] for h in hist if h["step"] > 0]
            label = f"{opt} sc{sc} wd{wd}" if i == 0 else None
            axes[0].plot(steps, [h["stable_rank_mean"] for h in hist if h["step"] > 0],
                         color=COLORS[opt], alpha=0.55, label=label,
                         ls="-" if sc == 1.0 else "--")
            axes[1].plot(steps, [h["rot_rate"] for h in hist if h["step"] > 0],
                         color=COLORS[opt], alpha=0.55, label=label,
                         ls="-" if sc == 1.0 else "--")
    for ax, ylab, title in [(axes[0], "stable rank (mean over hidden mats)",
                             "Spectral flatness"),
                            (axes[1], "rotation per eval (1 - cos prev)",
                             "Direction-change rate")]:
        ax.set_xscale("log")
        ax.set_xlabel("step (log)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[1].set_yscale("log")
    fig.tight_layout()
    fn = os.path.join(FIG_DIR, "fig_mech_trajectories.png")
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    print("wrote", fn)


if __name__ == "__main__":
    main()

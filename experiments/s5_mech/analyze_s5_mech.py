"""Direction 002 analysis — P1-P4 verdicts for S5 mechanism runs.

Reads:
  results/s5_mech/   (this direction's mech runs, op=s5)
  results/mech/      (001's modular-add mech runs — the P4 comparison arm)
  results/task_s5/   (acc-only behavioral baseline — probe non-perturbation check)

Writes (to results/figures-002/):
  s5_norm_ratio_table.json   P1: wn_hidden(T_grok)/wn_hidden(T_mem) per cell
  s5_behavior_reconciliation.json  probe vs baseline memorize/grok steps
  fig_s5_route_map.png       P1/P2: Δlog norm vs rotation angle, S5 vs mod-add overlay
  fig_s5_mech_trajectories.png  P2/P3: cos_mem / eff_rank / stable_rank trajectories
  fig_s5_vs_add_rank.png     P4: eff_rank at T_grok, S5 vs add, per optimizer
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

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "experiments", "results")
S5_DIR = os.path.join(RES, "s5_mech")
ADD_DIR = os.path.join(RES, "mech")
BASE_DIR = os.path.join(RES, "task_s5")
FIG = os.path.join(RES, "figures-002")
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


def rec_at(hist, step):
    return min(hist, key=lambda h: abs(h["step"] - step))


def load_dir(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        meta, summ, hist = load(p)
        if meta and summ and hist:
            out.append((meta, summ, hist))
    return out


def route_point(summ, hist):
    """(dlog_norm, angle_deg, rot_per_100) at T_grok rel. T_mem, or None."""
    T_mem, T_grok = summ["memorize_step"], summ["grok_step"]
    if T_mem is None or T_grok is None:
        return None
    h_mem, h_grok = rec_at(hist, T_mem), rec_at(hist, T_grok)
    cosm = h_grok.get("cos_mem")
    if cosm is None or not h_mem.get("wn_hidden"):
        return None
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosm))))
    dlog = math.log(h_grok["wn_hidden"] / h_mem["wn_hidden"])
    return dlog, angle, angle / max(T_grok - T_mem, 1) * 100.0


def main():
    os.makedirs(FIG, exist_ok=True)
    s5 = load_dir(S5_DIR)
    add = load_dir(ADD_DIR)
    base = load_dir(BASE_DIR)
    print(f"loaded: s5_mech={len(s5)} add_mech={len(add)} s5_baseline={len(base)}")

    # ---------- P1 norm-ratio table + behavior reconciliation ----------
    table, recon = {}, {}
    cells = defaultdict(list)
    for meta, summ, hist in s5:
        cells[(meta["optimizer"], meta.get("init_scale", 1.0))].append((meta, summ, hist))
    base_cells = defaultdict(list)
    for meta, summ, hist in base:
        base_cells[(meta["optimizer"], meta.get("init_scale", 1.0))].append(summ)

    for (opt, sc), runs in sorted(cells.items()):
        ratios, groks, mems = [], [], []
        for meta, summ, hist in runs:
            T_mem, T_grok = summ["memorize_step"], summ["grok_step"]
            if T_mem is not None:
                mems.append(T_mem)
            if T_mem is not None and T_grok is not None:
                h_mem, h_grok = rec_at(hist, T_mem), rec_at(hist, T_grok)
                if h_mem.get("wn_hidden"):
                    ratios.append(h_grok["wn_hidden"] / h_mem["wn_hidden"])
                groks.append(T_grok)
        key = f"{opt}_sc{sc}"
        table[key] = {
            "n_runs": len(runs), "n_grokked": len(groks),
            "norm_ratio_mean": float(np.mean(ratios)) if ratios else None,
            "norm_ratio_std": float(np.std(ratios)) if ratios else None,
            "grok_step_mean": float(np.mean(groks)) if groks else None,
            "memorize_step_mean": float(np.mean(mems)) if mems else None,
        }
        bsumms = base_cells.get((opt, sc), [])
        if bsumms:
            bgrok = [s["grok_step"] for s in bsumms if s["grok_step"] is not None]
            recon[key] = {
                "baseline_n": len(bsumms), "baseline_n_grokked": len(bgrok),
                "baseline_grok_mean": float(np.mean(bgrok)) if bgrok else None,
                "mech_n": len(runs), "mech_n_grokked": len(groks),
                "mech_grok_mean": float(np.mean(groks)) if groks else None,
            }

    for name, obj in [("s5_norm_ratio_table.json", table),
                      ("s5_behavior_reconciliation.json", recon)]:
        with open(os.path.join(FIG, name), "w") as f:
            json.dump(obj, f, indent=2)
        print("wrote", name)
    print(f"{'cell':16s} {'grok':>7s} {'ratio':>14s}")
    for k, v in table.items():
        r = (f"{v['norm_ratio_mean']:.3f}±{v['norm_ratio_std']:.3f}"
             if v["norm_ratio_mean"] is not None else "  --")
        print(f"{k:16s} {v['n_grokked']}/{v['n_runs']:<5d} {r:>14s}")

    # ---------- route map: S5 (filled) vs mod-add (faded) ----------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for meta, summ, hist in add:
        pt = route_point(summ, hist)
        if pt:
            ax.scatter(pt[0], pt[1], s=35, color=COLORS[meta["optimizer"]],
                       alpha=0.18, marker="o")
    for meta, summ, hist in s5:
        pt = route_point(summ, hist)
        if pt:
            sc = meta.get("init_scale", 1.0)
            ax.scatter(pt[0], pt[1], s=90, color=COLORS[meta["optimizer"]],
                       alpha=0.85, marker="^" if sc == 3.0 else "o",
                       edgecolors="k", linewidths=0.5)
    ax.axvline(0, color="gray", lw=1)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=c, label=f"{o} (S5)")
               for o, c in COLORS.items()]
    handles.append(Line2D([0], [0], marker="o", ls="", color="gray", alpha=0.3,
                          label="mod-add (001 arm, faded)"))
    handles.append(Line2D([0], [0], marker="^", ls="", color="gray", label="S5 sc=3"))
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel("Δlog hidden norm  (T_mem → T_grok)")
    ax.set_ylabel("angular distance (deg, T_mem → T_grok)")
    ax.set_title("Route map on S5 (filled) vs modular-add (faded): P1/P2 verdict")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_s5_route_map.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_s5_route_map.png")

    # ---------- trajectories: cos_mem / eff_rank / stable_rank ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for (opt, sc), runs in sorted(cells.items()):
        for j, (meta, summ, hist) in enumerate(runs):
            steps = [h["step"] for h in hist if h["step"] > 0]
            ls = "-" if sc == 1.0 else "--"
            lab = f"{opt} sc{sc}" if j == 0 else None
            axes[0].plot(steps, [h.get("cos_mem") if h.get("cos_mem") is not None
                                 else np.nan for h in hist if h["step"] > 0],
                         ls, color=COLORS[opt], alpha=0.4, label=lab)
            axes[1].plot(steps, [h["eff_rank_mean"] for h in hist if h["step"] > 0],
                         ls, color=COLORS[opt], alpha=0.4, label=lab)
            axes[2].plot(steps, [h["stable_rank_mean"] for h in hist if h["step"] > 0],
                         ls, color=COLORS[opt], alpha=0.4, label=lab)
    for ax, t in zip(axes, ["cos(θ_t, θ_mem)", "effective rank (entropy)",
                            "stable rank"]):
        ax.set_xscale("log")
        ax.set_xlabel("step (log)")
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("S5 mechanism trajectories (P2 rotation, P3 spectral non-collapse)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_s5_mech_trajectories.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_s5_mech_trajectories.png")

    # ---------- P4: eff_rank at T_grok, S5 vs add ----------
    def rank_at_grok(runs_list):
        out = defaultdict(list)
        for meta, summ, hist in runs_list:
            if summ["grok_step"] is None:
                continue
            h = rec_at(hist, summ["grok_step"])
            if "eff_rank_mean" in h:
                out[meta["optimizer"]].append(h["eff_rank_mean"])
        return out

    r_s5, r_add = rank_at_grok(s5), rank_at_grok(add)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    opts = sorted(set(r_s5) | set(r_add))
    x = np.arange(len(opts))
    for j, (rk, lab, alpha) in enumerate([(r_add, "mod-add", 0.55), (r_s5, "S5", 0.95)]):
        means = [np.mean(rk[o]) if rk.get(o) else 0 for o in opts]
        errs = [np.std(rk[o]) if rk.get(o) else 0 for o in opts]
        ax.bar(x + (j - 0.5) * 0.38, means, 0.38, yerr=errs, capsize=4,
               color=[COLORS[o] for o in opts], alpha=alpha, label=lab,
               edgecolor="k" if j else None)
    ax.set_xticks(x)
    ax.set_xticklabels(opts)
    ax.set_ylabel("effective rank of hidden matrices at T_grok")
    ax.set_title("P4: representation rank at grokking — S5 (dark) vs mod-add (light)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_s5_vs_add_rank.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_s5_vs_add_rank.png")


if __name__ == "__main__":
    main()

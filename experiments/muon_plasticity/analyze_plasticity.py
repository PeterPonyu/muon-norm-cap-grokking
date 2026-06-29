"""Direction 005 analysis — P1–P5 verdicts for Muon-as-plasticity-therapy.

Reads results/muon_plasticity/*.jsonl (opt × arm × 5 seeds, warm-start, 100 tasks).
Writes to results/figures-005/:
  plasticity_verdicts.json
  fig_fit_speed.png      steps-to-threshold vs task index (the plasticity curve)
  fig_spectra.png        feature eff-rank & weight eff-rank trajectories
  fig_dissociation.png   weight-spectrum retention vs feature retention vs outcome
Definitions:
  censor_onset  first task index whose steps_to_threshold is None
  fit_slope     OLS slope of steps_to_threshold over tasks fitted BEFORE censoring
  feat_retention = feat_eff_rank(last) / feat_eff_rank(task0); same for weights
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
DIR = os.path.join(RES, "muon_plasticity")
FIG = os.path.join(RES, "figures-005")
PAPERS_FIGS = os.path.join(RES, "..", "..", "papers", "figs")
COLORS = {k: figstyle.OPT[k] for k in ("adamw", "muon", "sgdm")}
OPT_LABEL = {"adamw": "AdamW", "muon": "Muon", "sgdm": "SGDM"}


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


def run_metrics(hist):
    tasks = [h["task"] for h in hist]
    stt = [h["steps_to_threshold"] for h in hist]
    fer = [h["probes"]["feat_eff_rank"] for h in hist]
    wer = [np.mean([h["probes"][k] for k in h["probes"] if k.startswith("w_eff_rank")])
           for h in hist]
    dead = [h["probes"]["dead_frac"] for h in hist]
    censor = next((t for t, s in zip(tasks, stt) if s is None), None)
    fitted = [(t, s) for t, s in zip(tasks, stt) if s is not None]
    slope = None
    if len(fitted) >= 5:
        xs, ys = zip(*fitted)
        slope = float(np.polyfit(xs, ys, 1)[0])
    return {
        "censor_onset": censor,
        "n_fit": len(fitted),
        "fit_slope": slope,
        "stt": stt, "fer": fer, "wer": wer, "dead": dead,
        "feat_retention": fer[-1] / fer[0] if fer[0] else None,
        "weight_retention": wer[-1] / wer[0] if wer[0] else None,
        "final_dead": dead[-1],
    }


def main():
    os.makedirs(FIG, exist_ok=True)
    cells = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(DIR, "*.jsonl"))):
        meta, summ, hist = load(p)
        if not (meta and summ and hist):
            continue
        cells[(summ["optimizer"], summ["arm"])].append(run_metrics(hist))

    table = {}
    print(f"{'cell':24s} {'censor@':>8s} {'n_fit':>6s} {'slope':>8s} "
          f"{'featR':>6s} {'wR':>6s} {'dead':>6s}")
    for (opt, arm), ms in sorted(cells.items()):
        def agg(key):
            vals = [m[key] for m in ms if m[key] is not None]
            return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals \
                else (None, None, 0)
        row = {k: agg(k) for k in ["censor_onset", "n_fit", "fit_slope",
                                   "feat_retention", "weight_retention",
                                   "final_dead"]}
        table[f"{opt}_{arm}"] = row
        print(f"{opt+'_'+arm:24s} "
              f"{row['censor_onset'][0] if row['censor_onset'][0] is not None else 'never':>8} "
              f"{row['n_fit'][0]:6.1f} "
              f"{row['fit_slope'][0] if row['fit_slope'][0] is not None else float('nan'):8.2f} "
              f"{row['feat_retention'][0]:6.2f} {row['weight_retention'][0]:6.2f} "
              f"{row['final_dead'][0]:6.2f}")

    with open(os.path.join(FIG, "plasticity_verdicts.json"), "w") as f:
        json.dump(table, f, indent=2)
    print("wrote plasticity_verdicts.json")

    # ---- fit-speed curves ----
    arms = sorted({a for (_, a) in cells})
    fig, axes = plt.subplots(1, len(arms), figsize=(7 * len(arms), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        for opt in ["adamw", "sgdm", "muon"]:
            ms = cells.get((opt, arm), [])
            if not ms:
                continue
            n = min(len(m["stt"]) for m in ms)
            mat = np.array([[s if s is not None else np.nan for s in m["stt"][:n]]
                            for m in ms], dtype=float)
            mean = np.nanmean(mat, axis=0)
            ax.plot(range(n), mean, color=COLORS[opt], label=opt, alpha=0.85)
            frac_c = np.isnan(mat).mean(0)
            ax.fill_between(range(n), 0, frac_c * np.nanmax(mean),
                            color=COLORS[opt], alpha=0.08)
        ax.set_xlabel("task index")
        ax.set_title(f"arm: {arm}  (shade ∝ censored fraction)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("steps to fit threshold (mean over seeds)")
    fig.suptitle("P1/P2: per-task fit speed across 100 tasks")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_fit_speed.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_fit_speed.png")

    # ---- spectra trajectories ----
    fig, axes = plt.subplots(2, len(arms), figsize=(7 * len(arms), 8), sharex=True)
    axes = axes.reshape(2, len(arms))
    for j, arm in enumerate(arms):
        for opt in ["adamw", "sgdm", "muon"]:
            ms = cells.get((opt, arm), [])
            if not ms:
                continue
            n = min(len(m["fer"]) for m in ms)
            fer = np.mean([m["fer"][:n] for m in ms], axis=0)
            wer = np.mean([m["wer"][:n] for m in ms], axis=0)
            axes[0, j].plot(range(n), fer, color=COLORS[opt], label=opt)
            axes[1, j].plot(range(n), wer, color=COLORS[opt], label=opt)
        axes[0, j].set_title(f"feature eff-rank — {arm}")
        axes[1, j].set_title(f"weight eff-rank (layer mean) — {arm}")
        axes[1, j].set_xlabel("task index")
        for ax in (axes[0, j], axes[1, j]):
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("P3/P4: which spectrum moves — features vs weights")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_spectra.png"), dpi=130)
    plt.close(fig)
    print("wrote fig_spectra.png")

    # ---- dissociation scatter ----
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    seen_opt = set()
    for (opt, arm), ms in sorted(cells.items()):
        for m in ms:
            ax.scatter(m["weight_retention"], m["feat_retention"],
                       s=30 + 200 * (m["n_fit"] / 100), color=COLORS[opt],
                       alpha=0.7, marker="o" if arm == "proj_shift" else "^")
            seen_opt.add(opt)
    ax.set_xlabel("weight eff-rank retention (last/first)")
    ax.set_ylabel("feature eff-rank retention (last/first)")
    # title removed (caption carries it); add explicit optimizer + arm legends
    from matplotlib.lines import Line2D
    opt_handles = [Line2D([0], [0], marker="o", ls="", color=COLORS[o],
                          label=OPT_LABEL[o]) for o in ("adamw", "muon", "sgdm")
                   if o in seen_opt]
    arm_handles = [Line2D([0], [0], marker="o", ls="", color="0.4", label="proj_shift"),
                   Line2D([0], [0], marker="^", ls="", color="0.4", label="label_refit")]
    leg1 = ax.legend(handles=opt_handles, loc="upper left", title="optimizer")
    ax.add_artist(leg1)
    ax.legend(handles=arm_handles, loc="lower right", title="arm (marker size $\\propto$ # fittable)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dissociation.png"))
    shutil.copy(os.path.join(FIG, "fig_dissociation.png"),
                os.path.join(PAPERS_FIGS, "A_plasticity.png"))
    plt.close(fig)
    print("wrote fig_dissociation.png + copied to papers/figs/A_plasticity.png")


if __name__ == "__main__":
    main()

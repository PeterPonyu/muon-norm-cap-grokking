"""Render the 021 norm-control causal figure for paper-A §3.2.

Reads results/figures-021/normctl_verdict.json (produced by analyze_normctl.py)
and draws the manipulated-cause result: a downward-only hidden-norm ceiling
k*||W||_init PRESERVES grokking 8/8 at every k and ACCELERATES it ~10x on S5
(eval-floored on mod-add), while ||W_hidden|| falls monotonically with the cap.

Usage: python plot_normctl.py
Writes results/figures-021/fig_normctl.png and ../../papers/figs/A_normctl.png
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle
figstyle.apply()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__))
C_S5, C_ADD = figstyle.OPT["muon"], figstyle.OPT["adamw"]
FIG = os.path.join(_THIS, "..", "results", "figures-021")
PAPERS_FIGS = os.path.join(_THIS, "..", "..", "papers", "figs")
K_ORDER = ["kinf", "k3", "k2", "k1p5", "k1"]
K_LABEL = {"kinf": r"$\infty$", "k3": "3", "k2": "2", "k1p5": "1.5", "k1": "1"}


def series(arm, field):
    rows = arm["rows"]
    return [rows[k][field] for k in K_ORDER if k in rows]


def labels(arm):
    return [K_LABEL[k] for k in K_ORDER if k in arm["rows"]]


def grok_text(arm):
    return [f"{arm['rows'][k]['n_grok']}/{arm['rows'][k]['n']}"
            for k in K_ORDER if k in arm["rows"]]


def main():
    with open(os.path.join(FIG, "normctl_verdict.json")) as f:
        v = json.load(f)
    s5, add = v["s5"], v["add"]
    x = list(range(len(K_ORDER)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.WIDTH_IN["col2_full"], 3.0))

    # Panel (a): grok step vs ceiling (the causal acceleration)
    ax1.plot(x, series(s5, "grok_step_med"), "o-", color=C_S5, lw=2,
             ms=8, label="S5 (non-abelian)")
    ax1.plot(x, series(add, "grok_step_med"), "s--", color=C_ADD, lw=2,
             ms=7, label="modular addition")
    for xi, gs, gt in zip(x, series(s5, "grok_step_med"), grok_text(s5)):
        ax1.annotate(gt, (xi, gs), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8, color=C_S5)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([K_LABEL[k] for k in K_ORDER])
    ax1.set_xlabel(r"hidden-norm ceiling $k$  ($\|W\|\leq k\,\|W\|_{\mathrm{init}}$)")
    ax1.set_ylabel("median grok step")
    ax1.set_title("(a) grok step vs ceiling")
    ax1.invert_xaxis()  # tighter cap to the right
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.annotate(r"$\sim$10$\times$ faster", (x[-1], series(s5, "grok_step_med")[-1]),
                 textcoords="offset points", xytext=(8, 18), fontsize=9,
                 color=C_S5,
                 arrowprops=dict(arrowstyle="->", color=C_S5))

    # Panel (b): final hidden norm vs ceiling (the cap actually bit)
    ax2.plot(x, series(s5, "wn_hidden_med"), "o-", color=C_S5, lw=2,
             ms=8, label="S5")
    ax2.plot(x, series(add, "wn_hidden_med"), "s--", color=C_ADD, lw=2,
             ms=7, label="modular addition")
    ax2.set_xticks(x)
    ax2.set_xticklabels([K_LABEL[k] for k in K_ORDER])
    ax2.set_xlabel(r"hidden-norm ceiling $k$")
    ax2.set_ylabel(r"median final $\|W_{\mathrm{hidden}}\|$")
    ax2.set_title("(b) final norm vs ceiling")
    ax2.invert_xaxis()
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(PAPERS_FIGS, exist_ok=True)
    for out in (os.path.join(FIG, "fig_normctl.png"),
                os.path.join(PAPERS_FIGS, "A_normctl.png")):
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A norm-match arm: TWO-SIDED constant-norm projection across optimizers.

The A norm-cap arm is one-sided (downward ceiling on Muon growth). This arm
answers the remaining objection -- "is it the norm LEVEL or the update
GEOMETRY?" -- by pinning every 2-D hidden matrix to EXACTLY its init norm
after every optimizer step (scale up AND down), for AdamW, Muon, and SGDM
alike. All optimizers then walk the same constant-norm sphere (the Omnigrok
constraint), so any remaining grok-speed difference is attributable to update
geometry alone.

Predictions: (i) matched-norm Muon still groks, and faster than matched-norm
AdamW (geometry, not norm level); (ii) matched-norm SGDM still fails on S5
(the family floor is not a norm-level artifact); (iii) doubles as the
Omnigrok-sphere baseline row for the acceleration-landscape table.

Reuses experiments/grokking/train.py via the build_optimizer monkeypatch,
appending a pseudo-optimizer that applies the projection after the real
step(s). Same operating point as s5_normctl (wd=0.01, init_scale=1, mech).

Modes: --smoke | --pilot (1 seed x 3 opts x {s5,add}) | --full (8 seeds)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_GROKKING = os.path.abspath(os.path.join(_THIS, "..", "grokking"))
if _GROKKING not in sys.path:
    sys.path.insert(0, _GROKKING)
_EXP = os.path.dirname(_THIS)
if _EXP not in sys.path:
    sys.path.append(_EXP)

import train as TRAIN  # noqa: E402
from muon import split_params_for_muon  # noqa: E402

OUT = os.path.join(_THIS, "..", "..", "experiments", "results", "normmatch")
STEPS = 20000
EVAL_EVERY = 50
WD = 0.01
INIT_SCALE = 1.0
OPTS = ["muon", "adamw", "sgdm"]
OPS = ["s5", "add"]

_ORIG_BUILD = TRAIN.build_optimizer


class ConstantNormProjector:
    """Pseudo-optimizer: two-sided projection of each hidden matrix onto its
    init-norm sphere. Appended after the real optimizer(s) so the trainer's
    `for opt in optimizers: opt.step()` applies it every step."""

    def __init__(self, hidden_params):
        self.params = list(hidden_params)
        self.target = {id(p): float(p.detach().norm()) for p in self.params}

    def zero_grad(self, set_to_none=True):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        for p in self.params:
            n = float(p.detach().norm())
            t = self.target[id(p)]
            if n > 0 and t > 0:
                p.mul_(t / n)
        return None


def matched_builder(model, cfg):
    opts = _ORIG_BUILD(model, cfg)
    hidden, _ = split_params_for_muon(model)
    return list(opts) + [ConstantNormProjector(hidden)]


def already_done(path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if size == 0:
            return False
        fh.seek(max(0, size - 4096))
        return '"_summary"' in fh.read().decode("utf-8", errors="replace")


def run_cells(cells):
    os.makedirs(OUT, exist_ok=True)
    TRAIN.build_optimizer = matched_builder
    for i, (op, optname, seed) in enumerate(cells):
        name = f"match_{op}_{optname}_s{seed}"
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        cfg = TRAIN.Config(op=op, optimizer=optname, init_scale=INIT_SCALE,
                           weight_decay=WD, seed=seed, steps=STEPS,
                           eval_every=EVAL_EVERY, mech=True)
        t0 = time.time()
        s, _ = TRAIN.run(cfg, out_path=path)
        print(f"[{i+1}/{len(cells)}] {name}: grok={s['grok_step']} "
              f"test={s['final_test_acc']:.3f} "
              f"wn_hidden={s['final_wn_hidden']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)


def run_smoke():
    # 1. projector unit test: two-sided (inflates small, shrinks big).
    torch.manual_seed(0)
    w_big = torch.nn.Parameter(torch.randn(8, 8))
    w_small = torch.nn.Parameter(torch.randn(8, 8))
    proj = ConstantNormProjector([w_big, w_small])
    with torch.no_grad():
        w_big.mul_(5.0)
        w_small.mul_(0.1)
    proj.step()
    for w in (w_big, w_small):
        n = float(w.detach().norm())
        t = proj.target[id(w)]
        assert abs(n - t) / t < 1e-4, f"projection missed: {n} vs {t}"
    print("SMOKE 1: two-sided projection pins norms exactly — OK")

    # 2. short runs, one per optimizer, CPU, no writes; hidden norm must stay
    #    at init while training proceeds.
    TRAIN.build_optimizer = matched_builder
    for optname in OPTS:
        cfg = TRAIN.Config(op="add", optimizer=optname, init_scale=INIT_SCALE,
                           weight_decay=WD, seed=0, steps=30, eval_every=15,
                           mech=True, device="cpu")
        s, hist = TRAIN.run(cfg, out_path=None)
        wn0, wn_last = hist[0]["wn_hidden"], hist[-1]["wn_hidden"]
        assert abs(wn_last - wn0) / wn0 < 0.02, \
            f"{optname}: hidden norm drifted {wn0:.2f}->{wn_last:.2f}"
        print(f"SMOKE 2 [{optname}]: wn_hidden pinned "
              f"{wn0:.2f}->{wn_last:.2f}; loss finite")
    print("RUN_NORMMATCH SMOKE PASS: zero writes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return
    if args.pilot:
        run_cells([(op, o, 0) for op in OPS for o in OPTS])
        return
    if args.full:
        run_cells([(op, o, s) for op in OPS for o in OPTS
                   for s in range(args.seeds)])
        return
    ap.print_help()


if __name__ == "__main__":
    main()

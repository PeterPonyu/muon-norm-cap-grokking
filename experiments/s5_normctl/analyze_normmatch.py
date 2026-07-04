#!/usr/bin/env python3
"""Analysis for the two-sided constant-norm (sphere) arm vs vanilla baselines.

Prints, per (task, optimizer): vanilla grok-step stats vs sphere grok-step
stats, grok rates, and the pinned-norm sanity (wn_hidden must sit at init).
Baselines: s5_mech (S5, 5 seeds) and grid_main (mod-add, 5 seeds), both at
wd=0.01, init_scale=1.0.
"""
import json
import glob
import os
from statistics import median

_THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(_THIS, "..", "results")


def summaries(pat, **filt):
    out = []
    for f in glob.glob(os.path.join(RES, pat)):
        s = None
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if "_summary" in r:
                    s = r["_summary"]
        if s and all(s.get(k) == v for k, v in filt.items()):
            out.append(s)
    return out


def stats(runs):
    groks = [s["grok_step"] for s in runs]
    ok = [g for g in groks if g is not None]
    return {
        "n": len(runs),
        "grok_rate": f"{len(ok)}/{len(runs)}",
        "grok_med": median(ok) if ok else None,
        "grok_range": (min(ok), max(ok)) if ok else None,
        "wn_hidden_med": round(median(s["final_wn_hidden"] for s in runs), 2)
        if runs else None,
    }


BASE = {
    "s5": ("s5_mech/*.jsonl", dict(weight_decay=0.01, init_scale=1.0)),
    "add": ("grid_main/*.jsonl", dict(op="add", weight_decay=0.01,
                                      init_scale=1.0, train_frac=0.4)),
}

for op in ("s5", "add"):
    print(f"== {op} ==")
    for opt in ("muon", "adamw", "sgdm"):
        bpat, bfilt = BASE[op]
        van = stats(summaries(bpat, optimizer=opt, **({} if op == "add"
                    else dict(op="s5")), **bfilt))
        sph = stats(summaries(f"normmatch/match_{op}_{opt}_s*.jsonl"))
        print(f"  {opt:6s} vanilla: {van}")
        print(f"         sphere : {sph}")

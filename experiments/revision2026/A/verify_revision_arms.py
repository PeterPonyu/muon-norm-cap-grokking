#!/usr/bin/env python3
"""Verify launch-agent numbers for Paper A revision integration (2026-07-17).

Arms:
  P1-A1  pilot-AB/p1a1_lrcross      S5 lr-cross (adamw/muon x native/crossed, seeds 10-17)
  P1-A2  pilot-AB/p1a2_capfine      cap fine-cadence (k1, kinf=uncapped, seeds 10-17)
  P2-A1m t1rem-AB/p2a1m_lrcross_add mod-add lr-cross (seeds 20-27)
  P2-A2r t1rem-AB/p2a2r_capfine     cap dose (adamw? k2 k4, seeds 20-27)

For each run: grok step recomputed from the eval timeseries as the first eval
step with test_acc >= grok_thresh (and cross-checked against _summary.grok_step).
Reports per-arm: n, grok fraction, median grok step (over grokked runs and
over all runs treating non-grok as censored/inf -> median only if fraction>0.5),
plus the _meta lr/muon_lr/eval_every/op audit.
"""
from pathlib import Path
import json, glob, os, statistics, sys
from collections import defaultdict

BASE = str(Path(__file__).resolve().parents[3] / 'experiments' / 'revision2026')
ARMS = {
    "P1-A1_s5_lrcross": os.path.join(BASE, "pilot-AB/p1a1_lrcross"),
    "P1-A2_capfine":    os.path.join(BASE, "pilot-AB/p1a2_capfine"),
    "P2-A1m_add_lrcross": os.path.join(BASE, "t1rem-AB/p2a1m_lrcross_add"),
    "P2-A2r_capdose":   os.path.join(BASE, "t1rem-AB/p2a2r_capfine"),
}

def load_run(path):
    meta = None; summ = None; series = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "_meta" in rec:
                meta = rec["_meta"]
            elif "_summary" in rec:
                summ = rec["_summary"]
            else:
                series.append(rec)
    return meta, summ, series

def first_cross(series, thresh):
    for rec in series:
        if rec.get("test_acc") is not None and rec["test_acc"] >= thresh:
            return rec["step"]
    return None

def arm_key(fname):
    # strip seed suffix: e.g. muon_native_s10.jsonl -> muon_native
    base = os.path.basename(fname).replace(".jsonl", "")
    parts = base.split("_s")
    return parts[0]

out = {}
for arm, d in ARMS.items():
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        if path.endswith("_smoke.jsonl"):
            continue
        meta, summ, series = load_run(path)
        if meta is None or summ is None:
            print(f"WARN incomplete run {path}", file=sys.stderr)
            continue
        thresh = meta.get("grok_thresh", 0.95)
        recomputed = first_cross(series, thresh)
        summary_gs = summ.get("grok_step")
        mismatch = (recomputed != summary_gs)
        groups[arm_key(path)].append({
            "file": os.path.basename(path),
            "seed": meta.get("seed"),
            "grok_step_summary": summary_gs,
            "grok_step_recomputed": recomputed,
            "mismatch": mismatch,
            "lr": meta.get("lr"),
            "muon_lr": meta.get("muon_lr"),
            "eval_every": meta.get("eval_every"),
            "op": meta.get("op"),
            "optimizer": meta.get("optimizer"),
            "steps": meta.get("steps"),
            "cap_k": meta.get("cap_k", meta.get("norm_cap_k")),
            "final_test_acc": summ.get("final_test_acc"),
        })
    arm_out = {}
    for g, runs in sorted(groups.items()):
        groks = [r["grok_step_recomputed"] for r in runs if r["grok_step_recomputed"] is not None]
        n = len(runs)
        frac = len(groks) / n if n else 0.0
        med = statistics.median(groks) if groks else None
        mism = [r["file"] for r in runs if r["mismatch"]]
        # config audit
        lrs = sorted({(r["lr"], r["muon_lr"]) for r in runs})
        evals = sorted({r["eval_every"] for r in runs})
        ops = sorted({r["op"] for r in runs})
        opts = sorted({r["optimizer"] for r in runs})
        caps = sorted({str(r["cap_k"]) for r in runs})
        arm_out[g] = {
            "n": n, "grok_frac": f"{len(groks)}/{n}", "median_grok_step": med,
            "grok_steps_sorted": sorted(groks),
            "summary_vs_recompute_mismatches": mism,
            "lr_muonlr": lrs, "eval_every": evals, "op": ops,
            "optimizer": opts, "cap_k": caps,
        }
    out[arm] = arm_out

print(json.dumps(out, indent=2))
with open(os.path.join(BASE, "A", "verify_revision_arms.json"), "w") as f:
    json.dump(out, f, indent=2)

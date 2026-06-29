"""A-ISSUE-3 — is norm GROWTH causal for grokking, or a byproduct?

Norm-controlled Muon (downward-only ceiling = k * init weight-norm) with
k ∈ {inf, 3, 2, 1.5, 1}, 8 seeds. If growth were the mechanism, capping it should
DELAY or block grokking. findings-021 (S5, results/s5_normctl) found the opposite:
grok survives at every ceiling incl. k=1, and capping ACCELERATES grok ~10× while
‖W_hidden‖ falls monotonically — growth is a byproduct of the orthogonalized
update, not the cause.

This analyzer adjudicates BOTH the original S5 arm (sanity) and the mod-add
replication (results/add_normctl). Replication holds iff, on mod-add:
  (a) grok rate stays high (≈8/8) at every ceiling incl. k=1;
  (b) median grok_step does NOT increase as the ceiling tightens (no delay; ideally
      accelerates);
  (c) ‖W_hidden‖ falls monotonically with k (the cap actually bit);
  (d) kinf reproduces the uncapped growth baseline (largest ‖W_hidden‖).

Usage: python analyze_normctl.py            # both arms
Writes results/figures-021/normctl_verdict.json (no plot — packaging deferred).
"""
from __future__ import annotations
import glob, json, os, re
from collections import defaultdict
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
DIRS = {"s5": os.path.join(_THIS, "..", "results", "s5_normctl"),
        "add": os.path.join(_THIS, "..", "results", "add_normctl")}
FIG = os.path.join(_THIS, "..", "results", "figures-021")
K_ORDER = ["kinf", "k3", "k2", "k1p5", "k1"]
K_VAL = {"kinf": float("inf"), "k3": 3.0, "k2": 2.0, "k1p5": 1.5, "k1": 1.0}


def summ(path):
    last = None
    with open(path) as f:
        for l in f:
            if l.strip():
                last = l
    if last is None:
        return None
    try:
        return json.loads(last).get("_summary")
    except json.JSONDecodeError:
        return None


def load(folder):
    by_k = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
        m = re.match(r"(k\w+?)_s\d+\.jsonl", os.path.basename(path))
        if not m:
            continue
        s = summ(path)
        if s:
            by_k[m.group(1)].append(s)
    return by_k


def grokked(s):
    gs = s.get("grok_step")
    return gs is not None and gs < s.get("steps", 1e9)


def arm_verdict(by_k):
    rows = {}
    for k in K_ORDER:
        runs = by_k.get(k)
        if not runs:
            continue
        gk = [r for r in runs if grokked(r)]
        gsteps = [r["grok_step"] for r in gk]
        wn = [r.get("final_wn_hidden") for r in runs if r.get("final_wn_hidden") is not None]
        acc = [r.get("final_test_acc") for r in runs if r.get("final_test_acc") is not None]
        rows[k] = {
            "k": K_VAL[k], "n": len(runs), "n_grok": len(gk),
            "grok_step_med": float(np.median(gsteps)) if gsteps else None,
            "wn_hidden_med": float(np.median(wn)) if wn else None,
            "test_acc_med": float(np.median(acc)) if acc else None,
        }
    # checks
    ks_present = [k for k in K_ORDER if k in rows]
    grok_all = all(rows[k]["n_grok"] == rows[k]["n"] for k in ks_present)
    wn_seq = [rows[k]["wn_hidden_med"] for k in ks_present]
    wn_monotone_down = all(a >= b - 1e-6 for a, b in zip(wn_seq, wn_seq[1:]))
    # grok delay: compare tightest cap (k1) vs uncapped (kinf)
    delay = None
    if "k1" in rows and "kinf" in rows and rows["k1"]["grok_step_med"] and rows["kinf"]["grok_step_med"]:
        delay = rows["k1"]["grok_step_med"] / rows["kinf"]["grok_step_med"]
    return {"rows": rows, "grok_at_every_ceiling": grok_all,
            "wn_monotone_decreasing": wn_monotone_down,
            "grok_step_ratio_k1_over_kinf": delay}


def main():
    os.makedirs(FIG, exist_ok=True)
    out = {}
    for arm, folder in DIRS.items():
        if os.path.isdir(folder):
            out[arm] = arm_verdict(load(folder))
    p = os.path.join(FIG, "normctl_verdict.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"wrote {p}\n")
    for arm in ("s5", "add"):
        if arm not in out:
            continue
        v = out[arm]
        label = "S5 (findings-021 baseline)" if arm == "s5" else "MOD-ADD (replication)"
        print(f"== {label} ==")
        print("  k       n  grok  grok_step(med)  ‖W_hidden‖(med)  test_acc(med)")
        for k in K_ORDER:
            r = v["rows"].get(k)
            if r:
                print(f"  {k:5s} {r['n']:3d} {r['n_grok']:4d}/{r['n']}  "
                      f"{str(r['grok_step_med']):>12s}  {r['wn_hidden_med']:13.2f}  {r['test_acc_med']:.3f}")
        ratio = v["grok_step_ratio_k1_over_kinf"]
        print(f"  -> grok at EVERY ceiling incl k=1: {v['grok_at_every_ceiling']}")
        print(f"  -> ‖W_hidden‖ monotone DOWN with tighter cap: {v['wn_monotone_decreasing']}")
        if ratio is not None:
            verb = "ACCELERATES" if ratio < 1 else ("DELAYS" if ratio > 1 else "unchanged")
            print(f"  -> grok_step(k1)/grok_step(kinf) = {ratio:.3f}  => capping {verb} grokking")
        print()


if __name__ == "__main__":
    main()

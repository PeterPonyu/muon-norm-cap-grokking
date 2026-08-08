#!/usr/bin/env python3
"""Bootstrap CIs for the 2026-07-17 revision arms (P1-A1/A2, P2-A1m/A2r).

Reuses the ratio-of-medians bootstrap from bootstrap_cis.py (same B=20000,
percentile + BCa, seed-level independent within-arm resampling).

Entries:
  - cap fine-cadence (eval=5, fresh seeds): kinf/k1 (reproduces 10.8x headline
    defloored), kinf/k2, kinf/k4 (new dose point). kinf baseline is the pilot
    batch (seeds 10-17); k2/k4 are t1rem batch (seeds 20-27) -> cross-batch,
    noted.
  - mod-add lr-cross: AdamW/Muon ratio at each hidden-lr operating point
    (1e-3 and 2e-2) and at each optimizer's native point.
"""
from pathlib import Path
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap_cis as bc

BASE = str(Path(__file__).resolve().parents[3] / 'experiments' / 'revision2026')
OUT = os.path.join(BASE, "A")

def arm(folder, pattern, expect_n=8, allow_nongrok=False):
    d, ng, _ = bc.grok_steps(folder, pattern)
    if not allow_nongrok:
        assert ng == 0, (folder, pattern, ng)
    assert len(d) + ng == expect_n, (folder, pattern, len(d), ng)
    return d

capfine = os.path.join(BASE, "pilot-AB", "p1a2_capfine")
capdose = os.path.join(BASE, "t1rem-AB", "p2a2r_capfine")
addlr = os.path.join(BASE, "t1rem-AB", "p2a1m_lrcross_add")

kinf = arm(capfine, "kinf_s*.jsonl")
k1 = arm(capfine, "k1_s*.jsonl")
k2 = arm(capdose, "k2_s*.jsonl")
k4 = arm(capdose, "k4_s*.jsonl")

aw_nat = arm(addlr, "adamw_native_s*.jsonl")
aw_cro = arm(addlr, "adamw_crossed_s*.jsonl")
mu_nat = arm(addlr, "muon_native_s*.jsonl")
mu_cro = arm(addlr, "muon_crossed_s*.jsonl")

entries = [
    bc.entry("capfine eval=5: kinf/k1", kinf, k1,
             "pilot-AB/p1a2_capfine, 8 fresh seeds/arm, eval_every=5 (defloored); reproduces 10.8x headline"),
    bc.entry("capdose eval=5: kinf/k2", kinf, k2,
             "kinf pilot batch (s10-17) vs k2 t1rem batch (s20-27), eval_every=5, cross-batch"),
    bc.entry("capdose eval=5: kinf/k4", kinf, k4,
             "kinf pilot batch (s10-17) vs k4 t1rem batch (s20-27), eval_every=5, cross-batch; NEW dose point"),
    bc.entry("mod-add lr-cross: AdamW/Muon at native lrs (AdamW 1e-3, Muon 2e-2)", aw_nat, mu_nat,
             "t1rem-AB/p2a1m_lrcross_add, 8 seeds/arm, eval_every=50"),
    bc.entry("mod-add lr-cross: AdamW/Muon at hidden lr 1e-3 (Muon crossed)", aw_nat, mu_cro,
             "both optimizers at hidden lr 1e-3"),
    bc.entry("mod-add lr-cross: AdamW/Muon at hidden lr 2e-2 (AdamW crossed)", aw_cro, mu_nat,
             "both optimizers at hidden lr 2e-2"),
    bc.entry("mod-add lr-cross: AdamW crossed / Muon crossed", aw_cro, mu_cro,
             "each at the other's lr"),
]
out = {"B": bc.B, "statistic": "ratio of group medians", "entries": entries}
with open(os.path.join(OUT, "bootstrap_cis_revarms.json"), "w") as f:
    json.dump(out, f, indent=2)
for e in entries:
    print(f"{e['name']}: ratio {e['ratio']:.2f} pct[{e['ci95_percentile'][0]:.2f},{e['ci95_percentile'][1]:.2f}]"
          f" bca[{e['ci95_bca'][0]:.2f},{e['ci95_bca'][1]:.2f}] (med {e['median_base']}/{e['median_treat']})")

#!/usr/bin/env python3
"""S-A real-text pretraining analysis (CPU-only, raw jsonls).

Families analyzed SEPARATELY (never pooled):
  sa10: 10M-param byte-LM, TinyStories, ~0.3B tokens, arms {muon, adamw, cap1, cap2, sphere} x 3 seeds
  sa50: 57M-param byte-LM, FineWeb, ~0.6B tokens, arms {muon, adamw, cap1, cap2} x 3 seeds

Outputs analyze_stier_sa.json next to this script.

Computes, per family:
  - config audit (_meta fields shared vs differing across arms)
  - matched-token final val loss: per-seed + median, at the LARGEST token count
    present in every run of the family (all runs share cadence, so this is the
    common final grid point)
  - trajectory val loss at 10/25/50/75/100% of matched tokens (median of 3)
  - hidden-norm trajectory: init norm, final norm, ratio final/init, and for cap
    arms the first token count where wn_hidden >= 0.98 * k * init (cap binding)
  - sphere norm flatness (max |wn_hidden/init - 1| over trajectory)
  - early phase: muon vs adamw val loss at each of the first 8 eval points
  - tokens/s median per arm (sanity)
"""
import json, glob, os, statistics

BASE = "/home/zeyufu/Desktop/dl-research/experiments/revision2026/stier-AB"
FAMILIES = {
    "sa10": ["muon", "adamw", "cap1", "cap2", "sphere"],
    "sa50": ["muon", "adamw", "cap1", "cap2"],
}
SEEDS = [0, 1, 2]

def load(fam, arm, seed):
    p = os.path.join(BASE, f"stier_{fam}", f"{fam}_{arm}_s{seed}.jsonl")
    meta, summary, recs = None, None, []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            elif "_summary" in d:
                summary = d["_summary"]
            else:
                recs.append(d)
    assert meta is not None and summary is not None, p
    return meta, summary, recs

def med(xs):
    return statistics.median(xs)

out = {}
for fam, arms in FAMILIES.items():
    runs = {}
    for arm in arms:
        for s in SEEDS:
            runs[(arm, s)] = load(fam, arm, s)

    # ---- config audit ----
    metas = {k: v[0] for k, v in runs.items()}
    keys = set().union(*[set(m) for m in metas.values()])
    shared, differing = {}, {}
    for key in sorted(keys):
        vals = {f"{a}_s{s}": metas[(a, s)].get(key) for (a, s) in metas}
        uniq = set(map(str, vals.values()))
        if len(uniq) == 1:
            shared[key] = metas[(arms[0], 0)].get(key)
        else:
            # collapse per-arm if seed-invariant
            per_arm = {}
            for a in arms:
                av = {str(metas[(a, s)].get(key)) for s in SEEDS}
                per_arm[a] = metas[(a, 0)].get(key) if len(av) == 1 else {s: metas[(a, s)].get(key) for s in SEEDS}
            differing[key] = per_arm
    fam_out = {"config_shared": shared, "config_differing_across_arms": differing}

    # ---- matched tokens = min over runs of max tokens_seen ----
    max_tok = {k: max(r["tokens_seen"] for r in v[2]) for k, v in runs.items()}
    matched = min(max_tok.values())
    fam_out["max_tokens_per_run"] = {f"{a}_s{s}": max_tok[(a, s)] for (a, s) in runs}
    fam_out["matched_tokens"] = matched

    def val_at(recs, tok):
        """val loss at the last record with tokens_seen <= tok."""
        best = None
        for r in recs:
            if r["tokens_seen"] <= tok and r["val_loss"] is not None:
                best = r
        return best["val_loss"], best["tokens_seen"]

    # ---- final + trajectory ----
    final = {}
    for a in arms:
        per_seed, toks = [], []
        for s in SEEDS:
            v, t = val_at(runs[(a, s)][2], matched)
            per_seed.append(round(v, 4)); toks.append(t)
        final[a] = {"per_seed": per_seed, "median": round(med(per_seed), 4),
                    "at_tokens": toks}
    fam_out["final_val_loss_at_matched_tokens"] = final

    traj = {}
    for frac in (0.10, 0.25, 0.50, 0.75, 1.00):
        tok = matched * frac
        row = {}
        for a in arms:
            vs = [val_at(runs[(a, s)][2], tok)[0] for s in SEEDS]
            row[a] = round(med(vs), 4)
        traj[f"{int(frac*100)}pct"] = row
    fam_out["trajectory_median_val_loss"] = traj

    # ---- hidden norms ----
    norms = {}
    for a in arms:
        rows = []
        for s in SEEDS:
            recs = runs[(a, s)][2]
            init = recs[0]["wn_hidden"]
            fin = recs[-1]["wn_hidden"]
            row = {"init": round(init, 2), "final": round(fin, 2),
                   "final_over_init": round(fin / init, 3)}
            k = metas[(a, s)].get("ceiling_k") or 0
            if a in ("cap1", "cap2") and k:
                bind = next((r["tokens_seen"] for r in recs
                             if r["wn_hidden"] >= 0.98 * k * init), None)
                row["cap_k"] = k
                row["cap_binds_at_tokens"] = bind
            if a == "sphere":
                row["max_abs_dev_from_init"] = round(
                    max(abs(r["wn_hidden"] / init - 1) for r in recs), 4)
            rows.append(row)
        norms[a] = rows
    fam_out["hidden_norms"] = norms

    # ---- early phase muon vs adamw ----
    early = []
    n_evals = min(len(runs[(a, s)][2]) for a in ("muon", "adamw") for s in SEEDS)
    for i in range(1, min(9, n_evals)):
        tok = runs[("muon", 0)][2][i]["tokens_seen"]
        early.append({
            "tokens": tok,
            "muon_med": round(med([runs[("muon", s)][2][i]["val_loss"] for s in SEEDS]), 4),
            "adamw_med": round(med([runs[("adamw", s)][2][i]["val_loss"] for s in SEEDS]), 4),
        })
    fam_out["early_phase_muon_vs_adamw"] = early

    # ---- tokens/s sanity ----
    tps = {}
    for a in arms:
        vs = []
        for s in SEEDS:
            xs = [r["tokens_per_s"] for r in runs[(a, s)][2] if r["tokens_per_s"] > 0]
            vs.append(med(xs))
        tps[a] = round(med(vs))
    fam_out["tokens_per_s_median"] = tps

    out[fam] = fam_out

dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_stier_sa.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({fam: {
    "matched_tokens": out[fam]["matched_tokens"],
    "config_differing": {k: v for k, v in out[fam]["config_differing_across_arms"].items()},
    "final": out[fam]["final_val_loss_at_matched_tokens"],
    "traj": out[fam]["trajectory_median_val_loss"],
    "tps": out[fam]["tokens_per_s_median"],
} for fam in out}, indent=1))

#!/usr/bin/env python3
"""A-T0-1 (revision plan 2026-07-16): bootstrap CIs + per-seed values for every
Paper-A headline acceleration ratio, plus a quantification of the 10.8x-vs-15.9x
between-batch discrepancy at the nominal d=128/depth-2 S5 configuration.

Data: RAW per-run jsonls (never derived tables, per LESSONS-AND-ERRATA rule 3).
  - results/s5_normctl/            primary 8-seed S5 dose-response (10.8x)
  - results/add_normctl/           mod-add replication (eval-floored)
  - results/s5_normctl_scale/      capwd_* width x depth grid (15.9x..38.2x),
                                   capscale_* modulus ladder
  - results/ieee_gap_20260705/A/   normaccel (A5 1.76x, D60 1.33x), depth4
  - results/normmatch/             two-sided sphere control (S5 Muon ~10x;
                                   mod-add AdamW ~3.4x vs grid_main baseline)
  - results/grid_main/             AdamW wd=0.01 mod-add baseline for the sphere

Statistic: ratio of group medians, median(grok_step | baseline)/median(grok_step | treat).
CI: nonparametric bootstrap, B=20000 resamples, independent resampling within each
arm; percentile AND BCa intervals (BCa via z0 from the bootstrap distribution +
jackknife acceleration over both samples). Grok steps are quantized to the 50-step
eval grid, so CI endpoints land on ratio values of grid medians.

Also: variance decomposition (between-batch vs within-batch, log10 grok steps) and
Mann-Whitney U for the s5_normctl-vs-capwd_d128_L2 batch comparison.

CPU-only, stdlib+numpy(+scipy if available). Output: bootstrap_cis.json + .csv here.
"""
from __future__ import annotations
import glob, json, math, os, re
import numpy as np

RES = "/home/zeyufu/Desktop/dl-research/experiments/results"
OUT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260716)
B = 20000

try:
    from scipy.stats import mannwhitneyu, norm as scipy_norm
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


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


def grok_steps(folder, pattern):
    """Return {seed: grok_step} for files matching pattern (grokked runs only),
    plus the set of seeds present and count of non-grokked runs."""
    out, nongrok, metas = {}, 0, []
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        m = re.search(r"_s(\d+)\.jsonl$", os.path.basename(path))
        if not m:
            continue
        s = summ(path)
        if s is None:
            continue
        metas.append(s)
        gs = s.get("grok_step")
        budget = s.get("steps", 10**9)
        grokked = s.get("grokked", None)
        if grokked is None:
            grokked = gs is not None and gs < budget
        if grokked and gs is not None:
            out[int(m.group(1))] = float(gs)
        else:
            nongrok += 1
    return out, nongrok, metas


def ratio_of_medians(base, treat):
    return float(np.median(base) / np.median(treat))


def bootstrap_ratio(base, treat, B=B):
    base, treat = np.asarray(base, float), np.asarray(treat, float)
    obs = ratio_of_medians(base, treat)
    nb, nt = len(base), len(treat)
    bs = np.median(base[RNG.integers(0, nb, (B, nb))], axis=1) / \
         np.median(treat[RNG.integers(0, nt, (B, nt))], axis=1)
    pct = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    # BCa
    bca = None
    try:
        # z0: bias correction (guard against 0/1 proportions)
        prop = np.clip(np.mean(bs < obs) + 0.5 * np.mean(bs == obs), 1e-6, 1 - 1e-6)
        if HAVE_SCIPY:
            ppf, cdf = scipy_norm.ppf, scipy_norm.cdf
        else:
            ppf = lambda p: math.sqrt(2) * erfinv(2 * p - 1)
            cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        z0 = ppf(prop)
        # jackknife over both samples for acceleration
        jack = []
        for i in range(nb):
            jack.append(ratio_of_medians(np.delete(base, i), treat))
        for i in range(nt):
            jack.append(ratio_of_medians(base, np.delete(treat, i)))
        jack = np.asarray(jack)
        jm = jack.mean()
        num = np.sum((jm - jack) ** 3)
        den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
        a = num / den if den != 0 else 0.0
        lo_p = cdf(z0 + (z0 + ppf(0.025)) / (1 - a * (z0 + ppf(0.025))))
        hi_p = cdf(z0 + (z0 + ppf(0.975)) / (1 - a * (z0 + ppf(0.975))))
        bca = (float(np.percentile(bs, 100 * lo_p)),
               float(np.percentile(bs, 100 * hi_p)))
    except Exception:
        pass
    return obs, pct, bca


def erfinv(x):  # fallback if no scipy
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


def entry(name, base_dict, treat_dict, note=""):
    base = [base_dict[k] for k in sorted(base_dict)]
    treat = [treat_dict[k] for k in sorted(treat_dict)]
    obs, pct, bca = bootstrap_ratio(base, treat)
    return {
        "name": name, "note": note,
        "n_base": len(base), "n_treat": len(treat),
        "base_per_seed": {str(k): base_dict[k] for k in sorted(base_dict)},
        "treat_per_seed": {str(k): treat_dict[k] for k in sorted(treat_dict)},
        "median_base": float(np.median(base)), "median_treat": float(np.median(treat)),
        "ratio": obs,
        "ci95_percentile": pct, "ci95_bca": bca,
    }


def main():
    results = {"B": B, "statistic": "ratio of group medians (uncapped/baseline over capped/treated)",
               "resampling": "independent within-arm, seed-level", "entries": []}
    E = results["entries"]

    # 1. primary S5 dose-response (10.8x headline) + full k grid
    s5 = {}
    for k in ["kinf", "k3", "k2", "k1p5", "k1"]:
        s5[k], ng, _ = grok_steps(os.path.join(RES, "s5_normctl"), f"{k}_s*.jsonl")
        assert ng == 0 and len(s5[k]) == 8, (k, ng, len(s5[k]))
    for k in ["k3", "k2", "k1p5", "k1"]:
        E.append(entry(f"S5 primary: kinf/{k}", s5["kinf"], s5[k],
                       "results/s5_normctl, 8 seeds/arm; paper headline 10.8x at k=1"))

    # 2. mod-add replication (eval-floored)
    add = {}
    for k in ["kinf", "k1"]:
        add[k], ng, _ = grok_steps(os.path.join(RES, "add_normctl"), f"{k}_s*.jsonl")
        assert ng == 0, (k, ng)
    E.append(entry("mod-add primary: kinf/k1", add["kinf"], add["k1"],
                   "results/add_normctl; eval-floored (all medians at step 100)"))

    # 3. width x depth grid (15.9x .. 38.2x)
    for d in [128, 256, 512]:
        for L in [2, 4]:
            b, ngb, _ = grok_steps(os.path.join(RES, "s5_normctl_scale"),
                                   f"capwd_d{d}_L{L}_kinf_muon_s*.jsonl")
            t, ngt, _ = grok_steps(os.path.join(RES, "s5_normctl_scale"),
                                   f"capwd_d{d}_L{L}_k1_muon_s*.jsonl")
            assert ngb == 0 and ngt == 0
            E.append(entry(f"S5 grid d={d} L={L}: kinf/k1", b, t,
                           "results/s5_normctl_scale capwd, 5 seeds/arm (replication batch)"))

    # 4. group-ladder tasks A5 / D60 (1.76x, 1.33x)
    for task, note in [("A5", "rung-60, 8 seeds"), ("D60", "rung-120, 5 seeds")]:
        b, ngb, _ = grok_steps(os.path.join(RES, "ieee_gap_20260705", "A", "normaccel"),
                               f"{task}_kinf_s*.jsonl")
        t, ngt, _ = grok_steps(os.path.join(RES, "ieee_gap_20260705", "A", "normaccel"),
                               f"{task}_k1_s*.jsonl")
        assert ngb == 0 and ngt == 0
        E.append(entry(f"{task}: kinf/k1", b, t,
                       f"results/ieee_gap_20260705/A/normaccel, {note}"))

    # 5. depth-4 S5 replication
    b, ngb, _ = grok_steps(os.path.join(RES, "ieee_gap_20260705", "A", "depth4"),
                           "L4_kinf_s*.jsonl")
    t, ngt, _ = grok_steps(os.path.join(RES, "ieee_gap_20260705", "A", "depth4"),
                           "L4_k1_s*.jsonl")
    assert ngb == 0 and ngt == 0
    E.append(entry("S5 depth-4 (gap battery): kinf/k1", b, t,
                   "results/ieee_gap_20260705/A/depth4, 5 seeds/arm"))

    # 6. sphere control: S5 Muon sphere vs uncapped Muon (cross-batch: normmatch vs s5_normctl)
    sph, ngs, _ = grok_steps(os.path.join(RES, "normmatch"), "match_s5_muon_s*.jsonl")
    assert ngs == 0
    E.append(entry("S5 sphere-Muon: uncapped kinf / sphere", s5["kinf"], sph,
                   "sphere arm results/normmatch (8 seeds) vs s5_normctl kinf (8 seeds); cross-batch"))

    # 7. sphere control: mod-add AdamW sphere vs unconstrained AdamW wd=0.01 (grid_main)
    sph_a, ngs, _ = grok_steps(os.path.join(RES, "normmatch"), "match_add_adamw_s*.jsonl")
    base_a, ngb, _ = grok_steps(os.path.join(RES, "grid_main"),
                                "adamw_wd0.01_sc1.0_tf0.4_s*.jsonl")
    E.append(entry("mod-add sphere-AdamW: unconstrained / sphere", base_a, sph_a,
                   f"sphere results/normmatch (n={len(sph_a)}, {ngs} non-grok) vs "
                   f"grid_main adamw wd=0.01 (n={len(base_a)}, {ngb} non-grok); cross-batch"))

    # ---- batch discrepancy: 10.8x (s5_normctl, 8 seeds) vs 15.9x (capwd d128 L2, 5 seeds)
    b2, _, _ = grok_steps(os.path.join(RES, "s5_normctl_scale"), "capwd_d128_L2_kinf_muon_s*.jsonl")
    t2, _, _ = grok_steps(os.path.join(RES, "s5_normctl_scale"), "capwd_d128_L2_k1_muon_s*.jsonl")
    unc1 = np.array([s5["kinf"][k] for k in sorted(s5["kinf"])])
    unc2 = np.array([b2[k] for k in sorted(b2)])
    cap1 = np.array([s5["k1"][k] for k in sorted(s5["k1"])])
    cap2 = np.array([t2[k] for k in sorted(t2)])
    disc = {
        "batch1_uncapped_per_seed": unc1.tolist(), "batch2_uncapped_per_seed": unc2.tolist(),
        "batch1_capped_per_seed": cap1.tolist(), "batch2_capped_per_seed": cap2.tolist(),
        "batch1_ratio": float(np.median(unc1) / np.median(cap1)),
        "batch2_ratio": float(np.median(unc2) / np.median(cap2)),
    }
    if HAVE_SCIPY:
        u_unc = mannwhitneyu(unc1, unc2, alternative="two-sided")
        u_cap = mannwhitneyu(cap1, cap2, alternative="two-sided")
        disc["mannwhitney_uncapped"] = {"U": float(u_unc.statistic), "p": float(u_unc.pvalue)}
        disc["mannwhitney_capped"] = {"U": float(u_cap.statistic), "p": float(u_cap.pvalue)}
    # variance decomposition on log10 uncapped grok steps (one-way, unbalanced)
    g = [np.log10(unc1), np.log10(unc2)]
    allv = np.concatenate(g)
    grand = allv.mean()
    ss_between = sum(len(x) * (x.mean() - grand) ** 2 for x in g)
    ss_within = sum(((x - x.mean()) ** 2).sum() for x in g)
    disc["variance_decomposition_log10_uncapped"] = {
        "ss_between": float(ss_between), "ss_within": float(ss_within),
        "frac_between": float(ss_between / (ss_between + ss_within)),
    }
    # bootstrap CI for the difference of the two batch ratios
    r1o, r1pct, _ = bootstrap_ratio(unc1, cap1)
    r2o, r2pct, _ = bootstrap_ratio(unc2, cap2)
    bs1 = np.median(unc1[RNG.integers(0, len(unc1), (B, len(unc1)))], axis=1) / \
          np.median(cap1[RNG.integers(0, len(cap1), (B, len(cap1)))], axis=1)
    bs2 = np.median(unc2[RNG.integers(0, len(unc2), (B, len(unc2)))], axis=1) / \
          np.median(cap2[RNG.integers(0, len(cap2), (B, len(cap2)))], axis=1)
    diff = bs2 - bs1
    disc["ratio_batch1_ci95_pct"] = r1pct
    disc["ratio_batch2_ci95_pct"] = r2pct
    disc["ratio_diff_batch2_minus_batch1_ci95_pct"] = (
        float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
    results["batch_discrepancy_10p8_vs_15p9"] = disc

    # ---- config audit: unique operating points across all arms used above
    cfgs = {}
    for d, pat in [("s5_normctl", "*.jsonl"), ("add_normctl", "*.jsonl"),
                   ("s5_normctl_scale", "*.jsonl"), ("normmatch", "*.jsonl"),
                   ("grid_main", "*.jsonl"), ("wd_sweep", "*.jsonl"),
                   (os.path.join("ieee_gap_20260705", "A", "normlaw"), "*.jsonl"),
                   (os.path.join("ieee_gap_20260705", "A", "depth4"), "*.jsonl")]:
        for path in sorted(glob.glob(os.path.join(RES, d, pat)))[:400]:
            with open(path) as f:
                first = f.readline()
            try:
                m = json.loads(first).get("_meta", {})
            except json.JSONDecodeError:
                continue
            key = (m.get("optimizer"), m.get("lr"), m.get("muon_lr"),
                   m.get("weight_decay"), m.get("steps"), m.get("eval_every"))
            cfgs.setdefault(str(key), set()).add(d)
    results["config_audit_optimizer_lr_muonlr_wd_steps_evalevery"] = {
        k: sorted(v) for k, v in sorted(cfgs.items())}

    with open(os.path.join(OUT, "bootstrap_cis.json"), "w") as f:
        json.dump(results, f, indent=1)

    # CSV summary
    with open(os.path.join(OUT, "bootstrap_cis.csv"), "w") as f:
        f.write("name,n_base,n_treat,median_base,median_treat,ratio,ci95_lo_pct,ci95_hi_pct,ci95_lo_bca,ci95_hi_bca\n")
        for e in E:
            bca = e["ci95_bca"] or (float("nan"), float("nan"))
            f.write(f"\"{e['name']}\",{e['n_base']},{e['n_treat']},{e['median_base']},"
                    f"{e['median_treat']},{e['ratio']:.3f},{e['ci95_percentile'][0]:.3f},"
                    f"{e['ci95_percentile'][1]:.3f},{bca[0]:.3f},{bca[1]:.3f}\n")

    for e in E:
        bca = e["ci95_bca"]
        print(f"{e['name']:45s} n={e['n_base']}/{e['n_treat']} "
              f"med {e['median_base']:.0f}/{e['median_treat']:.0f} "
              f"ratio {e['ratio']:6.2f}  pct[{e['ci95_percentile'][0]:6.2f},{e['ci95_percentile'][1]:6.2f}]"
              + (f"  BCa[{bca[0]:6.2f},{bca[1]:6.2f}]" if bca else ""))
    print("\nBatch discrepancy:", json.dumps(disc, indent=1)[:1500])
    print("\nConfig audit:")
    for k, v in results["config_audit_optimizer_lr_muonlr_wd_steps_evalevery"].items():
        print(" ", k, "->", v)


if __name__ == "__main__":
    main()

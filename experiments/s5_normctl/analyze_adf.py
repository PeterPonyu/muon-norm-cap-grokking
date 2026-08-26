#!/usr/bin/env python3
"""Analyze A-DF with the seed-level ratio-of-medians bootstrap."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

try:
    from scipy.stats import norm as scipy_norm
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "revision2026" / "gpu2026" / "adf"


def load_runs(root):
    rows = []
    for path in sorted(Path(root).glob("adf_*_k*_s*.jsonl")):
        m = re.match(r"adf_(add|D60)_(kinf|k1)_s(\d+)\.jsonl$", path.name)
        if not m:
            continue
        summary = None
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_summary" in rec:
                summary = rec["_summary"]
        if summary is not None:
            rows.append({"task": m.group(1), "cap": m.group(2),
                         "seed": int(m.group(3)), "path": str(path),
                         "summary": summary})
    return rows


def validate_grid(rows, root, require_complete=True):
    man_path = Path(root) / "MANIFEST.json"
    if not man_path.exists():
        if require_complete:
            raise ValueError(f"missing {man_path}")
        return {"complete": False, "missing": [], "extra": [], "duplicates": []}
    manifest = json.loads(man_path.read_text())
    expected = {(r["task"], "kinf" if r["ceiling_k"] == "inf" else "k1", int(r["seed"]))
                for r in manifest.get("runs", [])}
    seen = [(r["task"], r["cap"], r["seed"]) for r in rows]
    duplicates = sorted({k for k in seen if seen.count(k) > 1})
    actual = set(seen)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    complete = not missing and not extra and not duplicates and len(rows) == len(expected)
    if require_complete and not complete:
        raise ValueError(f"incomplete A-DF grid: missing={len(missing)} extra={len(extra)} duplicates={len(duplicates)}")
    return {"complete": complete, "missing": missing, "extra": extra,
            "duplicates": duplicates, "expected_n": len(expected)}


def ratio_of_medians(base, treat):
    return float(np.median(base) / np.median(treat))


def _erfinv(x):
    a = 0.147
    ln = math.log(1 - x * x)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), x)


def bootstrap_ratio(base, treat, n_resamples=20000, seed=20260726):
    base, treat = np.asarray(base, float), np.asarray(treat, float)
    obs = ratio_of_medians(base, treat)
    rng = np.random.default_rng(seed)
    nb, nt = len(base), len(treat)
    bs = (np.median(base[rng.integers(0, nb, (n_resamples, nb))], axis=1) /
          np.median(treat[rng.integers(0, nt, (n_resamples, nt))], axis=1))
    pct = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
    bca = None
    if nb >= 2 and nt >= 2:
        try:
            if HAVE_SCIPY:
                ppf, cdf = scipy_norm.ppf, scipy_norm.cdf
            else:
                ppf = lambda p: math.sqrt(2) * _erfinv(2 * p - 1)
                cdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
            prop = np.clip(np.mean(bs < obs) + 0.5 * np.mean(bs == obs),
                           1e-6, 1 - 1e-6)
            z0 = ppf(prop)
            jack = [ratio_of_medians(np.delete(base, i), treat) for i in range(nb)]
            jack += [ratio_of_medians(base, np.delete(treat, i)) for i in range(nt)]
            jack = np.asarray(jack)
            jm = jack.mean()
            den = 6 * np.sum((jm - jack) ** 2) ** 1.5
            acc = np.sum((jm - jack) ** 3) / den if den else 0.0
            qs = []
            for alpha in (0.025, 0.975):
                za = ppf(alpha)
                qs.append(cdf(z0 + (z0 + za) / (1 - acc * (z0 + za))))
            bca = [float(np.percentile(bs, 100 * q)) for q in qs]
        except Exception:
            bca = None
    return {"ratio": obs, "ci95_percentile": pct, "ci95_bca": bca}


def summarize(rows, bootstrap_resamples=20000):
    report = {"n_runs": len(rows), "tasks_seen": sorted({r["task"] for r in rows}),
              "bootstrap_resamples": bootstrap_resamples, "entries": []}
    for task in ("add", "D60"):
        arms = {}
        for cap in ("kinf", "k1"):
            vals = {r["seed"]: r["summary"].get("grok_step") for r in rows
                    if r["task"] == task and r["cap"] == cap}
            arms[cap] = {s: v for s, v in vals.items() if v is not None}
        entry = {"task": task, "n_uncapped": len(arms["kinf"]),
                 "n_capped": len(arms["k1"]),
                 "uncapped_per_seed": arms["kinf"], "capped_per_seed": arms["k1"],
                 "nongrok_uncapped": sum(r["task"] == task and r["cap"] == "kinf" and
                                          r["summary"].get("grok_step") is None for r in rows),
                 "nongrok_capped": sum(r["task"] == task and r["cap"] == "k1" and
                                        r["summary"].get("grok_step") is None for r in rows)}
        if (len(arms["kinf"]) == 8 and len(arms["k1"]) == 8 and
                entry["nongrok_uncapped"] == 0 and entry["nongrok_capped"] == 0 and
                set(arms["kinf"]) == set(range(70, 78)) and
                set(arms["k1"]) == set(range(70, 78))):
            base = [arms["kinf"][s] for s in sorted(arms["kinf"])]
            treat = [arms["k1"][s] for s in sorted(arms["k1"])]
            entry.update({"median_uncapped": float(np.median(base)),
                          "median_capped": float(np.median(treat)),
                          "inference_status": "complete_uncensored"})
            entry.update(bootstrap_ratio(base, treat, bootstrap_resamples,
                                         seed=20260726 + (task == "D60")))
        else:
            entry["inference_status"] = "withheld_incomplete_or_censored"
        report["entries"].append(entry)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="diagnostic only: report partial cells but withhold inference")
    args = ap.parse_args()
    rows = load_runs(args.root)
    grid = validate_grid(rows, args.root, require_complete=not args.allow_incomplete)
    report = summarize(rows, args.bootstrap)
    report["grid_validation"] = grid
    out = args.out or args.root / "analysis_adf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"A-DF: {report['n_runs']} completed runs")
    for e in report["entries"]:
        if "ratio" not in e:
            print(f"  {e['task']}: incomplete ({e['n_uncapped']}/{e['n_capped']})")
            continue
        print(f"  {e['task']}: n={e['n_uncapped']}/{e['n_capped']} "
              f"med={e['median_uncapped']:g}/{e['median_capped']:g} "
              f"ratio={e['ratio']:.3f} pct={e['ci95_percentile']} BCa={e['ci95_bca']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

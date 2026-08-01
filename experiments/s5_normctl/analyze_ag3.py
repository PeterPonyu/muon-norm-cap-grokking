#!/usr/bin/env python3
"""A-G3 readout — does any update-geometry control reproduce Muon's grokking?

The arm substitutes Muon's orthogonalized update UV^T with transforms that keep
some of its properties and break others, at a fixed param set (see
run_ag3_update_control.py's PARAM-SET DISCIPLINE note). This script reads the
resulting grok steps and reports the preregistered comparisons.

PREREGISTERED READING (stated in the runner's docstring before any data existed):

  specflat ~ muon      POSITIVE control. specflat is exact-SVD spectrum
                       flattening, which is what Newton-Schulz approximates, so
                       it MUST reproduce muon. If it does not, the arm is not
                       measuring update geometry and nothing else here is
                       interpretable.
  randorth fails/slow  NEGATIVE control. A fresh random semi-orthogonal matrix
                       each step, rescaled to the NS update's Frobenius norm:
                       same energy per step, zero gradient alignment. If this
                       groks like muon, the mechanism is energy injection, not
                       geometry.
  specinv separates    Gradient-aligned but anti-spectrally-weighted (step budget
                       into the gradient's WEAKEST directions). Separates "aligned
                       with the gradient" from "flattens the spectrum".

Grok step is read from each run's `_summary.grok_step`; `null` means the run never
reached the grok threshold inside its step budget, which is a RIGHT-CENSORED
observation at `steps`, not a missing value. Medians are therefore reported with
the censored count beside them, and a median is called censored when more than
half the arm never groks.

Usage
-----
    python3 analyze_ag3.py
    python3 analyze_ag3.py --root ../revision2026/gpu2026/ag3
    python3 analyze_ag3.py --json
    python3 analyze_ag3.py --allow-incomplete    # read a partial arm mid-flight
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "revision2026" / "gpu2026" / "ag3"

# Display order is the argument order, not alphabetical: reference, positive
# control, then the two probes that carry the actual dissociation.
ARM_ORDER = ["muon", "specflat", "randorth", "specinv", "adamw"]
ARM_ROLE = {
    "muon":     "reference (Newton-Schulz)",
    "specflat": "positive control (exact SVD, flat spectrum)",
    "randorth": "negative control (random semi-orth, energy-matched)",
    "specinv":  "probe (gradient-aligned, spectrum inverted)",
    "adamw":    "between-family reference (single AdamW)",
}


def tail_summary(path: Path) -> dict | None:
    """Return the `_summary` dict, or None if the run never closed its file.

    Read from the tail rather than parsing every line: these logs carry one
    eval record per 50 steps over a 20k budget, so the body is large and
    entirely irrelevant to this readout.
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.strip().split("\n")):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_summary" in rec:
            return rec["_summary"]
    return None


def median_censored(values: list, censored: int, budget: int):
    """Median of grok steps where `censored` runs never groked.

    Censored runs are known to exceed every observed value, so they sort to the
    top. If they occupy the upper half, the median itself is censored and no
    finite value is defensible — report the bound instead of silently dropping
    them, which would bias the median downward exactly in the arms designed not
    to grok.
    """
    n = len(values) + censored
    if n == 0:
        return None, False
    ordered = sorted(values) + [budget + 1] * censored
    mid = n // 2
    if n % 2:
        val = ordered[mid]
    else:
        lo, hi = ordered[mid - 1], ordered[mid]
        val = (lo + hi) / 2
    is_censored = val > budget
    return (None if is_censored else val), is_censored


def collect(root: Path, allow_incomplete: bool) -> dict:
    """Walk MANIFEST.json and read each planned run's summary."""
    man_path = root / "MANIFEST.json"
    if not man_path.exists():
        raise SystemExit(f"no MANIFEST.json under {root} — has the arm been planned?")
    manifest = json.loads(man_path.read_text())
    runs = manifest.get("runs") or []

    cells, absent, partial = [], [], []
    for run in runs:
        path = root / run["file"]
        if not path.exists():
            absent.append(run["file"])
            continue
        summary = tail_summary(path)
        if summary is None:
            partial.append(run["file"])
            continue
        cells.append({
            "op": run["op"], "arm": run["arm"], "seed": run["seed"],
            "steps": run["steps"],
            "grok_step": summary.get("grok_step"),
            "memorize_step": summary.get("memorize_step"),
            "final_test_acc": summary.get("final_test_acc"),
            "final_wn_hidden": summary.get("final_wn_hidden"),
            "stopped_step": summary.get("stopped_step"),
        })

    if (absent or partial) and not allow_incomplete:
        raise SystemExit(
            f"arm is incomplete: {len(absent)} absent, {len(partial)} partial "
            f"(of {len(runs)} planned).\nRe-run the queue, or pass "
            f"--allow-incomplete to read what is on disk.")
    return {"manifest": manifest, "cells": cells,
            "absent": absent, "partial": partial, "n_planned": len(runs)}


def summarize(data: dict) -> dict:
    """Per (op, arm) grok-step medians with censoring accounted for."""
    cells = data["cells"]
    ops = [o for o in (data["manifest"].get("ops") or []) ] or \
          sorted({c["op"] for c in cells})
    out = {}
    for op in ops:
        per_arm = {}
        for arm in ARM_ORDER:
            sel = [c for c in cells if c["op"] == op and c["arm"] == arm]
            if not sel:
                continue
            budget = sel[0]["steps"]
            groked = [c["grok_step"] for c in sel if c["grok_step"] is not None]
            censored = len(sel) - len(groked)
            med, med_censored = median_censored(groked, censored, budget)
            per_arm[arm] = {
                "n": len(sel), "n_grok": len(groked), "n_censored": censored,
                "budget": budget,
                "median_grok_step": med,
                "median_is_censored": med_censored,
                "grok_steps": sorted(groked),
                "final_test_acc_median": _median([
                    c["final_test_acc"] for c in sel
                    if c["final_test_acc"] is not None]),
                "final_wn_hidden_median": _median([
                    c["final_wn_hidden"] for c in sel
                    if c["final_wn_hidden"] is not None]),
            }
        out[op] = per_arm
    return out


def _median(xs: list):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# The positive control passes when specflat's grok rate matches muon's and its
# median sits within this factor. NS-5 is a five-step quintic in bfloat16 that
# only pushes singular values toward 1, so it approximates exact flattening
# rather than reproducing it (measured cos 0.9752-0.9999 across conditioning;
# see LESSONS-AND-ERRATA 6j). Demanding tight agreement in grok STEP would be
# the same category error as that round's over-strict assert.
POSCTRL_FACTOR = 2.0


def adjudicate(summary: dict) -> list[dict]:
    """Apply the preregistered reading to each op. Never softens a failure."""
    verdicts = []
    for op, arms in summary.items():
        muon = arms.get("muon")
        v = {"op": op, "checks": [], "positive_control": None}
        if not muon:
            v["checks"].append({"name": "muon reference", "verdict": "ABSENT",
                                "detail": "no muon runs — nothing is comparable"})
            verdicts.append(v)
            continue

        # --- positive control: specflat must reproduce muon -----------------
        sf = arms.get("specflat")
        if sf is None:
            v["positive_control"] = "ABSENT"
            v["checks"].append({
                "name": "positive control (specflat ~ muon)", "verdict": "ABSENT",
                "detail": "specflat missing; the arm cannot be validated"})
        else:
            rate_match = sf["n_grok"] == muon["n_grok"]
            both_med = (sf["median_grok_step"] is not None
                        and muon["median_grok_step"] is not None)
            if both_med:
                ratio = sf["median_grok_step"] / muon["median_grok_step"]
                within = (1 / POSCTRL_FACTOR) <= ratio <= POSCTRL_FACTOR
            else:
                ratio, within = None, (sf["median_is_censored"]
                                       == muon["median_is_censored"])
            ok = rate_match and within
            v["positive_control"] = "PASS" if ok else "FAIL"
            v["checks"].append({
                "name": "positive control (specflat ~ muon)",
                "verdict": "PASS" if ok else "FAIL",
                "detail": (f"grok {sf['n_grok']}/{sf['n']} vs muon "
                           f"{muon['n_grok']}/{muon['n']}"
                           + (f"; median ratio {ratio:.2f}x (bar {POSCTRL_FACTOR}x)"
                              if ratio is not None else "; medians censored")),
                "blocking": not ok})

        # --- negative control: randorth must NOT match muon ----------------
        ro = arms.get("randorth")
        if ro is not None:
            matches = (ro["n_grok"] == muon["n_grok"]
                       and ro["median_grok_step"] is not None
                       and muon["median_grok_step"] is not None
                       and ro["median_grok_step"] <= POSCTRL_FACTOR
                       * muon["median_grok_step"])
            v["checks"].append({
                "name": "negative control (randorth != muon)",
                "verdict": "MECHANISM IS GEOMETRY" if not matches
                           else "MECHANISM IS ENERGY INJECTION",
                "detail": (f"randorth grok {ro['n_grok']}/{ro['n']}"
                           + (f", median {ro['median_grok_step']:.0f}"
                              if ro["median_grok_step"] is not None
                              else f", never groks within {ro['budget']}")
                           + f" vs muon {muon['n_grok']}/{muon['n']}"
                           + (f", median {muon['median_grok_step']:.0f}"
                              if muon["median_grok_step"] is not None else ""))})

        # --- probe: specinv separates alignment from flattening ------------
        si = arms.get("specinv")
        if si is not None:
            v["checks"].append({
                "name": "probe (specinv: aligned but anti-spectral)",
                "verdict": ("ALIGNMENT SUFFICES" if si["n_grok"] == si["n"]
                            else "FLATTENING REQUIRED" if si["n_grok"] == 0
                            else "PARTIAL"),
                "detail": (f"specinv grok {si['n_grok']}/{si['n']}"
                           + (f", median {si['median_grok_step']:.0f}"
                              if si["median_grok_step"] is not None
                              else f", never groks within {si['budget']}"))})
        verdicts.append(v)
    return verdicts


def render(data: dict, summary: dict, verdicts: list[dict]) -> str:
    L = []
    L.append(f"A-G3 update-geometry readout — {len(data['cells'])}"
             f"/{data['n_planned']} runs read")
    if data["absent"] or data["partial"]:
        L.append(f"  INCOMPLETE: {len(data['absent'])} absent, "
                 f"{len(data['partial'])} partial — numbers below are provisional")
    L.append("")

    for op, arms in summary.items():
        L.append(f"=== op {op} ===")
        L.append(f"  {'arm':<9} {'role':<48} {'grok':>7} {'median':>9}  seeds")
        for arm in ARM_ORDER:
            a = arms.get(arm)
            if a is None:
                continue
            med = ("censored" if a["median_is_censored"]
                   else f"{a['median_grok_step']:.0f}")
            steps = ",".join(str(s) for s in a["grok_steps"]) or "-"
            L.append(f"  {arm:<9} {ARM_ROLE[arm]:<48} "
                     f"{a['n_grok']}/{a['n']:<5} {med:>9}  {steps}")
        L.append("")

    L.append("=== preregistered verdicts ===")
    for v in verdicts:
        L.append(f"  op {v['op']}:")
        for c in v["checks"]:
            mark = "!!" if c.get("blocking") else "  "
            L.append(f"  {mark} {c['name']}: {c['verdict']}")
            L.append(f"       {c['detail']}")
    blocking = [c for v in verdicts for c in v["checks"] if c.get("blocking")]
    unvalidated = [v["op"] for v in verdicts
                   if v["positive_control"] in (None, "ABSENT")]
    L.append("")
    if blocking:
        L.append("  POSITIVE CONTROL FAILED — specflat did not reproduce muon.")
        L.append("  The arm is not measuring update geometry; do NOT interpret")
        L.append("  the randorth/specinv verdicts until this is explained.")
    elif unvalidated:
        L.append("  POSITIVE CONTROL NOT EVALUATED for op(s): "
                 + ", ".join(unvalidated) + ".")
        L.append("  Nothing here is interpretable yet — the reference and/or the")
        L.append("  specflat control have not landed.")
    else:
        L.append("  Positive control holds; the geometry comparisons are readable.")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="A-G3 readout: does any update-geometry control reproduce Muon?")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"arm results dir (default {DEFAULT_ROOT})")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="read a partially-complete arm instead of exiting")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--out", type=Path, default=None,
                    help="also write the JSON verdict here")
    args = ap.parse_args()

    data = collect(args.root, args.allow_incomplete)
    summary = summarize(data)
    verdicts = adjudicate(summary)
    payload = {"root": str(args.root), "n_planned": data["n_planned"],
               "n_read": len(data["cells"]), "absent": data["absent"],
               "partial": data["partial"], "per_op": summary,
               "verdicts": verdicts, "posctrl_factor": POSCTRL_FACTOR}

    if args.json:
        print(json.dumps(payload, indent=1))
    else:
        print(render(data, summary, verdicts))
    if args.out:
        args.out.write_text(json.dumps(payload, indent=1))
        if not args.json:
            print(f"\n  wrote {args.out}")

    blocking = [c for v in verdicts for c in v["checks"] if c.get("blocking")]
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""A-DF -- defloor the two unresolved acceleration ratios.

Two ratios are pinned to their evaluation grids: mod-add p=97 has uncapped/k=1
medians 100/100 at cadence 50, and D60 has 100/75 at cadence 25. This arm
changes only eval cadence to 5 and uses eight fresh seeds per arm:

  task in {add, D60} x cap in {inf, 1} x seed in {70..77} = 32 runs.

The add cells reuse grokking.train.run and run_s5_normctl.make_builder. D60
reuses tools/run_20260705_gapA_normaccel.py's trainer and operating point.
Companion analyze_adf.py reproduces the 20,000-resample percentile+BCa
seed bootstrap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
THIS = EXP / "s5_normctl"
TOOLS = EXP / "tools"
for p in (THIS, EXP / "grokking", EXP, TOOLS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _early_gpu_pin(argv):
    for i, arg in enumerate(argv):
        if arg == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]
        elif arg.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]


_early_gpu_pin(sys.argv)

import torch  # noqa: E402
import train as TRAIN  # noqa: E402
import run_20260705_gapA_normaccel as GAPA  # noqa: E402
from run_s5_normctl import (  # noqa: E402
    INIT_SCALE, STEPS, WD, _ktag, already_done, make_builder,
)
from runner_utils import add_shard_args, shard_cells, validate_shard_args  # noqa: E402

OUT = EXP / "revision2026" / "gpu2026" / "adf"
TASKS = ("add", "D60")
KS = (None, 1.0)
SEEDS = tuple(range(70, 78))
EVAL_EVERY = 5
RATE_SEC_PER_STEP = {"add": 0.035, "D60": 0.42}
EXPECTED_STEPS = {"add": 900, "D60": 650}


def make_cells(tasks=TASKS, ks=KS, seeds=SEEDS):
    return [(task, k, seed) for task in tasks for k in ks for seed in seeds]


def cell_name(cell):
    task, k, seed = cell
    return f"adf_{task}_{_ktag(k)}_s{seed}"


def cell_path(cell, out_dir=OUT):
    return Path(out_dir) / f"{cell_name(cell)}.jsonl"


def read_summary(path):
    summary = None
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "_summary" in rec:
                summary = rec["_summary"]
    return summary


def make_add_cfg(seed, **overrides):
    kw = dict(op="add", p=97, d_model=128, n_layers=2,
              optimizer="muon", lr=1e-3, muon_lr=0.02,
              init_scale=INIT_SCALE, weight_decay=WD, train_frac=0.4,
              seed=seed, steps=STEPS, eval_every=EVAL_EVERY, mech=True)
    kw.update(overrides)
    return TRAIN.Config(**kw)


def run_add(k, seed, path, **overrides):
    TRAIN.build_optimizer = make_builder(k)
    return TRAIN.run(make_add_cfg(seed, **overrides), out_path=str(path))


def run_d60(k, seed, path, steps=STEPS):
    GAPA.RG.build_optimizer = make_builder(k)
    saved = GAPA.RG.EVAL_EVERY
    GAPA.RG.EVAL_EVERY = EVAL_EVERY
    try:
        return GAPA.RG.run_cell(
            "D60", "muon", seed, d_model=GAPA.D_MODEL,
            train_frac=GAPA.TRAIN_FRAC, steps=steps,
            weight_decay=WD, out_path=str(path))
    finally:
        GAPA.RG.EVAL_EVERY = saved


def run_cell(cell, out_dir=OUT, **overrides):
    task, k, seed = cell
    path = cell_path(cell, out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if task == "add":
        return run_add(k, seed, path, **overrides)
    return run_d60(k, seed, path, steps=overrides.get("steps", STEPS))


def estimate_gpu_hours(cells):
    seconds = sum(RATE_SEC_PER_STEP[t] * EXPECTED_STEPS[t] for t, _, _ in cells)
    ceiling = sum(RATE_SEC_PER_STEP[t] * STEPS for t, _, _ in cells)
    return {"expected": round(seconds / 3600, 2),
            "ceiling": round(ceiling / 3600, 2)}


def write_manifest(out_dir=OUT):
    cells = make_cells()
    runs = [{"name": cell_name(c), "file": cell_name(c) + ".jsonl",
             "task": c[0], "ceiling_k": "inf" if c[1] is None else c[1],
             "seed": c[2], "steps": STEPS, "eval_every": EVAL_EVERY}
            for c in cells]
    manifest = {
        "arm": "A-DF", "n_runs": len(runs), "runs": runs,
        "purpose": "defloor mod-add and D60 ratios whose prior CI lower bound touched 1.0",
        "tasks": list(TASKS), "caps": ["inf", 1.0],
        "seed_block": [SEEDS[0], SEEDS[-1]],
        "seed_rationale": "70-77 disjoint from prior A arms (0-59) and A-G3 (60-64)",
        "eval_every": EVAL_EVERY,
        "d60_reuse": "tools/run_20260705_gapA_normaccel.py trainer and operating point",
        "add_reuse": "grokking.train.run + run_s5_normctl.make_builder",
        "bootstrap": {"resamples": 20000, "unit": "seed",
                      "intervals": ["percentile", "BCa"]},
        "est_gpu_hours": estimate_gpu_hours(cells),
        "sentinel": "_summary in jsonl tail",
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def run_smoke():
    import shutil
    import analyze_adf as AN

    torch.set_num_threads(1)
    tmp = Path(tempfile.mkdtemp(prefix="adf_smoke_"))
    try:
        man = write_manifest(tmp)
        assert man["n_runs"] == 32 and man["eval_every"] == 5

        add_cell = ("add", 1.0, SEEDS[0])
        _, h_add = run_cell(add_cell, tmp, p=7, d_model=32, steps=20,
                            eval_every=5, device="cpu")
        assert len(h_add) == 5 and already_done(str(cell_path(add_cell, tmp)))
        assert read_summary(cell_path(add_cell, tmp))["eval_every"] == 5

        d_cell = ("D60", None, SEEDS[0])
        s_d, h_d = run_cell(d_cell, tmp, steps=10)
        assert len(h_d) == 3 and GAPA.RG.already_done(str(cell_path(d_cell, tmp)))

        partial = tmp / "partial.jsonl"
        partial.write_text(json.dumps({"_meta": {}}) + "\n")
        assert not already_done(str(partial))

        rows = AN.load_runs(tmp)
        grid = AN.validate_grid(rows, tmp, require_complete=False)
        report = AN.summarize(rows, bootstrap_resamples=200)
        assert len(rows) == 2 and report["n_runs"] == 2
        assert not grid["complete"] and len(grid["missing"]) == 30
        assert set(report["tasks_seen"]) == {"add", "D60"}
        print("SMOKE PASS: A-DF manifest=32; cap injection reused unchanged; "
              f"add cadence-5 jsonl ({len(h_add)} evals); "
              f"D60 reused trainer cadence-5 jsonl ({len(h_d)} evals, test={s_d['final_test_acc']:.3f}); "
              "resume rejects partial; analyzer read-back; zero writes outside tmp")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on cell name")
    ap.add_argument("--gpu", default=None,
                    help="CUDA id; absent means inherit CUDA_VISIBLE_DEVICES")
    add_shard_args(ap)
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return

    cells = make_cells()
    manifest = write_manifest(OUT)
    if args.only:
        cells = [c for c in cells if args.only in cell_name(c)]
    validate_shard_args(args)
    cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for cell in cells:
            print(cell_name(cell))
        est = manifest["est_gpu_hours"]
        print(f"\n{len(cells)} runs selected (full grid 32) | tasks={list(TASKS)} "
              f"caps=inf,1 seeds={SEEDS[0]}-{SEEDS[-1]} eval_every={EVAL_EVERY}")
        print(f"est {est['expected']:.2f} GPU-h expected, "
              f"{est['ceiling']:.2f} GPU-h all-full-budget ceiling")
        print(f"-> {OUT}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for i, cell in enumerate(cells, 1):
        path = cell_path(cell)
        if already_done(str(path)):
            print(f"[{i}/{len(cells)}] skip {path.name}", flush=True)
            continue
        t0 = time.time()
        summary, _ = run_cell(cell)
        print(f"[{i}/{len(cells)}] {path.name}: grok={summary.get('grok_step')} "
              f"test={summary['final_test_acc']:.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    print("[adf] DONE; run analyze_adf.py --root " + str(OUT), flush=True)


if __name__ == "__main__":
    main()

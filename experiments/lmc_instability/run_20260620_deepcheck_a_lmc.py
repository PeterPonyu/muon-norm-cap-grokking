#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "lmc_instability"
sys.path.insert(0, str(EXP))
sys.path.append(str(ROOT / "experiments"))
from train_fork import Config, run  # noqa:E402
from run_lmc import SPAWN_STEPS, N_CHILDREN, _cell_name  # noqa:E402

OUT = ROOT / "experiments/results/ultragoal_20260620_deepcheck/a_lmc_short"


def archive_incomplete(p: Path, reason: str) -> None:
    target = p.with_name(f"{p.name}.{reason}.{int(time.time())}")
    p.rename(target)
    print(f"archived incomplete evidence {p} -> {target}", flush=True)


def done(p: Path):
    if not p.exists():
        return None
    last = ""
    for line in p.read_text().splitlines():
        if line.strip():
            last = line
    if not last:
        archive_incomplete(p, "empty")
        return None
    try:
        obj = json.loads(last)
    except json.JSONDecodeError:
        archive_incomplete(p, "corrupt")
        return None
    summary = obj.get("_summary")
    if summary is None:
        archive_incomplete(p, "partial")
    return summary


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizers", default="muon,adamw")
    ap.add_argument("--seeds", default="10,11")
    ap.add_argument("--short", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ckpt").mkdir(exist_ok=True)
    spawn = list(SPAWN_STEPS)
    nchild = N_CHILDREN
    if a.short:
        spawn = spawn[:3]
        nchild = min(3, N_CHILDREN)
    summaries = []
    cells = [
        (o.strip(), s)
        for o in a.optimizers.split(",")
        if o.strip()
        for s in parse_seeds(a.seeds)
    ]
    for i, (opt, seed) in enumerate(cells, 1):
        name = _cell_name(opt, seed)
        p = OUT / (name + ".jsonl")
        s = done(p)
        if s is not None:
            print(f"[{i}/{len(cells)}] skip {name}", flush=True)
            summaries.append(s)
            continue
        print(
            f"[{i}/{len(cells)}] run {name} spawn={spawn} nchild={nchild}", flush=True
        )
        t = time.time()
        cfg = Config(optimizer=opt, seed=seed, spawn_steps=spawn, n_children=nchild)
        s, _ = run(cfg, out_path=str(p), ckpt_dir=str(OUT / "ckpt" / name))
        s["elapsed_wall_sec"] = time.time() - t
        summaries.append(s)
        print(
            json.dumps(
                {
                    "name": name,
                    "k_star": s.get("k_star"),
                    "elapsed": s["elapsed_wall_sec"],
                }
            ),
            flush=True,
        )
    (OUT / "summaries.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=True) + "\n"
    )


if __name__ == "__main__":
    main()

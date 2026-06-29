#!/usr/bin/env python3
"""G003 permuted-MNIST low-lr Muon seed extension cell runner."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')

ROOT = Path('/home/zeyufu/Desktop/dl-research')
EXP = ROOT / 'experiments' / 'muon_plasticity'
OUT = ROOT / 'experiments' / 'results' / 'muon_plasticity_perm_mnist_ext'
if str(EXP) in sys.path:
    sys.path.remove(str(EXP))
sys.path.insert(0, str(EXP))
from run_perm_mnist import base_cfg  # noqa: E402
from train_plasticity import run  # noqa: E402
try:
    import torch  # noqa: E402
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except Exception:
    pass


def lr_slug(x: float) -> str:
    return f"{x:g}".replace('.', 'p').replace('-', 'm')


def done(path: Path) -> bool:
    if not path.exists():
        return False
    last = ''
    with path.open() as f:
        for line in f:
            if line.strip():
                last = line
    try:
        return '_summary' in json.loads(last)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--muon-lr', type=float, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n-tasks', type=int, default=150)
    ap.add_argument('--steps', type=int, default=250)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"muon_mlr{lr_slug(args.muon_lr)}_s{args.seed}.jsonl"
    if done(path):
        print(json.dumps({'status': 'skip', 'path': str(path), 'muon_lr': args.muon_lr, 'seed': args.seed}), flush=True)
        return 0
    if path.exists():
        path.unlink()
    cfg = base_cfg('muon', args.seed, n_tasks=args.n_tasks, steps=args.steps, muon_lr=args.muon_lr)
    summary, _ = run(cfg, out_path=str(path))
    print(json.dumps({
        'status': 'done',
        'path': str(path),
        'muon_lr': args.muon_lr,
        'seed': args.seed,
        'first_task_steps': summary['first_task_steps'],
        'last_task_steps': summary['last_task_steps'],
        'n_tasks_fit': summary['n_tasks_fit'],
        'final_feat_eff_rank': summary['final_feat_eff_rank'],
        'final_dead_frac': summary['final_dead_frac'],
    }), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

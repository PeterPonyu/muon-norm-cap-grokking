#!/usr/bin/env python3
from __future__ import annotations
import glob, json, os
from pathlib import Path
from statistics import median

ROOT=Path('/home/zeyufu/Desktop/dl-research')
R=ROOT/'experiments/results'
OUT=R/'muon_plasticity_perm_mnist_ext'
VERDICT=OUT/'perm_mnist_ext_verdict.json'

def load_task_summary(path: Path):
    tasks=[]; summary=None
    with path.open() as f:
        for line in f:
            if not line.strip(): continue
            j=json.loads(line)
            if '_summary' in j: summary=j['_summary']
            elif 'task' in j: tasks.append(j)
    if summary is None: raise RuntimeError(f'incomplete {path}')
    fits=[t['steps_to_threshold'] for t in tasks if t.get('steps_to_threshold') is not None]
    return {
      'path': str(path.relative_to(ROOT)),
      'optimizer': summary['optimizer'],
      'muon_lr': float(summary['muon_lr']),
      'seed': int(summary['seed']),
      'n_tasks_fit': int(summary['n_tasks_fit']),
      'first10_median_steps': median(fits[:10]),
      'last10_median_steps': median(fits[-10:]),
      'slowdown': median(fits[-10:]) / median(fits[:10]),
      'final_feat_eff_rank': float(summary['final_feat_eff_rank']),
      'final_dead_frac': float(summary['final_dead_frac']),
      'last_task_steps': int(summary['last_task_steps']),
    }

rows=[]
for p in sorted((R/'muon_plasticity_mnist').glob('adamw_s*.jsonl')): rows.append(load_task_summary(p))
for p in sorted((R/'muon_plasticity_mnist').glob('sgdm_s*.jsonl')): rows.append(load_task_summary(p))
for p in sorted((R/'muon_plasticity_mnist').glob('muon_s*.jsonl')): rows.append(load_task_summary(p))
for p in sorted((R/'muon_plasticity_mnist_lrctl').glob('muon_mlr0p005_s*.jsonl')): rows.append(load_task_summary(p))
for p in sorted((R/'muon_plasticity_mnist_lrctl').glob('muon_mlr0p01_s*.jsonl')): rows.append(load_task_summary(p))
for p in sorted((R/'muon_plasticity_perm_mnist_ext').glob('muon_mlr*.jsonl')): rows.append(load_task_summary(p))

def key(r):
    if r['optimizer']!='muon': return f"{r['optimizer']}_base"
    return f"muon_lr{r['muon_lr']:g}"
by={}
for r in rows: by.setdefault(key(r),[]).append(r)
agg={}
for k, vals in sorted(by.items()):
    agg[k]={
      'n': len(vals),
      'seeds': sorted(v['seed'] for v in vals),
      'median_slowdown': median(v['slowdown'] for v in vals),
      'median_final_feat_eff_rank': median(v['final_feat_eff_rank'] for v in vals),
      'median_final_dead_frac': median(v['final_dead_frac'] for v in vals),
      'median_last10_steps': median(v['last10_median_steps'] for v in vals),
      'all_tasks_fit': all(v['n_tasks_fit']==150 for v in vals),
    }
verdict={
  'new_namespace': 'experiments/results/muon_plasticity_perm_mnist_ext',
  'new_cells_complete': len(list(OUT.glob('*.jsonl'))),
  'aggregate': agg,
  'interpretation': 'Fresh low-lr Muon seeds keep high feature rank and near-zero dead units, but still slow down more than AdamW/SGDM on permuted-MNIST; synthetic plasticity extension does not transfer.',
}
VERDICT.write_text(json.dumps(verdict, indent=2)+'\n')
print(json.dumps(verdict, indent=2))

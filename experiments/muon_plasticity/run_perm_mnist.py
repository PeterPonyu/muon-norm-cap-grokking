"""Direction 005 scale-up — permuted-MNIST loss-of-plasticity benchmark.

The canonical recognized loss-of-plasticity benchmark (Dohare et al. 2024): a
long sequence of permuted-MNIST tasks (fixed images/labels, a new fixed pixel
permutation per task). We compare {AdamW, Muon, SGDM} on the SAME continual loop
and probe suite used for the synthetic 005 arms (train_plasticity.run), to test
whether the toy-scale finding (Muon EXTENDS, not cures, plasticity; weight
spectrum preserved while feature rank collapses) holds on a real benchmark.

Usage:
  python run_perm_mnist.py --smoke         # 1 opt, 3 tasks, prints, no files
  python run_perm_mnist.py                 # full grid -> results/muon_plasticity_mnist/
"""
from __future__ import annotations
import argparse, json, os, sys, time

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
from train_plasticity import Config, run  # noqa: E402

OUT = os.path.abspath(os.path.join(_THIS, "..", "results", "muon_plasticity_mnist"))
OPTIMIZERS = ["adamw", "muon", "sgdm"]
SEEDS = [0, 1, 2]


def base_cfg(opt: str, seed: int, n_tasks: int, steps: int,
             muon_lr: float = 0.02) -> Config:
    return Config(
        arm="perm_mnist", n_classes=10, n_examples=5000,
        n_tasks=n_tasks, steps_per_task=steps, batch_size=256, probe_n=512,
        acc_threshold=0.80, lam_iters=20, width=512, n_layers=3,
        optimizer=opt, lr=1e-3, muon_lr=muon_lr, weight_decay=0.0,
        warm_start=True, seed=seed, device="cuda",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_tasks", type=int, default=150)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--optimizers", nargs="+", default=OPTIMIZERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--tag", default="")  # output subdir suffix (e.g. _lrctl)
    a = ap.parse_args()

    if a.smoke:
        cfg = base_cfg("muon", 0, n_tasks=3, steps=40)
        t = time.time()
        summ, hist = run(cfg, out_path=None)
        print(f"SMOKE perm_mnist OK in {time.time()-t:.1f}s | params={summ['n_params']} "
              f"task0 steps_to_thr={hist[0]['steps_to_threshold']} "
              f"feat_eff_rank={hist[-1]['probes']['feat_eff_rank']:.2f}")
        return

    out_dir = OUT + a.tag
    os.makedirs(out_dir, exist_ok=True)
    grid = [(o, s) for o in a.optimizers for s in a.seeds]
    mlr_tok = f"_mlr{a.muon_lr:g}".replace(".", "p") if a.muon_lr != 0.02 else ""
    print(f"perm_mnist grid: {len(grid)} runs (muon_lr={a.muon_lr}) -> {out_dir}")
    for i, (opt, seed) in enumerate(grid):
        cfg = base_cfg(opt, seed, a.n_tasks, a.steps, muon_lr=a.muon_lr)
        out_path = os.path.join(out_dir, f"{opt}{mlr_tok}_s{seed}.jsonl")
        t = time.time()
        summ, _ = run(cfg, out_path=out_path)
        print(f"[{i+1}/{len(grid)}] {opt} mlr{a.muon_lr} s{seed}: "
              f"first_task={summ['first_task_steps']} last_task={summ['last_task_steps']} "
              f"n_fit={summ['n_tasks_fit']}/{a.n_tasks} "
              f"final_feat_rank={summ['final_feat_eff_rank']:.2f} "
              f"({time.time()-t:.0f}s)", flush=True)
    print("DONE perm_mnist grid")


if __name__ == "__main__":
    main()

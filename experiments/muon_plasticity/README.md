# Direction 005 — Muon × loss of plasticity

Causal test of the **spectral-collapse drives loss of plasticity** hypothesis
(arXiv:2509.22335) at the OPTIMIZER level. A small MLP is trained on a long
continual sequence of tasks (warm-start: weights carried across tasks); we
measure per-task **fit speed** decay (= loss of plasticity) under
`{muon, adamw, sgdm}`, and run a spectral/feature diagnostic suite after every
task to test whether Muon's orthogonalized (spectrum-preserving) updates keep
the network trainable and/or preserve feature/weight spectra where AdamW and
SGD-momentum lose plasticity.

Two fully-synthetic, zero-download benchmark arms (deterministic per seed+task):
- **proj_shift** — shifting random-projection classification: shared input pool,
  fresh random projection `P_t` + linear teacher `W_t` per task,
  `y = argmax(W_t · relu(P_t · x))`.
- **label_refit** — same inputs, fresh uniform-random labels per task (pure
  memorization-capacity probe).

An optional **perm_mnist** arm exists behind `--arm perm_mnist`, guarded by a
`try: import torchvision`; it requires MNIST already present locally
(`download=False`) and is NEVER touched by smoke.

Diagnostics (`probes.py`, read-only `@no_grad` except the documented two-backward
optimization-readiness probe): feature effective rank + stable rank (penultimate
activations), dead-unit fraction (ReLU zero-rate), per-hidden-matrix weight
effective rank, gradient-norm strength + cross-minibatch gradient cosine, and
Hessian λmax (reused by import from `experiments/eos_tiny/hessian.py`).

## Smoke check (no files written, <60 s)
```
python train_plasticity.py --smoke     # labeled smoke lines (incl. plasticity probe)
python probes.py                        # probe self-test (PASS on known low-rank matrix)
```

## Dry run (prints planned cells, launches nothing)
```
python run_plasticity.py --dry-run                     # 30 cells (warm-start grid)
python run_plasticity.py --dry-run --include-coldstart # 60 cells (+ cold-start contrast)
```

## Real grid (run when ready — do NOT launch yet)
```
python run_plasticity.py    # muon/adamw/sgdm × {proj_shift,label_refit} × seeds 0-4 = 30 cells
```
Results land in `experiments/results/muon_plasticity/` (created only by real
runs). Resume-aware: a cell whose jsonl already ends with a `_summary` line is
skipped.

## Import discipline
This directory's modules win (LOCAL dir inserted at `sys.path[0]`); reused infra
is APPENDED at the back: the `Muon` optimizer class from `experiments/grokking/`
and `top_eigenvalue` from `experiments/eos_tiny/hessian.py`. `model.py` carries
its own MLP and a LOCAL `split_params_for_muon` (the grokking split is
GrokTransformer-specific). Per the root README hard rule, files under
`experiments/{grokking,eos_tiny,...}/` are reused by import only and never
modified.

## Reference
See `directions/005-muon-plasticity.md` for the full research write-up.

# Direction 012 — When do orthogonalized updates lock the basin?

**Muon × linear-mode-connectivity onset k\*** on the grokking testbed.

Question: as training proceeds, at what point do two children forked from the
same parent checkpoint become **linearly mode-connected** — i.e. you can linearly
interpolate between their final weights without crossing a loss barrier? That
onset spawn step is **k\*** (Frankle et al., *Linear Mode Connectivity and the
Lottery Ticket Hypothesis*, arXiv:1912.05671). We measure k\* across
`{muon, adamw, sgdm}` to ask whether Muon's orthogonalized (spectrum-controlled)
updates **lock the basin earlier** than AdamW / SGD-momentum.

## The instrument (spawn-fork)

1. **Parent**: train a `GrokTransformer` on modular addition (full-batch,
   deterministic), snapshotting a checkpoint at each of ~9 log-spaced **spawn
   steps** `{0,50,100,250,500,1000,2000,4000,8000}`.
2. **Fork**: at each spawn step, fork **2 children**, make them **diverge** via an
   explicit noise source, and train each to a shared fixed end step.
3. **Barrier**: between the two children's final weights, evaluate loss/acc along
   the linear interpolation `theta(alpha) = (1-alpha)*a + alpha*b` (forward
   passes only); the **barrier** is `max_alpha loss(alpha) - max(loss(0),
   loss(1))`.
4. **k\***: the earliest spawn step after which the barrier stays below
   `barrier_threshold` (children land in a shared basin).

## Child-divergence design note (CRITICAL)

The grokking testbed is **full-batch deterministic** — two same-seed children
would be bit-identical and their barrier would be trivially 0. We therefore
implement **two** divergence mechanisms behind `--child_noise`:

- **`perturb`** (default primary): at the spawn checkpoint, add a tiny Gaussian
  perturbation, per-tensor std `= perturb_scale * rms(W)` so the **relative L2**
  perturbation is `~perturb_scale` (default `1e-3`) for every tensor regardless
  of its native norm/size. The two children get **different** perturbation seeds;
  training is otherwise identical full-batch.
- **`minibatch`**: the spawn checkpoint is copied **unperturbed**; the two
  children train with **large-minibatch** sampling (`--minibatch_size`, e.g. 2048
  of the train pool) under **different shuffle seeds**. Divergence comes from the
  SGD noise of distinct minibatch orders.

Both are documented and switchable; `perturb` is the headline mechanism because
it injects a controlled, measurable divergence into an otherwise deterministic
trainer.

`fork.py` note on the barrier of a *pure* perturbation: perturbing a single point
and interpolating back gives a **monotone** path (no midpoint bump above the
worse endpoint), so its barrier is genuinely 0 — a real positive barrier requires
two endpoints that have **diverged via training**, which is exactly what the
protocol produces. The smoke fork-probe reflects this: it forks two children,
gives each one step, and reports a small *positive* pairwise barrier.

## Smoke check (no files written, <60 s)

```
python train_fork.py --smoke   # the 5 labeled smoke lines (incl. fork probe)
python fork.py                 # fork.py self-test (PASS):
                               #   self-barrier == 0 exactly,
                               #   independent-solution barrier >> 0,
                               #   perturb relative-norm ~= scale,
                               #   minibatch-mode copy is unperturbed,
                               #   k_star recovers a known onset pattern.
```

Smoke contract (exact, exit 0):
```
SMOKE DATASET SHAPE: ...
SMOKE PARAM COUNT: <n>
SMOKE FORWARD LOSS: <float>
SMOKE OPTIMIZER STEP: OK
SMOKE FORK PROBE: self_barrier=0.000000 perturbed_barrier=<f> rel_perturb=<f>
```

## Dry run (prints the cell plan, launches nothing)

```
python run_lmc.py --dry-run
```
Reports: **9 parent training runs** (optimizer × seed), **81 parent checkpoints**
(optimizer × spawn × seed; the task's "~27 parents" is optimizer × spawn at one
seed), and **162 child runs** (optimizer × spawn × child × seed). One grid CELL =
one (optimizer, seed) parent run that internally forks all spawn × child
children and emits one jsonl with per-spawn barriers + k\*.

## Real grid (run when ready — do NOT launch yet)

```
python run_lmc.py    # muon/adamw/sgdm × 3 seeds = 9 parent cells (162 children)
```
Results land in `experiments/results/lmc_instability/<opt>_s<seed>.jsonl` and
spawn checkpoints in `experiments/results/lmc_instability/ckpt/<name>/` (created
only by real runs). Resume-aware: a cell whose jsonl already ends with a
`_summary` line is skipped. `--num-shards/--shard-id` split the 9 cells across
machines.

## Import discipline (root README hard rule)

This directory is inserted at `sys.path[0]`; `experiments/grokking/` is
**appended** at the back, so our local modules win. We reuse **only**
`GrokTransformer` (model.py), `make_modular_dataset` / `train_test_split`
(data.py), and `Muon` / `split_params_for_muon` (muon.py) from grokking by
import — those files are **never modified**. The new load-bearing module is
`fork.py` (spawn / barrier / k\*). The grid runner additionally imports the
shared `experiments/runner_utils.py` sharding helpers.

## Reference

See `directions/012-lmc-basin-locking.md` for the full research write-up.
```

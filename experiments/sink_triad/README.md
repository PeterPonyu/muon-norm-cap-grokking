# Direction 008 — Attention-sink triad adjudication × Muon

Does the **extreme-token triad** — attention sink + massive activation + value
drain — form under Muon's orthogonalized updates the way it does under Adam? And
**why** does it form? Three hypotheses are adjudicated factorially:

1. **Optimization artifact** — the triad is driven by coordinate-adaptivity / a
   gradient-sink mechanism (2410.13835, 2603.17771). Prediction: it tracks the
   OPTIMIZER family and weakens/vanishes off the Adam adaptivity line (Muon, SGDM).
2. **Pre-norm architecture artifact** — pre-LN lets a token park a massive,
   LayerNorm-rescaled activation in the residual stream (2603.05498). Prediction:
   sandwich norm (LN before AND after each sublayer's residual add) suppresses it.
3. **Functionally necessary** — the sink does real work. Prediction: ablating it
   costs loss (`ablation_cost` large) regardless of optimizer / norm position.

## Task — Bigram-Backcopy (BB)
Synthetic online causal-LM stream (`data.py`): BOS-anchored sequences over a
~64-token vocab; a fixed bigram Markov chain except after TRIGGER tokens, where
the next token deterministically BACKCOPIES the pre-trigger token. This is the
native toy setting (2410.13835) in which heads must be ACTIVE after triggers and
may go DORMANT (attend to BOS = sink) on plain bigram positions — exactly the
active-dormant dynamic the triad rides on. Fresh batch per step; deterministic
per `(chain_seed, batch seed)`. Role masks `backcopy_mask` / `bigram_mask` align
with the next-token targets.

## Model — `model.py` (SELF-CONTAINED local `SinkTransformer`)
A self-contained tiny causal LM mirroring the grokking architecture (2 layers,
4 heads, d=128, learned PE, causal mask, full-sequence logits) with a
**`norm_position ∈ {pre, sandwich}`** switch:
- `pre` — grokking-identical pre-LN (sublayer output added raw to the residual);
  bit-for-bit the 007 baseline (same submodule names + init).
- `sandwich` — extra LayerNorm on each sublayer's OUTPUT before the residual add
  (2603.05498's decoupling intervention), so a massive activation cannot
  accumulate in the residual stream.

**Why self-contained (not a GrokTransformer subclass):** the sandwich variant
rewires the residual-add INSIDE the block, which subclassing cannot express (the
grokking block is a fixed pre-LN module). We still IMPORT the grokking
`GrokTransformer` (verifying the `induction_emergence` import chain resolves —
LOCAL dir `insert(0)`, grokking dir appended, `importlib` to dodge the
`model.py` name collision) to assert the chain and reuse its exact init recipe.
The grokking files are NOT modified. `split_params_for_muon` is the grokking
name-based splitter, re-exported; it applies unchanged across both norm
positions (the sandwich LayerNorms are 1-D → AdamW).

Read-only instrumentation: `forward_with_attn(idx) -> (logits, attn_list)` with
`attn_list[l]` of shape `[B, H, T, T]`; `forward_with_triad(idx)` additionally
returns per-head value norms, post-block residuals, and hidden states.

## Probes — `probes.py` (all `@no_grad`, never mutate live weights)
- (i) `sink_ratio` — mean attention mass on column 0 (per layer/head + aggregate).
- (ii) `spike_magnitude` — max |hidden coordinate| per layer (massive-activation).
- (iii) `value_drain` — `‖v_0‖ / mean_{t>0} ‖v_t‖` (per layer/head).
- (iv) `residual_peak` — residual-stream norm at position 0 vs the rest.
- (v) `ablation_cost(model, batch)` — masked-loss delta when attention to
  position 0 is zeroed + renormalized, computed on a deep COPY's read-only
  forward (no weight is touched).

## Trainer — `train_sink.py`
Online fresh-batch causal-LM loop (`steps` default 12k, eval every 150 on a fixed
held-out stream). `Config` dataclass + dynamic argparse; optimizer families via
the imported Muon hybrid split. Per-eval jsonl line: role acc/loss + per-layer
triad metrics + `ablation_cost`. Summary: final triad values + formation steps.

## Smoke (no files written, <60 s)
```
python data.py                       # data self-test (PASS)
python probes.py                     # probe self-test (PASS on designed triad)
python train_sink.py --smoke         # labeled smoke lines
python run_sink.py --smoke           # delegates to the trainer smoke
```
The `--smoke` contract prints exactly:
```
SMOKE DATASET SHAPE: ...
SMOKE PARAM COUNT: <n>
SMOKE FORWARD LOSS: <float>
SMOKE OPTIMIZER STEP: OK
SMOKE SINK PROBE: sink_ratio=<f> spike=<f> ablation_cost=<f>
```
(≤1 step, no jsonl, exit 0; untrained triad values are near baseline.)

## Dry run (prints planned cells, launches nothing)
```
python run_sink.py --dry-run             # 30 core cells
python run_sink.py --dry-run --depth-arm # 30 + 18 depth dose-response cells = 48
```

## Real grid (run when ready — do NOT launch yet)
```
python run_sink.py              # muon/adamw/sgdm × pre/sandwich × seeds 0-4 = 30
python run_sink.py --depth-arm  # + depth dose-response (n_layers∈{1,3}, +18)
```
Results land in `experiments/results/sink_triad/`. Resume-aware: a cell whose
`.jsonl` already ends with a `_summary` line is skipped.

## Discipline (root README conventions)
- Imports: LOCAL dir `insert(0)` on `sys.path`, grokking dir appended last; only
  grokking `model.py`(GrokTransformer) / `muon.py` are pulled, both UNMODIFIED.
- Results: the runner writes ONLY `experiments/results/sink_triad/`; smoke writes
  nothing.
- Two manipulated variables: optimizer family AND norm position (the 2603.05498
  factor is now an axis, not held constant).

## Reference
See `directions/008-sink-triad-muon.md` for the full research write-up.

# Findings 008 — The sink triad off the Adam–SGD line: three-way dissociation; massive activations track effective step size, and Muon's stable regime sits deep inside the spike zone

**Direction:** `directions/008-sink-triad-muon.md` · **Novelty record:** `.omc/research/novelty-008.md`
**Data:** `results/sink_triad/` — 48 runs on the remote 4080 (Codex executor):
main factorial {muon, adamw, sgdm} × {pre, sandwich} × 5 seeds (30) + depth arm
{muon, adamw, sgdm} × L{1,3} × 3 seeds (18). BB-style BOS-anchored causal LM
(SinkTransformer, 427k params, L=128, online fresh-batch). Probes: sink ratio
(col-0 attention mass), spike magnitude (max |hidden|), value drain (‖v₀‖/median),
residual peak (‖h₀‖/median), ablation cost (Δloss, col-0 zeroed). All cells reach
backcopy accuracy ≈ 1.0 — no failure-mode confound.

## Headline table (main arm, 5-seed means)

| cell | sink | spike | drain | peak | ablation cost |
|---|---|---|---|---|---|
| adamw_pre | 0.963 | 245.6 | 0.073 | 0.67 | 0.035 |
| adamw_sandwich | 0.614 | 41.2 | 2.37 | 0.91 | 0.173 |
| **muon_pre** | 0.926 | **4787.1** | 0.325 | 0.40 | 0.104 |
| muon_sandwich | 0.967 | 52.8 | 3.50 | 0.59 | 0.105 |
| **sgdm_pre** | **0.301** | 14.9 | 0.182 | 1.05 | **0.0002** |
| sgdm_sandwich | 0.254 | 44.2 | 1.04 | 0.99 | 0.119 |

## P1 — Sink under orthogonalized updates: **parity for Muon; SGDM bounds the necessity claim**

Muon forms the sink at Adam parity (0.926 vs 0.963). But **bare SGD-momentum
forms only a weak sink (0.30)** while still solving the task perfectly — and its
sink is functionally free (ablation cost 0.0002). The softmax-geometric
"provably necessary" reading (2603.11487) holds for Adam/Muon dynamics but is
**bounded by the SGDM arm in its own empirical regime**: a 0.30-mass, zero-cost
sink in a task-solving model is not a necessity.

## P2 — Coordinate-adaptivity attribution: **REFUTED; lr arm reframes the amplification**

Prediction (2410.13835 / 2603.17771 mechanism): without per-coordinate
adaptivity, the massive-activation spike should decay toward SGDM level.
Measured at the standard operating points: **Muon's spike is 4787 — 20× AdamW,
320× SGDM.** The adaptivity attribution is falsified in its own native toy
setting — orthogonalized updates without any per-coordinate adaptivity produce
the *largest* spikes observed.

**LR-robustness arm (completed 2026-06-11, `results/sink_triad_lr/`, 15 runs):
the amplification is operating-point-dependent, monotone-in-lr within BOTH
families** —

| muon lr | 0.005 | 0.01 | 0.02 (main) | 0.04 |
|---|---|---|---|---|
| spike | ~195 | ~680 | 4787 | **~21,000** |

| adamw lr | 0.0003 | 0.001 (main) | 0.003 |
|---|---|---|---|
| spike | ~12 | 245.6 | ~1,440 |

Matched-spike pairs exist across families (adamw@3e-3 ≈ muon@~0.01), so the
honest claim is: **spike magnitude is primarily an effective-step-size
phenomenon; what optimizer *family* controls is the operating point at which
stable training happens** — and Muon trains stably at step sizes whose spikes
are 1–2 orders above Adam's stable range (21k spike at lr 0.04, task still
solved). The adaptivity attribution stays refuted; the "20× amplifier" headline
is downgraded to "Muon's stable operating regime sits deep in the
massive-activation zone". Two further lr-arm observations: AdamW's *sink*
weakens toward SGDM level at lr 3e-4 (0.18–0.19 in 2/3 seeds, with bimodal
seed behavior) — sink formation itself needs sufficient effective step; and
ablation cost is non-monotone in lr for Muon (0.001 → 0.17–0.33 → 0.001 across
0.005→0.01→0.04), so functional necessity has an interior maximum.

## P3 — Sandwich decoupling replicates at small scale (multi-seed)

Sandwich-norm collapses the spike for the optimizers that had one — AdamW
245.6 → 41.2 (6×), **Muon 4787 → 52.8 (91×)** — while the sink survives
(Muon 0.97 unchanged; AdamW 0.61 partial; SGDM unchanged-low). 2603.05498's 7B
single-config headline (spike is a pre-norm architectural artifact; sink is not)
**replicates at 427k params with 5 seeds**, and the factorial adds what 7B could
not: the architecture lever works *regardless of optimizer*, including on
Muon's 20×-amplified spike.

## P4 — Triad coupling: **three-way dissociation (neither published 2-way split)**

Across all 48 runs: corr(drain, peak) = 0.007, corr(drain, sink) = 0.094 —
the value drain tracks **neither**. The factorial shows each triad component has
a different governor: **sink ← optimizer dynamics** (collapses only when leaving
both adaptive and orthogonalized families), **spike ← architecture × optimizer**
(norm position sets whether it exists; optimizer sets magnitude), **drain ←
architecture alone** (pre: drained 0.07–0.33; sandwich: anti-drained 1.0–3.5,
all optimizers). The "triad" is not a triad — it is three phenomena sharing
position 0.

## P5 — Functional necessity is dynamics-contingent

Ablation cost spans **three orders of magnitude by cell**: sgdm_pre ≈ 0.0002
(pure artifact) vs adamw_pre 0.035 vs muon_pre 0.104 vs sandwich arms 0.10–0.17.
The same intervention is free in one optimizer's solution and costly in
another's — the outcome the direction doc pre-registered as "the result none of
the four papers can express." Necessity claims about sinks must be indexed by
training dynamics, not stated architecture-wide.

## Depth arm

Triad signatures persist across L ∈ {1, 2, 3} with the same optimizer ordering
(figures); no depth inversion observed within this range (`fig_depth.png`).

## Limitations

- Toy scale (427k params, synthetic BB stream, vocab 64, L=128); the 20× Muon
  spike amplification needs a scale check before any claim about real LMs.
- Ablation cost measured by col-0 zeroing + renormalization only (one ablation
  semantics); drain/peak are layer-aggregated scalars.
- Sandwich arm re-wires residual-add (per scaffold README) — it is the
  2603.05498 intervention class, not a parameter-matched control.
- SGDM weak-sink cells solve the task with near-zero sink; we did not test
  whether *forcing* sink removal during training (not just at eval) hurts SGDM.
- ~~Single (lr, wd) per optimizer family~~ **resolved 2026-06-11**: the
  lr-robustness arm (15 runs) is folded into P2 above; remaining gap is wd
  (weight decay) robustness, untested.

## Figures

`results/figures-008/`: `fig_triad_factorial.png`, `fig_ablation.png`,
`fig_depth.png`, `sink_verdicts.json`.

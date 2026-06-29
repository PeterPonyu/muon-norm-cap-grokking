# Findings 005 — Muon is a plasticity *prolonger*, not a therapy; the operative spectrum is features, not weights

**Direction:** `directions/005-muon-plasticity.md` · **Novelty:** `.omc/research/novelty-008.md`-adjacent record + night-rescan addendum; pre-launch SPHERE check PASSED (2026-06-11, recorded in the direction doc).
**Data:** `results/muon_plasticity/` — 30 runs: {muon, adamw, sgdm} × {proj_shift,
label_refit} × 5 seeds, warm-start, 100-task continual sequence, MLP (167k params),
per-task probes (feature eff-rank, weight eff-rank per layer, dead fraction, λmax,
grad stats). Fit threshold per task; `steps_to_threshold = None` ⇒ task unfittable
(plasticity lost); censor_onset = first such task.

## Headline table (5-seed means)

| cell | censor onset (of 100) | fit-speed slope (steps/task) | feat-rank retention | weight-rank retention | dead frac |
|---|---|---|---|---|---|
| sgdm_label_refit | 7.0 | +19.6 | 0.00 | 0.98 | 1.00 |
| adamw_label_refit | 10.4 | +10.8 | 0.00 | 0.97 | 1.00 |
| sgdm_proj_shift | 22.2 | +5.3 | 0.00 | 0.92 | 0.92 |
| adamw_proj_shift | 34.4 | +2.9 | 0.00 | 0.94 | 1.00 |
| **muon_label_refit** | **75.0** | **+0.88** | 0.00 | 0.91 | 1.00 |
| **muon_proj_shift** | **80.6** | **+0.51** | 0.03 | 0.92 | 0.73 |

## P1 — Reproduction gate: **HOLDS**

AdamW and SGDM lose plasticity exactly as 2509.22335 describes: fit speed decays
steeply (slopes +2.9 to +19.6 steps/task), tasks become unfittable early
(censoring at task 7–35), feature eff-rank collapses to ~0 retention, dead
fraction → ~1.0. The regime and instrument are valid.

## P2 — Muon as optimizer-level therapy: **REFUTED in strict form; large prolongation effect is real**

Strict P2 (slope statistically indistinguishable from 0) fails: Muon's slope is
+0.51/+0.88 — small but positive — and **Muon too eventually loses plasticity**
(censoring at task 75–81 of 100). What survives is the largest optimizer-level
plasticity effect measurable in this design: **2.3–10.7× more fittable tasks**
than AdamW/SGDM and **6–12× shallower decay**, with zero plasticity-specific
machinery. Vanilla Muon is a *prolonger*, not a cure.

## P3/P4 — Which spectrum is causal: **weight spectra exonerated universally; feature collapse is the operative correlate**

The sharpest result, visible in `fig_dissociation.png`:
- **Weight eff-rank retention is 0.91–0.98 for EVERY optimizer** — including
  AdamW and SGDM, whose plasticity dies by task 7–35. Weight-spectrum collapse
  simply never occurs in this regime, so it cannot be the driver of plasticity
  loss — for anyone. (This also empirically separates update-geometry
  intervention from weight-spectrum regularization, P4's dissociation: Muon
  preserves weight rank no better than the optimizers that fail 10× earlier.)
- **Feature eff-rank retention is ≈0.00 for EVERY optimizer** (Muon 0.03 at
  best): feature collapse tracks plasticity loss universally — Muon delays the
  collapse (and keeps dead fraction at 0.73 vs ~1.0 in its best arm) but does
  not prevent it.

Verdict for the 2509.22335 causal law: **survives only at the feature/
representation level; any weight- or update-spectrum reading is demoted** —
an optimizer that provably flattens update spectra and preserves weight rank
still marches into the same feature-collapse wall, just later.

## P5 — Warm-start bridge: **not run** (severable by design; the cold-start
contrast cells require `--include-coldstart` and were not part of the headline
grid).

## Limitations

- 100-task budget: Muon censors at ~75–81, so its post-collapse behavior is
  observed in only ~20 tasks; a longer sequence would sharpen the "delays but
  does not prevent" claim.
- One architecture (3-layer MLP, width per scaffold), one task family pair,
  single (lr, wd) per optimizer (001's standard hybrid); no lr-matched control
  for the prolongation magnitude (Muon's effective step could partly explain
  delay — though it cannot explain the universal weight/feature dissociation).
- Feature eff-rank measured on a fixed probe set; dead fraction is
  activation-zero based.
- P2's "statistically indistinguishable from 0" was evaluated by effect size
  (slope ratio), not a formal test — slopes' seed std available in
  `plasticity_verdicts.json`.

## Figures

`results/figures-005/`: `fig_fit_speed.png` (P1/P2), `fig_spectra.png` (P3/P4
trajectories), `fig_dissociation.png` (the weight-vs-feature retention map),
`plasticity_verdicts.json`.

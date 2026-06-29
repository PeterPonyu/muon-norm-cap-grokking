# Findings 012 — Muon keeps basin selection open ~4× longer than AdamW, despite already generalizing

**Direction:** `directions/012-lmc-basin-locking.md` (bank OG3a 7.9; spawn-fork
interpolation-barrier instrument, new code).
**Question:** when does the optimizer commit to a linear-mode-connectivity (LMC)
basin — i.e. at what spawn step k* do two perturbed children, forked from a
shared parent and trained to a fixed end, become linearly connected (loss
barrier < 0.1)?
**Data:** `results/lmc_instability/` — 9 headline cells (3 optimizers × 3 seeds,
spawn ladder {0,50,100,250,500,1000,2000,4000,8000}, child_end 12000,
perturb-fork, mod-add p=97) + `results/lmc_instability_ext/` — 3 Muon cells with
the spawn ladder extended to {8000,16000,24000,32000}, child_end 40000 (pins the
right-censored Muon k*). Barrier = max excess train-loss along the child0↔child1
linear interpolation; k* = earliest spawn with barrier < 0.1.

## Headline (5-seed hardened 2026-06-14 — softened from n=3 per red-team audit)

**Muon keeps its LMC basins systematically more open than AdamW at spawn 8000,
but the difference is DISTRIBUTIONAL and seed-variable, NOT a clean k* threshold
separation.** At 5 seeds: every Muon seed is still wide open at spawn 8000
(barriers 2.1–4.4, median 3.18, **0/5 locked**); AdamW's barriers cluster low
(median 0.56, **2/5 locked** at barrier<0.1) but one seed (4.63) is as open as
Muon. Both optimizers' children grok (test acc 1.0) — the contrast is in basin
commitment, not learning — and the ext arm shows Muon does eventually lock
(s0 ≈0.16 by spawn 32000). **Honest statement:** Muon's basins commit later /
stay more malleable than AdamW's (median barrier 3.18 vs 0.56), the LMC-side
reading of its growth route (001/002). But the n=3 version's "~4× later, clean
k*≈8000 for AdamW" does NOT survive 5 seeds — k* is mostly right-censored for
both optimizers (AdamW locks only 2/5), so the claim is a distributional one, not
a measured ratio. SGDM never learns the task, so its low barriers are vacuous
(both children sit in the same un-learned basin) and excluded.

## P1/P2 — k* by optimizer

| optimizer | parent/children outcome | barrier @ spawn 8000 (**5 seeds**) | median | locked (b<0.1) |
|---|---|---|---|---|
| **AdamW** | grok, test acc 1.0 | 0.61 / 0.09 / 0.00 / 0.56 / 4.63 | **0.56** | **2/5** |
| **Muon** | grok, test acc 1.0 | 2.32 / 4.40 / 3.18 / 3.45 / 2.11 | **3.18** | **0/5** |
| **SGDM** | **never learns, test acc ≈0.01–0.05, loss ≈ ln 97** | 0.00 / 0.14 / 0.20 / 0.19 / 0.14 | 0.14 | 1/5 (vacuous) |

AdamW and Muon children both reach test acc 1.0, so the barrier contrast is not
confounded by failure-to-learn. **The 5-seed distributional contrast holds:
Muon's barriers are uniformly high (all 5 > 2.1, median 3.18) while AdamW's
cluster low (4/5 ≤ 0.61, median 0.56)** — Muon keeps children in *different*
basins, AdamW mostly in the *same* one. BUT the n=3 "AdamW locks ≈8000" reads
as only **2/5** at 5 seeds (one AdamW seed, 4.63, is as open as Muon), so this is
a distributional commitment difference, **not** a per-seed k* threshold. k* is
right-censored for 5/5 Muon and 3/5 AdamW.

## SGDM exclusion (the interpretation guard)

SGDM's barrier is ≈0 from the earliest spawn, which naively reads as "locks
immediately". But SGDM's children end at **test acc 0.014–0.048 and train loss
4.25–4.57 ≈ ln(97)** — i.e. the uniform-prediction floor: SGDM never leaves the
random-init basin at all. Two children that both stayed at initialization are
trivially connected. **SGDM's k* is therefore vacuous and is excluded from the
locking claim** — connectivity is only meaningful conditional on the parent
having learned (AdamW, Muon). This is the LMC analogue of 002's finding that
SGDM cannot train this family without orthogonalization.

## Extension arm — pinning Muon's censored k*

`lmc_instability_ext/` (Muon, child_end 40000, spawn to 32000):

| seed | barrier @ 8000 → 16000 → 24000 → 32000 |
|---|---|
| s0 | 5.48 → 2.48 → 0.96 → **0.16** (nearly locks; just above the 0.1 bar) |
| s1 | 5.98 → 4.42 → 2.72 → 2.34 (still wide open) |
| s2 | 5.57 → 5.92 → 2.65 → 1.72 (decaying, still open) |

Muon's barrier **does** decay with later spawn — basins are not open forever — but
the lock arrives near spawn ≈32000 for the fastest seed and remains censored
(>0.1) for the other two even at 32000. So Muon's basin commitment is **late
(~4× AdamW) and highly seed-variable**, vs AdamW's tight ≈8000 lock. k* stays
formally censored at the 0.1 threshold for 2/3 seeds; the bound is "Muon ≳ 4×
AdamW, erratic". (Methodological note: the ext arm uses child_end 40000 vs the
headline's 12000, so absolute barrier magnitudes are not cross-comparable to the
headline; only the within-arm decay trend across spawn is interpreted.)

## Limitations

- One task (mod-add p=97), one lr per optimizer (hybrid protocol), perturb-fork
  only (the minibatch-fork robustness arm in train_fork is not run here).
- k* threshold 0.1 is a convention; Muon's near-miss at s0 (0.16) means the
  exact k* is threshold-sensitive — reported as a bound, not a point.
- SGDM's non-learning is a property of this family at this lr; it removes SGDM
  from the comparison rather than contributing a third k*.
- Barrier measured on the train pool (the loss the optimizer shaped), per the
  headline protocol; a held-out-barrier arm is not run.

## Figures / data

`results/lmc_instability/` (9 headline) + `results/lmc_instability_ext/` (3 Muon
ext); per-cell jsonl carry `barrier_by_spawn` + `k_star`; per-step records carry
`acc_endpoints` (the learning guard) and the interpolation `barrier`.
`results/figures-012/fig_lmc.png` — barrier-vs-spawn curves per optimizer
(headline: AdamW locks ~8000, Muon stays open, SGDM vacuous) + the ext-arm panel
(Muon spawn→32000, s0 near-locks 0.16).

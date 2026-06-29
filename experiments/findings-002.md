# Findings 002 — The two-route model on a non-abelian task (S5 × mechanism probes)

**Direction:** `directions/002-s5-mech-two-route.md` · **Novelty record:** `.omc/research/novelty-002.md`
**Data:** `results/s5_mech/` (30 runs, mech=True), behavioral baseline `results/task_s5/`
(20 acc-only runs), mod-add comparison arm `results/mech/` (30 runs, 001 pipeline).
**Setup:** S5 composition (120 perms, 14,400 pairs), GrokTransformer d=128/2L (425,856
params), full-batch, wd=0.01, 5 seeds/cell, eval_every=50, budgets 20k (sc=1) / 12k (sc=3).

## TL;DR

On the non-abelian task the two-route picture survives — in a sharper form than on
modular arithmetic: **only the growth/directional route reaches generalization at all.**
Muon groks 5/5 (sc=1) with hidden-norm **growth ≈ 6.5×** and 70–80° of directional
rotation; AdamW and SGD-momentum **never grok (0/20 combined)** within budget at either
init scale. The "grokking ⇔ spectral/rank compression" reading is contradicted on S5:
Muon generalizes at effective rank ≈ 111/128, never dipping below ~101 en route. One
prediction fails honestly: S5 does **not** demand higher effective rank than mod-add
(111 vs 119 — slightly lower, not higher).

## Behavioral results + reconciliation (probe purity)

| cell | grok (mech runs) | baseline (acc-only) | reconciliation |
|---|---|---|---|
| muon sc=1 | **5/5** (2150–10750) | 5/5 (2150–10750) | per-seed grok steps **exactly equal** (2150=2150, 3400=3400, …) |
| muon sc=3 | 1/5 (s1@8250) | 1/5 (s1@8250) | exact |
| adamw sc=1 | 0/5 (test ≈ 0.01 = chance) | 0/5 | consistent |
| adamw sc=3 | 0/5 | 0/5 | consistent |
| sgdm sc=1 | **0/5** (memorizes @150–200, test ≤ 0.06) | — (new cell) | — |
| sgdm sc=3 | **0/5** | — (new cell) | — |

The MechProbe is read-only and training is seed-deterministic: mech runs reproduce the
baseline *exactly*, so all mechanism quantities describe unperturbed training.

## P1 — Two-route universality: **HOLDS (Muon side); AdamW takes the predicted failure branch**

Every grokking event on S5 (6 total: 5×sc1 + 1×sc3, all Muon) has
`wn_hidden(T_grok)/wn_hidden(T_mem)` = **6.52 ± 0.68** (sc=1) and 6.59 (sc=3) — the
norm *grows* through grokking, by far more than on mod-add (1.5–10× there, ~1.9× at
matched sc=1). AdamW never contracts its way to generalization — it simply never
generalizes within budget (P1's stated alternative). No grokking event anywhere on S5
sits in the contraction half-plane (`figures-002/fig_s5_route_map.png`).

## P2 — Directional route: **HOLDS (in the only available sense)**

Muon's rotation from the memorization direction at grok is **70–80°** (sc=1 cluster;
79.8° at sc=3) — larger than any route-map point we have measured on any task.
Matched-cell comparison against AdamW is impossible on S5 (AdamW has no grok events);
the honest framing: AdamW rotates ~60° from init over its full 20k-step run *without
ever generalizing*, so rotation alone is not sufficient — Muon's spectrum-flattened
rotation is what reaches the generalizing region. On matched mod-add cells (001
Result 10), Muon's rotation rate exceeds AdamW's by 15–100×.

**Red-team tempering (audit 2026-06-10):** on S5 the rotation *rate* is NOT a clean
discriminator — Muon's per-step rate (7.5–34°/1k steps over mem→grok windows)
overlaps AdamW's non-grokking baseline (3.5–9.6°/1k over matched windows). The
S5-route signature is the **norm-growth axis plus the destination** (Muon's rotation
ends at a generalizing region; AdamW's does not), not rotation speed. The 15–100×
rate separation is a mod-add result and does not transfer to S5.

## P3 — Spectral-compression non-universality: **HOLDS**

Muon groks on S5 at **effective rank 111.3 ± 2.6 of 128**, and its per-run eff-rank
*minimum* en route never drops below ~101. There is no monotone collapse toward low
rank at the moment of generalization — contradicting the universal reading of
"grokking ⇔ rank/spectral compression" (Yunis et al. 2024; openreview 6NHnsjsYXH),
which was established on AdamW/SGD + abelian tasks. Side observation: *memorizing*
AdamW (which never groks) ends with stable rank ≈ 23 — spectral concentration is a
signature of the memorization solution here, not of generalization.

**Red-team strengthening (audit 2026-06-10):** the "is the rank probe just frozen?"
attack found the opposite — the instrument detects collapse when it happens:
**SGDM on S5 sc=1 collapses massively (eff-rank 118 → min ~10, stable rank → 1–2)
while never generalizing.** Together this is a double dissociation: compression
without generalization (SGDM) and generalization without compression (Muon).
Disclosure for completeness: Muon's *stable* rank does decline moderately (≈57 → 34)
while its entropy eff-rank stays near-full — "no collapse" is precise for the
entropy measure; the stable-rank measure shows partial concentration.

## P4 — Non-abelian rank signature: **REFUTED in the predicted direction**

Prediction: S5's higher-dimensional irreps should demand *higher* effective rank at
grok than mod-add. Measured: **S5 111.3 ± 2.6 vs mod-add 118.8 ± 1.7** (Muon cells) —
slightly *lower*, not higher (`fig_s5_vs_add_rank.png`). Both sit near full rank, so
the practical content is "no compression on either task", but the directional
prediction as stated fails. Possible confound: at d=128 both tasks may be far from
rank-limited; a width sweep would be needed to expose representation-dimension demands.

## New finding (not among P1–P4): orthogonalization becomes *necessary*, not just faster

On mod-add, SGD-momentum (same lr/momentum/hybrid as Muon, no Newton-Schulz) grokked —
slowly and unstably. On S5 it **never groks (0/10 across both init scales)** despite
memorizing fine, and AdamW also 0/10. Task hardness promotes the orthogonalization from
an accelerator to a *requirement* for generalization within practical budgets. This
strengthens 001's headline in the direction that matters for practice.

**Rescue-sweep adjudication (audit fix, completed 2026-06-11, `results/s5_rescue/`,
24 runs):** the claim splits cleanly by optimizer family.
- **AdamW: rescuable — claim downgraded** to "necessary at standard settings".
  lr=0.01 + wd=1.0 groks 2/2 (850/1400 steps, test 1.0); lr=3e-3 + wd=1.0 groks 1/2
  (16700, barely within budget). No rescue at wd=0.01 for any lr (0/6), and
  lr=0.03 + wd=1.0 destabilizes (0/2). Mirror of the mod-add pattern: AdamW reaches
  S5 generalization only via the ~100× weight-decay contraction route.
- **SGDM: NOT rescuable — claim upgraded.** 0/12 across lr {5e-3, 2e-2, 8e-2} ×
  wd {0.01, 1.0}: no (lr, wd) in the sweep lets bare SGD-momentum generalize on S5.
  Within this sweep, orthogonalization (or AdamW's coordinate-wise adaptivity +
  extreme wd) is genuinely required.
End-of-budget trend check supports the within-budget claims: AdamW test accuracy is
flat at chance (last-quarter Δ ≤ +0.002), SGDM shows only a faint drift (max final
0.060 vs threshold 0.95). Muon remains the only optimizer that groks S5 at standard
hyperparameters.

## Limitations

- "Never groks" = within 20k/12k full-batch steps; the direction doc's contingency
  (extend sc=3 budget) was not exercised for AdamW/SGDM — longer budgets might
  eventually grok (mod-add behavior suggests AdamW's growth route is merely ~slow,
  but S5 sc=1 AdamW sits at chance test accuracy 0.005–0.012 with no upward trend).
- Single weight decay (0.01) — the S5 λ-sweep remains untested (also flagged in 001).
- Muon sc=3 on S5 is 1/5 — the strong-init + non-abelian corner is mostly beyond
  Muon's budget too; route claims there rest on one seed plus sc=1's five.
- Effective rank is entropy-based on singular values of d=128 matrices; both tasks may
  be unconstrained at this width (see P4 confound).
- eval_every=50 quantizes T_mem/T_grok; rotation/norm ratios use nearest-eval records.

## Figures

`results/figures-002/`: `fig_s5_route_map.png` (P1/P2), `fig_s5_mech_trajectories.png`
(P2/P3 trajectories), `fig_s5_vs_add_rank.png` (P4), `s5_norm_ratio_table.json`,
`s5_behavior_reconciliation.json`.

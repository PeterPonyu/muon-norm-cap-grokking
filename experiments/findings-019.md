# Findings 019 (Stage-A) — P0 capacity control FIRES: the group ladder is confounded by dataset size, not just complexity (B1 necessity-ladder needs redesign before Stage-B)

**Direction:** `directions/019-orthogonalization-necessity-threshold.md` (bank B1 8.2).
**Status: CLOSED. Stage-A (P0 capacity control) caught the |G|² data-size
confound; Stage-B (comparable-|G|² de-confounded ladder) then REFUTES the
complexity-graded necessity threshold — see Stage-B verdict below.**
**Data:** `results/group_complexity/` Stage-A — 18 runs: AdamW × {A4, A5, S5} ×
d_model{128, 256} × 3 seeds, full Cayley datasets, train_frac 0.5, 30k steps.
Question (P0): does each non-abelian group even GROK under AdamW at generous
capacity — establishing a non-trivial success rate to drive to zero in Stage-B?

## Headline

The answer is **no in a way that invalidates the naive ladder**: grok-success
under AdamW does NOT decrease with group complexity — it INCREASES with group
ORDER (= dataset size |G|²). The smallest non-abelian group **A₄ (12 elements,
144 pairs) fails to grok entirely (0/6, test acc ≈ chance)**, A₅ (60 elements,
3600 pairs) groks only at d256 (2/3), and S₅ (120 elements, 14400 pairs) groks
everywhere (6/6). Since group order and dataset size co-vary, **B1's "necessity
threshold vs group complexity" cannot be read off this ladder** — a small
non-abelian group fails for DATA-STARVATION reasons, not for orthogonalization
reasons. This is exactly the capacity artifact the killer-sweep flagged as B1's
collapse mode.

## P0 capacity-control table

| group | |G| | pairs |G|² | commutator density | AdamW grok @ d128 | @ d256 |
|---|---|---|---|---|---|---|
| A₄ | 12 | 144 | 0.667 | **0/3** (test 0.056) | **0/3** (0.042) |
| A₅ | 60 | 3600 | 0.917 | 0/3 (0.009) | **2/3** (0.999, grok 2300) |
| S₅ | 120 | 14400 | 0.942 | **3/3** (1.0, grok 1500) | **3/3** (1.0, grok 1100) |

Grok-success is **monotone in |G|² (dataset size), inverse to the naive
"smaller = easier" intuition** and orthogonal to commutator density: A₄ has the
*lowest* commutator density yet *fails*, S₅ has the *highest* yet *succeeds*. The
driver is data volume (a 12-element group offers only 72 train pairs at
train_frac 0.5 — too few to learn the group structure; test acc at/below chance
1/12), not complexity.

## Consequence for B1 (the necessity ladder, Stage-B)

The chartered Stage-B ({Z₉₇, D₁₂, A₄, A₅, S₅} × {muon, adamw, sgdm}) would
**confound optimizer-necessity with dataset size**: any "AdamW fails on A₄ but
Muon succeeds" would be uninterpretable, because AdamW fails on A₄ for data
reasons even at generous capacity. **Stage-B is therefore gated** pending a
de-confound. Options (decision needed):
1. **Comparable-|G|² ladder**: use groups of similar order (drop A₄; e.g. compare
   non-abelian groups all ≥ ~60 elements, or sub-sample S₅/A₅ to matched train
   counts) so complexity varies at fixed data volume.
2. **Fixed-train-count control**: hold the number of train pairs constant across
   groups (down-sample the large groups to A₄'s count) and ask whether the
   optimizer boundary appears at fixed data — isolating complexity.
3. **Re-anchor the difficulty axis** on commutator density at matched data, not
   on group order.

The killer-sweep's P0 precondition did its job: it caught the confound *before*
the 75-run Stage-B was spent.

## What DID survive

- The group generators (`groups.py`) are correct (self-test: order/identity/
  inverse/associativity, commutator density monotone Z97=0 → S5=0.942).
- S₅ groks reliably under AdamW (3/3 at both widths, grok 1100–1500) — a solid
  reference cell, consistent with Power et al. / 002's S5 grokking.
- A₅ at d256 groks (2/3) — the capacity dependence is real and measurable.

## Limitations / next

- One train_frac (0.5), one step budget; A₄'s failure could in principle be
  rescued by a much higher train_frac (more of its 144 pairs) — a quick control
  would confirm data-starvation vs intrinsic.
- norm-growth-ratio (P2) not yet analyzed; it is logged per cell for the
  surviving (grokking) groups.

## Stage-B verdict (comparable-|G|² ladder) — the complexity-graded necessity threshold is REFUTED; the only necessity is SGDM's wholesale floor

Stage-B holds dataset size FIXED (group order = |G|², so |G|² pairs fixed within
a rung) and varies commutator density: rung-60 {Z60 cd0, D30 cd.70, A5 cd.92} and
rung-120 {Z120 cd0, D60 cd.72, S5 cd.94}, × {muon, adamw, sgdm} × 5 seeds, d256.

**grok-success rate (5 seeds), de-confounded:**

| |G| | group | cd | Muon | AdamW | SGDM |
|---|---|---|---|---|---|---|
| 60 | Z60 | 0.00 | 5/5 | 5/5 | **0/5** |
| 60 | D30 | 0.70 | 5/5 | 5/5 | **0/5** |
| 60 | A5 | 0.92 | 5/5 | **3/5** | **0/5** |
| 120 | Z120 | 0.00 | 5/5 | 5/5 | **0/5** |
| 120 | D60 | 0.72 | 5/5 | 5/5 | **0/5** |
| 120 | S5 | 0.94 | 5/5 | 5/5 | **0/5** |

**P1 (necessity threshold): REFUTED.** Once dataset size is controlled, **AdamW
groks the entire abelian→S5 ladder** (5/5 everywhere except a mild A5 dip to 3/5),
and Muon is 5/5 throughout — so **orthogonalization is NOT necessary for AdamW at
any complexity rung** (S5 cd 0.94 at |G|=120: AdamW 5/5). There is no
complexity-graded Muon-necessity crossing; the A5 3/5 dip is non-monotone in cd
(S5 has higher cd yet AdamW 5/5 at the larger rung), so it reads as a residual
data/seed effect, not a threshold. **The earlier "necessity" signal was the |G|²
data-size confound that Stage-A (P0) caught.**

**The ONE real necessity is SGDM's wholesale failure**: SGDM is 0/5 on EVERY
group including abelian Z60/Z120 (cd 0). That is an optimizer-family floor
("plain SGD+momentum cannot train this Cayley-table family at all, regardless of
complexity"), NOT a complexity threshold.

**Refinement of findings-002:** 002 read "S5 necessity" (SGDM 0/10 on S5) as
complexity-driven; the de-confounded ladder shows SGDM fails even on the abelian
Z60, so 002's signal is the wholesale SGDM floor + the data-size confound, not a
graded complexity necessity. The "Muon becomes necessary as the group gets
harder" hypothesis (B1's premise) does not survive the de-confound.

**P2 (Muon norm-growth ∝ complexity): EQUIVOCAL.** rung-60 supports it (abelian
Z60 lowest growth 1.18 vs D30/A5 1.86/1.81), but rung-120 reverses (Z120 highest
1.90 vs S5 1.60). Non-monotone — no clean norm-growth × complexity law.

## Figures / data

`results/figures-019/fig_group.png` — de-confounded grok-success by group ×
optimizer at fixed |G|² (Muon/AdamW succeed across the ladder, SGDM floors at 0).
Raw: `results/group_complexity/` per-cell jsonl (Stage-A 18 + Stage-B ladder 90;
grokked, grok_step, final_test_acc, norm_growth_ratio, commutator_density).

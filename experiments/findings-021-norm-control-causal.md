# Findings 021 (Tier-2 / A-ISSUE-3) — Muon's hidden-norm growth is NOT causally necessary for grokking; capping it preserves and *accelerates* grokking

**Status:** Tier-2 follow-up to 002 / Bundle-A (resolves A-ISSUE-3, the red-team's
load-bearing objection that the "growth route" may be a restatement of the Muon
update geometry rather than a mechanism). NOT yet formally chartered (no novelty
killer-sweep run) — documented as a Tier-2 causal result; formal chartering +
novelty deferred. The causal verdict below stands on its own.
**Data:** `results/s5_normctl/` — 40 runs: ceiling k ∈ {∞ (vanilla Muon), 3, 2,
1.5, 1} × 8 seeds. Reuses grokking `train.run` (op=s5, init_scale=1.0,
weight_decay=0.01, steps=20000, eval_every=50, mech=True) — the SAME operating
point as s5_mech/002 where the 6.5× growth was measured, so results are directly
comparable. Intervention: `NormControlledMuon` — a downward-only per-matrix
Frobenius ceiling ‖W‖ ≤ k·‖W‖_init (caps GROWTH only, never inflates, so it does
NOT cancel decoupled weight decay).

## Headline

**Capping Muon's hidden-matrix norm growth does not block grokking on S5 — it
preserves grokking 8/8 at every ceiling (incl. k=1, held at init) and ACCELERATES
it ~10× (median grok step 3525 → 325).** The norm growth is therefore a *byproduct*
of the orthogonalized update (fixed spectral norm ⇒ Frobenius norm inflates each
step), NOT the generalization mechanism. This refutes the A-paper §3.2 framing of
norm growth as "the mechanistic core."

## Dose-response (the manipulated-cause arm)

| ceiling k | grok-rate | median grok_step | median final ‖W_hidden‖ | median test_acc |
|---|---|---|---|---|
| ∞ (vanilla Muon) | 8/8 | 3525 | 311.5 | 0.999 |
| 3   | 8/8 | 575 | 37.6 | 1.000 |
| 2   | 8/8 | 425 | 25.1 | 1.000 |
| 1.5 | 8/8 | 325 | 18.8 | 1.000 |
| 1   | 8/8 | 325 | 12.5 | 1.000 |

Two facts:
1. **The intervention bit.** Final ‖W_hidden‖ drops monotonically with the ceiling
   (311.5 → 12.5); at k=1 the norm is held at init (~12.5) vs vanilla Muon's ~25×
   inflation. So the experiment actually suppressed the growth signature.
2. **Grokking is unaffected in OUTCOME, improved in TIME.** 8/8 grok to test 1.0 at
   every k including k=1; median grok step falls monotonically 3525 → 325 as the
   cap tightens. Suppressing growth does not merely leave grokking intact — letting
   the norm grow freely is mildly *detrimental* (≈10× slower) on this task.

## Interpretation (what it does and does not say)

- **Causal verdict:** on S5, norm growth is **not necessary** for Muon grokking. The
  "growth route" (002/A §3.2) is a *descriptive byproduct* of the orthogonalized
  update, not the mechanism. A-ISSUE-3 is resolved on the NOT-causal side.
- **Consistent with the norm-clock account** (009 / 2603.13331): smaller weights →
  faster grokking, so capping growth toward init accelerates the transition. The
  acceleration is the *expected* direction under the geometric clock, which makes
  the byproduct reading more, not less, credible.
- **Does NOT say** Muon's other behaviours are byproducts — only the S5 hidden-norm
  growth. The acceleration finding is itself a (small, clean) positive contribution.

## Mod-add replication (A-ISSUE-3-add, 2026-06-15) — non-causality REPLICATES

Same NormControlledMuon dose-response on **modular addition** (`results/add_normctl/`,
op=add, k ∈ {∞,3,2,1.5,1} × 8 seeds; `analyze_normctl.py`, which also reproduces the
S5 table above exactly as a sanity check):

| ceiling k | grok-rate | median grok_step | median ‖W_hidden‖ | median test_acc |
|---|---|---|---|---|
| ∞ | 8/8 | 100 | 148.8 | 1.000 |
| 3   | 8/8 | 100 | 37.6 | 1.000 |
| 2   | 8/8 | 100 | 25.1 | 1.000 |
| 1.5 | 8/8 | 100 | 18.8 | 0.999 |
| 1   | 8/8 | 100 | 12.5 | 0.999 |

- **Core causal claim replicates:** capping growth neither blocks nor delays
  grokking — 8/8 grok to test 1.0 at every ceiling incl. k=1, with ‖W_hidden‖ capped
  ~12× (148.8 → 12.5; k=∞ reproduces the uncapped growth). Growth is a byproduct on
  mod-add as on S5.
- **The ~10× acceleration does NOT register here — eval-floor, not a contradiction:**
  under Muon, mod-add groks at the very first eval (step 100, eval_every=100) *even
  uncapped*, so grok_step floors at 100 for every k (k1/k∞ ratio = 1.000). This is
  consistent with findings-018 (Muon groks mod-add near-instantly at λ=0); resolving
  any acceleration would need a finer eval cadence. Acceleration remains an S5 result;
  mod-add confirms only survival + the cap biting.
- **Net:** A-ISSUE-3 non-causality generalizes beyond S5. Verdict:
  `results/figures-021/normctl_verdict.json`.

## Red-team / caveats (self-audited)

- ~~One task (S5), n=8~~ — **now replicated on mod-add** (op=add, n=8/k, above):
  survival + cap-bite reproduce; acceleration is eval-floored on mod-add.
- Downward-only cap is weight-decay-safe by construction (caps growth only; never
  scales up, so never undoes the ηλw shrink) — the wd-cancellation confound the
  naive "project to init" design had is avoided.
- The grok is real (S5 test_acc transitions chance→1.0; grok_step = first ≥0.95).
- The model stays expressive under the cap (groks to 1.0 at k=1), so "function
  clamped too small" is not the failure mode; function-output scale was not logged
  separately (proxy = ‖W_hidden‖, which is the capped quantity) — a logit-RMS log is
  a cheap future add.
- grok_step 325 (k=1) is 6.5 eval intervals (eval_every=50), well above the
  eval-floor; the 3525→325 contrast is far larger than eval-cadence noise.

## Implication for Bundle A (science-substance change)

paper-A §3.2 "norm-growth is the mechanistic core" → **corrected to byproduct**:
the manipulated-cause arm shows growth is not necessary (and capping it
accelerates grokking). The route stays DESCRIPTIVE (as the draft already hedged),
now with a *positive causal test* backing the hedge — this strengthens A (it adds
the manipulated-cause experiment the red-team said was missing) rather than
weakening it. Abstract/§4/§6 "whether growth is causal awaits a norm-controlled
arm" → answered: not causal.

## Novelty / prior-art (killer-sweep 2026-06-15) — PROCEED with scoping

A focused prior-art scan (arXiv/OpenReview/web, 2023–2026) found **no double-kill**,
but it DID surface a parent paper that narrows the claim and MUST be cited-and-distinguished:

- **arXiv:2504.16041 "Muon Optimizer Accelerates Grokking"** already asserts —
  *correlationally* — that "weight-norm growth accompanies grokking but does not
  drive it" (it attributes Muon's speedup to the spectral-norm constraint / softmax-
  collapse avoidance; mod-arithmetic only, no S5, NO norm-cap intervention). So the
  bare statement "growth doesn't drive grokking" is NOT ours to claim as new.
- **Omnigrok (2210.01117):** a *symmetric* constant-norm-sphere projection (can inflate
  AND deflate) nearly eliminates grokking; SGD/Adam, modular+MNIST, not Muon, not a
  downward-only ceiling.
- **Norm-separation delay law (2603.13331):** correlational norm-clock (AdamW/SGD, no
  Muon, no S5, no imposed ceiling) — predicts smaller norm ratio → faster grok.
- **Spectral entropy collapse (2604.13123):** a norm-matched control concludes norm is
  not the driver, but the control *delays* grokking (opposite direction), AdamW only.

**What remains genuinely ours (the gap):** the **interventional, Muon-specific causal
test** — a *downward-only* per-matrix Frobenius ceiling ‖W‖≤k·‖W‖_init (caps growth
without inflating, so it cannot be confounded with weight decay), with a clean
dose-response (k∈{∞,3,2,1.5,1}) showing **preserve-grokking-8/8 + ~10× acceleration**,
replicated on **non-abelian S5** (where the Muon-grokking/norm-clock literature is
otherwise absent). Asserting non-causality (2504.16041) vs. *demonstrating* it by
intervention is exactly the contribution.

**Required framing (do NOT overclaim):** claim the *causal cap*, the downward-only
design, the preserve-8/8 + ~10× result, and the S5 extension — NOT the bare
"growth doesn't drive grokking" (2504.16041 said it), nor "norm constraints can
affect grokking in general" (Omnigrok), nor "small-norm solutions generalize" (norm-
minimization theory 2511.01938/2505.20172). Residual risk: OpenReview submission
queues not exhaustively scanned. **Verdict: PROCEED (scoped).**

## Figures / data

`results/s5_normctl/` (40 runs). Dose-response figure (grok_step + ‖W‖ vs ceiling)
DEFERRED to the final unified packaging pass (science-before-packaging directive).

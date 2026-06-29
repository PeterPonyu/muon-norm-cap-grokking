# Mathematical backing for Papers A & B — feasibility audit (2026-06-21)

**Scope:** READ-ONLY analysis. Goal: identify which empirical findings in Paper A
(Muon non-contraction / growth route) and Paper B (grokking clock: precision ×
objective × weight decay) admit a rigorous or semi-rigorous analytic argument that
would *predict or bound* the measured quantity, strengthening the papers for a
dynamics-framed SCI venue (Physica D / Neurocomputing).

**Sources read:** `papers/A/main.tex`, `papers/B/main.tex`,
`experiments/findings-{002,009,018,021}.md`, `experiments/grokking/muon.py`,
`experiments/grokking/train.py`, `experiments/grok_numerics/{run_numerics.py,train_numerics.py}`.

**Verdict in one line:** Three arguments are worth adding. The **bf16 underflow
closed-form threshold (Paper B)** is *tractable now, predictive, low-risk* — it
should be added as a short proposition. The **orthogonalized-update norm-growth
lower bound (Paper A)** is *tractable-with-modest-work* and genuinely predictive of
the growth signature (it ties √(rank)/step growth to the measured 6.5×). The
**norm-clock → delay-slope argument (Paper B fp32 −0.50)** is only *plausible* and
mostly *rationalizes* rather than predicts the specific exponent — include only as a
heuristic scaling remark, flagged as such. The **causal-cap acceleration (Paper A)**
admits a clean heuristic ("norm-clock") argument but not a rigorous one; include as
a one-line consistency remark, not a theorem.

---

## Summary

The single highest-value, lowest-risk addition is the **bf16 weight-decay
underflow proposition for Paper B**. It converts the paper's current empirical
"smoking gun" (final ‖w‖ is λ-invariant at ≈37) into a *predicted* result: a
one-line closed-form threshold `λ* = 2^-(p-1)/η` (with p = 8 bf16 significand bits)
predicts, *before looking at any data*, that the decay is silently rounded away for
every λ in the swept range and that ‖w‖ is λ-invariant. With η = 1e-3 the threshold
is λ* ≈ 3.9, well above the largest swept λ = 1.0 — so the flatness is forced, not
observed. This is exactly the kind of "closed-form threshold corroborating an
empirical regime" that raises acceptance odds at a dynamics venue, and it is
referee-safe because it is elementary floating-point arithmetic that the data then
confirm.

---

## Paper B — per-claim table

| # | Claim (quoted) | Empirical quantity | Proposed argument | Feasibility | Worth it? | Risk |
|---|---|---|---|---|---|---|
| B1 | "the final weight norm is λ-invariant … the decoupled weight-decay update term ηλw falls below the local bf16 rounding threshold … so the decay is effectively rounded away" (sec 3.1 ii; case study 3.1.1) | bf16 final ‖w‖ = 37.03–37.08 across λ∈{0,0.01,0.1,1.0}; fp32 falls 42.6→22.9 | **Closed-form underflow threshold** (full derivation below). Decoupled decay is a relative shrink of size ηλ per step; in round-to-nearest bf16 it is annihilated when ηλ < 2^-(p-1) (p=8 ⇒ 2^-8 ≈ 3.9e-3). With η=1e-3, λ*=2^-8/η≈3.9 > max swept λ=1.0 ⇒ flat for the WHOLE sweep, predicted. | **Tractable-now** | **Yes — strongest candidate** | **Low.** Elementary FP arithmetic; data confirm. Only nuance: gradient term in the same add can mask decay even more easily (makes the bound *conservative*, i.e. safe). State as a sufficient condition. |
| B2 | "at fp32 the cross-entropy slope is −0.50 … The norm-clock prediction (slope → −1 as γ_eff=ηλ dominates) is only half-realized" (sec 3.1 i) | fp32 delay–λ log-log slope −0.50 ± 0.09 | **Norm-clock timescale argument**: weight-norm relaxation time τ ~ 1/(ηλ) under decoupled decay; if grok onset ∝ time to cross a norm threshold, delay ∝ 1/(ηλ) ⇒ slope −1 asymptotically. The −0.50 is the *non-asymptotic* regime where the gradient-driven norm-growth competes with decay. | **Plausible-with-work** (the −1 asymptote is derivable; the measured −0.50 is NOT predicted, only bounded as |slope| ≤ 1) | Partial — include as a heuristic scaling remark bounding the slope to [−1,0], NOT as a derivation of −0.50 | **Medium.** A referee will note the argument predicts −1, not −0.50; the paper already says "half-realized." Honesty: this RATIONALIZES the sign and the ≤1 magnitude, does not PREDICT the value. |
| B3 | "fp64 … traps low-weight-decay runs, which fail to generalize at all" (sec 3.1 iii); "fp64 repair … saturation rate → 1.0 with 0/3 grokking" (findings-009 P2b) | fp64 λ=0: 0/3 grok, saturation→1.0 | **Softmax-absorption argument**: once logit gap Δ exceeds the mantissa width, softmax prob saturates to exactly 1 and the CE gradient on fitted examples underflows to 0 cleanly; wider mantissa (fp64, 52 bits) merely *delays then completes* absorption with no escape noise. Can bound the step at which gradient → 0 as function of mantissa bits + logit growth rate. | **Plausible-with-work** | Optional — a short "why higher precision traps" paragraph would be elegant and on-theme (it inverts the naive "more precision = better" intuition) | **Medium.** The "absorption noise as escape mechanism" is speculative (findings-009 flags it). Keep to the clean direction: more mantissa ⇒ later but *cleaner* gradient nulling ⇒ no escape. |
| B4 | "probability-space MSE has a finite λ_c≈0.03 … cross-entropy, StableMax, logit-MSE … λ_c≈0" (sec 3.2) | prob-MSE λ_c∈(0.01,0.03); others ≈0 | **Loss-curvature / fixed-point argument**: at the memorized solution, prob-space MSE has vanishing gradient on saturated outputs (sigmoid/softmax derivative → 0), so without a decay force the iterate is near a stable non-generalizing fixed point; CE keeps an O(1) logit-scaling gradient. A linear-stability analysis of the memorized fixed point could predict *which objectives need a finite λ_c*. | **Hard/Open** (requires modeling the memorized fixed point's Jacobian per objective; non-trivial) | No — too much work for the payoff; the within-MSE parameterization point is already the honest contribution | **High.** Easy to get wrong; a hand-wavy version invites rejection. Leave empirical. |
| B5 | "the nominal Edge-of-Stability 2/η marker is loss-dependent … MSE loses a finite sharpness readout at large step size" (sec 3.3) | MSE diverges 6/6 at η=0.05; CE finite | **None new worth adding.** The objective-dependence of EoS sharpness is already Cohen 2021 (prior art); adaptive λmax ≪ 2/η is Cohen 2022. The paper correctly reports this as a negative control. | n/a | No | n/a — adding theory here would re-derive prior art. |

### FULL WORKED DERIVATION — B1 (bf16 weight-decay underflow), numbers plugged in

**Setup (verified from code).** `train_numerics.py` casts the *entire* model to the
target dtype (`model = model.to(torch.bfloat16)`) with **no fp32 master weights**;
`torch.optim.AdamW` then updates the bf16 parameters in place. Decoupled (AdamW)
weight decay applies the multiplicative shrink

    w ← w · (1 − ηλ)            (equivalently  w ← w − ηλ·w , decoupled from the Adam step)

with η = 1e-3 (`cfg.lr`) and λ the swept weight decay ∈ {0, 0.01, 0.03, 0.1, 0.3, 1.0}.
(Confirmed: `train.py:32 lr=1e-3`; Muon's own decay `p.mul_(1 − lr*wd)` at
`muon.py:83` is the same form for the hybrid path.)

**bf16 format.** bfloat16 has a 7-bit stored fraction ⇒ p = 8 significand bits
(1 implicit + 7 stored). For a stored value w with 2^e ≤ |w| < 2^(e+1), the spacing
between representable numbers (the ULP) is

    ULP(w) = 2^(e − (p−1)) = 2^(e−7),

so the *relative* spacing is ULP(w)/|w| ∈ [2^-7, 2^-8) (it is 2^-7 just above a
power of two and 2^-8 just below the next). Round-to-nearest-even leaves a value
unchanged under a perturbation δ iff |δ| < ½·ULP(w), i.e. iff the **relative**
perturbation satisfies

    |δ|/|w|  <  ½ · ULP(w)/|w|  ∈  [2^-8, 2^-7).

**The decay step's relative size.** The amount subtracted by decoupled decay is
δ = ηλ·w, whose relative size is exactly

    |δ|/|w| = ηλ      (independent of the weight magnitude — this is the key point).

**Underflow condition.** Decay is silently rounded away on a parameter iff
ηλ < ½·ULP relative spacing, i.e. (taking the conservative round-to-nearest bound)

    ┌─────────────────────────────────────────────┐
    │   ηλ  <  2^-(p-1)  =  2^-8  ≈  3.906e-3        │   (sufficient condition for annihilation)
    └─────────────────────────────────────────────┘

Equivalently a **closed-form critical weight decay** below which bf16 decay is inert:

    λ*  =  2^-(p-1) / η  =  2^-8 / η.

**Numbers plugged in (η = 1e-3, p = 8):**

| λ | relative decay ηλ | bf16 half-ULP 2^-8 ≈ 3.91e-3 | outcome |
|---|---|---|---|
| 0.01 | 1.0e-5 | 3.91e-3 | rounds away |
| 0.03 | 3.0e-5 | 3.91e-3 | rounds away |
| 0.10 | 1.0e-4 | 3.91e-3 | rounds away |
| 0.30 | 3.0e-4 | 3.91e-3 | rounds away |
| 1.00 | 1.0e-3 | 3.91e-3 | **rounds away** |

Critical value:  **λ* = 2^-8 / 1e-3 ≈ 3.91.**  (Truncation/full-ULP convention
gives λ* = 2^-7/η ≈ 7.81 — even larger.) Either way **every swept λ ≤ 1.0 lies
below λ***, so the decoupled decay never changes a single stored bf16 weight at
round-to-nearest. The model's norm is therefore set entirely by the gradient
dynamics, which equilibrate ‖w‖ ≈ 37 *independent of λ* — exactly the measured
λ-invariant 37.03–37.08 (findings-009 P1; Paper B sec 3.1.1). The decay would only
begin to "bite" at λ ≳ 3.9, outside the experiment.

**Why this is a prediction, not a rationalization.** The threshold λ* = 2^-8/η is
derived from the format and η *alone*; it forecasts (i) flatness for the entire
swept range, (ii) the exact λ above which flatness would break (≈3.9), and (iii)
that raising η would lower λ* and partially restore decay (a falsifiable
cross-check the paper could cite or run). None of these uses the measured ‖w‖.

**Conservativeness / honesty note.** In the actual AdamW step the decay add is
combined with the (typically larger) Adam update in the same accumulation, so the
small δ = ηλw can be swamped *even more* readily than the isolated-step bound says.
Thus ηλ < 2^-8 is a **sufficient** condition for annihilation; the true onset of
"decay starts to matter" may sit slightly higher than λ*, never lower. State the
proposition as a sufficient condition and the data confirm it. Also state the
scope honestly (the manuscript already does): this is a *pure-bf16 / no-fp32-master*
path; mixed-precision stacks with fp32 masters keep the decay and are unaffected.

**Suggested manuscript insertion (Paper B sec 3.1, ~4 lines):** a boxed
"Proposition (bf16 decay underflow)" giving λ* = 2^-(p-1)/η, the plugged-in
λ* ≈ 3.9 > 1.0, and one sentence that the λ-invariant ‖w‖≈37 is the predicted
consequence. The scale-up audit (p=251, d=256, sec 4) where bf16 ‖w‖ is again
λ-invariant (≈58) is a free second confirmation of the *same* η-only threshold —
note that the prediction is scale-free (λ* does not depend on width or modulus),
which is exactly why the artifact reproduced at the larger scale.

---

## Paper A — per-claim table

| # | Claim (quoted) | Empirical quantity | Proposed argument | Feasibility | Worth it? | Risk |
|---|---|---|---|---|---|---|
| A1 | "Every Muon grok event on S5 shows hidden-norm-ratio growth of 6.5±0.7, versus 0/20 growth-signature events for AdamW and SGDM" (sec 3.2) | S5 ‖W‖ growth ratio 6.5±0.7 | **Orthogonalized-update growth lower bound** (derivation below). Newton–Schulz pushes each hidden-matrix update to unit singular values, so per-step ‖ΔW‖_F ≈ √r (r = rank ≈ min(d, ...)), and without a contraction force ‖W‖ grows at least like a random-walk/ballistic accumulation ⇒ monotone growth, contrasted with AdamW whose normalized coordinate updates do not have a fixed Frobenius scale. Predicts the *direction and monotonicity* of the signature and a growth-rate floor. | **Tractable-with-work** | **Yes (2nd best)** — gives the central "growth route" descriptor a quantitative floor and a clean contrast with AdamW | **Medium.** The exact 6.5× is set by step count to grok and decay, so DON'T claim it predicts 6.5. Claim it predicts (a) growth is forced under Muon, (b) AdamW has no analogous fixed per-step Frobenius scale, (c) a per-step lower bound. |
| A2 | "capping Muon's hidden-norm growth … PRESERVES grokking 8/8 … and ACCELERATES it ∼10× (grok step 3525→325)" (sec 3.2, findings-021) | S5 grok step 3525→325 under cap | **Norm-clock consistency argument**: if grokking onset is governed by the weight-norm-separation clock (smaller ‖w‖ ⇒ faster grok, the geometric account of Paper B), then forcing ‖w‖ to stay near init short-circuits the slow growth phase ⇒ earlier grok. This *rationalizes the sign* of the acceleration. | **Plausible (heuristic only)** | Partial — one-line consistency remark linking to the norm-clock; NOT a theorem | **Medium-high.** It explains why acceleration is in the *expected* direction but cannot derive 10×. findings-021 itself frames it as "consistent with the norm-clock," which is the right altitude. Do not over-formalize. |
| A3 | "Grokking occurs at effective rank 111/128, refuting a universal grokking⇔rank-compression reading" (sec 3.2) | eff-rank 111/128 at grok; memorizing soln ≈23 | **None needed** — this is a *falsification* of an existing theory claim; a counterexample needs no positive theory. Optionally: orthogonalized updates *preserve* spectral entropy by construction (unit singular values ⇒ flat spectrum ⇒ high effective rank), which would *predict* high eff-rank under Muon. | Orthogonalization⇒high-rank link is **Tractable-now** as a short remark | Optional — a one-sentence "Muon's unit-singular-value updates bias toward flat spectra, hence high eff-rank" *predicts* the 111/128 direction and strengthens A3 | **Low** if scoped to "biases toward, does not force." The 6.5× growth coexisting with flat spectrum is itself consistent (Frobenius grows while singular values stay balanced). |
| A4 | "Muon's inter-child loss barrier is uniformly high … 0/5 basins locked vs AdamW 2/5" (sec 3.3, LMC) | barrier median 3.18 (Muon) vs 0.56 (AdamW) | No tractable theory; LMC barriers at n=5 are underpowered (Fisher p≈0.22). | Hard/Open | No | High — paper already flags as directional; theory would over-promise. |
| A5 | "massive-activation spike … monotone in effective step size within each family" (sec 3.4) | spike 4787 (Muon) vs 245 (AdamW), overlapping by lr | **None worth adding.** Operating-point statement; a scaling "larger steps ⇒ larger transient activations" is folklore and the paper already frames it as operating-point, not amplification. | n/a | No | n/a |
| A6 | "SGDM fails to train S5 at all — an optimizer-family floor" (sec 3.6) | SGDM 0/5 every rung incl. abelian Z60 | No clean theory (would need a trainability/conditioning argument for plain momentum vs adaptive/orthogonalized on these losses). | Hard/Open | No | High — the cliff is empirical (width×lr×target); a theory would be a separate project. |

### Derivation sketch — A1 (orthogonalized-update norm growth lower bound)

**Mechanism (verified from `muon.py`).** Each Muon step is
`p.add_(update, alpha=−ηλ_scale)` where `update = NS5(buf)` has been passed through
the quintic Newton–Schulz iteration, which pushes the singular values of the update
toward 1. So the applied increment is `ΔW = −η·s·U` where U ≈ semi-orthogonal
(σ_i(U) ≈ 1) and s = max(1, d_out/d_in)^½ is a fixed shape scale. Hence each step
has an (approximately) **fixed Frobenius norm**

    ‖ΔW‖_F ≈ η·s·√r ,     r = number of nonzero singular values ≈ min(d_out, d_in).

This is the crucial contrast with AdamW: AdamW's per-coordinate normalized update
has Frobenius norm ≈ η·√(#params) but with *no spectral structure* and, more
importantly, AdamW's contraction-via-decay route drives ‖W‖ *down*. Muon's update
injects a near-constant-energy, full-rank increment every step.

**Growth lower bound.** Write W_t = W_0 + Σ_{k<t} ΔW_k. Two regimes bound ‖W_t‖_F:

- *Aligned / ballistic upper-ish behavior*: if successive updates retain directional
  correlation (momentum 0.95 in Muon enforces strong step-to-step correlation), the
  increments add coherently and ‖W_t‖_F grows ~ linearly, ‖W_t‖_F ≳ ‖W_0‖_F + c·η·s·√r·t
  minus the decay pull ηλ‖W‖ (small).
- *Decorrelated / diffusive lower bound*: even if increments were orthogonal
  (worst case for growth), E‖W_t‖_F² = ‖W_0‖² + Σ‖ΔW_k‖² ⇒ ‖W_t‖_F ≳ √(‖W_0‖² + η²s²r·t),
  i.e. growth at least like √t.

Either way ‖W_t‖_F is **monotone increasing** until decay balances it at the
fixed point ‖W*‖ where ηλ‖W*‖ ≈ η·s·√r·(effective per-step coherence), i.e.

    ‖W*‖_F ≈ (s·√r / λ) · (coherence factor).

Two predictions follow that the data support: (i) the growth ratio is *positive and
≫1* for Muon by construction (vs AdamW's contraction) — this is the 0/20 vs 6/6
dissociation in findings-002; (ii) at small λ (=0.01 here) the balance norm ‖W*‖ is
large (∝ 1/λ), consistent with the large pre-cap final ‖W_hidden‖ ≈ 309 (case
study) and the 6.5× ratio over the memorization-time norm.

**Honest limits.** The bound predicts *direction, monotonicity, a √t floor, and a
1/λ balance scaling* — it does NOT predict the number 6.5 (that requires the
grok-time t and the coherence factor, both empirical). The "coherence factor"
hides the momentum/curvature interaction; a referee could ask for it. Recommended
framing: present as a **growth-rate floor / sign argument** ("orthogonalized updates
inject a fixed-Frobenius-energy, full-rank increment per step, so absent
contraction the hidden norm grows at least like √t and balances at ‖W*‖ ∝ 1/λ;
AdamW has no such fixed per-step Frobenius scale and contracts via decay") — this
*predicts the growth signature and the AdamW contrast*, which is the load-bearing
claim, while staying honest that the magnitude is empirical.

This same construction *predicts A3* for free: unit-singular-value updates bias the
weight spectrum toward flatness, so the effective (entropy) rank stays high —
consistent with eff-rank 111/128 and the explicit "Frobenius grows while singular
values stay balanced" picture.

---

## Cross-cutting recommendations (prioritized)

1. **ADD: Paper B bf16 underflow proposition** — Tractable-now, predictive,
   low-risk, high venue-fit. ~4–6 lines + boxed λ* = 2^-(p-1)/η. (Worked above.)
   Impact: converts the paper's central artifact from "observed" to "predicted,"
   and the scale-up reproduction becomes a confirmation of a scale-free prediction.

2. **ADD: Paper A orthogonalized-update growth-floor argument** —
   Tractable-with-work, predicts direction/monotonicity/√t-floor/1/λ-balance and
   the AdamW contrast; explicitly disclaim the magnitude. Doubles as the A3
   high-rank prediction. Impact: gives the "growth route" a quantitative spine.

3. **INCLUDE AS HEURISTIC REMARK (not theorem): Paper B norm-clock slope bound**
   (|slope| ≤ 1, asymptote −1, measured −0.50 = non-asymptotic) and **Paper A
   causal-cap sign** (norm-clock ⇒ acceleration expected). Both *rationalize*; flag
   them as scaling heuristics. Do NOT claim either predicts its number.

4. **DO NOT ADD theory for:** B4 (MSE λ_c), B5 (EoS, prior art), A4 (LMC,
   underpowered), A5 (activations), A6 (SGDM floor). These are Hard/Open or
   prior-art; a weak argument would *lower* acceptance odds.

---

## Honesty ledger (predict vs rationalize)

| Argument | Predicts (forecasts the number/regime a priori) | Only rationalizes (explains sign/direction post hoc) |
|---|---|---|
| B1 bf16 underflow | **YES** — λ*≈3.9, flat for all swept λ, scale-free | — |
| A1 growth floor | Partial — direction, monotonicity, √t floor, 1/λ balance | the magnitude 6.5× |
| A3 high-rank from flat-spectrum | direction (high eff-rank) | — |
| B3 fp64 trap | direction (more mantissa ⇒ later+cleaner null) | the escape-noise speculation |
| B2 slope bound | bound |slope|≤1, asymptote −1 | the specific −0.50 |
| A2 cap acceleration | — | the sign of the speed-up |

---

## References (file:line)

- `papers/B/main.tex:160-169` — bf16 "rounding artifact" claim, the target of B1.
- `papers/B/main.tex:194-212` — bf16 case study: ‖w‖=37.03–37.08 λ-invariant; the
  exact numbers the B1 proposition predicts.
- `papers/B/main.tex:152-158` — fp32 slope −0.50, "half-realized norm clock" (B2).
- `papers/B/main.tex:171-176` — fp64 low-λ trap (B3).
- `papers/B/main.tex:229-235` — prob-MSE λ_c≈0.03 (B4).
- `experiments/grok_numerics/train_numerics.py:16-17,108-110` — full-model dtype
  cast `model.to(dtype)`, NO fp32 master weights (B1 mechanism confirmed).
- `experiments/grok_numerics/train_numerics.py:86,117-119` — lr=1e-3, AdamW
  decoupled decay (B1 η value).
- `experiments/grokking/train.py:32` — `lr: float = 1e-3` (η for λ* computation).
- `experiments/grokking/muon.py:17-36` — Newton–Schulz `zeropower_via_newtonschulz5`
  (unit singular values; A1 mechanism).
- `experiments/grokking/muon.py:79-84` — `update = NS5(...); p.add_(update, alpha=-lr*scale)`
  with `scale = max(1, d_out/d_in)^0.5`; decoupled decay `p.mul_(1-lr*wd)` (A1 + B1 hybrid).
- `papers/A/main.tex:159-168` — S5 norm-growth 6.5±0.7, eff-rank 111/128 (A1, A3).
- `papers/A/main.tex:170-179` — causal cap 3525→325, "norm-clock (smaller weights
  grok faster)" (A2).
- `experiments/findings-009.md:58-78` — bf16 underflow mechanism + verified ‖w‖
  table (B1 ground truth).
- `experiments/findings-021-norm-control-causal.md:20-52` — dose-response table,
  "consistent with the norm-clock account" (A2).
- `experiments/findings-002.md:36-42` — 6.52±0.68 growth, 0/20 AdamW/SGDM (A1).

"""Direction 008 — Bigram-Backcopy (BB) synthetic task (2410.13835 setting).

Sequences come from a fixed bigram Markov chain over a small vocabulary, except
after TRIGGER tokens, where the next token is a deterministic BACKCOPY of the
token that preceded the trigger:

    x[0] = BOS
    x[1] ~ uniform over non-trigger, non-BOS tokens
    x[t] = x[t-2]                      if x[t-1] is a trigger and x[t-2] is not
         ~ P( . | x[t-1])              otherwise

This is the native toy setting in which 2410.13835 derives the active-dormant /
mutual-reinforcement mechanism for the extreme-token triad: heads must be ACTIVE
(attend locally) after triggers and can go DORMANT (attend to BOS = sink) on
plain bigram positions. The chain matrix P is fixed per task seed (NOT per
sequence), row-stochastic, never emits BOS, never transitions trigger->trigger,
and boosts trigger columns so backcopy positions are a non-trivial fraction of
the stream.

Edge rule: if both x[t-1] and x[t-2] are triggers (possible only via a copied
trigger, which the trigger->trigger ban makes rare) we fall back to chain
sampling and the position is excluded from BOTH role masks.

Batch layout (mirrors induction_emergence/data.py): X = seq[:, :-1],
Y = seq[:, 1:], and two boolean role masks aligned with Y positions:

    backcopy_mask[b, t]  — Y[b, t] is a deterministic backcopy target
                            (x[t] trigger, x[t-1] non-trigger), predictable
                            only by ATTENDING BACK — the "active" role.
    bigram_mask[b, t]    — Y[b, t] is a plain chain transition (x[t] is not a
                            trigger, t >= 1) — the "dormant" role.

target_mask = backcopy_mask | bigram_mask (position 0 — predicting x[1] from
BOS — is unpredictable and excluded). Everything is deterministic given
(chain_seed, batch seed); generation is CPU-side then moved to device.

Self-test: `python data.py` checks shapes, the backcopy oracle (Y equals
X[t-1] on every backcopy position), role-mask disjointness, trigger frequency,
and cross-seed chain determinism.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BBSpec:
    vocab_size: int = 64          # includes BOS = 0; emissions are 1..V-1
    seq_len: int = 128            # full sequence length (X/Y are seq_len-1)
    n_triggers: int = 3           # trigger token ids: V-n_triggers .. V-1
    trigger_boost: float = 4.0    # multiplier on trigger columns of P
    row_temp: float = 1.0         # softmax temperature of random row logits
    chain_seed: int = 1234        # fixes the Markov matrix P (task identity)

    @property
    def bos(self) -> int:
        return 0

    @property
    def trigger_ids(self) -> tuple[int, ...]:
        return tuple(range(self.vocab_size - self.n_triggers, self.vocab_size))


def batch_seed(seed: int, j: int) -> int:
    """Deterministic per-(run seed, step/batch index) sampling seed."""
    return (seed * 1_000_003 + j * 7919 + 17) % (2**31 - 1)


def transition_matrix(spec: BBSpec) -> torch.Tensor:
    """Fixed row-stochastic bigram matrix P [V, V] (CPU, float32).

    Rows are softmax of Gaussian logits (deterministic via chain_seed);
    column BOS is zeroed (never emitted); trigger columns are boosted by
    trigger_boost; trigger->trigger transitions are zeroed; rows renormalized.
    """
    g = torch.Generator().manual_seed(spec.chain_seed)
    V = spec.vocab_size
    logits = torch.randn(V, V, generator=g) / max(spec.row_temp, 1e-6)
    P = torch.softmax(logits, dim=-1)
    P[:, spec.bos] = 0.0
    trig = torch.tensor(spec.trigger_ids, dtype=torch.long)
    P[:, trig] *= spec.trigger_boost
    for a in spec.trigger_ids:                  # no trigger -> trigger
        P[a, trig] = 0.0
    P = P / P.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    return P


def is_trigger_lut(spec: BBSpec) -> torch.Tensor:
    """Boolean lookup table [V]: True iff token id is a trigger."""
    lut = torch.zeros(spec.vocab_size, dtype=torch.bool)
    lut[list(spec.trigger_ids)] = True
    return lut


def sample_batch(spec: BBSpec, n: int, seed: int, device: str = "cpu"):
    """Sample n BB sequences; return (X, Y, backcopy_mask, bigram_mask).

    X, Y: int64 [n, seq_len-1]; masks: bool [n, seq_len-1] aligned with Y.
    """
    g = torch.Generator().manual_seed(seed)
    P = transition_matrix(spec)
    trig_lut = is_trigger_lut(spec)
    L = spec.seq_len
    V = spec.vocab_size

    seq = torch.empty(n, L, dtype=torch.long)
    seq[:, 0] = spec.bos
    # x[1]: uniform over non-BOS, non-trigger tokens
    plain = torch.tensor(
        [v for v in range(1, V) if v not in spec.trigger_ids], dtype=torch.long)
    idx = torch.randint(len(plain), (n,), generator=g)
    seq[:, 1] = plain[idx]

    for t in range(2, L):
        prev = seq[:, t - 1]
        prev2 = seq[:, t - 2]
        do_copy = trig_lut[prev] & ~trig_lut[prev2]
        sampled = torch.multinomial(P[prev], 1, generator=g).squeeze(1)
        seq[:, t] = torch.where(do_copy, prev2, sampled)

    X = seq[:, :-1]
    Y = seq[:, 1:]
    T = X.shape[1]
    pos = torch.arange(T)
    x_is_trig = trig_lut[X]                                  # [n, T]
    prev_is_trig = torch.zeros_like(x_is_trig)
    prev_is_trig[:, 1:] = trig_lut[X[:, :-1]]
    backcopy_mask = x_is_trig & ~prev_is_trig & (pos >= 1)[None]
    bigram_mask = ~x_is_trig & (pos >= 1)[None]

    return (X.to(device), Y.to(device),
            backcopy_mask.to(device), bigram_mask.to(device))


# ---------------------------------------------------------------------------
# Self-test: `python data.py`
# ---------------------------------------------------------------------------
def _self_test() -> int:
    spec = BBSpec()
    ok = True

    X, Y, bc, bg = sample_batch(spec, n=64, seed=0)
    print(f"SELF-TEST BB batch: X={tuple(X.shape)} Y={tuple(Y.shape)} "
          f"backcopy={tuple(bc.shape)} bigram={tuple(bg.shape)}")

    # 1. roles are disjoint and exclude position 0
    if bool((bc & bg).any()):
        ok = False; print("  FAIL: role masks overlap")
    if bool(bc[:, 0].any()) or bool(bg[:, 0].any()):
        ok = False; print("  FAIL: position 0 must be excluded from both roles")

    # 2. backcopy oracle: on every backcopy position, Y[t] == X[t-1]
    rows, cols = torch.nonzero(bc, as_tuple=True)
    oracle = X[rows, cols - 1]
    match = float((Y[rows, cols] == oracle).float().mean()) if len(rows) else 0.0
    print(f"  backcopy positions = {len(rows)} "
          f"({len(rows) / bc.numel():.3f} of stream), oracle match = {match:.4f}")
    if len(rows) == 0 or match < 1.0:
        ok = False; print("  FAIL: backcopy oracle must match exactly")

    # 3. trigger frequency in a sane band (signal present, chain dominant)
    frac = len(rows) / bc.numel()
    if not (0.02 <= frac <= 0.40):
        ok = False; print(f"  FAIL: backcopy fraction {frac:.3f} outside [0.02, 0.40]")

    # 4. BOS only at position 0; chain matrix deterministic across calls
    if bool((X[:, 1:] == spec.bos).any()) or bool((Y == spec.bos).any()):
        ok = False; print("  FAIL: BOS emitted mid-sequence")
    if not torch.equal(transition_matrix(spec), transition_matrix(spec)):
        ok = False; print("  FAIL: transition matrix not deterministic")
    X2, _, _, _ = sample_batch(spec, n=64, seed=0)
    if not torch.equal(X, X2):
        ok = False; print("  FAIL: sample_batch not deterministic for fixed seed")

    print("DATA SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())

"""Direction 019 — group-complexity ladder generators (the TASK-SPACE axis).

Generalizes 002/grokking's `make_s5_dataset` (S5 Cayley table) to a complexity
ladder of finite groups, so the SAME GrokTransformer testbed can be swept from
abelian to non-abelian:

  - Z_n  (cyclic, ABELIAN)        : compose = (a+b) mod n            commutator density 0
  - D_n  (dihedral, non-abelian)  : elements (flip, rot), 2n total
  - A_n  (alternating, non-abelian): even permutations of n points
  - S_n  (symmetric, non-abelian) : all permutations of n points

Each group exposes `elements()` (canonical list) and `compose(i, j)` on element
indices; `make_group_dataset(spec)` builds the full Cayley-table dataset
X=[a,b,EQ], Y=index(a·b) exactly like make_s5_dataset, with vocab = |G|+1.
`commutator_density(spec)` = fraction of ordered pairs with a·b ≠ b·a (the
preregistered B1 difficulty proxy; 0 for abelian, rises with non-commutativity).

Ladder used by 019: Z_97 (|G|=97) → D_12 (24) → A_4 (12) → A_5 (60) → S_5 (120).
Run `python groups.py` for the self-test (closure, identity, associativity,
inverses, order counts, commutator-density monotonicity abelian<non-abelian).
"""
from __future__ import annotations

import itertools

import torch


# ---------------------------------------------------------------------------
# group element tables: each returns (elements_list, compose_fn_on_indices)
# ---------------------------------------------------------------------------
def _cyclic(n):
    els = list(range(n))                      # element k = rotation by k
    idx = {k: k for k in els}
    def comp(i, j):
        return (els[i] + els[j]) % n
    return els, idx, comp


def _dihedral(n):
    # element (f, r): f in {0,1} flip, r in {0..n-1} rotation; |D_n| = 2n.
    # multiplication (canonical): (f1,r1)·(f2,r2) =
    #   (f1 xor f2,  (r1 + (-1)^f1 * r2) mod n)
    els = [(f, r) for f in (0, 1) for r in range(n)]
    idx = {e: i for i, e in enumerate(els)}
    def comp(i, j):
        f1, r1 = els[i]; f2, r2 = els[j]
        f = f1 ^ f2
        r = (r1 + (1 - 2 * f1) * r2) % n
        return idx[(f, r)]
    return els, idx, comp


def _perm_group(n, even_only):
    perms = [p for p in itertools.permutations(range(n))
             if (not even_only or _is_even(p))]
    idx = {p: i for i, p in enumerate(perms)}
    def comp(i, j):
        pa, pb = perms[i], perms[j]
        c = tuple(pa[pb[k]] for k in range(n))
        return idx[c]
    return perms, idx, comp


def _is_even(p):
    # parity of a permutation via cycle decomposition
    n = len(p); seen = [False] * n; transpositions = 0
    for k in range(n):
        if seen[k]:
            continue
        j = k; clen = 0
        while not seen[j]:
            seen[j] = True; j = p[j]; clen += 1
        transpositions += clen - 1
    return transpositions % 2 == 0


def get_group(spec: str):
    """spec like 'Z97','D12','A4','A5','S5' -> (elements, idx, compose_fn)."""
    kind, n = spec[0], int(spec[1:])
    if kind == "Z":
        return _cyclic(n)
    if kind == "D":
        return _dihedral(n)
    if kind == "A":
        return _perm_group(n, even_only=True)
    if kind == "S":
        return _perm_group(n, even_only=False)
    raise ValueError(f"unknown group spec {spec!r}")


def group_order(spec: str) -> int:
    els, _, _ = get_group(spec)
    return len(els)


def make_group_dataset(spec: str, device: str = "cpu"):
    """Full Cayley-table dataset for group `spec` (mirrors make_s5_dataset).

    X: [|G|^2, 3] = [a, b, EQ]; Y = index(element_a · element_b). vocab = |G|+1.
    """
    els, _, comp = get_group(spec)
    n = len(els)
    A, B, Y = [], [], []
    for i in range(n):
        for j in range(n):
            A.append(i); B.append(j); Y.append(comp(i, j))
    X = torch.stack([torch.tensor(A), torch.tensor(B),
                     torch.full((n * n,), n)], dim=1).to(device)
    return X, torch.tensor(Y, device=device)


def commutator_density(spec: str) -> float:
    """Fraction of ordered pairs (a,b) with a·b != b·a (B1 difficulty proxy)."""
    els, _, comp = get_group(spec)
    n = len(els)
    noncomm = sum(1 for i in range(n) for j in range(n)
                  if comp(i, j) != comp(j, i))
    return noncomm / (n * n)


LADDER = ["Z97", "D12", "A4", "A5", "S5"]


# ---------------------------------------------------------------------------
def _self_test() -> int:
    ok = True
    expected_order = {"Z97": 97, "D12": 24, "A4": 12, "A5": 60, "S5": 120}
    print("group   |G|  commutator_density  checks")
    for spec in LADDER:
        els, idx, comp = get_group(spec)
        n = len(els)
        # order
        order_ok = (n == expected_order[spec])
        # identity: find e with comp(e,x)=x for all x (sample)
        ident = [e for e in range(n)
                 if all(comp(e, x) == x for x in range(min(n, 12)))]
        ident_ok = len(ident) >= 1
        e = ident[0] if ident else 0
        # inverses: every element has one (sample up to 12)
        inv_ok = all(any(comp(a, b) == e for b in range(n))
                     for a in range(min(n, 12)))
        # associativity (sample 200 triples deterministically)
        assoc_ok = True
        rng = range(n)
        cnt = 0
        for a in rng:
            for b in rng:
                for c in rng:
                    if comp(comp(a, b), c) != comp(a, comp(b, c)):
                        assoc_ok = False
                    cnt += 1
                    if cnt >= 200:
                        break
                if cnt >= 200:
                    break
            if cnt >= 200:
                break
        cd = commutator_density(spec)
        abelian_ok = (cd == 0.0) if spec[0] == "Z" else (cd > 0.0)
        cell_ok = order_ok and ident_ok and inv_ok and assoc_ok and abelian_ok
        ok = ok and cell_ok
        print(f"{spec:6s} {n:4d}  {cd:.4f}             "
              f"order={order_ok} ident={ident_ok} inv={inv_ok} "
              f"assoc={assoc_ok} abelian_axis={abelian_ok}"
              + ("" if cell_ok else "  <-- FAIL"))
    # commutator density should be 0 for Z and strictly positive & non-trivial
    # for the non-abelian rungs (monotone-ish complexity, not required strict)
    cds = {s: commutator_density(s) for s in LADDER}
    if not (cds["Z97"] == 0 and min(cds["D12"], cds["A4"], cds["A5"], cds["S5"]) > 0):
        ok = False
        print("    FAIL: commutator-density abelian/non-abelian split wrong")
    # dataset shape sanity on a small group
    X, Y = make_group_dataset("A4")
    shape_ok = (X.shape == (144, 3) and Y.shape == (144,)
                and int(X[:, 2].min()) == 12 and int(Y.max()) < 12)
    print(f"dataset A4: X{tuple(X.shape)} Y{tuple(Y.shape)} EQ={int(X[0,2])} "
          f"shape_ok={shape_ok}")
    ok = ok and shape_ok
    print("GROUPS SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())

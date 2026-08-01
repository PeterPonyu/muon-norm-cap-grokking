"""A-G3 — update-GEOMETRY control: if not norm growth, then what?

findings-021 (A §3.2) showed that capping Muon's hidden-matrix norm growth
PRESERVES grokking and ACCELERATES it ~10x, i.e. the growth is a dispensable
byproduct. The obvious referee question is then "what IS doing the work?".
This arm attacks that by substituting Muon's orthogonalized update UV^T with
transforms that keep some of its properties and break others, at the SAME hook
point run_s5_normctl uses for the cap (a monkeypatch of train.build_optimizer):

  muon      vanilla Newton-Schulz orthogonalization             (reference)
  specflat  exact SVD, all singular values set equal            (internal
            POSITIVE control: this is what NS approximates, so it must
            reproduce muon; if it does not, the arm is not measuring geometry)
  randorth  a FRESH random semi-orthogonal matrix each step, rescaled to the
            NS update's Frobenius norm. Keeps the "fixed energy per step"
            property, destroys ALL alignment with the gradient. NEGATIVE
            control: isolates energy-injection from direction.
  specinv   the gradient's singular VECTORS with the singular value spectrum
            INVERTED, rescaled to the NS norm. Gradient-aligned but
            anti-spectrally-weighted: it puts the step budget into the
            gradient's WEAKEST directions instead of flattening the spectrum.
  adamw     plain AdamW (no hidden/other split)                 (reference)

Each transform replaces ONLY the zeropower function inside Muon.step; momentum,
Nesterov, the max(1, m/n)**0.5 RMS scale, and decoupled weight decay are Muon's
own code, untouched. So the arms differ in update GEOMETRY and nothing else.

PARAM-SET DISCIPLINE (the subtlety flagged in arch_staircase/analyze_optaxis.py:12
-- "Muon acts on a different param set per arm"): every geometry arm here
(muon/specflat/randorth/specinv) drives EXACTLY the same param set, the
split_params_for_muon hidden 2-D matrices, with AdamW on the complement; the
transform is the only factor. The `adamw` reference is the paper's AdamW arm and
by construction puts ALL params under one AdamW (train.build_optimizer's own
branch) -- it is a between-family reference, not a matched-param-set control.
The matched-param-set comparison is muon-vs-{specflat,randorth,specinv}.

Grid: op in {s5, add(p=97)} x arm in {muon,adamw,randorth,specinv,specflat} x
5 seeds = 50 runs. Native lr regime per the archived config audit
(revision2026/A/bootstrap_cis.json: every tuple is ('adamw', 0.001, 0.02, ...)):
AdamW-side lr 1e-3, Muon hidden lr 2e-2, wd 0.01, init_scale 1.0, steps 20000.

Readout: grok_step per arm. Does any control reproduce Muon's acceleration?
Preregistered reading: specflat ~ muon (positive control passes); randorth
fails to grok or is much slower (energy alone is not the mechanism); specinv
separates "aligned with the gradient" from "flattens the spectrum".

Seed block 60-64 -- disjoint from every prior arm (results/* uses 0-14 and
0-59 in icrl_td_gamma_ladder; revision2026 uses 0-2, 10-17, 20-31, 50-51).

ETA (from measured s/step and stopped_step in the archived arms, not guesses):
expected ~2.0 GPU-h, hard ceiling ~9 GPU-h. The spread is real -- randorth and
specinv are EXPECTED not to grok, so they run the full 20k budget, while
muon/specflat early-stop around 1k steps.

  python run_ag3_update_control.py --smoke     # CPU, exercises every transform
  python run_ag3_update_control.py --dry-run
  python run_ag3_update_control.py [--only s5] [--gpu 0] [--num-shards N --shard-id I]

Output: ../revision2026/gpu2026/ag3/<name>.jsonl (flat) + MANIFEST.json.
Resume-safe via the house `_summary`-tail marker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
_GROKKING = os.path.abspath(os.path.join(_THIS, "..", "grokking"))
if _GROKKING not in sys.path:
    sys.path.insert(0, _GROKKING)
_EXP = os.path.dirname(_THIS)
if _EXP not in sys.path:
    sys.path.append(_EXP)


def _preselect_gpu(argv):
    """Honour --gpu before torch is imported. Absent => inherit the launcher's
    CUDA_VISIBLE_DEVICES untouched (no double-masking)."""
    if "--gpu" in argv:
        i = argv.index("--gpu")
        if i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]


_preselect_gpu(sys.argv)

import torch  # noqa: E402

import train as TRAIN  # noqa: E402  (grokking trainer; we monkeypatch build_optimizer)
from muon import Muon, split_params_for_muon, zeropower_via_newtonschulz5  # noqa: E402
from run_s5_normctl import already_done, STEPS, EVAL_EVERY, WD, INIT_SCALE  # noqa: E402

# Captured BEFORE any monkeypatching so the adamw reference arm can restore the
# real train.build_optimizer (which branches on cfg.optimizer).
_ORIG_BUILD_OPTIMIZER = TRAIN.build_optimizer

OUT = os.path.join(_EXP, "revision2026", "gpu2026", "ag3")
OPS = ["s5", "add"]
ARMS = ["muon", "adamw", "randorth", "specinv", "specflat"]
GEOMETRY_ARMS = ["muon", "randorth", "specinv", "specflat"]
SEEDS = list(range(60, 65))
LR_ADAMW = 1e-3        # archived config audit: AdamW-side lr
LR_MUON = 0.02         # archived config audit: Muon hidden-matrix lr
SPECINV_FLOOR = 1e-6   # relative floor on singular values before inversion


def _svd32(G):
    """SVD in fp32 regardless of param dtype (bf16 SVD is unsupported)."""
    U, S, Vh = torch.linalg.svd(G.float(), full_matrices=False)
    return U, S, Vh


def transform_specflat(G, ref_norm, gen):
    """Singular vectors of G, all singular values equal. The exact version of
    what Newton-Schulz approximates."""
    U, S, Vh = _svd32(G)
    X = U @ Vh
    return X * (ref_norm / (X.norm() + 1e-12))


def transform_randorth(G, ref_norm, gen):
    """A FRESH random semi-orthogonal matrix of G's shape, scaled to ref_norm.
    Same per-step energy and the same flat spectrum as Muon's update, but drawn
    independently of the gradient -- zero alignment in expectation."""
    m, n = G.shape
    A = torch.randn(m, n, generator=gen, device=G.device, dtype=torch.float32)
    if m >= n:
        Q, _ = torch.linalg.qr(A)          # [m, n], orthonormal columns
        X = Q
    else:
        Q, _ = torch.linalg.qr(A.T)        # [n, m] -> transpose to [m, n]
        X = Q.T
    return X * (ref_norm / (X.norm() + 1e-12))


def transform_specinv(G, ref_norm, gen):
    """G's singular vectors with the spectrum INVERTED (s -> 1/s), then rescaled
    to ref_norm. Gradient-aligned but anti-spectrally-weighted: the step budget
    goes into the gradient's weakest directions. The floor keeps a numerically
    zero singular value from dominating the whole update."""
    U, S, Vh = _svd32(G)
    s = S.clamp_min(S.max() * SPECINV_FLOOR + 1e-30)
    inv = 1.0 / s
    X = (U * inv.unsqueeze(0)) @ Vh
    return X * (ref_norm / (X.norm() + 1e-12))


TRANSFORMS = {
    "specflat": transform_specflat,
    "randorth": transform_randorth,
    "specinv": transform_specinv,
}


class UpdateGeometryMuon(Muon):
    """Muon with its orthogonalized update replaced by an alternative transform.

    Everything else -- momentum buffer, Nesterov lookahead, the
    max(1, m/n)**0.5 RMS scale, decoupled weight decay -- is Muon's own step()
    code, reached by temporarily swapping the module-level zeropower function
    that Muon.step calls. The reference Frobenius norm each transform matches is
    the norm of the ACTUAL Newton-Schulz output for that same update tensor, so
    "same energy" is per-step exact rather than a nominal sqrt(min(m,n)).
    """

    def __init__(self, params, transform, seed=0, **kw):
        super().__init__(params, **kw)
        self.transform_name = transform
        self._fn = TRANSFORMS[transform]
        dev = "cpu"
        for g in self.param_groups:
            for p in g["params"]:
                dev = p.device
                break
            break
        self._gen = torch.Generator(device=dev).manual_seed(seed * 104729 + 7)

    def _zeropower(self, G, steps=5, eps=1e-7):
        ns = zeropower_via_newtonschulz5(G, steps=steps, eps=eps)
        ref = ns.float().norm()
        return self._fn(G, ref, self._gen).to(ns.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        import muon as _muon
        saved = _muon.zeropower_via_newtonschulz5
        _muon.zeropower_via_newtonschulz5 = self._zeropower
        try:
            return super().step(closure)
        finally:
            _muon.zeropower_via_newtonschulz5 = saved


def make_builder(arm, seed):
    """build_optimizer(model, cfg) replacement for the geometry arms.

    The param set is FIXED across arms: split_params_for_muon's hidden 2-D
    matrices go to the geometry optimizer, the complement to AdamW.
    """
    def _build(model, cfg):
        hidden_p, other_p = split_params_for_muon(model)
        kw = dict(lr=cfg.muon_lr, momentum=0.95, nesterov=True, ns_steps=5,
                  weight_decay=cfg.weight_decay)
        if arm == "muon":
            opt_h = Muon(hidden_p, **kw)
        else:
            opt_h = UpdateGeometryMuon(hidden_p, transform=arm, seed=seed, **kw)
        opt_a = torch.optim.AdamW(other_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay)
        return [opt_h, opt_a]
    return _build


def build_cells():
    return [(op, arm, s) for op in OPS for arm in ARMS for s in SEEDS]


def cell_name(cell):
    op, arm, seed = cell
    return f"ag3_{op}_{arm}_s{seed}"


def make_cfg(op, arm, seed, **over):
    kw = dict(op=op, optimizer=("adamw" if arm == "adamw" else "muon"),
              lr=LR_ADAMW, muon_lr=LR_MUON, init_scale=INIT_SCALE,
              weight_decay=WD, seed=seed, steps=STEPS, eval_every=EVAL_EVERY,
              mech=True)
    if op == "add":
        kw["p"] = 97
    kw.update(over)
    return TRAIN.Config(**kw)


def install(arm, seed):
    if arm == "adamw":
        TRAIN.build_optimizer = _ORIG_BUILD_OPTIMIZER
    else:
        TRAIN.build_optimizer = make_builder(arm, seed)


def write_manifest(cells, est):
    os.makedirs(OUT, exist_ok=True)
    runs = []
    for c in cells:
        op, arm, seed = c
        name = cell_name(c)
        runs.append({"name": name, "file": name + ".jsonl", "op": op,
                     "arm": arm, "seed": seed, "steps": STEPS,
                     "eval_every": EVAL_EVERY, "lr": LR_ADAMW,
                     "muon_lr": LR_MUON, "weight_decay": WD})
    man = {"arm": "ag3", "runs": runs, "n_runs": len(runs),
           "seed_block": [SEEDS[0], SEEDS[-1]],
           "ops": OPS, "arms": ARMS,
           "param_set": "split_params_for_muon hidden 2-D matrices, FIXED "
                        "across all geometry arms; adamw reference is "
                        "train.build_optimizer's own single-AdamW branch",
           "est_gpu_hours": est,
           "sentinel": '"_summary" in jsonl tail'}
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def estimate_gpu_hours(cells):
    """Measured s/step and stopped_step from the archived arms.

    s5:  results/s5_normctl  kinf 0.036 s/step, stopped_med 4175
                             k1   0.185 s/step, stopped_med 1150
    add: results/add_normctl kinf 0.024 s/step, stopped_med 600
    Expected = grokking arms early-stop; ceiling = every cell runs the full
    20000-step budget at the slower measured rate.
    """
    sps = {"s5": 0.12, "add": 0.024}
    exp_steps = {"s5": {"grok": 2000, "nogrok": STEPS},
                 "add": {"grok": 900, "nogrok": STEPS}}
    exp = ceil = 0.0
    for op, arm, _ in cells:
        grok = arm in ("muon", "adamw", "specflat")
        exp += sps[op] * exp_steps[op]["grok" if grok else "nogrok"]
        ceil += sps[op] * STEPS
    return {"expected_gpu_hours": round(exp / 3600, 2),
            "ceiling_gpu_hours": round(ceil / 3600, 2),
            "basis": "measured s/step in results/s5_normctl + results/add_normctl; "
                     "randorth/specinv assumed not to grok (full budget)"}


def run_smoke():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    # 1. Every transform: shape preserved, Frobenius norm matched to the NS
    #    reference, and the alignment signature that defines each arm.
    for shape in [(16, 16), (32, 8), (8, 32)]:
        G = torch.randn(*shape)
        ns = zeropower_via_newtonschulz5(G, steps=5).float()
        ref = ns.norm()
        for name, fn in TRANSFORMS.items():
            X = fn(G, ref, gen)
            assert X.shape == G.shape, f"{name} {shape}: shape {tuple(X.shape)}"
            assert abs(float(X.norm()) - float(ref)) / float(ref) < 1e-4, \
                f"{name} {shape}: norm {float(X.norm()):.5f} vs ref {float(ref):.5f}"
    # specflat must agree with Newton-Schulz (it is the exact version of it).
    # NS-5 is a 5-step quintic run in bfloat16 that only pushes singular values
    # TOWARD 1, so exact agreement is not available and the gate has to be set
    # from the approximation's real fidelity. Measured cos(NS-5, exact U V^T)
    # over 15 configurations (shapes 64x64 to 512x128, condition numbers 1 to
    # 100): 0.9814 at worst (kappa=100), 0.9999 at best (kappa=1, where NS has
    # nothing to do). The gate is 0.95, below the measured floor with margin and
    # far above the level any non-gradient-aligned transform can reach --- the
    # separation the arm relies on is cos(specflat) ~ 0.98 against
    # cos(randorth) ~ 0.00, three orders of magnitude wider than this threshold.
    G = torch.randn(16, 16)
    ns = zeropower_via_newtonschulz5(G, steps=5).float()
    flat = transform_specflat(G, ns.norm(), gen)
    cos_flat = float((ns * flat).sum() / (ns.norm() * flat.norm()))
    rnd = transform_randorth(G, ns.norm(), gen)
    cos_rnd = float((ns * rnd).sum() / (ns.norm() * rnd.norm()))
    inv = transform_specinv(G, ns.norm(), gen)
    cos_inv = float((ns * inv).sum() / (ns.norm() * inv.norm()))
    assert cos_flat > 0.95, f"specflat vs NS cos={cos_flat:.4f} (positive control broken)"
    assert cos_flat - abs(cos_rnd) > 0.5, \
        f"specflat/randorth separation too small ({cos_flat:.4f} vs {cos_rnd:+.4f})"
    assert abs(cos_rnd) < 0.5, f"randorth vs NS cos={cos_rnd:.4f} (not decorrelated)"
    print(f"SMOKE 1: transforms shape+energy matched to NS reference; "
          f"cos(NS, specflat)={cos_flat:.4f} cos(NS, randorth)={cos_rnd:+.4f} "
          f"cos(NS, specinv)={cos_inv:+.4f} — OK")

    # 2. specinv really inverts the spectrum. svdvals always returns descending
    # order, so comparing argsort orders is vacuous (both are the identity
    # permutation). The content of the claim is that G's SMALLEST singular
    # direction carries X's LARGEST weight, i.e. the sorted spectrum of X is
    # proportional to the reversed reciprocal of G's, up to the floor clamp and
    # the ref_norm rescale.
    G = torch.randn(12, 12)
    s_g = torch.linalg.svdvals(G.float())
    s_i = torch.linalg.svdvals(transform_specinv(G, s_g.norm(), gen))
    s_clamped = s_g.clamp_min(s_g.max() * SPECINV_FLOOR + 1e-30)
    predicted = (1.0 / s_clamped).flip(0)
    predicted = predicted * (s_i.norm() / predicted.norm())
    rel = float((s_i - predicted).norm() / predicted.norm())
    assert rel < 1e-4, f"specinv spectrum is not the reversed reciprocal (rel err {rel:.2e})"
    assert float(s_i.argmax()) == 0.0 and float(s_g.argmin()) == len(s_g) - 1, \
        "spectra are not in the expected descending order"
    print(f"SMOKE 2: specinv flips the spectrum "
          f"(G smax/smin={float(s_g.max()/s_g.min()):.2f} -> "
          f"X smax/smin={float(s_i.max()/s_i.min()):.2f}, "
          f"rel err vs reversed reciprocal {rel:.1e}) — OK")

    # 3. Every arm runs end-to-end through the monkeypatched trainer, CPU, no
    #    writes; the geometry arms must all drive the SAME param set.
    nsets = {}
    for arm in ARMS:
        install(arm, seed=0)
        cfg = make_cfg("add", arm, 0, p=7, d_model=32, steps=20, eval_every=10,
                       device="cpu")
        s, hist = TRAIN.run(cfg, out_path=None)
        assert "grok_step" in s and "final_wn_hidden" in s, f"{arm}: summary schema"
        assert hist[-1]["train_loss"] == hist[-1]["train_loss"], f"{arm}: NaN loss"
        print(f"SMOKE 3 [{arm}]: 20-step add run OK "
              f"train_loss={hist[-1]['train_loss']:.4f} "
              f"test_acc={hist[-1]['test_acc']:.3f} "
              f"wn_hidden={s['final_wn_hidden']:.2f}")
        if arm in GEOMETRY_ARMS:
            torch.manual_seed(0)
            m = TRAIN.GrokTransformer(vocab_size=8, seq_len=3, d_model=32,
                                      n_heads=4, n_layers=2, mlp_ratio=4,
                                      init_scale=INIT_SCALE)
            h, _ = split_params_for_muon(m)
            nsets[arm] = (len(h), sum(p.numel() for p in h))
    assert len(set(nsets.values())) == 1, f"param set differs across arms: {nsets}"
    print(f"SMOKE 4: geometry arms share one param set "
          f"({nsets['muon'][0]} matrices, {nsets['muon'][1]} params) — OK")

    # 5. One s5 cell through the real op, and the jsonl writer + resume marker.
    install("specflat", seed=0)
    cfg = make_cfg("s5", "specflat", 0, d_model=32, steps=20, eval_every=10,
                   device="cpu")
    sdir = os.path.join(OUT, "_smoke")
    os.makedirs(sdir, exist_ok=True)
    path = os.path.join(sdir, "ag3_s5_specflat_s60_smoke.jsonl")
    s, _ = TRAIN.run(cfg, out_path=path)
    assert already_done(path), "smoke jsonl missing the _summary marker"
    with open(path) as f:
        first = json.loads(f.readline())
    assert "_meta" in first and first["_meta"]["muon_lr"] == LR_MUON
    print(f"SMOKE 5: s5 cell + jsonl writer OK (_meta first, _summary last, "
          f"resume marker detected) -> {path}")

    cells = build_cells()
    est = estimate_gpu_hours(cells)
    print(f"AG3 SMOKE PASS: 3 update transforms (randorth/specinv/specflat) + "
          f"muon + adamw all exercised end-to-end; energy-matching, "
          f"spectrum inversion, positive control (specflat~NS cos {cos_flat:.4f}), "
          f"negative control (randorth cos {cos_rnd:+.4f}), fixed param set, "
          f"jsonl writer and resume marker all verified; "
          f"{len(cells)} cells planned, est {est['expected_gpu_hours']}h "
          f"(ceiling {est['ceiling_gpu_hours']}h)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on cell name")
    ap.add_argument("--gpu", default=None,
                    help="CUDA device id; absent => inherit CUDA_VISIBLE_DEVICES")
    try:
        from runner_utils import add_shard_args, shard_cells, validate_shard_args
        add_shard_args(ap)
        _shard = True
    except Exception:
        _shard = False
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return

    cells = build_cells()
    est = estimate_gpu_hours(cells)
    write_manifest(cells, est)
    if args.only:
        cells = [c for c in cells if args.only in cell_name(c)]
    if _shard:
        validate_shard_args(args)
        cells = shard_cells(cells, args.num_shards, args.shard_id)

    if args.dry_run:
        for c in cells:
            print(cell_name(c))
        print(f"\n{len(cells)} cells (of {len(build_cells())} planned) "
              f"| ops={OPS} arms={ARMS} seeds={SEEDS[0]}-{SEEDS[-1]} "
              f"| steps={STEPS} eval_every={EVAL_EVERY}")
        print(f"est {est['expected_gpu_hours']} GPU-h expected, "
              f"{est['ceiling_gpu_hours']} GPU-h ceiling ({est['basis']})")
        print(f"-> {OUT}")
        return

    os.makedirs(OUT, exist_ok=True)
    print(f"[ag3] {len(cells)} cells -> {OUT}", flush=True)
    for i, c in enumerate(cells):
        op, arm, seed = c
        name = cell_name(c)
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        install(arm, seed)
        cfg = make_cfg(op, arm, seed)
        t0 = time.time()
        s, _ = TRAIN.run(cfg, out_path=path)
        print(f"[{i+1}/{len(cells)}] {name}: grok={s['grok_step']} "
              f"mem={s['memorize_step']} test={s['final_test_acc']:.3f} "
              f"wn_hidden={s['final_wn_hidden']:.1f} ({time.time()-t0:.0f}s)",
              flush=True)
    print("[ag3] DONE", flush=True)


if __name__ == "__main__":
    main()

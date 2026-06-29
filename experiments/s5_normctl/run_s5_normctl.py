"""Tier-2 (A-ISSUE-3) — norm-controlled Muon on S5: is the growth route CAUSAL?

The A red-team objected that Muon's 6.5× hidden-norm growth (findings-002) may be a
restatement of the orthogonalized update (fixed spectral norm ⇒ Frobenius norm
inflates every step) rather than a mechanism. This runner is the manipulated-cause
arm: cap the hidden-matrix growth and ask whether grokking survives.

RED-TEAM-CORRECTED design (agent ac44b66b, see tier2-normctl-muon-design.md):
- DOSE-RESPONSE is primary: ceiling k ∈ {∞ (vanilla Muon = growth control), 3, 2,
  1.5, 1}. Only a MONOTONE grok-vs-ceiling curve distinguishes "growth causal" from
  a single confounded flat point.
- RELATIVE GROWTH BUDGET, downward-only: cap each 2-D hidden matrix at k·‖W‖_init
  and ONLY ever scale DOWN (n > cap). Downward-only means the projection NEVER
  cancels Muon's decoupled weight decay (which also shrinks) — fixing the
  wd-cancellation confound the naive "project to init" had. k=∞ ⇒ vanilla Muon.
- CONFIG MATCHES s5_mech (NOT grokking/train.py defaults): op=s5, init_scale=1.0,
  weight_decay=0.01, steps=20000, eval_every=50, mech=True — so the growth arm
  reproduces 002's 6.5× and grok_step / wn_hidden are directly comparable.

Non-invasive: reuses grokking/train.run via a monkeypatch of train.build_optimizer.
train.run already calls opt.step() in its loop, so NormControlledMuon.step()
(= super().step() then project) drops in with no loop change.

KILL CRITERIA (preregistered):
- flat/low-k arms grok at comparable rate/step → growth NOT causal (A §3.2 demoted
  to descriptive; the honest, likely-publishable negative).
- monotone grok-rate collapse as k→1 → growth IS causal (A §3.2 upheld as mechanism).
- sanity: growth arm (k=∞) must reproduce ~6.5× (final_wn_hidden); capped arms must
  show wn_hidden pinned at ≤ k·init (logged) — else the intervention didn't bite.
- confound watch: log wn_hidden (the represented-scale proxy); a future logit-RMS
  add would sharpen "function got too small" (train.run eval doesn't log it yet).

  python run_s5_normctl.py --smoke          # CPU: projection unit-test + short s5 run
  python run_s5_normctl.py --dry-run
  python run_s5_normctl.py [--ks inf,3,2,1.5,1] [--seeds 8] [--num-shards N --shard-id I]

Output: ../../experiments/results/s5_normctl/<name>.jsonl  (k<inf|float>_s<seed>)
Resume-aware. Compare against results/s5_mech/ (the growth-arm reference).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_GROKKING = os.path.abspath(os.path.join(_THIS, "..", "grokking"))
if _GROKKING not in sys.path:
    sys.path.insert(0, _GROKKING)
_EXP = os.path.dirname(_THIS)
if _EXP not in sys.path:
    sys.path.append(_EXP)

import train as TRAIN  # noqa: E402  (grokking trainer; we monkeypatch build_optimizer)
from muon import Muon, split_params_for_muon  # noqa: E402

OUT = os.path.join(_THIS, "..", "..", "experiments", "results", "s5_normctl")
STEPS = 20000
EVAL_EVERY = 50
WD = 0.01            # matches s5_mech (NOT train.py default 1.0)
INIT_SCALE = 1.0


class NormControlledMuon(Muon):
    """Muon with a downward-only per-matrix Frobenius-norm ceiling = k·‖W‖_init.

    Caps GROWTH only (never inflates), so it does not cancel decoupled weight decay.
    k=inf would be vanilla Muon (use plain Muon for that arm instead).
    """

    def __init__(self, params, ceiling_k=1.0, **kw):
        super().__init__(params, **kw)
        self.ceiling_k = float(ceiling_k)
        self._init_norm = {}
        for g in self.param_groups:
            for p in g["params"]:
                self._init_norm[id(p)] = float(p.detach().norm())

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for g in self.param_groups:
            for p in g["params"]:
                cap = self.ceiling_k * self._init_norm[id(p)]
                n = float(p.detach().norm())
                if cap > 0 and n > cap:
                    p.mul_(cap / n)
        return loss


def make_builder(ceiling_k):
    """build_optimizer(model, cfg) replacement. ceiling_k=None ⇒ vanilla Muon."""
    def _build(model, cfg):
        muon_p, adamw_p = split_params_for_muon(model)
        if ceiling_k is None:
            opt_m = Muon(muon_p, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                         ns_steps=5, weight_decay=cfg.weight_decay)
        else:
            opt_m = NormControlledMuon(muon_p, ceiling_k=ceiling_k, lr=cfg.muon_lr,
                                       momentum=0.95, nesterov=True, ns_steps=5,
                                       weight_decay=cfg.weight_decay)
        opt_a = torch.optim.AdamW(adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                  weight_decay=cfg.weight_decay)
        return [opt_m, opt_a]
    return _build


def _ktag(k):
    return "kinf" if k is None else ("k" + (f"{k:g}").replace(".", "p"))


def _parse_ks(s):
    ks = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        ks.append(None if tok.lower() in ("inf", "none", "vanilla") else float(tok))
    return ks


def build_cells(ks, n_seeds):
    return [(k, s) for k in ks for s in range(n_seeds)]


def already_done(path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if size == 0:
            return False
        fh.seek(max(0, size - 4096))
        return '"_summary"' in fh.read().decode("utf-8", errors="replace")


def run_smoke():
    # 1. DIRECT projection unit-test: ceiling caps an inflated matrix to k·init,
    #    downward-only (never inflates a small one).
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 8))
    init_n = float(w.detach().norm())
    opt = NormControlledMuon([w], ceiling_k=1.0, lr=0.02)
    with torch.no_grad():
        w.mul_(5.0)                       # inflate to 5× init
    w.grad = torch.zeros_like(w)          # zero grad ⇒ super().step() ~ no-op
    opt.step()
    capped = float(w.detach().norm())
    assert abs(capped - init_n) / init_n < 1e-3, \
        f"ceiling did not cap: {capped:.4f} vs init {init_n:.4f}"
    # downward-only: a matrix below ceiling is untouched
    torch.manual_seed(1)
    w2 = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
    small_n = float(w2.detach().norm())
    opt2 = NormControlledMuon([w2], ceiling_k=3.0, lr=0.02)
    w2.grad = torch.zeros_like(w2)
    opt2.step()
    assert abs(float(w2.detach().norm()) - small_n) / small_n < 1e-3, \
        "downward-only violated: small matrix was changed"
    print("SMOKE 1: projection caps growth to k·init, downward-only — OK")

    # 2. short s5 run via the monkeypatched trainer (k=1 flat arm), CPU, no files.
    TRAIN.build_optimizer = make_builder(1.0)
    cfg = TRAIN.Config(op="s5", optimizer="muon", init_scale=INIT_SCALE,
                       weight_decay=WD, seed=0, steps=30, eval_every=15,
                       mech=True, device="cpu")
    summary, hist = TRAIN.run(cfg, out_path=None)
    assert "grok_step" in summary and "final_wn_hidden" in summary
    wn0 = hist[0]["wn_hidden"]
    wn_last = hist[-1]["wn_hidden"]
    assert wn_last <= wn0 * 1.05, \
        f"flat arm hidden-norm grew ({wn0:.2f}→{wn_last:.2f}); ceiling not biting"
    print(f"SMOKE 2: k=1 s5 run hidden-norm held {wn0:.2f}→{wn_last:.2f} "
          f"(capped), summary schema OK")
    print("RUN_S5_NORMCTL SMOKE PASS: dose-response normctl ready; reuses "
          "train.run (op=s5/wd=0.01/mech) for s5_mech comparability; zero writes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ks", default="inf,3,2,1.5,1",
                    help="ceiling ladder (inf=vanilla Muon growth control)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--op", default="s5", choices=["s5", "add"],
                   help="s5 (default, s5_mech operating point) | add (mod-add replication of A-ISSUE-3)")
    try:
        from runner_utils import add_shard_args, shard_cells, validate_shard_args
        add_shard_args(ap); _shard = True
    except Exception:
        _shard = False
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return
    ks = _parse_ks(args.ks)
    cells = build_cells(ks, args.seeds)
    if _shard:
        validate_shard_args(args)
        cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        for k, s in cells:
            print(f"{_ktag(k)}_s{s}")
        print(f"{len(cells)} cells; ks={args.ks} seeds={args.seeds}")
        return
    out_dir = OUT if args.op == "s5" else os.path.join(os.path.dirname(OUT), "add_normctl")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[s5_normctl op={args.op}] {len(cells)} cells -> {out_dir}", flush=True)
    for i, (k, seed) in enumerate(cells):
        name = f"{_ktag(k)}_s{seed}"
        path = os.path.join(out_dir, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        TRAIN.build_optimizer = make_builder(k)
        cfg = TRAIN.Config(op=args.op, optimizer="muon", init_scale=INIT_SCALE,
                           weight_decay=WD, seed=seed, steps=STEPS,
                           eval_every=EVAL_EVERY, mech=True)
        t0 = time.time()
        s, _ = TRAIN.run(cfg, out_path=path)
        print(f"[{i+1}/{len(cells)}] {name}: grok={s['grok_step']} "
              f"test={s['final_test_acc']:.3f} wn_hidden={s['final_wn_hidden']:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[s5_normctl] DONE", flush=True)


if __name__ == "__main__":
    main()

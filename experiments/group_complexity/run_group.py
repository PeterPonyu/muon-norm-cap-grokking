"""Direction 019 — group-complexity necessity-threshold runner.

Self-contained training loop (mirrors grokking/train.run) that feeds the
group-ladder Cayley datasets from groups.py through the SHARED GrokTransformer +
build_optimizer (AdamW/Muon/SGDM) — grokking/train.run hardcodes its dataset to
{s5, modular}, so the loop is reproduced here with make_group_dataset swapped in.

STAGE A (--stage A, run FIRST) = P0 CAPACITY CONTROL (mandatory per the
killer-sweep, novelty-chartering-2026-0613.md): does each non-abelian group even
GROK under AdamW at a generous capacity/budget? Establishes a non-trivial
success rate to drive to zero — without it the "necessity threshold" collapses
into a capacity artifact.
  AdamW × {A4, A5, S5} × d_model{128, 256} × 3 seeds.

STAGE B (--stage B) = NECESSITY LADDER:
  {Z97, D12, A4, A5, S5} × {adamw, muon, sgdm} × 5 seeds at fixed width.
  Readout: grok-success (test acc >= grok_thresh within budget) + norm-growth.

Output: ../../experiments/results/group_complexity/<name>.jsonl. Resume-safe.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_THIS)                  # experiments/ — holds runner_utils
_GK = os.path.join(_EXP, "grokking")
for p in (_THIS, _GK, _EXP):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import GrokTransformer          # noqa: E402
from data import train_test_split          # noqa: E402
from train import Config, build_optimizer, evaluate, weight_norms  # noqa: E402
from groups import make_group_dataset, group_order, commutator_density, LADDER  # noqa: E402

OUT = os.path.join(_THIS, "..", "..", "experiments", "results", "group_complexity")
STEPS = 30000
EVAL_EVERY = 100


def run_cell(group: str, optimizer: str, seed: int, d_model: int = 128,
             train_frac: float = 0.5, steps: int = STEPS,
             out_path: str | None = None):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = make_group_dataset(group, device=device)
    vocab_size = group_order(group) + 1
    (Xtr, Ytr), (Xte, Yte) = train_test_split(X, Y, train_frac, seed=seed)

    cfg = Config(optimizer=optimizer, d_model=d_model, weight_decay=1.0,
                 seed=seed, steps=steps, eval_every=EVAL_EVERY)
    model = GrokTransformer(vocab_size=vocab_size, seq_len=3, d_model=d_model,
                            n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                            mlp_ratio=cfg.mlp_ratio,
                            init_scale=cfg.init_scale).to(device)
    optimizers = build_optimizer(model, cfg)

    history = []
    mem_step = grok_step = None
    t0 = time.time()
    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": {"group": group, "optimizer": optimizer,
                                      "seed": seed, "d_model": d_model,
                                      "vocab_size": vocab_size,
                                      "train_frac": train_frac,
                                      "commutator_density": commutator_density(group),
                                      "group_order": group_order(group)}}) + "\n")
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0:
            tr_loss, tr_acc = evaluate(model, Xtr, Ytr)
            te_loss, te_acc = evaluate(model, Xte, Yte)
            wn_total, wn_hidden, wn_embed = weight_norms(model)
            if mem_step is None and tr_acc >= cfg.memorize_thresh:
                mem_step = step
            if grok_step is None and te_acc >= cfg.grok_thresh:
                grok_step = step
            rec = {"step": step, "train_loss": tr_loss, "train_acc": tr_acc,
                   "test_loss": te_loss, "test_acc": te_acc,
                   "wn_total": wn_total, "wn_hidden": wn_hidden}
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n"); f.flush()
            if grok_step is not None and te_acc >= 0.999 and step - grok_step >= 500:
                break
        model.train()
        loss = F.cross_entropy(model(Xtr), Ytr)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    wn0 = history[0]["wn_hidden"]
    summary = {"group": group, "optimizer": optimizer, "seed": seed,
               "d_model": d_model, "train_frac": train_frac,
               "group_order": group_order(group),
               "commutator_density": commutator_density(group),
               "memorize_step": mem_step, "grok_step": grok_step,
               "grokked": bool(grok_step is not None),
               "final_train_acc": history[-1]["train_acc"],
               "final_test_acc": history[-1]["test_acc"],
               "norm_growth_ratio": (history[-1]["wn_hidden"] / wn0) if wn0 else None,
               "n_params": sum(p.numel() for p in model.parameters()),
               "elapsed_sec": time.time() - t0,
               "stopped_step": history[-1]["step"]}
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n"); f.close()
    return summary, history


def already_done(path):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as fh:
        fh.seek(0, 2); sz = fh.tell()
        if sz == 0:
            return False
        fh.seek(max(0, sz - 4096))
        return b'"_summary"' in fh.read()


def build_cells(stage):
    if stage == "A":   # P0 capacity control
        return [(g, "adamw", s, d)
                for g in ["A4", "A5", "S5"] for d in [128, 256]
                for s in range(3)]
    # Stage B (2026-06-13 REDESIGN per findings-019 P0): COMPARABLE-|G|^2 ladders.
    # P0 showed grok-success tracks dataset size |G|^2, not complexity, so the
    # naive Z97/D12/A4/A5/S5 ladder confounds the two. Fix: vary commutator
    # density at FIXED group order (=fixed dataset size). Two rungs:
    #   order-60  : Z60(cd0) -> D30(cd.70) -> A5(cd.92)   (3600 pairs each)
    #   order-120 : Z120(cd0) -> D60(cd.73) -> S5(cd.94)  (14400 pairs each)
    # d256 (P0: A5 needs it); 5 seeds; 3 optimizers. 6 groups x 3 x 5 = 90 cells.
    rung60 = ["Z60", "D30", "A5"]
    rung120 = ["Z120", "D60", "S5"]
    return [(g, opt, s, 256)
            for g in rung60 + rung120 for opt in ["adamw", "muon", "sgdm"]
            for s in range(5)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage", choices=["A", "B"], default="A")
    try:
        from runner_utils import add_shard_args, shard_cells, validate_shard_args
        add_shard_args(ap); _shard = True
    except Exception:
        _shard = False
    args = ap.parse_args()

    if args.smoke:
        s, h = run_cell("A4", "adamw", 0, d_model=64, steps=300)
        print(f"SMOKE A4/adamw 300-step: params={s['n_params']} "
              f"final_train_acc={s['final_train_acc']:.3f} "
              f"commutator_density={s['commutator_density']:.3f} OK")
        print("RUN_GROUP SMOKE PASS")
        sys.exit(0)

    cells = build_cells(args.stage)
    if _shard:
        validate_shard_args(args)
        cells = shard_cells(cells, args.num_shards, args.shard_id)
    if args.dry_run:
        print(f"[group_complexity] stage {args.stage}: {len(cells)} cells")
        for g, opt, s, d in cells[:10]:
            print(f"  {g}_{opt}_d{d}_s{s}")
        sys.exit(0)

    os.makedirs(OUT, exist_ok=True)
    print(f"[group_complexity] stage {args.stage}: {len(cells)} cells -> {OUT}",
          flush=True)
    for i, (g, opt, s, d) in enumerate(cells):
        tag = "capA" if args.stage == "A" else "ladder"
        name = f"{tag}_{g}_{opt}_d{d}_s{s}"
        path = os.path.join(OUT, name + ".jsonl")
        if already_done(path):
            print(f"[{i+1}/{len(cells)}] skip {name}", flush=True)
            continue
        t0 = time.time()
        summ, _ = run_cell(g, opt, s, d_model=d, out_path=path)
        print(f"[{i+1}/{len(cells)}] {name}: grokked={summ['grokked']} "
              f"grok_step={summ['grok_step']} test={summ['final_test_acc']:.3f} "
              f"norm_growth={summ['norm_growth_ratio']:.2f} ({time.time()-t0:.0f}s)",
              flush=True)
    print("[group_complexity] DONE", flush=True)


if __name__ == "__main__":
    main()

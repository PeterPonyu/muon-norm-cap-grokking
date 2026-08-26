"""Full-batch grokking trainer with rich metric logging.

One run = one (optimizer, weight_decay, train_frac, seed) configuration.
Metrics are appended as JSON lines to an output file; a final summary dict is
returned and also written. Designed to be called from run_grid.py or stand-alone.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from data import make_modular_dataset, make_s5_dataset, train_test_split
from model import GrokTransformer
from muon import Muon, split_params_for_muon


@dataclass
class Config:
    p: int = 97
    op: str = "add"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    optimizer: str = "adamw"        # "adamw" | "muon" | "sgdm"
    lr: float = 1e-3                # AdamW lr (and AdamW side of hybrids)
    muon_lr: float = 0.02           # Muon/SGDM lr for hidden matrices
    init_scale: float = 1.0         # Omnigrok-style init multiplier
    weight_decay: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.98
    train_frac: float = 0.4
    steps: int = 30000
    eval_every: int = 100
    seed: int = 0
    device: str = "cuda"
    memorize_thresh: float = 0.99
    grok_thresh: float = 0.95
    mech: bool = False              # log directional/spectral mechanism metrics


@torch.no_grad()
def evaluate(model, X, Y):
    logits = model(X)
    loss = F.cross_entropy(logits, Y).item()
    acc = (logits.argmax(-1) == Y).float().mean().item()
    return loss, acc


@torch.no_grad()
def weight_norms(model):
    """Return (total_l2, hidden_l2, embed_l2)."""
    total_sq = 0.0
    hidden_sq = 0.0
    embed_sq = 0.0
    for name, p in model.named_parameters():
        s = p.detach().float().pow(2).sum().item()
        total_sq += s
        if p.ndim == 2 and "emb" not in name and "unembed" not in name:
            hidden_sq += s
        if "emb" in name or "unembed" in name:
            embed_sq += s
    return total_sq ** 0.5, hidden_sq ** 0.5, embed_sq ** 0.5


class MechProbe:
    """Directional + spectral probes of the hidden (Muon-governed) matrices.

    Distinguishes the two grokking routes directly:
      - contraction route: norm drops, direction barely rotates mem→grok
      - directional route: norm grows, direction rotates substantially
    Logged per eval: cos_init, cos_mem (after T_mem), rot_rate (per-eval rotation),
    eff_rank_mean (entropy effective rank), stable_rank_mean.
    """

    def __init__(self, model):
        from muon import split_params_for_muon
        self.mats, _ = split_params_for_muon(model)
        self.v_init = self._vec()
        self.v_prev = self.v_init.clone()
        self.v_mem = None

    @torch.no_grad()
    def _vec(self):
        return torch.cat([p.detach().reshape(-1) for p in self.mats]).float()

    @staticmethod
    def _cos(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

    @torch.no_grad()
    def mark_memorize(self):
        self.v_mem = self._vec()

    @torch.no_grad()
    def metrics(self):
        v = self._vec()
        out = {
            "cos_init": self._cos(v, self.v_init),
            "cos_mem": self._cos(v, self.v_mem) if self.v_mem is not None else None,
            "rot_rate": 1.0 - self._cos(v, self.v_prev),
        }
        self.v_prev = v
        eff_ranks, stable_ranks = [], []
        for p in self.mats:
            s = torch.linalg.svdvals(p.detach().float())
            ps = s / (s.sum() + 1e-12)
            eff_ranks.append(float(torch.exp(-(ps * (ps + 1e-12).log()).sum())))
            stable_ranks.append(float((s ** 2).sum() / (s[0] ** 2 + 1e-12)))
        out["eff_rank_mean"] = sum(eff_ranks) / len(eff_ranks)
        out["stable_rank_mean"] = sum(stable_ranks) / len(stable_ranks)
        return out


def build_optimizer(model, cfg: Config):
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(), lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)]
    elif cfg.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model)
        opt_muon = Muon(muon_p, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                        ns_steps=5, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_muon, opt_adamw]
    elif cfg.optimizer == "sgdm":
        # Mechanistic control: same hybrid split & momentum as Muon, but plain
        # SGD-momentum on the hidden matrices (NO Newton-Schulz orthogonalization).
        # Isolates the effect of orthogonalization vs the optimizer family/lr.
        sgd_p, adamw_p = split_params_for_muon(model)
        opt_sgd = torch.optim.SGD(sgd_p, lr=cfg.muon_lr, momentum=0.95,
                                  nesterov=True, weight_decay=cfg.weight_decay)
        opt_adamw = torch.optim.AdamW(
            adamw_p, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay)
        return [opt_sgd, opt_adamw]
    else:
        raise ValueError(cfg.optimizer)


def run(cfg: Config, out_path: str | None = None):
    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    if cfg.op == "s5":
        X, Y = make_s5_dataset(device=device)
        vocab_size = 121  # 120 perms + EQ
    else:
        X, Y = make_modular_dataset(cfg.p, cfg.op, device=device)
        vocab_size = cfg.p + 1
    (Xtr, Ytr), (Xte, Yte) = train_test_split(X, Y, cfg.train_frac, seed=cfg.seed)

    model = GrokTransformer(
        vocab_size=vocab_size, seq_len=3, d_model=cfg.d_model,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, mlp_ratio=cfg.mlp_ratio,
        init_scale=cfg.init_scale,
    ).to(device)

    optimizers = build_optimizer(model, cfg)
    probe = MechProbe(model) if cfg.mech else None

    history = []
    memorize_step = None
    grok_step = None
    t0 = time.time()

    f = open(out_path, "w") if out_path else None
    if f:
        f.write(json.dumps({"_meta": asdict(cfg)}) + "\n")

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            tr_loss, tr_acc = evaluate(model, Xtr, Ytr)
            te_loss, te_acc = evaluate(model, Xte, Yte)
            wn_total, wn_hidden, wn_embed = weight_norms(model)
            if memorize_step is None and tr_acc >= cfg.memorize_thresh:
                memorize_step = step
                if probe is not None:
                    probe.mark_memorize()
            if grok_step is None and te_acc >= cfg.grok_thresh:
                grok_step = step
            rec = {
                "step": step,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "test_loss": te_loss, "test_acc": te_acc,
                "wn_total": wn_total, "wn_hidden": wn_hidden, "wn_embed": wn_embed,
            }
            if probe is not None:
                rec.update(probe.metrics())
            history.append(rec)
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            # early stop once grokked and stable
            if grok_step is not None and te_acc >= 0.999 and step - grok_step >= 500:
                break

        # full-batch gradient step
        model.train()
        logits = model(Xtr)
        loss = F.cross_entropy(logits, Ytr)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()

    elapsed = time.time() - t0
    delay_ratio = (grok_step / memorize_step) if (memorize_step and grok_step) else None
    summary = {
        **asdict(cfg),
        "memorize_step": memorize_step,
        "grok_step": grok_step,
        "delay_ratio": delay_ratio,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "final_wn_total": history[-1]["wn_total"],
        "final_wn_hidden": history[-1]["wn_hidden"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_sec": elapsed,
        "stopped_step": history[-1]["step"],
    }
    if f:
        f.write(json.dumps({"_summary": summary}) + "\n")
        f.close()
    return summary, history


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    for k, v in asdict(Config()).items():
        ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    a = ap.parse_args()
    return Config(**vars(a))


if __name__ == "__main__":
    cfg = parse_args()
    out = None
    summary, _ = run(cfg, out)
    print(json.dumps(summary, indent=2))

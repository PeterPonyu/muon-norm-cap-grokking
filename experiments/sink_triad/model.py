"""Direction 008 — full-sequence transformer + norm-position variant + triad
instrumentation.

THE NORM-POSITION QUESTION
--------------------------
2603.05498 argues the extreme-token triad (attention sink + massive activation +
value drain) is partly a PRE-NORM architecture artifact: because pre-LN reads the
RAW (un-normalized) residual stream into the residual add, a token can park a
huge, near-constant activation in the residual stream (the "sink token") that the
LayerNorm downstream rescales away — making the peak free to grow. Their decoupling
intervention is SANDWICH norm: LayerNorm BEFORE the attention/MLP sublayer AND a
second LayerNorm AFTER it, before the residual add, so the quantity actually added
to the residual stream is normalized and the peak cannot run away.

We therefore need a model with BOTH norm positions sharing every other detail so
the optimizer × norm-position factorial is clean. The grokking `Block` is pre-LN
only and we may not modify it, so we write a LOCAL `SinkBlock` parameterized by
`norm_position ∈ {pre, sandwich}`:

    pre       (grokking-identical):
        x = x + proj(attn(ln1(x)))
        x = x + fc2(gelu(fc1(ln2(x))))
    sandwich  (2603.05498 decoupling):
        x = x + ln1b( proj(attn(ln1(x))) )
        x = x + ln2b( fc2(gelu(fc1(ln2(x)))) )

`SinkBlock(pre, ...)` is bit-for-bit the grokking Block (same submodule names
`ln1/ln2/qkv/proj/fc1/fc2`, same init), so the pre arm reproduces 007's pre-LN
regime exactly; the two extra LayerNorms (`ln1b/ln2b`) exist only in the sandwich
arm. We still REUSE the grokking GrokTransformer by import (verifying the
induction_emergence import chain resolves) to source the canonical init and to keep
the param-count baseline identical on the pre arm; the local model only swaps the
block class.

MODEL CHOICE: SELF-CONTAINED local `SinkTransformer` (NOT a subclass).
Rationale: the sandwich variant changes the residual-add wiring INSIDE the block,
which subclassing GrokTransformer cannot express (its blocks are fixed pre-LN
modules). A self-contained model that mirrors the grokking architecture
(2 layers, 4 heads, d=128, learned PE, causal mask, full-sequence logits) is the
minimal way to support both norm positions. We import GrokTransformer to ASSERT
the chain resolves and to share the exact init recipe, but instantiate our own.

Read-only instrumentation (no architecture/weight change):
- `forward(idx)`                  -> logits [B, T, vocab] (all positions).
- `forward_with_attn(idx)`        -> (logits, attn_list) with attn_list[l] =
                                     [B, H, T, T] post-softmax weights (the probe
                                     contract from induction_emergence).
- `forward_with_triad(idx)`       -> (logits, attn_list, vnorm_list, resid_list,
                                     hidden_list) adding per-head value norms
                                     [B,H,T], post-block residual [B,T,C], and the
                                     pre-final-norm hidden states [B,T,C] used by
                                     the spike / residual-peak probes.

Muon hybrid split: `split_params_for_muon` from grokking/muon.py is name-based
(2-D params not named "emb"/"unembed" -> Muon). Our module names match the
grokking convention (`tok_emb`, `pos_emb`, `blocks.*.{qkv,proj,fc1,fc2}`,
`unembed`) and the sandwich LayerNorms are 1-D, so the grokking splitter applies
unchanged (LN weights/biases -> AdamW). Re-exported locally.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import grokking infra without modifying it: LOCAL dir first (our data/model/
# probes win name collisions), grokking dir APPENDED last (we only pull
# model/muon, which do not collide).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
_GROKKING_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "grokking"))
if _GROKKING_DIR not in sys.path:
    sys.path.append(_GROKKING_DIR)

# grokking/model.py -> GrokTransformer; load the module object explicitly to
# dodge the name collision with THIS file (also model.py) on the local path.
# We do not subclass it (the sandwich block rewires the residual add), but we
# (a) prove the induction_emergence import chain resolves here too and
# (b) reuse its exact init recipe so the pre arm matches the 007 baseline.
import importlib.util as _ilu

_grok_model_path = os.path.join(_GROKKING_DIR, "model.py")
_spec = _ilu.spec_from_file_location("grokking_model", _grok_model_path)
_grok_model = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_grok_model)  # type: ignore[union-attr]
GrokTransformer = _grok_model.GrokTransformer  # imported -> chain verified

from muon import split_params_for_muon as _grok_split  # noqa: E402 (grokking infra)

NORM_POSITIONS = ("pre", "sandwich")


class SinkBlock(nn.Module):
    """Transformer block with selectable norm position.

    pre       : grokking-identical pre-LN (sublayer output added raw).
    sandwich  : extra LayerNorm on each sublayer OUTPUT before the residual add
                (2603.05498's decoupling — normalizes what enters the residual
                stream so a massive activation cannot accumulate there).

    Submodule names `ln1/ln2/qkv/proj/fc1/fc2` match grokking's Block so the pre
    arm is bit-for-bit identical; `ln1b/ln2b` exist only in the sandwich arm.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                 norm_position: str = "pre"):
        super().__init__()
        assert norm_position in NORM_POSITIONS, norm_position
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm_position = norm_position
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, mlp_ratio * d_model, bias=False)
        self.fc2 = nn.Linear(mlp_ratio * d_model, d_model, bias=False)
        if norm_position == "sandwich":
            self.ln1b = nn.LayerNorm(d_model)   # post-attention, pre-add
            self.ln2b = nn.LayerNorm(d_model)   # post-MLP, pre-add

    def _attn(self, x):
        B, T, C = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)

    def _mlp(self, x):
        return self.fc2(F.gelu(self.fc1(self.ln2(x))))

    def forward(self, x):
        if self.norm_position == "sandwich":
            x = x + self.ln1b(self._attn(x))
            x = x + self.ln2b(self._mlp(x))
        else:  # pre
            x = x + self._attn(x)
            x = x + self._mlp(x)
        return x


class SinkTransformer(nn.Module):
    """Self-contained tiny causal LM (grokking architecture) with selectable
    norm position and full-sequence logits + triad instrumentation.

    Defaults mirror the grokking spec (2 layers, 4 heads, d=128, learned PE).
    Uses the grokking init recipe so the pre arm reproduces the 007 baseline.
    """

    def __init__(self, vocab_size: int, seq_len: int = 128, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, mlp_ratio: int = 4,
                 init_scale: float = 1.0, norm_position: str = "pre"):
        super().__init__()
        assert norm_position in NORM_POSITIONS, norm_position
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [SinkBlock(d_model, n_heads, mlp_ratio, norm_position)
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len
        self.init_scale = init_scale
        self.norm_position = norm_position
        self.apply(self._init)

    def _init(self, m):
        # Grokking init recipe (single multiplier inflates init scale).
        s = self.init_scale
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 * s)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02 * s)

    def _embed(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        return self.tok_emb(idx) + self.pos_emb(pos)[None]

    def forward(self, idx):
        x = self._embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.unembed(x)                       # [B, T, vocab]

    @torch.no_grad()
    def forward_with_attn(self, idx):
        """Return (logits [B,T,vocab], attn_list) where attn_list[l] is the
        per-head causal attention weight tensor [B, n_heads, T, T] for layer l.

        Matches the induction_emergence probe contract. Re-runs the block math
        read-only (output identical to `forward`).
        """
        logits, attn_list, _, _, _ = self.forward_with_triad(idx)
        return logits, attn_list

    @torch.no_grad()
    def forward_with_triad(self, idx):
        """Read-only instrumented forward.

        Returns (logits, attn_list, vnorm_list, resid_list, hidden_list):
            attn_list[l]   [B, H, T, T]  post-softmax attention weights
            vnorm_list[l]  [B, H, T]     L2 norm of each token's VALUE vector
            resid_list[l]  [B, T, C]     residual stream AFTER block l (post-adds)
            hidden_list[l] [B, T, C]     same residual stream (spike/peak source)
        Recomputed from each block's own q/k/v with the SAME wiring as
        `forward` (pre vs sandwich), so logits are identical to `forward`.
        """
        T = idx.shape[1]
        x = self._embed(idx)
        attn_list, vnorm_list, resid_list = [], [], []
        causal = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        for blk in self.blocks:
            B, _, C = x.shape
            h = blk.ln1(x)
            qkv = blk.qkv(h).view(B, T, 3, blk.n_heads, blk.d_head)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)                    # [B, H, T, d_head]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (blk.d_head ** 0.5)
            scores = scores.masked_fill(~causal[None, None], float("-inf"))
            attn = scores.softmax(dim=-1)            # [B, H, T, T]
            attn_list.append(attn)
            vnorm_list.append(v.norm(dim=-1))        # [B, H, T]
            ctx = (attn @ v).transpose(1, 2).reshape(B, T, C)
            attn_out = blk.proj(ctx)
            if blk.norm_position == "sandwich":
                x = x + blk.ln1b(attn_out)
                x = x + blk.ln2b(blk._mlp(x))
            else:
                x = x + attn_out
                x = x + blk._mlp(x)
            resid_list.append(x)                     # [B, T, C] post-block
        # hidden states feeding ln_f (same as last residual) — spike source.
        hidden_list = list(resid_list)
        logits = self.unembed(self.ln_f(x))
        return logits, attn_list, vnorm_list, resid_list, hidden_list


def split_params_for_muon(model):
    """Partition params into (muon_2d, adamw_other); grokking splitter re-export.

    Name-based: 2-D block matrices -> Muon; embeddings, unembed, and ALL
    LayerNorm params (incl. sandwich ln1b/ln2b, which are 1-D) -> AdamW.
    Our module names match the grokking convention so the splitter applies
    unchanged across both norm positions.
    """
    return _grok_split(model)

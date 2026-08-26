"""Tiny decoder-only transformer for grokking experiments.

Deliberately minimal and standard (close to Nanda's grokking setup):
learned token + positional embeddings, pre-norm blocks, GELU MLP,
weight-tied or separate unembed (separate here for clean param grouping).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.fc1 = nn.Linear(d_model, mlp_ratio * d_model, bias=False)
        self.fc2 = nn.Linear(mlp_ratio * d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, n_heads, d_head]
        q = q.transpose(1, 2)  # [B, n_heads, T, d_head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(attn)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class GrokTransformer(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int = 3, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, mlp_ratio: int = 4,
                 init_scale: float = 1.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len
        self.init_scale = init_scale
        self.apply(self._init)

    def _init(self, m):
        # Omnigrok-style: a single multiplier inflates the initialization,
        # pushing the model into the strong-grokking (large-weight-norm) regime.
        s = self.init_scale
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02 * s)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02 * s)

    def forward(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.unembed(x)  # [B, T, vocab]
        return logits[:, -1, :]   # predict at final position only

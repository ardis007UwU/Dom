#!/usr/bin/env python3
"""Dom SLM v2.0 model: ~4.66M parameters, GQA + FlashAttention + tied embeddings.

Architecture (defaults match the Dom v2.0 spec):
  vocab 2048 (uint16 ids) | d_model 256 | 6 layers | 8 query heads |
  2 key/value heads (GQA, 4 query groups) | head_dim 32 | d_ffn 1024 |
  GELU | RoPE | RMSNorm | causal FlashAttention (SDPA SRAM tiling).

Memory story:
  * Input/output embeddings tied (tok_emb.weight is lm_head.weight): with
    V=2048, d=256 the single 0.52M table keeps 90%+ of capacity in blocks.
  * GQA shrinks the autoregressive KV cache 4x vs MHA (2 KV heads instead
    of 8): per-token cache = 2 kv * 6 layers * 2 heads * 32 dim.
  * No biases, no learned position tables (RoPE), persistent=False caches.

Exact parameter count (defaults): 4,656,384. Requires: torch>=2.0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return ((xf * torch.rsqrt(var + self.eps)) * self.weight).to(x.dtype)


def build_rope_cache(max_seq: int, head_dim: int, base: float = 10000.0,
                     device: torch.device | str = "cpu"):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                           device=device) / head_dim))
    t = torch.arange(max_seq, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]  # (1,1,T,D)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0) -> torch.Tensor:
    """x: (B,H,T,D). offset shifts positions for cached generation."""
    t = x.shape[-2]
    cos = cos[:, :, offset:offset + t, :].to(x.dtype)
    sin = sin[:, :, offset:offset + t, :].to(x.dtype)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


# torch>=2.5 supports enable_gqa (KV broadcast inside SRAM tiles); older
# 2.x raises TypeError, where we fall back to an explicit exact broadcast.
_SDPA_GQA_OK: bool | None = None


def _sdpa_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool) -> torch.Tensor:
    global _SDPA_GQA_OK
    if _SDPA_GQA_OK is False:
        return _sdpa_gqa_fallback(q, k, v, causal)
    try:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                             enable_gqa=True)
    except TypeError:
        _SDPA_GQA_OK = False
        return _sdpa_gqa_fallback(q, k, v, causal)
    _SDPA_GQA_OK = True
    return out


def _sdpa_gqa_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       causal: bool) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


class GQABlock(nn.Module):
    """8 query heads sharing 2 KV heads (4 groups); SDPA FlashAttention."""

    def __init__(self, d_model: int, n_qhead: int, n_kvhead: int, d_ffn: int) -> None:
        super().__init__()
        assert d_model % n_qhead == 0, "d_model must split over query heads"
        assert n_qhead % n_kvhead == 0, "query heads must group evenly over KV"
        self.n_qhead = n_qhead
        self.n_kvhead = n_kvhead
        self.head_dim = d_model // n_qhead
        self.n_group = n_qhead // n_kvhead
        self.n1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_qhead * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kvhead * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kvhead * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.n2 = RMSNorm(d_model)
        self.f1 = nn.Linear(d_model, d_ffn, bias=False)
        self.f2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
                offset: int = 0):
        b, t, _ = x.shape
        q = self.q_proj(self.n1(x)).view(b, t, self.n_qhead, self.head_dim)
        k = self.k_proj(self.n1(x)).view(b, t, self.n_kvhead, self.head_dim)
        v = self.v_proj(self.n1(x)).view(b, t, self.n_kvhead, self.head_dim)
        q = apply_rope(q.permute(0, 2, 1, 3), cos, sin, offset)
        k = apply_rope(k.permute(0, 2, 1, 3), cos, sin, offset)
        v = v.permute(0, 2, 1, 3)
        if kv_cache is not None:  # paged append during generation
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        # enable_gqa broadcasts 2 KV heads over 8 query heads inside SRAM
        # tiles without materializing repeated K/V in HBM (torch>=2.5;
        # exact repeat fallback on older 2.x, see _sdpa_gqa).
        a = _sdpa_gqa(q, k, v, kv_cache is None)
        a = a.permute(0, 2, 1, 3).reshape(b, t, -1)
        x = x + self.o_proj(a)
        h = self.n2(x)
        x = x + self.f2(F.gelu(self.f1(h), approximate="tanh"))
        return x, new_cache


class DomSLMv2(nn.Module):
    """Tied-embedding GQA decoder. lm_head shares tok_emb storage."""

    def __init__(self, vocab_size: int = 2048, d_model: int = 256,
                 n_layer: int = 6, n_qhead: int = 8, n_kvhead: int = 2,
                 d_ffn: int = 1024, max_seq: int = 512) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [GQABlock(d_model, n_qhead, n_kvhead, d_ffn) for _ in range(n_layer)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # one 0.52M table
        cos, sin = build_rope_cache(max_seq, d_model // n_qhead)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            block.o_proj.weight.data.normal_(0.0, 0.02 / math.sqrt(2 * n_layer))
            block.f2.weight.data.normal_(0.0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(0.0, 0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied storage counted once

    def kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        n_kv = self.blocks[0].n_kvhead if len(self.blocks) else 0
        hd = self.d_model // self.blocks[0].n_qhead if len(self.blocks) else 0
        return 2 * len(self.blocks) * n_kv * hd * dtype_bytes  # K + V

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x, _ = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int,
                 temperature: float = 1.0) -> torch.Tensor:
        """Greedy/temperature autoregressive generation with paged KV cache."""
        self.eval()
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.blocks)
        generated = idx
        offset = 0
        for _ in range(max_new):
            x = self.tok_emb(generated[:, -1:])
            for bi, block in enumerate(self.blocks):
                x, caches[bi] = block(x, self.rope_cos, self.rope_sin,
                                      caches[bi], offset)
            offset += 1
            logits = self.lm_head(self.final_norm(x))[:, -1, :]
            if temperature <= 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.softmax(logits / temperature, dim=-1).multinomial(1)
            generated = torch.cat([generated, nxt], dim=1)
            if generated.shape[1] >= self.max_seq:
                break
        self.train()
        return generated

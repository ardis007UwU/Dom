#!/usr/bin/env python3
"""DomLM pre-training engine: high-throughput PyTorch pipeline (5M SLM).

Model (DomSLM): vocab 2048 (tied in/out embeddings), d_model 256, 6 layers,
8 heads (head_dim 32), d_ffn 1024 with GELU, RoPE positions, RMSNorm,
causal FlashAttention via ``F.scaled_dot_product_attention``.
Exact parameter count: 5,246,208 (~5.25M; ~90% in transformer blocks).

Data: zero-copy streaming from uint16 ``.bin`` streams (see
``scripts/prepare_dataset.py``) with ``numpy.memmap`` -- the file is never
fully loaded into RAM.

Precision auto-switch: bfloat16 on Ampere/Hopper (sm>=8.0, e.g. A100),
float16 with GradScaler on older GPUs (T4/V100), float32 on CPU.

Optimizer: AdamW (fused when on CUDA) + linear warmup + cosine decay
(max_lr=3e-3 -> min_lr=3e-4), gradient accumulation, grad-clip 1.0.

Logging: loss, lr, tokens/sec and estimated TFLOPS (6*N*D convention)
every ``--log-every`` steps; validation loss every ``--eval-every`` steps;
checkpoints to ``dom_5m_checkpoint.pt`` (+ best to ``dom_5m_best.pt``).

Requires: torch>=2.0, numpy.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def domlm_version() -> str:
    """Read the single-source-of-truth <repo-root>/VERSION (locked in sync
    with the C++ core's domlm/Version.hpp)."""
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "VERSION"), "r", encoding="ascii") as f:
        ver = f.read().strip()
    if ver.count(".") != 2 or not all(p.isdigit() for p in ver.split(".")):
        raise ValueError(f"malformed VERSION file: {ver!r}")
    return ver


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        normed = xf * torch.rsqrt(var + self.eps)
        return (normed * self.weight).to(x.dtype)


def build_rope_cache(max_seq: int, head_dim: int, base: float = 10000.0,
                     device: torch.device | str = "cpu"):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                           device=device) / head_dim))
    t = torch.arange(max_seq, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None], emb.sin()[None, None]  # (1,1,T,D)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B,H,T,D). cos/sin: (1,1,Tmax,D); sliced to T inside."""
    t = x.shape[-2]
    cos = cos[:, :, :t, :].to(x.dtype)
    sin = sin[:, :, :t, :].to(x.dtype)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ffn: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model must split evenly over heads"
        self.n1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.n2 = RMSNorm(d_model)
        self.f1 = nn.Linear(d_model, d_ffn, bias=False)
        self.f2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(self.n1(x)).view(b, t, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,H,T,Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # FlashAttention backend; exact softmax math, causal mask.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.permute(0, 2, 1, 3).reshape(b, t, -1)
        x = x + self.proj(a)
        h = self.n2(x)
        x = x + self.f2(F.gelu(self.f1(h), approximate="tanh"))
        return x


class DomSLM(nn.Module):
    """5M decoder-only SLM. Tied embeddings: lm_head shares tok_emb storage."""

    def __init__(self, vocab_size: int = 2048, d_model: int = 256,
                 n_layer: int = 6, n_head: int = 8, d_ffn: int = 1024,
                 max_seq: int = 512) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq = max_seq
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, d_ffn) for _ in range(n_layer)])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied storage, one copy
        cos, sin = build_rope_cache(max_seq, d_model // n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # Scale residual-output projections (GPT-2 style) for stable depth.
        for block in self.blocks:
            block.proj.weight.data.normal_(0.0, 0.02 / math.sqrt(2 * n_layer))
            block.f2.weight.data.normal_(0.0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(0.0, 0.02)

    def num_params(self) -> int:
        # parameters() dedups the tied embedding/head storage via memo.
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size),
                                   targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Data: zero-copy memmap streaming
# ---------------------------------------------------------------------------


def load_memmap(path: str, seq_len: int) -> np.memmap:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) < seq_len + 2:
        raise ValueError(f"{path}: only {len(data)} tokens, need > {seq_len + 1}")
    return data


def sample_batch(mem: np.memmap, batch_size: int, seq_len: int,
                 device: torch.device):
    """Random (x, y) views; disk bytes stay on disk until indexed (memmap)."""
    hi = len(mem) - seq_len - 1
    starts = torch.randint(0, hi, (batch_size,)).tolist()
    xs = [torch.from_numpy(np.array(mem[s:s + seq_len], dtype=np.int64))
          for s in starts]
    ys = [torch.from_numpy(np.array(mem[s + 1:s + seq_len + 1], dtype=np.int64))
          for s in starts]
    x = torch.stack(xs).to(device, non_blocking=True)
    y = torch.stack(ys).to(device, non_blocking=True)
    return x, y


# ---------------------------------------------------------------------------
# Training setup helpers
# ---------------------------------------------------------------------------


def pick_precision():
    """bf16 on Ampere/Hopper+, fp16 on older GPUs, fp32 on CPU."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16, False
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.bfloat16, False
    if torch.cuda.is_available():
        return torch.float16, True
    return torch.float32, False


def make_scaler(use_scaler: bool):
    if not use_scaler:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except AttributeError:  # torch < 2.0 fallback
        return torch.cuda.amp.GradScaler()


def lr_at(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(1, max_steps - warmup)
    t = min(t, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def estimate_loss(model: nn.Module, mem: np.memmap, batches: int,
                  batch_size: int, seq_len: int, device: torch.device,
                  amp_dtype, use_amp: bool) -> float:
    model.eval()
    total = 0.0
    dev = "cuda" if device.type == "cuda" else "cpu"
    for _ in range(batches):
        x, y = sample_batch(mem, batch_size, seq_len, device)
        with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / max(batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=f"DomLM v{domlm_version()} 5M SLM pre-training")
    ap.add_argument("--version", action="version",
                    version=f"DomLM {domlm_version()}")
    ap.add_argument("--data-dir", default="data",
                    help="dir with train.bin / valid.bin (uint16 LE)")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--d-ffn", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-lr", type=float, default=3e-3)
    ap.add_argument("--min-lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--ckpt-name", default="dom_5m_checkpoint.pt")
    ap.add_argument("--resume", default="",
                    help="checkpoint path to resume from")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (extra startup cost)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    amp_dtype, use_scaler = pick_precision()
    use_amp = device.type == "cuda"
    scaler = make_scaler(use_scaler and device.type == "cuda")
    print(f"DomLM v{domlm_version()}", flush=True)
    print(f"device={device} amp={'off' if not use_amp else amp_dtype} "
          f"scaler={'on' if scaler is not None else 'off'}", flush=True)

    train_mem = load_memmap(os.path.join(args.data_dir, "train.bin"), args.seq_len)
    valid_mem = load_memmap(os.path.join(args.data_dir, "valid.bin"), args.seq_len)
    print(f"train tokens={len(train_mem)} valid tokens={len(valid_mem)}", flush=True)

    model = DomSLM(args.vocab_size, args.d_model, args.n_layer,
                   args.n_head, args.d_ffn, args.seq_len).to(device)
    n_params = model.num_params()
    print(f"DomLM DomSLM params={n_params} (~{n_params / 1e6:.2f}M)", flush=True)
    if args.compile:
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.max_lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay,
                            fused=(device.type == "cuda"))
    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("val_loss", best_val)
        print(f"resumed {args.resume} at step {start_step}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if device.type == "cuda" else "cpu"
    toks_per_step = args.batch_size * args.seq_len * args.grad_accum
    interval_toks = 0
    t_last = time.perf_counter()

    model.train()
    for step in range(start_step, args.max_steps):
        lr = lr_at(step, args.warmup_steps, args.max_steps,
                   args.max_lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = sample_batch(train_mem, args.batch_size, args.seq_len, device)
            with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        interval_toks += toks_per_step
        if (step + 1) % args.log_every == 0:
            now = time.perf_counter()
            dt = max(now - t_last, 1e-9)
            tps = interval_toks / dt
            tflops = 6 * n_params * interval_toks / dt / 1e12
            with torch.no_grad():
                rep_loss = loss.item() * args.grad_accum
            print(f"step {step + 1}/{args.max_steps} loss {rep_loss:.4f} "
                  f"lr {lr:.2e} tok/s {tps:.0f} TFLOPS {tflops:.3f}", flush=True)
            interval_toks = 0
            t_last = now

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.max_steps:
            val = estimate_loss(model, valid_mem, args.eval_batches,
                                args.batch_size, args.seq_len, device,
                                amp_dtype, use_amp)
            print(f"  [eval] step {step + 1} val_loss {val:.4f}", flush=True)
            if val < best_val:
                best_val = val
                torch.save({"step": step, "config": vars(args),
                            "model": model.state_dict(),
                            "optim": opt.state_dict(),
                            "val_loss": val},
                           os.path.join(args.out_dir, "dom_5m_best.pt"))
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            torch.save({"step": step, "config": vars(args),
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "val_loss": best_val},
                       os.path.join(args.out_dir, args.ckpt_name))
            print(f"  [ckpt] saved {args.ckpt_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

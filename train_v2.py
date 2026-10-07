#!/usr/bin/env python3
"""Dom SLM v2.0 training engine: GQA model + paged streaming + mixed precision.

Pipeline: uint16 ``train.bin``/``valid.bin`` (see scripts/prepare_dataset.py)
-> PagedStreamer (bounded-RAM random pages, bulk disk reads amortized over
dozens of batches) -> GQA DomSLMv2 -> AdamW + warmup/cosine, grad accumulation,
tok/s + TFLOPS logging, ``dom_v3_checkpoint.pt`` saver (+ ``dom_v3_best.pt``).

Requires: torch>=2.0, numpy. Optional ``--gen-prompt`` demo decodes with the
v2 byte vocabulary (no extra dependencies).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from dom_model_v2 import DomSLMv2

# ---------------------------------------------------------------------------
# Paged streaming dataset: bounded RAM, amortized bulk reads
# ---------------------------------------------------------------------------


class PagedStreamer:
    """Random 1 MiB pages copied from the memmap into RAM; batches sample
    uniform random windows *within* the resident page, so disk traffic is
    one sequential bulk read per dozens of batches while RAM stays flat
    (~2 MB + one batch). Near-uniform globally once pages turn over."""

    def __init__(self, path: str, page_tokens: int = 1 << 20,
                 batches_per_page: int = 64) -> None:
        self.mem = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.mem) == 0:
            raise ValueError(f"{path}: empty token stream")
        if page_tokens < 1:
            raise ValueError("--page-tokens must be >= 1")
        self.page_tokens = min(page_tokens, len(self.mem))
        self.batches_per_page = max(1, batches_per_page)
        self.page = np.empty(self.page_tokens, dtype=np.uint16)
        self.left = 0
        self.refill()

    def refill(self) -> None:
        hi = len(self.mem) - self.page_tokens
        base = random.randrange(hi + 1) if hi > 0 else 0
        # Single bulk copy (C-speed, GIL released); the only disk traffic.
        self.page[:] = self.mem[base:base + self.page_tokens]
        self.left = self.batches_per_page

    def batch(self, batch_size: int, seq_len: int, device: torch.device):
        if len(self.page) < seq_len + 2:
            raise ValueError("page smaller than seq_len + 2; lower --seq-len")
        if self.left <= 0:
            self.refill()
        self.left -= 1
        hi = len(self.page) - seq_len - 1
        starts = torch.randint(0, hi, (batch_size,)).tolist()
        xs = [torch.from_numpy(np.array(self.page[s:s + seq_len], dtype=np.int64))
              for s in starts]
        ys = [torch.from_numpy(np.array(self.page[s + 1:s + seq_len + 1], dtype=np.int64))
              for s in starts]
        x = torch.stack(xs).to(device, non_blocking=True)
        y = torch.stack(ys).to(device, non_blocking=True)
        return x, y


# ---------------------------------------------------------------------------
# Shared training helpers (precision / schedule / eval)
# ---------------------------------------------------------------------------


def pick_precision():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
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
    t = min((step - warmup) / max(1, max_steps - warmup), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def estimate_loss(model: nn.Module, stream: PagedStreamer, batches: int,
                  batch_size: int, seq_len: int, device: torch.device,
                  amp_dtype, use_amp: bool) -> float:
    model.eval()
    total = 0.0
    dev = "cuda" if device.type == "cuda" else "cpu"
    for _ in range(batches):
        x, y = stream.batch(batch_size, seq_len, device)
        with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / max(batches, 1)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Dom SLM v2.0 pre-training (GQA)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="checkpoints_v2")
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-qhead", type=int, default=8)
    ap.add_argument("--n-kvhead", type=int, default=2)
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
    ap.add_argument("--page-tokens", type=int, default=1 << 20)
    ap.add_argument("--batches-per-page", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--ckpt-name", default="dom_v3_checkpoint.pt")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--vocab-json", default="tokenizer/vocab.json",
                    help="byte vocab for the optional --gen-prompt demo")
    ap.add_argument("--merges", default="tokenizer/merges.txt")
    ap.add_argument("--gen-prompt", default="",
                    help="post-training generation demo prompt (empty = skip)")
    ap.add_argument("--gen-tokens", type=int, default=50,
                    help="tokens to generate for --gen-prompt")
    return ap.parse_args(argv)


def maybe_generate(model: DomSLMv2, device: torch.device, args) -> None:
    if not args.gen_prompt:
        return
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    from prepare_dataset import BpeEncoder, load_merges, load_vocab
    vocab = load_vocab(args.vocab_json)
    rank = load_merges(args.merges)
    enc = BpeEncoder(vocab, rank)
    ids = enc.encode_text(args.gen_prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, args.gen_tokens)
    gen_ids = out[0].tolist()[len(ids):]
    print("prompt:", args.gen_prompt)
    print("gen ids:", gen_ids)
    print("gen bytes:", b"".join(vocab[i] for i in gen_ids))


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
    print(f"device={device} amp={'off' if not use_amp else amp_dtype} "
          f"scaler={'on' if scaler is not None else 'off'}", flush=True)

    train_stream = PagedStreamer(os.path.join(args.data_dir, "train.bin"),
                                 args.page_tokens, args.batches_per_page)
    valid_stream = PagedStreamer(os.path.join(args.data_dir, "valid.bin"),
                                 args.page_tokens, args.batches_per_page)
    print(f"train tokens={len(train_stream.mem)} "
          f"valid tokens={len(valid_stream.mem)}", flush=True)

    model = DomSLMv2(args.vocab_size, args.d_model, args.n_layer,
                     args.n_qhead, args.n_kvhead, args.d_ffn,
                     args.seq_len).to(device)
    n_params = model.num_params()
    print(f"DomSLMv2 params={n_params} (~{n_params / 1e6:.2f}M) "
          f"kv_bytes_per_tok={model.kv_cache_bytes_per_token()}", flush=True)
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
            x, y = train_stream.batch(args.batch_size, args.seq_len, device)
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
            val = estimate_loss(model, valid_stream, args.eval_batches,
                                args.batch_size, args.seq_len, device,
                                amp_dtype, use_amp)
            print(f"  [eval] step {step + 1} val_loss {val:.4f}", flush=True)
            if val < best_val:
                best_val = val
                torch.save({"step": step, "config": vars(args),
                            "model": model.state_dict(),
                            "optim": opt.state_dict(),
                            "val_loss": val},
                           os.path.join(args.out_dir, "dom_v3_best.pt"))
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            torch.save({"step": step, "config": vars(args),
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "val_loss": best_val},
                       os.path.join(args.out_dir, args.ckpt_name))
            print(f"  [ckpt] saved {args.ckpt_name}", flush=True)

    maybe_generate(model, device, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

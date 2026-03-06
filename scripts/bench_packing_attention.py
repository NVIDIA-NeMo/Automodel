#!/usr/bin/env python3
"""Benchmark attention backends for sequence packing scenarios.

Compares:
  1. SDPA with 4D block-causal bool mask  (current implementation)
  2. SDPA with 4D float mask (-inf)
  3. SDPA with is_causal=True, no mask    (baseline / no packing isolation)
  4. FlashAttention2 varlen (flash_attn)  via cu_seqlens
  5. FlexAttention with BlockMask         (PyTorch native)

Usage:
  python scripts/bench_packing_attention.py [--seq-len 8192] [--num-docs 4] [--batch 1]
"""

import argparse
import time
from functools import partial

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_packed_inputs(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    num_docs: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create Q, K, V and packing metadata for benchmarking."""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

    # Create document boundaries: split seq_len into num_docs roughly equal parts
    doc_lens = []
    remaining = seq_len
    for i in range(num_docs):
        if i < num_docs - 1:
            length = seq_len // num_docs
        else:
            length = remaining
        doc_lens.append(length)
        remaining -= length

    # Build indexed attention mask [B, S]: each position has its 1-based doc index
    doc_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
    offset = 0
    for doc_idx, length in enumerate(doc_lens):
        doc_ids[offset : offset + length] = doc_idx + 1
        offset += length
    doc_ids_batch = doc_ids.unsqueeze(0).expand(batch_size, -1)  # [B, S]

    # Build cu_seqlens for flash_attn varlen
    cu_seqlens_list = [0]
    for b in range(batch_size):
        for length in doc_lens:
            cu_seqlens_list.append(cu_seqlens_list[-1] + length)
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)
    max_seqlen = max(doc_lens)

    return q, k, v, doc_ids_batch, doc_ids, doc_lens, cu_seqlens, max_seqlen


def build_4d_block_causal_mask(doc_ids_batch: torch.Tensor) -> torch.Tensor:
    """Replicate _indexed_mask_to_4d_block_causal from the codebase."""
    B, S = doc_ids_batch.shape
    mask_q = doc_ids_batch.unsqueeze(2)  # [B, S, 1]
    mask_k = doc_ids_batch.unsqueeze(1)  # [B, 1, S]
    same_doc = mask_q == mask_k  # [B, S, S]
    causal = torch.ones(S, S, dtype=torch.bool, device=doc_ids_batch.device).tril()
    not_pad_q = (doc_ids_batch > 0).unsqueeze(2)
    not_pad_k = (doc_ids_batch > 0).unsqueeze(1)
    mask_4d = same_doc & causal.unsqueeze(0) & not_pad_q & not_pad_k
    return mask_4d.unsqueeze(1)  # [B, 1, S, S]


def bench(fn, warmup: int = 10, repeats: int = 50, label: str = ""):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    print(f"  {label:45s}  avg={avg_ms:7.2f} ms  min={min_ms:7.2f} ms  max={max_ms:7.2f} ms")
    return avg_ms


# ---------------------------------------------------------------------------
# Attention implementations
# ---------------------------------------------------------------------------

def attn_sdpa_4d_bool(q, k, v, mask_4d_bool):
    """SDPA with 4D bool mask (current packing implementation)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d_bool)


def attn_sdpa_4d_float(q, k, v, mask_4d_float):
    """SDPA with 4D float mask (-inf for masked positions)."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d_float)


def attn_sdpa_causal(q, k, v):
    """SDPA with is_causal=True (no packing isolation, baseline)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def attn_flash_varlen(q, k, v, cu_seqlens, max_seqlen, batch_size, num_docs):
    """FlashAttention2 varlen interface with cu_seqlens."""
    from flash_attn import flash_attn_varlen_func

    # flash_attn_varlen_func expects [total_tokens, num_heads, head_dim]
    # q/k/v are [B, H, S, D] -> reshape to [B*S, H, D]
    B, H, S, D = q.shape
    q_flat = q.transpose(1, 2).reshape(B * S, H, D)
    k_flat = k.transpose(1, 2).reshape(B * S, H, D)
    v_flat = v.transpose(1, 2).reshape(B * S, H, D)

    out = flash_attn_varlen_func(
        q_flat, k_flat, v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )
    return out.reshape(B, S, H, D).transpose(1, 2)


def attn_flex(q, k, v, block_mask):
    """FlexAttention with block-causal document mask."""
    from torch.nn.attention.flex_attention import flex_attention
    return flex_attention(q, k, v, block_mask=block_mask)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark packing attention backends")
    parser.add_argument("--seq-len", type=int, default=8192, help="Total packed sequence length")
    parser.add_argument("--num-docs", type=int, default=4, help="Number of documents packed together")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=50, help="Timed iterations")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"=== Packing Attention Benchmark ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  seq_len={args.seq_len}, num_docs={args.num_docs}, batch={args.batch}")
    print(f"  num_heads={args.num_heads}, head_dim={args.head_dim}, dtype={args.dtype}")
    print(f"  warmup={args.warmup}, repeats={args.repeats}")
    print()

    q, k, v, doc_ids_batch, doc_ids, doc_lens, cu_seqlens, max_seqlen = make_packed_inputs(
        args.batch, args.seq_len, args.num_heads, args.head_dim, args.num_docs, dtype=dtype,
    )

    print(f"  Doc lengths: {doc_lens}")
    print()

    results = {}

    # --- 1. SDPA with 4D bool mask ---
    mask_4d_bool = build_4d_block_causal_mask(doc_ids_batch)
    results["sdpa_4d_bool"] = bench(
        partial(attn_sdpa_4d_bool, q, k, v, mask_4d_bool),
        warmup=args.warmup, repeats=args.repeats,
        label="SDPA + 4D bool mask (current)",
    )
    del mask_4d_bool

    # --- 2. SDPA with 4D float mask ---
    mask_4d_bool2 = build_4d_block_causal_mask(doc_ids_batch)
    mask_4d_float = torch.where(
        mask_4d_bool2,
        torch.zeros(1, dtype=dtype, device="cuda"),
        torch.full((1,), float("-inf"), dtype=dtype, device="cuda"),
    )
    del mask_4d_bool2
    results["sdpa_4d_float"] = bench(
        partial(attn_sdpa_4d_float, q, k, v, mask_4d_float),
        warmup=args.warmup, repeats=args.repeats,
        label="SDPA + 4D float mask (-inf)",
    )
    del mask_4d_float

    # --- 3. SDPA causal (baseline, no packing isolation) ---
    results["sdpa_causal"] = bench(
        partial(attn_sdpa_causal, q, k, v),
        warmup=args.warmup, repeats=args.repeats,
        label="SDPA + is_causal=True (no packing)",
    )

    # --- 4. FlashAttention2 varlen ---
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        results["flash_varlen"] = bench(
            partial(attn_flash_varlen, q, k, v, cu_seqlens, max_seqlen, args.batch, args.num_docs),
            warmup=args.warmup, repeats=args.repeats,
            label="FlashAttn2 varlen (cu_seqlens)",
        )
    except ImportError:
        print("  FlashAttn2 varlen: SKIPPED (flash_attn not installed)")

    # --- 5. FlexAttention ---
    try:
        from torch.nn.attention.flex_attention import (
            create_block_mask,
            flex_attention as _flex_attention,
        )

        # Build block mask from document IDs
        # doc_ids_flat: [S] tensor on CUDA
        doc_ids_flat = doc_ids.clone()  # [S]

        def doc_causal_mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (doc_ids_flat[q_idx] == doc_ids_flat[kv_idx])

        print("  (compiling FlexAttention block mask...)")
        block_mask = create_block_mask(
            doc_causal_mask,
            B=args.batch, H=args.num_heads,
            Q_LEN=args.seq_len, KV_LEN=args.seq_len,
            device="cuda",
        )

        compiled_flex = torch.compile(_flex_attention)

        # Extra warmup for compile
        for _ in range(3):
            compiled_flex(q, k, v, block_mask=block_mask)
        torch.cuda.synchronize()

        results["flex_attention"] = bench(
            partial(attn_flex, q, k, v, block_mask),
            warmup=args.warmup, repeats=args.repeats,
            label="FlexAttention (compiled, BlockMask)",
        )
    except Exception as e:
        print(f"  FlexAttention: SKIPPED ({e})")

    # --- Summary ---
    print()
    print("=== Summary (sorted by speed) ===")
    for name, ms in sorted(results.items(), key=lambda x: x[1]):
        speedup = results.get("sdpa_4d_bool", ms) / ms if ms > 0 else 0
        bar = "#" * int(speedup * 20)
        print(f"  {name:25s}  {ms:7.2f} ms  {speedup:5.2f}x vs current  {bar}")


if __name__ == "__main__":
    main()

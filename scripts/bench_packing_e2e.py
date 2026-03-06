#!/usr/bin/env python3
"""End-to-end mock benchmark: packing + sdpa vs packing + flash_attention_2.

Simulates a realistic Qwen3-VL-8B forward+backward pass with packed sequences
using mock data. No real model weights or dataset needed.

Usage:
  python scripts/bench_packing_e2e.py [--seq-len 8192] [--num-docs 4] [--layers 4]
"""

import argparse
import gc
import time

import torch
import torch.nn.functional as F


def build_mock_batch(
    batch_size: int,
    seq_len: int,
    num_docs: int,
    vocab_size: int,
    attn_implementation: str,
    device: str = "cuda",
):
    """Build a mock packed batch mimicking neat_packed_vlm_collater output."""
    # input_ids and labels
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Build indexed attention mask: [1,1,...,2,2,...,num_docs,...,0,0]
    doc_len = seq_len // num_docs
    attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    for d in range(num_docs):
        start = d * doc_len
        end = start + doc_len
        attention_mask[:, start:end] = d + 1

    # position_ids: mRoPE [3, B, S]
    pos = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=device)
    for d in range(num_docs):
        start = d * doc_len
        end = start + doc_len
        pos[:, :, start:end] = torch.arange(doc_len, device=device).view(1, 1, -1)

    if attn_implementation == "flash_attention_2":
        # Keep indexed [B, S] mask
        mask_out = attention_mask
    else:
        # Convert to 4D block-causal mask
        B, S = attention_mask.shape
        mask_q = attention_mask.unsqueeze(2)
        mask_k = attention_mask.unsqueeze(1)
        same_doc = mask_q == mask_k
        causal = torch.ones(S, S, dtype=torch.bool, device=device).tril()
        not_pad_q = (attention_mask > 0).unsqueeze(2)
        not_pad_k = (attention_mask > 0).unsqueeze(1)
        mask_out = (same_doc & causal.unsqueeze(0) & not_pad_q & not_pad_k).unsqueeze(1)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": mask_out,
        "position_ids": pos,
    }


def build_model(
    attn_implementation: str,
    num_layers: int = 4,
    device: str = "cuda",
):
    """Load a real Qwen3-VL-8B model with reduced layers for benchmarking."""
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(
        "/mnt/amlfs-01/home/zhiqil/checkpoints/Qwen3-VL-8B-Instruct",
        attn_implementation=attn_implementation,
    )
    # Reduce layers for faster benchmark
    config.text_config.num_hidden_layers = num_layers

    model = AutoModelForImageTextToText.from_config(config).to(dtype=torch.bfloat16, device=device)
    model.gradient_checkpointing_enable()
    model.train()
    return model


def _get_seqlens_in_batch(attention_mask: torch.Tensor) -> torch.Tensor:
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_num = torch.max(attention_mask).item()
    counts = torch.zeros((bsz, max_num), dtype=dtype, device=device)
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)
    counts = counts.flatten()
    return counts[counts.nonzero().squeeze(dim=-1)]


def _get_unpad_data(attention_mask: torch.Tensor):
    seqlens = _get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def apply_packing_patches(attn_implementation: str):
    """Apply monkey-patches for flash attention packing."""
    if attn_implementation != "flash_attention_2":
        return

    import transformers.modeling_flash_attention_utils
    transformers.modeling_flash_attention_utils._get_unpad_data = _get_unpad_data

    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl
        def _passthrough_mask(config, input_embeds, attention_mask, cache_position,
                              past_key_values, position_ids, **kwargs):
            return attention_mask
        modeling_qwen3_vl.create_causal_mask = _passthrough_mask
    except ImportError:
        pass


def benchmark_one(
    attn_impl: str,
    batch_size: int,
    seq_len: int,
    num_docs: int,
    num_layers: int,
    warmup: int,
    repeats: int,
):
    """Run forward+backward benchmark for one attention implementation."""
    device = "cuda"

    apply_packing_patches(attn_impl)
    model = build_model(attn_impl, num_layers=num_layers, device=device)
    vocab_size = model.config.text_config.vocab_size

    batch = build_mock_batch(batch_size, seq_len, num_docs, vocab_size, attn_impl, device)

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def step():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            logits_to_keep=1,
        )
        logits = out.logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["labels"][:, -1:].reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        model.zero_grad()
        return loss.detach().item()

    # Warmup
    print(f"  [{attn_impl}] Warming up ({warmup} steps)...")
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    # Timed runs
    print(f"  [{attn_impl}] Benchmarking ({repeats} steps)...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    losses = []

    for i in range(repeats):
        start_events[i].record()
        l = step()
        end_events[i].record()
        losses.append(l)

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Cleanup
    del model, batch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return avg_ms, min_ms, max_ms, peak_mem, losses[0]


def main():
    parser = argparse.ArgumentParser(description="E2E packing attention benchmark")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-docs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=4, help="Number of transformer layers (fewer = faster)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    print(f"=== E2E Packing Benchmark (fwd + bwd) ===")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  seq_len={args.seq_len}, num_docs={args.num_docs}, batch={args.batch}")
    print(f"  layers={args.layers}, warmup={args.warmup}, repeats={args.repeats}")
    print()

    results = {}

    for attn_impl in ["sdpa", "flash_attention_2"]:
        print(f"--- {attn_impl} ---")
        avg, mn, mx, mem, loss = benchmark_one(
            attn_impl, args.batch, args.seq_len, args.num_docs,
            args.layers, args.warmup, args.repeats,
        )
        results[attn_impl] = (avg, mn, mx, mem, loss)
        print(f"  avg={avg:.1f} ms  min={mn:.1f} ms  max={mx:.1f} ms  peak_mem={mem:.2f} GB  loss={loss:.4f}")
        print()

    # Summary
    sdpa_avg = results["sdpa"][0]
    flash_avg = results["flash_attention_2"][0]
    speedup = sdpa_avg / flash_avg

    print("=== Summary ===")
    print(f"  SDPA  (4D mask):       {sdpa_avg:7.1f} ms/step  peak={results['sdpa'][3]:.2f} GB")
    print(f"  Flash (cu_seqlens):    {flash_avg:7.1f} ms/step  peak={results['flash_attention_2'][3]:.2f} GB")
    print(f"  Speedup:               {speedup:.2f}x")
    print(f"  Memory saving:         {results['sdpa'][3] - results['flash_attention_2'][3]:.2f} GB")


if __name__ == "__main__":
    main()

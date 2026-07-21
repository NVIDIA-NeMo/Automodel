#!/usr/bin/env python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark: Triton fused soft CE vs PyTorch baseline.

Usage:
    python nemo_automodel/components/loss/benchmark/soft_ce_loss.py
    CUDA_VISIBLE_DEVICES=2 python nemo_automodel/components/loss/benchmark/soft_ce_loss.py
"""

import time

import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.triton.soft_cross_entropy import fused_soft_cross_entropy


def pytorch_soft_ce(logits, target_probs, mask):
    """Reference PyTorch soft cross-entropy used as the benchmark baseline."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_token_loss = -(target_probs.float() * log_probs).sum(dim=-1)
    valid_mask = mask.squeeze(-1).float()
    valid_count = valid_mask.sum().clamp_min(1.0)
    return (per_token_loss * valid_mask).sum() / valid_count


def bench_fn(fn, logits, target_probs, mask, warmup=5, iters=10, backward=True):
    """Time ``fn`` (optionally including backward) and return seconds per iteration."""
    for _ in range(warmup):
        loss = fn(logits, target_probs, mask)
        if backward:
            loss.backward()
            logits.grad = None
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        loss = fn(logits, target_probs, mask)
        if backward:
            loss.backward()
            logits.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def measure_memory(fn, logits, target_probs, mask, backward=True):
    """Return peak CUDA memory (bytes) for a single forward/backward of ``fn``."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loss = fn(logits, target_probs, mask)
    if backward:
        loss.backward()
        logits.grad = None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def run_shape_benchmark(shapes, mask_ratio=0.8, dtype=torch.bfloat16):
    """Print a speed/memory/accuracy comparison table across the given tensor shapes."""
    device = "cuda"
    print(f"\n{'=' * 110}")
    print(f" Shape Benchmark | mask_ratio={mask_ratio}, dtype={dtype}")
    print(f"{'=' * 110}")
    print(
        f"{'Shape':<22} | {'PyTorch':>9} | {'Triton':>9} | {'Speedup':>7} "
        f"| {'PT Mem':>8} | {'Tri Mem':>8} | {'Saved':>8} | {'Loss diff':>10} | {'Grad cos':>10}"
    )
    print("-" * 110)

    for B, S, V in shapes:
        torch.cuda.empty_cache()
        try:
            logits = torch.randn(B, S, V, dtype=dtype, device=device, requires_grad=True)
            logits_ref = logits.detach().clone().requires_grad_(True)
            target_probs = torch.softmax(torch.randn(B, S, V, device=device), dim=-1).to(dtype)
            mask = (torch.rand(B, S, device=device) < mask_ratio).unsqueeze(-1).float()

            # Precision
            ref_loss = pytorch_soft_ce(logits_ref, target_probs, mask)
            ref_loss.backward()
            tri_loss = fused_soft_cross_entropy(logits, target_probs, mask)
            tri_loss.backward()
            loss_diff = abs(tri_loss.item() - ref_loss.item())
            grad_cos = F.cosine_similarity(
                logits.grad.flatten().float(), logits_ref.grad.flatten().float(), dim=0
            ).item()
            logits.grad = None
            logits_ref.grad = None
            del logits_ref, ref_loss, tri_loss

            # Speed
            t_pt = bench_fn(pytorch_soft_ce, logits, target_probs, mask)
            t_tri = bench_fn(fused_soft_cross_entropy, logits, target_probs, mask)

            # Memory
            mem_pt = measure_memory(pytorch_soft_ce, logits, target_probs, mask)
            mem_tri = measure_memory(fused_soft_cross_entropy, logits, target_probs, mask)

            speedup = t_pt / t_tri
            saved = mem_pt - mem_tri
            tag = f"[{B},{S},{V}]"

            print(
                f"{tag:<22} | {t_pt * 1000:7.2f}ms | {t_tri * 1000:7.2f}ms | {speedup:5.1f}x  "
                f"| {mem_pt / 1e9:6.2f}GB | {mem_tri / 1e9:6.2f}GB | {saved / 1e9:6.2f}GB "
                f"| {loss_diff:10.2e} | {grad_cos:10.8f}"
            )

            del logits, target_probs, mask
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{f'[{B},{S},{V}]':<22} | OOM")
            torch.cuda.empty_cache()


def run_mask_ratio_benchmark(B=1, S=4096, V=32000, dtype=torch.bfloat16):
    """Print speed/accuracy of the fused kernel as the valid-position mask ratio varies."""
    device = "cuda"
    mask_ratios = [1.0, 0.8, 0.5, 0.2, 0.05]

    print(f"\n{'=' * 110}")
    print(f" Mask Ratio Benchmark | shape=[{B},{S},{V}], dtype={dtype}")
    print(f"{'=' * 110}")
    print(
        f"{'Mask ratio':<12} | {'Valid':>7} | {'PyTorch':>9} | {'Triton':>9} "
        f"| {'Speedup':>7} | {'Loss diff':>10} | {'Grad cos':>12}"
    )
    print("-" * 85)

    for ratio in mask_ratios:
        torch.cuda.empty_cache()

        logits = torch.randn(B, S, V, dtype=dtype, device=device, requires_grad=True)
        logits_ref = logits.detach().clone().requires_grad_(True)
        target_probs = torch.softmax(torch.randn(B, S, V, device=device), dim=-1).to(dtype)
        mask = (
            (torch.rand(B, S, device=device) < ratio).unsqueeze(-1).float()
            if ratio < 1.0
            else torch.ones(B, S, 1, device=device)
        )
        valid = int(mask.sum().item())

        # Precision
        ref_loss = pytorch_soft_ce(logits_ref, target_probs, mask)
        ref_loss.backward()
        tri_loss = fused_soft_cross_entropy(logits, target_probs, mask)
        tri_loss.backward()
        loss_diff = abs(tri_loss.item() - ref_loss.item())
        grad_cos = F.cosine_similarity(logits.grad.flatten().float(), logits_ref.grad.flatten().float(), dim=0).item()
        logits.grad = None
        logits_ref.grad = None
        del logits_ref, ref_loss, tri_loss

        # Speed
        t_pt = bench_fn(pytorch_soft_ce, logits, target_probs, mask)
        t_tri = bench_fn(fused_soft_cross_entropy, logits, target_probs, mask)

        print(
            f"{ratio:<12} | {valid:>7} | {t_pt * 1000:7.2f}ms | {t_tri * 1000:7.2f}ms "
            f"| {t_pt / t_tri:5.1f}x  | {loss_diff:10.2e} | {grad_cos:12.10f}"
        )

        del logits, target_probs, mask
        torch.cuda.empty_cache()


def main():
    """Run the shape and mask-ratio benchmarks on the current CUDA device."""
    torch.manual_seed(42)
    device = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {device} | {free / 1e9:.1f}GB free / {total / 1e9:.1f}GB total")

    shapes = [
        (1, 128, 1024),
        (1, 256, 8192),
        (1, 1024, 32000),
        (1, 2048, 32000),
        (1, 4096, 32000),
        (1, 8192, 32000),
        (1, 16384, 32000),
        (1, 32768, 32000),
    ]

    run_shape_benchmark(shapes, mask_ratio=0.8)
    run_mask_ratio_benchmark(B=1, S=4096, V=32000)


if __name__ == "__main__":
    main()

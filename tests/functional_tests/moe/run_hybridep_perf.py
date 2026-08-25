# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure HybridEP dispatch/combine with equal and unequal EP-rank extents.

This is a temporary PR-performance harness, not a CI correctness test.  The
``manual`` mode is intended for the PR base: it pads each rank to a known
aligned group maximum before entering the unmodified dispatcher.  The
``native`` mode is intended for the PR head: it passes the original unequal
extents and lets the dispatcher discover and pad to the group maximum.  Both
modes therefore present identical tensors to the HybridEP kernels, including
all-False/zero metadata for padded rows.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.components.moe.megatron.token_dispatcher import _HybridEPManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # NeMo-CI's generic finetune launcher always supplies this argument.
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=("native", "manual"), default=os.environ.get("HYBRIDEP_BENCH_MODE", "native"))
    parser.add_argument(
        "--token-counts",
        default=os.environ.get(
            "HYBRIDEP_BENCH_TOKEN_COUNTS",
            "2048;2047,2013,1987,1901,1843,1799,1711,1667;2048,1792,1536,1280,1024,768,512,256",
        ),
        help="Semicolon-separated scenarios; each scenario has one count or one count per rank.",
    )
    parser.add_argument("--hidden-size", type=int, default=int(os.environ.get("HYBRIDEP_BENCH_HIDDEN_SIZE", "2048")))
    parser.add_argument("--num-experts", type=int, default=int(os.environ.get("HYBRIDEP_BENCH_NUM_EXPERTS", "256")))
    parser.add_argument("--topk", type=int, default=int(os.environ.get("HYBRIDEP_BENCH_TOPK", "8")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("HYBRIDEP_BENCH_WARMUP", "20")))
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("HYBRIDEP_BENCH_ITERATIONS", "100")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("HYBRIDEP_BENCH_REPEATS", "3")))
    args, _unknown = parser.parse_known_args()
    return args


def _pad_inputs(
    hidden: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    target_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_tokens = target_tokens - hidden.shape[0]
    if pad_tokens <= 0:
        return hidden, routing_map, probs
    return (
        F.pad(hidden, (0, 0, 0, pad_tokens)),
        F.pad(routing_map, (0, 0, 0, pad_tokens)),
        F.pad(probs, (0, 0, 0, pad_tokens)),
    )


def main() -> None:
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.num_experts % world_size:
        raise ValueError(f"num_experts={args.num_experts} must be divisible by world_size={world_size}")
    device = torch.device("cuda", local_rank)
    group = dist.new_group(ranks=list(range(world_size)))
    manager = _HybridEPManager(
        group=group,
        num_local_experts=args.num_experts // world_size,
        num_experts=args.num_experts,
        router_topk=args.topk,
    )

    for scenario, count_spec in enumerate(args.token_counts.split(";")):
        counts = [int(value) for value in count_spec.split(",")]
        if len(counts) == 1:
            counts *= world_size
        if len(counts) != world_size:
            raise ValueError(f"Expected 1 or {world_size} token counts, got {counts}")

        num_tokens = counts[rank]
        target_tokens = ((max(counts) + 3) // 4) * 4
        generator = torch.Generator(device=device).manual_seed(1234 + rank + 1000 * scenario)
        hidden = torch.randn(
            num_tokens,
            args.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        token_ids = torch.arange(num_tokens, device=device).unsqueeze(1)
        expert_offsets = torch.arange(args.topk, device=device).unsqueeze(0)
        token_indices = (token_ids * args.topk + expert_offsets + rank * args.topk) % args.num_experts
        routing_map = torch.zeros((num_tokens, args.num_experts), dtype=torch.bool, device=device)
        routing_map.scatter_(1, token_indices, True)
        probs = torch.zeros((num_tokens, args.num_experts), dtype=torch.float32, device=device)
        probs.scatter_(1, token_indices, 1.0 / args.topk)

        def run_once() -> torch.Tensor:
            iter_hidden, iter_routing_map, iter_probs = hidden, routing_map, probs
            if args.mode == "manual":
                iter_hidden, iter_routing_map, iter_probs = _pad_inputs(
                    iter_hidden,
                    iter_routing_map,
                    iter_probs,
                    target_tokens,
                )
            manager.setup_metadata(iter_routing_map, iter_probs)
            output = manager.combine(manager.dispatch(iter_hidden))
            return output[:num_tokens]

        for _ in range(args.warmup):
            output = run_once()
        torch.cuda.synchronize()
        dist.barrier(group=group)

        for repeat in range(args.repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            stop_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            wall_start = time.perf_counter()
            for _ in range(args.iterations):
                output = run_once()
            stop_event.record()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1_000
            cuda_ms = start_event.elapsed_time(stop_event)

            # Report the slowest rank, which determines distributed iteration latency.
            timings = torch.tensor([wall_ms, cuda_ms], dtype=torch.float64, device=device)
            dist.all_reduce(timings, op=dist.ReduceOp.MAX, group=group)
            if rank == 0:
                result = {
                    "mode": args.mode,
                    "scenario": scenario,
                    "repeat": repeat,
                    "world_size": world_size,
                    "token_counts": counts,
                    "target_tokens": target_tokens,
                    "padding_ratio": world_size * target_tokens / sum(counts),
                    "hidden_size": args.hidden_size,
                    "num_experts": args.num_experts,
                    "topk": args.topk,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "wall_us_per_iteration": timings[0].item() * 1_000 / args.iterations,
                    "cuda_us_per_iteration": timings[1].item() * 1_000 / args.iterations,
                    "output_checksum": output.float().sum().item(),
                }
                print("HYBRIDEP_PERF " + json.dumps(result, sort_keys=True), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

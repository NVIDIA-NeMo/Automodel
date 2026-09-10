# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Actual HybridEP layout and FP32 combine audit; launch with torchrun on >=2 GPUs."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from nemo_automodel.components.moe.megatron.token_dispatcher import MoEFlexTokenDispatcher, TokenDispatcherConfig


def _run_case(pad_multiple: int | None, empty_rank: bool) -> dict:
    rank, world = dist.get_rank(), dist.get_world_size()
    local_experts, hidden, topk = 4, 256, 6
    global_experts = local_experts * world
    tokens = [23 - 4 * owner for owner in range(world)]
    if min(tokens) <= 0:
        raise ValueError("This bounded diagnostic supports at most six ranks")
    generator = torch.Generator().manual_seed(9123)
    all_hidden, all_routes, all_probs = [], [], []
    for owner, count in enumerate(tokens):
        values = (torch.randn(count, hidden, generator=generator) / 8).bfloat16().cuda()
        values[:, 0] = owner + 1
        values[:, 1] = torch.arange(count, device="cuda")
        routes = torch.stack([torch.randperm(global_experts, generator=generator)[:topk] for _ in range(count)])
        routes[-2:] = -1  # User padding, distinct from the dispatcher's alignment padding.
        if empty_rank and owner == world - 1:
            routes.fill_(-1)
        all_hidden.append(values)
        all_routes.append(routes.cuda())
        all_probs.append((torch.rand(count, topk, generator=generator) / 8).cuda())

    config = TokenDispatcherConfig(
        moe_flex_dispatcher_backend="hybridep",
        num_moe_experts=global_experts,
        moe_router_topk=topk,
        moe_permute_fusion=True,
        moe_share_token_dispatcher=False,
        moe_combine_in_fp32=True,
    )
    dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=local_experts,
        local_expert_indices=list(range(rank * local_experts, (rank + 1) * local_experts)),
        config=config,
        ep_group=dist.group.WORLD,
    )
    dispatcher._comm_manager.pad_multiple = pad_multiple
    actual_hidden = all_hidden[rank].detach().clone().requires_grad_()
    actual_probs = all_probs[rank].detach().clone().requires_grad_()
    dispatched, counts, dispatched_probs = dispatcher.token_permutation2(
        actual_hidden,
        num_local_tokens=tokens[rank],
        token_probs=actual_probs,
        token_indices=all_routes[rank],
    )
    raw_counts = dispatcher.get_unpadded_tokens_per_expert().tolist()
    counts = counts.tolist()
    reference_packed, reference_probs, packed_routes = [], [], []
    for expert in range(rank * local_experts, (rank + 1) * local_experts):
        entries = []
        for owner in range(world):
            rows, slots = torch.where(all_routes[owner] == expert)
            entries.extend((owner, int(row), int(slot)) for row, slot in zip(rows, slots))
        padded = counts[expert - rank * local_experts]
        assert raw_counts[expert - rank * local_experts] == len(entries)
        assert padded >= len(entries)
        for owner, row, slot in entries:
            reference_packed.append(all_hidden[owner][row])
            reference_probs.append(all_probs[owner][row, slot])
            packed_routes.append((owner, row, expert))
        for _ in range(padded - len(entries)):
            reference_packed.append(torch.zeros(hidden, dtype=torch.bfloat16, device="cuda"))
            reference_probs.append(torch.zeros((), device="cuda"))
            packed_routes.append((-1, -1, expert))
    torch.testing.assert_close(dispatched, torch.stack(reference_packed), rtol=0, atol=0)
    torch.testing.assert_close(dispatched_probs, torch.stack(reference_probs), rtol=0, atol=0)

    # Independent synthetic expert outputs depend on source identity and expert
    # ID, so incorrect ordering cannot accidentally pass on repeated tokens.
    def contribution(owner, row, expert):
        return (all_hidden[owner][row].float() * (expert + 1) / 16 + expert / 128).bfloat16()

    routed_values = torch.stack(
        [
            contribution(owner, row, expert)
            if owner >= 0
            else torch.full((hidden,), 99, dtype=torch.bfloat16, device="cuda")
            for owner, row, expert in packed_routes
        ]
    ).requires_grad_()
    actual = dispatcher.token_unpermutation(routed_values)
    expected = torch.zeros(tokens[rank], hidden, device="cuda", dtype=torch.float32)
    for expert in range(global_experts):
        rows, _ = torch.where(all_routes[rank] == expert)
        for row in rows.tolist():
            expected[row] += contribution(rank, row, expert).float()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    all_upstream = [torch.randint(-3, 4, (count, hidden), generator=generator).cuda().float() / 8 for count in tokens]
    (actual * all_upstream[rank]).sum().backward()
    expected_gradient = torch.stack(
        [
            all_upstream[owner][row].bfloat16()
            if owner >= 0
            else torch.zeros(hidden, dtype=torch.bfloat16, device="cuda")
            for owner, row, _ in packed_routes
        ]
    )
    torch.testing.assert_close(routed_values.grad, expected_gradient, rtol=0, atol=0)

    # A second dispatch checks the existing HybridEP backward together with the
    # new combine, using exact integer gradients to expose missing/extra routes.
    second, _, _ = dispatcher.token_permutation2(
        actual_hidden,
        num_local_tokens=tokens[rank],
        token_probs=actual_probs,
        token_indices=all_routes[rank],
    )
    combined = dispatcher.token_unpermutation(second)
    combined.sum().backward()
    degree = (all_routes[rank] >= 0).sum(1)
    expected_input_gradient = degree[:, None].expand_as(actual_hidden).bfloat16()
    torch.testing.assert_close(actual_hidden.grad, expected_input_gradient, rtol=0, atol=0)
    return {
        "rank": rank,
        "source_token_counts": tokens,
        "expert_pad_multiple": pad_multiple,
        "empty_source_rank": empty_rank,
        "padded_expert_counts": counts,
        "raw_expert_counts": raw_counts,
        "expert_order_exact": True,
        "routing_probabilities_exact": True,
        "fp32_combine_exact": True,
        "expert_output_gradient_exact": True,
        "input_gradient_exact": True,
        "source_rows": int(actual.numel() // hidden),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    reports = [_run_case(padding, empty) for padding in (None, 16) for empty in (False, True)]
    dist.barrier()
    report = {"rank": dist.get_rank(), "world": dist.get_world_size(), "cases": reports, "passed": True}
    output = args.output.with_name(f"{args.output.stem}.rank{dist.get_rank()}{args.output.suffix}")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

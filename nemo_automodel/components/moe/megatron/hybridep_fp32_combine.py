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

"""Optional FP32 output reduction for the existing HybridEP dispatch path.

HybridEP transports BF16 activations and rounds its local expert sum before
communication. Models requiring a single cast after the routed and shared sum
instead return individual expert contributions with NCCL all-to-all. Only
active routes are exchanged; memory and payload scale with routed tokens, not
the total EP world size. HybridEP still performs the forward token dispatch.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class HybridEPCombineMetadata:
    """Immutable inverse routes belonging to one dispatch, including recomputes.

    ``source_rows`` indexes this rank's padded input tokens in destination-rank
    order. ``source_slots`` [padded local tokens, maximum selected experts]
    indexes returned rows in expert order, with -1 for unused slots.
    ``expert_rows`` indexes the received routes in HybridEP's padded,
    expert-major output. Splits count active routes exchanged with each EP rank.
    """

    group: dist.ProcessGroup
    source_rows: torch.Tensor
    source_slots: torch.Tensor
    expert_rows: torch.Tensor
    send_splits: tuple[int, ...]
    receive_splits: tuple[int, ...]
    num_tokens: int


@torch.no_grad()
def build_hybridep_combine_metadata(
    routing_map: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    raw_tokens_per_expert: torch.Tensor,
    group: dist.ProcessGroup,
) -> HybridEPCombineMetadata:
    """Reproduce the public HybridEP expert order with sparse route metadata.

    Args:
        routing_map: Boolean routes [padded local tokens, global experts].
        tokens_per_expert: GEMM row counts [local experts], including padding.
        raw_tokens_per_expert: Public handle's real row counts [local experts].
        group: EP process group; experts occupy contiguous equal rank ranges.

    Returns:
        Per-dispatch inverse map. No activation tensor is gathered globally.

    HybridEP visits source ranks and tokens in increasing order within each
    expert. NCCL's rank-ordered receive followed by a stable expert sort yields
    that same ordering. Expert padding is skipped explicitly. Token dropping is
    rejected collectively because it changes the inverse route map.
    """
    world = dist.get_world_size(group)
    local_experts = tokens_per_expert.numel()
    if routing_map.ndim != 2 or routing_map.shape[1] != world * local_experts:
        raise ValueError("HybridEP FP32 combine requires equal contiguous expert ranges across EP ranks")
    source_rows, experts = routing_map.nonzero(as_tuple=True)
    destinations = experts // local_experts
    order = destinations.argsort(stable=True)
    # Preserve expert-ascending order within every source token. Each entry is
    # the row returned by the reverse exchange, or -1 for an unused route slot.
    source_counts = torch.bincount(source_rows, minlength=routing_map.shape[0])
    max_routes = int(source_counts.max()) if source_counts.numel() else 0
    source_slots = torch.full((routing_map.shape[0], max_routes), -1, device=routing_map.device, dtype=torch.long)
    source_starts = source_counts.cumsum(0) - source_counts
    slot_ids = torch.arange(source_rows.numel(), device=routing_map.device) - source_starts[source_rows]
    return_rows = torch.empty_like(order)
    return_rows[order] = torch.arange(order.numel(), device=routing_map.device)
    source_slots[source_rows, slot_ids] = return_rows
    source_rows, experts = source_rows[order], experts[order]
    send_counts = torch.bincount(destinations, minlength=world)
    receive_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(receive_counts, send_counts, group=group)
    send_splits = tuple(send_counts.cpu().tolist())
    receive_splits = tuple(receive_counts.cpu().tolist())

    # Only local expert IDs need transport: source token order remains with the
    # sending rank, and NCCL's reverse exchange restores it without sending IDs.
    send_experts = (experts % local_experts).to(torch.int32).contiguous()
    received_experts = torch.empty(sum(receive_splits), dtype=torch.int32, device=routing_map.device)
    dist.all_to_all_single(
        received_experts,
        send_experts,
        output_split_sizes=receive_splits,
        input_split_sizes=send_splits,
        group=group,
    )
    received_experts = received_experts.long()
    raw_counts = torch.bincount(received_experts, minlength=local_experts)
    padded_counts = tokens_per_expert.to(device=routing_map.device, dtype=torch.long)
    raw_actual = raw_tokens_per_expert.to(device=routing_map.device, dtype=torch.long)
    invalid = ((raw_counts != raw_actual) | (padded_counts < raw_counts)).any().to(torch.int32)
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=group)
    if invalid.item():
        raise ValueError("HybridEP FP32 combine requires complete routing without expert token dropping")

    sorted_order = received_experts.argsort(stable=True)
    sorted_experts = received_experts[sorted_order]
    raw_starts = raw_counts.cumsum(0) - raw_counts
    padded_starts = padded_counts.cumsum(0) - padded_counts
    within_expert = torch.arange(received_experts.numel(), device=routing_map.device) - raw_starts[sorted_experts]
    expert_rows = torch.empty_like(received_experts)
    expert_rows[sorted_order] = padded_starts[sorted_experts] + within_expert
    return HybridEPCombineMetadata(
        group, source_rows, source_slots, expert_rows, send_splits, receive_splits, routing_map.shape[0]
    )


class _HybridEPCombineInFP32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output: torch.Tensor, metadata: HybridEPCombineMetadata) -> torch.Tensor:
        """Return individual BF16 expert contributions before a local FP32 sum."""
        selected = expert_output.index_select(0, metadata.expert_rows).contiguous()
        returned = expert_output.new_empty((metadata.source_rows.numel(), expert_output.shape[1]))
        dist.all_to_all_single(
            returned,
            selected,
            output_split_sizes=metadata.send_splits,
            input_split_sizes=metadata.receive_splits,
            group=metadata.group,
        )
        output = torch.zeros(
            metadata.num_tokens, expert_output.shape[1], device=expert_output.device, dtype=torch.float32
        )
        # Sum experts in increasing global expert order, matching the original
        # FP32 accumulation without atomic ordering or a tree reduction. Bound
        # the transient FP32 gather independently of sequence length/top-k.
        chunk_rows = max(1, (4 * 1024 * 1024) // expert_output.shape[1])
        for start in range(0, metadata.num_tokens, chunk_rows):
            stop = start + chunk_rows
            for slot in range(metadata.source_slots.shape[1]):
                rows = metadata.source_slots[start:stop, slot]
                values = returned.index_select(0, rows.clamp_min(0)).float()
                values.masked_fill_(rows[:, None] < 0, 0)
                output[start:stop].add_(values)
        ctx.route_metadata = metadata
        ctx.input_shape = expert_output.shape
        ctx.input_dtype = expert_output.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Reverse the same sparse exchange; padding has exactly zero gradient."""
        metadata = ctx.route_metadata
        selected = grad_output.index_select(0, metadata.source_rows).to(ctx.input_dtype).contiguous()
        returned = selected.new_empty((metadata.expert_rows.numel(), grad_output.shape[1]))
        dist.all_to_all_single(
            returned,
            selected,
            output_split_sizes=metadata.receive_splits,
            input_split_sizes=metadata.send_splits,
            group=metadata.group,
        )
        grad_experts = torch.zeros(ctx.input_shape, dtype=ctx.input_dtype, device=grad_output.device)
        grad_experts.index_copy_(0, metadata.expert_rows, returned)
        return grad_experts, None


def hybridep_combine_in_fp32(expert_output: torch.Tensor, metadata: HybridEPCombineMetadata) -> torch.Tensor:
    """Combine [padded expert rows, hidden] into FP32 [padded local tokens, hidden]."""
    return _HybridEPCombineInFP32.apply(expert_output, metadata)

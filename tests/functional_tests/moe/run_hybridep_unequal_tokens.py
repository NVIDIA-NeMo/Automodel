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

"""HybridEP dispatch/combine and router-gradient parity with unequal token counts.

Ranks retain their real token counts while HybridEP pads routing metadata to a
common slot extent. Compare outputs and activation/router gradients for negotiated
slots and a larger fixed capacity, including checkpoint recomputation.

Run:
    torchrun --standalone --nproc_per_node=2 \
        tests/functional_tests/moe/run_hybridep_unequal_tokens.py

Repeat with --compact-routing, --permute-fusion, and both flags to exercise
all routing/fusion combinations. Each variant checks a local reference with
different expert scales and compares equal and unequal token extents. Add
--activation-checkpointing to compare checkpoint replay with the same unequal
computation without checkpointing. Add --capacity-factor F to run HybridEP
capacity mode: the equal-count run is the blocking calibration dispatch, the
unequal run then dispatches non-blocking into capacity-sized buffers (rows
past the routed count are padding), and the checkpoint replay reuses the host
extent. The pytest launcher uses --all-variants to run the complete matrix,
including dense and compact capacity-mode variants, in one worker launch and
process group. It also runs the dynamic-batch regression from PR #3641 with no
fixed capacity: counts change between calls, the larger rank swaps, and either
rank can be empty, with and without checkpointing, across routing/fusion variants.
Profiler assertions also check that fixed slot capacity removes token-count
collectives and that nested scopes restore negotiation and capacity correctly.
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.moe.megatron.fused_a2a import reset_hybrid_ep_buffer, store_hybrid_ep_jit_cache
from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    TokenDispatcherConfig,
    hybrid_ep_token_capacity,
)
from nemo_automodel.components.moe.parallelizer import _replay_hybridep_dispatch_on_recompute

# Match the MXFP4 EP fixture so both groups can reuse dispatch/combine kernels.
HIDDEN = 512
NUM_EXPERTS = 4
TOPK = 2
FULL_TOKENS = int(os.environ.get("FULL_TOKENS", "5"))
SHORT_TOKENS = int(os.environ.get("SHORT_TOKENS", "3"))  # rank 1 in the unequal run


def run_dispatch_combine(
    dispatcher: MoEFlexTokenDispatcher,
    hidden: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
    *,
    activation_checkpointing: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute distinctly scaled experts and return output and input gradients.

    Args:
        dispatcher: Initialized expert-parallel dispatcher.
        hidden: Hidden states with shape [tokens, hidden].
        indices: Global expert IDs with shape [tokens, top_k], with -1 for masked slots.
        probs: Routing probabilities with shape [tokens, top_k].
        activation_checkpointing: Whether to replay the forward dispatch layout during recomputation.

    Returns:
        Combined output and hidden-state gradient, both with shape [tokens, hidden],
        and probability gradient with shape [tokens, top_k].
    """
    hidden = hidden.detach().requires_grad_(True)
    probs = probs.detach().requires_grad_(True)
    dispatch_handles: list[object] = []

    def dispatch_combine(hidden_states: torch.Tensor, token_probs: torch.Tensor) -> torch.Tensor:
        """Dispatch, scale each expert's rows, and combine on this EP rank.

        Args:
            hidden_states: Per-rank hidden states with shape [tokens, hidden].
            token_probs: Per-rank routing probabilities with shape [tokens, top_k].

        Returns:
            Combined per-rank hidden states with shape [tokens, hidden].
        """
        out, tokens_per_expert, permuted_probs = dispatcher.token_permutation2(
            hidden_states=hidden_states,
            num_local_tokens=hidden_states.shape[0],
            token_probs=token_probs,
            token_indices=indices,
        )
        dispatch_handles.append(dispatcher._comm_manager.handle)
        # Give expert e the scale e+1, so a wrong destination or expert grouping
        # changes the output rather than being hidden by identical experts.
        expert_scales = torch.tensor(dispatcher.local_expert_indices, device=out.device, dtype=out.dtype) + 1
        row_scales = expert_scales.repeat_interleave(tokens_per_expert.to(device=out.device, dtype=torch.int64))
        # Capacity mode hands back capacity-sized buffers: rows past the routed count are padding that the
        # combine never reads, so scale them by zero like a grouped GEMM that stops at offs[-1].
        if row_scales.shape[0] < out.shape[0]:
            row_scales = torch.nn.functional.pad(row_scales, (0, out.shape[0] - row_scales.shape[0]))
        return dispatcher.token_unpermutation(out * (permuted_probs * row_scales.float()).unsqueeze(-1).to(out.dtype))

    if activation_checkpointing:
        context_fn = _replay_hybridep_dispatch_on_recompute(lambda: (nullcontext(), nullcontext()))
        combined = checkpoint(dispatch_combine, hidden, probs, use_reentrant=False, context_fn=context_fn)
    else:
        combined = dispatch_combine(hidden, probs)
    combined.float().square().sum().backward()
    if activation_checkpointing:
        assert len(dispatch_handles) == 2, "checkpoint backward must recompute the dispatch"
        assert dispatch_handles[1] is dispatch_handles[0], "recompute must reuse the checkpoint-forward layout"
    # BF16 dispatch rounds each expert's weighted copy before summing. Compare
    # with a local FP32 oracle using tolerances that allow this rounding.
    reference_hidden = hidden.detach().float().requires_grad_(True)
    reference_probs = probs.detach().clone().requires_grad_(True)
    reference_weights = reference_probs.masked_fill(indices == -1, 0) * (indices + 1)
    reference = reference_hidden * reference_weights.sum(dim=1, keepdim=True)
    reference.square().sum().backward()
    torch.testing.assert_close(combined.float(), reference, rtol=0.02, atol=0.02)
    torch.testing.assert_close(hidden.grad.float(), reference_hidden.grad, rtol=0.02, atol=0.02)
    torch.testing.assert_close(probs.grad, reference_probs.grad, rtol=0.02, atol=0.02)
    return combined.detach(), hidden.grad, probs.grad


def _run_variant(
    ep_group: dist.ProcessGroup,
    *,
    compact_routing: bool,
    permute_fusion: bool,
    activation_checkpointing: bool,
    capacity_factor: float | None = None,
) -> None:
    """Check one routing/fusion variant using the existing two-rank process group.

    With ``capacity_factor`` the first (equal-count) run calibrates HybridEP capacity mode and the
    unequal run dispatches non-blocking into buffers of that capacity.
    """
    rank = dist.get_rank()
    torch.manual_seed(1234 + rank)
    config = TokenDispatcherConfig(
        moe_flex_dispatcher_backend="hybridep",
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_share_token_dispatcher=False,
        moe_hybridep_compact_routing=compact_routing,
        moe_hybridep_permute_fusion=permute_fusion,
        moe_hybridep_capacity_factor=capacity_factor,
    )
    num_local = NUM_EXPERTS // dist.get_world_size()
    dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=num_local,
        local_expert_indices=list(range(rank * num_local, (rank + 1) * num_local)),
        config=config,
        ep_group=ep_group,
    )
    # These direct EP calls use lazy initialization, sharing one buffer across
    # per-call routing/fusion options. PP resource-signature validation is tested separately.
    hidden = torch.randn(FULL_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    indices = torch.stack([torch.randperm(NUM_EXPERTS, device="cuda")[:TOPK] for _ in range(FULL_TOKENS)])
    indices[0] = -1
    probs = torch.rand(FULL_TOKENS, TOPK, dtype=torch.float32, device="cuda")
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Reference: every rank dispatches FULL_TOKENS (equal, unaligned counts by default).
    reference, reference_grad, reference_prob_grad = run_dispatch_combine(dispatcher, hidden.clone(), indices, probs)

    # Unequal: rank 1 truncates to SHORT_TOKENS, forcing the padding path.
    keep = FULL_TOKENS if rank == 0 else SHORT_TOKENS
    unequal, unequal_grad, unequal_prob_grad = run_dispatch_combine(
        dispatcher, hidden[:keep].clone(), indices[:keep], probs[:keep]
    )

    assert unequal.shape == (keep, HIDDEN), f"rank {rank}: got {tuple(unequal.shape)}"
    torch.testing.assert_close(unequal, reference[:keep], rtol=0, atol=0)
    torch.testing.assert_close(unequal_grad, reference_grad[:keep], rtol=0, atol=0)
    torch.testing.assert_close(unequal_prob_grad, reference_prob_grad[:keep], rtol=0, atol=0)
    print(
        f"[rank {rank}] OK: compact={compact_routing}, fusion={permute_fusion}, capacity={capacity_factor}; "
        "output, hidden gradient, and router gradient match the local oracle and equal-count run",
        flush=True,
    )
    if activation_checkpointing:
        checkpointed, checkpointed_grad, checkpointed_prob_grad = run_dispatch_combine(
            dispatcher, hidden[:keep], indices[:keep], probs[:keep], activation_checkpointing=True
        )
        torch.testing.assert_close(checkpointed, unequal, rtol=0, atol=0)
        torch.testing.assert_close(checkpointed_grad, unequal_grad, rtol=0, atol=0)
        torch.testing.assert_close(checkpointed_prob_grad, unequal_prob_grad, rtol=0, atol=0)
        print(
            f"[rank {rank}] OK: checkpoint replay reuses the forward layout; "
            "output, hidden gradient, and router gradient match without checkpointing",
            flush=True,
        )


def _run_dynamic_token_counts(ep_group: dist.ProcessGroup, *, compact_routing: bool, permute_fusion: bool) -> None:
    """Exercise PR #3641's negotiated path across changing two-rank batches.

    Reuse one dispatcher so a stale group maximum or shape cannot be hidden by
    reconstructing it between steps. No fixed/equal-count/static-routing shortcut
    is enabled; the production NCCL collective and HybridEP kernels run normally.
    """
    assert dist.get_world_size(ep_group) == 2
    rank = dist.get_rank(ep_group)
    torch.manual_seed(3641 + rank)
    num_local = NUM_EXPERTS // 2
    dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=num_local,
        local_expert_indices=list(range(rank * num_local, (rank + 1) * num_local)),
        config=TokenDispatcherConfig(
            moe_flex_dispatcher_backend="hybridep",
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=TOPK,
            moe_share_token_dispatcher=False,
            moe_hybridep_equal_token_counts=False,
            moe_benchmark_static_routing=False,
            moe_hybridep_compact_routing=compact_routing,
            moe_hybridep_permute_fusion=permute_fusion,
        ),
        ep_group=ep_group,
    )
    # Original unequal/unaligned cases, then changing the max-owning rank,
    # crossing upstream's 16-token slot boundary, empty ranks, and aligned counts.
    token_counts = ((8, 3), (6, 6), (5, 3), (13, 7), (3, 13), (17, 5), (5, 17), (0, 7), (7, 0), (4, 4))
    with hybrid_ep_token_capacity(None):
        for counts in token_counts:
            num_tokens = counts[rank]
            hidden = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
            indices = (torch.arange(num_tokens * TOPK, device="cuda").reshape(num_tokens, TOPK) + rank) % NUM_EXPERTS
            indices[:1] = -1
            probs = torch.rand(num_tokens, TOPK, dtype=torch.float32, device="cuda").softmax(dim=-1)
            # Each execution also checks output and both gradients against the
            # local FP32 oracle with distinct expert scales.
            eager = run_dispatch_combine(dispatcher, hidden, indices, probs)
            recomputed = run_dispatch_combine(dispatcher, hidden, indices, probs, activation_checkpointing=True)
            for actual, expected in zip(recomputed, eager):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(
                f"[rank {rank}] OK: PR #3641 dynamic counts={counts}, "
                f"compact={compact_routing}, fusion={permute_fusion}; negotiated slots, "
                "output, hidden/router gradients, and checkpoint replay",
                flush=True,
            )


def _run_slot_capacity_collectives(ep_group: dist.ProcessGroup) -> None:
    """Verify real collective counts and numerical parity as capacity scopes change."""
    rank = dist.get_rank(ep_group)
    torch.manual_seed(744 + rank)
    num_local = NUM_EXPERTS // dist.get_world_size(ep_group)
    dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=num_local,
        local_expert_indices=list(range(rank * num_local, (rank + 1) * num_local)),
        config=TokenDispatcherConfig(
            moe_flex_dispatcher_backend="hybridep",
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=TOPK,
            moe_share_token_dispatcher=False,
            moe_hybridep_equal_token_counts=False,
            moe_benchmark_static_routing=False,
        ),
        ep_group=ep_group,
    )
    num_tokens = (17, 5)[rank]
    hidden = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    indices = (torch.arange(num_tokens * TOPK, device="cuda").reshape(num_tokens, TOPK) + rank) % NUM_EXPERTS
    probs = torch.rand(num_tokens, TOPK, dtype=torch.float32, device="cuda").softmax(dim=-1)
    with hybrid_ep_token_capacity(None):
        # Warm up outside profiling and retain a numerical baseline for every scope.
        reference = run_dispatch_combine(dispatcher, hidden, indices, probs)
        for activation_checkpointing in (False, True):

            def check_collectives(expected: int) -> None:
                """Profile real c10d calls while also checking outputs and both gradients."""
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    result = run_dispatch_combine(
                        dispatcher, hidden, indices, probs, activation_checkpointing=activation_checkpointing
                    )
                collective_count = sum(event.count for event in prof.key_averages() if event.key == "c10d::allreduce_")
                assert collective_count == expected, (
                    f"rank {rank}, checkpoint={activation_checkpointing}: "
                    f"expected {expected} token-count all-reduces, got {collective_count}"
                )
                for actual, baseline in zip(result, reference):
                    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

            negotiated_count = 2 if activation_checkpointing else 1
            check_collectives(negotiated_count)
            with hybrid_ep_token_capacity(32):
                check_collectives(0)
                with hybrid_ep_token_capacity(None):
                    check_collectives(negotiated_count)
                check_collectives(0)
            check_collectives(negotiated_count)
            print(
                f"[rank {rank}] OK: slot-capacity scopes, checkpoint={activation_checkpointing}; "
                f"all-reduces={negotiated_count}/0/{negotiated_count}/0/{negotiated_count}, "
                "output and hidden/router gradients match",
                flush=True,
            )


def main() -> None:
    """Run individual diagnostics or the complete CI matrix in one worker launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-routing", action="store_true")
    parser.add_argument("--permute-fusion", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument(
        "--capacity-factor",
        type=float,
        default=None,
        help="HybridEP capacity mode: calibrate on the first dispatch, then dispatch non-blocking at this factor",
    )
    parser.add_argument("--all-variants", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    ep_group = dist.new_group(ranks=list(range(dist.get_world_size())))
    if args.all_variants:
        variants = [
            (compact, fusion, compact and fusion, None) for compact in (False, True) for fusion in (False, True)
        ]
        # Capacity mode, dense and compact routing, with checkpoint replay of the host extent.
        variants += [(compact, False, True, 1.5) for compact in (False, True)]
    else:
        variants = [(args.compact_routing, args.permute_fusion, args.activation_checkpointing, args.capacity_factor)]
    try:
        if args.all_variants:
            for compact in (False, True):
                for fusion in (False, True):
                    _run_dynamic_token_counts(ep_group, compact_routing=compact, permute_fusion=fusion)
            _run_slot_capacity_collectives(ep_group)
        for compact, fusion, activation_checkpointing, capacity_factor in variants:
            start = time.perf_counter()
            for slot_capacity in (None, max(32, FULL_TOKENS)):
                with hybrid_ep_token_capacity(slot_capacity):
                    _run_variant(
                        ep_group,
                        compact_routing=compact,
                        permute_fusion=fusion,
                        activation_checkpointing=activation_checkpointing,
                        capacity_factor=capacity_factor,
                    )
            # Reuse the buffer: routing/fusion are per-call options, and all variants share its shape.
            # Keep the native buffer and loaded kernels alive until all variants finish.
            torch.cuda.synchronize()
            dist.barrier(group=ep_group)
            store_hybrid_ep_jit_cache()
            print(
                f"HybridEP parity: compact={compact} fusion={fusion} capacity={capacity_factor} "
                f"elapsed={time.perf_counter() - start:.2f}s",
                flush=True,
            )
    finally:
        reset_hybrid_ep_buffer()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

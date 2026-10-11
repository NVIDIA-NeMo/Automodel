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

Every rank in a HybridEP group must dispatch the same token extent; the
dispatcher now pads shorter ranks up to the group maximum. With distinct
linear expert scales, each token's combined output depends only on its own routing, so
running the same data once with equal unaligned counts and once with rank 1
truncated must produce identical outputs for the common tokens. Both runs
exercise alignment padding, and truncation additionally exercises unequal counts.

Run:
    torchrun --standalone --nproc_per_node=2 \
        tests/functional_tests/moe/run_hybridep_unequal_tokens.py

Repeat with --compact-routing, --permute-fusion, and both flags to exercise
all routing/fusion combinations. Each variant checks a local reference with
different expert scales and compares equal and unequal token extents. Add
--activation-checkpointing to compare checkpoint replay with the same unequal
computation without checkpointing. The pytest launcher uses --all-variants to
run the complete matrix in one worker launch and process group. Add
--prepared-extent to exercise a PP-prepared dispatch bound with changing and
empty rank extents, multiple microbatches, checkpoint replay, and scope reset.
Add --static-routing to compare fixed, all-valid balanced dispatch against the
normal dynamic path, including actual combine backward and checkpoint replay.
"""

import argparse
import os
import time
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.moe.megatron import fused_a2a
from nemo_automodel.components.moe.megatron.fused_a2a import reset_hybrid_ep_buffer, store_hybrid_ep_jit_cache
from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    TokenDispatcherConfig,
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
    ep_group: dist.ProcessGroup, *, compact_routing: bool, permute_fusion: bool, activation_checkpointing: bool
) -> None:
    """Check one routing/fusion variant using the existing two-rank process group."""
    rank = dist.get_rank()
    torch.manual_seed(1234 + rank)
    config = TokenDispatcherConfig(
        moe_flex_dispatcher_backend="hybridep",
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_share_token_dispatcher=False,
        moe_hybridep_compact_routing=compact_routing,
        moe_hybridep_permute_fusion=permute_fusion,
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
        f"[rank {rank}] OK: compact={compact_routing}, fusion={permute_fusion}; "
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


def _run_prepared_extent(ep_group: dist.ProcessGroup) -> None:
    """Exercise changing PP step bounds, empty ranks, and checkpoint recomputation."""
    rank = dist.get_rank()
    config = TokenDispatcherConfig(
        moe_flex_dispatcher_backend="hybridep",
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_share_token_dispatcher=False,
        moe_hybridep_compact_routing=True,
        moe_hybridep_permute_fusion=True,
    )
    num_local = NUM_EXPERTS // dist.get_world_size()
    dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=num_local,
        local_expert_indices=list(range(rank * num_local, (rank + 1) * num_local)),
        config=config,
        ep_group=ep_group,
    )
    (initializer,) = dispatcher.get_pipeline_runtime_initializers(hidden_dim=HIDDEN, dtype=torch.bfloat16)
    torch.manual_seed(5678 + rank)
    # Fixed local shape while the peer grows detects unsafe local-shape caching.
    # Shrink must discard the previous step's extent; an empty rank still joins.
    for step_index, rank_tokens in enumerate(((3, 5), (3, 13), (3, 3), (1, 0))):
        count = rank_tokens[rank]
        hidden = torch.randn(count, HIDDEN, device="cuda", dtype=torch.bfloat16)
        indices = torch.stack([torch.randperm(NUM_EXPERTS, device="cuda")[:TOPK] for _ in range(max(count, 1))])[:count]
        probs = torch.rand(count, TOPK, device="cuda", dtype=torch.float32)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        reference = run_dispatch_combine(dispatcher, hidden, indices, probs)
        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as collectives:
            initializer.prepare(num_tokens=max(count, 1), device=torch.device("cuda", torch.cuda.current_device()))
            assert collectives.call_count >= 1
            calls_before_scope = collectives.call_count
            with initializer.step_context():
                actual = run_dispatch_combine(dispatcher, hidden, indices, probs, activation_checkpointing=True)
                # A second microbatch has different routing probabilities and
                # fewer valid tokens but uses the same pre-negotiated upper bound.
                shorter = count // 2
                run_dispatch_combine(
                    dispatcher,
                    hidden[:shorter],
                    indices[:shorter],
                    probs[:shorter] * 0.75,
                    activation_checkpointing=True,
                )
            assert collectives.call_count == calls_before_scope
        for result, expected in zip(actual, reference, strict=True):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        # Exiting scope must restore the real dynamic all-reduce path.
        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as collectives:
            after = run_dispatch_combine(dispatcher, hidden, indices, probs)
            assert collectives.call_count == 1
        for result, expected in zip(after, reference, strict=True):
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        print(f"[rank {rank}] prepared extent step {step_index}: tokens={count}, parity/AC/reset OK", flush=True)


def _run_static_routing(ep_group: dist.ProcessGroup, *, compact_routing: bool) -> None:
    """Compare static and dynamic communication on the same fixed routing.

    Args:
        ep_group: Existing two-rank NCCL expert-parallel group.
        compact_routing: Whether the actual dispatcher consumes compact indices.
    """
    rank = dist.get_rank(ep_group)
    num_local = NUM_EXPERTS // dist.get_world_size(ep_group)
    dispatchers = []
    for static in (False, True):
        config = TokenDispatcherConfig(
            moe_flex_dispatcher_backend="hybridep",
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=TOPK,
            moe_share_token_dispatcher=False,
            moe_hybridep_compact_routing=compact_routing,
            moe_benchmark_static_routing=static,
        )
        dispatchers.append(
            MoEFlexTokenDispatcher(
                num_local_experts=num_local,
                local_expert_indices=list(range(rank * num_local, (rank + 1) * num_local)),
                config=config,
                ep_group=ep_group,
            )
        )
    indices = torch.arange(FULL_TOKENS * TOPK, device="cuda").reshape(FULL_TOKENS, TOPK) % NUM_EXPERTS
    probs = torch.full((FULL_TOKENS, TOPK), 1.0 / TOPK, device="cuda", dtype=torch.float32)
    torch.manual_seed(4321 + rank)
    hidden = torch.randn(FULL_TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    calls: list[tuple[int | None, int, bool]] = []
    for step in range(2):
        data = hidden + step * 0.125
        for ac in (False, True):
            reference = run_dispatch_combine(dispatchers[0], data, indices, probs, activation_checkpointing=ac)
            original_dispatch = fused_a2a._hybrid_ep_buffer.dispatch_with_permute

            def observe_dispatch(
                **kwargs: object,
            ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, object]:
                """Observe exact dispatched row counts without retaining CUDA tensors."""
                extent = kwargs.get("num_permuted_tokens")
                assert extent is None or type(extent) is int, "static FW/replay/BW extent must stay on host"
                result = original_dispatch(**kwargs)
                assert extent is None or result[0].shape[0] == extent, "static mode must not add capacity rows"
                handle = kwargs.get("handle") or result[4]
                torch._assert_async((handle[-1] == 0).reshape(()), "static dispatch overflow")
                calls.append((extent, result[0].shape[0], kwargs.get("handle") is not None))
                return result

            with patch.object(fused_a2a._hybrid_ep_buffer, "dispatch_with_permute", observe_dispatch):
                actual = run_dispatch_combine(dispatchers[1], data, indices, probs, activation_checkpointing=ac)
            for observed, expected in zip(actual, reference):
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert calls[0][0] is None
    assert all(type(extent) is int for extent, _, _ in calls[1:])
    assert any(cached for _, _, cached in calls), "backward and AC must exercise cached handles"
    print(f"[rank {rank}] static host extent: compact={compact_routing}, calls={calls}; exact FW/BW", flush=True)


def main() -> None:
    """Run individual diagnostics or the complete CI matrix in one worker launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-routing", action="store_true")
    parser.add_argument("--permute-fusion", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--prepared-extent", action="store_true")
    parser.add_argument("--static-routing", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    ep_group = dist.new_group(ranks=list(range(dist.get_world_size())))
    variants = (
        [(compact, fusion, compact and fusion) for compact in (False, True) for fusion in (False, True)]
        if args.all_variants
        else [(args.compact_routing, args.permute_fusion, args.activation_checkpointing)]
    )
    try:
        for compact, fusion, activation_checkpointing in variants:
            start = time.perf_counter()
            _run_variant(
                ep_group,
                compact_routing=compact,
                permute_fusion=fusion,
                activation_checkpointing=activation_checkpointing,
            )
            # Reuse the buffer: routing/fusion are per-call options, and all variants share its shape.
            # Keep the native buffer and loaded kernels alive until all variants finish.
            torch.cuda.synchronize()
            dist.barrier(group=ep_group)
            store_hybrid_ep_jit_cache()
            print(
                f"HybridEP parity: compact={compact} fusion={fusion} elapsed={time.perf_counter() - start:.2f}s",
                flush=True,
            )
        if args.static_routing:
            for compact in (False, True):
                _run_static_routing(ep_group, compact_routing=compact)
        if args.prepared_extent:
            _run_prepared_extent(ep_group)
            torch.cuda.synchronize()
            dist.barrier(group=ep_group)
    finally:
        reset_hybrid_ep_buffer()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

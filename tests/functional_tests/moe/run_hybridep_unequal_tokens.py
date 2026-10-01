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
running the same data once with equal counts (no padding path) and once with
rank 1 truncated (padding path) must produce identical outputs for the common
tokens.

Run:
    torchrun --standalone --nproc_per_node=2 \
        tests/functional_tests/moe/run_hybridep_unequal_tokens.py

Repeat with --compact-routing, --permute-fusion, and both flags to exercise
all routing/fusion combinations. Each variant checks a local reference with
different expert scales and compares equal and unequal token extents.
"""

import argparse
import os

import torch
import torch.distributed as dist

from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    TokenDispatcherConfig,
)

HIDDEN = 256
NUM_EXPERTS = 4
TOPK = 2
FULL_TOKENS = int(os.environ.get("FULL_TOKENS", "5"))
SHORT_TOKENS = int(os.environ.get("SHORT_TOKENS", "3"))  # rank 1 in the unequal run


def run_dispatch_combine(
    dispatcher: MoEFlexTokenDispatcher,
    hidden: torch.Tensor,
    indices: torch.Tensor,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute distinctly scaled experts and return output and input gradients.

    Args:
        dispatcher: Initialized expert-parallel dispatcher.
        hidden: Hidden states with shape [tokens, hidden].
        indices: Global expert IDs with shape [tokens, top_k], with -1 for masked slots.
        probs: Routing probabilities with shape [tokens, top_k].

    Returns:
        Combined output and hidden-state gradient, both with shape [tokens, hidden],
        and probability gradient with shape [tokens, top_k].
    """
    hidden = hidden.detach().requires_grad_(True)
    probs = probs.detach().requires_grad_(True)
    out, tokens_per_expert, permuted_probs = dispatcher.token_permutation2(
        hidden_states=hidden,
        num_local_tokens=hidden.shape[0],
        token_probs=probs,
        token_indices=indices,
    )
    # Give expert e the scale e+1, so a wrong destination or expert grouping
    # changes the output rather than being hidden by identical experts.
    expert_scales = torch.tensor(dispatcher.local_expert_indices, device=out.device, dtype=out.dtype) + 1
    row_scales = expert_scales.repeat_interleave(tokens_per_expert.to(device=out.device, dtype=torch.int64))
    combined = dispatcher.token_unpermutation(out * (permuted_probs * row_scales.float()).unsqueeze(-1).to(out.dtype))
    combined.float().square().sum().backward()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-routing", action="store_true")
    parser.add_argument("--permute-fusion", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    torch.manual_seed(1234 + rank)

    ep_group = dist.new_group(ranks=list(range(dist.get_world_size())))
    config = TokenDispatcherConfig(
        moe_flex_dispatcher_backend="hybridep",
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_share_token_dispatcher=False,
        moe_hybridep_compact_routing=args.compact_routing,
        moe_hybridep_permute_fusion=args.permute_fusion,
    )
    num_local = NUM_EXPERTS // dist.get_world_size()
    dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=num_local,
        local_expert_indices=list(range(rank * num_local, (rank + 1) * num_local)),
        config=config,
        ep_group=ep_group,
    )
    for initializer in dispatcher.get_pipeline_runtime_initializers(hidden_dim=HIDDEN, dtype=torch.bfloat16):
        initializer.prepare(num_tokens=FULL_TOKENS, device=torch.device("cuda", int(os.environ["LOCAL_RANK"])))

    hidden = torch.randn(FULL_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    indices = torch.stack([torch.randperm(NUM_EXPERTS, device="cuda")[:TOPK] for _ in range(FULL_TOKENS)])
    indices[0] = -1
    probs = torch.rand(FULL_TOKENS, TOPK, dtype=torch.float32, device="cuda")
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Reference: every rank dispatches FULL_TOKENS (equal counts).
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
        f"[rank {rank}] OK: compact={args.compact_routing}, fusion={args.permute_fusion}; "
        "output, hidden gradient, and router gradient match the local oracle and equal-count run"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

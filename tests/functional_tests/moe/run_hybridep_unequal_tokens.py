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

"""HybridEP dispatch/combine parity with unequal per-rank token counts.

Compare negotiated slots and a larger fixed capacity through the production wrapper.
Identity experts allow exact output and activation/routing-gradient comparisons
across unequal and empty ranks. Fixed capacity must never call all_reduce.

Run:
    torchrun --standalone --nproc_per_node=2 \
        tests/functional_tests/moe/run_hybridep_unequal_tokens.py
"""

import os

import torch
import torch.distributed as dist

from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    TokenDispatcherConfig,
    hybrid_ep_token_capacity,
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
    """Run weighted identity experts and differentiate their squared output.

    Args:
        dispatcher: Production EP dispatcher.
        hidden: BF16 tensor of shape [local_tokens, hidden].
        indices: Expert indices of shape [local_tokens, topk].
        probs: Float32 routing weights of shape [local_tokens, topk].

    Returns:
        Combined output and activation gradient [local_tokens, hidden], and
        routing gradient [local_tokens, topk].
    """
    hidden = hidden.detach().requires_grad_(True)
    probs = probs.detach().requires_grad_(True)
    out, _tokens_per_expert, permuted_probs = dispatcher.token_permutation2(
        hidden_states=hidden,
        num_local_tokens=hidden.shape[0],
        token_probs=probs,
        token_indices=indices,
    )
    # Identity experts: combine returns each token's prob-weighted sum of its
    # own dispatched copies, independent of every other token.
    combined = dispatcher.token_unpermutation((out.float() * permuted_probs.unsqueeze(-1)).to(out.dtype))
    combined.float().square().sum().backward()
    return combined.detach(), hidden.grad, probs.grad


def main():
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
    )
    num_local = NUM_EXPERTS // dist.get_world_size()
    dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=num_local,
        local_expert_indices=list(range(rank * num_local, (rank + 1) * num_local)),
        config=config,
        ep_group=ep_group,
    )

    hidden = torch.randn(FULL_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    indices = torch.stack([torch.randperm(NUM_EXPERTS, device="cuda")[:TOPK] for _ in range(FULL_TOKENS)])
    probs = torch.rand(FULL_TOKENS, TOPK, dtype=torch.float32, device="cuda")
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Reference: every rank dispatches FULL_TOKENS (equal counts).
    reference, reference_grad, reference_prob_grad = run_dispatch_combine(dispatcher, hidden.clone(), indices, probs)

    original_all_reduce = dist.all_reduce

    def unexpected_all_reduce(*args, **kwargs):
        raise AssertionError("Fixed-capacity dispatch must not negotiate token counts")

    for capacity in (None, max(32, FULL_TOKENS)):
        with hybrid_ep_token_capacity(capacity):
            if capacity is not None:
                dist.all_reduce = unexpected_all_reduce
            try:
                for count in (SHORT_TOKENS, 0, FULL_TOKENS):
                    keep = FULL_TOKENS if rank == 0 else count
                    actual = run_dispatch_combine(dispatcher, hidden[:keep], indices[:keep], probs[:keep])
                    for value, expected in zip(actual, (reference, reference_grad, reference_prob_grad)):
                        torch.testing.assert_close(value, expected[:keep], rtol=0, atol=0)
                    assert actual[0].shape == (keep, HIDDEN)
            finally:
                dist.all_reduce = original_all_reduce
    print(f"[rank {rank}] PASS: dynamic/fixed slots, unequal/empty ranks, output + hidden/prob gradients")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

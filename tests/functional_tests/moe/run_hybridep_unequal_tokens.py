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

Every rank in a HybridEP group must dispatch the same token extent; the
dispatcher now pads shorter ranks up to the group maximum. With identity
"experts", each token's combined output depends only on its own routing, so
running the same data once with equal counts (no padding path) and once with
rank 1 truncated (padding path) must produce identical outputs for the common
tokens.

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
)

HIDDEN = 256
NUM_EXPERTS = 4
TOPK = 2
FULL_TOKENS = int(os.environ.get("FULL_TOKENS", "5"))
SHORT_TOKENS = int(os.environ.get("SHORT_TOKENS", "3"))  # rank 1 in the unequal run


def run_dispatch_combine(dispatcher: MoEFlexTokenDispatcher, hidden, indices, probs):
    hidden = hidden.detach().requires_grad_(True)
    out, _tokens_per_expert, _permuted_probs = dispatcher.token_permutation2(
        hidden_states=hidden,
        num_local_tokens=hidden.shape[0],
        token_probs=probs,
        token_indices=indices,
    )
    # Identity experts: combine returns each token's prob-weighted sum of its
    # own dispatched copies, independent of every other token.
    combined = dispatcher.token_unpermutation(out)
    combined.float().square().sum().backward()
    return combined.detach(), hidden.grad


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
    reference, reference_grad = run_dispatch_combine(dispatcher, hidden.clone(), indices, probs)

    # Unequal: rank 1 truncates to SHORT_TOKENS, forcing the padding path.
    keep = FULL_TOKENS if rank == 0 else SHORT_TOKENS
    unequal, unequal_grad = run_dispatch_combine(dispatcher, hidden[:keep].clone(), indices[:keep], probs[:keep])

    assert unequal.shape == (keep, HIDDEN), f"rank {rank}: got {tuple(unequal.shape)}"
    torch.testing.assert_close(unequal, reference[:keep], rtol=0, atol=0)
    torch.testing.assert_close(unequal_grad, reference_grad[:keep], rtol=0, atol=0)
    print(f"[rank {rank}] OK: unequal-count forward AND backward bitwise-match the equal-count run")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

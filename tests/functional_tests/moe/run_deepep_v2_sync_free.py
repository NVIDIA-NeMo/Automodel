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

"""DeepEP V2 sync-free dispatch/combine parity on a tiny distributed MoE.

Run:
    torchrun --standalone --nproc_per_node=2 \
        tests/functional_tests/moe/run_deepep_v2_sync_free.py
"""

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nemo_automodel.components.moe.megatron.fused_a2a import free_buffer
from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
    TokenDispatcherConfig,
)

HIDDEN = 256
NUM_EXPERTS = 4
TOPK = 2
TOKENS = 16


def run_tiny_moe(dispatcher, hidden, indices, probs):
    """Run identity local experts through sync-free dispatch and combine."""
    hidden = hidden.detach().requires_grad_(True)
    dispatched, tokens_per_expert, dispatched_probs = dispatcher.token_permutation2(
        hidden_states=hidden,
        num_local_tokens=hidden.shape[0],
        token_probs=probs,
        token_indices=indices,
    )
    offsets = tokens_per_expert.cumsum(dim=0).to(torch.int32)
    identity = torch.eye(HIDDEN, dtype=hidden.dtype, device=hidden.device).expand(tokens_per_expert.numel(), -1, -1)
    expert_output = torch._grouped_mm(dispatched, identity, offsets)
    valid_rows = torch.arange(expert_output.shape[0], device=expert_output.device) < offsets[-1]
    expert_output = expert_output.masked_fill(~valid_rows.unsqueeze(-1), 0)
    expert_output = expert_output * dispatched_probs.unsqueeze(-1).to(expert_output.dtype)
    combined = dispatcher.token_unpermutation(expert_output)
    combined.float().square().sum().backward()
    return combined.detach(), hidden.grad.detach()


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    torch.manual_seed(1234 + rank)

    group = dist.new_group(ranks=list(range(dist.get_world_size())))
    num_local_experts = NUM_EXPERTS // dist.get_world_size()
    dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=num_local_experts,
        local_expert_indices=list(range(rank * num_local_experts, (rank + 1) * num_local_experts)),
        config=TokenDispatcherConfig(
            moe_flex_dispatcher_backend="deepep",
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=TOPK,
            moe_share_token_dispatcher=False,
            moe_deepep_sync_free=True,
        ),
        ep_group=group,
    )

    hidden = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    indices = torch.stack([torch.randperm(NUM_EXPERTS, device="cuda")[:TOPK] for _ in range(TOKENS)])
    probs = torch.rand(TOKENS, TOPK, dtype=torch.float32, device="cuda")
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Initialize the ElasticBuffer and JIT kernels outside the measured region.
    run_tiny_moe(dispatcher, hidden, indices, probs)
    torch.cuda.synchronize()
    dist.barrier()

    should_profile = rank == 0 and "PROFILE_PATH" in os.environ
    profile_context = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        if should_profile
        else nullcontext()
    )
    with profile_context as prof:
        with torch.profiler.record_function("deepep_v2_measured"):
            actual, actual_grad = run_tiny_moe(dispatcher, hidden, indices, probs)

    torch.testing.assert_close(actual, hidden, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_grad, 2 * hidden, rtol=2e-2, atol=2e-2)

    if should_profile:
        prof.export_chrome_trace(os.environ["PROFILE_PATH"])
        measured = next(event for event in prof.events() if event.name == "deepep_v2_measured")
        sync_names = {
            event.name
            for event in prof.events()
            if measured.time_range.start <= event.time_range.start
            and event.time_range.end <= measured.time_range.end
            and (
                "cudaDeviceSynchronize" in event.name
                or "_local_scalar_dense" in event.name
                or "aten::item" in event.name
            )
        }
        assert not sync_names, f"unexpected CPU-GPU synchronization: {sorted(sync_names)}"

    print(f"[rank {rank}] DeepEP V2 sync-free forward/backward parity OK", flush=True)
    free_buffer()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

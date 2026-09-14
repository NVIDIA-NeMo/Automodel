# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Two-node comms smoke test for the EP16 run: NCCL all-reduce, then DeepEP internode.

Launched through launch_2node_v4_88k.sh with ENTRY pointing at this file. Passes when the
all-reduce bus bandwidth looks like RDMA (hundreds of Gbps, not single digits) and a
DeepEP dispatch -> combine round trip over 16 ranks reproduces its input.
"""

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    """Run the NCCL and DeepEP checks on every rank; rank 0 prints the results."""
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    log(f"world={world} torch={torch.__version__} cuda={torch.version.cuda} nccl={torch.cuda.nccl.version()}")

    for mib in (64, 1024):
        t = torch.ones(mib * 2**20 // 4, device=device)
        for _ in range(3):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        iters = 10
        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(t)
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / iters
        busbw_gbps = mib * 8 / 1024 / seconds * 2 * (world - 1) / world
        log(f"all_reduce {mib:>5} MiB: {seconds * 1e3:7.1f} ms  busbw {busbw_gbps:7.1f} Gbps")

    from deep_ep import Buffer

    log(f"deep_ep NVSHMEM (sm90) compiled: {Buffer.is_sm90_compiled()}")
    # Qwen3.6-35B-A3B: hidden 2048, 256 experts, top-8, bf16 activations.
    hidden, num_experts, topk, num_tokens = 2048, 256, 8, 8192
    hidden_bytes = hidden * 2
    num_nvl_bytes = num_rdma_bytes = 0
    for config in (Buffer.get_dispatch_config(world), Buffer.get_combine_config(world)):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, world), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, world), num_rdma_bytes)
    buffer = Buffer(dist.group.WORLD, num_nvl_bytes, num_rdma_bytes, explicitly_destroy=True)
    log(f"Buffer created: nvl={num_nvl_bytes / 2**30:.2f} GiB rdma={num_rdma_bytes / 2**30:.2f} GiB")

    torch.manual_seed(1234 + rank)
    x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    topk_idx = torch.rand(num_tokens, num_experts, device=device).topk(topk, dim=-1).indices
    topk_weights = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)

    def round_trip() -> tuple[torch.Tensor, torch.Tensor]:
        per_rank, per_rdma_rank, per_expert, in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts)
        recv_x, _, _, _, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=per_rank,
            num_tokens_per_rdma_rank=per_rdma_rank,
            is_token_in_rank=in_rank,
            num_tokens_per_expert=per_expert,
        )
        combined, _, _ = buffer.combine(recv_x, handle)
        return combined, in_rank

    for _ in range(3):
        round_trip()
    torch.cuda.synchronize()
    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        combined, in_rank = round_trip()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1e3

    # combine sums the copies a token sent to each destination rank.
    expected = x.float() * in_rank.sum(dim=1, keepdim=True).float()
    err = ((combined.float() - expected).abs().max() / expected.abs().max()).reshape(1)
    dist.all_reduce(err, op=dist.ReduceOp.MAX)
    log(f"deepep dispatch+combine x{num_tokens} tokens: {ms:.1f} ms/round-trip, max rel err {err.item():.2e}")
    log("COMM_CHECK_PASS" if err.item() < 1e-2 else "COMM_CHECK_FAIL")

    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

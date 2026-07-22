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

"""Multi-rank functional test for the CPTokenLayout token verbs (L1, 2+ GPUs).

The single-process unit suite can only exercise the identity early-returns of
``CPTokenLayout.gather``; this driver runs the real collectives:

  - shard -> gather round-trips a
    caller-coordinate tensor through the round-robin layout (fill/trim);
  - the gather is differentiable: backward through the gathered full-sequence
    tensor routes gradients back to each rank's own local shard, in local
    (head-tail) order.

Run:
    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_cp_sharder_token_verbs.py
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed import ContextParallelRuntime
from nemo_automodel.components.distributed.context_parallel import ContextParallelismSharder


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]

    # Shard a batch whose length needs CP padding (6 -> 8 at cp=2) so the
    # captured layout (original=6, padded=8) drive fill/trim below.
    seq_len = 3 * world
    batch = {
        "input_ids": torch.arange(seq_len, device=device).unsqueeze(0),
        "labels": torch.arange(seq_len, device=device).unsqueeze(0),
    }
    tokens = ContextParallelRuntime(device_mesh=mesh).prepare_forward(None, batch).tokens
    padded = tokens.shard_layout.padded_seq_len
    assert tokens.shard_layout.original_seq_len == seq_len, tokens.shard_layout.original_seq_len
    assert padded == seq_len + (-seq_len) % (2 * world), padded

    # --- down: caller-coordinate tensor rides the same layout -------------
    full = torch.arange(float(seq_len), device=device).unsqueeze(0)
    local = tokens.shard(full, fill=-1.0)
    indices = ContextParallelismSharder.sdpa().local_token_global_indices(cp_mesh, padded, device)
    expected_local = torch.where(indices < seq_len, indices.float(), torch.tensor(-1.0, device=device)).unsqueeze(0)
    assert torch.equal(local, expected_local), (rank, local.tolist(), expected_local.tolist())

    # --- up: differentiable gather + trim back to caller coordinates ------
    local_leaf = local.detach().clone().requires_grad_(True)
    gathered = tokens.gather(local_leaf)
    assert gathered.shape == (1, seq_len), gathered.shape
    # global order restored: position i holds the token with global index i
    assert torch.equal(gathered, full), (rank, gathered.tolist())

    # weight each global position differently; backward must route each
    # position's gradient to the rank that owns it, in local head-tail order.
    # Every rank computes the same replicated full-sequence loss and the
    # differentiable all-gather SUMS the gradient contributions of all
    # consumers, so the local grad is world_size x the single-loss grad — the
    # semantics a consumer must normalize for (cf. the cp_size loss scaling in
    # replicated-loss CP trainers).
    weights = (torch.arange(float(seq_len), device=device) + 1.0).unsqueeze(0)
    (gathered * weights).sum().backward()
    expected_grad = world * torch.where(
        indices < seq_len, (indices + 1).float(), torch.tensor(0.0, device=device)
    ).unsqueeze(0)
    assert torch.equal(local_leaf.grad, expected_grad), (rank, local_leaf.grad.tolist(), expected_grad.tolist())

    dist.barrier()
    if rank == 0:
        print("CP sharder token-verb functional test: OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

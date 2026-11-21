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
"""CUDA parity for explicit FSDP block prefetch scheduling."""

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def _prefetch_worker(
    rank: int,
    init_file: str,
    checkpointed: bool,
) -> None:
    from nemo_automodel.components.distributed.parallelizer_utils import configure_fsdp_prefetch
    from nemo_automodel.components.distributed.utils import get_sync_ctx

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(72)
        blocks = [
            torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.SiLU(), torch.nn.Linear(128, 64)) for _ in range(4)
        ]
        reference = (
            torch.nn.Sequential(*[checkpoint_wrapper(b) if checkpointed else b for b in blocks]).cuda().bfloat16()
        )
        prefetched = copy.deepcopy(reference)
        mesh = init_device_mesh("cuda", (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for candidate in (reference, prefetched):
            for block in candidate:
                fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
            fully_shard(candidate, mesh=mesh, mp_policy=policy)
        configure_fsdp_prefetch(list(prefetched), forward_depth=2, backward_depth=2)
        torch.manual_seed(81 + rank)
        inputs = [torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        upstream = [torch.randn_like(x) for x in inputs]
        runs = []
        for candidate in (reference, prefetched):
            optimizer = torch.optim.SGD(candidate.parameters(), lr=0.02, weight_decay=0.1)
            records = []
            for step, (x_base, grad) in enumerate(zip(inputs, upstream, strict=True)):
                records.append([])
                microbatches = 2 if step == 0 else 1
                for microbatch in range(microbatches):
                    x = (x_base * (microbatch + 1)).requires_grad_()
                    with get_sync_ctx(
                        candidate,
                        is_optim_step=microbatch == microbatches - 1,
                        defer_fsdp_grad_sync=True,
                    ):
                        output = candidate(x)
                        output.backward(grad)
                    records[-1].extend((output.detach(), x.grad.clone()))
                records[-1].extend(p.grad.full_tensor() for p in candidate.parameters())
                optimizer.step()
                records[-1].extend(p.full_tensor() for p in candidate.parameters())
                optimizer.zero_grad(set_to_none=True)
            runs.append(records)
        for expected_step, actual_step in zip(*runs, strict=True):
            for expected, actual in zip(expected_step, actual_step, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@pytest.mark.parametrize("checkpointed", [False, True])
def test_explicit_fsdp_prefetch_preserves_gradients_and_updates(tmp_path: Path, checkpointed: bool) -> None:
    mp.spawn(_prefetch_worker, args=(str(tmp_path / "init"), checkpointed), nprocs=2, join=True)

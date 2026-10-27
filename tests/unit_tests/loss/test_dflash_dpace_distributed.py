# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Distributed optimizer-step parity for D-PACE normalization."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.dllm_loss import DFlashDecayLoss

_BATCH = 2
_BLOCKS = 3
_POSITIONS = 3
_HIDDEN = 5
_VOCAB = 11
_MICROBATCHES = 2


def _inputs(rank: int, microbatch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(100 + 10 * rank + microbatch)
    hidden = torch.randn(_BATCH, _BLOCKS, _POSITIONS, _HIDDEN, generator=generator)
    targets = torch.randint(_VOCAB, (_BATCH, _BLOCKS, _POSITIONS), generator=generator)
    mask = torch.zeros(_BATCH, _BLOCKS, _POSITIONS)
    mask[:, : 1 + 2 * rank] = 1
    return hidden, targets, mask


def _dpace_loss(
    model: torch.nn.Module,
    rank: int,
    microbatch: int,
    denominator: int,
) -> torch.Tensor:
    hidden, targets, mask = _inputs(rank, microbatch)
    logits = model(hidden)
    return DFlashDecayLoss(loss_type="dpace")(
        logits,
        targets,
        mask,
        num_tokens=denominator,
        total_blocks=_BLOCKS,
    ).total_loss


def _distributed_step(rank: int, world_size: int, init_file: str, output_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(7)
        model = DistributedDataParallel(torch.nn.Linear(_HIDDEN, _VOCAB, bias=False))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        denominator = world_size * _MICROBATCHES * _BATCH * _BLOCKS
        for microbatch in range(_MICROBATCHES):
            sync = get_sync_ctx(model, microbatch + 1 == _MICROBATCHES, defer_fsdp_grad_sync=True)
            with sync:
                (_dpace_loss(model, rank, microbatch, denominator) * world_size).backward()
        optimizer.step()
        if rank == 0:
            torch.save(model.module.weight.detach(), output_file)
    finally:
        dist.destroy_process_group()


def test_dpace_ddp_accumulation_matches_dedicated_recipe(tmp_path):
    """The generic distributed path matches the dedicated recipe's local mean."""
    world_size = 2
    init_file = tmp_path / "process-group"
    output_file = tmp_path / "distributed-weight.pt"
    mp.spawn(
        _distributed_step,
        args=(world_size, str(init_file), str(output_file)),
        nprocs=world_size,
        join=True,
    )

    torch.manual_seed(7)
    reference = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    for rank in range(world_size):
        for microbatch in range(_MICROBATCHES):
            local_mean = _dpace_loss(reference, rank, microbatch, _BATCH * _BLOCKS)
            (local_mean / (world_size * _MICROBATCHES)).backward()
    optimizer.step()

    distributed_weight = torch.load(output_file, weights_only=True)
    torch.testing.assert_close(distributed_weight, reference.weight)

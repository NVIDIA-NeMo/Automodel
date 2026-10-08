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

"""Exact forward/backward oracle for windows spanning multiple CP ranks.

Launch with torchrun --standalone --nproc_per_node=3.
"""

import os
from itertools import product

import torch
import torch.distributed as dist
from torch.nn import functional as F

from nemo_automodel.components.models.step5.cp import gather_halo


def main() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world, length = dist.get_rank(), dist.get_world_size(), 7
    assert world >= 3
    global_input = torch.arange(2 * world * length * 3, device="cuda", dtype=torch.float32).view(2, world * length, 3)
    # Zero, short, exact-shard, and multi-shard halos exercise different peers.
    for halo, padding in product((0, 3, 7, 19, world * length), (0, rank + 1)):
        local = global_input[:, rank * length : (rank + 1) * length].clone().requires_grad_()
        actual = gather_halo(local, dist.group.WORLD, halo, left_padding=padding)
        first = max(0, rank * length - halo)
        torch.testing.assert_close(
            actual, F.pad(global_input[:, first : (rank + 1) * length], (0, 0, padding, 0)), atol=0, rtol=0
        )
        ((rank + 1) * actual.square().sum()).backward()
        expected = torch.zeros_like(global_input)
        for consumer in range(world):
            start, stop = max(0, consumer * length - halo), (consumer + 1) * length
            expected[:, start:stop] += 2 * (consumer + 1) * global_input[:, start:stop]
        torch.testing.assert_close(local.grad, expected[:, rank * length : (rank + 1) * length], atol=0, rtol=0)
    if rank == 0:
        print("SWA halo forward/backward: all ten window/padding cases passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

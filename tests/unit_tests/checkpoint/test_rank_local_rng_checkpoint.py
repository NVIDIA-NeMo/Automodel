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

"""Distributed tests for rank-local RNG checkpoint restoration."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig
from nemo_automodel.components.training.rng import StatefulRNG, init_all_rng


def _run_rank_local_rng_roundtrip(rank: int, world_size: int, init_file: str, checkpoint_dir: str) -> None:
    """Verify one rank in a simulated TP2/PP2 topology restores its own RNG stream."""
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        rng = StatefulRNG(seed=123, ranked=True)
        checkpointer = CheckpointingConfig(
            checkpoint_dir=checkpoint_dir,
            save_consolidated=False,
        ).build(
            dp_rank=0,
            tp_rank=rank % 2,
            pp_rank=rank // 2,
            process_group=dist.group.WORLD,
        )

        checkpointer.save_on_global_ranks(rng, "rng", checkpoint_dir)
        expected = (random.random(), float(np.random.rand()), torch.rand(1).item())

        init_all_rng(999)
        checkpointer.load_on_global_ranks(rng, "rng", checkpoint_dir)
        actual = (random.random(), float(np.random.rand()), torch.rand(1).item())
        assert actual == expected

        all_expected = [None] * world_size
        dist.all_gather_object(all_expected, expected)
        assert len(set(all_expected)) == world_size

        dist.barrier()
        if rank == 0:
            rng_dir = os.path.join(checkpoint_dir, "rng")
            assert sorted(os.listdir(rng_dir)) == [f"rng_global_rank_{i}.pt" for i in range(world_size)]
    finally:
        dist.destroy_process_group()


def test_rank_local_rng_roundtrip_with_composed_parallelism(tmp_path):
    """TP/PP peers with one DP rank retain distinct, exactly restorable RNG streams."""
    world_size = 4
    mp.spawn(
        _run_rank_local_rng_roundtrip,
        args=(world_size, str(tmp_path / "dist_init"), str(tmp_path / "checkpoint")),
        nprocs=world_size,
        join=True,
    )

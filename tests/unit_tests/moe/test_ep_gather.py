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

"""Two-rank Gloo coverage for the expert-parallel input gather of the torch dispatcher."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.moe.experts import GroupedExperts

_WORLD_SIZE = 2
_HIDDEN = 8
_TOPK = 2
# Equal counts, ragged counts, and a rank that contributes no tokens.
_LENGTHS = ([4, 4], [3, 6], [5, 0])


def _shard(case: int, rank: int, rows: int) -> dict[str, torch.Tensor]:
    """Return one rank's deterministic router outputs and upstream gradients for a gather case."""
    generator = torch.Generator().manual_seed(100 * case + rank)
    return {
        "x": torch.randn(rows, _HIDDEN, generator=generator),
        "weights": torch.rand(rows, _TOPK, generator=generator),
        "indices": torch.randint(0, 16, (rows, _TOPK), generator=generator),
        "mask": torch.rand(rows, generator=generator) > 0.3,
        "grad_x": torch.randn(rows, _HIDDEN, generator=generator),
        "grad_weights": torch.randn(rows, _TOPK, generator=generator),
    }


def _run_gather(rank: int, init_file: str) -> None:
    """Check every gathered tensor and both input gradients against values rebuilt from each rank's seed."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=_WORLD_SIZE)
    try:
        for case, lengths in enumerate(_LENGTHS):
            shards = [_shard(case, source, lengths[source]) for source in range(_WORLD_SIZE)]
            local = shards[rank]
            x = local["x"].clone().requires_grad_()
            weights = local["weights"].clone().requires_grad_()

            gx, gweights, gindices, gmask, gathered_lens = GroupedExperts._gather_ep_inputs(
                x, local["mask"], weights, local["indices"], ep_group=dist.group.WORLD, ep_size=_WORLD_SIZE
            )

            assert gathered_lens == lengths
            for name, gathered in (("x", gx), ("weights", gweights), ("indices", gindices), ("mask", gmask)):
                expected = torch.cat([shard[name] for shard in shards])
                assert gathered.dtype == expected.dtype
                assert torch.equal(gathered, expected)

            # Every rank sees the same gathered tokens and sends back the gradient of its own slice of each
            # source rank's upstream gradient, so a source rank receives the sum over ranks of its slice.
            upstream_x = torch.cat([shard["grad_x"] for shard in shards]) * (rank + 1)
            upstream_weights = torch.cat([shard["grad_weights"] for shard in shards]) * (rank + 1)
            torch.autograd.backward([gx, gweights], [upstream_x, upstream_weights])
            scale = sum(source + 1 for source in range(_WORLD_SIZE))
            assert torch.allclose(x.grad, local["grad_x"] * scale)
            assert torch.allclose(weights.grad, local["grad_weights"] * scale)
    finally:
        dist.destroy_process_group()


def test_gather_ep_inputs_concatenates_ragged_shards(tmp_path) -> None:
    """Gathered tokens follow rank order without padding, keep dtypes, and return gradients to their source rank."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with Gloo is unavailable")

    mp.spawn(_run_gather, args=(str(tmp_path / "ep_gather_pg"),), nprocs=_WORLD_SIZE, join=True)

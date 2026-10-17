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

"""Two-rank Gloo coverage for the expert-parallel output combine of the torch dispatcher."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.moe.experts import GroupedExperts

_WORLD_SIZE = 2
_HIDDEN = 8
# Equal counts, ragged counts, and a rank that contributes no tokens.
_LENGTHS = ([4, 4], [3, 6], [5, 0])
# [tokens, hidden] and the [tokens, topk, hidden] layout used when routing weights apply after the down projection.
_TRAILING = ((_HIDDEN,), (2, _HIDDEN))


def _rows(seed: int, rows: int, trailing: tuple[int, ...]) -> torch.Tensor:
    """Return a deterministic FP32 tensor of shape [rows, *trailing]."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((rows,) + trailing, generator=generator)


def _run_combine(rank: int, init_file: str) -> None:
    """Check the combined values and gradients on every rank against sums computed locally."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=_WORLD_SIZE)
    try:
        group = dist.group.WORLD
        for case, lengths in enumerate(_LENGTHS):
            start, count = sum(lengths[:rank]), lengths[rank]
            for trailing in _TRAILING:
                # Each rank holds its local experts' contributions for every rank's tokens, in rank order.
                contributions = _rows(100 * case + rank, sum(lengths), trailing).requires_grad_()
                x = torch.randn(sum(lengths), _HIDDEN, requires_grad=True)
                out = GroupedExperts._combine_ep_outputs(contributions * 1.0, x, ep_group=group, gathered_lens=lengths)

                expected = sum(_rows(100 * case + source, sum(lengths), trailing) for source in range(_WORLD_SIZE))
                assert out.shape == (count,) + trailing
                assert torch.equal(out, expected[start : start + count])

                upstream = _rows(1000 + 100 * case + rank, count, trailing)
                out.backward(upstream)
                expected_grad = torch.cat(
                    [_rows(1000 + 100 * case + source, lengths[source], trailing) for source in range(_WORLD_SIZE)]
                )
                assert torch.equal(contributions.grad, expected_grad)
                # The gradient anchor keeps the gathered activations in the graph with a zero gradient.
                assert x.grad is not None and torch.count_nonzero(x.grad) == 0
    finally:
        dist.destroy_process_group()


def test_combine_reduce_scatters_ragged_ep_outputs(tmp_path) -> None:
    """Each rank receives the group sum of its own tokens; backward returns every rank's upstream gradient."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with Gloo is unavailable")

    mp.spawn(_run_combine, args=(str(tmp_path / "ep_combine_pg"),), nprocs=_WORLD_SIZE, join=True)

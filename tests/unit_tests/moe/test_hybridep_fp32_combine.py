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

"""CPU edge cases for the optional sparse HybridEP return transport.

Gloo tests exercise the sparse inverse transport with two real processes.
They do not construct the CUDA-only HybridEP dispatcher or claim NCCL coverage.
"""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.moe.megatron.hybridep_fp32_combine import (
    build_hybridep_combine_metadata,
    hybridep_combine_in_fp32,
)


@pytest.fixture
def group(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    yield dist.group.WORLD
    dist.destroy_process_group()


def test_fp32_combine_preserves_cancellation_and_ignores_padding(group):
    # Both tokens route to experts 0 and 2. Expert 1 has no rows, and padding
    # deliberately contains large values that must never enter the reduction.
    routing = torch.tensor([[True, False, True], [True, False, True], [False, False, False]])
    metadata = build_hybridep_combine_metadata(routing, torch.tensor([4, 0, 4]), torch.tensor([2, 0, 2]), group)
    values = torch.tensor([[256.0], [-256.0], [99.0], [99.0], [0.5], [0.5], [99.0], [99.0]])
    values = values.bfloat16().requires_grad_()
    output = hybridep_combine_in_fp32(values, metadata)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.tensor([[256.5], [-255.5], [0.0]]), rtol=0, atol=0)
    (output * torch.tensor([[2.0], [-3.0], [7.0]])).sum().backward()
    expected_gradient = torch.tensor([[2], [-3], [0], [0], [2], [-3], [0], [0]], dtype=torch.bfloat16)
    torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)


def test_fp32_combine_empty_routes_have_zero_output_and_gradient(group):
    metadata = build_hybridep_combine_metadata(
        torch.zeros(3, 2, dtype=torch.bool), torch.tensor([0, 0]), torch.tensor([0, 0]), group
    )
    values = torch.empty(0, 4, dtype=torch.bfloat16, requires_grad=True)
    output = hybridep_combine_in_fp32(values, metadata)
    torch.testing.assert_close(output, torch.zeros(3, 4), rtol=0, atol=0)
    output.sum().backward()
    assert values.grad.shape == values.shape


def test_fp32_combine_rejects_dropped_expert_rows(group):
    with pytest.raises(ValueError, match="token dropping"):
        build_hybridep_combine_metadata(
            torch.ones(3, 2, dtype=torch.bool), torch.tensor([2, 3]), torch.tensor([2, 3]), group
        )


def _route_values(source, token, expert):
    # The first component exposes the loss from early BF16 accumulation.
    return torch.tensor([256.0 if expert % 2 == 0 else 0.5, source * 16 + token * 2 + expert], dtype=torch.bfloat16)


def _source_gradient(source, token):
    return torch.tensor([source + 2, 1 - token], dtype=torch.float32)


def _two_rank_combine_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        for case in ("unequal", "empty_requester", "all_empty"):
            # Three experts per owner, with expert 4 always unused. Lengths
            # include trailing padding and differ across the source ranks.
            routes = [torch.zeros(4, 6, dtype=torch.bool), torch.zeros(2, 6, dtype=torch.bool)]
            if case != "all_empty":
                routes[0][0, [0, 1, 3, 5]] = True
                routes[0][1, [2, 3]] = True
                routes[0][2, [0, 5]] = True
            if case == "unequal":
                routes[1][0, [1, 2, 5]] = True
            if case == "empty_requester":
                routes[1] = torch.zeros(0, 6, dtype=torch.bool)

            values, gradients, counts, raw_counts = [], [], [], []
            # Independent expected layout: expert-major, then source rank,
            # then token. Never read the implementation's inverse metadata.
            for expert in range(rank * 3, (rank + 1) * 3):
                entries = [
                    (source, token)
                    for source in range(2)
                    for token in range(len(routes[source]))
                    if routes[source][token, expert]
                ]
                raw_counts.append(len(entries))
                count = ((len(entries) + 3) // 4) * 4
                counts.append(count)
                values.extend(_route_values(source, token, expert) for source, token in entries)
                gradients.extend(_source_gradient(source, token).bfloat16() for source, token in entries)
                values.extend(torch.full((2,), 999.0, dtype=torch.bfloat16) for _ in range(count - len(entries)))
                gradients.extend(torch.zeros(2, dtype=torch.bfloat16) for _ in range(count - len(entries)))
            expert_values = (
                torch.stack(values) if values else torch.empty(0, 2, dtype=torch.bfloat16)
            ).requires_grad_()
            expected_gradient = torch.stack(gradients) if gradients else torch.empty_like(expert_values)
            metadata = build_hybridep_combine_metadata(
                routes[rank], torch.tensor(counts), torch.tensor(raw_counts), dist.group.WORLD
            )
            output = hybridep_combine_in_fp32(expert_values, metadata)
            expected = torch.zeros(len(routes[rank]), 2)
            upstream = torch.empty_like(expected)
            for token in range(len(routes[rank])):
                upstream[token] = _source_gradient(rank, token)
                for expert in range(6):
                    if routes[rank][token, expert]:
                        expected[token] += _route_values(rank, token, expert).float()
            assert output.dtype == torch.float32
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            output.backward(upstream)
            torch.testing.assert_close(expert_values.grad, expected_gradient, rtol=0, atol=0)

        # Only one owner's public count is corrupt. Both ranks must reject
        # after metadata exchange rather than leaving the healthy rank hung.
        routes = torch.ones(rank + 1, 6, dtype=torch.bool)
        raw_counts = torch.full((3,), 3)
        if rank == 1:
            raw_counts[0] -= 1
        with pytest.raises(ValueError, match="token dropping"):
            build_hybridep_combine_metadata(routes, torch.full((3,), 4), raw_counts, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


def test_two_rank_fp32_combine_forward_backward_and_collective_rejection(tmp_path):
    mp.spawn(_two_rank_combine_worker, args=(str(tmp_path / "combine-rendezvous"),), nprocs=2, join=True)

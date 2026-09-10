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

The actual HybridEP/NCCL distributed layout and backward are checked separately
by ``run_hybridep_precision.py`` with unequal source lengths and expert padding.
"""

import pytest
import torch
import torch.distributed as dist

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

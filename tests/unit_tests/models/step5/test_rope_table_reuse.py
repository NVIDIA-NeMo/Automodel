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

"""Paired Q/K rotation preserves table and shared-projection gradient semantics."""

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding


@pytest.mark.parametrize("span", [0, 64, 192])
@pytest.mark.parametrize("sequence", [0, 1, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_paired_rope_reuses_tables_and_preserves_strided_qkv_gradients(
    span: int, sequence: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(987)
    config = Step5Config()
    rope = Step5RotaryEmbedding(config, 0, span=span)
    source = torch.randn(2, sequence, 4, 18, 192, dtype=dtype)
    expected_input = source.clone().requires_grad_()
    actual_input = source.clone().requires_grad_()
    positions = torch.arange(sequence).expand(2, -1) + 131071
    positions[:, sequence // 2 :] = torch.arange(sequence - sequence // 2)

    def qk(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value[..., :16, :].reshape(2, sequence, 64, 192), value[..., 16, :]

    expected_q, expected_k = qk(expected_input)
    actual_q, actual_k = qk(actual_input)
    with patch.object(rope, "_tables", wraps=rope._tables) as tables:
        expected = rope(expected_q, positions), rope(expected_k, positions)
        assert tables.call_count == (2 if sequence and span else 0)
        tables.reset_mock()
        actual = rope.forward_pair(actual_q, actual_k, positions)
        assert tables.call_count == (1 if sequence and span else 0)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    upstream = tuple(torch.randn_like(value) for value in expected)
    expected_grad = torch.autograd.grad(expected, expected_input, upstream)[0]
    actual_grad = torch.autograd.grad(actual, actual_input, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    if sequence == 0 or span == 0:
        assert actual[0] is actual_q and actual[1] is actual_k


def test_paired_rope_rejects_different_rounding_dtypes() -> None:
    rope = Step5RotaryEmbedding(Step5Config(), 0, span=64)
    query = torch.randn(1, 1, 64, 192, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 4, 192, dtype=torch.float32)
    with pytest.raises(ValueError, match="matching Q/K"):
        rope.forward_pair(query, key, torch.ones(1, 1, dtype=torch.long))

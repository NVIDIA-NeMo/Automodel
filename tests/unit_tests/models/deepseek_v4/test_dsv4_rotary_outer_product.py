# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4RotaryEmbedding


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotary_outer_product_without_batched_matmul(dtype: torch.dtype, batch_size: int) -> None:
    """RoPE tables preserve FP32 phase math without a lazy batched-matmul kernel."""
    rotary = DeepseekV4RotaryEmbedding(
        rope_theta=10000.0, head_dim=16, partial_rotary_factor=0.5, attention_scaling=1.1
    )
    positions = torch.tensor([[0, 3, 17, 127, 4095], [2, 9, 31, 1023, 8191]])[:batch_size]
    x = torch.empty(batch_size, positions.shape[1], 16, dtype=dtype, requires_grad=True)
    phase = torch.einsum("d,bs->bsd", rotary.inv_freq, positions.float())
    phase = torch.cat((phase, phase), dim=-1)
    expected_cos = (phase.cos() * 1.1).to(dtype)
    expected_sin = (phase.sin() * 1.1).to(dtype)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        cos, sin = rotary(x, positions)

    torch.testing.assert_close(cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(sin, expected_sin, rtol=0, atol=0)
    assert not cos.requires_grad and not sin.requires_grad
    assert cos.shape == (batch_size, positions.shape[1], 8)
    assert "aten::bmm" not in {event.key for event in profile.key_averages()}

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

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("seq_len", [4096, 8192])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_rotary_tables_match_batched_matmul_cuda(seq_len: int, batch_size: int) -> None:
    """The replacement preserves the previous CUDA tables exactly at benchmark lengths."""
    rotary = DeepseekV4RotaryEmbedding(
        rope_theta=160000.0,
        head_dim=512,
        partial_rotary_factor=64 / 512,
        attention_scaling=1.1,
        device=torch.device("cuda"),
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )
    positions = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
    positions = positions + torch.arange(batch_size, device="cuda").unsqueeze(1) * 17
    frequencies = rotary.inv_freq[None, :, None].expand(batch_size, -1, 1)
    phases = torch.bmm(frequencies, positions[:, None, :].float()).transpose(1, 2)
    phases = torch.cat((phases, phases), dim=-1)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.empty(batch_size, seq_len, 1, device="cuda", dtype=dtype, requires_grad=True)
        cos, sin = rotary(x, positions)
        torch.testing.assert_close(cos, (phases.cos() * 1.1).to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(sin, (phases.sin() * 1.1).to(dtype), rtol=0, atol=0)
        assert not cos.requires_grad and not sin.requires_grad

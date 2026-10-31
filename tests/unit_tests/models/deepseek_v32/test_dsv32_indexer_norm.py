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

"""Check indexer key normalization independently of HF and sparse top-k selection."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.layers import DeepseekV32Indexer

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("key_scale", [0.0, 1e-4, 1.0])
@pytest.mark.parametrize("packed", [False, True])
def test_key_norm_matches_layer_norm_equation(key_scale: float, packed: bool) -> None:
    """Check key outputs and input/affine gradients against the LayerNorm equation."""
    config = DeepseekV32Config(
        hidden_size=8,
        q_lora_rank=8,
        index_n_heads=2,
        index_head_dim=8,
        qk_rope_head_dim=4,
        index_topk=4,
        torch_dtype="float32",
    )
    indexer = DeepseekV32Indexer(config, BackendConfig(linear="torch", attn="sdpa", rms_norm="torch")).float()
    generator = torch.Generator().manual_seed(29)
    shape = (12, 8) if packed else (2, 12, 8)
    keys = (torch.randn(shape, generator=generator) * key_scale).requires_grad_()
    with torch.no_grad():
        indexer.k_norm.weight.copy_(torch.linspace(0.7, 1.3, 8))
        indexer.k_norm.bias.copy_(torch.linspace(-0.2, 0.2, 8))
    reference_keys = keys.detach().double().requires_grad_()
    reference_weight = indexer.k_norm.weight.detach().double().requires_grad_()
    reference_bias = indexer.k_norm.bias.detach().double().requires_grad_()
    centered = reference_keys - reference_keys.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    expected = centered * torch.rsqrt(variance + 1e-6) * reference_weight + reference_bias
    actual = indexer.k_norm(keys)
    torch.testing.assert_close(actual, expected.float(), rtol=1e-6, atol=1e-6)

    upstream = torch.randn(actual.shape, generator=generator)
    actual_grads = torch.autograd.grad(actual, (keys, indexer.k_norm.weight, indexer.k_norm.bias), upstream)
    expected_grads = torch.autograd.grad(
        expected, (reference_keys, reference_weight, reference_bias), upstream.double()
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad.float(), rtol=2e-5, atol=1e-5)

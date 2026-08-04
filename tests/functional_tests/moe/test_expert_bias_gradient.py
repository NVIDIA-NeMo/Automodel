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

import pytest
import torch

from nemo_automodel.components.moe.experts import _apply_bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_bias_has_deterministic_bf16_bias_gradient():
    """Imbalanced BF16 routing produces the same trainable bias gradient on every CUDA backward."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    n_experts = 64
    n_tokens = 4096
    hidden = 512

    torch.manual_seed(1234)
    value = torch.randn(n_tokens, hidden, dtype=torch.bfloat16, device=device)
    bias_data = torch.randn(n_experts, hidden, dtype=torch.bfloat16, device=device)
    tokens_per_expert = torch.zeros(n_experts, dtype=torch.long, device=device)
    tokens_per_expert[0] = n_tokens
    upstream_grad = torch.randn_like(value)

    reference_bias = bias_data.clone().requires_grad_()
    reference = torch.cat(
        [
            expert_value + expert_bias
            for expert_value, expert_bias in zip(torch.split(value, tokens_per_expert.tolist()), reference_bias)
        ]
    )
    reference.backward(upstream_grad)
    expected_grad = reference_bias.grad
    assert expected_grad is not None

    for _ in range(10):
        bias = bias_data.clone().requires_grad_()
        result = _apply_bias(value, bias=bias, tokens_per_expert=tokens_per_expert)
        result.backward(upstream_grad)

        assert bias.grad is not None
        torch.testing.assert_close(bias.grad, expected_grad, rtol=0, atol=0)

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

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.layers import Step3p5Attention


@pytest.mark.parametrize("partial_rotary_factor", [0.5, 1.0])
@pytest.mark.parametrize("head_gate", [False, True])
def test_thd_attention_matches_separate_documents_forward_backward(partial_rotary_factor, head_gate):
    """Flat reset positions preserve real RoPE and attention gradients under THD."""
    torch.manual_seed(12)
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=32,
        partial_rotary_factors=[partial_rotary_factor],
        torch_dtype="float32",
        use_head_wise_attn_gate=head_gate,
    )
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    reference = Step3p5Attention(config, layer_idx=0, backend=backend)
    packed = copy.deepcopy(reference)
    packed.backend.attn = "te"

    def cpu_varlen_attention(q, k, v, *, cu_seqlens_q, **kwargs):
        # Substitute only the CUDA TE kernel. Actual Q/K projections, norms,
        # RoPE, THD preprocessing, head gate and output projection still run.
        assert kwargs["qkv_format"] == "thd"
        outputs = []
        boundaries = cu_seqlens_q.tolist()
        for start, end in zip(boundaries, boundaries[1:]):
            heads = [tensor[start:end].transpose(0, 1).unsqueeze(0) for tensor in (q, k, v)]
            out = F.scaled_dot_product_attention(*heads, is_causal=True, enable_gqa=True)
            outputs.append(out.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)

    packed.attn_func = cpu_varlen_attention
    x = torch.randn(7, config.hidden_size, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3])
    output = packed(x, position_ids=positions, qkv_format="thd", cu_seqlens=torch.tensor([0, 3, 7]))
    expected = torch.cat(
        [
            reference(reference_x[:3].unsqueeze(0), position_ids=positions[:3].unsqueeze(0)).squeeze(0),
            reference(reference_x[3:].unsqueeze(0), position_ids=positions[3:].unsqueeze(0)).squeeze(0),
        ]
    )
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    weight = torch.randn_like(output)
    (output * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=1e-6, rtol=1e-5)
    for parameter, reference_parameter in zip(packed.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=2e-6, rtol=1e-5)

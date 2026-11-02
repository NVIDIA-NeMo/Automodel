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

"""Numerical and ownership coverage for M3's opt-in RMSNorm compilation."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3RMSNorm
from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForCausalLM
from nemo_automodel.components.utils.model_utils import freeze_minimax_m3_indexer_params


def _oracle(
    x: torch.Tensor, weight: torch.Tensor, grad: torch.Tensor, *, gemma: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate an FP64 norm and its analytic gradients independently of autograd.

    Args:
        x: Tensor of shape [tokens, hidden].
        weight: Tensor of shape [hidden].
        grad: Output gradient of shape [tokens, hidden].
        gemma: Whether the scale is one plus weight.

    Returns:
        Output and input gradient of shape [tokens, hidden], and weight gradient
        of shape [hidden], all in float64.
    """
    values = x.double()
    scale = weight.double() + int(gemma)
    inverse_rms = torch.sqrt(values.square().sum(-1, keepdim=True) / values.shape[-1] + 1e-6).reciprocal()
    weighted_grad = grad.double() * scale
    dx = inverse_rms * weighted_grad - values * inverse_rms.pow(3) * (weighted_grad * values).mean(-1, keepdim=True)
    dw = (grad.double() * values * inverse_rms).sum(0)
    return values * inverse_rms * scale, dx, dw


@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize(
    "compiled",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.runtime_budget(
                60, hard_timeout=120, reason="Actual cold CPU Inductor forward/backward compilation took 25 seconds."
            ),
        ),
    ],
)
def test_norm_matches_fp64_analytic_forward_and_gradients(gemma: bool, compiled: bool) -> None:
    torch.manual_seed(17)
    layer = MiniMaxM3RMSNorm(32, gemma=gemma, compile_norm=compiled)
    with torch.no_grad():
        layer.weight.uniform_(-0.5, 0.5)
    x = torch.randn(7, 32, requires_grad=True)
    grad = torch.randn_like(x)
    expected, expected_dx, expected_dw = _oracle(x.detach(), layer.weight.detach(), grad, gemma=gemma)
    output = layer(x)
    dx, dw = torch.autograd.grad(output, (x, layer.weight), grad)
    torch.testing.assert_close(output.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(dx.double(), expected_dx, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(dw.double(), expected_dw, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compilation_is_per_instance_and_preserves_initialization(sparse_text_config, dtype: torch.dtype) -> None:
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        compile_norm=True,
    )
    model = MiniMaxM3SparseForCausalLM(sparse_text_config, backend=backend)
    model.initialize_weights(dtype=dtype)
    freeze_minimax_m3_indexer_params(model)
    compiled_norms = [module for module in model.modules() if isinstance(module, MiniMaxM3RMSNorm)]
    assert len(compiled_norms) == 17  # 4 norms/dense layer, 6/sparse layer, final norm.
    assert all(torch.count_nonzero(module.weight).item() == 0 for module in compiled_norms)
    assert all(module.weight.dtype == dtype for module in compiled_norms)
    eager = MiniMaxM3RMSNorm(32)
    assert all(module._norm is not eager._norm for module in compiled_norms)
    for block in model.model.layers.values():
        if block.is_moe_layer:
            assert block.mlp.gate.weight.dtype == torch.float32
            assert block.mlp.gate.e_score_correction_bias.dtype == torch.float32
        if block.self_attn.indexer is not None:
            assert not any(parameter.requires_grad for parameter in block.self_attn.indexer.parameters())

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Parity of the fused Qwen3.5 norms with the modules they replace."""

from unittest import mock

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNormGated

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5 import layers
from nemo_automodel.components.models.qwen3_5.layers import Qwen3_5FusedRMSNorm, Qwen3_5FusedRMSNormGated
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5DenseTextBackbone
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm

# (fused module, module it replaces, number of tensor inputs)
CASES = [
    (Qwen3_5FusedRMSNorm, Qwen3NextRMSNorm, 1),
    (Qwen3_5FusedRMSNormGated, Qwen3_5MoeRMSNormGated, 2),
]


def _modules(fused_cls, reference_cls, hidden, device, weight_dtype):
    torch.manual_seed(0)
    reference = reference_cls(hidden, eps=1e-6).to(device=device, dtype=weight_dtype)
    with torch.no_grad():
        reference.weight.normal_(0.0, 0.5)
    fused = fused_cls(hidden, eps=1e-6).to(device=device, dtype=weight_dtype)
    fused.load_state_dict(reference.state_dict())
    return fused, reference


def _forward_backward(module, inputs, grad_output):
    """Run forward and backward with a fixed upstream gradient.

    Args:
        module: Norm module under test.
        inputs: Tensors of shape [..., hidden]; the gated norm takes (hidden_states, gate).
        grad_output: Tensor of shape [..., hidden] used as the upstream gradient.

    Returns:
        Output of shape [..., hidden], input gradients of the same shape, and the
        weight gradient of shape [hidden].
    """
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = module(*leaves)
    out.backward(grad_output)
    return [out.detach(), *(t.grad for t in leaves), module.weight.grad]


def _inputs(n_inputs, shape, device):
    torch.manual_seed(1)
    inputs = [torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(n_inputs)]
    return inputs, torch.randn(shape, device=device, dtype=torch.bfloat16)


def test_backbone_uses_fused_norms_with_unchanged_parameter_names():
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        attn_implementation="eager",
        torch_dtype="float32",
    )
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", rope_fusion=False, dispatcher="torch")
    backbone = Qwen3_5DenseTextBackbone(config, backend)
    linear, full = backbone.layers["0"], backbone.layers["1"]

    for norm in (
        linear.input_layernorm,
        linear.post_attention_layernorm,
        full.input_layernorm,
        full.post_attention_layernorm,
        full.self_attn.q_norm,
        full.self_attn.k_norm,
        backbone.norm,
    ):
        assert type(norm) is Qwen3_5FusedRMSNorm
    assert type(linear.linear_attn.norm) is Qwen3_5FusedRMSNormGated
    # Checkpoint keys stay those of the replaced modules.
    assert {
        "layers.0.input_layernorm.weight",
        "layers.0.linear_attn.norm.weight",
        "layers.1.self_attn.q_norm.weight",
        "layers.1.self_attn.k_norm.weight",
        "norm.weight",
    } <= set(backbone.state_dict())


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused_cls,reference_cls,n_inputs", CASES)
def test_cpu_matches_replaced_module_exactly(fused_cls, reference_cls, n_inputs, weight_dtype):
    fused, reference = _modules(fused_cls, reference_cls, 64, "cpu", weight_dtype)
    inputs, grad_output = _inputs(n_inputs, (3, 5, 64), "cpu")

    for got, expected in zip(
        _forward_backward(fused, inputs, grad_output), _forward_backward(reference, inputs, grad_output)
    ):
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernels run on CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused_cls,reference_cls,n_inputs", CASES)
def test_cuda_fused_kernel_matches_replaced_module(fused_cls, reference_cls, n_inputs, weight_dtype):
    hidden = 256
    fused, reference = _modules(fused_cls, reference_cls, hidden, "cuda", weight_dtype)
    # An odd, non-power-of-two row count exercises the dynamic-shape kernel.
    inputs, grad_output = _inputs(n_inputs, (2, 37, hidden), "cuda")
    compiled_name = "_zero_centered_rms_norm_compiled" if n_inputs == 1 else "_rms_norm_gated_silu_compiled"
    # The gated norm keeps eager when its weight is wider than the activations.
    expect_fused = n_inputs == 1 or weight_dtype == torch.bfloat16

    with mock.patch.object(layers, compiled_name, wraps=getattr(layers, compiled_name)) as compiled:
        got = _forward_backward(fused, inputs, grad_output)
    assert compiled.called == expect_fused
    expected = _forward_backward(reference, inputs, grad_output)
    for name, a, b in zip(("output", *(f"input{i}_grad" for i in range(n_inputs)), "weight_grad"), got, expected):
        assert a.dtype == b.dtype, name
        if a.dtype == torch.float32:
            # The fp32 weight gradient sums 74 rows; only the summation order differs from eager.
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5, msg=name)
        else:
            torch.testing.assert_close(a, b, rtol=torch.finfo(a.dtype).eps, atol=1e-6, msg=name)

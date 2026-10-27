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

"""Parity of the fused Qwen3.5 layers with the modules and ops they replace."""

import inspect
import math
from unittest import mock

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_modeling
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNormGated

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5 import layers
from nemo_automodel.components.models.qwen3_5.layers import (
    Qwen3_5FusedMLP,
    Qwen3_5FusedRMSNorm,
    Qwen3_5FusedRMSNormGated,
    Qwen3_5FusedSSMGate,
)
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5DenseTextBackbone
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import SSMGate
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm
from nemo_automodel.components.moe.layers import MLP

HIDDEN = 64


def _norm_pair(fused_cls, reference_cls, device, dtype):
    reference = reference_cls(HIDDEN, eps=1e-6).to(device=device, dtype=dtype)
    with torch.no_grad():
        reference.weight.normal_(0.0, 0.5)
    fused = fused_cls(HIDDEN, eps=1e-6).to(device=device, dtype=dtype)
    fused.load_state_dict(reference.state_dict())
    return fused, reference


def _mlp_pair(device, dtype):
    reference = MLP(HIDDEN, 2 * HIDDEN, "torch", dtype=dtype).to(device)
    with torch.no_grad():
        for p in reference.parameters():
            p.normal_(0.0, 0.1)
    fused = Qwen3_5FusedMLP(HIDDEN, 2 * HIDDEN, "torch", dtype=dtype).to(device)
    fused.load_state_dict(reference.state_dict())
    return fused, reference


def _gate_pair(device, dtype):
    reference = SSMGate(HIDDEN, dtype=dtype).to(device)
    with torch.no_grad():
        reference.A_log.copy_(torch.empty(HIDDEN).uniform_(0.1, 16).log())
        reference.dt_bias.normal_(1.0, 0.5)
    fused = Qwen3_5FusedSSMGate(HIDDEN, dtype=dtype).to(device)
    fused.load_state_dict(reference.state_dict())
    return fused, reference


def _case(name, device, dtype):
    """Build ``(fused_fn, reference_fn, fused_params, reference_params, input_dtypes, compiled_name)``."""
    torch.manual_seed(0)
    bf16 = torch.bfloat16
    if name == "rms_norm":
        fused, ref = _norm_pair(Qwen3_5FusedRMSNorm, Qwen3NextRMSNorm, device, dtype)
        fused_fn, ref_fn, inputs, compiled = fused, ref, [bf16], "_zero_centered_rms_norm_compiled"
    elif name == "add_rms_norm":
        fused, ref = _norm_pair(Qwen3_5FusedRMSNorm, Qwen3NextRMSNorm, device, dtype)

        def fused_fn(x, residual):
            return fused(x, residual=residual)

        def ref_fn(x, residual):
            summed = x + residual
            return summed, ref(summed)

        inputs, compiled = [bf16, bf16], "_add_zero_centered_rms_norm_compiled"
    elif name == "rms_norm_gated":
        fused, ref = _norm_pair(Qwen3_5FusedRMSNormGated, Qwen3_5MoeRMSNormGated, device, dtype)
        fused_fn, ref_fn, inputs, compiled = fused, ref, [bf16, bf16], "_rms_norm_gated_silu_compiled"
    elif name == "mlp":
        # Linear layers need activations in the weight dtype.
        fused, ref = _mlp_pair(device, dtype)
        fused_fn, ref_fn, inputs, compiled = fused, ref, [dtype], "_silu_mul_compiled"
    else:
        fused, ref = _gate_pair(device, dtype)
        fused_fn, ref_fn, inputs, compiled = fused, ref, [bf16], "_ssm_gate_compiled"
    return fused_fn, ref_fn, list(fused.parameters()), list(ref.parameters()), inputs, compiled


def _forward_backward(fn, inputs, params):
    """Run forward and backward with fixed random upstream gradients.

    Args:
        fn: Callable under test returning a tensor or a tuple of tensors.
        inputs: Input tensors, each of shape [..., features].
        params: Parameters whose gradients are returned.

    Returns:
        Outputs, input gradients, then parameter gradients.
    """
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    outputs = fn(*leaves)
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    generator = torch.Generator(device=outputs[0].device).manual_seed(2)
    grads = [torch.randn(o.shape, dtype=o.dtype, device=o.device, generator=generator) for o in outputs]
    torch.autograd.backward(outputs, grads)
    return [o.detach() for o in outputs] + [t.grad for t in leaves] + [p.grad for p in params]


def _inputs(dtypes, shape, device):
    generator = torch.Generator(device=device).manual_seed(1)
    return [torch.randn(shape, dtype=dtype, device=device, generator=generator) for dtype in dtypes]


CASES = ["rms_norm", "add_rms_norm", "rms_norm_gated", "mlp", "ssm_gate"]


def test_backbone_uses_fused_layers_with_unchanged_parameter_names():
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
    assert type(linear.mlp) is Qwen3_5FusedMLP and type(full.mlp) is Qwen3_5FusedMLP
    gate = linear.linear_attn._fp32_params
    assert type(gate) is Qwen3_5FusedSSMGate
    # The gate holds the GatedDeltaNet's initialized fp32 parameters, not fresh storage.
    assert gate.A_log.dtype == torch.float32
    assert torch.equal(gate.dt_bias, torch.ones_like(gate.dt_bias))
    assert torch.all(gate.A_log <= math.log(16))
    # Checkpoint keys stay those of the replaced modules.
    assert {
        "layers.0.input_layernorm.weight",
        "layers.0.linear_attn.norm.weight",
        "layers.0.linear_attn._fp32_params.A_log",
        "layers.0.linear_attn._fp32_params.dt_bias",
        "layers.0.mlp.gate_proj.weight",
        "layers.1.self_attn.q_norm.weight",
        "layers.1.mlp.down_proj.weight",
        "norm.weight",
    } <= set(backbone.state_dict())


def test_backbone_forward_backward_is_finite():
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
    backbone = Qwen3_5DenseTextBackbone(config, backend).float().train()
    linear_attn = backbone.layers["0"].linear_attn
    # Run the real CPU GatedDeltaNet math even when FLA is installed.
    linear_attn.causal_conv1d_fn = None
    linear_attn.chunk_gated_delta_rule = inspect.unwrap(hf_modeling.torch_chunk_gated_delta_rule)

    hidden = backbone(input_ids=torch.tensor([[1, 2, 3, 4, 5]])).last_hidden_state
    hidden.backward(torch.randn_like(hidden))

    assert hidden.shape == (1, 5, 16) and torch.isfinite(hidden).all()
    for name, param in backbone.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", CASES)
def test_cpu_matches_replaced_computation_exactly(name, dtype):
    fused_fn, ref_fn, fused_params, ref_params, input_dtypes, _ = _case(name, "cpu", dtype)
    inputs = _inputs(input_dtypes, (3, 5, HIDDEN), "cpu")

    for got, expected in zip(
        _forward_backward(fused_fn, inputs, fused_params), _forward_backward(ref_fn, inputs, ref_params)
    ):
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernels run on CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", CASES)
def test_cuda_fused_kernel_matches_replaced_computation(name, dtype):
    fused_fn, ref_fn, fused_params, ref_params, input_dtypes, compiled_name = _case(name, "cuda", dtype)
    # An odd, non-power-of-two row count exercises the dynamic-shape kernel.
    inputs = _inputs(input_dtypes, (2, 37, HIDDEN), "cuda")
    # The gated norm keeps eager when its weight is wider than the activations.
    expect_fused = not (name == "rms_norm_gated" and dtype == torch.float32)

    with mock.patch.object(layers, compiled_name, wraps=getattr(layers, compiled_name)) as compiled:
        got = _forward_backward(fused_fn, inputs, fused_params)
    assert compiled.called == expect_fused
    expected = _forward_backward(ref_fn, inputs, ref_params)
    for index, (a, b) in enumerate(zip(got, expected)):
        assert a.dtype == b.dtype, index
        if a.dtype == torch.float32:
            # fp32 results differ from eager only in reduction order and transcendental rounding.
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5, msg=f"{name} tensor {index}")
        else:
            torch.testing.assert_close(a, b, rtol=torch.finfo(a.dtype).eps, atol=1e-6, msg=f"{name} tensor {index}")

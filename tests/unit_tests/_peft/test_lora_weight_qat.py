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

"""CPU additive-LoRA routing and merged-weight QAT integration tests."""

import copy
from typing import Literal
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DeviceMesh, Replicate, distribute_tensor

from nemo_automodel.components._peft.lora import LinearLoRA, TritonLinearLoRA, patch_linear_module
from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components._peft.lora_mlp import _fusible, fused_lora_swiglu_mlp
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.quantization.weight_qat import WeightFakeQuantizer, WeightQuantizationConfig


@pytest.fixture(autouse=True)
def _eager_cpu():
    # Exercise the real activation/autograd implementations without CPU compilation.
    with torch.compiler.set_stance("force_eager"):
        yield


@pytest.fixture(params=["fp8", "mxfp4"])
def quant_config(request):
    return WeightQuantizationConfig(format=request.param, block_size=(32, 32) if request.param == "fp8" else (1, 32))


def _reference_qdq(weight: torch.Tensor, quantizer: WeightFakeQuantizer) -> torch.Tensor:
    """Use encoded bytes plus an explicitly constructed identity-gradient STE.

    Args:
        weight: Local FP32 tensor [..., out, in] in canonical order.
        quantizer: Real quantizer; only its non-differentiable encoding API is used.

    Returns:
        Tensor [..., out, in] with decoded values and identity weight gradient.
    """
    decoded = quantizer.quantize(weight.detach()).dequantize(dtype=weight.dtype)
    return decoded + (weight - weight.detach())


@pytest.mark.parametrize("implementation", ["linear", "triton", "patched_triton"])
@pytest.mark.parametrize("bias", [False, True])
def test_dense_effective_qdq_forward_gradients_and_step(quant_config, implementation, bias):
    torch.manual_seed(29)
    base = nn.Linear(32, 64, bias=bias)
    if implementation == "patched_triton":
        layer = patch_linear_module(base, dim=3, alpha=7, use_triton=True)
    else:
        cls = TritonLinearLoRA if implementation == "triton" else LinearLoRA
        layer = cls(base, dim=3, alpha=7)
    assert layer.weight_fake_quantizer is None
    with torch.no_grad():
        layer.lora_A.weight.normal_(std=0.15)
        layer.lora_B.weight.normal_(std=0.15)
    layer.weight_fake_quantizer = quant_config.build()
    assert dict(layer.named_children())["weight_fake_quantizer"] is layer.weight_fake_quantizer
    frozen_weight = layer.weight.detach().clone()
    frozen_bias = layer.bias.detach().clone() if bias else None
    a = layer.lora_A.weight.detach().clone().requires_grad_()
    b = layer.lora_B.weight.detach().clone().requires_grad_()
    x = torch.randn(2, 3, 32, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    effective = frozen_weight + layer.scale * (b @ a)
    expected = F.linear(ref_x, _reference_qdq(effective, quant_config.build()), frozen_bias)
    # A nonzero adapter must be inside QDQ, not added to an already quantized base.
    wrong = F.linear(ref_x, quant_config.build()(frozen_weight), frozen_bias)
    wrong = wrong + F.linear(F.linear(ref_x, a), b) * layer.scale
    assert not torch.allclose(expected, wrong, atol=1e-6, rtol=1e-6)
    with patch("nemo_automodel.components._peft.lora.apply_memory_efficient_lora", side_effect=AssertionError):
        actual = layer(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=1e-6, rtol=1e-5)
    for param, ref in ((layer.lora_A.weight, a), (layer.lora_B.weight, b)):
        torch.testing.assert_close(param.grad, ref.grad, atol=1e-5, rtol=1e-5)
        assert torch.count_nonzero(param.grad) > 0
    assert layer.weight.grad is None and not layer.weight.requires_grad
    if bias:
        assert layer.bias.grad is None and not layer.bias.requires_grad
    before = layer.lora_A.weight.detach().clone()
    torch.optim.SGD(layer.parameters(), lr=0.01).step()
    assert not torch.equal(before, layer.lora_A.weight)
    torch.testing.assert_close(layer.lora_A.weight, a.detach() - 0.01 * a.grad)
    torch.testing.assert_close(layer.lora_B.weight, b.detach() - 0.01 * b.grad)
    assert torch.equal(layer.weight, frozen_weight)
    if bias:
        assert torch.equal(layer.bias, frozen_bias)


@pytest.mark.parametrize("cls", [LinearLoRA, TritonLinearLoRA])
@pytest.mark.parametrize("position", ["pre", "post"])
def test_dense_dropout_guard_and_eval(quant_config, cls, position):
    layer = cls(nn.Linear(32, 32), dropout=0.2, dropout_position=position)
    layer.weight_fake_quantizer = quant_config.build()
    x = torch.randn(2, 32)
    with pytest.raises(RuntimeError, match="training dropout"):
        layer(x)
    layer.eval()
    expected = F.linear(x, _reference_qdq(layer.materialize_effective_weight(), quant_config.build()), layer.bias)
    torch.testing.assert_close(layer(x), expected)


@pytest.mark.parametrize("cls", [LinearLoRA, TritonLinearLoRA])
def test_dense_dora_and_delegation_guards(quant_config, cls):
    layer = cls(nn.Linear(32, 32), use_dora=True)
    layer.weight_fake_quantizer = quant_config.build()
    x = torch.randn(2, 32)
    with pytest.raises(NotImplementedError, match="DoRA"):
        layer(x)
    layer.use_dora = False
    layer.super_fwd = Mock(side_effect=AssertionError("delegated GEMM must not run"))
    with pytest.raises(NotImplementedError, match="delegated"):
        layer(x)
    layer.super_fwd.assert_not_called()


@pytest.fixture
def cpu_mesh(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        yield DeviceMesh("cpu", [0])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("target", ["input", "weight", "bias", "lora_A", "lora_B"])
@pytest.mark.parametrize("cls", [LinearLoRA, TritonLinearLoRA])
def test_dense_rejects_dtensor_before_computation(cpu_mesh, target, cls):
    layer = cls(nn.Linear(32, 32))
    layer.weight_fake_quantizer = WeightQuantizationConfig(format="fp8", block_size=(32, 32)).build()
    x = torch.randn(2, 32)
    if target == "input":
        x = distribute_tensor(x, cpu_mesh, [Replicate()])
    else:
        owner = getattr(layer, target) if target.startswith("lora_") else layer
        name = "weight" if target.startswith("lora_") else target
        old = getattr(owner, name)
        setattr(owner, name, nn.Parameter(distribute_tensor(old.detach(), cpu_mesh, [Replicate()]), old.requires_grad))
    with pytest.raises(NotImplementedError, match="DTensor"):
        layer(x)


@pytest.mark.parametrize("projection", ["gate", "up", "down"])
def test_qat_excludes_fused_mlp(quant_config, projection):
    layers = {name: LinearLoRA(nn.Linear(32, 32, bias=False)) for name in ("gate", "up", "down")}
    assert all(_fusible(layer) for layer in layers.values())
    layers[projection].weight_fake_quantizer = quant_config.build()
    assert not _fusible(layers[projection])
    assert fused_lora_swiglu_mlp(layers["gate"], layers["up"], layers["down"], torch.randn(2, 32)) is None
    layers[projection].weight_fake_quantizer = None
    assert _fusible(layers[projection])


def _experts_config(
    *,
    expert_bias: bool = False,
    activation: Literal["swiglu", "relu2"] = "swiglu",
    after_down: bool = False,
) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=32,
        inter_dim=64,
        moe_inter_dim=64,
        norm_topk_prob=False,
        expert_bias=expert_bias,
        expert_activation=activation,
        apply_router_weight_after_down=after_down,
        dtype=torch.float32,
    )


def _make_experts(
    *, expert_bias: bool = False, activation: Literal["swiglu", "relu2"] = "swiglu", after_down: bool = False
) -> GroupedExpertsLoRA:
    base = GroupedExperts(_experts_config(expert_bias=expert_bias, activation=activation, after_down=after_down))
    with torch.no_grad():
        for param in base.parameters():
            param.normal_(std=0.08)
    layer = GroupedExpertsLoRA(base, lora_dim=3, alpha=7)
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if name.startswith("lora_"):
                param.normal_(std=0.08)
    assert not layer.use_torch_mm
    assert layer.weight_fake_quantizer is None
    return layer


def _reference_experts(
    layer: GroupedExpertsLoRA | GroupedExpertsDeepEPLoRA,
    x: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    quantizer: WeightFakeQuantizer,
) -> torch.Tensor:
    """Independent token-by-token reference, using F.linear in canonical order.

    Args:
        layer: Reference module containing expert weights [experts, in, out],
            adapters A [experts, in, rank] and B [experts, rank, out].
        x: FP32 CPU tensor [tokens, hidden].
        mask: Boolean tensor [tokens].
        weights: Routing tensor [tokens, top_k].
        indices: Integer tensor [tokens, top_k].
        quantizer: Real encoder/decoder used to build a manual STE reference.

    Returns:
        FP32 tensor [tokens, hidden], with gradients to input, routing and adapters.
    """
    up = layer.gate_and_up_projs + layer.scale * (layer.lora_gate_and_up_A @ layer.lora_gate_and_up_B)
    down = layer.down_projs + layer.scale * (layer.lora_down_A @ layer.lora_down_B)
    up = _reference_qdq(up.transpose(-2, -1), quantizer)
    down = _reference_qdq(down.transpose(-2, -1), quantizer)
    rows = []
    for token in range(x.size(0)):
        row = x[token] * 0
        if mask[token]:
            for slot in range(indices.size(1)):
                expert = int(indices[token, slot])
                h = F.linear(x[token], up[expert])
                if layer.expert_bias:
                    h = h + layer.gate_up_proj_bias[expert]
                if layer.is_gated:
                    gate, value = h.chunk(2)
                    h = F.silu(gate) * value
                else:
                    h = F.relu(h).square()
                probability = weights[token, slot]
                result = F.linear(h * probability, down[expert])
                if layer.expert_bias:
                    result = result + layer.down_proj_bias[expert] * probability
                row = row + result
        rows.append(row)
    return torch.stack(rows)


@pytest.mark.parametrize("expert_bias", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "relu2"])
def test_expert_loop_qdq_forward_gradients_and_step(quant_config, expert_bias, activation):
    torch.manual_seed(41)
    layer = _make_experts(expert_bias=expert_bias, activation=activation)
    reference = copy.deepcopy(layer)
    frozen = {name: param.detach().clone() for name, param in layer.named_parameters() if not param.requires_grad}
    layer.weight_fake_quantizer = quant_config.build()
    x = torch.randn(4, 32, requires_grad=True)
    weights = torch.rand(4, 2, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    mask = torch.tensor([True, True, False, True])
    expected = _reference_experts(reference, ref_x, mask, ref_weights, indices, quant_config.build())
    actual = layer(x, mask, weights, indices)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(weights.grad, ref_weights.grad, atol=2e-6, rtol=2e-5)
    refs = dict(reference.named_parameters())
    for name, param in layer.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, refs[name].grad, atol=2e-6, rtol=2e-5)
            assert torch.count_nonzero(param.grad) > 0
        else:
            assert param.grad is None
    before = layer.lora_gate_and_up_A.detach().clone()
    torch.optim.SGD(layer.parameters(), lr=0.03).step()
    torch.optim.SGD(reference.parameters(), lr=0.03).step()
    assert not torch.equal(layer.lora_gate_and_up_A, before)
    for name, param in layer.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param, refs[name], atol=2e-6, rtol=2e-5)
        else:
            assert torch.equal(param, frozen[name])


@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("grouped_mm", [False, True])
@pytest.mark.parametrize("after_down", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("qat", [False, True])
def test_expert_empty_or_masked_gradients(quant_config, tokens, grouped_mm, after_down, masked, qat):
    layer = _make_experts(after_down=after_down)
    layer.use_torch_mm = grouped_mm
    layer.weight_fake_quantizer = quant_config.build() if qat else None
    x = torch.randn(tokens, 32, requires_grad=True)
    weights = torch.rand(tokens, 2, requires_grad=True)
    mask = torch.full((tokens,), not masked, dtype=torch.bool)
    indices = torch.full((tokens, 2), -1, dtype=torch.long)
    if masked:
        indices.zero_()
    out = layer(x, mask, weights, indices)
    assert out.shape == x.shape and torch.count_nonzero(out) == 0
    out.sum().backward()
    for tensor in (x, weights, *(p for p in layer.parameters() if p.requires_grad)):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0
    assert layer.gate_and_up_projs.grad is None
    assert layer.down_projs.grad is None


def _cpu_grouped_mm(x: torch.Tensor, operands: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    """Differentiable CPU stand-in for GEMM plumbing, not a CUDA kernel test.

    Args:
        x: CPU tensor [routed_tokens, in], grouped by expert.
        operands: Contiguous CPU tensor [local_experts, in, out].
        offs: Integer tensor [local_experts] of cumulative token counts.

    Returns:
        CPU tensor [routed_tokens, out], in the same expert-grouped order.
    """
    assert operands.is_contiguous()
    bounds = [0, *offs.tolist()]
    return torch.cat([x[bounds[i] : bounds[i + 1]] @ operand for i, operand in enumerate(operands)])


def _cpu_gmm(x: torch.Tensor, operands: torch.Tensor, counts: torch.Tensor, *, trans_b: bool) -> torch.Tensor:
    """CPU stand-in for grouped_gemm's untransposed operand interface.

    Args:
        x: CPU tensor [routed_tokens, in], grouped by expert.
        operands: Contiguous CPU tensor [local_experts, in, out].
        counts: Integer tensor [local_experts], tokens per expert.
        trans_b: Must be False for the expert operand contract.

    Returns:
        CPU tensor [routed_tokens, out] retaining input and weight gradients.
    """
    assert trans_b is False
    return _cpu_grouped_mm(x, operands, offs=counts.cumsum(0))


@pytest.mark.parametrize("expert_bias", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "relu2"])
@pytest.mark.parametrize("grouped_mm", [False, True])
@pytest.mark.parametrize("after_down", [False, True])
def test_expert_qat_exact_loaded_parent_outputs_and_gradients(
    quant_config, expert_bias, activation, grouped_mm, after_down
):
    torch.manual_seed(67)
    layer = _make_experts(expert_bias=expert_bias, activation=activation, after_down=after_down)
    layer.use_torch_mm = grouped_mm
    # Bias storage may differ: the parent casts biases to the activation dtype.
    if expert_bias:
        layer.gate_up_proj_bias = nn.Parameter(layer.gate_up_proj_bias.double(), requires_grad=False)
        layer.down_proj_bias = nn.Parameter(layer.down_proj_bias.double(), requires_grad=False)
    reference = copy.deepcopy(layer)
    parent = GroupedExperts(copy.deepcopy(layer.config))
    parent.use_torch_mm = grouped_mm
    effective = {}
    for name, a, b in (
        ("gate_and_up_projs", reference.lora_gate_and_up_A, reference.lora_gate_and_up_B),
        ("down_projs", reference.lora_down_A, reference.lora_down_B),
    ):
        merged = getattr(reference, name) + reference.scale * (a @ b)
        effective[name] = _reference_qdq(merged.transpose(-2, -1), quant_config.build()).transpose(-2, -1).contiguous()
    state = {name: value.detach() for name, value in effective.items()}
    if expert_bias:
        state.update(gate_up_proj_bias=layer.gate_up_proj_bias, down_proj_bias=layer.down_proj_bias)
    parent.load_state_dict(state, strict=True)
    layer.weight_fake_quantizer = quant_config.build()
    x = torch.randn(5, 32, requires_grad=True)
    weights = torch.rand(5, 2, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    indices = torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]])
    mask = torch.tensor([True, True, False, True, True])
    with patch.object(torch, "_grouped_mm", side_effect=_cpu_grouped_mm, create=True):
        expected = parent(ref_x, mask, ref_weights, indices)
        # QAT must bypass the overridden additive-LoRA kernels, not call module.forward.
        with (
            patch.object(layer, "_forward_loop", side_effect=AssertionError("legacy LoRA kernel")),
            patch.object(layer, "_forward_grouped_mm", side_effect=AssertionError("legacy LoRA kernel")),
        ):
            actual = layer(x, mask, weights, indices)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.autograd.backward(tuple(effective.values()), (parent.gate_and_up_projs.grad, parent.down_projs.grad))
    torch.testing.assert_close(x.grad, ref_x.grad, atol=0, rtol=0)
    torch.testing.assert_close(weights.grad, ref_weights.grad, atol=0, rtol=0)
    for name, param in layer.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, reference.get_parameter(name).grad, atol=0, rtol=0)
            assert torch.count_nonzero(param.grad) > 0
        else:
            assert param.grad is None


@pytest.mark.parametrize("deep_ep", [False, True])
@pytest.mark.parametrize(
    "target",
    [
        "input",
        "gate_and_up_projs",
        "down_projs",
        "lora_gate_and_up_A",
        "lora_gate_and_up_B",
        "lora_down_A",
        "lora_down_B",
    ],
)
def test_expert_qat_rejects_mixed_dtype_before_conversion(deep_ep, target):
    layer = GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP(_experts_config())) if deep_ep else _make_experts()
    layer.use_torch_mm = True
    layer.token_dispatcher = Mock()
    layer.weight_fake_quantizer = Mock(spec=nn.Module)
    x = torch.randn(2, 32)
    if target == "input":
        x = x.bfloat16()
    else:
        old = layer.get_parameter(target)
        setattr(layer, target, nn.Parameter(old.bfloat16(), requires_grad=old.requires_grad))
    with (
        patch("nemo_automodel.components._peft.lora_experts._to_grouped_mm_operand", side_effect=AssertionError),
        pytest.raises(ValueError, match="dtype"),
    ):
        layer(x, torch.ones(2, dtype=torch.bool), torch.rand(2, 2), torch.zeros(2, 2, dtype=torch.long))
    layer.weight_fake_quantizer.assert_not_called()
    layer.token_dispatcher.token_permutation2.assert_not_called()


@pytest.mark.parametrize("deep_ep", [False, True])
@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_expert_qat_rejects_autocast_before_dispatch(deep_ep, device_type):
    layer = GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP(_experts_config())) if deep_ep else _make_experts()
    layer.use_torch_mm = True
    layer.token_dispatcher = Mock()
    layer.weight_fake_quantizer = Mock(spec=nn.Module)
    # CUDA availability is mocked only to enable the autocast state on CPU CI;
    # the guard must run before any CUDA allocation or native backend call.
    with patch("torch.cuda.is_available", return_value=True), torch.autocast(device_type):
        assert torch.is_autocast_enabled(device_type)
        with pytest.raises(ValueError, match="autocast"):
            layer(
                torch.randn(2, 32),
                torch.ones(2, dtype=torch.bool),
                torch.rand(2, 2),
                torch.zeros(2, 2, dtype=torch.long),
            )
    layer.weight_fake_quantizer.assert_not_called()
    layer.token_dispatcher.token_permutation2.assert_not_called()


@pytest.mark.parametrize("backend", ["grouped_mm", "deepep_torch_mm", "deepep_gmm"])
def test_grouped_gemm_qat_operand_plumbing(quant_config, backend):
    """CPU GEMM/dispatcher stand-ins test math and call counts, not distributed execution."""
    torch.manual_seed(59)
    source = _make_experts()
    reference = copy.deepcopy(source)
    if backend == "grouped_mm":
        layer = source
    else:
        layer = GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP(_experts_config()), lora_dim=3, alpha=7)
        layer.load_state_dict(source.state_dict())
    layer.use_torch_mm = backend != "deepep_gmm"
    layer.weight_fake_quantizer = quant_config.build()
    x = torch.randn(3, 32, requires_grad=True)
    weights = torch.rand(3, 1, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    indices = torch.tensor([[0], [0], [1]])
    mask = torch.ones(3, dtype=torch.bool)
    if backend != "grouped_mm":
        layer.token_dispatcher = Mock()
        layer.token_dispatcher.token_permutation2.return_value = (x, torch.tensor([2, 1]), weights[:, 0])
        layer.token_dispatcher.token_unpermutation = nn.Identity()
    expected = _reference_experts(reference, ref_x, mask, ref_weights, indices, quant_config.build())
    with (
        patch.object(torch, "_grouped_mm", side_effect=_cpu_grouped_mm, create=True) as torch_mm,
        patch("nemo_automodel.components._peft.lora_experts.ops") as ops,
    ):
        ops.gmm.side_effect = _cpu_gmm
        actual = layer(x, mask, weights, indices)
        gemm = ops.gmm if backend == "deepep_gmm" else torch_mm
        assert gemm.call_count == 2  # Only effective gate/up and down GEMMs; no additive adapter GEMMs.
        assert gemm.call_args_list[0].args[1].shape == (2, 32, 128)
        assert gemm.call_args_list[1].args[1].shape == (2, 64, 32)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(weights.grad, ref_weights.grad, atol=2e-6, rtol=2e-5)
    ref_params = dict(reference.named_parameters())
    for name, param in layer.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, ref_params[name].grad, atol=2e-6, rtol=2e-5)
        else:
            assert param.grad is None


@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("after_down", [False, True])
@pytest.mark.parametrize("qat", [False, True])
@pytest.mark.parametrize("grouped_mm", [False, True])
def test_deepep_empty_dispatch_gradients(quant_config, tokens, after_down, qat, grouped_mm):
    source = _make_experts(after_down=after_down)
    layer = GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP(_experts_config(after_down=after_down)), lora_dim=3, alpha=7)
    layer.load_state_dict(source.state_dict())
    layer.use_torch_mm = grouped_mm
    layer.weight_fake_quantizer = quant_config.build() if qat else None
    x = torch.randn(tokens, 32, requires_grad=True)
    weights = torch.rand(tokens, 1, requires_grad=True)
    layer.token_dispatcher = Mock()
    layer.token_dispatcher.token_permutation2.return_value = (x[:0], torch.zeros(2, dtype=torch.long), weights[:0, 0])
    layer.token_dispatcher.token_unpermutation = nn.Identity()
    # Inspect the local empty output passed to combine; this stand-in performs no communication.
    with patch("nemo_automodel.components._peft.lora_experts.ops"):
        out = layer(x, torch.zeros(tokens, dtype=torch.bool), weights, torch.zeros(tokens, 1, dtype=torch.long))
    assert out.shape == (0, 32)
    out.sum().backward()
    for tensor in (x, weights, *(p for p in layer.parameters() if p.requires_grad)):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) == 0


@pytest.mark.parametrize("expert_bias", [False, True])
@pytest.mark.parametrize("grouped_mm", [False, True])
@pytest.mark.parametrize("after_down", [False, True])
def test_deepep_qat_exact_parent_outputs_and_gradients(quant_config, cpu_mesh, expert_bias, grouped_mm, after_down):
    """Real parent math with CPU dispatcher/GEMM stand-ins, not native DeepEP validation."""
    torch.manual_seed(71)
    source = _make_experts(expert_bias=expert_bias, after_down=after_down)
    config = copy.deepcopy(source.config)
    layer = GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP(config), lora_dim=3, alpha=7)
    layer.load_state_dict(source.state_dict())
    parent = GroupedExpertsDeepEP(copy.deepcopy(config))
    effective = {}
    for name, a, b in (
        ("gate_and_up_projs", source.lora_gate_and_up_A, source.lora_gate_and_up_B),
        ("down_projs", source.lora_down_A, source.lora_down_B),
    ):
        merged = source.get_parameter(name) + source.scale * (a @ b)
        effective[name] = _reference_qdq(merged.transpose(-2, -1), quant_config.build()).transpose(-2, -1).contiguous()
    state = {name: value.detach() for name, value in effective.items()}
    if expert_bias:
        state.update(gate_up_proj_bias=source.gate_up_proj_bias, down_proj_bias=source.down_proj_bias)
    parent.load_state_dict(state, strict=True)
    # The parent requires DTensor storage. A real one-rank CPU mesh supplies
    # that contract; dispatch and grouped kernels below remain explicit stand-ins.
    for name, param in tuple(parent.named_parameters()):
        value = param.detach()
        if "bias" in name:
            value = value.double()
            setattr(layer, name, nn.Parameter(value.clone(), requires_grad=False))
        setattr(parent, name, nn.Parameter(distribute_tensor(value, cpu_mesh, [Replicate()]), "bias" not in name))
    parent.n_routed_experts = config.n_routed_experts
    parent.ep_size = 1
    parent.use_torch_mm = layer.use_torch_mm = grouped_mm
    layer.weight_fake_quantizer = quant_config.build()
    x = torch.randn(5, 32, requires_grad=True)
    weights = torch.rand(5, 1, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    indices = torch.tensor([[0], [0], [0], [1], [1]])
    mask = torch.ones(5, dtype=torch.bool)
    for module, hidden, probs in ((layer, x, weights), (parent, ref_x, ref_weights)):
        module.token_dispatcher = Mock()
        module.token_dispatcher.token_permutation2.return_value = (hidden, torch.tensor([3, 2]), probs[:, 0])
        module.token_dispatcher.token_unpermutation = nn.Identity()
    with (
        patch.object(torch, "_grouped_mm", side_effect=_cpu_grouped_mm, create=True),
        patch("nemo_automodel.components._peft.lora_experts.ops") as lora_ops,
        patch("nemo_automodel.components.moe.experts.ops", create=True) as parent_ops,
    ):
        lora_ops.gmm.side_effect = parent_ops.gmm.side_effect = _cpu_gmm
        actual = layer(x, mask, weights, indices)
        expected = parent(ref_x, mask, ref_weights, indices)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.autograd.backward(
        tuple(effective.values()), (parent.gate_and_up_projs.grad.to_local(), parent.down_projs.grad.to_local())
    )
    torch.testing.assert_close(x.grad, ref_x.grad, atol=0, rtol=0)
    torch.testing.assert_close(weights.grad, ref_weights.grad, atol=0, rtol=0)
    for name, param in layer.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param.grad, source.get_parameter(name).grad, atol=0, rtol=0)
            assert torch.count_nonzero(param.grad) > 0
        else:
            assert param.grad is None


@pytest.mark.parametrize("after_down", [False, True])
def test_deepep_qat_missing_backend_fails_before_dispatch(after_down):
    layer = GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP(_experts_config(after_down=after_down)))
    layer.weight_fake_quantizer = Mock(spec=nn.Module)
    layer.token_dispatcher = Mock()
    assert not layer.use_torch_mm
    with (
        patch("nemo_automodel.components._peft.lora_experts.ops", None),
        pytest.raises(RuntimeError, match="requires grouped_gemm or use_torch_mm=True"),
    ):
        layer(
            torch.randn(2, 32), torch.ones(2, dtype=torch.bool), torch.rand(2, 2), torch.zeros(2, 2, dtype=torch.long)
        )
    layer.token_dispatcher.token_permutation2.assert_not_called()
    layer.weight_fake_quantizer.assert_not_called()


def test_deepep_qat_rejects_dispatch_dtype_change():
    layer = GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP(_experts_config()))
    layer.use_torch_mm = True
    layer.weight_fake_quantizer = Mock(spec=nn.Module)
    layer.token_dispatcher = Mock()
    layer.token_dispatcher.token_permutation2.return_value = (
        torch.randn(2, 32, dtype=torch.bfloat16),
        torch.tensor([1, 1]),
        torch.rand(2),
    )
    with pytest.raises(ValueError, match="dispatch to preserve activation dtype"):
        layer(
            torch.randn(2, 32), torch.ones(2, dtype=torch.bool), torch.rand(2, 2), torch.zeros(2, 2, dtype=torch.long)
        )
    layer.weight_fake_quantizer.assert_not_called()
    layer.token_dispatcher.token_unpermutation.assert_not_called()


@pytest.mark.parametrize(
    "cls,base_cls", [(GroupedExpertsLoRA, GroupedExperts), (GroupedExpertsDeepEPLoRA, GroupedExpertsDeepEP)]
)
def test_expert_constructor_preserves_source_dtype(cls, base_cls):
    base = base_cls(_experts_config(expert_bias=True)).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.normal_()
    layer = cls(base, lora_dim=3)
    assert layer.weight_fake_quantizer is None
    for name, param in base.named_parameters():
        actual = dict(layer.named_parameters())[name]
        assert actual.dtype == param.dtype
        assert torch.equal(actual, param)
        assert not actual.requires_grad


@pytest.mark.parametrize("deep_ep", [False, True])
def test_experts_reject_mxfp8_before_dispatch_or_quantization(deep_ep):
    if deep_ep:
        base = GroupedExpertsDeepEP(_experts_config())
        base.use_mxfp8 = True
        layer = GroupedExpertsDeepEPLoRA(base)
        layer.token_dispatcher = Mock()
    else:
        base = GroupedExperts(_experts_config())
        base.use_mxfp8 = True
        layer = GroupedExpertsLoRA(base)
    layer.weight_fake_quantizer = WeightQuantizationConfig(format="mxfp4").build()
    with pytest.raises(NotImplementedError, match="MXFP8"):
        layer(
            torch.empty(0, 32), torch.empty(0, dtype=torch.bool), torch.empty(0, 2), torch.empty(0, 2, dtype=torch.long)
        )
    if deep_ep:
        layer.token_dispatcher.token_permutation2.assert_not_called()


@pytest.mark.parametrize("after_down", [False, True])
def test_detaching_quantizer_restores_original_forward(quant_config, after_down):
    torch.manual_seed(51)
    layer = _make_experts(after_down=after_down)
    x = torch.randn(3, 32)
    weights = torch.rand(3, 2)
    indices = torch.tensor([[0, 1], [1, 0], [0, 1]])
    mask = torch.ones(3, dtype=torch.bool)
    original = layer(x, mask, weights, indices)
    layer.weight_fake_quantizer = quant_config.build()
    quantized = layer(x, mask, weights, indices)
    assert not torch.allclose(original, quantized, atol=1e-6, rtol=1e-6)
    layer.weight_fake_quantizer = None
    torch.testing.assert_close(layer(x, mask, weights, indices), original, atol=0, rtol=0)


def _ordinary_expert_pair(backend, dtype, expert_bias, activation, after_down):
    config = _experts_config(expert_bias=expert_bias, activation=activation, after_down=after_down)
    config.n_routed_experts = 4
    config.n_activated_experts = 3
    config.swiglu_limit = 0.4  # Exercise FP32 clamping, not only small linear activations.
    config.dtype = dtype
    deep_ep = backend.startswith("deepep")
    parent = (GroupedExpertsDeepEP(config) if deep_ep else GroupedExperts(config)).to(dtype)
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.normal_(std=0.2)
    parent.use_torch_mm = backend in ("grouped_mm", "deepep_torch_mm")
    # Aligned rank avoids conflating router placement with padding-dependent GEMM rounding.
    layer = (GroupedExpertsDeepEPLoRA if deep_ep else GroupedExpertsLoRA)(parent, lora_dim=8, alpha=13)
    return layer, parent


def _ordinary_routing_inputs(dtype):
    x = torch.randn(7, 32, dtype=dtype, requires_grad=True)
    weights = (0.05 + 2.4 * torch.rand(7, 3)).requires_grad_()
    indices = torch.rand(7, 4).argsort(dim=1)[:, :3].contiguous()
    mask = torch.tensor([True, True, False, True, True, True, True])
    assert (weights > 1).any() and (weights < 1).any()
    assert (indices[:, 1:] < indices[:, :-1]).any()
    return x, mask, weights, indices


def _cpu_slot_dispatcher(n_experts):
    """CPU routing stand-in; no native dispatch, communication, or kernel claims."""
    slots = []
    shape = None

    def permute(*, hidden_states, num_local_tokens, token_probs, token_indices):
        """Group valid slots by expert using explicit enumeration.

        Args:
            hidden_states: CPU tensor [tokens, hidden].
            num_local_tokens: Number of local tokens.
            token_probs: FP32 CPU tensor [tokens, top_k].
            token_indices: Integer CPU tensor [tokens, top_k], with -1 for masked slots.

        Returns:
            Hidden states [routes, hidden], counts [experts], and probabilities
            [routes], all in expert order, retaining gradients to the inputs.
        """
        nonlocal slots, shape
        shape = (num_local_tokens, token_indices.size(1), hidden_states.size(1))
        slots = [
            (token, slot)
            for expert in range(n_experts)
            for token in range(num_local_tokens)
            for slot in range(token_indices.size(1))
            if token_indices[token, slot] == expert
        ]
        counts = torch.tensor([(token_indices == expert).sum() for expert in range(n_experts)])
        token_ids = torch.tensor([token for token, _ in slots], dtype=torch.long)
        slot_ids = torch.tensor([slot for _, slot in slots], dtype=torch.long)
        return hidden_states[token_ids], counts, token_probs[token_ids, slot_ids]

    def combine(outputs):
        """Restore slots before reducing, without simulating native combine precision.

        Args:
            outputs: CPU tensor [routes, hidden] in expert order and activation dtype.

        Returns:
            CPU tensor [tokens, hidden], summed over top-k in FP32 then cast back.
        """
        result = torch.zeros(shape, dtype=torch.float32)
        for route, (token, slot) in enumerate(slots):
            result[token, slot] = outputs[route].float()
        return result.sum(dim=1).to(outputs.dtype)

    return Mock(token_permutation2=permute, token_unpermutation=combine)


def _reference_additive_experts(layer, x, mask, weights, indices, *, grouped, deep_ep):
    """Explicit additive projections and activation math, never merged weights.

    Args:
        layer: Local expert module with base [experts, in, out], A [experts,
            in, rank], B [experts, rank, out], and optional biases [experts, out].
        x: CPU tensor [tokens, hidden] in FP32 or BF16.
        mask: Boolean CPU tensor [tokens].
        weights: FP32 CPU tensor [tokens, top_k], not necessarily normalized.
        indices: Integer CPU tensor [tokens, top_k] with unique experts per token.
        grouped: Whether to round bias addition to the GEMM output dtype.
        deep_ep: Whether to cast each routed output before dispatcher combine.

    Returns:
        CPU tensor [tokens, hidden] in x's dtype with additive adapter gradients.
    """
    slots = torch.zeros((*indices.shape, x.size(1)), dtype=torch.float32)
    legacy_sum = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(layer.config.n_routed_experts):
        routes = [
            (token, slot)
            for token in range(x.size(0))
            for slot in range(indices.size(1))
            if mask[token] and indices[token, slot] == expert
        ]
        if not routes:
            continue
        tokens, top = zip(*routes)
        tokens, top = list(tokens), list(top)
        hidden = x[tokens]
        h = F.linear(hidden, layer.gate_and_up_projs[expert].T)
        h = (
            h
            + F.linear(F.linear(hidden, layer.lora_gate_and_up_A[expert].T), layer.lora_gate_and_up_B[expert].T)
            * layer.scale
        )
        if layer.expert_bias:
            h = h + layer.gate_up_proj_bias[expert]
        if layer.is_gated:
            gate, up = h.float().chunk(2, dim=-1)
            h = F.silu(gate.clamp(max=layer.config.swiglu_limit)) * up.clamp(
                -layer.config.swiglu_limit, layer.config.swiglu_limit
            )
        else:
            h = F.relu(h).square()
        probability = weights[tokens, top, None]
        if not layer.config.apply_router_weight_after_down:
            h = h * probability
        h = h.to(x.dtype)
        result = F.linear(h, layer.down_projs[expert].T)
        result = result + F.linear(F.linear(h, layer.lora_down_A[expert].T), layer.lora_down_B[expert].T) * layer.scale
        if layer.expert_bias:
            bias = layer.down_proj_bias[expert]
            result = result + (bias if layer.config.apply_router_weight_after_down else bias * probability)
            if grouped:
                result = result.to(x.dtype)
        if layer.config.apply_router_weight_after_down:
            result = result.float() * probability
        if deep_ep:
            result = result.to(x.dtype)
        slots[tokens, top] = result.float()
        legacy_sum[tokens] = legacy_sum[tokens] + result.float()
    result = slots.sum(dim=1) if layer.config.apply_router_weight_after_down or deep_ep else legacy_sum
    return result.to(x.dtype)


@pytest.mark.parametrize("backend", ["loop", "grouped_mm", "deepep_torch_mm", "deepep_gmm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("expert_bias", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "relu2"])
@pytest.mark.parametrize("after_down", [False, True])
def test_ordinary_zero_b_exact_parent(cpu_mesh, backend, dtype, expert_bias, activation, after_down):
    torch.manual_seed(83)
    layer, parent = _ordinary_expert_pair(backend, dtype, expert_bias, activation, after_down)
    x, mask, weights, indices = _ordinary_routing_inputs(dtype)
    if backend.startswith("deepep"):
        for name, parameter in tuple(parent.named_parameters()):
            setattr(parent, name, nn.Parameter(distribute_tensor(parameter.detach(), cpu_mesh, [Replicate()])))
        parent.n_routed_experts = parent.config.n_routed_experts
        parent.ep_size = 1
        parent.token_dispatcher = _cpu_slot_dispatcher(4)
        layer.token_dispatcher = _cpu_slot_dispatcher(4)
    with (
        patch.object(torch, "_grouped_mm", side_effect=_cpu_grouped_mm, create=True),
        patch("nemo_automodel.components._peft.lora_experts.ops") as lora_ops,
        patch("nemo_automodel.components.moe.experts.ops", create=True) as parent_ops,
    ):
        lora_ops.gmm.side_effect = parent_ops.gmm.side_effect = _cpu_gmm
        actual = layer(x, mask, weights, indices)
        expected = parent(x, mask, weights, indices)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[~mask]) == 0


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ordinary_after_down_reduces_in_topk_not_expert_order(backend, dtype):
    layer, parent = _ordinary_expert_pair(backend, dtype, True, "swiglu", True)
    with torch.no_grad():
        for parameter in parent.parameters():
            parameter.zero_()
        parent.down_proj_bias[0].fill_(-(2**24))
        parent.down_proj_bias[1].fill_(0.25)
        parent.down_proj_bias[2].fill_(2**26)
        for name, parameter in parent.named_parameters():
            layer.get_parameter(name).copy_(parameter)
    x = torch.zeros(1, 32, dtype=dtype)
    mask = torch.ones(1, dtype=torch.bool)
    indices = torch.tensor([[2, 0, 1]])
    weights = torch.tensor([[0.5, 2.0, 4.0]])
    # Slot order: (2**25 - 2**25) + 1 == 1. Expert-order FP32
    # accumulation loses the 1: (-2**25 + 1) + 2**25 == 0.
    with patch.object(torch, "_grouped_mm", side_effect=_cpu_grouped_mm, create=True):
        expected = parent(x, mask, weights, indices)
        actual = layer(x, mask, weights, indices)
    torch.testing.assert_close(expected, torch.ones_like(x), atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["loop", "grouped_mm", "deepep_torch_mm", "deepep_gmm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("expert_bias", [False, True])
@pytest.mark.parametrize("activation", ["swiglu", "relu2"])
@pytest.mark.parametrize("after_down", [False, True])
def test_ordinary_nonzero_additive_reference(backend, dtype, expert_bias, activation, after_down):
    torch.manual_seed(89)
    layer, _ = _ordinary_expert_pair(backend, dtype, expert_bias, activation, after_down)
    with torch.no_grad():
        layer.lora_gate_and_up_B.normal_(std=0.15)
        layer.lora_down_B.normal_(std=0.15)
    reference = copy.deepcopy(layer)
    x, mask, weights, indices = _ordinary_routing_inputs(dtype)
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    deep_ep = backend.startswith("deepep")
    if deep_ep:
        layer.token_dispatcher = _cpu_slot_dispatcher(4)
    expected = _reference_additive_experts(
        reference, ref_x, mask, ref_weights, indices, grouped=backend != "loop", deep_ep=deep_ep
    )
    with (
        patch.object(torch, "_grouped_mm", side_effect=_cpu_grouped_mm, create=True),
        patch("nemo_automodel.components._peft.lora_experts.ops") as ops,
    ):
        ops.gmm.side_effect = _cpu_gmm
        actual = layer(x, mask, weights, indices)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # The independent FP32 autograd oracle verifies routing and every additive
    # adapter path. BF16 forward remains exact; its custom activation backward
    # intentionally uses different intermediate precision from naive autograd.
    if dtype == torch.float32:
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        for value, ref in ((x, ref_x), (weights, ref_weights)):
            torch.testing.assert_close(value.grad, ref.grad, atol=2e-6, rtol=2e-5)
        for name, parameter in layer.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, atol=2e-6, rtol=2e-5)
                assert torch.count_nonzero(parameter.grad) > 0
            else:
                assert parameter.grad is None

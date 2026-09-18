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

"""CPU regressions for ordinary expert LoRA routing and checkpoint compatibility.

Only GEMM kernels and the DeepEP transport are replaced by CPU doubles. Real
expert activation, permutation, bias and router autograd paths remain active.
These tests do not establish native DeepEP or multi-rank collective correctness.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components._peft import lora_experts
from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP

_ADAPTER_NAMES = {"lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"}
_BACKENDS = ("loop", "grouped_mm", "deepep_torch", "deepep_gmm")


@pytest.fixture(autouse=True)
def _cpu_kernels(monkeypatch):
    """Exercise real routing math without CUDA kernels or compiler dependencies."""
    monkeypatch.setattr(torch, "_grouped_mm", _grouped_mm, raising=False)
    monkeypatch.setattr(lora_experts, "ops", SimpleNamespace(gmm=_gmm))
    with torch.compiler.set_stance("force_eager"), torch.random.fork_rng(devices=[]):
        torch.manual_seed(913)
        yield


def _grouped_mm(x: torch.Tensor, weight: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    """Provide a differentiable CPU grouped GEMM double.

    Args:
        x: Tensor of shape [routed_tokens, in], grouped by expert.
        weight: Tensor of shape [experts, in, out], matching x's dtype/device.
        offs: Integer tensor of shape [experts], cumulative token counts.

    Returns:
        Tensor of shape [routed_tokens, out], in x's dtype/device.
    """
    assert x.dtype == weight.dtype
    assert x.device.type == weight.device.type == "cpu"
    boundaries = [0, *offs.tolist()]
    assert boundaries[-1] == x.shape[0]
    return torch.cat([x[start:end] @ w for start, end, w in zip(boundaries, boundaries[1:], weight)])


def _gmm(x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor, *, trans_b: bool) -> torch.Tensor:
    """Adapt grouped_gemm's count-based interface to the CPU double.

    Args:
        x: Tensor of shape [routed_tokens, in], grouped by expert.
        weight: Tensor of shape [experts, in, out], matching x's dtype/device.
        counts: Integer tensor of shape [experts], per-expert token counts.
        trans_b: Must be false for the expert weight layout.

    Returns:
        Tensor of shape [routed_tokens, out], in x's dtype/device.
    """
    assert not trans_b
    return _grouped_mm(x, weight, offs=counts.cumsum(0))


class _CPUDispatcher:
    """Model local dispatch/combine, including empty routes, without communication."""

    def token_permutation2(
        self,
        *,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Group valid top-k slots by expert with a differentiable gather.

        Args:
            hidden_states: CPU tensor of shape [tokens, hidden].
            num_local_tokens: Number of rows in hidden_states.
            token_probs: CPU tensor of shape [tokens, top_k].
            token_indices: Integer CPU tensor of shape [tokens, top_k]; -1 is masked.

        Returns:
            Tuple of hidden states [routed_tokens, hidden], integer counts
            [experts], and probabilities [routed_tokens], grouped by expert.
        """
        self.tokens = num_local_tokens
        self.top_k = token_indices.shape[1]
        locations = [(token_indices == expert).nonzero(as_tuple=True) for expert in range(3)]
        token_ids = torch.cat([loc[0] for loc in locations])
        slot_ids = torch.cat([loc[1] for loc in locations])
        self.flat_slots = token_ids * self.top_k + slot_ids
        self.input_dtype = hidden_states.dtype
        return (
            hidden_states[token_ids],
            torch.tensor([loc[0].numel() for loc in locations], dtype=torch.int64),
            token_probs[token_ids, slot_ids],
        )

    def token_unpermutation(self, output: torch.Tensor) -> torch.Tensor:
        """Combine already-weighted expert outputs, without applying probabilities again.

        Args:
            output: CPU tensor of shape [routed_tokens, hidden], in dispatch order.

        Returns:
            CPU tensor of shape [tokens, hidden], summed over top-k slots.
        """
        assert output.ndim == 2
        assert output.shape[0] == self.flat_slots.numel()
        assert output.dtype == self.input_dtype
        slots = output.new_zeros(self.tokens * self.top_k, output.shape[-1])
        slots = slots.index_add(0, self.flat_slots, output)
        return slots.view(self.tokens, self.top_k, output.shape[-1]).float().sum(1).to(output.dtype)


def _source(*, backend: str, after_down: bool, bias: bool, dtype: torch.dtype, activation: str = "relu2"):
    config = MoEConfig(
        n_routed_experts=3,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=4,
        inter_dim=4,
        moe_inter_dim=4,
        norm_topk_prob=False,
        expert_bias=bias,
        expert_activation=activation,
        swiglu_limit=0.7,
        apply_router_weight_after_down=after_down,
        dtype=torch.bfloat16,
    )
    if backend.startswith("deepep"):
        source = GroupedExpertsDeepEP(
            config,
            dispatcher_backend="hybridep",
            dispatcher_num_sms=24,
            dispatcher_share_token_dispatcher=False,
            dispatcher_async_dispatch=True,
        )
        source.n_routed_experts = 3
        source.ep_size = 1
        source.ep_rank = 0
        source.token_dispatcher = _CPUDispatcher()
    else:
        source = GroupedExperts(config)
    source.to(dtype=dtype)
    source.use_torch_mm = backend in ("grouped_mm", "deepep_torch")
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.7, 0.7)
    return source


def _wrap(source, *, lora_dtype=None):
    cls = (
        lora_experts.GroupedExpertsDeepEPLoRA
        if isinstance(source, GroupedExpertsDeepEP)
        else lora_experts.GroupedExpertsLoRA
    )
    return cls(source, lora_dim=2, alpha=3, lora_dtype=lora_dtype)


def _inputs(dtype: torch.dtype):
    x = torch.tensor([[0.2, -0.7, 1.1, 0.9], [-0.3, 0.8, 0.6, -0.4], [0.9, 0.3, -0.8, 0.2]], dtype=dtype)
    weights = torch.tensor([[0.17321, 0.82679], [0.31987, 0.68013], [0.73129, 0.26871]])
    # Reversed expert/slot order, uneven group sizes, and a masked token.
    indices = torch.tensor([[2, 0], [1, 0], [0, 2]])
    return x.requires_grad_(), torch.tensor([True, True, False]), weights.requires_grad_(), indices


def _dense_reference(
    module,
    x: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    merged: bool,
    dispatch_rounding: bool,
) -> torch.Tensor:
    """Compute every expert densely, then select routes as an independent oracle.

    Args:
        module: Expert module with base [experts, in, out] and low-rank parameters.
        x: CPU tensor of shape [tokens, hidden].
        mask: Boolean CPU tensor of shape [tokens].
        weights: FP32 CPU tensor of shape [tokens, top_k].
        indices: Integer CPU tensor of shape [tokens, top_k].
        merged: Use algebraically merged weights for FP32 gradient comparisons.
        dispatch_rounding: Round each weighted output before top-k combine, as DeepEP requires.

    Returns:
        CPU tensor of shape [tokens, hidden], in x's dtype.
    """
    gate_up = module.gate_and_up_projs.to(x.dtype)
    down = module.down_projs.to(x.dtype)
    up_a = module.lora_gate_and_up_A.to(x.dtype)
    up_b = module.lora_gate_and_up_B.to(x.dtype)
    down_a = module.lora_down_A.to(x.dtype)
    down_b = module.lora_down_B.to(x.dtype)
    if merged:
        projected = x @ (gate_up + module.scale * (up_a @ up_b))
    else:
        projected = x @ gate_up + ((x @ up_a) @ up_b) * module.scale
    if module.expert_bias:
        projected = projected + module.gate_up_proj_bias.to(x.dtype)[:, None, :]
    if module.config.expert_activation == "relu2":
        activated = F.relu(projected).square()
    else:
        gate, up = projected.chunk(2, dim=-1)
        activated = F.silu(gate.float().clamp(max=0.7)) * up.float().clamp(-0.7, 0.7)
    # [experts, tokens, intermediate] -> [tokens, top_k, intermediate].
    token_ids = torch.arange(x.shape[0])[:, None]
    selected = activated[indices, token_ids]
    after_down = module.config.apply_router_weight_after_down
    selected = (selected * (1.0 if after_down else weights[..., None])).to(x.dtype)
    if merged:
        effective_down = down + module.scale * (down_a @ down_b)
        output = (selected.unsqueeze(-2) @ effective_down[indices]).squeeze(-2)
    else:
        output = (selected.unsqueeze(-2) @ down[indices]).squeeze(-2)
        adapter = (selected.unsqueeze(-2) @ down_a[indices]) @ down_b[indices]
        output = output + adapter.squeeze(-2) * module.scale
    if module.expert_bias:
        bias = module.down_proj_bias.to(x.dtype)[indices]
        output = output + bias * (1.0 if after_down else weights[..., None])
        if module.use_torch_mm or dispatch_rounding:
            output = output.to(x.dtype)
    if after_down:
        output = output.float() * weights[..., None]
        if dispatch_rounding:
            output = output.to(x.dtype)
    return (output.float() * mask[:, None, None]).sum(1).to(x.dtype)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lora_dtype", [None, "float32"])
@pytest.mark.parametrize("use_mxfp8", [False, True])
def test_wrapper_preserves_storage_flags_and_parameter_paths(backend, dtype, lora_dtype, use_mxfp8):
    """Wrapping preserves source storage and paths while freezing only base parameters."""
    source = _source(backend=backend, after_down=True, bias=True, dtype=dtype)
    source.use_mxfp8 = use_mxfp8
    before = deepcopy(source.state_dict())
    wrapped = _wrap(source, lora_dtype=lora_dtype)
    assert wrapped.config is source.config
    assert wrapped.use_torch_mm is source.use_torch_mm
    assert wrapped.use_mxfp8 is source.use_mxfp8
    assert set(dict(wrapped.named_parameters())) == set(before) | _ADAPTER_NAMES
    assert set(wrapped.state_dict()) == set(before) | _ADAPTER_NAMES
    for name, parameter in wrapped.named_parameters():
        if name in _ADAPTER_NAMES:
            assert parameter.requires_grad
            assert parameter.dtype == (torch.float32 if lora_dtype else dtype)
            assert parameter.device == source.gate_and_up_projs.device
            assert torch.isfinite(parameter).all()
            if name.endswith("_B"):
                assert torch.count_nonzero(parameter) == 0
        else:
            assert not parameter.requires_grad
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            assert parameter.data_ptr() != source.get_parameter(name).data_ptr()
            assert source.get_parameter(name).requires_grad
    if backend.startswith("deepep"):
        for name in (
            "dispatcher_backend",
            "dispatcher_num_sms",
            "dispatcher_share_token_dispatcher",
            "dispatcher_async_dispatch",
            "ep_size",
            "ep_rank",
            "n_routed_experts",
        ):
            assert getattr(wrapped, name) == getattr(source, name)
        assert wrapped.token_dispatcher is source.token_dispatcher
    clone = _wrap(source, lora_dtype=lora_dtype)
    clone.load_state_dict(wrapped.state_dict(), strict=True)
    for name, value in clone.state_dict().items():
        torch.testing.assert_close(value, wrapped.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["loop", "deepep_torch"])
def test_peft_entry_uses_the_imported_expert_class(backend):
    """The ordinary patch entry point must use the same canonical module as these tests."""
    source = _source(backend=backend, after_down=True, bias=True, dtype=torch.float32)
    expected = (
        lora_experts.GroupedExpertsDeepEPLoRA if backend.startswith("deepep") else lora_experts.GroupedExpertsLoRA
    )
    patched = patch_moe_module(source, dim=2)
    assert isinstance(patched, expected)
    assert patched.use_torch_mm is source.use_torch_mm
    assert set(source.state_dict()).issubset(patched.state_dict())


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("after_down", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", ["relu2", "swiglu"])
def test_zero_b_matches_parent_forward_and_router_gradients(backend, dtype, after_down, bias, activation):
    """Zero adapters preserve the parent output exactly for each supported routing flag."""
    source = _source(backend=backend, after_down=after_down, bias=bias, dtype=dtype, activation=activation)
    wrapped = _wrap(source)
    x, mask, weights, indices = _inputs(dtype)
    parent_x = x.detach().clone().requires_grad_()
    parent_weights = weights.detach().clone().requires_grad_()
    expected = source(parent_x, mask, parent_weights, indices)
    actual = wrapped(x, mask, weights, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, parent_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, parent_weights.grad, rtol=0, atol=0)
    assert all(parameter.grad is None for name, parameter in wrapped.named_parameters() if name not in _ADAPTER_NAMES)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("after_down", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("activation", ["relu2", "swiglu"])
def test_nonzero_adapters_match_dense_forward_backward_and_step(backend, after_down, bias, activation):
    """An independent dense merged FP32 oracle verifies every trainable gradient and update."""
    source = _source(backend=backend, after_down=after_down, bias=bias, dtype=torch.float32, activation=activation)
    wrapped = _wrap(source)
    with torch.no_grad():
        wrapped.lora_gate_and_up_B.uniform_(-0.2, 0.2)
        wrapped.lora_down_B.uniform_(-0.2, 0.2)
    reference = deepcopy(wrapped)
    base_before = {
        name: parameter.detach().clone() for name, parameter in wrapped.named_parameters() if name not in _ADAPTER_NAMES
    }
    x, mask, weights, indices = _inputs(torch.float32)
    ref_x = x.detach().clone().requires_grad_()
    ref_weights = weights.detach().clone().requires_grad_()
    actual = wrapped(x, mask, weights, indices)
    expected = _dense_reference(
        reference, ref_x, mask, ref_weights, indices, merged=True, dispatch_rounding=backend.startswith("deepep")
    )
    # FP32 merged versus additive projection reassociation is not bitwise identical.
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(weights.grad, ref_weights.grad, rtol=2e-5, atol=2e-6)
    for name in _ADAPTER_NAMES:
        grad = wrapped.get_parameter(name).grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
        torch.testing.assert_close(grad, reference.get_parameter(name).grad, rtol=2e-5, atol=2e-6)
    torch.optim.SGD(wrapped.parameters(), lr=0.05).step()
    torch.optim.SGD(reference.parameters(), lr=0.05).step()
    for name, parameter in wrapped.named_parameters():
        if name in _ADAPTER_NAMES:
            torch.testing.assert_close(parameter, reference.get_parameter(name), rtol=2e-5, atol=2e-6)
        else:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, base_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("after_down", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_bf16_routing_matches_dense_additive_rounding(backend, after_down, bias):
    """BF16 exposes routing placement and DeepEP's per-slot cast before combine."""
    wrapped = _wrap(_source(backend=backend, after_down=after_down, bias=bias, dtype=torch.bfloat16))
    with torch.no_grad():
        wrapped.lora_gate_and_up_B.uniform_(-0.2, 0.2)
        wrapped.lora_down_B.uniform_(-0.2, 0.2)
    x, mask, weights, indices = _inputs(torch.bfloat16)
    actual = wrapped(x, mask, weights, indices)
    expected = _dense_reference(
        wrapped, x, mask, weights, indices, merged=False, dispatch_rounding=backend.startswith("deepep")
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    other = deepcopy(wrapped)
    other.config = deepcopy(wrapped.config)
    other.config.apply_router_weight_after_down = not after_down
    alternative = _dense_reference(
        other, x, mask, weights, indices, merged=False, dispatch_rounding=backend.startswith("deepep")
    )
    assert not torch.equal(expected, alternative), "the fixture must distinguish pre- and post-down rounding"


@pytest.mark.parametrize("backend", ["loop", "grouped_mm"])
def test_post_down_reduces_in_topk_slot_order(backend):
    """Cancellation distinguishes top-k slot reduction from expert-order accumulation."""
    source = _source(backend=backend, after_down=True, bias=True, dtype=torch.float32)
    source.config.n_activated_experts = 3
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.zero_()
        source.down_proj_bias.copy_(torch.tensor([[1e20] * 4, [-1e20] * 4, [3.0] * 4]))
    wrapped = _wrap(source)
    x = torch.ones(1, 4)
    weights = torch.ones(1, 3)
    indices = torch.tensor([[2, 0, 1]])
    mask = torch.ones(1, dtype=torch.bool)
    actual = wrapped(x, mask, weights, indices)
    expected = source.down_proj_bias[indices].sum(1)
    assert torch.equal(expected, torch.zeros_like(expected))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    source.config.apply_router_weight_after_down = False
    legacy = wrapped(x, mask, weights, indices)
    torch.testing.assert_close(legacy, torch.full_like(legacy, 3.0), rtol=0, atol=0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("after_down", [False, True])
@pytest.mark.parametrize("empty_case", ["zero_tokens", "masked", "no_local_routes"])
def test_empty_routes_preserve_shape_and_zero_gradients(backend, dtype, after_down, empty_case):
    """Empty inputs and fully inactive routes remain attached with exactly zero gradients."""
    wrapped = _wrap(_source(backend=backend, after_down=after_down, bias=True, dtype=dtype))
    x, mask, weights, indices = _inputs(dtype)
    if empty_case == "zero_tokens":
        x = x[:0].detach().requires_grad_()
        weights = weights[:0].detach().requires_grad_()
        mask, indices = mask[:0], indices[:0]
    elif empty_case == "masked":
        mask = torch.zeros_like(mask)
    else:
        indices = torch.full_like(indices, 3)
    actual = wrapped(x, mask, weights, indices)
    assert actual.shape == x.shape and actual.dtype == dtype
    assert torch.isfinite(actual).all() and torch.count_nonzero(actual) == 0
    actual.sum().backward()
    for tensor in (x, weights, *(wrapped.get_parameter(name) for name in _ADAPTER_NAMES)):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all() and torch.count_nonzero(tensor.grad) == 0
    assert all(parameter.grad is None for name, parameter in wrapped.named_parameters() if name not in _ADAPTER_NAMES)

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

"""Unit tests for the ERNIE 5.5 gate and latent MoE block."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5 import moe as moe_module
from nemo_automodel.components.models.ernie5_5.layers import ernie_situ
from nemo_automodel.components.models.ernie5_5.moe import Ernie5_5Gate, Ernie5_5MoE
from nemo_automodel.components.moe import experts as shared_experts
from nemo_automodel.components.moe import layers as shared_layers
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP

# No test of this file uses CUDA.
pytestmark = pytest.mark.cpu_only

# Ernie5_5Gate


def _gate_config() -> MoEConfig:
    """Build a tiny gate configuration."""
    return MoEConfig(
        n_routed_experts=3,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.1,
        aux_loss_coeff=0,
        score_func="sigmoid",
        route_scale=3.0,
        dim=2,
        inter_dim=4,
        moe_inter_dim=4,
        norm_topk_prob=True,
        gate_dtype=torch.float32,
        router_weights_fp32=True,
    )


def test_gate_routes_and_updates_bias_for_valid_tokens() -> None:
    gate = Ernie5_5Gate(_gate_config())
    with torch.no_grad():
        gate.weight.zero_()
        gate.weight_1.zero_()
        gate.e_score_correction_bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
        gate.routed_scaling_factor_param.copy_(torch.tensor([2.0, 4.0, 6.0]))
    hidden = torch.ones(2, 2, requires_grad=True)
    weights, indices, aux_loss = gate(hidden, torch.tensor([True, False]), None)
    assert indices.tolist() == [[2, 1], [2, 1]]
    torch.testing.assert_close(weights, torch.tensor([[3.0, 2.0], [3.0, 2.0]]))
    assert aux_loss is None
    assert gate._cumulative_expert_load.tolist() == [0, 1, 1]
    weights.sum().backward()
    assert gate.weight.grad is not None
    assert gate.weight_1.grad is not None
    assert gate.routed_scaling_factor_param.grad is not None
    gate.update_bias()
    torch.testing.assert_close(gate.e_score_correction_bias, torch.tensor([0.1, 0.9, 1.9]))
    assert gate._cumulative_expert_load is None


def test_gate_rejects_unsupported_aux_loss() -> None:
    config = _gate_config()
    config.aux_loss_coeff = 0.1
    with pytest.raises(ValueError, match="Auxiliary loss"):
        Ernie5_5Gate(config)


def test_gate_rejects_incompatible_score_contract() -> None:
    config = _gate_config()
    config.score_func = "softmax"
    with pytest.raises(ValueError, match="sigmoid"):
        Ernie5_5Gate(config)
    config.score_func = "sigmoid"
    config.router_weight_uses_score_correction_bias = True
    with pytest.raises(ValueError, match="selection-only"):
        Ernie5_5Gate(config)
    config.router_weight_uses_score_correction_bias = False
    config.router_weights_fp32 = False
    with pytest.raises(ValueError, match="FP32 routing weights"):
        Ernie5_5Gate(config)


def test_gate_state_dict_roundtrip_preserves_fp32_values() -> None:
    source = Ernie5_5Gate(_gate_config())
    with torch.no_grad():
        source.weight.fill_(0.123456)
        source.weight_1.fill_(-0.432109)
        source.routed_scaling_factor_param.copy_(torch.tensor([1.125, 2.25, 3.5]))
        source.e_score_correction_bias.copy_(torch.tensor([0.007, -0.011, 0.017]))
    restored = Ernie5_5Gate(_gate_config())
    restored.load_state_dict(source.state_dict(), strict=True)
    for name, tensor in source.state_dict().items():
        assert restored.state_dict()[name].dtype == torch.float32
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


def test_gate_meta_initialization_keeps_parameters_on_meta_device() -> None:
    with torch.device("meta"):
        gate = Ernie5_5Gate(_gate_config())
    assert all(parameter.device.type == "meta" for parameter in gate.parameters())


def test_gate_counts_reentrant_recompute_once() -> None:
    gate = Ernie5_5Gate(_gate_config())
    hidden = torch.ones(2, 2, requires_grad=True)
    token_mask = torch.tensor([True, False])
    with torch.no_grad():
        gate(hidden, token_mask, None)
    assert gate._cumulative_expert_load is None
    _, indices, _ = gate(hidden, token_mask, None)
    expected = torch.bincount(indices[0], minlength=gate.n_experts)
    torch.testing.assert_close(gate._cumulative_expert_load, expected)


def test_gate_initializes_new_parameters_after_meta_materialization() -> None:
    with torch.device("meta"):
        gate = Ernie5_5Gate(_gate_config())
    gate.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.fill_(float("nan"))
        gate.e_score_correction_bias.fill_(float("nan"))
    gate.init_weights(torch.device("cpu"))
    for parameter in gate.parameters():
        assert torch.isfinite(parameter).all()
    torch.testing.assert_close(gate.routed_scaling_factor_param, torch.full((3,), 3.0))
    torch.testing.assert_close(gate.e_score_correction_bias, torch.zeros(3))


# Ernie5_5MoE


def _moe_config() -> MoEConfig:
    """Build a tiny latent MoE configuration."""
    return MoEConfig(
        n_routed_experts=3,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0,
        score_func="sigmoid",
        route_scale=3.0,
        dim=4,
        inter_dim=5,
        moe_inter_dim=5,
        moe_latent_size=3,
        norm_topk_prob=True,
        force_e_score_correction_bias=True,
        dtype=torch.float32,
        gate_dtype=torch.float32,
        router_weights_fp32=True,
    )


def _moe_model() -> Ernie5_5MoE:
    """Build the tiny MoE on the torch backends."""
    return Ernie5_5MoE(_moe_config(), BackendConfig(experts="torch", dispatcher="torch", linear="torch"))


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Evaluate SiTU with beta 4 and linear beta 25 without intermediate rounding.

    Args:
        gate: Gate projection of shape [..., intermediate], with arbitrary leading dimensions.
        up: Up projection of shape [..., intermediate].

    Returns:
        Activation of shape [..., intermediate].
    """
    return 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))


def _explicit_reference(model: Ernie5_5MoE, x: torch.Tensor) -> torch.Tensor:
    """Recompute the MoE output token by token from the module parameters.

    Args:
        model: MoE built by ``_moe_model``.
        x: Token states of shape [tokens, hidden].

    Returns:
        Output of shape [tokens, hidden].
    """
    gate = model.gate
    scores = F.linear(x, gate.weight).sigmoid() + F.linear(x, gate.weight_1).sigmoid()
    indices = (scores + gate.e_score_correction_bias).topk(2, dim=-1).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * gate.routed_scaling_factor_param[indices]
    latent = F.linear(x, model.fc1_latent_proj.weight)
    rows = []
    for row in range(x.shape[0]):
        values = []
        for slot in range(2):
            expert = indices[row, slot]
            projected = latent[row] @ model.experts.gate_and_up_projs[expert]
            gate_proj, up_proj = projected.chunk(2)
            values.append((_situ(gate_proj, up_proj) * weights[row, slot]) @ model.experts.down_projs[expert])
        rows.append(sum(values))
    routed = F.rms_norm(torch.stack(rows), (3,), model.latent_norm.weight, 1e-6)
    routed = F.linear(routed, model.fc2_latent_proj.weight)
    shared = model.shared_experts
    common = _situ(F.linear(x, shared.gate_proj.weight), F.linear(x, shared.up_proj.weight))
    return routed + F.linear(common, shared.down_proj.weight)


def test_moe_matches_explicit_forward_and_input_gradients():
    torch.manual_seed(5571)
    model = _moe_model()
    model.init_weights(torch.device("cpu"))
    x = torch.randn(3, 4, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = model(x)
    expected = _explicit_reference(model, reference_x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), x)[0]
    reference_grad = torch.autograd.grad(expected.square().sum(), reference_x)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-4, atol=1e-6)


def test_moe_meta_initialization_and_fp32_guard():
    with torch.device("meta"):
        model = _moe_model()
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    model.init_weights(torch.device("cpu"))
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert torch.all(model.gate.routed_scaling_factor_param == 3.0)
    model.to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="latent projection parameters must remain FP32"):
        model(torch.zeros(1, 4, dtype=torch.bfloat16))


def test_moe_rejects_unsharded_distributed_execution_and_context_parallel(monkeypatch):
    model = _moe_model()
    inputs = torch.zeros(2, 4)
    with pytest.raises(RuntimeError, match="does not support context parallelism"):
        model(inputs, cp_mesh=object())
    monkeypatch.setattr(moe_module, "get_world_size_safe", lambda: 2)
    with pytest.raises(RuntimeError, match="requires sharded expert DTensors"):
        model(inputs)


# Expert and dispatcher backends

ONE_PROCESS_WARNING = "'hybridep' dispatcher is enabled .* world size is 1"


def _hybridep_backend() -> BackendConfig:
    """Return the grouped-GEMM experts with the HybridEP token dispatcher."""
    return BackendConfig(experts="torch_mm", dispatcher="hybridep", linear="torch")


def _build_for_job(monkeypatch: pytest.MonkeyPatch, ranks: int) -> None:
    """Let the shared MoE constructor see a job of ``ranks`` processes; no process group exists in these tests."""
    monkeypatch.setattr(shared_layers, "get_world_size_safe", lambda: ranks)


def _state_layout(model: Ernie5_5MoE) -> dict[str, tuple[torch.Size, torch.dtype]]:
    """Return shape and dtype of every parameter and buffer of the MoE by state-dict name."""
    return {name: (tensor.shape, tensor.dtype) for name, tensor in model.state_dict().items()}


def test_moe_installs_situ_on_grouped_experts_for_the_torch_pair_and_in_one_process():
    gate_up, weights = torch.randn(4, 10), torch.rand(4, 1)
    expected = ernie_situ(gate_up, beta=4.0, linear_beta=25.0, routing_weight=weights)
    with pytest.warns(UserWarning, match=ONE_PROCESS_WARNING):
        one_process = Ernie5_5MoE(_moe_config(), _hybridep_backend())
    for model in (_moe_model(), one_process):
        assert type(model.experts) is GroupedExperts
        actual = model.experts.expert_activation_grouped(gate_up, weights)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_moe_installs_situ_on_the_hybridep_experts_of_a_job_and_keeps_the_state_layout(monkeypatch):
    reference = _state_layout(_moe_model())
    _build_for_job(monkeypatch, 2)
    model = Ernie5_5MoE(_moe_config(), _hybridep_backend())
    assert type(model.experts) is GroupedExpertsDeepEP
    gate_up, weights = torch.randn(4, 10), torch.rand(4, 1)
    expected = ernie_situ(gate_up, beta=4.0, linear_beta=25.0, routing_weight=weights)
    torch.testing.assert_close(model.experts.expert_activation(gate_up, weights), expected, rtol=0, atol=0)
    # Both expert classes hold the same tensors under the same names, so a checkpoint of one fits the other.
    assert _state_layout(model) == reference
    # The torch pair keeps the experts without a token dispatcher in a job as well.
    assert type(_moe_model().experts) is GroupedExperts


def test_moe_forward_in_one_process_matches_the_explicit_reference_with_the_hybridep_pair(monkeypatch):
    """In one process the HybridEP pair runs the grouped-GEMM path of ``GroupedExperts``."""

    def dense_grouped_mm(value: torch.Tensor, weights: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        """Stand in for ``torch._grouped_mm``, which not every torch build has for CPU.

        Args:
            value: Tensor of shape [routed_tokens, input], with the rows of each expert together.
            weights: Tensor of shape [experts, input, output].
            offs: Integer tensor of shape [experts] with the cumulative row count after each expert.

        Returns:
            Tensor of shape [routed_tokens, output].
        """
        starts = [0, *offs.tolist()]
        return torch.cat([value[starts[index] : starts[index + 1]] @ weights[index] for index in range(len(weights))])

    monkeypatch.setattr(shared_experts, "select_grouped_mm", lambda use_mxfp8: dense_grouped_mm)
    torch.manual_seed(5571)
    with pytest.warns(UserWarning, match=ONE_PROCESS_WARNING):
        model = Ernie5_5MoE(_moe_config(), _hybridep_backend())
    model.init_weights(torch.device("cpu"))
    x = torch.randn(3, 4, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = model(x)
    expected = _explicit_reference(model, reference_x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual_grad = torch.autograd.grad(actual.square().sum(), x)[0]
    reference_grad = torch.autograd.grad(expected.square().sum(), reference_x)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-4, atol=1e-6)


def test_moe_rejects_activations_that_are_not_bf16_before_a_hybridep_dispatch(monkeypatch):
    """HybridEP exchanges 16-bit activations only, so FP32 activations are refused before the experts run."""
    _build_for_job(monkeypatch, 2)
    model = Ernie5_5MoE(_moe_config(), _hybridep_backend())
    model.init_weights(torch.device("cpu"))
    # These experts have no token dispatcher: a forward that reached them would fail with an AttributeError.
    with pytest.raises(RuntimeError, match=r"requires BF16 activations at the routed experts, got torch\.float32"):
        model(torch.zeros(2, 4))


def test_moe_rejects_a_hybridep_backend_whose_experts_were_built_for_one_process(monkeypatch):
    with pytest.warns(UserWarning, match=ONE_PROCESS_WARNING):
        model = Ernie5_5MoE(_moe_config(), _hybridep_backend())
    monkeypatch.setattr(moe_module, "get_world_size_safe", lambda: 2)
    with pytest.raises(RuntimeError, match="HybridEP dispatcher was requested, but this ERNIE MoE was built for one"):
        model(torch.zeros(2, 4))


@pytest.mark.parametrize(
    ("experts", "dispatcher"),
    [("torch", "hybridep"), ("te", "hybridep"), ("torch_mm", "torch"), ("torch_mm", "deepep")],
)
def test_moe_rejects_unsupported_expert_and_dispatcher_pairs(experts, dispatcher):
    backend = BackendConfig(experts=experts, dispatcher=dispatcher, linear="torch")
    message = rf"with dispatcher='hybridep'; got experts='{experts}' with dispatcher='{dispatcher}'"
    with pytest.raises(ValueError, match=message):
        Ernie5_5MoE(_moe_config(), backend)

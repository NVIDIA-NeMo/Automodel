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
import importlib.util
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import _mask_routing_metadata
from nemo_automodel.components.moe.layers import Gate, MoE
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler


@pytest.fixture(autouse=True)
def _restore_aux_loss_scale():
    previous = MoEAuxLossAutoScaler.main_loss_backward_scale
    yield
    MoEAuxLossAutoScaler.main_loss_backward_scale = previous


def _gpt_oss_moe_config(**overrides) -> MoEConfig:
    values = dict(
        n_routed_experts=8,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.03,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        router_bias=True,
        expert_bias=True,
        dtype=torch.float32,
    )
    values.update(overrides)
    return MoEConfig(**values)


def _emulate_te_fused_topk(**kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference with TE's dense output contract."""
    assert kwargs["score_function"] == "softmax"
    assert kwargs["use_pre_softmax"] is False
    assert kwargs["num_groups"] is None
    assert kwargs["group_topk"] is None
    assert kwargs["expert_bias"] is None

    logits = kwargs["logits"]
    top_values, top_indices = torch.topk(logits, k=kwargs["topk"], dim=-1)
    top_probs = torch.softmax(top_values, dim=-1, dtype=torch.float32).type_as(logits)
    scaling_factor = kwargs["scaling_factor"]
    if scaling_factor is not None:
        top_probs = top_probs * scaling_factor
    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, top_probs)
    routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, top_indices, True)
    return routing_probs, routing_map


def _compact_to_dense(
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = weights.new_zeros((weights.shape[0], num_experts)).scatter(1, indices, weights)
    routing_map = torch.zeros(
        (weights.shape[0], num_experts),
        dtype=torch.bool,
        device=indices.device,
    ).scatter(1, indices, True)
    return probs, routing_map


def _dense_gate_output(gate: Gate, x: torch.Tensor, token_mask: torch.Tensor):
    weights, indices, aux_loss = gate(x, token_mask, cp_mesh=None)
    if indices.dtype == torch.bool:
        return weights, indices, aux_loss
    probs, routing_map = _compact_to_dense(weights, indices, gate.n_experts)
    return probs, routing_map, aux_loss


def _assert_optimizer_state_close(left: torch.optim.Optimizer, right: torch.optim.Optimizer) -> None:
    left_states = list(left.state.values())
    right_states = list(right.state.values())
    assert len(left_states) == len(right_states)
    for left_state, right_state in zip(left_states, right_states):
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            torch.testing.assert_close(left_state[key], right_state[key], rtol=1e-6, atol=1e-7)


def test_te_router_fusion_matches_dynamic_multistep_training_on_cpu():
    """Different routing each step preserves outputs, aux gradients, and AdamW state."""
    torch.manual_seed(1234)
    baseline = Gate(_gpt_oss_moe_config())
    fused = Gate(_gpt_oss_moe_config(), fuse_router=True)
    fused.load_state_dict(copy.deepcopy(baseline.state_dict()))
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=2e-3, weight_decay=0.01)
    fused_optimizer = torch.optim.AdamW(fused.parameters(), lr=2e-3, weight_decay=0.01)
    token_mask = torch.tensor([True, True, False, True, True, True, False, True])
    observed_routes = []
    MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0)

    with patch(
        "nemo_automodel.components.moe.layers._te_fused_topk_with_score_function",
        side_effect=_emulate_te_fused_topk,
    ):
        for step in range(4):
            generator = torch.Generator().manual_seed(9000 + step)
            x = torch.randn(8, 16, generator=generator)
            expert_targets = torch.randn(8, 8, generator=generator)

            baseline_optimizer.zero_grad(set_to_none=True)
            fused_optimizer.zero_grad(set_to_none=True)
            baseline_probs, baseline_map, baseline_aux = _dense_gate_output(baseline, x, token_mask)
            fused_probs, fused_map, fused_aux = _dense_gate_output(fused, x, token_mask)

            torch.testing.assert_close(fused_map, baseline_map)
            torch.testing.assert_close(fused_probs, baseline_probs, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(fused_aux, baseline_aux, rtol=1e-6, atol=1e-7)
            observed_routes.append(fused_map.detach().clone())

            valid = token_mask.unsqueeze(-1)
            baseline_loss = (baseline_probs * expert_targets * valid).sum()
            fused_loss = (fused_probs * expert_targets * valid).sum()
            baseline_loss.backward()
            fused_loss.backward()

            for baseline_parameter, fused_parameter in zip(baseline.parameters(), fused.parameters()):
                torch.testing.assert_close(fused_parameter.grad, baseline_parameter.grad, rtol=1e-6, atol=1e-7)
            assert fused.bias is not None and fused.bias.grad is not None

            baseline_optimizer.step()
            fused_optimizer.step()
            for baseline_parameter, fused_parameter in zip(baseline.parameters(), fused.parameters()):
                torch.testing.assert_close(fused_parameter, baseline_parameter, rtol=1e-6, atol=1e-7)
            _assert_optimizer_state_close(baseline_optimizer, fused_optimizer)

    assert any(not torch.equal(observed_routes[0], route) for route in observed_routes[1:])


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_te_router_fusion_is_checkpoint_recomputation_safe_on_cpu(use_reentrant):
    """Checkpoint recomputation re-evaluates current logits without retaining routing state."""
    torch.manual_seed(4321)
    eager_gate = Gate(_gpt_oss_moe_config(), fuse_router=True)
    checkpoint_gate = Gate(_gpt_oss_moe_config(), fuse_router=True)
    checkpoint_gate.load_state_dict(copy.deepcopy(eager_gate.state_dict()))
    eager_x = torch.randn(8, 16, requires_grad=True)
    checkpoint_x = eager_x.detach().clone().requires_grad_()
    token_mask = torch.tensor([True, True, True, False, True, False, True, True])
    expert_targets = torch.randn(8, 8)

    def objective(gate: Gate, hidden_states: torch.Tensor) -> torch.Tensor:
        probs, _, _ = _dense_gate_output(gate, hidden_states, token_mask)
        return (probs * expert_targets * token_mask.unsqueeze(-1)).sum()

    with patch(
        "nemo_automodel.components.moe.layers._te_fused_topk_with_score_function",
        side_effect=_emulate_te_fused_topk,
    ):
        eager_loss = objective(eager_gate, eager_x)
        checkpoint_loss = checkpoint(
            lambda hidden_states: objective(checkpoint_gate, hidden_states),
            checkpoint_x,
            use_reentrant=use_reentrant,
        )
        eager_loss.backward()
        checkpoint_loss.backward()

    torch.testing.assert_close(checkpoint_loss, eager_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(checkpoint_x.grad, eager_x.grad, rtol=1e-6, atol=1e-7)
    for eager_parameter, checkpoint_parameter in zip(eager_gate.parameters(), checkpoint_gate.parameters()):
        torch.testing.assert_close(checkpoint_parameter.grad, eager_parameter.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"score_func": "sigmoid"}, "score_func must be 'softmax'"),
        ({"softmax_before_topk": True}, "softmax_before_topk must be False"),
        ({"norm_topk_prob": True}, "norm_topk_prob must be False"),
        ({"n_expert_groups": 2, "n_limited_groups": 1}, "expert-group routing"),
        ({"force_e_score_correction_bias": True}, "correction bias"),
        ({"enable_routing_replay": True}, "routing replay"),
    ],
)
def test_te_router_fusion_rejects_non_exact_semantics(overrides, match):
    with pytest.raises(ValueError, match=match):
        Gate(_gpt_oss_moe_config(**overrides), fuse_router=True)


@pytest.mark.parametrize("fuse_router", [False, True])
def test_softmax_after_topk_skips_full_aux_scores_when_disabled(fuse_router):
    gate = Gate(_gpt_oss_moe_config(aux_loss_coeff=0.0), fuse_router=fuse_router)
    logits = torch.randn(8, gate.n_experts)
    with patch(
        "nemo_automodel.components.moe.layers._te_fused_topk_with_score_function",
        side_effect=_emulate_te_fused_topk,
    ):
        _, _, original_scores = gate._route_scores(logits)

    assert original_scores is None


def test_dense_hybridep_metadata_masks_padding_without_static_routing():
    probs = torch.tensor(
        [[0.6, 0.0, 0.4, 0.0], [0.0, 0.3, 0.0, 0.7]],
        requires_grad=True,
    )
    routing_map = probs.detach() != 0
    masked_probs, masked_map = _mask_routing_metadata(
        probs,
        routing_map,
        torch.tensor([True, False]),
    )

    torch.testing.assert_close(masked_probs[0], probs[0])
    torch.testing.assert_close(masked_probs[1], torch.zeros_like(probs[1]))
    torch.testing.assert_close(masked_map[0], routing_map[0])
    assert not masked_map[1].any()
    masked_probs.sum().backward()
    torch.testing.assert_close(probs.grad[0], torch.ones_like(probs[0]))
    torch.testing.assert_close(probs.grad[1], torch.zeros_like(probs[1]))


def test_moe_enables_fusion_only_when_hybridep_is_active():
    backend = BackendConfig(
        experts="torch_mm",
        dispatcher="hybridep",
        moe_router_fusion=True,
    )
    with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=2):
        distributed_moe = MoE(_gpt_oss_moe_config(), backend)
    with pytest.warns(UserWarning, match="world size is 1"):
        with patch("nemo_automodel.components.moe.layers.get_world_size_safe", return_value=1):
            single_rank_moe = MoE(_gpt_oss_moe_config(), backend)

    assert distributed_moe.gate.fuse_router is True
    assert single_rank_moe.gate.fuse_router is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("transformer_engine") is None,
    reason="requires CUDA and Transformer Engine's fused router",
)
def test_real_te_bf16_router_matches_eager_forward_and_backward():
    """Real TE kernel parity gate for the GB200 image."""
    try:
        from transformer_engine.pytorch.router import fused_topk_with_score_function  # noqa: F401
    except ImportError:
        pytest.skip("installed Transformer Engine does not provide fused_topk_with_score_function")

    torch.manual_seed(2026)
    config = _gpt_oss_moe_config(
        n_routed_experts=32,
        n_activated_experts=4,
        dim=64,
        dtype=torch.bfloat16,
        aux_loss_coeff=0.0,
    )
    baseline = Gate(config).cuda()
    fused = Gate(config, fuse_router=True).cuda()
    fused.load_state_dict(copy.deepcopy(baseline.state_dict()))
    baseline_x = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    fused_x = baseline_x.detach().clone().requires_grad_()
    token_mask = torch.ones(128, device="cuda", dtype=torch.bool)
    target = torch.randn(128, 32, device="cuda", dtype=torch.float32)

    baseline_probs, baseline_map, _ = _dense_gate_output(baseline, baseline_x, token_mask)
    fused_probs, fused_map, _ = _dense_gate_output(fused, fused_x, token_mask)
    torch.testing.assert_close(fused_map, baseline_map)
    torch.testing.assert_close(fused_probs, baseline_probs, rtol=2e-2, atol=2e-3)

    (baseline_probs.float() * target).sum().backward()
    (fused_probs.float() * target).sum().backward()
    torch.testing.assert_close(fused_x.grad, baseline_x.grad, rtol=3e-2, atol=3e-3)
    for baseline_parameter, fused_parameter in zip(baseline.parameters(), fused.parameters()):
        torch.testing.assert_close(fused_parameter.grad, baseline_parameter.grad, rtol=3e-2, atol=3e-3)

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

"""CPU coverage of GroupedExpertsTE under HybridEP capacity mode (device-side split sizes)."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig

NUM_EXPERTS = 4
TAIL_ROWS = 17  # capacity rows past the routed (and padded) total


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=NUM_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=32,
        inter_dim=64,
        moe_inter_dim=16,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.float32,
    )


@pytest.fixture
def fake_te(monkeypatch):
    """TE grouped-tensor GEMMs and a HybridEP capacity-mode dispatcher, on CPU.

    The GEMM multiplies each expert's rows by that expert's weight and returns NaN for the rows past the split
    total, as TE leaves them uninitialized. The dispatcher lays the routed rows out per local expert, padded with
    zero rows to the configured pad multiple as HybridEP's permute does, fills the buffer tail with garbage, and
    combines only the routed rows. Any leak of padding or tail rows into outputs or gradients therefore shows.
    """
    gemm_splits = []
    dispatcher_configs = []
    padding_calls = []

    class GroupedLinear(torch.nn.Module):
        def __init__(self, *, num_gemms, in_features, out_features, bias, params_dtype, device, use_grouped_tensor):
            super().__init__()
            assert use_grouped_tensor
            self.num_gemms = num_gemms
            self.weight = torch.nn.Parameter(torch.randn(num_gemms, out_features, in_features) * 0.3)

        def forward(self, values, splits):
            assert splits.dtype == torch.int64 and splits.device == values.device and splits.numel() == self.num_gemms
            gemm_splits.append(splits.clone())
            sizes = splits.tolist()  # test-only host read
            assert sum(sizes) <= values.shape[0]
            outputs, start = [], 0
            for expert, rows in enumerate(sizes):
                outputs.append(values[start : start + rows] @ self.weight[expert].T)
                start += rows
            outputs.append(values.new_full((values.shape[0] - start, self.weight.shape[1]), float("nan")))
            return torch.cat(outputs)

    class Padding(torch.nn.Module):
        def __init__(self, num_gemms, align_size=None):
            super().__init__()

        def forward(self, values, splits):
            raise AssertionError("TE's host-list padding must not run in capacity mode")

    class CapacityDispatcher:
        def __init__(self, num_local_experts, local_expert_indices, config, ep_group, hidden_dim=None):
            self.num_local_experts = num_local_experts
            self.pad = config.moe_router_expert_pad_multiple or 1
            dispatcher_configs.append(config)

        def token_permutation2(self, hidden_states, num_local_tokens, token_probs, token_indices):
            experts = token_indices[:, 0]
            counts = torch.bincount(experts, minlength=self.num_local_experts)
            padded = (counts + self.pad - 1) // self.pad * self.pad
            self.order = torch.argsort(experts, stable=True)
            sorted_experts = experts[self.order]
            rank_in_expert = torch.arange(self.order.numel()) - (counts.cumsum(0) - counts)[sorted_experts]
            self.rows = (padded.cumsum(0) - padded)[sorted_experts] + rank_in_expert
            routed_total = int(padded.sum())
            hidden = torch.cat(
                [
                    hidden_states.new_zeros(routed_total, hidden_states.shape[1]),
                    hidden_states.new_full((TAIL_ROWS, hidden_states.shape[1]), 1e30),
                ]
            ).index_put((self.rows,), hidden_states[self.order])
            probs = torch.cat([torch.zeros(routed_total), torch.full((TAIL_ROWS,), float("nan"))])
            probs = probs.index_put((self.rows,), token_probs[self.order, 0].float())
            return hidden, padded, probs

        def token_unpermutation(self, values):
            routed = values[self.rows]
            return torch.zeros_like(routed).index_copy(0, self.order, routed)

    state = Mock()
    state.is_fp8_enabled.return_value = False
    te = ModuleType("transformer_engine.pytorch")
    te.GroupedLinear, te.Fp8Padding, te.Fp8Unpadding = GroupedLinear, Padding, Padding
    quantization = ModuleType("transformer_engine.pytorch.quantization")
    quantization.FP8GlobalStateManager = state
    quantization.get_align_size_for_quantization = lambda recipe: 16
    grouped_linear = ModuleType("transformer_engine.pytorch.module.grouped_linear")
    grouped_linear.is_module_grouped_tensor_path_supported = lambda recipe, dtype: True
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.quantization", quantization)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module.grouped_linear", grouped_linear)
    monkeypatch.setattr("nemo_automodel.components.moe.experts.is_te_min_version", lambda version: True)
    monkeypatch.setattr("nemo_automodel.components.models.common.utils._patch_te_modules", lambda: None)
    monkeypatch.setattr("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher", CapacityDispatcher)

    def no_host_read(*args, **kwargs):
        raise AssertionError("capacity mode must not read the split sizes on the host")

    monkeypatch.setattr("nemo_automodel.components.moe.experts._resolve_m_splits", no_host_read)

    from nemo_automodel.components.moe import experts as experts_module

    real_pad = experts_module.pad_expert_rows

    def counted_pad(*args, **kwargs):
        padding_calls.append(kwargs["align"])
        return real_pad(*args, **kwargs)

    monkeypatch.setattr(experts_module, "pad_expert_rows", counted_pad)
    return SimpleNamespace(
        state=state, gemm_splits=gemm_splits, dispatcher_configs=dispatcher_configs, padding_calls=padding_calls
    )


def _capacity_experts(moe_config, **backend_overrides):
    from nemo_automodel.components.moe.experts import GroupedExpertsTE

    backend = BackendConfig(
        experts="te",
        dispatcher="hybridep",
        te_grouped_tensor=True,
        dispatcher_capacity_factor=1.5,
        **backend_overrides,
    )
    experts = GroupedExpertsTE(moe_config, backend=backend)
    mesh = Mock()
    mesh.size.return_value = 1
    mesh.get_local_rank.return_value = 0
    mesh.mesh_dim_names = ("ep",)
    experts.init_token_dispatcher(mesh)
    experts.expert_activation = _activation
    return experts


def _activation(values: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Weighted SwiGLU over [rows, 2 * inter] values with [rows, 1] weights."""
    gate, up = values.chunk(2, -1)
    return F.silu(gate) * up * probs


def _recipe(mxfp8: bool) -> Mock:
    recipe = Mock()
    recipe.mxfp8.return_value = mxfp8
    recipe.float8_block_scaling.return_value = False
    return recipe


@pytest.mark.parametrize(
    "active_recipe,fp8_configured,dispatch_pad,triton_align",
    [
        ("bf16", False, None, None),  # no quantization: raw counts, no padding anywhere
        ("mxfp8", True, 128, None),  # FP8 configured: HybridEP pads, no padding copy
        ("current", True, 128, None),  # 16-row recipe: HybridEP's 128-row segments satisfy it
        ("mxfp8", False, None, 128),  # FP8 autocast the config did not announce: the Triton padding aligns
        ("current", False, None, 16),
    ],
)
@pytest.mark.parametrize("after_down", [False, True])
def test_capacity_buffer_matches_per_token_experts(
    moe_config, fake_te, active_recipe, fp8_configured, dispatch_pad, triton_align, after_down
):
    moe_config.apply_router_weight_after_down = after_down
    experts = _capacity_experts(moe_config, te_fp8={"recipe": "mxfp8"} if fp8_configured else None)
    assert experts.dispatch_pad_multiple == dispatch_pad
    assert fake_te.dispatcher_configs[-1].moe_router_expert_pad_multiple == dispatch_pad
    if active_recipe != "bf16":
        fake_te.state.is_fp8_enabled.return_value = True
        fake_te.state.get_fp8_recipe.return_value = _recipe(mxfp8=active_recipe == "mxfp8")

    torch.manual_seed(0)
    tokens = 23
    expert_ids = torch.tensor([2] * 9 + [0] * 3 + [3] * 11)[torch.randperm(tokens)]  # expert 1 receives nothing
    x = torch.randn(tokens, moe_config.dim, requires_grad=True)
    weights = torch.rand(tokens, 1, requires_grad=True)
    out = experts(x, torch.ones(tokens, dtype=torch.bool), weights, expert_ids[:, None])

    x_ref = x.detach().clone().requires_grad_(True)
    weights_ref = weights.detach().clone().requires_grad_(True)
    w1 = experts.gate_up_linear.weight.detach().clone().requires_grad_(True)
    w2 = experts.down_linear.weight.detach().clone().requires_grad_(True)
    hidden = torch.einsum("td,tod->to", x_ref, w1[expert_ids])
    hidden = _activation(hidden, torch.ones_like(weights_ref) if after_down else weights_ref)
    expected = torch.einsum("ti,toi->to", hidden, w2[expert_ids])
    if after_down:
        expected = expected * weights_ref

    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(weights.grad, weights_ref.grad)
    torch.testing.assert_close(experts.gate_up_linear.weight.grad, w1.grad)
    torch.testing.assert_close(experts.down_linear.weight.grad, w2.grad)
    assert not experts.gate_up_linear.weight.grad[1].any() and not experts.down_linear.weight.grad[1].any()

    counts = torch.tensor([3, 0, 9, 11])
    align = dispatch_pad or triton_align or 1
    assert fake_te.padding_calls == ([triton_align] * 2 if triton_align else [])
    for splits in fake_te.gemm_splits:
        assert splits.tolist() == ((counts + align - 1) // align * align).tolist()


def test_capacity_settings_reach_the_token_dispatcher(moe_config, fake_te):
    _capacity_experts(moe_config, dispatcher_equal_token_counts=True)
    (config,) = fake_te.dispatcher_configs
    assert config.moe_hybridep_capacity_factor == 1.5
    assert config.moe_hybridep_equal_token_counts is True
    assert config.moe_router_expert_pad_multiple is None


def test_capacity_mode_rejects_expert_bias(moe_config, fake_te):
    moe_config.expert_bias = True
    with pytest.raises(ValueError, match="expert_bias"):
        _capacity_experts(moe_config)

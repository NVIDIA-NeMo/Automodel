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

"""Chunked Transformer Engine grouped projections: NVFP4 with more than 64 local experts."""

import sys
from types import ModuleType

import pytest
import torch

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsTE, _ChunkedGroupedLinear, _grouped_parameter


class _FakeGroupedLinear(torch.nn.Module):
    """CPU stand-in for TE ``GroupedLinear``: one ``weight{i}`` per expert, rows split by ``m_splits``."""

    def __init__(self, *, num_gemms, in_features, out_features, bias, params_dtype, device, use_grouped_tensor=False):
        super().__init__()
        self.num_gemms = num_gemms
        self.use_bias = bias
        for index in range(num_gemms):
            weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=params_dtype))
            self.register_parameter(f"weight{index}", weight)

    def forward(self, values, splits):
        assert len(splits) == self.num_gemms and sum(splits) == values.shape[0]
        outputs, row = [], 0
        for index, rows in enumerate(splits):
            outputs.append(torch.nn.functional.linear(values[row : row + rows], getattr(self, f"weight{index}")))
            row += rows
        return torch.cat(outputs)


class _FakePadding(torch.nn.Module):
    def __init__(self, num_gemms, align_size=None):
        super().__init__()


@pytest.fixture
def fake_te(monkeypatch):
    """Stand in for Transformer Engine so the grouped projections run on CPU."""
    te = ModuleType("transformer_engine.pytorch")
    te.GroupedLinear, te.Fp8Padding, te.Fp8Unpadding = _FakeGroupedLinear, _FakePadding, _FakePadding
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te)
    monkeypatch.setattr("nemo_automodel.components.moe.experts.is_te_min_version", lambda version: True)
    monkeypatch.setattr("nemo_automodel.components.models.common.utils._patch_te_modules", lambda: None)


def _moe_config(n_routed_experts: int) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=n_routed_experts,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=8,
        inter_dim=4,
        moe_inter_dim=4,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )


def _chunked(num_gemms: int, in_features: int = 8, out_features: int = 12) -> _ChunkedGroupedLinear:
    return _ChunkedGroupedLinear(
        num_gemms, in_features, out_features, max_gemms=64, bias=False, params_dtype=torch.float32, device="cpu"
    )


@pytest.mark.parametrize("num_gemms", [65, 96, 128, 129, 384])
def test_chunks_cover_the_experts_in_order_within_the_limit(fake_te, num_gemms):
    layer = _chunked(num_gemms)
    sizes = [chunk.num_gemms for chunk in layer.chunks]

    assert len(sizes) == -(-num_gemms // 64)
    assert sum(sizes) == num_gemms
    assert max(sizes) <= 64
    assert max(sizes) - min(sizes) <= 1
    expected = [(chunk, local) for chunk in layer.chunks for local in range(chunk.num_gemms)]
    for index, (chunk, local) in enumerate(expected):
        owner, owner_local = layer.locate(index)
        assert owner is chunk
        assert owner_local == local


@pytest.mark.parametrize(
    "splits",
    [
        [1 + index % 3 for index in range(96)],
        # No rows for the 48 experts of the first chunk.
        [0] * 48 + [index % 4 for index in range(48)],
        # No rows for the 48 experts of the last chunk.
        [index % 4 for index in range(48)] + [0] * 48,
    ],
)
def test_chunked_projection_matches_one_grouped_linear(fake_te, splits):
    torch.manual_seed(0)
    reference = _FakeGroupedLinear(
        num_gemms=96, in_features=8, out_features=12, bias=False, params_dtype=torch.float32, device="cpu"
    )
    chunked = _chunked(96)
    with torch.no_grad():
        for index in range(96):
            _grouped_parameter(chunked, "weight", index).copy_(_grouped_parameter(reference, "weight", index))
    x = torch.randn(sum(splits), 8, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_()

    output = chunked(x, splits)
    expected = reference(x_reference, splits)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    expected.backward(upstream)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("recipe", "n_experts", "chunked"),
    [("nvfp4", 96, True), ("nvfp4", 64, False), ("mxfp8", 96, False), (None, 96, False)],
)
def test_te_experts_chunk_only_nvfp4_above_the_limit(fake_te, recipe, n_experts, chunked):
    te_fp8 = None if recipe is None else {"recipe": recipe}
    backend = BackendConfig(experts="te", dispatcher="deepep", te_fp8=te_fp8)

    experts = GroupedExpertsTE(_moe_config(n_experts), backend=backend)

    for layer in (experts.gate_up_linear, experts.down_linear):
        assert isinstance(layer, _ChunkedGroupedLinear) == chunked
        assert layer.num_gemms == n_experts


def test_te_grouped_tensor_rejects_nvfp4_above_the_limit(fake_te):
    backend = BackendConfig(experts="te", dispatcher="deepep", te_grouped_tensor=True, te_fp8={"recipe": "nvfp4"})

    with pytest.raises(ValueError, match="at most 64 local experts"):
        GroupedExpertsTE(_moe_config(96), backend=backend)


def test_stacked_expert_weights_round_trip_through_chunks(fake_te):
    backend = BackendConfig(experts="te", dispatcher="deepep", te_fp8={"recipe": "nvfp4"})
    config = _moe_config(96)
    experts = GroupedExpertsTE(config, backend=backend)
    stacked = torch.randn(96, config.dim, 2 * config.moe_inter_dim)

    experts.gate_and_up_projs = stacked

    assert torch.equal(experts.gate_and_up_projs, stacked)
    # Local expert 70 is expert 22 of the second 48-expert chunk; TE stores [out, in].
    assert torch.equal(experts.gate_up_linear.chunks[1].weight22, stacked[70].T)

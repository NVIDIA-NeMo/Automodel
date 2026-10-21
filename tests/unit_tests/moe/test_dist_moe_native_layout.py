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

"""Native storage preserves logical weights, updates, and checkpoint aliases."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.dist_moe_experts import GroupedExpertsDistMoE
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.components.moe.layers import _init_weights


def _config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=24,
        moe_inter_dim=24,
        norm_topk_prob=False,
    )


def _experts(dtype: torch.dtype = torch.float32) -> GroupedExpertsDistMoE:
    return GroupedExpertsDistMoE(
        replace(_config(), dtype=dtype),
        BackendConfig(dispatcher="dist_moe", experts="torch_mm", dist_moe={"max_tokens_per_rank": 4}),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_initialization_and_checkpoint_aliases(dtype: torch.dtype) -> None:
    reference = GroupedExperts(replace(_config(), dtype=dtype))
    actual = _experts(dtype)
    torch.manual_seed(1234)
    with torch.no_grad():
        _init_weights(reference, torch.device("cpu"))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(1234)
    _init_weights(actual, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert set(dict(actual.named_parameters())) == {"routed_gate_up_weights", "routed_down_weights"}
    for name, native in (
        ("gate_and_up_projs", actual.routed_gate_up_weights),
        ("down_projs", actual.routed_down_weights),
    ):
        assert native.is_contiguous()
        canonical = actual.state_dict()[name]
        torch.testing.assert_close(canonical, reference.state_dict()[name], rtol=0, atol=0)
        assert canonical.data_ptr() == native.data_ptr()
    restored = _experts(dtype)
    parameters = dict(restored.named_parameters())
    restored.load_state_dict(reference.state_dict())
    for name, parameter in restored.named_parameters():
        assert parameter is parameters[name]
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
    reference.load_state_dict(restored.state_dict())


def test_checkpoint_partial_keys_and_strict_errors() -> None:
    module = _experts()
    module.init_weights(torch.device("cpu"))
    before = module.down_projs.detach().clone()
    gate = torch.full_like(module.gate_and_up_projs, 2)
    missing = module.load_state_dict({"gate_and_up_projs": gate}, strict=False)
    assert missing.missing_keys == ["down_projs"]
    torch.testing.assert_close(module.gate_and_up_projs, gate)
    torch.testing.assert_close(module.down_projs, before)
    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_state_dict({"gate_and_up_projs": gate})
    with pytest.raises(RuntimeError, match="Unexpected key"):
        module.load_state_dict({**module.state_dict(), "wrong": gate})
    with pytest.raises(RuntimeError, match="size mismatch"):
        module.load_state_dict({"gate_and_up_projs": gate[:, :1], "down_projs": before})


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_consumes_native_layout_after_optimizer_update(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    module = _experts(dtype)
    module.init_weights(torch.device("cpu"))
    module.context = object()
    calls = []

    def routed(inputs, ids, scores, gate_up, down, context):
        assert gate_up.is_contiguous() and down.is_contiguous()
        assert gate_up.shape == (2, 48, 16) and down.shape == (2, 16, 24)
        if dtype == torch.bfloat16:
            assert gate_up.data_ptr() == module.routed_gate_up_weights.data_ptr()
            assert down.data_ptr() == module.routed_down_weights.data_ptr()
        calls.append((gate_up.detach().clone(), down.detach().clone()))
        gate, up = F.linear(inputs, gate_up[0]).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, down[0]) * scores[:, :1].to(inputs.dtype)

    package = SimpleNamespace(routed_experts=routed)
    monkeypatch.setattr("nemo_automodel.components.moe.dist_moe_experts.safe_import", lambda *a, **k: (True, package))
    optimizer = torch.optim.SGD(module.parameters(), lr=1)
    hidden = torch.randn(4, 16, dtype=torch.bfloat16)
    scores = torch.ones(4, 1)
    mask = torch.ones(4, dtype=torch.bool)
    indices = torch.zeros(4, 1, dtype=torch.int64)
    for _ in range(2):
        optimizer.zero_grad()
        expected_up = module.gate_and_up_projs.detach().clone().requires_grad_()
        expected_down = module.down_projs.detach().clone().requires_grad_()
        # Canonical matmul is independent of the native-layout fake kernel.
        gate, up = (hidden @ expected_up[0].bfloat16()).chunk(2, dim=-1)
        expected = (F.silu(gate) * up) @ expected_down[0].bfloat16()
        actual = module(hidden, mask, scores, indices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().sum().backward()
        expected.float().sum().backward()
        torch.testing.assert_close(module.routed_gate_up_weights.grad.transpose(-1, -2), expected_up.grad)
        torch.testing.assert_close(module.routed_down_weights.grad.transpose(-1, -2), expected_down.grad)
        optimizer.step()
    assert not torch.equal(calls[0][0], calls[1][0])
    assert not torch.equal(calls[0][1], calls[1][1])

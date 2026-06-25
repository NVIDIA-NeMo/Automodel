# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from unittest.mock import Mock

import pytest
import torch
from peft.tuners.lora.layer import ParamWrapper
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


class _ExpertParams(nn.Module):
    def __init__(self, parameter_name: str, shape: tuple[int, int, int]):
        super().__init__()
        self.register_parameter(parameter_name, nn.Parameter(torch.zeros(shape)))


@pytest.fixture
def adapter():
    return Qwen3MoeStateDictAdapter(
        config=Mock(),
        moe_config=MoEConfig(
            dim=5,
            inter_dim=7,
            moe_inter_dim=7,
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=0.0,
            norm_topk_prob=False,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            activation_alpha=1.702,
            activation_limit=7.0,
            softmax_before_topk=True,
        ),
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        ),
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    (
        "native_a_suffix",
        "native_b_suffix",
        "parameter_name",
        "paramwrapper_prefix",
        "in_features",
        "out_features",
    ),
    [
        ("lora_gate_and_up_A", "lora_gate_and_up_B", "gate_up_proj", "base_layer.", 5, 14),
        ("lora_down_A", "lora_down_B", "down_proj", "", 7, 5),
    ],
)
def test_paramwrapper_export_preserves_expert_lora_delta(
    adapter,
    native_a_suffix,
    native_b_suffix,
    parameter_name,
    paramwrapper_prefix,
    in_features,
    out_features,
):
    n_experts = adapter.moe_config.n_routed_experts
    rank = 3
    native_a = torch.randn(n_experts, in_features, rank)
    native_b = torch.randn(n_experts, rank, out_features)
    base_key = "model.layers.0.mlp.experts"

    converted = {}
    for suffix, tensor in ((native_a_suffix, native_a), (native_b_suffix, native_b)):
        converted.update(adapter._convert_lora_to_paramwrapper(f"{base_key}.{suffix}", tensor))

    wrapper = ParamWrapper(
        _ExpertParams(parameter_name, (n_experts, out_features, in_features)),
        adapter_name="default",
        parameter_name=parameter_name,
        r=rank,
        lora_alpha=rank,
        init_lora_weights=False,
    )
    wrapper.lora_A["default"].weight.data.copy_(converted[f"{base_key}.{paramwrapper_prefix}lora_A.weight"])
    wrapper.lora_B["default"].weight.data.copy_(converted[f"{base_key}.{paramwrapper_prefix}lora_B.weight"])

    expected_delta = torch.einsum("eir,ero->eio", native_a, native_b).transpose(1, 2)
    torch.testing.assert_close(wrapper.get_delta_weight("default"), expected_delta)

    restored = adapter._convert_paramwrapper_to_native(converted)
    torch.testing.assert_close(restored[f"{base_key}.{native_a_suffix}"], native_a)
    torch.testing.assert_close(restored[f"{base_key}.{native_b_suffix}"], native_b)


def test_from_hf_converts_fused_expert_base_weights(adapter):
    n_experts = adapter.moe_config.n_routed_experts
    hidden_size = adapter.moe_config.dim
    inter_size = adapter.moe_config.moe_inter_dim
    base_key = "model.layers.0.mlp.experts"
    gate_up = torch.randn(n_experts, 2 * inter_size, hidden_size)
    down = torch.randn(n_experts, hidden_size, inter_size)

    converted = adapter.from_hf(
        {
            f"{base_key}.gate_up_proj": gate_up,
            f"{base_key}.down_proj": down,
        }
    )

    torch.testing.assert_close(converted[f"{base_key}.gate_and_up_projs"], gate_up.transpose(1, 2))
    torch.testing.assert_close(converted[f"{base_key}.down_projs"], down.transpose(1, 2))


def test_to_hf_emits_fused_expert_base_weights_for_init_load(adapter):
    n_experts = adapter.moe_config.n_routed_experts
    hidden_size = adapter.moe_config.dim
    inter_size = adapter.moe_config.moe_inter_dim
    base_key = "model.layers.0.mlp.experts"
    gate_and_up = torch.randn(n_experts, hidden_size, 2 * inter_size)
    down = torch.randn(n_experts, inter_size, hidden_size)

    converted = adapter.to_hf(
        {
            f"{base_key}.gate_and_up_projs": gate_and_up,
            f"{base_key}.down_projs": down,
        },
        is_init_step=True,
    )

    assert set(converted) == {f"{base_key}.gate_up_proj", f"{base_key}.down_proj"}
    torch.testing.assert_close(converted[f"{base_key}.gate_up_proj"], gate_and_up.transpose(1, 2))
    torch.testing.assert_close(converted[f"{base_key}.down_proj"], down.transpose(1, 2))


def test_to_hf_keeps_legacy_split_expert_base_weights_by_default(adapter):
    n_experts = adapter.moe_config.n_routed_experts
    hidden_size = adapter.moe_config.dim
    inter_size = adapter.moe_config.moe_inter_dim
    base_key = "model.layers.0.mlp.experts"
    gate_and_up = torch.randn(n_experts, hidden_size, 2 * inter_size)
    down = torch.randn(n_experts, inter_size, hidden_size)

    converted = adapter.to_hf(
        {
            f"{base_key}.gate_and_up_projs": gate_and_up,
            f"{base_key}.down_projs": down,
        }
    )

    assert f"{base_key}.gate_up_proj" not in converted
    assert f"{base_key}.down_proj" not in converted
    assert f"{base_key}.0.gate_proj.weight" in converted
    assert f"{base_key}.0.up_proj.weight" in converted
    assert f"{base_key}.0.down_proj.weight" in converted

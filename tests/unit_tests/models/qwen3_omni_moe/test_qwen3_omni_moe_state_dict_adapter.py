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

"""Tests for Qwen3 Omni MoE state dict adapter."""

from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.models.qwen3_omni_moe.state_dict_adapter import Qwen3OmniMoeStateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 32
    cfg.intermediate_size = 64
    cfg.moe_intermediate_size = 32
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.num_experts = 2
    cfg.num_experts_per_tok = 1
    return cfg


@pytest.fixture
def moe_config(config):
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=0,
        n_activated_experts=config.num_experts_per_tok,
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
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class StubQwen3OmniMoeStateDictAdapter(Qwen3OmniMoeStateDictAdapter):
    def convert_single_tensor_to_hf(self, fqn: str, tensor, **kwargs):
        return [(fqn, tensor)]


@pytest.fixture
def adapter(config, moe_config, backend_config):
    return StubQwen3OmniMoeStateDictAdapter(
        config=config, moe_config=moe_config, backend=backend_config, dtype=torch.float32
    )


class TestInitialization:
    def test_sets_default_prefix_flags(self, adapter, config, moe_config, backend_config):
        assert adapter.config is config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend_config
        assert adapter.dtype == torch.float32
        assert adapter._uses_thinker_prefix is True
        assert adapter._uses_model_prefix is True


class TestToHF:
    def test_adds_thinker_prefix(self, adapter):
        state_dict = {"model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(4, 4)}

        with patch.object(adapter, "_to_hf_w_split_experts", return_value=state_dict):
            out = adapter.to_hf(state_dict)

        assert all(key.startswith("thinker.") for key in out.keys())

    def test_respects_exclude_regex(self, adapter):
        state_dict = {"keep.me": torch.ones(1), "drop.me": torch.ones(1)}

        with patch.object(adapter, "_to_hf_w_split_experts", return_value=state_dict):
            out = adapter.to_hf(state_dict, exclude_key_regex=r"^drop")

        assert "drop.me" not in out
        assert "thinker.keep.me" in out


class TestFromHF:
    def test_strips_thinker_prefix_before_merge(self, adapter):
        hf_state = {
            "thinker.model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(4, 4),
            "thinker.model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(4, 4),
            "thinker.model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(4, 4),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts", return_value={}) as mock_from_hf:
            adapter.from_hf(hf_state)

        passed = mock_from_hf.call_args[0][0]
        assert all(not key.startswith("thinker.") for key in passed.keys())
        assert adapter._uses_thinker_prefix is True
        assert adapter._uses_model_prefix is True

    def test_handles_state_without_thinker_prefix(self, adapter):
        hf_state = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(4, 4),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts", return_value={}) as mock_from_hf:
            adapter.from_hf(hf_state)

        passed = mock_from_hf.call_args[0][0]
        assert "model.layers.0.mlp.experts.0.gate_proj.weight" in passed
        assert adapter._uses_thinker_prefix is False

    def test_preserves_non_expert_keys(self, adapter):
        hf_state = {
            "thinker.model.layers.0.mlp.router.weight": torch.randn(4, 4),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts", return_value=hf_state) as mock_from_hf:
            out = adapter.from_hf(hf_state)

        assert "thinker.model.layers.0.mlp.router.weight" in out
        passed = mock_from_hf.call_args[0][0]
        assert "model.layers.0.mlp.router.weight" in passed


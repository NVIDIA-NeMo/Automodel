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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie4_5.state_dict_adapter import (
    Ernie4_5_MoeStateDictAdapter,
    Ernie4_5StateDictAdapter,
)
from nemo_automodel.components.moe.config import MoEConfig


@pytest.fixture
def dense_config():
    return SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )


@pytest.fixture
def moe_hf_config():
    return SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        moe_num_experts=4,
        moe_num_shared_experts=1,
        moe_k=2,
        moe_layer_interval=1,
        moe_layer_start_index=1,
        moe_layer_end_index=3,
        use_bias=False,
    )


@pytest.fixture
def moe_config():
    return MoEConfig(
        dim=64,
        inter_dim=128,
        moe_inter_dim=32,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax_with_bias",
        route_scale=1.0,
        aux_loss_coeff=0.001,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
        force_e_score_correction_bias=True,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class TestErnie4_5StateDictAdapter:
    """Tests for the dense (passthrough) adapter."""

    def test_init_stores_config(self, dense_config):
        adapter = Ernie4_5StateDictAdapter(dense_config)
        assert adapter.config is dense_config

    def test_from_hf_returns_copy(self, dense_config):
        adapter = Ernie4_5StateDictAdapter(dense_config)
        hf_state = {"model.embed_tokens.weight": torch.randn(8, 4)}
        out = adapter.from_hf(hf_state)
        assert out == hf_state
        # Must be a copy, not the same object.
        assert out is not hf_state

    def test_to_hf_passthrough(self, dense_config):
        adapter = Ernie4_5StateDictAdapter(dense_config)
        state = {"model.embed_tokens.weight": torch.randn(8, 4), "lm_head.weight": torch.randn(8, 4)}
        out = adapter.to_hf(state)
        assert set(out.keys()) == set(state.keys())
        assert out is not state

    def test_to_hf_filters_with_regex(self, dense_config):
        adapter = Ernie4_5StateDictAdapter(dense_config)
        state = {
            "model.embed_tokens.weight": torch.randn(2, 2),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(2, 2),
            "lm_head.weight": torch.randn(2, 2),
        }
        out = adapter.to_hf(state, exclude_key_regex=r".*lm_head.*")
        assert "lm_head.weight" not in out
        assert "model.embed_tokens.weight" in out
        assert "model.layers.0.self_attn.q_proj.weight" in out

    def test_convert_single_tensor_to_hf_basic(self, dense_config):
        adapter = Ernie4_5StateDictAdapter(dense_config)
        tensor = torch.randn(4, 4)
        out = adapter.convert_single_tensor_to_hf("model.layers.0.mlp.up_proj.weight", tensor)
        assert len(out) == 1
        assert out[0][0] == "model.layers.0.mlp.up_proj.weight"
        assert out[0][1] is tensor

    def test_convert_single_tensor_to_hf_excluded(self, dense_config):
        adapter = Ernie4_5StateDictAdapter(dense_config)
        tensor = torch.randn(4, 4)
        out = adapter.convert_single_tensor_to_hf("lm_head.weight", tensor, exclude_key_regex=r"lm_head.*")
        assert out == []


class TestErnie4_5_MoeStateDictAdapter:
    """Tests for the MoE adapter."""

    def test_init_stores_attributes(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config, dtype=torch.float32)
        assert adapter.config is moe_hf_config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend_config
        assert adapter.dtype is torch.float32
        assert adapter._uses_model_prefix is True

    def test_hf_key_to_native_renames_moe_statics(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        assert (
            adapter._hf_key_to_native("model.layers.1.mlp.moe_statics.e_score_correction_bias")
            == "model.layers.1.mlp.gate.e_score_correction_bias"
        )

    def test_hf_key_to_native_unchanged_for_unrelated_keys(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        key = "model.layers.0.self_attn.q_proj.weight"
        assert adapter._hf_key_to_native(key) == key

    def test_native_key_to_hf_renames_gate_bias(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        assert (
            adapter._native_key_to_hf("model.layers.1.mlp.gate.e_score_correction_bias")
            == "model.layers.1.mlp.moe_statics.e_score_correction_bias"
        )

    def test_native_key_to_hf_unchanged_for_unrelated_keys(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        key = "model.embed_tokens.weight"
        assert adapter._native_key_to_hf(key) == key

    def test_from_hf_drops_mtp_keys(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        hf_state = {
            "model.embed_tokens.weight": torch.randn(8, 4),
            "model.mtp_block.0.weight": torch.randn(4, 4),
            "model.mtp_emb.weight": torch.randn(4, 4),
        }
        with patch.object(adapter, "_from_hf_w_merged_experts", side_effect=lambda sd, _: sd):
            out = adapter.from_hf(hf_state)
        assert "model.mtp_block.0.weight" not in out
        assert "model.mtp_emb.weight" not in out
        assert "model.embed_tokens.weight" in out

    def test_from_hf_renames_moe_statics(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        hf_state = {
            "model.layers.1.mlp.moe_statics.e_score_correction_bias": torch.zeros(4),
            "model.embed_tokens.weight": torch.randn(2, 2),
        }
        with patch.object(adapter, "_from_hf_w_merged_experts", side_effect=lambda sd, _: sd):
            out = adapter.from_hf(hf_state)
        assert "model.layers.1.mlp.gate.e_score_correction_bias" in out
        assert "model.layers.1.mlp.moe_statics.e_score_correction_bias" not in out

    def test_from_hf_squeezes_2d_bias(self, moe_hf_config, moe_config, backend_config):
        """HF saves e_score_correction_bias as shape (1, n_experts); native expects (n_experts,)."""
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        hf_state = {
            "model.layers.1.mlp.moe_statics.e_score_correction_bias": torch.zeros(1, 4),
        }
        with patch.object(adapter, "_from_hf_w_merged_experts", side_effect=lambda sd, _: sd):
            out = adapter.from_hf(hf_state)
        assert out["model.layers.1.mlp.gate.e_score_correction_bias"].shape == (4,)

    def test_from_hf_keeps_1d_bias_as_is(self, moe_hf_config, moe_config, backend_config):
        """A bias already shaped (n_experts,) should pass through the rename without reshape."""
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        bias = torch.arange(4, dtype=torch.float32)
        hf_state = {
            "model.layers.1.mlp.moe_statics.e_score_correction_bias": bias.clone(),
        }
        with patch.object(adapter, "_from_hf_w_merged_experts", side_effect=lambda sd, _: sd):
            out = adapter.from_hf(hf_state)
        torch.testing.assert_close(out["model.layers.1.mlp.gate.e_score_correction_bias"], bias)

    def test_from_hf_calls_merged_experts(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        hf_state = {"model.embed_tokens.weight": torch.randn(2, 2)}
        sentinel = {"merged": torch.tensor(1.0)}
        mesh = Mock()
        with patch.object(adapter, "_from_hf_w_merged_experts", return_value=sentinel) as mock_merge:
            out = adapter.from_hf(hf_state, device_mesh=mesh)
        mock_merge.assert_called_once()
        assert mock_merge.call_args[0][1] is mesh
        assert out is sentinel

    def test_convert_single_tensor_to_hf_non_expert(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        tensor = torch.randn(4, 4)
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            out = adapter.convert_single_tensor_to_hf("model.layers.0.self_attn.q_proj.weight", tensor)
        assert len(out) == 1
        assert out[0][0] == "model.layers.0.self_attn.q_proj.weight"
        assert out[0][1] is tensor

    def test_convert_single_tensor_to_hf_renames_gate_bias(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        bias = torch.zeros(4)
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            out = adapter.convert_single_tensor_to_hf("model.layers.1.mlp.gate.e_score_correction_bias", bias)
        assert len(out) == 1
        key, value = out[0]
        assert key == "model.layers.1.mlp.moe_statics.e_score_correction_bias"
        # Should be unsqueezed back to 2D (1, n_experts) for HF.
        assert value.shape == (1, 4)

    def test_convert_single_tensor_to_hf_expert_path(self, moe_hf_config, moe_config, backend_config):
        """When merged-expert split returns tensors, names are mapped through _native_key_to_hf."""
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        tensor = torch.randn(4, 64, 32)
        split_pairs = [
            ("model.layers.1.mlp.experts.0.up_proj.weight", torch.randn(32, 64)),
            ("model.layers.1.mlp.experts.0.gate_proj.weight", torch.randn(32, 64)),
        ]
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=split_pairs):
            out = adapter.convert_single_tensor_to_hf("model.layers.1.mlp.experts.gate_and_up_projs", tensor)
        keys = [k for k, _ in out]
        assert keys == [
            "model.layers.1.mlp.experts.0.up_proj.weight",
            "model.layers.1.mlp.experts.0.gate_proj.weight",
        ]

    def test_convert_single_tensor_to_hf_exclude_regex(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        tensor = torch.randn(4, 4)
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            out = adapter.convert_single_tensor_to_hf("lm_head.weight", tensor, exclude_key_regex=r"lm_head.*")
        assert out == []

    def test_to_hf_iterates_all_keys(self, moe_hf_config, moe_config, backend_config):
        adapter = Ernie4_5_MoeStateDictAdapter(moe_hf_config, moe_config, backend_config)
        state = {
            "model.embed_tokens.weight": torch.randn(2, 2),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.zeros(4),
        }
        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts", return_value=None):
            out = adapter.to_hf(state)
        # gate bias should be renamed back to moe_statics in HF
        assert "model.layers.1.mlp.moe_statics.e_score_correction_bias" in out
        assert "model.embed_tokens.weight" in out

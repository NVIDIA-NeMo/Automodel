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

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import (
    NemotronV3StateDictAdapter,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")


class MockNemotronV3Config:
    """Mock configuration for NemotronV3 model."""

    def __init__(self, **overrides):
        self.num_hidden_layers = 2
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.intermediate_size = 512

        for key, value in overrides.items():
            setattr(self, key, value)


class TestNemotronV3StateDictAdapter:
    """Test NemotronV3StateDictAdapter."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=4,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_bias=False,
            expert_activation="relu2",  # NemotronV3 uses relu2
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            enable_deepep=False,
        )

    def test_adapter_init(self, config, moe_config, backend):
        """Test adapter initialization."""
        adapter = NemotronV3StateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend,
            dtype=torch.bfloat16,
        )

        assert adapter.config == config
        assert adapter.moe_config == moe_config
        assert adapter.backend == backend
        assert adapter.dtype == torch.bfloat16
        assert adapter._uses_model_prefix is True

    def test_hf_prefix_property(self, config, moe_config, backend):
        """Test _hf_prefix property returns 'backbone.'"""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._hf_prefix == "backbone."

    def test_expert_path_segment_property(self, config, moe_config, backend):
        """Test _expert_path_segment property returns 'mixer.experts'."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._expert_path_segment == "mixer.experts"

    def test_from_hf_map_structure(self, config, moe_config, backend):
        """Test from_hf_map structure."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        # Check that mapping uses 'mixer.experts' path
        assert "model.layers.{}.mixer.experts.{}.up_proj.weight" in adapter.from_hf_map
        assert "model.layers.{}.mixer.experts.{}.down_proj.weight" in adapter.from_hf_map


class TestNemotronV3AdapterToHf:
    """Test to_hf conversion."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_to_hf_converts_keys(self, config, moe_config, backend):
        """Test to_hf converts internal keys to HF format."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, 256),
            "model.norm.weight": torch.randn(256),
            "lm_head.weight": torch.randn(100, 256),
        }

        hf_state_dict = adapter.to_hf(state_dict)

        # model → backbone
        assert "backbone.embeddings.weight" in hf_state_dict
        # norm → norm_f
        assert "backbone.norm_f.weight" in hf_state_dict
        # lm_head stays the same
        assert "lm_head.weight" in hf_state_dict

    def test_to_hf_exclude_key_regex(self, config, moe_config, backend):
        """Test to_hf with exclude_key_regex."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.embed_tokens.weight": torch.randn(100, 256),
            "exclude_me.weight": torch.randn(10, 10),
        }

        hf_state_dict = adapter.to_hf(state_dict, exclude_key_regex=r"exclude.*")

        assert "backbone.embeddings.weight" in hf_state_dict
        assert "exclude_me.weight" not in hf_state_dict


class TestNemotronV3AdapterFromHf:
    """Test from_hf conversion."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_from_hf_renames_backbone_to_model(self, config, moe_config, backend):
        """Test from_hf renames backbone → model."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "backbone.embeddings.weight": torch.randn(100, 256),
            "backbone.norm_f.weight": torch.randn(256),
            "lm_head.weight": torch.randn(100, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {
                "model.embed_tokens.weight": torch.randn(100, 256),
                "model.norm.weight": torch.randn(256),
                "lm_head.weight": torch.randn(100, 256),
            }

            result = adapter.from_hf(hf_state_dict)

            # Check that _from_hf_w_merged_experts was called with renamed state dict
            call_args = mock_merge.call_args[0][0]
            assert "model.embed_tokens.weight" in call_args
            assert "model.norm.weight" in call_args
            assert "backbone.embeddings.weight" not in call_args
            assert "backbone.norm_f.weight" not in call_args

    def test_from_hf_detects_backbone_prefix(self, config, moe_config, backend):
        """Test from_hf detects backbone prefix from HF checkpoint."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "backbone.layers.0.mixer.experts.0.up_proj.weight": torch.randn(128, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {}
            adapter.from_hf(hf_state_dict)

            # _uses_model_prefix should be False when starting with 'backbone.'
            assert adapter._uses_model_prefix is False

    def test_from_hf_detects_model_prefix(self, config, moe_config, backend):
        """Test from_hf detects model prefix from HF checkpoint."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "model.layers.0.mixer.experts.0.up_proj.weight": torch.randn(128, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {}
            adapter.from_hf(hf_state_dict)

            assert adapter._uses_model_prefix is True


class TestNemotronV3AdapterConvertSingleTensor:
    """Test convert_single_tensor_to_hf method."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_convert_embed_tokens(self, config, moe_config, backend):
        """Test converting embed_tokens weight."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(100, 256)
        fqn = "model.embed_tokens.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "backbone.embeddings.weight"
        assert torch.equal(result[0][1], tensor)

    def test_convert_norm_weight(self, config, moe_config, backend):
        """Test converting norm weight."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256)
        fqn = "model.norm.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "backbone.norm_f.weight"
        assert torch.equal(result[0][1], tensor)

    def test_convert_layer_weight(self, config, moe_config, backend):
        """Test converting layer weight (model → backbone)."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256, 256)
        fqn = "model.layers.0.mixer.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor)

        assert len(result) == 1
        assert result[0][0] == "backbone.layers.0.mixer.weight"
        assert torch.equal(result[0][1], tensor)

    def test_convert_expert_tensor(self, config, moe_config, backend):
        """Test converting merged expert tensor to split experts."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        # For relu2 (non-gated), gate_and_up_projs has shape [n_experts, dim, inter_dim]
        # instead of [n_experts, dim, 2*inter_dim]
        tensor = torch.randn(2, 256, 128)
        fqn = "model.layers.0.mixer.experts.gate_and_up_projs"

        with patch.object(adapter, "_convert_single_merged_expert_to_hf_split_experts") as mock_convert:
            mock_convert.return_value = [
                ("backbone.layers.0.mixer.experts.0.up_proj.weight", torch.randn(128, 256)),
                ("backbone.layers.0.mixer.experts.1.up_proj.weight", torch.randn(128, 256)),
            ]

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            mock_convert.assert_called_once_with(fqn, tensor)
            assert len(result) == 2

    def test_convert_with_exclude_regex(self, config, moe_config, backend):
        """Test converting with exclude regex."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256)
        fqn = "exclude.this.weight"

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

        assert len(result) == 0


class TestNemotronV3AdapterNonGatedExperts:
    """Test adapter handling of non-gated (relu2) experts."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",  # Non-gated activation
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_is_gated_moe_property(self, config, moe_config, backend):
        """Test _is_gated_moe property returns False for relu2."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._is_gated_moe is False

    def test_gated_activation_returns_true(self, config, backend):
        """Test _is_gated_moe returns True for gated activations."""
        moe_config = MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="swiglu",  # Gated activation
            dtype=torch.bfloat16,
        )

        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._is_gated_moe is True


class TestNemotronV3AdapterMixerExperts:
    """Test adapter handling of mixer.experts path (vs mlp.experts)."""

    @pytest.fixture
    def config(self):
        return MockNemotronV3Config()

    @pytest.fixture
    def moe_config(self):
        return MoEConfig(
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            dim=256,
            inter_dim=512,
            moe_inter_dim=128,
            norm_topk_prob=False,
            expert_activation="relu2",
            dtype=torch.bfloat16,
        )

    @pytest.fixture
    def backend(self):
        return BackendConfig()

    def test_expert_path_segment(self, config, moe_config, backend):
        """Test that expert path segment is 'mixer.experts'."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        assert adapter._expert_path_segment == "mixer.experts"

    def test_from_hf_uses_mixer_experts_path(self, config, moe_config, backend):
        """Test that from_hf correctly parses mixer.experts paths."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        # Simulate HF checkpoint with mixer.experts path
        hf_state_dict = {
            "model.layers.0.mixer.experts.0.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mixer.experts.1.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mixer.experts.0.down_proj.weight": torch.randn(256, 128),
            "model.layers.0.mixer.experts.1.down_proj.weight": torch.randn(256, 128),
        }

        with patch.object(adapter, "_validate_expert_availability"):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local", side_effect=lambda x, *args: x):
                    result = adapter._from_hf_w_merged_experts(hf_state_dict)

        # Should have created merged expert tensors with mixer.experts path
        assert "model.layers.0.mixer.experts.gate_and_up_projs" in result
        assert "model.layers.0.mixer.experts.down_projs" in result

    def test_convert_merged_expert_to_hf_split_uses_mixer_path(self, config, moe_config, backend):
        """Test that convert_single_merged_expert uses mixer.experts path."""
        adapter = NemotronV3StateDictAdapter(config, moe_config, backend)

        # For relu2, tensor has shape [n_experts, dim, inter_dim]
        tensor = torch.randn(2, 256, 128)
        fqn = "model.layers.0.mixer.experts.gate_and_up_projs"

        result = adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)

        assert result is not None
        # Should produce up_proj weights with mixer.experts path (non-gated, so no gate_proj)
        keys = [k for k, _ in result]
        assert all("mixer.experts" in k for k in keys)
        # For relu2, only up_proj is created (no gate_proj)
        assert all("up_proj" in k for k in keys)

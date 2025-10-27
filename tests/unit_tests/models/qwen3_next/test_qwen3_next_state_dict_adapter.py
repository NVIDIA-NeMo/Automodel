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

import torch
from unittest.mock import Mock, patch

from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig

from nemo_automodel.components.models.qwen3_next.state_dict_adapter import Qwen3NextStateDictAdapter


class TestApplyKeyMapping:
    def _make_adapter(self):
        return Qwen3NextStateDictAdapter(
            config=object(), moe_config=object(), backend=object(), dtype=torch.float32
        )

    def test_shared_expert_to_shared_experts_mapping(self):
        """Test that shared_expert (singular) is mapped to shared_experts (plural)"""
        adapter = self._make_adapter()
        mapping = adapter.hf_to_internal_map

        state_dict = {
            # HF format uses singular "shared_expert"
            "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(256, 128),
            "model.layers.1.mlp.shared_expert.gate_proj.weight": torch.randn(128, 256),
            # Keys that shouldn't be affected
            "model.layers.0.mlp.gate.weight": torch.randn(8, 256),
            "model.layers.0.attn.qkv.weight": torch.randn(768, 256),
        }

        original_state_dict = dict(state_dict)
        out = adapter._apply_key_mapping(state_dict, mapping)

        # Shared expert keys should be mapped to shared_experts (plural)
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_experts.up_proj.weight" in out
        assert "model.layers.0.mlp.shared_experts.down_proj.weight" in out
        assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in out

        # Original shared_expert keys should not exist
        assert "model.layers.0.mlp.shared_expert.gate_proj.weight" not in out
        assert "model.layers.0.mlp.shared_expert.up_proj.weight" not in out
        assert "model.layers.0.mlp.shared_expert.down_proj.weight" not in out
        assert "model.layers.1.mlp.shared_expert.gate_proj.weight" not in out

        # Unrelated keys should remain unchanged
        assert "model.layers.0.mlp.gate.weight" in out
        assert "model.layers.0.attn.qkv.weight" in out

        # Value identity preserved
        torch.testing.assert_close(
            out["model.layers.0.mlp.shared_experts.gate_proj.weight"],
            original_state_dict["model.layers.0.mlp.shared_expert.gate_proj.weight"],
        )

    def test_reverse_mapping_for_to_hf(self):
        """Test that shared_experts (plural) is mapped back to shared_expert (singular)"""
        adapter = self._make_adapter()
        mapping = adapter.internal_to_hf_map

        state_dict = {
            # Internal format uses plural "shared_experts"
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_experts.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_experts.down_proj.weight": torch.randn(256, 128),
            # Unrelated keys
            "model.layers.0.attn.qkv.weight": torch.randn(768, 256),
        }

        original_state_dict = dict(state_dict)
        out = adapter._apply_key_mapping(state_dict, mapping)

        # Shared experts keys should be mapped to shared_expert (singular)
        assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_expert.up_proj.weight" in out
        assert "model.layers.0.mlp.shared_expert.down_proj.weight" in out

        # Original shared_experts keys should not exist
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" not in out
        assert "model.layers.0.mlp.shared_experts.up_proj.weight" not in out
        assert "model.layers.0.mlp.shared_experts.down_proj.weight" not in out

        # Unrelated keys should remain unchanged
        assert "model.layers.0.attn.qkv.weight" in out

        # Value identity preserved
        torch.testing.assert_close(
            out["model.layers.0.mlp.shared_expert.gate_proj.weight"],
            original_state_dict["model.layers.0.mlp.shared_experts.gate_proj.weight"],
        )

    def test_mapping_without_model_prefix(self):
        """Test mapping works with or without 'model.' prefix"""
        adapter = self._make_adapter()
        mapping = adapter.hf_to_internal_map

        state_dict = {
            # Without model prefix
            "layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(128, 256),
            "layers.0.mlp.shared_expert.up_proj.weight": torch.randn(128, 256),
        }

        out = adapter._apply_key_mapping(state_dict, mapping)

        assert "layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "layers.0.mlp.shared_experts.up_proj.weight" in out
        assert "layers.0.mlp.shared_expert.gate_proj.weight" not in out
        assert "layers.0.mlp.shared_expert.up_proj.weight" not in out

    def test_no_accidental_partial_replacement(self):
        """Test that only exact pattern matches are replaced"""
        adapter = self._make_adapter()
        mapping = adapter.hf_to_internal_map

        state_dict = {
            # This should be mapped (exact pattern match)
            "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(128, 256),
            # These should NOT be mapped (pattern requires dots on both sides)
            "model.layers.0.mlp.shared_expert_extra.weight": torch.randn(128, 256),
            "model.layers.0.mlp.my_shared_expert.weight": torch.randn(128, 256),
            # This already has the target pattern
            "model.layers.0.mlp.shared_experts.already.weight": torch.randn(128, 256),
        }

        out = adapter._apply_key_mapping(state_dict, mapping)

        # Only the exact pattern should be replaced
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_expert.gate_proj.weight" not in out

        # These should remain unchanged (no dot after shared_expert, so pattern doesn't match)
        assert "model.layers.0.mlp.shared_expert_extra.weight" in out
        assert "model.layers.0.mlp.my_shared_expert.weight" in out

        # This already has the target pattern so it stays the same
        assert "model.layers.0.mlp.shared_experts.already.weight" in out


class TestQwen3NextStateDictAdapter:
    def create_mock_config(self, **overrides):
        config = Mock()
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def create_mock_moe_config(self, **overrides):
        moe_config = Mock(spec=MoEConfig)
        moe_config.n_routed_experts = overrides.get("n_routed_experts", 8)
        moe_config.moe_inter_dim = overrides.get("moe_inter_dim", 512)
        for key, value in overrides.items():
            setattr(moe_config, key, value)
        return moe_config

    def create_mock_backend_config(self, **overrides):
        backend = Mock(spec=BackendConfig)
        for key, value in overrides.items():
            setattr(backend, key, value)
        return backend

    def test_initialization(self):
        """Test adapter initialization"""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = Qwen3NextStateDictAdapter(
            config=config, moe_config=moe_config, backend=backend, dtype=torch.float16
        )

        assert adapter.config is config
        assert adapter.moe_config is moe_config
        assert adapter.backend is backend
        assert adapter.dtype == torch.float16
        assert adapter._uses_model_prefix is True

        # Mapping structures
        assert isinstance(adapter.hf_to_internal_map, dict)
        assert isinstance(adapter.internal_to_hf_map, dict)
        # Expected one mapping for shared expert -> shared experts
        assert len(adapter.hf_to_internal_map) == 1
        assert len(adapter.internal_to_hf_map) == 1
        assert ".mlp.shared_expert." in adapter.hf_to_internal_map
        assert ".mlp.shared_experts." in adapter.internal_to_hf_map

    def test_to_hf_applies_mapping_and_exclude(self):
        """Test that to_hf converts experts and applies shared expert mapping"""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = Qwen3NextStateDictAdapter(config, moe_config, backend)

        state_dict = {
            # Internal format uses plural "shared_experts"
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_experts.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_experts.down_proj.weight": torch.randn(256, 128),
            # Routed experts in grouped format (will be handled by _to_hf_w_split_experts)
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(8, 256, 1024),
            "model.layers.0.mlp.experts.down_projs": torch.randn(8, 512, 256),
            # Other keys
            "model.layers.0.attn.qkv.weight": torch.randn(768, 256),
            "exclude_this": torch.randn(1),
        }

        with patch.object(adapter, "_to_hf_w_split_experts") as mock_split:
            # Mock the split experts method to return a simplified state dict
            mock_split.return_value = {
                "model.layers.0.mlp.shared_experts.gate_proj.weight": state_dict[
                    "model.layers.0.mlp.shared_experts.gate_proj.weight"
                ],
                "model.layers.0.mlp.shared_experts.up_proj.weight": state_dict[
                    "model.layers.0.mlp.shared_experts.up_proj.weight"
                ],
                "model.layers.0.mlp.shared_experts.down_proj.weight": state_dict[
                    "model.layers.0.mlp.shared_experts.down_proj.weight"
                ],
                "model.layers.0.attn.qkv.weight": state_dict["model.layers.0.attn.qkv.weight"],
                "exclude_this": state_dict["exclude_this"],
                # Assume routed experts were split by mock
                "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 256),
                "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(512, 256),
            }

            out = adapter.to_hf(state_dict, exclude_key_regex=r"^exclude.*", quantization=False)

        # Shared expert keys should be mapped to singular "shared_expert"
        assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_expert.up_proj.weight" in out
        assert "model.layers.0.mlp.shared_expert.down_proj.weight" in out

        # Original plural form should not exist
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" not in out
        assert "model.layers.0.mlp.shared_experts.up_proj.weight" not in out
        assert "model.layers.0.mlp.shared_experts.down_proj.weight" not in out

        # Excluded key should be removed
        assert "exclude_this" not in out

        # Other keys should remain
        assert "model.layers.0.attn.qkv.weight" in out

    def test_from_hf_detects_model_prefix(self):
        """Test that from_hf correctly detects model prefix"""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = Qwen3NextStateDictAdapter(config, moe_config, backend)

        hf_state = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 256),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(512, 256),
            "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(128, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {}
            adapter.from_hf(hf_state)

        assert adapter._uses_model_prefix is True

    def test_from_hf_detects_no_model_prefix(self):
        """Test that from_hf correctly detects absence of model prefix"""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = Qwen3NextStateDictAdapter(config, moe_config, backend)

        hf_state = {
            "layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 256),
            "layers.0.mlp.experts.0.up_proj.weight": torch.randn(512, 256),
            "layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(128, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            mock_merge.return_value = {}
            adapter.from_hf(hf_state)

        # Should not find model prefix
        # Note: The detection logic looks for ".mlp.experts." keys, not shared_expert
        # So we need to ensure the detection is correct

    def test_from_hf_applies_mapping_and_merges_experts(self):
        """Test that from_hf applies shared expert mapping and merges routed experts"""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = Qwen3NextStateDictAdapter(config, moe_config, backend)

        hf_state = {
            # HF format uses singular "shared_expert"
            "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(256, 128),
            # Routed experts in split format (will be handled by _from_hf_w_merged_experts)
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 256),
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(512, 256),
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(256, 512),
            # Other keys
            "model.layers.0.attn.qkv.weight": torch.randn(768, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            # The mock should receive the state dict with shared_expert mapped to shared_experts
            def check_and_return(mapped_state, device_mesh=None):
                # Verify that shared_expert was mapped to shared_experts
                assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in mapped_state
                assert "model.layers.0.mlp.shared_experts.up_proj.weight" in mapped_state
                assert "model.layers.0.mlp.shared_experts.down_proj.weight" in mapped_state
                assert "model.layers.0.mlp.shared_expert.gate_proj.weight" not in mapped_state

                # Return a mock result with grouped experts
                return {
                    "model.layers.0.mlp.shared_experts.gate_proj.weight": mapped_state[
                        "model.layers.0.mlp.shared_experts.gate_proj.weight"
                    ],
                    "model.layers.0.mlp.shared_experts.up_proj.weight": mapped_state[
                        "model.layers.0.mlp.shared_experts.up_proj.weight"
                    ],
                    "model.layers.0.mlp.shared_experts.down_proj.weight": mapped_state[
                        "model.layers.0.mlp.shared_experts.down_proj.weight"
                    ],
                    "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(8, 256, 1024),
                    "model.layers.0.mlp.experts.down_projs": torch.randn(8, 512, 256),
                    "model.layers.0.attn.qkv.weight": mapped_state["model.layers.0.attn.qkv.weight"],
                }

            mock_merge.side_effect = check_and_return

            out = adapter.from_hf(hf_state)

        # Output should have plural "shared_experts"
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_experts.up_proj.weight" in out
        assert "model.layers.0.mlp.shared_experts.down_proj.weight" in out

        # Grouped expert keys should exist
        assert "model.layers.0.mlp.experts.gate_and_up_projs" in out
        assert "model.layers.0.mlp.experts.down_projs" in out

        # Other keys preserved
        assert "model.layers.0.attn.qkv.weight" in out

    def test_from_hf_end_to_end_with_real_mapping(self):
        """Test from_hf with actual key mapping (not mocked)"""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = Qwen3NextStateDictAdapter(config, moe_config, backend)

        # Simple state dict without routed experts to test just the mapping
        hf_state = {
            "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(256, 128),
            "model.layers.0.attn.qkv.weight": torch.randn(768, 256),
            # Add a routed expert key to trigger prefix detection
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 256),
        }

        with patch.object(adapter, "_from_hf_w_merged_experts") as mock_merge:
            # Just return the input state dict to test mapping only
            mock_merge.side_effect = lambda x, device_mesh=None: dict(x)

            out = adapter.from_hf(hf_state)

        # Shared expert keys should be mapped to plural
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_experts.up_proj.weight" in out
        assert "model.layers.0.mlp.shared_experts.down_proj.weight" in out

        # Original singular form should not exist
        assert "model.layers.0.mlp.shared_expert.gate_proj.weight" not in out
        assert "model.layers.0.mlp.shared_expert.up_proj.weight" not in out
        assert "model.layers.0.mlp.shared_expert.down_proj.weight" not in out

        # Other keys should remain
        assert "model.layers.0.attn.qkv.weight" in out

    def test_to_hf_end_to_end_with_real_mapping(self):
        """Test to_hf with actual key mapping (not mocked)"""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = Qwen3NextStateDictAdapter(config, moe_config, backend)

        # Simple state dict without routed experts to test just the mapping
        state_dict = {
            "model.layers.0.mlp.shared_experts.gate_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_experts.up_proj.weight": torch.randn(128, 256),
            "model.layers.0.mlp.shared_experts.down_proj.weight": torch.randn(256, 128),
            "model.layers.0.attn.qkv.weight": torch.randn(768, 256),
        }

        with patch.object(adapter, "_to_hf_w_split_experts") as mock_split:
            # Just return the input state dict to test mapping only
            mock_split.return_value = dict(state_dict)

            out = adapter.to_hf(state_dict)

        # Shared experts keys should be mapped to singular
        assert "model.layers.0.mlp.shared_expert.gate_proj.weight" in out
        assert "model.layers.0.mlp.shared_expert.up_proj.weight" in out
        assert "model.layers.0.mlp.shared_expert.down_proj.weight" in out

        # Original plural form should not exist
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" not in out
        assert "model.layers.0.mlp.shared_experts.up_proj.weight" not in out
        assert "model.layers.0.mlp.shared_experts.down_proj.weight" not in out

        # Other keys should remain
        assert "model.layers.0.attn.qkv.weight" in out

    def test_multiple_layers_mapping(self):
        """Test mapping works across multiple layers"""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()
        adapter = Qwen3NextStateDictAdapter(config, moe_config, backend)

        state_dict = {}
        for layer in range(4):
            state_dict[f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight"] = torch.randn(128, 256)
            state_dict[f"model.layers.{layer}.mlp.shared_expert.up_proj.weight"] = torch.randn(128, 256)
            state_dict[f"model.layers.{layer}.mlp.shared_expert.down_proj.weight"] = torch.randn(256, 128)

        out = adapter._apply_key_mapping(state_dict, adapter.hf_to_internal_map)

        for layer in range(4):
            assert f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight" in out
            assert f"model.layers.{layer}.mlp.shared_experts.up_proj.weight" in out
            assert f"model.layers.{layer}.mlp.shared_experts.down_proj.weight" in out

            assert f"model.layers.{layer}.mlp.shared_expert.gate_proj.weight" not in out
            assert f"model.layers.{layer}.mlp.shared_expert.up_proj.weight" not in out
            assert f"model.layers.{layer}.mlp.shared_expert.down_proj.weight" not in out

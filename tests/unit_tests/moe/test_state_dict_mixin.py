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
import re

from nemo_automodel.components.moe.state_dict_mixin import MoEStateDictMixin


class MockMoEConfig:
    def __init__(self, n_routed_experts=8, moe_inter_dim=512):
        self.n_routed_experts = n_routed_experts
        self.moe_inter_dim = moe_inter_dim


class MockConfig:
    def __init__(self):
        pass


class MockBackend:
    def __init__(self):
        pass


class MockMoEStateDictMixin(MoEStateDictMixin):
    def __init__(self, n_experts=8, inter_dim=512, dtype=torch.float32, uses_model_prefix=True):
        self.moe_config = MockMoEConfig(n_experts, inter_dim)
        self.config = MockConfig()
        self.backend = MockBackend()
        self.dtype = dtype
        self._uses_model_prefix = uses_model_prefix
        self._last_expert_ids = []


class TestValidateExpertAvailability:
    def test_no_expert_weights_in_state_dict(self):
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {"layers.0.attention.weight": torch.randn(10, 10)}

        mixin._validate_expert_availability(hf_state_dict, 8)

    def test_all_experts_available_no_device_mesh(self):
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        for layer in range(2):
            for expert in range(8):
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        mixin._validate_expert_availability(hf_state_dict, 8)

    def test_missing_experts_no_device_mesh(self):
        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        # Only add experts 0-6, missing expert 7
        for layer in range(2):
            for expert in range(7):  # Missing expert 7
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        with pytest.raises(RuntimeError, match="Expert weights missing from checkpoint"):
            mixin._validate_expert_availability(hf_state_dict, 8)

    def test_without_model_prefix(self):
        mixin = MockMoEStateDictMixin(uses_model_prefix=False)
        hf_state_dict = {}

        # Add experts without "model." prefix
        for layer in range(2):
            for expert in range(8):
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        mixin._validate_expert_availability(hf_state_dict, 8)

    @patch("nemo_automodel.components.moe.state_dict_mixin.get_expert_range_for_rank_from_mesh")
    @patch("nemo_automodel.components.moe.state_dict_mixin.get_submesh")
    def test_with_device_mesh(self, mock_get_submesh, mock_get_expert_range):
        mock_get_expert_range.return_value = (2, 4)  # Only need experts 2-3

        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep"]

        mock_ep_mesh = Mock()
        mock_ep_mesh.get_rank.return_value = 1
        mock_get_submesh.return_value = mock_ep_mesh

        mixin = MockMoEStateDictMixin()
        hf_state_dict = {}

        # Only add experts 2-3 (required for this rank)
        for layer in range(2):
            for expert in [2, 3]:
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
                    hf_state_dict[key] = torch.randn(512, 1024)

        mixin._validate_expert_availability(hf_state_dict, 8, mock_device_mesh)


class TestSplitExpertsWeights:
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_regular_tensor(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin()
        weight = torch.randn(8, 512, 1024)

        result = mixin._split_experts_weights(weight, 8)

        assert len(result) == 8
        assert len(mixin._last_expert_ids) == 8
        assert mixin._last_expert_ids == list(range(8))
        for i, expert_weight in enumerate(result):
            assert expert_weight.shape == (512, 1024)
            assert torch.equal(expert_weight, weight[i])

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_shape_mismatch(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin()
        weight = torch.randn(6, 512, 1024)  # Wrong number of experts

        with pytest.raises(ValueError, match="Expected first dimension to be 8, got 6"):
            mixin._split_experts_weights(weight, 8)

    @patch("nemo_automodel.components.moe.state_dict_mixin.split_experts_weights_dtensor_aware")
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_dtensor(self, mock_is_dtensor, mock_split_dtensor):
        mock_is_dtensor.return_value = True
        mock_split_dtensor.return_value = ([torch.randn(512, 1024), torch.randn(512, 1024)], [2, 3])

        mixin = MockMoEStateDictMixin()
        mock_weight = Mock()

        result = mixin._split_experts_weights(mock_weight, 8)

        assert len(result) == 2
        assert mixin._last_expert_ids == [2, 3]
        mock_split_dtensor.assert_called_once_with(mock_weight, 8)


class TestConcatenateExpertWeights:
    def test_complete_experts_available(self):
        mixin = MockMoEStateDictMixin()

        expert_weights_by_layer = {
            "0": {
                "abstract_key": {
                    0: torch.randn(512, 1024),
                    1: torch.randn(512, 1024),
                    2: torch.randn(512, 1024),
                    3: torch.randn(512, 1024),
                }
            }
        }

        result = mixin._concatenate_expert_weights(expert_weights_by_layer, 4)

        assert result is not None
        assert result.shape == (4, 512, 1024)
        assert "0" not in expert_weights_by_layer  # Should be cleaned up

    def test_incomplete_experts(self):
        mixin = MockMoEStateDictMixin()

        expert_weights_by_layer = {
            "0": {
                "abstract_key": {
                    0: torch.randn(512, 1024),
                    1: torch.randn(512, 1024),
                    # Missing experts 2 and 3
                }
            }
        }

        result = mixin._concatenate_expert_weights(expert_weights_by_layer, 4)

        assert result is None
        assert "0" in expert_weights_by_layer  # Should not be cleaned up

    def test_multiple_layers_first_complete(self):
        mixin = MockMoEStateDictMixin()

        expert_weights_by_layer = {
            "0": {
                "abstract_key1": {
                    0: torch.randn(512, 1024),
                    1: torch.randn(512, 1024),
                },
                "abstract_key2": {
                    0: torch.randn(512, 1024),
                    1: torch.randn(512, 1024),
                }
            }
        }

        result = mixin._concatenate_expert_weights(expert_weights_by_layer, 2)

        assert result is not None
        assert result.shape == (2, 512, 1024)


class TestToHfGroupedExperts:
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_gate_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=4)

        # Mock the _split_experts_weights method
        gate_weights = [torch.randn(512, 1024) for _ in range(4)]
        mixin._split_experts_weights = Mock(return_value=gate_weights)
        mixin._last_expert_ids = [0, 1, 2, 3]

        state_dict = {
            "model.layers.0.mlp.experts.gate_projs": torch.randn(4, 512, 1024),
            "other_weight": torch.randn(10, 10)
        }

        result = mixin._to_hf_grouped_experts(state_dict)

        # Check that gate_proj weights were split correctly
        for expert_id in range(4):
            expected_key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            assert expected_key in result
            assert torch.equal(result[expected_key], gate_weights[expert_id])

        # Check that other weights are preserved
        assert "other_weight" in result

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_up_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=4)

        up_weights = [torch.randn(512, 1024) for _ in range(4)]
        mixin._split_experts_weights = Mock(return_value=up_weights)
        mixin._last_expert_ids = [0, 1, 2, 3]

        state_dict = {
            "model.layers.1.mlp.experts.up_projs": torch.randn(4, 512, 1024),
        }

        result = mixin._to_hf_grouped_experts(state_dict)

        for expert_id in range(4):
            expected_key = f"model.layers.1.mlp.experts.{expert_id}.up_proj.weight"
            assert expected_key in result

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_down_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=4)

        down_weights = [torch.randn(1024, 512) for _ in range(4)]
        mixin._split_experts_weights = Mock(return_value=down_weights)
        mixin._last_expert_ids = [0, 1, 2, 3]

        state_dict = {
            "model.layers.2.mlp.experts.down_projs": torch.randn(4, 1024, 512),
        }

        result = mixin._to_hf_grouped_experts(state_dict)

        for expert_id in range(4):
            expected_key = f"model.layers.2.mlp.experts.{expert_id}.down_proj.weight"
            assert expected_key in result

    @patch("nemo_automodel.components.moe.state_dict_mixin.validate_dtensor_expert_sharding")
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_dtensor_validation(self, mock_is_dtensor, mock_validate):
        mock_is_dtensor.return_value = True

        mixin = MockMoEStateDictMixin(n_experts=4)

        gate_weights = [torch.randn(512, 1024) for _ in range(4)]
        mixin._split_experts_weights = Mock(return_value=gate_weights)
        mixin._last_expert_ids = [0, 1, 2, 3]

        mock_dtensor = Mock()
        state_dict = {
            "model.layers.0.mlp.experts.gate_projs": mock_dtensor,
        }

        result = mixin._to_hf_grouped_experts(state_dict)

        mock_validate.assert_called_once_with(mock_dtensor, 4, "gate_projs layer 0")

    def test_without_model_prefix(self):
        mixin = MockMoEStateDictMixin(n_experts=4, uses_model_prefix=False)

        with patch.object(mixin, '_split_experts_weights') as mock_split:
            gate_weights = [torch.randn(512, 1024) for _ in range(4)]
            mock_split.return_value = gate_weights
            mixin._last_expert_ids = [0, 1, 2, 3]

            state_dict = {
                "model.layers.0.mlp.experts.gate_projs": torch.randn(4, 512, 1024),
            }

            with patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor", return_value=False):
                result = mixin._to_hf_grouped_experts(state_dict)

            # Without model prefix, keys should not have "model."
            for expert_id in range(4):
                expected_key = f"layers.0.mlp.experts.{expert_id}.gate_proj.weight"
                assert expected_key in result


class TestToHfDeepEP:
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_gate_and_up_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        # Create mock gate_and_up tensor [n_experts, dim, 2*inter_dim]
        gate_and_up_weights = [torch.randn(1024, 1024) for _ in range(2)]  # [dim, 2*inter_dim]
        mixin._split_experts_weights = Mock(return_value=gate_and_up_weights)
        mixin._last_expert_ids = [0, 1]

        state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(2, 1024, 1024),
        }

        result = mixin._to_hf_deepep(state_dict)

        # Check that gate_proj and up_proj weights were created
        for expert_id in range(2):
            gate_key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            up_key = f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
            assert gate_key in result
            assert up_key in result
            assert result[gate_key].shape == (512, 1024)  # [inter_dim, dim]
            assert result[up_key].shape == (512, 1024)    # [inter_dim, dim]

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_down_projs_conversion(self, mock_is_dtensor):
        mock_is_dtensor.return_value = False

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        # Create mock down tensor [n_experts, inter_dim, dim]
        down_weights = [torch.randn(512, 1024) for _ in range(2)]  # [inter_dim, dim]
        mixin._split_experts_weights = Mock(return_value=down_weights)
        mixin._last_expert_ids = [0, 1]

        state_dict = {
            "model.layers.0.mlp.experts.down_projs": torch.randn(2, 512, 1024),
        }

        result = mixin._to_hf_deepep(state_dict)

        # Check that down_proj weights were transposed correctly
        for expert_id in range(2):
            down_key = f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"
            assert down_key in result
            assert result[down_key].shape == (1024, 512)  # [dim, inter_dim]

    @patch("nemo_automodel.components.moe.state_dict_mixin.validate_dtensor_expert_sharding")
    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_dtensor_validation(self, mock_is_dtensor, mock_validate):
        mock_is_dtensor.return_value = True

        mixin = MockMoEStateDictMixin(n_experts=2, inter_dim=512)

        gate_and_up_weights = [torch.randn(1024, 1024) for _ in range(2)]
        mixin._split_experts_weights = Mock(return_value=gate_and_up_weights)
        mixin._last_expert_ids = [0, 1]

        mock_dtensor = Mock()
        state_dict = {
            "model.layers.0.mlp.experts.gate_and_up_projs": mock_dtensor,
        }

        result = mixin._to_hf_deepep(state_dict)

        mock_validate.assert_called_once_with(mock_dtensor, 2, "gate_and_up_projs layer 0")


class TestFromHfGroupedExperts:
    @patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local")
    @patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank")
    def test_basic_conversion(self, mock_should_load, mock_create_dtensor):
        mock_should_load.return_value = True
        mock_create_dtensor.side_effect = lambda x, *args: x  # Return local tensor as-is

        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {}
        # Add gate_proj weights for 2 experts in layer 0
        for expert_id in range(2):
            key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.randn(512, 1024)

        with patch.object(mixin, '_validate_expert_availability'):
            result = mixin._from_hf_grouped_experts(hf_state_dict)

        # Check that gate_projs tensor was created
        expected_key = "model.layers.0.mlp.experts.gate_projs"
        assert expected_key in result
        assert result[expected_key].shape == (2, 512, 1024)

    def test_partial_expert_loading(self):
        # Test that the method respects should_load_expert_for_rank filtering
        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {}
        for expert_id in range(2):
            key = f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.randn(512, 1024)

        with patch.object(mixin, '_validate_expert_availability'):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank") as mock_should_load:
                mock_should_load.side_effect = lambda expert_id, *args: expert_id == 1  # Only load expert 1
                with patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local", side_effect=lambda x, *args: x):
                    result = mixin._from_hf_grouped_experts(hf_state_dict)

        # When only partial experts are loaded, no tensor should be created until all are available
        # This is the expected behavior based on the code logic
        expected_key = "model.layers.0.mlp.experts.gate_projs"
        assert expected_key not in result  # No tensor created because we don't have all expected experts

    def test_without_model_prefix(self):
        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {}
        # Add weights without "model." prefix
        for expert_id in range(2):
            key = f"layers.0.mlp.experts.{expert_id}.gate_proj.weight"
            hf_state_dict[key] = torch.randn(512, 1024)

        with patch.object(mixin, '_validate_expert_availability'):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local", side_effect=lambda x, *args: x):
                    result = mixin._from_hf_grouped_experts(hf_state_dict)

        expected_key = "model.layers.0.mlp.experts.gate_projs"
        assert expected_key in result

    @patch("nemo_automodel.components.moe.state_dict_mixin.get_expert_range_for_rank_from_mesh")
    @patch("nemo_automodel.components.moe.state_dict_mixin.get_submesh")
    def test_with_device_mesh(self, mock_get_submesh, mock_get_expert_range):
        mock_get_expert_range.return_value = (0, 1)  # Only expert 0 for this rank

        mock_device_mesh = Mock()
        mock_device_mesh.mesh_dim_names = ["ep"]

        mock_ep_mesh = Mock()
        mock_ep_mesh.get_rank.return_value = 0
        mock_get_submesh.return_value = mock_ep_mesh

        mixin = MockMoEStateDictMixin(n_experts=2, dtype=torch.float32)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 1024)
        }

        with patch.object(mixin, '_validate_expert_availability'):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local", side_effect=lambda x, *args: x):
                    result = mixin._from_hf_grouped_experts(hf_state_dict, mock_device_mesh)

        expected_key = "model.layers.0.mlp.experts.gate_projs"
        assert expected_key in result
        assert result[expected_key].shape == (1, 512, 1024)


class TestFromHfDeepEP:
    @patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local")
    @patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank")
    def test_gate_and_up_combination(self, mock_should_load, mock_create_dtensor):
        mock_should_load.return_value = True
        mock_create_dtensor.side_effect = lambda x, *args: x

        mixin = MockMoEStateDictMixin(n_experts=1, inter_dim=512, dtype=torch.float32)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(512, 1024),  # [inter_dim, dim]
            "model.layers.0.mlp.experts.0.up_proj.weight": torch.randn(512, 1024),    # [inter_dim, dim]
        }

        with patch.object(mixin, '_validate_expert_availability'):
            result = mixin._from_hf_deepep(hf_state_dict)

        # Should create gate_and_up_projs tensor
        expected_key = "model.layers.0.mlp.experts.gate_and_up_projs"
        assert expected_key in result
        assert result[expected_key].shape == (1, 1024, 1024)  # [n_experts, dim, 2*inter_dim]

    @patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local")
    @patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank")
    def test_down_proj_transpose(self, mock_should_load, mock_create_dtensor):
        mock_should_load.return_value = True
        mock_create_dtensor.side_effect = lambda x, *args: x

        mixin = MockMoEStateDictMixin(n_experts=1, inter_dim=512, dtype=torch.float32)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(1024, 512),  # [dim, inter_dim]
        }

        with patch.object(mixin, '_validate_expert_availability'):
            result = mixin._from_hf_deepep(hf_state_dict)

        # Should create transposed down_projs tensor
        expected_key = "model.layers.0.mlp.experts.down_projs"
        assert expected_key in result
        assert result[expected_key].shape == (1, 512, 1024)  # [n_experts, inter_dim, dim]

    @patch("nemo_automodel.components.moe.state_dict_mixin.is_dtensor")
    def test_dtensor_input_handling(self, mock_is_dtensor):
        # Test when input tensors are already DTensors
        mock_is_dtensor.return_value = True

        mixin = MockMoEStateDictMixin(n_experts=1, inter_dim=512, dtype=torch.float32)

        # Mock DTensor inputs
        mock_gate_dtensor = Mock()
        mock_gate_dtensor.to_local.return_value = torch.randn(512, 1024)
        mock_up_dtensor = Mock()
        mock_up_dtensor.to_local.return_value = torch.randn(512, 1024)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": mock_gate_dtensor,
            "model.layers.0.mlp.experts.0.up_proj.weight": mock_up_dtensor,
        }

        with patch.object(mixin, '_validate_expert_availability'):
            with patch("nemo_automodel.components.moe.state_dict_mixin.should_load_expert_for_rank", return_value=True):
                with patch("nemo_automodel.components.moe.state_dict_mixin.create_dtensor_from_local", side_effect=lambda x, *args: x):
                    result = mixin._from_hf_deepep(hf_state_dict)

        # Verify to_local was called on DTensor inputs
        mock_gate_dtensor.to_local.assert_called_once()
        mock_up_dtensor.to_local.assert_called_once()

    def test_skip_scale_inv_keys(self):
        mixin = MockMoEStateDictMixin()

        hf_state_dict = {
            "some_weight": torch.randn(10, 10),
            "some_weight_scale_inv": torch.randn(10),  # Should be skipped
        }

        with patch.object(mixin, '_validate_expert_availability'):
            result = mixin._from_hf_deepep(hf_state_dict)

        assert "some_weight" in result
        assert "some_weight_scale_inv" not in result

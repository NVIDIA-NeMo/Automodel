# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for the parallelization strategy pattern."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
from abc import ABC

import pytest
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel

# Import the components under test
from nemo_automodel.components.distributed.parallelizer import (
    ParallelizationStrategy,
    DefaultParallelizationStrategy,
    NemotronHParallelizationStrategy,
    PARALLELIZATION_STRATEGIES,
    _DEFAULT_STRATEGY,
    get_parallelization_strategy,
    fsdp2_strategy_parallelize,
)


class MockModel(nn.Module):
    """Mock model for testing purposes."""

    def __init__(self, model_name="MockModel", num_attention_heads=8, num_key_value_heads=8):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )

        # Create mock model structure
        class MockInnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    self._create_mock_layer() for _ in range(2)
                ])

            def _create_mock_layer(self):
                """Create a mock transformer layer."""
                layer = nn.Module()
                layer.mlp = nn.Linear(10, 10)
                return layer

        self.model = MockInnerModel()

        # Set the class name for strategy selection
        self.__class__.__name__ = model_name

    def forward(self, x):
        return x


class MockNemotronHModel(nn.Module):
    """Mock NemotronH model for testing."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=8,
            num_key_value_heads=8,
        )

        # Create backbone structure specific to NemotronH
        class MockBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    self._create_mock_layer() for _ in range(2)
                ])

            def _create_mock_layer(self):
                layer = nn.Module()
                # Use setattr to avoid linter issues with dynamic attributes
                setattr(layer, 'block_type', "mlp")  # Set block type for NemotronH
                layer.mixer = nn.Module()
                layer.mixer.up_proj = nn.Linear(10, 10)
                layer.mixer.down_proj = nn.Linear(10, 10)
                return layer

        self.backbone = MockBackbone()
        self.__class__.__name__ = "NemotronHForCausalLM"

    def forward(self, x):
        return x


@pytest.fixture
def mock_device_mesh():
    """Create a mock device mesh for testing."""
    mesh = MagicMock(spec=DeviceMesh)
    mesh.device_type = "cuda"

    # Mock submeshes
    dp_replicate_mesh = MagicMock()
    dp_shard_mesh = MagicMock()
    tp_mesh = MagicMock()

    dp_replicate_mesh.size.return_value = 1
    dp_shard_mesh.size.return_value = 2
    tp_mesh.size.return_value = 1

    dp_replicate_mesh.ndim = 1
    dp_shard_mesh.ndim = 1
    tp_mesh.ndim = 1

    # Configure mesh access
    mesh.__getitem__.side_effect = lambda key: {
        "dp_replicate": dp_replicate_mesh,
        "dp_shard_cp": dp_shard_mesh,
        "tp": tp_mesh,
        ("dp_replicate", "dp_shard_cp"): dp_shard_mesh,  # Combined mesh
    }[key]

    return mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh


@pytest.fixture
def mock_distributed_env(monkeypatch):
    """Mock the distributed environment for strategy tests."""
    # Mock FSDP functions
    fully_shard_mock = MagicMock(side_effect=lambda model, **kwargs: model)
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.fully_shard",
        fully_shard_mock, raising=False
    )

    # Mock tensor parallel functions
    parallelize_module_mock = MagicMock()
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.parallelize_module",
        parallelize_module_mock, raising=False
    )

    # Mock checkpoint wrapper
    checkpoint_wrapper_mock = MagicMock(side_effect=lambda x: x)
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.checkpoint_wrapper",
        checkpoint_wrapper_mock, raising=False
    )

    # Mock apply_fsdp2_sharding_recursively
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively",
        apply_fsdp_mock, raising=False
    )

    # Mock _extract_model_layers
    extract_layers_mock = MagicMock(return_value=[])
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer._extract_model_layers",
        extract_layers_mock, raising=False
    )

    # Mock _get_parallel_plan
    get_plan_mock = MagicMock(return_value={"test.layer": ColwiseParallel()})
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer._get_parallel_plan",
        get_plan_mock, raising=False
    )

    # Mock validate_tp_mesh
    validate_tp_mock = MagicMock()
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.validate_tp_mesh",
        validate_tp_mock, raising=False
    )

    return {
        "fully_shard": fully_shard_mock,
        "parallelize_module": parallelize_module_mock,
        "checkpoint_wrapper": checkpoint_wrapper_mock,
        "apply_fsdp": apply_fsdp_mock,
        "extract_layers": extract_layers_mock,
        "get_plan": get_plan_mock,
        "validate_tp": validate_tp_mock,
    }


class TestParallelizationStrategy:
    """Test the abstract ParallelizationStrategy base class."""

    def test_is_abstract(self):
        """Test that ParallelizationStrategy is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ParallelizationStrategy()  # type: ignore

    def test_has_abstract_parallelize_method(self):
        """Test that the parallelize method is abstract."""
        assert hasattr(ParallelizationStrategy, 'parallelize')
        assert getattr(ParallelizationStrategy.parallelize, '__isabstractmethod__', False)

    def test_inherits_from_abc(self):
        """Test that ParallelizationStrategy inherits from ABC."""
        assert issubclass(ParallelizationStrategy, ABC)


class TestDefaultParallelizationStrategy:
    """Test the DefaultParallelizationStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create a DefaultParallelizationStrategy instance."""
        return DefaultParallelizationStrategy()

    def test_can_be_instantiated(self, strategy):
        """Test that DefaultParallelizationStrategy can be instantiated."""
        assert isinstance(strategy, DefaultParallelizationStrategy)
        assert isinstance(strategy, ParallelizationStrategy)

    def test_parallelize_method_signature(self, strategy):
        """Test that parallelize method has the correct signature."""
        method = strategy.parallelize
        assert callable(method)

        # Check that all required parameters are supported
        import inspect
        sig = inspect.signature(method)
        required_params = [
            'model', 'device_mesh', 'mp_policy', 'offload_policy',
            'sequence_parallel', 'activation_checkpointing', 'tp_shard_plan',
            'dp_replicate_mesh_name', 'dp_shard_cp_mesh_name', 'tp_mesh_name'
        ]

        for param in required_params:
            assert param in sig.parameters

    def test_parallelize_basic_flow(self, strategy, mock_device_mesh, mock_distributed_env):
        """Test the basic parallelization flow of DefaultParallelizationStrategy."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        model = MockModel()

        # Call the strategy
        result = strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        # Verify the strategy was called correctly
        assert result is model  # Should return the same model

        # Verify key functions were called
        mock_distributed_env["extract_layers"].assert_called_once_with(model)
        mock_distributed_env["apply_fsdp"].assert_called_once()
        mock_distributed_env["fully_shard"].assert_called()

    def test_parallelize_with_tensor_parallel(self, strategy, mock_device_mesh, mock_distributed_env):
        """Test parallelization with tensor parallelism enabled."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2  # Enable TP

        model = MockModel()

        result = strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        # Should call validate_tp_mesh, _get_parallel_plan, and parallelize_module
        mock_distributed_env["validate_tp"].assert_called_once_with(model, tp_mesh)
        mock_distributed_env["get_plan"].assert_called_once()
        mock_distributed_env["parallelize_module"].assert_called_once()

    def test_parallelize_with_activation_checkpointing(self, strategy, mock_device_mesh, mock_distributed_env):
        """Test parallelization with activation checkpointing enabled."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh

        # Mock layers with MLP
        mock_layer = MagicMock()
        mock_layer.mlp = nn.Linear(10, 10)
        mock_distributed_env["extract_layers"].return_value = [mock_layer]

        model = MockModel()

        result = strategy.parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=True,
        )

        # Should apply checkpoint wrapper to MLP layers
        mock_distributed_env["checkpoint_wrapper"].assert_called_with(mock_layer.mlp)

    def test_parallelize_with_custom_mesh_names(self, strategy, mock_device_mesh, mock_distributed_env):
        """Test parallelization with custom mesh names."""
        mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh = mock_device_mesh

        # Update mesh mock to support custom names
        mesh.__getitem__.side_effect = lambda key: {
            "custom_dp_replicate": dp_replicate_mesh,
            "custom_dp_shard": dp_shard_mesh,
            "custom_tp": tp_mesh,
            ("custom_dp_replicate", "custom_dp_shard"): dp_shard_mesh,
        }[key]

        model = MockModel()

        result = strategy.parallelize(
            model=model,
            device_mesh=mesh,
            dp_replicate_mesh_name="custom_dp_replicate",
            dp_shard_cp_mesh_name="custom_dp_shard",
            tp_mesh_name="custom_tp",
        )

        # Verify mesh access used custom names
        expected_calls = [
            call("custom_tp"),
            call(("custom_dp_replicate", "custom_dp_shard")),
        ]
        mesh.__getitem__.assert_has_calls(expected_calls, any_order=True)


class TestNemotronHParallelizationStrategy:
    """Test the NemotronHParallelizationStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create a NemotronHParallelizationStrategy instance."""
        return NemotronHParallelizationStrategy()

    @pytest.fixture
    def nemotron_model(self):
        """Create a mock NemotronH model."""
        return MockNemotronHModel()

    def test_can_be_instantiated(self, strategy):
        """Test that NemotronHParallelizationStrategy can be instantiated."""
        assert isinstance(strategy, NemotronHParallelizationStrategy)
        assert isinstance(strategy, ParallelizationStrategy)

    def test_sequence_parallel_not_supported(self, strategy, mock_device_mesh, nemotron_model):
        """Test that sequence parallelism raises assertion error."""
        mesh, _, _, _ = mock_device_mesh

        with pytest.raises(AssertionError, match="Sequence parallelism is not supported"):
            strategy.parallelize(
                model=nemotron_model,
                device_mesh=mesh,
                sequence_parallel=True,
            )

    def test_custom_tp_plan_not_supported(self, strategy, mock_device_mesh, nemotron_model):
        """Test that custom TP plans raise assertion error."""
        mesh, _, _, _ = mock_device_mesh

        with pytest.raises(AssertionError, match="Custom parallel plan is not supported"):
            strategy.parallelize(
                model=nemotron_model,
                device_mesh=mesh,
                tp_shard_plan={"test": ColwiseParallel()},
            )

    @patch("nemo_automodel.components.distributed.parallelizer.parallelize_module")
    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_nemotron_specific_parallelization(self, mock_fully_shard, mock_parallelize_module,
                                             strategy, mock_device_mesh, nemotron_model):
        """Test NemotronH-specific parallelization logic."""
        mesh, _, dp_shard_mesh, tp_mesh = mock_device_mesh
        mock_fully_shard.side_effect = lambda model, **kwargs: model

        result = strategy.parallelize(
            model=nemotron_model,
            device_mesh=mesh,
            activation_checkpointing=False,
        )

        # Should call parallelize_module for model-level TP plan
        assert mock_parallelize_module.call_count >= 1

        # Should call parallelize_module for each MLP layer
        expected_calls = len([layer for layer in nemotron_model.backbone.layers if layer.block_type == "mlp"])
        assert mock_parallelize_module.call_count == expected_calls + 1  # +1 for model level

        # Should call fully_shard for each layer and the root model
        expected_fully_shard_calls = len(nemotron_model.backbone.layers) + 1  # +1 for root
        assert mock_fully_shard.call_count == expected_fully_shard_calls

    @patch("nemo_automodel.components.distributed.parallelizer.checkpoint_wrapper")
    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    @patch("nemo_automodel.components.distributed.parallelizer.parallelize_module")
    def test_activation_checkpointing(self, mock_parallelize, mock_fully_shard, mock_checkpoint,
                                    strategy, mock_device_mesh, nemotron_model):
        """Test activation checkpointing for NemotronH models."""
        mesh, _, dp_shard_mesh, tp_mesh = mock_device_mesh
        mock_fully_shard.side_effect = lambda model, **kwargs: model
        mock_checkpoint.side_effect = lambda x: x

        # Add a mamba layer to test mamba checkpointing
        mamba_layer = nn.Module()
        setattr(mamba_layer, 'block_type', "mamba")
        nemotron_model.backbone.layers.append(mamba_layer)

        result = strategy.parallelize(
            model=nemotron_model,
            device_mesh=mesh,
            activation_checkpointing=True,
        )

        # Should apply checkpoint wrapper to both MLP and Mamba layers
        expected_checkpoint_calls = 3  # 2 MLP (from MockNemotronHModel) + 1 Mamba layer
        assert mock_checkpoint.call_count == expected_checkpoint_calls


class TestStrategyRegistry:
    """Test the strategy registry functionality."""

    def test_registry_contains_nemotron_strategy(self):
        """Test that the registry contains NemotronH strategy."""
        assert "NemotronHForCausalLM" in PARALLELIZATION_STRATEGIES
        assert isinstance(PARALLELIZATION_STRATEGIES["NemotronHForCausalLM"], NemotronHParallelizationStrategy)

    def test_default_strategy_exists(self):
        """Test that the default strategy exists."""
        assert _DEFAULT_STRATEGY is not None
        assert isinstance(_DEFAULT_STRATEGY, DefaultParallelizationStrategy)

    def test_get_parallelization_strategy_for_nemotron(self):
        """Test strategy selection for NemotronH model."""
        model = MockNemotronHModel()
        strategy = get_parallelization_strategy(model)

        assert isinstance(strategy, NemotronHParallelizationStrategy)

    def test_get_parallelization_strategy_for_regular_model(self):
        """Test strategy selection for regular models."""
        model = MockModel("RegularModel")
        strategy = get_parallelization_strategy(model)

        assert isinstance(strategy, DefaultParallelizationStrategy)
        assert strategy is _DEFAULT_STRATEGY

    def test_get_parallelization_strategy_unknown_model(self):
        """Test strategy selection for unknown model types."""
        model = MockModel("UnknownModelType")
        strategy = get_parallelization_strategy(model)

        assert isinstance(strategy, DefaultParallelizationStrategy)
        assert strategy is _DEFAULT_STRATEGY


class TestFsdp2StrategyParallelizeIntegration:
    """Test the main fsdp2_strategy_parallelize function with the new strategy pattern."""

    def test_delegates_to_strategy(self, mock_device_mesh, mock_distributed_env):
        """Test that fsdp2_strategy_parallelize delegates to the appropriate strategy."""
        mesh, _, _, _ = mock_device_mesh

        # Test with regular model (should use default strategy)
        model = MockModel("RegularModel")

        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            sequence_parallel=False,
            activation_checkpointing=False,
        )

        assert result is model
        # Verify that default strategy functions were called
        mock_distributed_env["extract_layers"].assert_called_once_with(model)

    def test_delegates_to_nemotron_strategy(self, mock_device_mesh):
        """Test that fsdp2_strategy_parallelize uses NemotronH strategy for NemotronH models."""
        mesh, _, _, _ = mock_device_mesh

        with patch("nemo_automodel.components.distributed.parallelizer.parallelize_module"):
            with patch("nemo_automodel.components.distributed.parallelizer.fully_shard", side_effect=lambda model, **kwargs: model):
                model = MockNemotronHModel()

                result = fsdp2_strategy_parallelize(
                    model=model,
                    device_mesh=mesh,
                    sequence_parallel=False,
                    activation_checkpointing=False,
                )

                assert result is model

    def test_backward_compatibility_arguments(self, mock_device_mesh, mock_distributed_env):
        """Test that all original function arguments are still supported."""
        mesh, _, _, _ = mock_device_mesh
        model = MockModel("RegularModel")

        # Test with all possible arguments
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            mp_policy=None,
            offload_policy=None,
            sequence_parallel=False,
            activation_checkpointing=True,
            tp_shard_plan=None,
            dp_replicate_mesh_name="dp_replicate",
            dp_shard_cp_mesh_name="dp_shard_cp",
            tp_mesh_name="tp",
        )

        assert result is model

    def test_preserves_function_signature(self):
        """Test that the main function preserves its original signature."""
        import inspect

        sig = inspect.signature(fsdp2_strategy_parallelize)

        # Check that all expected parameters are present
        expected_params = [
            'model', 'device_mesh', 'mp_policy', 'offload_policy',
            'sequence_parallel', 'activation_checkpointing', 'tp_shard_plan',
            'dp_replicate_mesh_name', 'dp_shard_cp_mesh_name', 'tp_mesh_name'
        ]

        for param in expected_params:
            assert param in sig.parameters

        # Check default values are preserved
        assert sig.parameters['sequence_parallel'].default is False
        assert sig.parameters['activation_checkpointing'].default is False
        assert sig.parameters['dp_replicate_mesh_name'].default == "dp_replicate"
        assert sig.parameters['dp_shard_cp_mesh_name'].default == "dp_shard_cp"
        assert sig.parameters['tp_mesh_name'].default == "tp"


class TestStrategyExtensibility:
    """Test the extensibility of the strategy pattern."""

    def test_can_add_new_strategy_to_registry(self):
        """Test that new strategies can be added to the registry."""
        # Create a custom strategy
        class CustomStrategy(ParallelizationStrategy):
            def parallelize(self, model, device_mesh, **kwargs):
                return model

        custom_strategy = CustomStrategy()

        # Add to registry
        original_registry = PARALLELIZATION_STRATEGIES.copy()
        PARALLELIZATION_STRATEGIES["CustomModel"] = custom_strategy

        try:
            # Test that it's selected
            model = MockModel("CustomModel")
            strategy = get_parallelization_strategy(model)

            assert strategy is custom_strategy
            assert isinstance(strategy, CustomStrategy)

        finally:
            # Clean up registry
            PARALLELIZATION_STRATEGIES.clear()
            PARALLELIZATION_STRATEGIES.update(original_registry)

    def test_strategy_isolation(self):
        """Test that strategies are isolated and don't interfere with each other."""
        # Get strategies for different models
        regular_model = MockModel("RegularModel")
        nemotron_model = MockNemotronHModel()

        regular_strategy = get_parallelization_strategy(regular_model)
        nemotron_strategy = get_parallelization_strategy(nemotron_model)

        # Strategies should be different instances
        assert regular_strategy is not nemotron_strategy
        assert type(regular_strategy) != type(nemotron_strategy)

        # Both should be proper strategy objects
        assert isinstance(regular_strategy, ParallelizationStrategy)
        assert isinstance(nemotron_strategy, ParallelizationStrategy)

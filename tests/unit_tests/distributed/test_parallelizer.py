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

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

from nemo_automodel.components.distributed.optimized_tp_plans import _get_class_qualname

# Import the function under test
from nemo_automodel.components.distributed.parallelizer import (
    _attention_is_head_sharded,
    _extract_model_layers,
    _get_parallel_plan,
    _update_attention_head_counts_for_tp,
    apply_fsdp2_sharding_recursively,
    get_hf_tp_shard_plan,
    import_class_from_path,
    megatron_fsdp_strategy_parallelize,
)


class MockModel(nn.Module):
    """Mock model for testing purposes."""

    def __init__(self, model_type="llama", num_attention_heads=8, num_key_value_heads=8):
        super().__init__()
        if model_type == "baichuan2":
            self.config = SimpleNamespace(
                num_attention_heads=num_attention_heads,
            )
        else:
            self.config = SimpleNamespace(
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )

        # Create mock model as a proper nn.Module so it gets picked up by named_children()
        class MockInnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockModel._create_mock_layer() for _ in range(2)])

        self.model = MockInnerModel()

        if model_type == "gemma3":
            self.language_model = SimpleNamespace()
            self.language_model.layers = self.model.layers
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                )
            )

    @staticmethod
    def _create_mock_layer():
        """Create a mock transformer layer."""
        layer = nn.Module()
        layer.mlp = nn.Linear(10, 10)  # Simple MLP for testing
        return layer

    def forward(self, x):
        return x


class MockGemma3Model(nn.Module):
    """Mock Gemma3 model that simulates Gemma3ForConditionalGeneration."""

    def __init__(self, num_attention_heads=8, num_key_value_heads=8):
        # Explicitly call nn.Module.__init__() to avoid MRO issues with multiple inheritance
        nn.Module.__init__(self)

        # Set up config structure for Gemma3 with both top-level and nested structure
        self.config = SimpleNamespace(
            # Top-level attributes for regular model compatibility
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            # Nested structure for Gemma3
            text_config=SimpleNamespace(
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            ),
        )

        # Create mock model as a proper nn.Module so it gets picked up by named_children()
        class MockInnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockGemma3Model._create_mock_layer() for _ in range(2)])

        self.model = MockInnerModel()

        # Create language_model structure expected by Gemma3 as a proper PyTorch module
        class LanguageModel(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = layers

        self.language_model = LanguageModel(self.model.layers)

    @staticmethod
    def _create_mock_layer():
        """Create a mock transformer layer."""
        layer = nn.Module()
        layer.mlp = nn.Linear(10, 10)  # Simple MLP for testing
        return layer

    def forward(self, x):
        return x


def create_gemma3_mock():
    """Factory function to create a mock that passes Gemma3 type checks."""

    # Create a simple hybrid class like in the functional test
    class MockGemma3ModelWithTypeCheck(MockGemma3Model, Gemma3ForConditionalGeneration):
        """Mock Gemma3 model that properly inherits from Gemma3ForConditionalGeneration."""

        def __init__(self, num_attention_heads=8, num_key_value_heads=8):
            # Explicitly call only MockGemma3Model.__init__ to avoid MRO issues
            MockGemma3Model.__init__(self, num_attention_heads, num_key_value_heads)

    # Create an instance of the hybrid class
    mock = MockGemma3ModelWithTypeCheck()
    return mock


@pytest.fixture
def mock_device_mesh_fsdp2():
    """Create a mock device mesh."""
    mesh = MagicMock(spec=DeviceMesh)

    # Mock device_type to return a valid string
    mesh.device_type = "cuda"

    # Mock submeshes
    dp_replicate_mesh = MagicMock()
    dp_shard_mesh = MagicMock()
    cp_mesh = MagicMock()
    tp_mesh = MagicMock()

    dp_replicate_mesh.size.return_value = 1
    dp_shard_mesh.size.return_value = 2
    tp_mesh.size.return_value = 1
    cp_mesh.size.return_value = 1

    dp_replicate_mesh.ndim = 1
    dp_shard_mesh.ndim = 1
    tp_mesh.ndim = 1
    cp_mesh.ndim = 1

    # Configure mesh access
    mesh.__getitem__.side_effect = lambda key: {
        "dp_replicate": dp_replicate_mesh,
        "dp_shard": dp_shard_mesh,
        "tp": tp_mesh,
        "cp": cp_mesh,
    }[key]

    return mesh, dp_replicate_mesh, dp_shard_mesh, tp_mesh, cp_mesh


@pytest.fixture
def mock_device_mesh_megatron_fsdp():
    """Create a mock device mesh."""
    mesh = MagicMock(spec=DeviceMesh)

    # Mock device_type to return a valid string
    mesh.device_type = "cuda"

    # Mock submeshes
    dp_mesh = MagicMock()
    cp_mesh = MagicMock()
    tp_mesh = MagicMock()

    dp_mesh.size.return_value = 2
    tp_mesh.size.return_value = 1
    cp_mesh.size.return_value = 1

    dp_mesh.ndim = 1
    tp_mesh.ndim = 1
    cp_mesh.ndim = 1

    # Configure mesh access
    mesh.__getitem__.side_effect = lambda key: {
        "dp": dp_mesh,
        "tp": tp_mesh,
        "cp": cp_mesh,
        "dp_cp": dp_mesh,
    }[key]

    return mesh, dp_mesh, tp_mesh, cp_mesh


@pytest.fixture
def mock_distributed_env(monkeypatch):
    """Mock the distributed environment."""
    # Mock torch.distributed
    dist_mock = SimpleNamespace()
    dist_mock.is_initialized = lambda: True
    dist_mock.get_rank = lambda: 0
    dist_mock.get_world_size = lambda: 2

    # Add device_mesh structure to dist_mock
    device_mesh_mock = SimpleNamespace()
    dist_mock.device_mesh = device_mesh_mock

    # Mock device mesh resources
    mesh_resources_mock = SimpleNamespace()
    mesh_resources_mock.root_to_flatten_mapping = MagicMock()
    mesh_resources_mock.root_to_flatten_mapping.get.return_value = {}
    device_mesh_mock._mesh_resources = mesh_resources_mock

    # Add FSDP structure to dist_mock
    fsdp_mock = SimpleNamespace()
    fsdp_mock.MixedPrecisionPolicy = MagicMock()
    fsdp_mock.CPUOffloadPolicy = MagicMock()
    fsdp_mock.fully_shard = MagicMock(side_effect=lambda model, **kwargs: model)
    dist_mock.fsdp = fsdp_mock

    # Add algorithms structure to dist_mock
    checkpoint_wrapper_mock = SimpleNamespace()
    checkpoint_wrapper_mock.checkpoint_wrapper = MagicMock(side_effect=lambda x: x)

    # Add tensor parallel structure to dist_mock
    tp_parallel_mock = SimpleNamespace()
    tp_parallel_mock.parallelize_module = MagicMock()
    tp_parallel_mock.checkpoint_wrapper = checkpoint_wrapper_mock.checkpoint_wrapper

    tensor_mock = SimpleNamespace()
    tensor_mock.parallel = tp_parallel_mock
    dist_mock.tensor = tensor_mock

    checkpoint_mock = SimpleNamespace()
    checkpoint_mock.checkpoint_wrapper = checkpoint_wrapper_mock

    algorithms_mock = SimpleNamespace()
    algorithms_mock._checkpoint = checkpoint_mock
    dist_mock.algorithms = algorithms_mock

    # Apply patches
    monkeypatch.setattr("torch.distributed", dist_mock, raising=False)
    # Patch the imported functions directly in the parallelizer module
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.fully_shard", fsdp_mock.fully_shard, raising=False
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.parallelize_module",
        tp_parallel_mock.parallelize_module,
        raising=False,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer.checkpoint_wrapper",
        checkpoint_wrapper_mock.checkpoint_wrapper,
        raising=False,
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer._mesh_resources", mesh_resources_mock, raising=False
    )

    return {
        "dist": dist_mock,
        "mesh_resources": mesh_resources_mock,
        "fsdp": fsdp_mock,
        "tensor_parallel": tp_parallel_mock,
    }


@pytest.fixture
def mock_optimized_tp_plans(monkeypatch):
    """Mock the PARALLELIZE_FUNCTIONS dictionary."""
    mock_plans = {}

    def mock_llama_plan(model, sequence_parallel=False):
        return {"model.layers.0.self_attn.q_proj": ColwiseParallel()}

    def mock_gemma3_plan(model, sequence_parallel=False):
        return {"language_model.layers.0.self_attn.q_proj": ColwiseParallel()}

    # Mock the import to avoid actual dependency
    with patch("nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS", mock_plans):
        # Add mock functions for different model types
        mock_plans[type(MockModel())] = mock_llama_plan
        mock_plans[type(create_gemma3_mock())] = mock_gemma3_plan
        yield mock_plans


class TestMegatronFSDPStrategyParallelize:
    """Test suite for megatron_fsdp_strategy_parallelize function."""

    @pytest.fixture
    def mock_megatron_fsdp_env(self, monkeypatch):
        """Mock Megatron FSDP environment and dependencies."""
        # Mock megatron_fsdp module
        megatron_fsdp_mock = SimpleNamespace()
        megatron_fsdp_mock.fully_shard = MagicMock(return_value=(MagicMock(), None))

        # Mock HAVE_MEGATRON_FSDP flag
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.HAVE_MEGATRON_FSDP", True, raising=False
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.megatron_fsdp_fully_shard",
            megatron_fsdp_mock.fully_shard,
            raising=False,
        )

        # Mock parallelize_module
        parallelize_module_mock = MagicMock()
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.parallelize_module",
            parallelize_module_mock,
            raising=False,
        )

        # Mock import_classes_from_paths
        import_classes_mock = MagicMock(return_value=[])
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.import_classes_from_paths",
            import_classes_mock,
            raising=False,
        )

        return {
            "megatron_fsdp": megatron_fsdp_mock,
            "parallelize_module": parallelize_module_mock,
            "import_classes": import_classes_mock,
        }

    def test_basic_megatron_fsdp_with_default_mesh_names(self, mock_device_mesh_megatron_fsdp, mock_megatron_fsdp_env):
        """Test basic Megatron FSDP with default mesh names."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh_megatron_fsdp
        tp_mesh.size.return_value = 1  # No tensor parallelism
        cp_mesh.size.return_value = 1  # No context parallelism

        model = MockModel()
        optimizer = MagicMock()

        result_model, result_optimizer = megatron_fsdp_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            optimizer=optimizer,
        )

        # Verify megatron_fsdp_fully_shard was called with default mesh names
        mock_megatron_fsdp_env["megatron_fsdp"].fully_shard.assert_called_once()
        call_kwargs = mock_megatron_fsdp_env["megatron_fsdp"].fully_shard.call_args[1]
        assert call_kwargs["dp_shard_dim"] == "dp"
        assert call_kwargs["tp_dim"] == "tp"

    def test_megatron_fsdp_with_custom_mesh_names(self, mock_megatron_fsdp_env):
        """Test Megatron FSDP with custom mesh names."""
        # Create a mock device mesh with custom keys
        mesh = MagicMock(spec=DeviceMesh)
        mesh.device_type = "cuda"

        # Mock custom submeshes
        custom_dp_mesh = MagicMock()
        custom_tp_mesh = MagicMock()
        custom_cp_mesh = MagicMock()

        custom_dp_mesh.size.return_value = 2
        custom_tp_mesh.size.return_value = 1
        custom_cp_mesh.size.return_value = 1
        custom_dp_mesh.ndim = 1
        custom_tp_mesh.ndim = 1
        custom_cp_mesh.ndim = 1

        # Configure mesh access with custom names
        mesh.__getitem__.side_effect = lambda key: {
            "my_dp": custom_dp_mesh,
            "my_tp": custom_tp_mesh,
            "my_cp": custom_cp_mesh,
        }[key]

        model = MockModel()
        optimizer = MagicMock()

        result_model, result_optimizer = megatron_fsdp_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            optimizer=optimizer,
            dp_shard_dim="my_dp",
            tp_dim="my_tp",
        )

        # Verify megatron_fsdp_fully_shard was called with custom mesh names
        mock_megatron_fsdp_env["megatron_fsdp"].fully_shard.assert_called_once()
        call_kwargs = mock_megatron_fsdp_env["megatron_fsdp"].fully_shard.call_args[1]
        assert call_kwargs["dp_shard_dim"] == "my_dp"
        assert call_kwargs["tp_dim"] == "my_tp"

    def test_megatron_fsdp_with_context_parallelism_custom_names(self, mock_megatron_fsdp_env):
        """Test Megatron FSDP with context parallelism and custom mesh names."""
        # Create a mock device mesh with custom keys
        mesh = MagicMock(spec=DeviceMesh)
        mesh.device_type = "cuda"

        # Mock custom submeshes
        custom_dp_mesh = MagicMock()
        custom_tp_mesh = MagicMock()
        custom_cp_mesh = MagicMock()
        custom_dp_cp_mesh = MagicMock()

        custom_dp_mesh.size.return_value = 2
        custom_tp_mesh.size.return_value = 1
        custom_cp_mesh.size.return_value = 2  # Enable CP
        custom_dp_cp_mesh.size.return_value = 4  # Mock flattening
        custom_dp_mesh.ndim = 1
        custom_tp_mesh.ndim = 1
        custom_cp_mesh.ndim = 1
        custom_dp_cp_mesh.ndim = 1

        # Configure mesh access with custom names
        mesh.__getitem__.side_effect = lambda key: {
            "dp_mesh": custom_dp_mesh,
            "tp_mesh": custom_tp_mesh,
            "cp_mesh": custom_cp_mesh,
            "dp_cp": custom_dp_cp_mesh,
        }[key]

        model = MockModel()
        optimizer = MagicMock()

        result_model, result_optimizer = megatron_fsdp_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            optimizer=optimizer,
            dp_shard_dim="dp_cp",
            tp_dim="tp_mesh",
        )

        # Verify megatron_fsdp_fully_shard was called with dp_cp_mesh_name set correctly
        mock_megatron_fsdp_env["megatron_fsdp"].fully_shard.assert_called_once()
        call_kwargs = mock_megatron_fsdp_env["megatron_fsdp"].fully_shard.call_args[1]
        assert call_kwargs["dp_shard_dim"] == "dp_cp"  # Should use default when CP > 1
        assert call_kwargs["tp_dim"] == "tp_mesh"

    def test_megatron_fsdp_not_available_error(self, mock_device_mesh_megatron_fsdp, monkeypatch):
        """Test error when Megatron FSDP is not available."""
        # Mock HAVE_MEGATRON_FSDP as False
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.HAVE_MEGATRON_FSDP", False, raising=False
        )

        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh_megatron_fsdp
        model = MockModel()

        with pytest.raises(AssertionError):
            megatron_fsdp_strategy_parallelize(
                model=model,
                device_mesh=mesh,
            )


class TestUtilityFunctions:
    """Test utility functions used by fsdp2_strategy_parallelize."""

    def test_import_class_from_path_success(self):
        """Test successful import of class from path."""
        # Test importing a real class
        cls = import_class_from_path("torch.nn.Linear")
        assert cls is torch.nn.Linear

    def test_import_class_from_path_error(self):
        """Test error handling in import_class_from_path."""
        with pytest.raises(Exception):
            import_class_from_path("nonexistent.module.Class")


class TestGetHfTpShardPlan:
    """Test suite for get_hf_tp_shard_plan function."""

    def test_standard_model_with_class_tp_plan(self):
        """Test standard model with TP plan defined on model class."""
        model = MockModel()
        model_cls = type(model)

        # Add TP plan to model class
        model_cls._tp_plan = {
            "layers.0.self_attn.q_proj": "colwise",
            "layers.0.self_attn.k_proj": "colwise",
            "layers.0.mlp.gate_proj": "colwise",
        }

        # Mock config for tied embeddings test
        model.config.tie_word_embeddings = True

        try:
            result = get_hf_tp_shard_plan(model)

            # Verify TP plan was applied correctly
            assert len(result) > 0
            assert "layers.0.self_attn.q_proj" in result
            assert isinstance(result["layers.0.self_attn.q_proj"], ColwiseParallel)

        finally:
            # Clean up class attribute
            if hasattr(model_cls, "_tp_plan"):
                delattr(model_cls, "_tp_plan")

    def test_standard_model_with_instance_tp_plan(self):
        """Test standard model with TP plan defined on model instance."""
        model = MockModel()

        # Add TP plan to model instance
        model._tp_plan = {
            "layers.0.self_attn.q_proj": "rowwise",
            "layers.0.mlp.down_proj": "rowwise",
        }
        model.config.tie_word_embeddings = False

        result = get_hf_tp_shard_plan(model)

        # Verify TP plan was applied correctly
        assert len(result) > 0
        assert "layers.0.self_attn.q_proj" in result
        assert isinstance(result["layers.0.self_attn.q_proj"], RowwiseParallel)

        # Should add embed_tokens since tie_word_embeddings=False
        assert "model.embed_tokens" in result
        assert isinstance(result["model.embed_tokens"], RowwiseParallel)

    def test_standard_model_with_inner_model_tp_plan(self):
        """Test standard model with TP plan defined on inner model."""
        model = MockModel()

        # Add TP plan to inner model
        model.model._tp_plan = {
            "layers.0.self_attn.v_proj": "colwise_rep",
            "layers.0.self_attn.o_proj": "rowwise_rep",
        }
        model.config.tie_word_embeddings = False

        result = get_hf_tp_shard_plan(model)

        # Verify TP plan was applied correctly with model prefix
        assert len(result) > 0
        assert "model.layers.0.self_attn.v_proj" in result
        assert isinstance(result["model.layers.0.self_attn.v_proj"], ColwiseParallel)
        assert "model.layers.0.self_attn.o_proj" in result
        assert isinstance(result["model.layers.0.self_attn.o_proj"], RowwiseParallel)

    def test_multiple_tp_plan_sources_precedence(self):
        """Test precedence when TP plans exist in multiple places."""
        model = MockModel()
        model_cls = type(model)

        # Add TP plans to all possible sources
        model_cls._tp_plan = {"layers.0.self_attn.q_proj": "colwise"}
        model._tp_plan = {"layers.0.self_attn.k_proj": "rowwise"}
        model.model._tp_plan = {"layers.0.self_attn.v_proj": "colwise_rep"}
        model.config.tie_word_embeddings = True

        try:
            result = get_hf_tp_shard_plan(model)

            # All plans should be merged
            assert "layers.0.self_attn.q_proj" in result  # from class
            assert "layers.0.self_attn.k_proj" in result  # from instance
            assert "model.layers.0.self_attn.v_proj" in result  # from inner model with prefix

            # Instance plan should take precedence over class plan if same key exists
            assert isinstance(result["layers.0.self_attn.q_proj"], ColwiseParallel)
        finally:
            # Clean up class attribute
            if hasattr(model_cls, "_tp_plan"):
                delattr(model_cls, "_tp_plan")

    def test_lm_head_optimization(self):
        """Test special optimization for lm_head with colwise_rep."""
        model = MockModel()

        model._tp_plan = {
            "lm_head": "colwise_rep",
            "layers.0.self_attn.q_proj": "colwise",
        }
        model.config.tie_word_embeddings = False

        result = get_hf_tp_shard_plan(model)

        # Verify lm_head gets special optimization
        assert "lm_head" in result
        lm_head_parallel = result["lm_head"]
        assert isinstance(lm_head_parallel, ColwiseParallel)
        # The optimization should set output_layouts=Shard(-1) and use_local_output=False
        assert not lm_head_parallel.use_local_output

    def test_lm_head_no_optimization_when_tied(self):
        """Test lm_head doesn't get optimization when embeddings are tied."""
        model = MockModel()

        model._tp_plan = {
            "lm_head": "colwise_rep",
            "layers.0.self_attn.q_proj": "colwise",
        }
        model.config.tie_word_embeddings = True

        result = get_hf_tp_shard_plan(model)

        # Verify lm_head gets standard translation, not optimization
        assert "lm_head" in result
        lm_head_parallel = result["lm_head"]
        assert isinstance(lm_head_parallel, ColwiseParallel)

    def test_embed_tokens_added_when_not_tied(self):
        """Test embed_tokens is added when tie_word_embeddings=False."""
        model = MockModel()

        model._tp_plan = {"layers.0.self_attn.q_proj": "colwise"}
        model.config.tie_word_embeddings = False

        result = get_hf_tp_shard_plan(model)

        assert "model.embed_tokens" in result
        assert isinstance(result["model.embed_tokens"], RowwiseParallel)

    def test_parallel_style_translations(self):
        """Test all parallel style string translations."""
        model = MockModel()

        model._tp_plan = {
            "layer1": "colwise",
            "layer2": "rowwise",
            "layer3": "colwise_rep",
            "layer4": "rowwise_rep",
            "layer5": "sequence_parallel",
        }
        model.config.tie_word_embeddings = True

        result = get_hf_tp_shard_plan(model)

        assert isinstance(result["layer1"], ColwiseParallel)
        assert isinstance(result["layer2"], RowwiseParallel)
        assert isinstance(result["layer3"], ColwiseParallel)
        assert isinstance(result["layer4"], RowwiseParallel)
        assert isinstance(result["layer5"], SequenceParallel)

    def test_no_tp_plan_error(self):
        """Test error when no TP plan is found."""
        model = MockModel()
        model.config.tie_word_embeddings = True

        with pytest.raises(AssertionError, match="Hugging Face tp plan is not supported"):
            get_hf_tp_shard_plan(model)

    def test_invalid_parallel_style_error(self):
        """Test error for invalid parallel style string."""
        model = MockModel()

        model._tp_plan = {"layers.0.self_attn.q_proj": "invalid_style"}
        model.config.tie_word_embeddings = True

        with pytest.raises(ValueError, match="Unknown parallel style"):
            get_hf_tp_shard_plan(model)


class TestApplyFsdpShardingRecursively:
    """Test class for apply_fsdp2_sharding_recursively utility function."""

    @pytest.fixture
    def mock_module_list(self):
        """Create a mock ModuleList with transformer blocks."""
        module_list = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
        return module_list

    @pytest.fixture
    def mock_single_module(self):
        """Create a mock module with child modules."""

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 10)
                self.layer2 = nn.Linear(10, 10)
                self.nested = nn.ModuleList([nn.Linear(5, 5)])

        return TestModule()

    @pytest.fixture
    def mock_mesh(self):
        """Create a mock device mesh."""
        mesh = MagicMock(spec=DeviceMesh)
        return mesh

    @pytest.fixture
    def mock_mp_policy(self):
        """Create a mock mixed precision policy."""
        from torch.distributed.fsdp import MixedPrecisionPolicy

        mp_policy = MagicMock(spec=MixedPrecisionPolicy)
        return mp_policy

    @pytest.fixture
    def mock_offload_policy(self):
        """Create a mock offload policy."""
        from torch.distributed.fsdp import CPUOffloadPolicy

        offload_policy = MagicMock(spec=CPUOffloadPolicy)
        return offload_policy

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_module_list(
        self, mock_fully_shard, mock_module_list, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with a ModuleList."""

        # Set up mock return values - add FSDP2 prefetch methods that fully_shard normally provides
        def mock_shard(x, **kwargs):
            x.set_modules_to_forward_prefetch = MagicMock()
            x.set_modules_to_backward_prefetch = MagicMock()
            return x

        mock_fully_shard.side_effect = mock_shard

        # Call the function
        apply_fsdp2_sharding_recursively(
            module=mock_module_list, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # Verify fully_shard was called for each layer in the ModuleList
        assert mock_fully_shard.call_count == 3

        # Verify the call parameters for each layer
        calls = mock_fully_shard.call_args_list
        for i, call in enumerate(calls):
            args, kwargs = call
            assert args[0] is mock_module_list[i]  # The transformer block
            assert kwargs["mesh"] is mock_mesh
            assert kwargs["mp_policy"] is mock_mp_policy
            assert kwargs["offload_policy"] is mock_offload_policy

            # Check reshard_after_forward optimization (last layer should be False)
            expected_reshard = i < len(mock_module_list) - 1
            assert kwargs["reshard_after_forward"] == expected_reshard

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_module_list_without_offload_policy(
        self, mock_fully_shard, mock_module_list, mock_mesh, mock_mp_policy
    ):
        """Test apply_fsdp2_sharding_recursively with a ModuleList and no offload policy."""

        # Set up mock return values - add FSDP2 prefetch methods that fully_shard normally provides
        def mock_shard(x, **kwargs):
            x.set_modules_to_forward_prefetch = MagicMock()
            x.set_modules_to_backward_prefetch = MagicMock()
            return x

        mock_fully_shard.side_effect = mock_shard

        # Call the function without offload_policy
        apply_fsdp2_sharding_recursively(module=mock_module_list, mesh=mock_mesh, mp_policy=mock_mp_policy)

        # Verify fully_shard was called with None offload_policy
        calls = mock_fully_shard.call_args_list
        for call in calls:
            args, kwargs = call
            assert kwargs["offload_policy"] is None

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_regular_module(
        self, mock_fully_shard, mock_single_module, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with a regular module (not ModuleList)."""
        # Set up mock return values
        mock_fully_shard.side_effect = lambda x, **kwargs: x

        # Call the function
        apply_fsdp2_sharding_recursively(
            module=mock_single_module, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # For regular modules, it should recursively call on children
        # It should call itself recursively for the nested ModuleList
        # The nested ModuleList should get fully_shard called on its children
        assert mock_fully_shard.call_count == 1  # Just the nested ModuleList's single layer

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_empty_module_list(
        self, mock_fully_shard, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with an empty ModuleList."""
        empty_module_list = nn.ModuleList([])

        # Call the function
        apply_fsdp2_sharding_recursively(
            module=empty_module_list, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # Should not call fully_shard for empty ModuleList
        assert mock_fully_shard.call_count == 0

    @patch("nemo_automodel.components.distributed.parallelizer.fully_shard")
    def test_apply_fsdp_sharding_single_item_module_list(
        self, mock_fully_shard, mock_mesh, mock_mp_policy, mock_offload_policy
    ):
        """Test apply_fsdp2_sharding_recursively with a single-item ModuleList."""
        single_module_list = nn.ModuleList([nn.Linear(10, 10)])
        mock_fully_shard.side_effect = lambda x, **kwargs: x

        # Call the function
        apply_fsdp2_sharding_recursively(
            module=single_module_list, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # Should call fully_shard once
        assert mock_fully_shard.call_count == 1

        # For single item, reshard_after_forward should be False (optimization)
        call_args = mock_fully_shard.call_args_list[0]
        assert call_args[1]["reshard_after_forward"] is False

    def test_apply_fsdp_sharding_no_children(self, mock_mesh, mock_mp_policy, mock_offload_policy):
        """Test apply_fsdp2_sharding_recursively with a module that has no children."""
        leaf_module = nn.Linear(10, 10)

        # This should complete without error (no children to recurse on)
        apply_fsdp2_sharding_recursively(
            module=leaf_module, mesh=mock_mesh, mp_policy=mock_mp_policy, offload_policy=mock_offload_policy
        )

        # Just verify it doesn't crash - leaf modules have no children to process


class TestUnshardFsdp2Model:
    """Test suite for unshard_fsdp2_model context manager."""

    def test_unshard_fsdp2_model_basic_functionality(self):
        """Test basic unshard/reshard functionality with FSDP modules."""
        # Import the function to test
        from nemo_automodel.components.distributed.parallelizer import unshard_fsdp2_model

        # Create a simple test double that can pass isinstance checks
        class TestFSDPModule:
            def __init__(self):
                self.unshard_called = False
                self.reshard_called = False

            def unshard(self):
                self.unshard_called = True

            def reshard(self):
                self.reshard_called = True

        test_fsdp_module = TestFSDPModule()

        # Create a mock model that returns our test module
        mock_model = MagicMock()
        mock_model.modules.return_value = [test_fsdp_module, nn.Linear(10, 10)]

        # Patch FSDPModule to be our test class
        with patch.object(
            sys.modules["nemo_automodel.components.distributed.parallelizer"], "FSDPModule", TestFSDPModule
        ):
            # Test the context manager
            with unshard_fsdp2_model(mock_model):
                assert test_fsdp_module.unshard_called is True
                assert test_fsdp_module.reshard_called is False

            # After exiting, reshard should be called
            assert test_fsdp_module.reshard_called is True

    def test_unshard_fsdp2_model_exception_handling(self):
        """Test that reshard is called even if an exception occurs."""
        # Import the function to test
        from nemo_automodel.components.distributed.parallelizer import unshard_fsdp2_model

        # Create a simple test double that can pass isinstance checks
        class TestFSDPModule:
            def __init__(self):
                self.unshard_called = False
                self.reshard_called = False

            def unshard(self):
                self.unshard_called = True

            def reshard(self):
                self.reshard_called = True

        test_fsdp_module = TestFSDPModule()

        mock_model = MagicMock()
        mock_model.modules.return_value = [test_fsdp_module]

        # Patch FSDPModule to be our test class
        with patch.object(
            sys.modules["nemo_automodel.components.distributed.parallelizer"], "FSDPModule", TestFSDPModule
        ):
            with pytest.raises(ValueError):
                with unshard_fsdp2_model(mock_model):
                    raise ValueError("Test exception")

            # Verify reshard was still called despite the exception
            assert test_fsdp_module.reshard_called is True


class TestGetParallelPlanClassNameFallback:
    """Test that _get_parallel_plan matches by qualified class name (module.qualname)."""

    def test_identity_match(self):
        """Exact class qualname in PARALLELIZE_FUNCTIONS is found."""
        sentinel_plan = {"layer": ColwiseParallel()}
        model = MockModel()

        with patch(
            "nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS",
            {_get_class_qualname(type(model)): lambda m, sp: sentinel_plan},
        ):
            plan = _get_parallel_plan(model, sequence_parallel=False, tp_shard_plan=None)
        assert plan is sentinel_plan

    def test_class_name_fallback(self):
        """A different class object with the same module.qualname still matches.

        With the old class-object-keyed dict, identity was required. With the new
        string-keyed dict, two distinct class objects that share ``__module__`` and
        ``__qualname__`` resolve to the same key and both match — which is exactly
        the NeMo-RL wrapping scenario this fix targets.
        """
        sentinel_plan = {"layer": ColwiseParallel()}

        # Create a *different* class object with the same name (and therefore the same
        # module.qualname since both are defined in this test module).
        DuplicateMockModel = type("MockModel", (nn.Module,), {"forward": lambda self, x: x})
        assert DuplicateMockModel is not MockModel
        assert _get_class_qualname(DuplicateMockModel) == _get_class_qualname(MockModel)

        model = MockModel()
        model.__class__ = DuplicateMockModel  # model's type is the duplicate

        with patch(
            "nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS",
            {_get_class_qualname(MockModel): lambda m, sp: sentinel_plan},
        ):
            plan = _get_parallel_plan(model, sequence_parallel=False, tp_shard_plan=None)
        # Matches because module.qualname is the same, even though the class object differs
        assert plan is sentinel_plan

    def test_nemo_rl_wrapped_class_match(self):
        """A different class object with the same module and qualname still matches.

        This simulates the NeMo-RL scenario: _get_mixin_wrapped_class() creates a new
        class via type(...) that preserves __module__ and __qualname__ from the original.
        Both the original and the wrapper resolve to the same _get_class_qualname() key.
        """
        sentinel_plan = {"layer": ColwiseParallel()}
        original_cls = type(MockModel())

        # Simulate _get_mixin_wrapped_class: create a *new* class object that copies
        # __module__ and __qualname__ from the original (same qualname, different object)
        WrappedCls = type(
            original_cls.__name__,
            (nn.Module,),
            {
                "forward": lambda self, x: x,
                "__module__": original_cls.__module__,
                "__qualname__": original_cls.__qualname__,
            },
        )
        assert WrappedCls is not original_cls
        assert _get_class_qualname(WrappedCls) == _get_class_qualname(original_cls)

        model = MockModel()
        model.__class__ = WrappedCls  # model's type is the wrapper

        with patch(
            "nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS",
            {_get_class_qualname(original_cls): lambda m, sp: sentinel_plan},
        ):
            plan = _get_parallel_plan(model, sequence_parallel=False, tp_shard_plan=None)
        assert plan is sentinel_plan

    def test_no_match_falls_through_to_default(self):
        """Completely unknown class qualname falls through to the default plan."""
        model = MockModel()
        model.__class__ = type("UnknownModel", (nn.Module,), {"forward": lambda self, x: x})

        with patch(
            "nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS",
            {_get_class_qualname(MockModel): lambda m, sp: {"x": ColwiseParallel()}},
        ):
            plan = _get_parallel_plan(model, sequence_parallel=False, tp_shard_plan=None)
        # Should get the default Llama3-style plan (has q_proj, k_proj, etc.)
        assert "model.layers.*.self_attn.q_proj" in plan


class TestUpdateAttentionHeadCountsForTP:
    """Tests for _update_attention_head_counts_for_tp."""

    @staticmethod
    def _make_model(num_heads=64, num_kv_heads=8, hidden_size=8192, architectures=None, model_type=None):
        model = nn.Module()
        cfg = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
        )
        if architectures is not None:
            cfg.architectures = architectures
        if model_type is not None:
            cfg.model_type = model_type
        model.config = cfg

        inner = nn.Module()
        layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            attn = nn.Module()
            attn.num_heads = num_heads
            attn.num_key_value_heads = num_kv_heads
            layer.self_attn = attn
            layers.append(layer)
        inner.layers = layers
        model.model = inner
        return model

    def test_noop_for_tp_size_1(self):
        model = self._make_model()
        _update_attention_head_counts_for_tp(model, tp_size=1)
        assert model.config.num_attention_heads == 64
        assert model.config.num_key_value_heads == 8

    def test_preserves_config_and_updates_layer_attrs(self):
        model = self._make_model(num_heads=64, num_kv_heads=8, hidden_size=8192)
        _update_attention_head_counts_for_tp(model, tp_size=2)
        assert model.config.num_attention_heads == 64
        assert model.config.num_key_value_heads == 8
        assert model.config.head_dim == 128
        for layer in model.model.layers:
            assert layer.self_attn.num_heads == 32
            assert layer.self_attn.num_key_value_heads == 4

    def test_preserves_existing_head_dim(self):
        model = self._make_model(num_heads=64, num_kv_heads=8, hidden_size=8192)
        model.config.head_dim = 128
        _update_attention_head_counts_for_tp(model, tp_size=2)
        assert model.config.head_dim == 128

    def test_computes_head_dim_when_missing(self):
        model = self._make_model(num_heads=32, num_kv_heads=8, hidden_size=4096)
        _update_attention_head_counts_for_tp(model, tp_size=2)
        assert model.config.head_dim == 128  # 4096 // 32

    def test_decilm_nemotron_nas_skips_config_update(self):
        model = self._make_model(
            num_heads=64,
            num_kv_heads=8,
            hidden_size=8192,
            architectures=["DeciLMForCausalLM"],
            model_type="nemotron-nas",
        )
        _update_attention_head_counts_for_tp(model, tp_size=2)
        # Config should NOT be updated for DeciLM (per-layer head counts differ)
        assert model.config.num_attention_heads == 64
        assert model.config.num_key_value_heads == 8
        # But per-layer attn modules should still be updated
        for layer in model.model.layers:
            assert layer.self_attn.num_heads == 32
            assert layer.self_attn.num_key_value_heads == 4

    def test_derives_kv_heads_from_num_key_value_groups(self):
        """When config.num_key_value_heads is None, fall back to num_key_value_groups."""
        model = self._make_model(num_heads=64, num_kv_heads=8, hidden_size=8192)
        model.config.num_key_value_heads = None
        for layer in model.model.layers:
            layer.self_attn.num_key_value_groups = 8
        _update_attention_head_counts_for_tp(model, tp_size=2)
        for layer in model.model.layers:
            assert layer.self_attn.num_heads == 32
            assert layer.self_attn.num_key_value_heads == 4  # 32 // 8

    def test_kv_heads_defaults_to_num_heads_without_groups(self):
        """When config.num_key_value_heads is None and no num_key_value_groups attr."""
        model = self._make_model(num_heads=64, num_kv_heads=8, hidden_size=8192)
        model.config.num_key_value_heads = None
        # num_key_value_groups is never set in _make_model, so already absent
        _update_attention_head_counts_for_tp(model, tp_size=2)
        for layer in model.model.layers:
            assert layer.self_attn.num_key_value_heads == 32  # same as local_num_attention_heads

    def test_language_model_inner_path(self):
        """Layers under model.language_model are found when model.model has no layers."""
        model = nn.Module()
        model.config = SimpleNamespace(
            num_attention_heads=64,
            num_key_value_heads=8,
            hidden_size=8192,
        )
        lang = nn.Module()
        layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            attn = nn.Module()
            attn.num_heads = 64
            attn.num_key_value_heads = 8
            layer.self_attn = attn
            layers.append(layer)
        lang.layers = layers
        model.language_model = lang
        _update_attention_head_counts_for_tp(model, tp_size=2)
        for layer in lang.layers:
            assert layer.self_attn.num_heads == 32
            assert layer.self_attn.num_key_value_heads == 4

    def test_noop_without_config(self):
        model = nn.Module()
        _update_attention_head_counts_for_tp(model, tp_size=2)

    def test_noop_without_layers(self):
        model = nn.Module()
        model.config = SimpleNamespace(num_attention_heads=8, hidden_size=64)
        _update_attention_head_counts_for_tp(model, tp_size=2)


class TestAttentionIsHeadSharded:
    """Tests for _attention_is_head_sharded."""

    def test_colwise_default_is_sharded(self):
        """ColwiseParallel() with default output (Shard) → heads are sharded."""
        plan = {
            "model.layers.*.self_attn.q_proj": ColwiseParallel(),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        }
        assert _attention_is_head_sharded(plan) is True

    def test_colwise_explicit_shard_is_sharded(self):
        plan = {
            "model.layers.*.self_attn.q_proj": ColwiseParallel(output_layouts=Shard(-1)),
        }
        assert _attention_is_head_sharded(plan) is True

    def test_rowwise_replicate_is_not_sharded(self):
        """Phi-3 style: RowwiseParallel with Replicate output → not sharded."""
        plan = {
            "model.layers.*.self_attn.qkv_proj": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
            "model.layers.*.self_attn.o_proj": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
        }
        assert _attention_is_head_sharded(plan) is False

    def test_colwise_replicate_output_is_not_sharded(self):
        """ColwiseParallel with explicit Replicate output → not sharded."""
        plan = {
            "model.layers.*.self_attn.q_proj": ColwiseParallel(output_layouts=Replicate()),
        }
        assert _attention_is_head_sharded(plan) is False

    def test_no_attn_keys_is_not_sharded(self):
        """Plan with only MLP entries → not sharded."""
        plan = {
            "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(),
        }
        assert _attention_is_head_sharded(plan) is False

    def test_empty_plan_is_not_sharded(self):
        assert _attention_is_head_sharded({}) is False


# ---------------------------------------------------------------------------
# Activation checkpointing + KV-sharing tests
# ---------------------------------------------------------------------------


class _FakeLayer(nn.Module):
    """Minimal transformer layer with mlp, self_attn, and layernorms."""

    def __init__(self, dim: int = 16):
        super().__init__()
        self.mlp = nn.Linear(dim, dim)
        self.self_attn = nn.Linear(dim, dim)
        self.input_layernorm = nn.Linear(dim, dim)
        self.post_attention_layernorm = nn.Linear(dim, dim)

    def forward(self, x):
        return x


def _make_model_for_ac(
    num_layers: int = 2,
    dim: int = 16,
    use_cache: bool = True,
    num_kv_shared_layers: int = 0,
    text_config_nested: bool = True,
):
    """Build a minimal model with configurable KV-sharing for activation-checkpointing tests.

    Args:
        text_config_nested: If True, place ``num_kv_shared_layers`` under
            ``config.text_config`` (VLM pattern).  If False, place it directly
            on ``config`` (flat LLM pattern).
    """

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_FakeLayer(dim) for _ in range(num_layers)])

    model = nn.Module()
    model.model = _Inner()  # type: ignore[attr-defined]

    if text_config_nested:
        text_cfg = SimpleNamespace(num_kv_shared_layers=num_kv_shared_layers)
        model.config = SimpleNamespace(use_cache=use_cache, text_config=text_cfg)  # type: ignore[attr-defined]
    else:
        model.config = SimpleNamespace(  # type: ignore[attr-defined]
            use_cache=use_cache,
            num_kv_shared_layers=num_kv_shared_layers,
        )
    model.forward = lambda x: x  # type: ignore[attr-defined]
    return model


class TestActivationCheckpointingKVSharing:
    """Tests for the KV-sharing–aware activation-checkpointing guards
    in ``DefaultParallelizationStrategy.parallelize``.
    """

    @pytest.fixture(autouse=True)
    def _patch_parallelizer(self, monkeypatch):
        """Patch heavy distributed primitives so we can call ``parallelize``
        without a real GPU mesh.  ``checkpoint_wrapper`` is replaced with a
        lightweight wrapper that records which module was wrapped.
        """

        class _Wrapped(nn.Module):
            """Sentinel wrapper so we can assert which sub-modules were checkpointed.

            Must inherit from ``nn.Module`` because PyTorch's ``__setattr__``
            rejects non-Module values when replacing a registered child module.
            """

            def __init__(self, inner):
                super().__init__()
                self._inner = inner

            def forward(self, x):
                return self._inner(x)

        self._Wrapped = _Wrapped

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.checkpoint_wrapper",
            _Wrapped,
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.fully_shard",
            lambda model, **kw: model,
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.apply_fsdp2_sharding_recursively",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer.get_fsdp_dp_mesh",
            lambda mesh, *a, **kw: MagicMock(),
        )

    def _run_parallelize(self, model, activation_checkpointing=True):
        """Invoke the strategy under test and return the model."""
        from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

        strategy = DefaultParallelizationStrategy()
        mesh = MagicMock(spec=DeviceMesh)
        tp_mesh = MagicMock()
        tp_mesh.size.return_value = 1  # no TP
        mesh.__getitem__ = lambda self_, key: tp_mesh
        return strategy.parallelize(
            model=model,
            device_mesh=mesh,
            activation_checkpointing=activation_checkpointing,
        )

    # ------------------------------------------------------------------ #
    # use_cache preservation / disabling
    # ------------------------------------------------------------------ #

    def test_use_cache_preserved_when_kv_sharing(self):
        """Models with num_kv_shared_layers > 0 must keep use_cache=True."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=20)
        self._run_parallelize(model)
        assert model.config.use_cache is True

    def test_use_cache_disabled_without_kv_sharing(self):
        """Standard models (num_kv_shared_layers=0) get use_cache=False."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=0)
        self._run_parallelize(model)
        assert model.config.use_cache is False

    def test_use_cache_preserved_flat_config(self):
        """KV-sharing detected through a flat config (no text_config nesting)."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=10, text_config_nested=False)
        self._run_parallelize(model)
        assert model.config.use_cache is True

    def test_use_cache_disabled_flat_config_no_sharing(self):
        """Flat config without KV sharing still disables cache."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=0, text_config_nested=False)
        self._run_parallelize(model)
        assert model.config.use_cache is False

    def test_use_cache_noop_when_already_false(self):
        """If use_cache is already False and no KV sharing, code path is a no-op."""
        model = _make_model_for_ac(use_cache=False, num_kv_shared_layers=0)
        self._run_parallelize(model)
        assert model.config.use_cache is False

    def test_no_config_does_not_crash(self, monkeypatch):
        """Model without a config attribute must not raise."""
        monkeypatch.setattr(
            "nemo_automodel.components.distributed.parallelizer._extract_model_layers",
            lambda m: [],
        )
        model = nn.Module()
        model.forward = lambda x: x  # type: ignore[attr-defined]
        # no model.config at all
        self._run_parallelize(model)  # should not raise

    # ------------------------------------------------------------------ #
    # self_attn checkpoint wrapping
    # ------------------------------------------------------------------ #

    def test_self_attn_not_wrapped_when_kv_sharing(self):
        """KV-shared models: self_attn must NOT be wrapped (would corrupt cache)."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=20)
        self._run_parallelize(model)
        for layer in model.model.layers:
            assert not isinstance(layer.self_attn, self._Wrapped), (
                "self_attn should NOT be checkpoint-wrapped for KV-shared models"
            )

    def test_self_attn_wrapped_without_kv_sharing(self):
        """Standard models: self_attn IS wrapped."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=0)
        self._run_parallelize(model)
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, self._Wrapped), (
                "self_attn should be checkpoint-wrapped for standard models"
            )

    def test_mlp_always_wrapped(self):
        """MLP is checkpoint-wrapped regardless of KV sharing."""
        for kv_shared in (0, 20):
            model = _make_model_for_ac(num_kv_shared_layers=kv_shared)
            self._run_parallelize(model)
            for layer in model.model.layers:
                assert isinstance(layer.mlp, self._Wrapped), (
                    f"mlp should always be wrapped (num_kv_shared_layers={kv_shared})"
                )

    def test_layernorms_always_wrapped(self):
        """Layernorms are checkpoint-wrapped regardless of KV sharing."""
        for kv_shared in (0, 20):
            model = _make_model_for_ac(num_kv_shared_layers=kv_shared)
            self._run_parallelize(model)
            for layer in model.model.layers:
                assert isinstance(layer.input_layernorm, self._Wrapped)
                assert isinstance(layer.post_attention_layernorm, self._Wrapped)

    def test_no_wrapping_without_activation_checkpointing(self):
        """When activation_checkpointing=False, nothing is wrapped."""
        model = _make_model_for_ac(num_kv_shared_layers=0)
        self._run_parallelize(model, activation_checkpointing=False)
        for layer in model.model.layers:
            assert not isinstance(layer.mlp, self._Wrapped)
            assert not isinstance(layer.self_attn, self._Wrapped)
        assert model.config.use_cache is True  # untouched

    # ------------------------------------------------------------------ #
    # HF native gradient-checkpointing path
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Exception / edge-case branches
    # ------------------------------------------------------------------ #

    def test_frozen_config_use_cache_except_branch(self):
        """When ``model.config.use_cache = False`` raises, the except branch runs."""
        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=0)

        class _FrozenConfig:
            use_cache = True
            text_config = SimpleNamespace(num_kv_shared_layers=0)

            def __setattr__(self, name, value):
                raise AttributeError("frozen")

        model.config = _FrozenConfig()  # type: ignore[attr-defined]
        self._run_parallelize(model)
        # use_cache stays True because the assignment raised and was caught
        assert model.config.use_cache is True

    def test_no_config_with_layers_does_not_crash(self):
        """Model without ``config`` but with extractable layers does not crash."""

        class _Bare(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([_FakeLayer() for _ in range(2)])  # type: ignore[attr-defined]

            def forward(self, x):
                return x

        model = _Bare()
        # no model.config → hasattr(model, "config") is False
        self._run_parallelize(model)
        # mlp should still be wrapped (activation_checkpointing still applies)
        for layer in model.model.layers:
            assert isinstance(layer.mlp, self._Wrapped)

    def test_layer_missing_self_attn(self):
        """Layers without ``self_attn`` are skipped gracefully."""

        class _MlpOnlyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Linear(16, 16)

            def forward(self, x):
                return x

        model = _make_model_for_ac(num_kv_shared_layers=0)
        model.model.layers = nn.ModuleList([_MlpOnlyLayer() for _ in range(2)])
        self._run_parallelize(model)
        for layer in model.model.layers:
            assert isinstance(layer.mlp, self._Wrapped)
            assert not hasattr(layer, "self_attn")

    def test_layer_missing_mlp(self):
        """Layers without ``mlp`` are skipped gracefully."""

        class _AttnOnlyLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = nn.Linear(16, 16)

            def forward(self, x):
                return x

        model = _make_model_for_ac(num_kv_shared_layers=0)
        model.model.layers = nn.ModuleList([_AttnOnlyLayer() for _ in range(2)])
        self._run_parallelize(model)
        for layer in model.model.layers:
            assert isinstance(layer.self_attn, self._Wrapped)
            assert not hasattr(layer, "mlp")

    # ------------------------------------------------------------------ #
    # HF native gradient-checkpointing path
    # ------------------------------------------------------------------ #

    @staticmethod
    def _setup_hf_native_model(monkeypatch, num_kv_shared_layers):
        """Helper: configure a model + fake transformers module for the HF native path."""
        import types

        class _FakeGradLayer(_FakeLayer):
            pass

        _FakeGradLayer.__module__ = "transformers.models.gemma4.modeling_gemma4"

        fake_module = types.ModuleType("transformers.modeling_layers")
        fake_module.GradientCheckpointingLayer = _FakeGradLayer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers.modeling_layers", fake_module)

        model = _make_model_for_ac(use_cache=True, num_kv_shared_layers=num_kv_shared_layers)
        for i in range(len(model.model.layers)):
            model.model.layers[i] = _FakeGradLayer()
        model.supports_gradient_checkpointing = True  # type: ignore[attr-defined]
        model.gradient_checkpointing_enable = MagicMock()  # type: ignore[attr-defined]
        return model

    def test_hf_native_grad_ckpt_preserves_use_cache_with_kv_sharing(self, monkeypatch):
        """Even when the HF native path is taken, use_cache stays True for KV-shared models."""
        model = self._setup_hf_native_model(monkeypatch, num_kv_shared_layers=20)
        self._run_parallelize(model)

        assert model.config.use_cache is True
        model.gradient_checkpointing_enable.assert_called_once()

    def test_hf_native_grad_ckpt_disables_use_cache_without_kv_sharing(self, monkeypatch):
        """HF native path + no KV sharing: use_cache is set to False."""
        model = self._setup_hf_native_model(monkeypatch, num_kv_shared_layers=0)
        self._run_parallelize(model)

        assert model.config.use_cache is False
        model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )


class TestExtractModelLayers:
    """Tests for ``_extract_model_layers`` flattening of ModuleList results.

    Covers the PR that replaced ``layers.extend(_reduce_attrs(...))`` with a
    helper that flattens ModuleList elements so each decoder layer ends up as
    its own list entry (what AC wrapping expects), while leaving non-ModuleList
    results (e.g. ModuleDict after PP split) appended as-is.
    """

    def _make_layers(self, n: int) -> nn.ModuleList:
        return nn.ModuleList([_FakeLayer() for _ in range(n)])

    @staticmethod
    def _bare_instance(cls):
        """Instantiate an HF model class without running HF ``__init__``.

        Needed because ``MODEL_CLS_TO_LAYERS`` is keyed by exact class identity
        (no subclass match), but the real classes require a config to
        construct. ``__new__`` + manual ``nn.Module.__init__`` gives us an
        instance where ``type(model) is cls`` while skipping the expensive
        construction path.
        """
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        return obj

    def test_class_keyed_single_fqn_flattens_modulelist(self):
        """GPT2LMHeadModel entry ``["transformer.h"]`` → individual layers.

        Before the fix, ``layers.extend(_reduce_attrs(...))`` put the ModuleList
        itself into ``layers`` as one element; hasattr(layer, 'mlp') then failed
        and AC silently skipped every layer. Flattening must restore the
        per-layer elements so the AC loop can wrap them.
        """
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        model = self._bare_instance(GPT2LMHeadModel)
        transformer = nn.Module()
        layers = self._make_layers(3)
        transformer.h = layers
        model.transformer = transformer

        result = _extract_model_layers(model)

        assert len(result) == 3, f"expected 3 flat layers, got {len(result)}"
        assert all(r is layers[i] for i, r in enumerate(result)), (
            "result should be the individual decoder layer objects, not a ModuleList container"
        )
        assert not any(isinstance(r, nn.ModuleList) for r in result)

    def test_string_keyed_arm_flattens_modulelist(self):
        """``NemotronHForCausalLM`` is string-keyed in MODEL_CLS_TO_LAYERS.

        Hits the ``model_cls.__name__ in MODEL_CLS_TO_LAYERS`` branch.
        """

        class NemotronHForCausalLM(nn.Module):
            def __init__(self, layers):
                super().__init__()
                backbone = nn.Module()
                backbone.layers = layers
                self.backbone = backbone

        layers = self._make_layers(4)
        result = _extract_model_layers(NemotronHForCausalLM(layers))

        assert len(result) == 4
        assert all(r is layers[i] for i, r in enumerate(result))

    def test_multi_fqn_flattens_each_modulelist(self):
        """Qwen2.5-VL entry ``["language_model.layers", "visual.blocks"]``.

        Both FQNs resolve to ModuleLists; both must be flattened so all decoder
        and vision blocks appear as individual elements in the final list.
        """
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )

        model = self._bare_instance(Qwen2_5_VLForConditionalGeneration)
        lang = self._make_layers(5)
        vis = self._make_layers(2)
        language_model = nn.Module()
        language_model.layers = lang
        model.language_model = language_model
        visual = nn.Module()
        visual.blocks = vis
        model.visual = visual

        result = _extract_model_layers(model)

        assert len(result) == 7
        assert [id(r) for r in result[:5]] == [id(item) for item in lang]
        assert [id(r) for r in result[5:]] == [id(item) for item in vis]
        assert not any(isinstance(r, nn.ModuleList) for r in result)

    def test_non_modulelist_element_appended_as_single_entry(self):
        """PP post-split: ``_reduce_attrs`` returns a ModuleDict.

        A ModuleDict is NOT an nn.ModuleList, so ``_extend_layers`` must fall
        through to ``layers.append(m)`` and keep it as a single element —
        same behaviour as before the fix (the AC loop then skips it via
        hasattr, which is the expected PP-path behaviour and handled
        elsewhere for the happy PP case).
        """
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        model = self._bare_instance(GPT2LMHeadModel)
        layer_dict = nn.ModuleDict({"0": _FakeLayer(), "1": _FakeLayer()})
        transformer = nn.Module()
        transformer.h = layer_dict
        model.transformer = transformer

        result = _extract_model_layers(model)

        # ModuleDict is not flattened — it stays as one element.
        assert len(result) == 1
        assert result[0] is layer_dict

    def test_fallback_branch_still_handles_modulelist(self):
        """Non-MODEL_CLS_TO_LAYERS models hit the ``hasattr(model.model, 'layers')``
        fallback, which is unchanged by the PR. Guard against accidental regression.
        """

        class GenericCausalLM(nn.Module):
            def __init__(self, layers):
                super().__init__()
                inner = nn.Module()
                inner.layers = layers
                self.model = inner

        layers = self._make_layers(2)
        result = _extract_model_layers(GenericCausalLM(layers))
        assert len(result) == 2
        assert all(r is layers[i] for i, r in enumerate(result))

    def test_fallback_branch_handles_moduledict(self):
        """Fallback branch already normalises ModuleDict via ``.values()``."""

        class GenericCausalLM(nn.Module):
            def __init__(self, layer_dict):
                super().__init__()
                inner = nn.Module()
                inner.layers = layer_dict
                self.model = inner

        layer_dict = nn.ModuleDict({"0": _FakeLayer(), "1": _FakeLayer(), "2": _FakeLayer()})
        result = _extract_model_layers(GenericCausalLM(layer_dict))
        assert len(result) == 3
        assert [id(r) for r in result] == [id(v) for v in layer_dict.values()]

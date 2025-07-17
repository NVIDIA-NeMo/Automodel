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
import types
from typing import Dict, Any
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    ParallelStyle,
)

from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

# Import the function under test
from nemo_automodel.components.distributed.parallelizer import (
    fsdp2_strategy_parallelize,
    import_class_from_path,
    get_hf_tp_shard_plan,
)


class MockModel(nn.Module):
    """Mock model for testing purposes."""
    
    def __init__(self, model_type="llama", num_attention_heads=8, num_key_value_heads=8):
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        
        # Create mock model as a proper nn.Module so it gets picked up by named_children()
        class MockInnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    MockModel._create_mock_layer() for _ in range(2)
                ])
        
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
            )
        )
        
        # Create mock model as a proper nn.Module so it gets picked up by named_children()
        class MockInnerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    MockGemma3Model._create_mock_layer() for _ in range(2)
                ])
        
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
def mock_device_mesh():
    """Create a mock device mesh."""
    mesh = MagicMock(spec=DeviceMesh)
    
    # Mock device_type to return a valid string
    mesh.device_type = "cuda"
    
    # Mock submeshes
    dp_mesh = MagicMock()
    tp_mesh = MagicMock()
    cp_mesh = MagicMock()
    
    dp_mesh.size.return_value = 2
    tp_mesh.size.return_value = 1
    cp_mesh.size.return_value = 1
    
    dp_mesh.ndim = 1
    tp_mesh.ndim = 1
    cp_mesh.ndim = 1
    
    # Configure mesh access
    mesh.__getitem__.side_effect = lambda key: {
        "data_parallel": dp_mesh,
        "tensor_parallel": tp_mesh,
        "context_parallel": cp_mesh,
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
    monkeypatch.setattr("nemo_automodel.components.distributed.parallelizer.fully_shard", fsdp_mock.fully_shard, raising=False)
    monkeypatch.setattr("nemo_automodel.components.distributed.parallelizer.parallelize_module", tp_parallel_mock.parallelize_module, raising=False)
    monkeypatch.setattr("nemo_automodel.components.distributed.parallelizer.checkpoint_wrapper", checkpoint_wrapper_mock.checkpoint_wrapper, raising=False)
    monkeypatch.setattr("nemo_automodel.components.distributed.parallelizer._mesh_resources", mesh_resources_mock, raising=False)
    
    return {
        "dist": dist_mock,
        "mesh_resources": mesh_resources_mock,
        "fsdp": fsdp_mock,
        "tp": tp_parallel_mock,
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


class TestFSDP2StrategyParallelize:
    """Test suite for fsdp2_strategy_parallelize function."""
    
    def test_basic_parallelization_dp_only(self, mock_device_mesh, mock_distributed_env):
        """Test basic parallelization with data parallelism only."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 1  # No tensor parallelism
        
        model = MockModel()
        
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
        )
        
        assert result is model
        mock_distributed_env["fsdp"].fully_shard.assert_called()
        mock_distributed_env["tp"].parallelize_module.assert_not_called()
    
    def test_tensor_parallelism_with_custom_plan(self, mock_device_mesh, mock_distributed_env):
        """Test tensor parallelism with custom parallel plan."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2  # Enable tensor parallelism
        
        model = MockModel()
        custom_plan = {"model.layers.0.self_attn.q_proj": ColwiseParallel()}
        
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            tp_shard_plan=custom_plan,
        )
        
        assert result is model
        mock_distributed_env["tp"].parallelize_module.assert_called_once()
        mock_distributed_env["fsdp"].fully_shard.assert_called()
    
    def test_gemma3_model_handling(self, mock_device_mesh, mock_distributed_env):
        """Test Gemma3 model type handling."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        
        model = create_gemma3_mock()
        
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
        )
        
        assert result is model
        mock_distributed_env["fsdp"].fully_shard.assert_called()
    
    def test_attention_heads_validation_error(self, mock_device_mesh, mock_distributed_env):
        """Test validation error when attention heads not divisible by TP size."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 3  # Not divisible by 8 heads
        
        model = MockModel(num_attention_heads=8, num_key_value_heads=8)
        
        with pytest.raises(AssertionError, match="num_key_value_heads.*must be divisible"):
            fsdp2_strategy_parallelize(
                model=model,
                device_mesh=mesh,
                tp_shard_plan={"dummy": ColwiseParallel()},
            )
    
    def test_activation_checkpointing(self, mock_device_mesh, mock_distributed_env):
        """Test activation checkpointing application."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        
        model = MockModel()
        
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            activation_checkpointing=True,
        )
        
        assert result is model
        # Verify checkpoint_wrapper was called for MLP layers
        mock_distributed_env["tp"].checkpoint_wrapper.assert_called()
    
    def test_optimized_tp_plan_usage(self, mock_device_mesh, mock_distributed_env, mock_optimized_tp_plans):
        """Test usage of optimized TP plan based on model type."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        
        model = MockModel()
        
        # Mock the PARALLELIZE_FUNCTIONS to include our model type
        with patch("nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS", mock_optimized_tp_plans):
            result = fsdp2_strategy_parallelize(
                model=model,
                device_mesh=mesh,
            )
        
        assert result is model
        mock_distributed_env["tp"].parallelize_module.assert_called_once()
    
    def test_hf_tp_plan_fallback(self, mock_device_mesh, mock_distributed_env):
        """Test fallback to HuggingFace TP plan."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        
        model = MockModel()
        
        with patch("nemo_automodel.components.distributed.parallelizer.get_hf_tp_shard_plan") as mock_hf_plan:
            mock_hf_plan.return_value = {"model.layers.0.self_attn.q_proj": ColwiseParallel()}
            
            result = fsdp2_strategy_parallelize(
                model=model,
                device_mesh=mesh,
                use_hf_tp_plan=True,
            )
        
        assert result is model
        mock_hf_plan.assert_called_once_with(model)
        mock_distributed_env["tp"].parallelize_module.assert_called_once()
    
    def test_string_tp_plan_import(self, mock_device_mesh, mock_distributed_env):
        """Test importing TP plan from string path."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        
        model = MockModel()
        
        with patch("nemo_automodel.components.distributed.parallelizer.import_class_from_path") as mock_import:
            mock_plan = {"model.layers.0.self_attn.q_proj": ColwiseParallel()}
            mock_import.return_value = mock_plan
            
            result = fsdp2_strategy_parallelize(
                model=model,
                device_mesh=mesh,
                tp_shard_plan="path.to.parallel.plan",
            )
        
        assert result is model
        mock_import.assert_called_once_with("path.to.parallel.plan")
        mock_distributed_env["tp"].parallelize_module.assert_called_once()
    
    def test_function_tp_plan_import(self, mock_device_mesh, mock_distributed_env):
        """Test importing TP plan from function path."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        
        model = MockModel()
        
        def mock_plan_function():
            return {"model.layers.0.self_attn.q_proj": ColwiseParallel()}
        
        with patch("nemo_automodel.components.distributed.parallelizer.import_class_from_path") as mock_import:
            mock_import.return_value = mock_plan_function
            
            result = fsdp2_strategy_parallelize(
                model=model,
                device_mesh=mesh,
                tp_shard_plan="path.to.plan.function",
            )
        
        assert result is model
        mock_import.assert_called_once_with("path.to.plan.function")
        mock_distributed_env["tp"].parallelize_module.assert_called_once()
    
    def test_invalid_tp_plan_error(self, mock_device_mesh, mock_distributed_env):
        """Test error handling for invalid TP plan."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        
        model = MockModel()
        
        with patch("nemo_automodel.components.distributed.parallelizer.import_class_from_path") as mock_import:
            mock_import.side_effect = ImportError("Module not found")
            
            with pytest.raises(ValueError, match="Custom parallel plan.*is not valid"):
                fsdp2_strategy_parallelize(
                    model=model,
                    device_mesh=mesh,
                    tp_shard_plan="invalid.path",
                )
    
    def test_mixed_precision_policy_creation(self, mock_device_mesh, mock_distributed_env):
        """Test mixed precision policy creation when not provided."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        
        model = MockModel()
        
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            param_dtype=torch.float16,
        )
        
        assert result is model
        # Verify mixed precision policy was created
        mock_distributed_env["fsdp"].fully_shard.assert_called()
    
    def test_cpu_offload_policy(self, mock_device_mesh, mock_distributed_env):
        """Test CPU offload policy configuration."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        
        model = MockModel()
        
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
            cpu_offload=True,
        )
        
        assert result is model
        mock_distributed_env["fsdp"].fully_shard.assert_called()
    
    def test_sequence_parallel_with_optimized_plan(self, mock_device_mesh, mock_distributed_env, mock_optimized_tp_plans):
        """Test sequence parallel with optimized plan."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        
        model = MockModel()
        
        with patch("nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS", mock_optimized_tp_plans):
            result = fsdp2_strategy_parallelize(
                model=model,
                device_mesh=mesh,
                sequence_parallel=True,
            )
        
        assert result is model
        mock_distributed_env["tp"].parallelize_module.assert_called_once()
    
    def test_sequence_parallel_with_hf_plan_error(self, mock_device_mesh, mock_distributed_env):
        """Test error when sequence parallel is used with HF plan."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        
        model = MockModel()
        
        with patch("nemo_automodel.components.distributed.parallelizer.get_hf_tp_shard_plan"):
            with pytest.raises(AssertionError, match="sequence_parallel is not supported in HF tp plan"):
                fsdp2_strategy_parallelize(
                    model=model,
                    device_mesh=mesh,
                    sequence_parallel=True,
                    use_hf_tp_plan=True,
                )
    
    def test_dp_cp_mesh_selection(self, mock_device_mesh, mock_distributed_env):
        """Test proper data parallel mesh selection when CP > 1."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        
        # Mock mesh resources to simulate dp_cp mesh
        mock_distributed_env["mesh_resources"].root_to_flatten_mapping.get.return_value = {"dp_cp": True}
        
        model = MockModel()
        
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
        )
        
        assert result is model
        mock_distributed_env["fsdp"].fully_shard.assert_called()
    
    def test_hybrid_sharding_not_supported_error(self, mock_device_mesh, mock_distributed_env):
        """Test error when hybrid sharding is attempted."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        dp_mesh.ndim = 2  # Multi-dimensional mesh (hybrid sharding)
        
        model = MockModel()
        
        with pytest.raises(AssertionError, match="Hybrid-sharding not supported"):
            fsdp2_strategy_parallelize(
                model=model,
                device_mesh=mesh,
            )
    
    def test_reshard_optimization(self, mock_device_mesh, mock_distributed_env):
        """Test reshard_after_forward optimization for transformer layers."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        
        model = MockModel()
        # Add more layers to test the optimization
        model.model.layers.extend([model._create_mock_layer() for _ in range(3)])
        
        result = fsdp2_strategy_parallelize(
            model=model,
            device_mesh=mesh,
        )
        
        assert result is model
        # Verify fully_shard was called multiple times (once per layer + root)
        assert mock_distributed_env["fsdp"].fully_shard.call_count > 1


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
            
            # Should not add embed_tokens since tie_word_embeddings=True
            assert "model.embed_tokens" not in result
        finally:
            # Clean up class attribute
            if hasattr(model_cls, '_tp_plan'):
                delattr(model_cls, '_tp_plan')
    
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
    
    def test_gemma3_model_with_tp_plan(self):
        """Test Gemma3 model with special prefix handling."""
        model = create_gemma3_mock()
        
        # Add TP plan to inner language model
        model.language_model._tp_plan = {
            "layers.0.self_attn.q_proj": "sequence_parallel",
            "layers.0.mlp.gate_proj": "colwise",
        }
        model.config.tie_word_embeddings = False
        
        result = get_hf_tp_shard_plan(model)
        
        # Verify TP plan uses language_model prefix for Gemma3
        assert len(result) > 0
        assert "language_model.layers.0.self_attn.q_proj" in result
        assert isinstance(result["language_model.layers.0.self_attn.q_proj"], SequenceParallel)
        assert "language_model.embed_tokens" in result
    
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
            if hasattr(model_cls, '_tp_plan'):
                delattr(model_cls, '_tp_plan')
    
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
    
    def test_embed_tokens_not_added_when_tied(self):
        """Test embed_tokens is not added when tie_word_embeddings=True."""
        model = MockModel()
        
        model._tp_plan = {"layers.0.self_attn.q_proj": "colwise"}
        model.config.tie_word_embeddings = True
        
        result = get_hf_tp_shard_plan(model)
        
        assert "model.embed_tokens" not in result
    
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
    
    def test_gemma3_embed_tokens_with_language_model_prefix(self):
        """Test embed_tokens gets correct prefix for Gemma3 models."""
        model = create_gemma3_mock()
        
        model.language_model._tp_plan = {"layers.0.self_attn.q_proj": "colwise"}
        model.config.tie_word_embeddings = False
        
        result = get_hf_tp_shard_plan(model)
        
        # Should use language_model prefix for Gemma3
        assert "language_model.embed_tokens" in result
        assert "model.embed_tokens" not in result


class TestFSDP2StrategyEndToEnd:
    """End-to-end tests that verify the FSDP2 strategy parallelization works with all components integrated."""
    
    def test_full_pipeline_mock(self, mock_device_mesh, mock_distributed_env, mock_optimized_tp_plans):
        """Test the full pipeline with all components mocked."""
        mesh, dp_mesh, tp_mesh, cp_mesh = mock_device_mesh
        tp_mesh.size.return_value = 2
        
        model = MockModel()
        
        with patch("nemo_automodel.components.distributed.parallelizer.PARALLELIZE_FUNCTIONS", mock_optimized_tp_plans):
            result = fsdp2_strategy_parallelize(
                model=model,
                device_mesh=mesh,
                param_dtype=torch.bfloat16,
                sequence_parallel=False,
                activation_checkpointing=True,
                cpu_offload=False,
            )
        
        assert result is model
        
        # Verify all expected calls were made
        mock_distributed_env["tp"].parallelize_module.assert_called_once()
        mock_distributed_env["tp"].checkpoint_wrapper.assert_called()
        mock_distributed_env["fsdp"].fully_shard.assert_called() 

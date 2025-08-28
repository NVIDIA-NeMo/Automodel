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

"""Unit tests for optimized_tp_plans module."""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.optimized_tp_plans import (
    RotaryEmbedParallel,
    _parallelize_gemma3,
    _parallelize_llama,
    _parallelize_qwen,
    PARALLELIZE_FUNCTIONS,
)
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class MockModel:
    """Mock model class for testing."""
    def __init__(self, model_type="llama", tie_word_embeddings=False):
        self.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.__class__ = {
            "llama": LlamaForCausalLM,
            "qwen2": Qwen2ForCausalLM,
            "qwen3": Qwen3ForCausalLM,
            "gemma3_causal": Gemma3ForCausalLM,
            "gemma3_conditional": Gemma3ForConditionalGeneration,
        }[model_type]


class MockDeviceMesh:
    """Mock device mesh for testing."""
    def __init__(self):
        pass


class TestRotaryEmbedParallel:
    """Test suite for RotaryEmbedParallel class."""

    def test_prepare_input_fn_with_dtensor(self):
        """Test _prepare_input_fn when input is already DTensor."""
        # Mock device mesh
        device_mesh = MockDeviceMesh()

        # Mock DTensor inputs
        mock_dtensor1 = Mock(spec=DTensor)
        mock_dtensor2 = Mock(spec=DTensor)
        inputs = (mock_dtensor1, mock_dtensor2)

        # Mock module
        mod = Mock()

        # Mock sequence sharding
        sequence_sharding = [Shard(1)]

        result = RotaryEmbedParallel._prepare_input_fn(
            sequence_sharding, mod, inputs, device_mesh
        )

        # Should return same type with original inputs unchanged
        assert type(result) == type(inputs)
        assert result[0] == mock_dtensor1
        assert result[1] == mock_dtensor2

    @patch('torch.distributed.get_rank')
    @patch.object(DTensor, 'from_local')
    def test_prepare_input_fn_with_tensor(self, mock_from_local, mock_get_rank):
        """Test _prepare_input_fn when input is regular tensor."""
        mock_get_rank.return_value = 0
        device_mesh = MockDeviceMesh()

        # Mock tensor inputs
        tensor1 = torch.randn(4, 8)
        tensor2 = torch.randn(4, 8)
        inputs = (tensor1, tensor2)

        # Mock DTensor creation
        mock_dtensor1 = Mock(spec=DTensor)
        mock_dtensor2 = Mock(spec=DTensor)
        mock_from_local.side_effect = [mock_dtensor1, mock_dtensor2]

        mod = Mock()
        sequence_sharding = [Shard(1)]

        result = RotaryEmbedParallel._prepare_input_fn(
            sequence_sharding, mod, inputs, device_mesh
        )

        # Should have called from_local twice
        assert mock_from_local.call_count == 2

        # First call should be for sequence parallel sharding
        first_call = mock_from_local.call_args_list[0]
        assert first_call[1]['local_tensor'] is tensor1
        assert first_call[1]['device_mesh'] is device_mesh
        assert first_call[1]['placements'] == sequence_sharding
        assert first_call[1]['run_check'] is True

        # Second call should be for replication
        second_call = mock_from_local.call_args_list[1]
        assert second_call[1]['local_tensor'] is tensor2
        assert second_call[1]['device_mesh'] is device_mesh
        assert second_call[1]['placements'] == (Replicate(),)
        assert second_call[1]['run_check'] is False

    @patch('torch.distributed.get_rank')
    @patch.object(DTensor, 'from_local')
    def test_prepare_input_fn_value_error(self, mock_from_local, mock_get_rank):
        """Test _prepare_input_fn handles ValueError properly."""
        mock_get_rank.return_value = 1
        mock_from_local.side_effect = ValueError("Shape mismatch")

        device_mesh = MockDeviceMesh()
        tensor = torch.randn(4, 8)
        inputs = (tensor, torch.randn(4, 8))
        mod = Mock()
        sequence_sharding = [Shard(1)]

        with pytest.raises(ValueError) as exc_info:
            RotaryEmbedParallel._prepare_input_fn(
                sequence_sharding, mod, inputs, device_mesh
            )

        # Should wrap original error with helpful context
        assert "Failed to shard tensor for sequence parallelism" in str(exc_info.value)
        assert "rank 1" in str(exc_info.value)
        assert "Shape mismatch" in str(exc_info.value)

    def test_prepare_output_fn_with_local_output(self):
        """Test _prepare_output_fn with use_local_output=True."""
        mock_dtensor1 = Mock(spec=DTensor)
        mock_dtensor1.to_local.return_value = torch.randn(4, 8)
        mock_dtensor2 = Mock(spec=DTensor)
        mock_dtensor2.to_local.return_value = torch.randn(4, 8)

        outputs = (mock_dtensor1, mock_dtensor2)
        mod = Mock()
        device_mesh = MockDeviceMesh()

        result = RotaryEmbedParallel._prepare_output_fn(
            True, mod, outputs, device_mesh
        )

        # Should call to_local on both outputs
        assert mock_dtensor1.to_local.called
        assert mock_dtensor2.to_local.called
        assert type(result) == type(outputs)

    def test_prepare_output_fn_without_local_output(self):
        """Test _prepare_output_fn with use_local_output=False."""
        mock_dtensor1 = Mock(spec=DTensor)
        mock_dtensor2 = Mock(spec=DTensor)
        outputs = (mock_dtensor1, mock_dtensor2)
        mod = Mock()
        device_mesh = MockDeviceMesh()

        result = RotaryEmbedParallel._prepare_output_fn(
            False, mod, outputs, device_mesh
        )

        # Should not call to_local
        assert not mock_dtensor1.to_local.called
        assert not mock_dtensor2.to_local.called
        assert result == outputs


class TestParallelizeFunctions:
    """Test suite for model-specific parallelization functions."""

    def test_parallelize_gemma3_causal_basic(self):
        """Test _parallelize_gemma3 with Gemma3ForCausalLM."""
        model = MockModel("gemma3_causal")

        result = _parallelize_gemma3(model, sequence_parallel=False)

        # Should return dict with proper module patterns
        assert isinstance(result, dict)

        # Check expected patterns for CausalLM (uses "model" prefix)
        expected_patterns = [
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.self_attn.o_proj",
            "model.layers.*.mlp.up_proj",
            "model.layers.*.mlp.gate_proj",
            "model.layers.*.mlp.down_proj",
        ]

        for pattern in expected_patterns:
            assert pattern in result

        # Check parallel styles
        assert isinstance(result["model.layers.*.self_attn.q_proj"], ColwiseParallel)
        assert isinstance(result["model.layers.*.self_attn.o_proj"], RowwiseParallel)

    def test_parallelize_gemma3_conditional_basic(self):
        """Test _parallelize_gemma3 with Gemma3ForConditionalGeneration."""
        model = MockModel("gemma3_conditional")

        result = _parallelize_gemma3(model, sequence_parallel=False)

        # Should use "model.language_model" prefix for conditional generation
        expected_patterns = [
            "model.language_model.layers.*.self_attn.q_proj",
            "model.language_model.layers.*.self_attn.k_proj",
            "model.language_model.layers.*.self_attn.v_proj",
            "model.language_model.layers.*.self_attn.o_proj",
        ]

        for pattern in expected_patterns:
            assert pattern in result

    def test_parallelize_gemma3_with_sequence_parallel(self):
        """Test _parallelize_gemma3 with sequence parallelism enabled."""
        model = MockModel("gemma3_causal")

        result = _parallelize_gemma3(model, sequence_parallel=True)

        # Should include additional sequence parallel patterns
        sequence_patterns = [
            "model.embed_tokens",
            "model.rotary_emb",
            "model.rotary_emb_local",
            "model.layers.*.input_layernorm",
            "model.norm",
            "lm_head",
        ]

        for pattern in sequence_patterns:
            assert pattern in result

        # Check specific types for sequence parallel components
        assert isinstance(result["model.embed_tokens"], RowwiseParallel)
        assert isinstance(result["model.rotary_emb"], RotaryEmbedParallel)
        assert isinstance(result["model.layers.*.input_layernorm"], SequenceParallel)

    def test_parallelize_llama_basic(self):
        """Test _parallelize_llama without sequence parallelism."""
        model = MockModel("llama", tie_word_embeddings=False)

        result = _parallelize_llama(model, sequence_parallel=False)

        # Check expected patterns
        expected_patterns = [
            "model.embed_tokens",
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.self_attn.o_proj",
            "model.layers.*.mlp.up_proj",
            "model.layers.*.mlp.gate_proj",
            "model.layers.*.mlp.down_proj",
            "lm_head",
        ]

        for pattern in expected_patterns:
            assert pattern in result

        # Check parallel styles
        assert isinstance(result["model.embed_tokens"], RowwiseParallel)
        assert isinstance(result["lm_head"], ColwiseParallel)

    def test_parallelize_llama_tied_embeddings_works(self):
        """Test _parallelize_llama works with tied embeddings."""
        model = MockModel("llama", tie_word_embeddings=True)

        # Should not raise an error
        result = _parallelize_llama(model, sequence_parallel=False)

        # Should return a valid parallelization plan
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_parallelize_llama_with_sequence_parallel(self):
        """Test _parallelize_llama with sequence parallelism."""
        model = MockModel("llama", tie_word_embeddings=False)

        result = _parallelize_llama(model, sequence_parallel=True)

        # Should include additional sequence parallel patterns
        sequence_patterns = [
            "model.norm",
            "model.layers.*.input_layernorm",
            "model.layers.*.post_attention_layernorm",
        ]

        for pattern in sequence_patterns:
            assert pattern in result

        # Check that embed_tokens has sequence parallel output layout
        embed_tokens = result["model.embed_tokens"]
        assert isinstance(embed_tokens, RowwiseParallel)

    def test_parallelize_qwen_basic(self):
        """Test _parallelize_qwen without sequence parallelism."""
        model = MockModel("qwen2", tie_word_embeddings=False)

        result = _parallelize_qwen(model, sequence_parallel=False)

        # Check expected patterns
        expected_patterns = [
            "lm_head",
            "model.embed_tokens",
            "model.layers.*.self_attn.q_proj",
            "model.layers.*.self_attn.k_proj",
            "model.layers.*.self_attn.v_proj",
            "model.layers.*.self_attn.o_proj",
            "model.layers.*.mlp.up_proj",
            "model.layers.*.mlp.gate_proj",
            "model.layers.*.mlp.down_proj",
        ]

        for pattern in expected_patterns:
            assert pattern in result

    def test_parallelize_qwen_tied_embeddings_works(self):
        """Test _parallelize_qwen works with tied embeddings."""
        model = MockModel("qwen2", tie_word_embeddings=True)

        # Should not raise an error
        result = _parallelize_qwen(model, sequence_parallel=False)

        # Should return a valid parallelization plan
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_parallelize_qwen_with_sequence_parallel(self):
        """Test _parallelize_qwen with sequence parallelism."""
        model = MockModel("qwen2", tie_word_embeddings=False)

        result = _parallelize_qwen(model, sequence_parallel=True)

        # Should include sequence parallel patterns
        sequence_patterns = [
            "model.rotary_emb",
            "model.norm",
            "model.layers.*.input_layernorm",
            "model.layers.*.self_attn.q_norm",
            "model.layers.*.self_attn.k_norm",
            "model.layers.*.post_attention_layernorm",
        ]

        for pattern in sequence_patterns:
            assert pattern in result

        # Check RotaryEmbedParallel is used
        assert isinstance(result["model.rotary_emb"], RotaryEmbedParallel)

        # Check that lm_head has sequence parallel input layout
        lm_head = result["lm_head"]
        assert isinstance(lm_head, ColwiseParallel)

    def test_parallelize_qwen3_with_sequence_parallel(self):
        """Test _parallelize_qwen with Qwen3 and sequence parallelism."""
        model = MockModel("qwen3", tie_word_embeddings=False)

        result = _parallelize_qwen(model, sequence_parallel=True)

        # Should include Qwen3-specific patterns like q_norm and k_norm
        assert "model.layers.*.self_attn.q_norm" in result
        assert "model.layers.*.self_attn.k_norm" in result


class TestParallelizeFunctionsMapping:
    """Test suite for PARALLELIZE_FUNCTIONS mapping."""

    def test_mapping_contains_all_model_types(self):
        """Test that PARALLELIZE_FUNCTIONS contains all expected model types."""
        expected_types = [
            Qwen2ForCausalLM,
            Qwen3ForCausalLM,
            LlamaForCausalLM,
            Gemma3ForCausalLM,
            Gemma3ForConditionalGeneration,
        ]

        for model_type in expected_types:
            assert model_type in PARALLELIZE_FUNCTIONS

    def test_mapping_functions_are_callable(self):
        """Test that all functions in the mapping are callable."""
        for model_type, func in PARALLELIZE_FUNCTIONS.items():
            assert callable(func)

    def test_mapping_functions_return_dict(self):
        """Test that all mapping functions return dictionaries."""
        for model_type, func in PARALLELIZE_FUNCTIONS.items():
            # Create a mock model of the appropriate type
            mock_model = Mock()
            mock_model.__class__ = model_type
            mock_model.config = SimpleNamespace(tie_word_embeddings=False)

            result = func(mock_model, sequence_parallel=False)
            assert isinstance(result, dict)

    def test_qwen2_and_qwen3_use_same_function(self):
        """Test that Qwen2 and Qwen3 models use the same parallelization function."""
        qwen2_func = PARALLELIZE_FUNCTIONS[Qwen2ForCausalLM]
        qwen3_func = PARALLELIZE_FUNCTIONS[Qwen3ForCausalLM]

        assert qwen2_func is qwen3_func
        assert qwen2_func is _parallelize_qwen

    def test_gemma3_models_use_same_function(self):
        """Test that both Gemma3 model types use the same function."""
        causal_func = PARALLELIZE_FUNCTIONS[Gemma3ForCausalLM]
        conditional_func = PARALLELIZE_FUNCTIONS[Gemma3ForConditionalGeneration]

        assert causal_func is conditional_func
        assert causal_func is _parallelize_gemma3


class TestParallelPlanStructure:
    """Test suite for validating parallel plan structure."""

    def test_parallel_plans_have_valid_styles(self):
        """Test that all parallel plans use valid ParallelStyle objects."""
        mock_models = [
            (MockModel("llama", tie_word_embeddings=False), _parallelize_llama),
            (MockModel("qwen2", tie_word_embeddings=False), _parallelize_qwen),
            (MockModel("gemma3_causal"), _parallelize_gemma3),
        ]

        valid_styles = (
            ColwiseParallel,
            RowwiseParallel,
            SequenceParallel,
            PrepareModuleInput,
            PrepareModuleOutput,
            RotaryEmbedParallel,
        )

        for model, func in mock_models:
            # Test without sequence parallel
            plan = func(model, sequence_parallel=False)
            for pattern, style in plan.items():
                assert isinstance(style, valid_styles), (
                    f"Invalid style {type(style)} for pattern {pattern}"
                )

            # Test with sequence parallel
            plan_sp = func(model, sequence_parallel=True)
            for pattern, style in plan_sp.items():
                assert isinstance(style, valid_styles), (
                    f"Invalid style {type(style)} for pattern {pattern} with SP"
                )

    def test_module_patterns_are_strings(self):
        """Test that all module patterns are strings."""
        mock_models = [
            (MockModel("llama", tie_word_embeddings=False), _parallelize_llama),
            (MockModel("qwen2", tie_word_embeddings=False), _parallelize_qwen),
            (MockModel("gemma3_causal"), _parallelize_gemma3),
        ]

        for model, func in mock_models:
            plan = func(model, sequence_parallel=False)
            for pattern in plan.keys():
                assert isinstance(pattern, str), f"Pattern {pattern} is not a string"
                assert len(pattern) > 0, "Pattern cannot be empty"

    def test_sequence_parallel_adds_patterns(self):
        """Test that enabling sequence parallel adds additional patterns."""
        mock_models = [
            (MockModel("llama", tie_word_embeddings=False), _parallelize_llama),
            (MockModel("qwen2", tie_word_embeddings=False), _parallelize_qwen),
            (MockModel("gemma3_causal"), _parallelize_gemma3),
        ]

        for model, func in mock_models:
            plan_basic = func(model, sequence_parallel=False)
            plan_sp = func(model, sequence_parallel=True)

            # Sequence parallel should add patterns, not remove them
            assert len(plan_sp) >= len(plan_basic)

            # All basic patterns should still be present
            for pattern in plan_basic:
                assert pattern in plan_sp


if __name__ == "__main__":
    pytest.main([__file__])

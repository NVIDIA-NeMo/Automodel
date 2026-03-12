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

"""
Unit tests for SimpleAdapter.

SimpleAdapter supports simple transformer models with a basic interface:
- hidden_states: noisy latents
- timestep: timestep values
- encoder_hidden_states: text embeddings

Tests cover:
- Input preparation
- Forward pass
- Text embedding handling
- Shape handling
"""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters import (
    FlowMatchingContext,
    SimpleAdapter,
)


class MockSimpleModel(nn.Module):
    """Mock model that mimics simple transformer interface."""

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_inputs = {}

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        self.call_count += 1
        self.last_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": return_dict,
        }
        # Return prediction with same shape as input
        output = torch.randn_like(hidden_states)
        return (output,)


@pytest.fixture
def simple_adapter():
    """Create a SimpleAdapter instance."""
    return SimpleAdapter()


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return MockSimpleModel()


@pytest.fixture
def sample_context():
    """Create a sample FlowMatchingContext."""
    batch = {
        "video_latents": torch.randn(2, 16, 4, 8, 8),
        "text_embeddings": torch.randn(2, 77, 4096),
    }
    return FlowMatchingContext(
        noisy_latents=torch.randn(2, 16, 4, 8, 8),
        latents=batch["video_latents"],
        timesteps=torch.rand(2) * 1000,
        sigma=torch.rand(2),
        task_type="t2v",
        data_type="video",
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg_dropout_prob=0.0,
        batch=batch,
    )


class TestSimpleAdapterInit:
    """Test SimpleAdapter initialization."""

    def test_adapter_creation(self):
        """Test that SimpleAdapter can be created."""
        adapter = SimpleAdapter()
        assert adapter is not None


class TestSimpleAdapterPrepareInputs:
    """Test SimpleAdapter.prepare_inputs method."""

    def test_prepare_inputs_basic(self, simple_adapter, sample_context):
        """Test basic input preparation."""
        inputs = simple_adapter.prepare_inputs(sample_context)

        assert "hidden_states" in inputs
        assert "timestep" in inputs
        assert "encoder_hidden_states" in inputs

    def test_prepare_inputs_hidden_states_shape(self, simple_adapter, sample_context):
        """Test that hidden_states has correct shape."""
        inputs = simple_adapter.prepare_inputs(sample_context)

        assert inputs["hidden_states"].shape == sample_context.noisy_latents.shape

    def test_prepare_inputs_timestep_dtype(self, simple_adapter, sample_context):
        """Test that timestep has correct dtype."""
        inputs = simple_adapter.prepare_inputs(sample_context)

        assert inputs["timestep"].dtype == sample_context.dtype

    def test_prepare_inputs_text_embeddings_shape(self, simple_adapter, sample_context):
        """Test text embeddings shape."""
        inputs = simple_adapter.prepare_inputs(sample_context)

        assert inputs["encoder_hidden_states"].shape == (2, 77, 4096)

    def test_prepare_inputs_2d_text_embeddings(self, simple_adapter):
        """Test that 2D text embeddings are properly unsqueezed to 3D."""
        batch = {
            "video_latents": torch.randn(16, 4, 8, 8),  # 4D (no batch dim)
            "text_embeddings": torch.randn(77, 4096),  # 2D (no batch dim)
        }

        context = FlowMatchingContext(
            noisy_latents=torch.randn(1, 16, 4, 8, 8),
            latents=batch["video_latents"].unsqueeze(0),
            timesteps=torch.rand(1) * 1000,
            sigma=torch.rand(1),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )

        inputs = simple_adapter.prepare_inputs(context)

        # Should be 3D after unsqueeze
        assert inputs["encoder_hidden_states"].ndim == 3
        assert inputs["encoder_hidden_states"].shape == (1, 77, 4096)

    def test_prepare_inputs_different_batch_sizes(self, simple_adapter):
        """Test input preparation with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            batch = {
                "video_latents": torch.randn(batch_size, 16, 4, 8, 8),
                "text_embeddings": torch.randn(batch_size, 77, 4096),
            }

            context = FlowMatchingContext(
                noisy_latents=torch.randn(batch_size, 16, 4, 8, 8),
                latents=batch["video_latents"],
                timesteps=torch.rand(batch_size) * 1000,
                sigma=torch.rand(batch_size),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=torch.float32,
                cfg_dropout_prob=0.0,
                batch=batch,
            )

            inputs = simple_adapter.prepare_inputs(context)

            assert inputs["hidden_states"].shape[0] == batch_size
            assert inputs["encoder_hidden_states"].shape[0] == batch_size
            assert inputs["timestep"].shape[0] == batch_size

    def test_prepare_inputs_different_dtypes(self, simple_adapter):
        """Test input preparation with different data types."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            batch = {
                "video_latents": torch.randn(2, 16, 4, 8, 8),
                "text_embeddings": torch.randn(2, 77, 4096),
            }

            context = FlowMatchingContext(
                noisy_latents=torch.randn(2, 16, 4, 8, 8),
                latents=batch["video_latents"],
                timesteps=torch.rand(2) * 1000,
                sigma=torch.rand(2),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=dtype,
                cfg_dropout_prob=0.0,
                batch=batch,
            )

            inputs = simple_adapter.prepare_inputs(context)

            assert inputs["timestep"].dtype == dtype
            assert inputs["encoder_hidden_states"].dtype == dtype

    def test_prepare_inputs_preserves_noisy_latents(self, simple_adapter, sample_context):
        """Test that noisy latents are passed through unchanged."""
        inputs = simple_adapter.prepare_inputs(sample_context)

        assert torch.equal(inputs["hidden_states"], sample_context.noisy_latents)


class TestSimpleAdapterForward:
    """Test SimpleAdapter.forward method."""

    def test_forward_basic(self, simple_adapter, mock_model, sample_context):
        """Test basic forward pass."""
        inputs = simple_adapter.prepare_inputs(sample_context)
        output = simple_adapter.forward(mock_model, inputs)

        assert output.shape == inputs["hidden_states"].shape

    def test_forward_calls_model_correctly(self, simple_adapter, mock_model, sample_context):
        """Test that forward pass calls model with correct arguments."""
        inputs = simple_adapter.prepare_inputs(sample_context)
        simple_adapter.forward(mock_model, inputs)

        assert mock_model.call_count == 1
        assert mock_model.last_inputs["return_dict"] is False
        assert torch.equal(mock_model.last_inputs["hidden_states"], inputs["hidden_states"])
        assert torch.equal(mock_model.last_inputs["timestep"], inputs["timestep"])

    def test_forward_output_shape(self, simple_adapter, mock_model):
        """Test forward output shapes for various inputs."""
        shapes = [
            (1, 16, 1, 8, 8),
            (2, 16, 4, 16, 16),
            (4, 32, 8, 32, 32),
        ]

        for shape in shapes:
            batch = {
                "video_latents": torch.randn(shape),
                "text_embeddings": torch.randn(shape[0], 77, 4096),
            }

            context = FlowMatchingContext(
                noisy_latents=torch.randn(shape),
                latents=batch["video_latents"],
                timesteps=torch.rand(shape[0]) * 1000,
                sigma=torch.rand(shape[0]),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=torch.float32,
                cfg_dropout_prob=0.0,
                batch=batch,
            )

            inputs = simple_adapter.prepare_inputs(context)
            output = simple_adapter.forward(mock_model, inputs)

            assert output.shape == shape, f"Shape mismatch for input {shape}"

    def test_forward_with_tuple_output(self, simple_adapter, sample_context):
        """Test that tuple outputs from model are handled correctly."""

        class TupleOutputModel(nn.Module):
            def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
                return (torch.randn_like(hidden_states), "extra_data", {"key": "value"})

        model = TupleOutputModel()
        inputs = simple_adapter.prepare_inputs(sample_context)
        output = simple_adapter.forward(model, inputs)

        # Should extract first element from tuple
        assert output.shape == inputs["hidden_states"].shape

    def test_forward_with_tensor_output(self, simple_adapter, sample_context):
        """Test that direct tensor outputs from model are handled correctly."""

        class TensorOutputModel(nn.Module):
            def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
                return torch.randn_like(hidden_states)

        model = TensorOutputModel()
        inputs = simple_adapter.prepare_inputs(sample_context)
        output = simple_adapter.forward(model, inputs)

        assert output.shape == inputs["hidden_states"].shape


class TestSimpleAdapterEndToEnd:
    """End-to-end tests for SimpleAdapter."""

    def test_full_workflow(self, simple_adapter, mock_model):
        """Test complete workflow from context to output."""
        batch = {
            "video_latents": torch.randn(2, 16, 4, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
        }

        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 16, 4, 8, 8),
            latents=batch["video_latents"],
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )

        # Prepare inputs
        inputs = simple_adapter.prepare_inputs(context)

        # Verify inputs
        assert "hidden_states" in inputs
        assert "timestep" in inputs
        assert "encoder_hidden_states" in inputs

        # Forward pass
        output = simple_adapter.forward(mock_model, inputs)

        # Verify output
        assert output.shape == context.noisy_latents.shape
        assert torch.isfinite(output).all()

    def test_multiple_forward_passes(self, simple_adapter, mock_model):
        """Test multiple consecutive forward passes."""
        for i in range(5):
            batch = {
                "video_latents": torch.randn(2, 16, 4, 8, 8),
                "text_embeddings": torch.randn(2, 77, 4096),
            }

            context = FlowMatchingContext(
                noisy_latents=torch.randn(2, 16, 4, 8, 8),
                latents=batch["video_latents"],
                timesteps=torch.rand(2) * 1000,
                sigma=torch.rand(2),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=torch.float32,
                cfg_dropout_prob=0.0,
                batch=batch,
            )

            inputs = simple_adapter.prepare_inputs(context)
            output = simple_adapter.forward(mock_model, inputs)

            assert output.shape == context.noisy_latents.shape

        assert mock_model.call_count == 5

    def test_with_different_task_types(self, simple_adapter, mock_model):
        """Test adapter with both t2v and i2v task types."""
        for task_type in ["t2v", "i2v"]:
            batch = {
                "video_latents": torch.randn(2, 16, 4, 8, 8),
                "text_embeddings": torch.randn(2, 77, 4096),
            }

            context = FlowMatchingContext(
                noisy_latents=torch.randn(2, 16, 4, 8, 8),
                latents=batch["video_latents"],
                timesteps=torch.rand(2) * 1000,
                sigma=torch.rand(2),
                task_type=task_type,
                data_type="video",
                device=torch.device("cpu"),
                dtype=torch.float32,
                cfg_dropout_prob=0.0,
                batch=batch,
            )

            inputs = simple_adapter.prepare_inputs(context)
            output = simple_adapter.forward(mock_model, inputs)

            # SimpleAdapter doesn't use condition latents, so output shape should match
            assert output.shape == context.noisy_latents.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

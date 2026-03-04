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
Unit tests for HunyuanAdapter.

HunyuanAdapter supports HunyuanVideo 1.5 style models with:
- Dual text encoders with attention masks
- Image embeddings for i2v conditioning
- Condition latents concatenation

Tests cover:
- Initialization with various parameters
- Input preparation with all optional fields
- Condition latents handling
- Image embeddings handling
- Forward pass
"""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters import (
    FlowMatchingContext,
    HunyuanAdapter,
)


# =============================================================================
# Mock Model
# =============================================================================


class MockHunyuanModel(nn.Module):
    """Mock model that mimics HunyuanVideo interface."""

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_inputs = {}

    def forward(
        self,
        latents,
        timesteps,
        encoder_hidden_states,
        encoder_attention_mask=None,
        encoder_hidden_states_2=None,
        encoder_attention_mask_2=None,
        image_embeds=None,
        return_dict=False,
    ):
        self.call_count += 1
        self.last_inputs = {
            "latents": latents,
            "timesteps": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_hidden_states_2": encoder_hidden_states_2,
            "encoder_attention_mask_2": encoder_attention_mask_2,
            "image_embeds": image_embeds,
            "return_dict": return_dict,
        }

        # Return prediction - note that latents may have condition channels
        # so we need to return prediction for original channels only
        if latents.shape[1] > 16:
            # Has condition latents concatenated, return only original channels
            output = torch.randn(latents.shape[0], 16, *latents.shape[2:])
        else:
            output = torch.randn_like(latents)
        return (output,)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def hunyuan_adapter():
    """Create a HunyuanAdapter with default settings."""
    return HunyuanAdapter()


@pytest.fixture
def hunyuan_adapter_no_condition():
    """Create a HunyuanAdapter without condition latents."""
    return HunyuanAdapter(use_condition_latents=False)


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return MockHunyuanModel()


@pytest.fixture
def sample_batch():
    """Create a sample batch with all fields."""
    return {
        "video_latents": torch.randn(2, 16, 4, 8, 8),
        "text_embeddings": torch.randn(2, 77, 4096),
        "text_mask": torch.ones(2, 77),
        "text_embeddings_2": torch.randn(2, 128, 2048),
        "text_mask_2": torch.ones(2, 128),
        "image_embeds": torch.randn(2, 729, 1152),
    }


@pytest.fixture
def minimal_batch():
    """Create a minimal batch with required fields only."""
    return {
        "video_latents": torch.randn(2, 16, 4, 8, 8),
        "text_embeddings": torch.randn(2, 77, 4096),
    }


def create_context(batch, task_type="t2v", data_type="video"):
    """Helper to create FlowMatchingContext."""
    return FlowMatchingContext(
        noisy_latents=torch.randn(batch["video_latents"].shape),
        latents=batch["video_latents"],
        timesteps=torch.rand(batch["video_latents"].shape[0]) * 1000,
        sigma=torch.rand(batch["video_latents"].shape[0]),
        task_type=task_type,
        data_type=data_type,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg_dropout_prob=0.0,
        batch=batch,
    )


# =============================================================================
# HunyuanAdapter Initialization Tests
# =============================================================================


class TestHunyuanAdapterInit:
    """Test HunyuanAdapter initialization."""

    def test_default_initialization(self):
        """Test adapter creation with default values."""
        adapter = HunyuanAdapter()

        assert adapter.default_image_embed_shape == (729, 1152)
        assert adapter.use_condition_latents is True
        print("✓ HunyuanAdapter default initialization test passed")

    def test_custom_image_embed_shape(self):
        """Test adapter with custom image embed shape."""
        adapter = HunyuanAdapter(default_image_embed_shape=(512, 768))

        assert adapter.default_image_embed_shape == (512, 768)
        print("✓ HunyuanAdapter custom image embed shape test passed")

    def test_disable_condition_latents(self):
        """Test adapter with condition latents disabled."""
        adapter = HunyuanAdapter(use_condition_latents=False)

        assert adapter.use_condition_latents is False
        print("✓ HunyuanAdapter disable condition latents test passed")


# =============================================================================
# HunyuanAdapter Input Preparation Tests
# =============================================================================


class TestHunyuanAdapterPrepareInputs:
    """Test HunyuanAdapter.prepare_inputs method."""

    def test_prepare_inputs_all_fields(self, hunyuan_adapter, sample_batch):
        """Test input preparation with all optional fields present."""
        context = create_context(sample_batch)
        inputs = hunyuan_adapter.prepare_inputs(context)

        expected_keys = [
            "latents",
            "timesteps",
            "encoder_hidden_states",
            "encoder_attention_mask",
            "encoder_hidden_states_2",
            "encoder_attention_mask_2",
            "image_embeds",
        ]

        for key in expected_keys:
            assert key in inputs, f"Missing key: {key}"

        print("✓ HunyuanAdapter prepare_inputs all fields test passed")

    def test_prepare_inputs_minimal_batch(self, hunyuan_adapter, minimal_batch):
        """Test input preparation with minimal batch (required fields only)."""
        context = create_context(minimal_batch)
        inputs = hunyuan_adapter.prepare_inputs(context)

        # Required fields should be present
        assert inputs["encoder_hidden_states"] is not None
        assert inputs["latents"] is not None

        # Optional masks should be None
        assert inputs["encoder_attention_mask"] is None
        assert inputs["encoder_hidden_states_2"] is None
        assert inputs["encoder_attention_mask_2"] is None

        # Image embeds should be zeros (default)
        assert inputs["image_embeds"] is not None
        assert torch.allclose(inputs["image_embeds"], torch.zeros_like(inputs["image_embeds"]))

        print("✓ HunyuanAdapter prepare_inputs minimal batch test passed")

    def test_prepare_inputs_latents_with_condition(self, hunyuan_adapter, sample_batch):
        """Test that latents include condition channels when enabled."""
        context = create_context(sample_batch, task_type="t2v")
        inputs = hunyuan_adapter.prepare_inputs(context)

        # With condition latents: C + (C+1) = 16 + 17 = 33 channels
        expected_channels = 16 + 17  # original + condition (with mask channel)
        assert inputs["latents"].shape[1] == expected_channels
        print("✓ HunyuanAdapter latents with condition test passed")

    def test_prepare_inputs_latents_without_condition(self, hunyuan_adapter_no_condition, sample_batch):
        """Test latents without condition channels."""
        context = create_context(sample_batch)
        inputs = hunyuan_adapter_no_condition.prepare_inputs(context)

        # Without condition latents: just original channels
        assert inputs["latents"].shape[1] == 16
        print("✓ HunyuanAdapter latents without condition test passed")

    def test_prepare_inputs_i2v_task(self, hunyuan_adapter, sample_batch):
        """Test input preparation for i2v task with image embeddings."""
        context = create_context(sample_batch, task_type="i2v")
        inputs = hunyuan_adapter.prepare_inputs(context)

        # Image embeddings should be from batch (not zeros)
        assert inputs["image_embeds"].shape == (2, 729, 1152)
        # Should NOT be all zeros since we have image_embeds in batch
        assert not torch.allclose(inputs["image_embeds"], torch.zeros_like(inputs["image_embeds"]))
        print("✓ HunyuanAdapter i2v task test passed")

    def test_prepare_inputs_i2v_without_image_embeds(self, hunyuan_adapter, minimal_batch):
        """Test i2v task without image embeddings (uses default zeros)."""
        context = create_context(minimal_batch, task_type="i2v")
        inputs = hunyuan_adapter.prepare_inputs(context)

        # Should create default zeros
        assert inputs["image_embeds"].shape == (2, 729, 1152)
        assert torch.allclose(inputs["image_embeds"], torch.zeros_like(inputs["image_embeds"]))
        print("✓ HunyuanAdapter i2v without image embeds test passed")

    def test_prepare_inputs_t2v_ignores_image_embeds(self, hunyuan_adapter, sample_batch):
        """Test that t2v task ignores image embeddings even if present."""
        context = create_context(sample_batch, task_type="t2v")
        inputs = hunyuan_adapter.prepare_inputs(context)

        # For t2v, should use default zeros, not the batch image_embeds
        assert torch.allclose(inputs["image_embeds"], torch.zeros_like(inputs["image_embeds"]))
        print("✓ HunyuanAdapter t2v ignores image embeds test passed")

    def test_prepare_inputs_custom_image_embed_shape(self):
        """Test with custom default image embed shape."""
        adapter = HunyuanAdapter(default_image_embed_shape=(256, 512))
        batch = {
            "video_latents": torch.randn(2, 16, 4, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
        }
        context = create_context(batch)

        inputs = adapter.prepare_inputs(context)

        assert inputs["image_embeds"].shape == (2, 256, 512)
        print("✓ HunyuanAdapter custom image embed shape test passed")

    def test_prepare_inputs_2d_text_embeddings(self, hunyuan_adapter):
        """Test that 2D text embeddings are properly unsqueezed."""
        batch = {
            "video_latents": torch.randn(16, 4, 8, 8),  # 4D
            "text_embeddings": torch.randn(77, 4096),  # 2D
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

        inputs = hunyuan_adapter.prepare_inputs(context)

        assert inputs["encoder_hidden_states"].ndim == 3
        assert inputs["encoder_hidden_states"].shape == (1, 77, 4096)
        print("✓ HunyuanAdapter 2D text embeddings handling test passed")

    def test_prepare_inputs_dtype_conversion(self, hunyuan_adapter, sample_batch):
        """Test that inputs are converted to correct dtype."""
        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 16, 4, 8, 8),
            latents=sample_batch["video_latents"],
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            cfg_dropout_prob=0.0,
            batch=sample_batch,
        )

        inputs = hunyuan_adapter.prepare_inputs(context)

        assert inputs["timesteps"].dtype == torch.bfloat16
        assert inputs["encoder_hidden_states"].dtype == torch.bfloat16
        assert inputs["image_embeds"].dtype == torch.bfloat16
        print("✓ HunyuanAdapter dtype conversion test passed")


# =============================================================================
# HunyuanAdapter Condition Latents Tests
# =============================================================================


class TestHunyuanAdapterConditionLatents:
    """Test condition latents handling in HunyuanAdapter."""

    def test_condition_latents_t2v(self, hunyuan_adapter, sample_batch):
        """Test condition latents for t2v task."""
        context = create_context(sample_batch, task_type="t2v")
        inputs = hunyuan_adapter.prepare_inputs(context)

        # Latents should be: [noisy_latents, condition_latents]
        # condition_latents for t2v should be zeros with mask channel
        assert inputs["latents"].shape == (2, 33, 4, 8, 8)

        # The condition part (last 17 channels) should be zeros for t2v
        condition_part = inputs["latents"][:, 16:]
        assert torch.allclose(condition_part, torch.zeros_like(condition_part))
        print("✓ HunyuanAdapter condition latents t2v test passed")

    def test_condition_latents_i2v(self, hunyuan_adapter, sample_batch):
        """Test condition latents for i2v task."""
        context = create_context(sample_batch, task_type="i2v")
        inputs = hunyuan_adapter.prepare_inputs(context)

        assert inputs["latents"].shape == (2, 33, 4, 8, 8)

        # For i2v, first frame should be copied to condition
        # The condition mask channel should be 1 for first frame
        condition_mask = inputs["latents"][:, -1, 0]  # Last channel, first frame
        assert (condition_mask == 1).all()
        print("✓ HunyuanAdapter condition latents i2v test passed")


# =============================================================================
# HunyuanAdapter Forward Tests
# =============================================================================


class TestHunyuanAdapterForward:
    """Test HunyuanAdapter.forward method."""

    def test_forward_basic(self, hunyuan_adapter, mock_model, sample_batch):
        """Test basic forward pass."""
        context = create_context(sample_batch)
        inputs = hunyuan_adapter.prepare_inputs(context)
        output = hunyuan_adapter.forward(mock_model, inputs)

        assert output.shape == (2, 16, 4, 8, 8)
        print("✓ HunyuanAdapter forward basic test passed")

    def test_forward_calls_model_correctly(self, hunyuan_adapter, mock_model, sample_batch):
        """Test that forward calls model with correct arguments."""
        context = create_context(sample_batch)
        inputs = hunyuan_adapter.prepare_inputs(context)
        hunyuan_adapter.forward(mock_model, inputs)

        assert mock_model.call_count == 1
        assert mock_model.last_inputs["return_dict"] is False
        assert mock_model.last_inputs["encoder_hidden_states"] is not None
        assert mock_model.last_inputs["image_embeds"] is not None
        print("✓ HunyuanAdapter forward calls model correctly test passed")

    def test_forward_passes_all_inputs(self, hunyuan_adapter, mock_model, sample_batch):
        """Test that all inputs are passed to the model."""
        context = create_context(sample_batch)
        inputs = hunyuan_adapter.prepare_inputs(context)
        hunyuan_adapter.forward(mock_model, inputs)

        # Check all inputs were passed
        assert mock_model.last_inputs["latents"] is not None
        assert mock_model.last_inputs["timesteps"] is not None
        assert mock_model.last_inputs["encoder_hidden_states"] is not None
        assert mock_model.last_inputs["encoder_attention_mask"] is not None
        assert mock_model.last_inputs["encoder_hidden_states_2"] is not None
        assert mock_model.last_inputs["encoder_attention_mask_2"] is not None
        assert mock_model.last_inputs["image_embeds"] is not None
        print("✓ HunyuanAdapter forward passes all inputs test passed")

    def test_forward_with_tuple_output(self, hunyuan_adapter, sample_batch):
        """Test handling of tuple output from model."""

        class TupleOutputModel(nn.Module):
            def forward(self, latents, timesteps, **kwargs):
                return (torch.randn(latents.shape[0], 16, *latents.shape[2:]), "extra")

        model = TupleOutputModel()
        context = create_context(sample_batch)
        inputs = hunyuan_adapter.prepare_inputs(context)
        output = hunyuan_adapter.forward(model, inputs)

        assert output.shape == (2, 16, 4, 8, 8)
        print("✓ HunyuanAdapter forward tuple output test passed")


# =============================================================================
# HunyuanAdapter End-to-End Tests
# =============================================================================


class TestHunyuanAdapterEndToEnd:
    """End-to-end tests for HunyuanAdapter."""

    def test_full_workflow_t2v(self, hunyuan_adapter, mock_model, sample_batch):
        """Test complete workflow for t2v task."""
        context = create_context(sample_batch, task_type="t2v")

        inputs = hunyuan_adapter.prepare_inputs(context)
        output = hunyuan_adapter.forward(mock_model, inputs)

        assert output.shape == context.noisy_latents.shape
        assert torch.isfinite(output).all()
        print("✓ HunyuanAdapter full workflow t2v test passed")

    def test_full_workflow_i2v(self, hunyuan_adapter, mock_model, sample_batch):
        """Test complete workflow for i2v task."""
        context = create_context(sample_batch, task_type="i2v")

        inputs = hunyuan_adapter.prepare_inputs(context)
        output = hunyuan_adapter.forward(mock_model, inputs)

        assert output.shape == context.noisy_latents.shape
        assert torch.isfinite(output).all()
        print("✓ HunyuanAdapter full workflow i2v test passed")

    def test_different_batch_sizes(self, hunyuan_adapter, mock_model):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            batch = {
                "video_latents": torch.randn(batch_size, 16, 4, 8, 8),
                "text_embeddings": torch.randn(batch_size, 77, 4096),
            }

            context = create_context(batch)
            inputs = hunyuan_adapter.prepare_inputs(context)
            output = hunyuan_adapter.forward(mock_model, inputs)

            assert output.shape[0] == batch_size

        print("✓ HunyuanAdapter different batch sizes test passed")

    def test_different_video_shapes(self, hunyuan_adapter, mock_model):
        """Test with different video shapes."""
        shapes = [
            (2, 16, 1, 8, 8),  # Single frame
            (2, 16, 8, 16, 16),  # Longer video, larger spatial
            (2, 16, 4, 32, 32),  # Large spatial
        ]

        for shape in shapes:
            batch = {
                "video_latents": torch.randn(shape),
                "text_embeddings": torch.randn(shape[0], 77, 4096),
            }

            context = create_context(batch)
            inputs = hunyuan_adapter.prepare_inputs(context)
            output = hunyuan_adapter.forward(mock_model, inputs)

            assert output.shape == shape

        print("✓ HunyuanAdapter different video shapes test passed")

    def test_multiple_forward_passes(self, hunyuan_adapter, mock_model):
        """Test multiple consecutive forward passes."""
        for i in range(5):
            batch = {
                "video_latents": torch.randn(2, 16, 4, 8, 8),
                "text_embeddings": torch.randn(2, 77, 4096),
            }

            context = create_context(batch)
            inputs = hunyuan_adapter.prepare_inputs(context)
            output = hunyuan_adapter.forward(mock_model, inputs)

            assert output.shape == (2, 16, 4, 8, 8)

        assert mock_model.call_count == 5
        print("✓ HunyuanAdapter multiple forward passes test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

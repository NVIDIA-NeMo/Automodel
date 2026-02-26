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
Unit tests for ModelAdapter base class and FlowMatchingContext dataclass.

Tests cover:
- FlowMatchingContext creation and attributes
- ModelAdapter base class functionality (post_process_prediction)
"""

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters import FlowMatchingContext


# =============================================================================
# FlowMatchingContext Tests
# =============================================================================


class TestFlowMatchingContext:
    """Test FlowMatchingContext dataclass."""

    def test_context_creation(self):
        """Test creating a FlowMatchingContext with all required fields."""
        batch = {"video_latents": torch.randn(2, 16, 4, 8, 8)}

        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 16, 4, 8, 8),
            latents=torch.randn(2, 16, 4, 8, 8),
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )

        assert context.task_type == "t2v"
        assert context.data_type == "video"
        assert context.noisy_latents.shape == (2, 16, 4, 8, 8)
        assert context.device == torch.device("cpu")
        assert context.dtype == torch.float32

    def test_context_with_i2v_task(self):
        """Test creating context for i2v task."""
        batch = {"video_latents": torch.randn(2, 16, 4, 8, 8)}

        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 16, 4, 8, 8),
            latents=torch.randn(2, 16, 4, 8, 8),
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="i2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )

        assert context.task_type == "i2v"

    def test_context_with_image_data(self):
        """Test creating context for image data."""
        batch = {"video_latents": torch.randn(2, 16, 1, 8, 8)}

        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 16, 1, 8, 8),
            latents=torch.randn(2, 16, 1, 8, 8),
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="t2v",
            data_type="image",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )

        assert context.data_type == "image"

    def test_context_batch_access(self):
        """Test accessing batch data through context."""
        batch = {
            "video_latents": torch.randn(2, 16, 4, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
            "custom_key": "custom_value",
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

        assert "text_embeddings" in context.batch
        assert context.batch["custom_key"] == "custom_value"

    def test_context_tensor_shapes(self):
        """Test various tensor shapes in context."""
        shapes = [
            (1, 16, 1, 8, 8),  # Single frame, single batch
            (4, 16, 8, 16, 16),  # Multiple frames, larger spatial
            (2, 32, 4, 32, 32),  # Different channel count
        ]

        for shape in shapes:
            batch = {"video_latents": torch.randn(shape)}
            context = FlowMatchingContext(
                noisy_latents=torch.randn(shape),
                latents=torch.randn(shape),
                timesteps=torch.rand(shape[0]) * 1000,
                sigma=torch.rand(shape[0]),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=torch.float32,
                cfg_dropout_prob=0.0,
                batch=batch,
            )

            assert context.noisy_latents.shape == shape
            assert context.latents.shape == shape
            assert context.timesteps.shape == (shape[0],)
            assert context.sigma.shape == (shape[0],)

    def test_context_different_dtypes(self):
        """Test context with different data types."""
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            batch = {"video_latents": torch.randn(2, 16, 4, 8, 8)}
            context = FlowMatchingContext(
                noisy_latents=torch.randn(2, 16, 4, 8, 8),
                latents=torch.randn(2, 16, 4, 8, 8),
                timesteps=torch.rand(2) * 1000,
                sigma=torch.rand(2),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=dtype,
                cfg_dropout_prob=0.0,
                batch=batch,
            )

            assert context.dtype == dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

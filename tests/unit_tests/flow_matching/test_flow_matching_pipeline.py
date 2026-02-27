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
Unit tests for FlowMatchingPipeline and related components:
- LinearInterpolationSchedule
- FlowMatchingPipeline class
- Factory functions (create_adapter, create_pipeline)
"""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters import (
    HunyuanAdapter,
    SimpleAdapter,
)
from nemo_automodel.components.flow_matching.pipeline import (
    FlowMatchingPipeline,
    LinearInterpolationSchedule,
    create_adapter,
    create_pipeline,
)


# =============================================================================
# Fixtures
# =============================================================================


class MockModel(nn.Module):
    """Mock model that returns predictions with same shape as input."""

    def __init__(self, output_scale=1.0):
        super().__init__()
        self.output_scale = output_scale
        self.linear = nn.Linear(1, 1)  # Dummy parameter for gradient tests

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
        # Return prediction with same shape as input hidden_states
        output = torch.randn_like(hidden_states) * self.output_scale
        # Add small scaled version of input to maintain gradient connection
        output = output + hidden_states * 0.0
        return (output,)


@pytest.fixture
def mock_model():
    """Create a mock model that returns predictions with gradients."""
    return MockModel()


@pytest.fixture
def simple_adapter():
    """Create a SimpleAdapter instance."""
    return SimpleAdapter()


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "video_latents": torch.randn(2, 16, 4, 8, 8),  # (B, C, T, H, W)
        "text_embeddings": torch.randn(2, 77, 4096),  # (B, seq_len, dim)
        "data_type": "video",
    }


@pytest.fixture
def image_batch():
    """Create a sample image batch for testing."""
    return {
        "video_latents": torch.randn(2, 16, 1, 8, 8),  # (B, C, T=1, H, W)
        "text_embeddings": torch.randn(2, 77, 4096),
        "data_type": "image",
    }


@pytest.fixture
def pipeline(simple_adapter):
    """Create a default FlowMatchingPipeline."""
    return FlowMatchingPipeline(
        model_adapter=simple_adapter,
        num_train_timesteps=1000,
        flow_shift=3.0,
        timestep_sampling="uniform",
        log_interval=1000,  # Reduce logging during tests
        summary_log_interval=1000,
    )


class TestLinearInterpolationSchedule:
    """Test the linear interpolation noise schedule."""

    def test_interpolation_at_sigma_zero(self):
        """At sigma=0, x_t should equal x_0 (clean latents)."""
        schedule = LinearInterpolationSchedule()
        x0 = torch.randn(2, 16, 4, 8, 8)
        x1 = torch.randn_like(x0)  # noise
        sigma = torch.zeros(2)

        x_t = schedule.forward(x0, x1, sigma)

        assert torch.allclose(x_t, x0, atol=1e-6), "At sigma=0, x_t should equal x_0"

    def test_interpolation_at_sigma_one(self):
        """At sigma=1, x_t should equal x_1 (noise)."""
        schedule = LinearInterpolationSchedule()
        x0 = torch.randn(2, 16, 4, 8, 8)
        x1 = torch.randn_like(x0)  # noise
        sigma = torch.ones(2)

        x_t = schedule.forward(x0, x1, sigma)

        assert torch.allclose(x_t, x1, atol=1e-6), "At sigma=1, x_t should equal x_1"

    def test_interpolation_at_intermediate_values(self):
        """Test interpolation at various sigma values."""
        schedule = LinearInterpolationSchedule()
        x0 = torch.randn(4, 16, 4, 8, 8)
        x1 = torch.randn_like(x0)

        for sigma_val in [0.25, 0.5, 0.75]:
            sigma = torch.full((4,), sigma_val)
            x_t = schedule.forward(x0, x1, sigma)

            # Manual calculation
            expected = (1.0 - sigma_val) * x0 + sigma_val * x1

            assert torch.allclose(x_t, expected, atol=1e-5), f"Interpolation at sigma={sigma_val} failed"

    def test_interpolation_shape_preservation(self):
        """Test that output shape matches input shape."""
        schedule = LinearInterpolationSchedule()

        shapes = [
            (1, 16, 1, 8, 8),
            (2, 16, 4, 16, 16),
            (4, 32, 8, 32, 32),
        ]

        for shape in shapes:
            x0 = torch.randn(shape)
            x1 = torch.randn_like(x0)
            sigma = torch.rand(shape[0])

            x_t = schedule.forward(x0, x1, sigma)

            assert x_t.shape == x0.shape, f"Shape mismatch: {x_t.shape} vs {x0.shape}"

    def test_interpolation_finite_values(self):
        """Test that interpolation produces finite values."""
        schedule = LinearInterpolationSchedule()
        x0 = torch.randn(2, 16, 4, 8, 8)
        x1 = torch.randn_like(x0)
        sigma = torch.rand(2)

        x_t = schedule.forward(x0, x1, sigma)

        assert torch.isfinite(x_t).all(), "Interpolated values should be finite"

    def test_interpolation_broadcast_sigma(self):
        """Test that sigma is properly broadcast across dimensions."""
        schedule = LinearInterpolationSchedule()
        x0 = torch.zeros(2, 16, 4, 8, 8)
        x1 = torch.ones(2, 16, 4, 8, 8)

        # Different sigma for each batch element
        sigma = torch.tensor([0.3, 0.7])

        x_t = schedule.forward(x0, x1, sigma)

        # Check that batch elements have different values
        assert torch.allclose(x_t[0], torch.full_like(x_t[0], 0.3), atol=1e-5)
        assert torch.allclose(x_t[1], torch.full_like(x_t[1], 0.7), atol=1e-5)


class TestTimestepSampling:
    """Test various timestep sampling strategies."""

    def test_uniform_sampling(self, simple_adapter):
        """Test uniform timestep sampling."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            timestep_sampling="uniform",
            flow_shift=3.0,
        )

        batch_size = 100
        sigma, timesteps, method = pipeline.sample_timesteps(batch_size, torch.device("cpu"))

        assert method == "uniform"
        assert sigma.shape == (batch_size,)
        assert timesteps.shape == (batch_size,)
        assert (sigma >= 0).all() and (sigma <= 1).all(), "Sigma should be in [0, 1]"
        assert (timesteps >= 0).all() and (timesteps <= 1000).all(), "Timesteps should be in [0, T]"

    def test_logit_normal_sampling(self, simple_adapter):
        """Test logit-normal timestep sampling."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            timestep_sampling="logit_normal",
            logit_mean=0.0,
            logit_std=1.0,
            mix_uniform_ratio=0.0,  # Disable uniform mixing
        )

        batch_size = 100
        sigma, timesteps, method = pipeline.sample_timesteps(batch_size, torch.device("cpu"))

        assert method == "logit_normal"
        assert sigma.shape == (batch_size,)
        assert (sigma >= 0).all() and (sigma <= 1).all()

    def test_mode_sampling(self, simple_adapter):
        """Test mode-based timestep sampling."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            timestep_sampling="mode",
            mix_uniform_ratio=0.0,
        )

        batch_size = 100
        sigma, timesteps, method = pipeline.sample_timesteps(batch_size, torch.device("cpu"))

        assert method == "mode"
        assert sigma.shape == (batch_size,)
        assert (sigma >= 0).all() and (sigma <= 1).all()

    def test_lognorm_sampling(self, simple_adapter):
        """Test log-normal timestep sampling."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            timestep_sampling="lognorm",
            mix_uniform_ratio=0.0,
        )

        batch_size = 100
        sigma, timesteps, method = pipeline.sample_timesteps(batch_size, torch.device("cpu"))

        assert method == "lognorm"
        assert sigma.shape == (batch_size,)
        assert (sigma >= 0).all() and (sigma <= 1).all()

    def test_mix_sampling_strategy(self, simple_adapter):
        """Test mixed uniform sampling ratio."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            timestep_sampling="logit_normal",
            mix_uniform_ratio=0.5,
        )

        methods_seen = set()
        for _ in range(50):
            _, _, method = pipeline.sample_timesteps(2, torch.device("cpu"))
            methods_seen.add(method)

    def test_flow_shift_transformation(self, simple_adapter):
        """Test flow shift transformation: σ = shift / (shift + (1/u - 1))."""
        shift = 3.0
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            timestep_sampling="uniform",
            flow_shift=shift,
        )

        # Sample multiple times to check transformation
        torch.manual_seed(42)
        for _ in range(10):
            sigma, _, _ = pipeline.sample_timesteps(100, torch.device("cpu"))

            # With flow shift > 1, sigma distribution is biased toward higher values
            # This is because σ = shift / (shift + (1/u - 1)) where u ∈ [0, 1]
            assert (sigma >= 0).all() and (sigma <= 1).all()

    def test_sigma_clamping(self, simple_adapter):
        """Test sigma clamping for finetuning."""
        sigma_min, sigma_max = 0.1, 0.8

        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            timestep_sampling="uniform",
            flow_shift=3.0,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        batch_size = 100
        sigma, _, _ = pipeline.sample_timesteps(batch_size, torch.device("cpu"))

        tolerance = 1e-6
        assert (sigma >= sigma_min - tolerance).all(), f"Sigma should be >= {sigma_min}"
        assert (sigma <= sigma_max + tolerance).all(), f"Sigma should be <= {sigma_max}"

    def test_different_flow_shift_values(self, simple_adapter):
        """Test with different flow shift values."""
        for shift in [1.0, 2.0, 3.0, 5.0, 10.0]:
            pipeline = FlowMatchingPipeline(
                model_adapter=simple_adapter,
                timestep_sampling="uniform",
                flow_shift=shift,
            )

            sigma, timesteps, _ = pipeline.sample_timesteps(50, torch.device("cpu"))

            assert (sigma >= 0).all() and (sigma <= 1).all()
            assert (timesteps >= 0).all() and (timesteps <= 1000).all()


class TestLossComputation:
    """Test loss computation with flow matching."""

    def test_loss_weighting_enabled(self, simple_adapter):
        """Test loss computation with weighting enabled."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            flow_shift=3.0,
            use_loss_weighting=True,
        )

        model_pred = torch.randn(2, 16, 4, 8, 8)
        target = torch.randn_like(model_pred)
        sigma = torch.tensor([0.3, 0.7])
        batch = {}

        # Returns 6 values for megatron compatibility
        weighted_loss, average_weighted_loss, unweighted_loss, average_unweighted_loss, loss_weight, _ = (
            pipeline.compute_loss(model_pred, target, sigma, batch)
        )

        # Verify shapes - average losses should be scalar
        assert average_weighted_loss.ndim == 0, "Average weighted loss should be scalar"
        assert average_unweighted_loss.ndim == 0, "Average unweighted loss should be scalar"

        # Verify weight formula: w = 1 + shift * σ
        expected_weights = 1.0 + 3.0 * sigma
        assert torch.allclose(loss_weight.squeeze(), expected_weights.view(-1, 1, 1, 1, 1).squeeze(), atol=1e-5)

        print(f"✓ Loss weighting enabled test passed - w: {loss_weight.squeeze()}")

    def test_loss_weighting_disabled(self, simple_adapter):
        """Test loss computation with weighting disabled."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            flow_shift=3.0,
            use_loss_weighting=False,
        )

        model_pred = torch.randn(2, 16, 4, 8, 8)
        target = torch.randn_like(model_pred)
        sigma = torch.tensor([0.3, 0.7])
        batch = {}

        weighted_loss, _, unweighted_loss, _, loss_weight, _ = pipeline.compute_loss(model_pred, target, sigma, batch)

        # Without weighting, weighted loss should equal unweighted loss
        assert torch.allclose(weighted_loss, unweighted_loss, atol=1e-6)

        # All weights should be 1
        assert torch.allclose(loss_weight, torch.ones_like(loss_weight))

    def test_loss_weight_formula(self, simple_adapter):
        """Test that loss weight follows formula: w = 1 + shift * σ."""
        flow_shift = 3.0
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            flow_shift=flow_shift,
            use_loss_weighting=True,
        )

        model_pred = torch.zeros(4, 16, 4, 8, 8)
        target = torch.ones_like(model_pred)
        batch = {}

        for sigma_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sigma = torch.full((4,), sigma_val)
            _, _, _, _, loss_weight, _ = pipeline.compute_loss(model_pred, target, sigma, batch)

            expected_weight = 1.0 + flow_shift * sigma_val
            actual_weight = loss_weight[0, 0, 0, 0, 0].item()

            assert abs(actual_weight - expected_weight) < 1e-5, (
                f"Weight mismatch at σ={sigma_val}: got {actual_weight}, expected {expected_weight}"
            )

    def test_loss_is_non_negative(self, simple_adapter):
        """Test that computed loss is non-negative."""
        pipeline = FlowMatchingPipeline(model_adapter=simple_adapter)

        model_pred = torch.randn(2, 16, 4, 8, 8)
        target = torch.randn_like(model_pred)
        sigma = torch.rand(2)
        batch = {}

        _, average_weighted_loss, _, average_unweighted_loss, _, _ = pipeline.compute_loss(
            model_pred, target, sigma, batch
        )

        assert average_weighted_loss >= 0, "Weighted loss should be non-negative"
        assert average_unweighted_loss >= 0, "Unweighted loss should be non-negative"

    def test_loss_is_finite(self, simple_adapter):
        """Test that computed loss is finite."""
        pipeline = FlowMatchingPipeline(model_adapter=simple_adapter)

        model_pred = torch.randn(2, 16, 4, 8, 8)
        target = torch.randn_like(model_pred)
        sigma = torch.rand(2)
        batch = {}

        _, average_weighted_loss, _, average_unweighted_loss, _, _ = pipeline.compute_loss(
            model_pred, target, sigma, batch
        )

        assert torch.isfinite(average_weighted_loss), "Weighted loss should be finite"
        assert torch.isfinite(average_unweighted_loss), "Unweighted loss should be finite"

    def test_loss_mse_correctness(self, simple_adapter):
        """Test that base loss is MSE."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            use_loss_weighting=False,
        )

        model_pred = torch.randn(2, 16, 4, 8, 8)
        target = torch.randn_like(model_pred)
        sigma = torch.rand(2)
        batch = {}

        _, _, _, average_unweighted_loss, _, _ = pipeline.compute_loss(model_pred, target, sigma, batch)

        # Manual MSE calculation
        expected_mse = nn.functional.mse_loss(model_pred.float(), target.float())

        assert torch.allclose(average_unweighted_loss, expected_mse, atol=1e-6)


class TestFullTrainingStep:
    """Test the complete training step."""

    def test_basic_training_step(self, pipeline, mock_model, sample_batch):
        """Test basic training step execution."""
        device = torch.device("cpu")
        dtype = torch.bfloat16

        # Returns: weighted_loss, average_weighted_loss, loss_mask, metrics
        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, sample_batch, device, dtype, global_step=0
        )

        # Verify loss
        assert isinstance(weighted_loss, torch.Tensor), "Weighted loss should be a tensor"
        assert isinstance(average_weighted_loss, torch.Tensor), "Average weighted loss should be a tensor"
        assert average_weighted_loss.ndim == 0, "Average weighted loss should be scalar"
        assert not torch.isnan(average_weighted_loss), "Loss should not be NaN"
        assert torch.isfinite(average_weighted_loss), "Loss should be finite"

        # Verify metrics
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert "loss" in metrics
        assert "sigma_min" in metrics
        assert "sigma_max" in metrics
        assert "weight_min" in metrics
        assert "weight_max" in metrics
        assert "timestep_min" in metrics
        assert "timestep_max" in metrics
        assert "sampling_method" in metrics
        print(f"✓ Basic training step test passed - Loss: {average_weighted_loss.item():.4f}")

    def test_step_with_different_batch_sizes(self, simple_adapter, mock_model):
        """Test training step with different batch sizes."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            log_interval=1000,
            summary_log_interval=1000,
        )
        device = torch.device("cpu")
        dtype = torch.bfloat16

        for batch_size in [1, 2, 4, 8]:
            batch = {
                "video_latents": torch.randn(batch_size, 16, 4, 8, 8),
                "text_embeddings": torch.randn(batch_size, 77, 4096),
            }

            weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
                mock_model, batch, device, dtype, global_step=0
            )

            assert isinstance(weighted_loss, torch.Tensor), f"Loss should be tensor for batch_size={batch_size}"
            assert not torch.isnan(average_weighted_loss), f"Loss should not be NaN for batch_size={batch_size}"

    def test_step_with_4d_video_latents(self, pipeline, mock_model):
        """Test that 4D video latents are handled (unsqueezed to 5D)."""
        device = torch.device("cpu")
        dtype = torch.bfloat16

        batch = {
            "video_latents": torch.randn(16, 4, 8, 8),  # 4D instead of 5D
            "text_embeddings": torch.randn(77, 4096),  # 2D instead of 3D
        }

        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, batch, device, dtype, global_step=0
        )

        assert isinstance(weighted_loss, torch.Tensor)
        assert not torch.isnan(average_weighted_loss)

    def test_step_metrics_collection(self, pipeline, mock_model, sample_batch):
        """Test that all expected metrics are collected."""
        device = torch.device("cpu")
        dtype = torch.bfloat16

        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, sample_batch, device, dtype, global_step=100
        )

        expected_keys = [
            "loss",
            "unweighted_loss",
            "sigma_min",
            "sigma_max",
            "sigma_mean",
            "weight_min",
            "weight_max",
            "timestep_min",
            "timestep_max",
            "noisy_min",
            "noisy_max",
            "sampling_method",
            "task_type",
            "data_type",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_step_sigma_in_valid_range(self, pipeline, mock_model, sample_batch):
        """Test that sigma values are in valid range."""
        device = torch.device("cpu")
        dtype = torch.bfloat16

        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, sample_batch, device, dtype, global_step=0
        )

        assert 0.0 <= metrics["sigma_min"] <= 1.0, "Sigma min should be in [0, 1]"
        assert 0.0 <= metrics["sigma_max"] <= 1.0, "Sigma max should be in [0, 1]"
        assert metrics["sigma_min"] <= metrics["sigma_max"], "Sigma min should be <= sigma max"

    def test_step_timesteps_in_valid_range(self, simple_adapter, mock_model, sample_batch):
        """Test that timesteps are in valid range [0, num_train_timesteps]."""
        num_timesteps = 1000
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            num_train_timesteps=num_timesteps,
            log_interval=1000,
            summary_log_interval=1000,
        )
        device = torch.device("cpu")
        dtype = torch.bfloat16

        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, sample_batch, device, dtype, global_step=0
        )

        assert 0.0 <= metrics["timestep_min"] <= num_timesteps
        assert 0.0 <= metrics["timestep_max"] <= num_timesteps

    def test_step_noisy_latents_are_finite(self, pipeline, mock_model, sample_batch):
        """Test that noisy latents don't contain NaN or Inf."""
        device = torch.device("cpu")
        dtype = torch.bfloat16

        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, sample_batch, device, dtype, global_step=0
        )

        assert torch.isfinite(torch.tensor(metrics["noisy_min"])), "Noisy min should be finite"
        assert torch.isfinite(torch.tensor(metrics["noisy_max"])), "Noisy max should be finite"

    def test_step_with_image_batch(self, pipeline, mock_model, image_batch):
        """Test training step with image data."""
        device = torch.device("cpu")
        dtype = torch.bfloat16

        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, image_batch, device, dtype, global_step=0
        )

        assert isinstance(weighted_loss, torch.Tensor)
        assert not torch.isnan(average_weighted_loss)
        assert metrics["data_type"] == "image"
        assert metrics["task_type"] == "t2v"  # Image always uses t2v

    def test_deterministic_with_seed(self, simple_adapter, mock_model, sample_batch):
        """Test that setting seed produces deterministic sigma/timesteps."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            timestep_sampling="uniform",
            log_interval=1000,
            summary_log_interval=1000,
        )
        device = torch.device("cpu")
        dtype = torch.bfloat16

        # First run
        torch.manual_seed(42)
        _, _, _, metrics1 = pipeline.step(mock_model, sample_batch, device, dtype, global_step=0)

        # Second run with same seed
        torch.manual_seed(42)
        _, _, _, metrics2 = pipeline.step(mock_model, sample_batch, device, dtype, global_step=0)

        # Sigma values should be identical
        assert abs(metrics1["sigma_min"] - metrics2["sigma_min"]) < 1e-6
        assert abs(metrics1["sigma_max"] - metrics2["sigma_max"]) < 1e-6


class TestFactoryFunctions:
    """Test factory functions for creating adapters and pipelines."""

    def test_create_simple_adapter(self):
        """Test creating SimpleAdapter via factory."""
        adapter = create_adapter("simple")
        assert isinstance(adapter, SimpleAdapter)

    def test_create_hunyuan_adapter(self):
        """Test creating HunyuanAdapter via factory."""
        adapter = create_adapter("hunyuan")
        assert isinstance(adapter, HunyuanAdapter)

    def test_create_adapter_invalid_type(self):
        """Test that invalid adapter type raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_adapter("invalid_adapter")

        assert "Unknown adapter type" in str(exc_info.value)

    def test_create_pipeline_simple(self):
        """Test creating pipeline with simple adapter."""
        pipeline = create_pipeline(
            adapter_type="simple",
            flow_shift=5.0,
            timestep_sampling="mode",
        )

        assert isinstance(pipeline, FlowMatchingPipeline)
        assert isinstance(pipeline.model_adapter, SimpleAdapter)
        assert pipeline.flow_shift == 5.0
        assert pipeline.timestep_sampling == "mode"

    def test_create_pipeline_hunyuan(self):
        """Test creating pipeline with hunyuan adapter."""
        pipeline = create_pipeline(
            adapter_type="hunyuan",
            adapter_kwargs={},
            flow_shift=3.0,
        )

        assert isinstance(pipeline, FlowMatchingPipeline)
        assert isinstance(pipeline.model_adapter, HunyuanAdapter)

    def test_create_pipeline_with_adapter_kwargs(self):
        """Test creating pipeline with adapter kwargs."""
        pipeline = create_pipeline(
            adapter_type="simple",
            adapter_kwargs={},
            sigma_min=0.1,
            sigma_max=0.9,
        )

        assert isinstance(pipeline, FlowMatchingPipeline)
        assert pipeline.sigma_min == 0.1
        assert pipeline.sigma_max == 0.9


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_empty_batch_handling(self, simple_adapter):
        """Test handling of minimal batch."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            log_interval=1000,
            summary_log_interval=1000,
        )

        # Batch with single sample
        batch = {
            "video_latents": torch.randn(1, 16, 1, 8, 8),
            "text_embeddings": torch.randn(1, 77, 4096),
        }

        mock_model = MockModel()
        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, batch, torch.device("cpu"), torch.float32, global_step=0
        )

        assert not torch.isnan(average_weighted_loss)

    def test_large_batch_handling(self, simple_adapter):
        """Test handling of larger batch sizes."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            log_interval=1000,
            summary_log_interval=1000,
        )

        batch = {
            "video_latents": torch.randn(16, 16, 4, 8, 8),
            "text_embeddings": torch.randn(16, 77, 4096),
        }

        mock_model = MockModel()
        weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
            mock_model, batch, torch.device("cpu"), torch.float32, global_step=0
        )

        assert not torch.isnan(average_weighted_loss)

    def test_extreme_flow_shift_values(self, simple_adapter):
        """Test with extreme flow shift values."""
        # Note: Very extreme values (50+) can produce losses > 100 which triggers
        # the pipeline's loss explosion safety check. We test a reasonable range
        # of shift values that covers edge cases without triggering false positives.
        for shift in [0.1, 0.5, 5.0, 15.0]:
            pipeline = FlowMatchingPipeline(
                model_adapter=simple_adapter,
                flow_shift=shift,
                log_interval=1000,
                summary_log_interval=1000,
            )

            batch = {
                "video_latents": torch.randn(2, 16, 4, 8, 8),
                "text_embeddings": torch.randn(2, 77, 4096),
            }

            mock_model = MockModel()
            weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
                mock_model, batch, torch.device("cpu"), torch.float32, global_step=0
            )

            assert torch.isfinite(average_weighted_loss), f"Loss should be finite for shift={shift}"

    def test_sigma_clamping_edge_cases(self, simple_adapter):
        """Test sigma clamping at boundary values."""
        # Very narrow range
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            sigma_min=0.5,
            sigma_max=0.5,  # Same value
            log_interval=1000,
            summary_log_interval=1000,
        )

        sigma, _, _ = pipeline.sample_timesteps(10, torch.device("cpu"))
        assert torch.allclose(sigma, torch.full_like(sigma, 0.5), atol=1e-5)


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_multiple_training_steps(self, simple_adapter):
        """Test multiple consecutive training steps."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            log_interval=1000,
            summary_log_interval=1000,
        )

        mock_model = MockModel()

        losses = []
        for step in range(10):
            batch = {
                "video_latents": torch.randn(2, 16, 4, 8, 8),
                "text_embeddings": torch.randn(2, 77, 4096),
            }

            weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
                mock_model, batch, torch.device("cpu"), torch.float32, global_step=step
            )
            losses.append(average_weighted_loss.item())

            assert not torch.isnan(average_weighted_loss), f"Loss became NaN at step {step}"
            assert torch.isfinite(average_weighted_loss), f"Loss became infinite at step {step}"

    def test_pipeline_with_all_sampling_methods(self, simple_adapter):
        """Test pipeline works with all sampling methods."""
        mock_model = MockModel()

        for method in ["uniform", "logit_normal", "mode", "lognorm", "mix"]:
            pipeline = FlowMatchingPipeline(
                model_adapter=simple_adapter,
                timestep_sampling=method,
                mix_uniform_ratio=0.3 if method == "mix" else 0.0,
                log_interval=1000,
                summary_log_interval=1000,
            )

            batch = {
                "video_latents": torch.randn(2, 16, 4, 8, 8),
                "text_embeddings": torch.randn(2, 77, 4096),
            }

            weighted_loss, average_weighted_loss, loss_mask, metrics = pipeline.step(
                mock_model, batch, torch.device("cpu"), torch.float32, global_step=0
            )

            assert not torch.isnan(average_weighted_loss), f"Loss should not be NaN for method={method}"

    def test_pipeline_state_consistency(self, simple_adapter):
        """Test that pipeline maintains consistent state."""
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            flow_shift=3.0,
            sigma_min=0.1,
            sigma_max=0.9,
        )

        # Verify state after initialization
        assert pipeline.flow_shift == 3.0
        assert pipeline.sigma_min == 0.1
        assert pipeline.sigma_max == 0.9

        # Run some steps
        mock_model = MockModel()
        batch = {
            "video_latents": torch.randn(2, 16, 4, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
        }

        for _ in range(5):
            pipeline.step(mock_model, batch, torch.device("cpu"), torch.float32, global_step=0)

        # Verify state is unchanged
        assert pipeline.flow_shift == 3.0
        assert pipeline.sigma_min == 0.1
        assert pipeline.sigma_max == 0.9


class TestPipelineEdgeCases:
    """Test edge cases for uncovered lines in pipeline.py."""

    def test_missing_batch_key_raises(self, simple_adapter):
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            log_interval=1000,
            summary_log_interval=1000,
        )
        mock_model = MockModel()
        batch = {"text_embeddings": torch.randn(2, 77, 4096)}  # no latents key
        with pytest.raises(KeyError, match="video_latents.*image_latents"):
            pipeline.step(mock_model, batch, torch.device("cpu"), torch.float32, global_step=0)

    def test_image_latents_key_works(self, simple_adapter):
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            log_interval=1000,
            summary_log_interval=1000,
        )
        mock_model = MockModel()
        batch = {
            "image_latents": torch.randn(2, 16, 1, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
            "data_type": "image",
        }
        _, avg_loss, _, metrics = pipeline.step(mock_model, batch, torch.device("cpu"), torch.float32, global_step=0)
        assert not torch.isnan(avg_loss)

    def test_loss_explosion_raises(self, simple_adapter):
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            flow_shift=1.0,
            use_loss_weighting=False,
            log_interval=1000,
            summary_log_interval=1000,
        )
        # Model that returns very large values to trigger loss > 100
        big_model = MockModel(output_scale=100.0)
        batch = {
            "video_latents": torch.randn(2, 16, 4, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
        }
        with pytest.raises(ValueError, match="Loss exploded"):
            pipeline.step(big_model, batch, torch.device("cpu"), torch.float32, global_step=0)

    def test_nan_loss_raises(self, simple_adapter):
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            flow_shift=1.0,
            log_interval=1000,
            summary_log_interval=1000,
        )

        class NanModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False):
                return (torch.full_like(hidden_states, float("nan")),)

        nan_model = NanModel()
        batch = {
            "video_latents": torch.randn(2, 16, 4, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
        }
        with pytest.raises(ValueError, match="Loss exploded"):
            pipeline.step(nan_model, batch, torch.device("cpu"), torch.float32, global_step=0)


class TestDebugLogging:
    """Test debug logging paths in pipeline.py."""

    def test_log_detailed_single_element_sigma(self, simple_adapter):
        pipeline = FlowMatchingPipeline(model_adapter=simple_adapter)
        # Single-element sigma triggers the `.item()` path
        pipeline._log_detailed(
            global_step=0,
            sampling_method="uniform",
            batch_size=1,
            sigma=torch.tensor([0.5]),
            timesteps=torch.tensor([500.0]),
            latents=torch.randn(1, 16, 4, 8, 8),
            noise=torch.randn(1, 16, 4, 8, 8),
            noisy_latents=torch.randn(1, 16, 4, 8, 8),
        )

    def test_log_detailed_multi_element_sigma(self, simple_adapter):
        pipeline = FlowMatchingPipeline(model_adapter=simple_adapter)
        pipeline._log_detailed(
            global_step=0,
            sampling_method="logit_normal",
            batch_size=4,
            sigma=torch.tensor([0.1, 0.3, 0.5, 0.9]),
            timesteps=torch.tensor([100.0, 300.0, 500.0, 900.0]),
            latents=torch.randn(4, 16, 4, 8, 8),
            noise=torch.randn(4, 16, 4, 8, 8),
            noisy_latents=torch.randn(4, 16, 4, 8, 8),
        )

    def test_log_detailed_large_noisy_triggers_warning(self, simple_adapter):
        pipeline = FlowMatchingPipeline(model_adapter=simple_adapter)
        latents = torch.randn(1, 16, 4, 8, 8) * 0.1
        noise = torch.randn(1, 16, 4, 8, 8) * 0.1
        # noisy_latents much larger than expected
        noisy_latents = torch.randn(1, 16, 4, 8, 8) * 100.0
        pipeline._log_detailed(
            global_step=0,
            sampling_method="uniform",
            batch_size=1,
            sigma=torch.tensor([0.5]),
            timesteps=torch.tensor([500.0]),
            latents=latents,
            noise=noise,
            noisy_latents=noisy_latents,
        )

    def test_log_loss_detailed(self, simple_adapter):
        pipeline = FlowMatchingPipeline(model_adapter=simple_adapter)
        pipeline._log_loss_detailed(
            global_step=0,
            model_pred=torch.randn(2, 16, 4, 8, 8),
            target=torch.randn(2, 16, 4, 8, 8),
            loss_weight=torch.tensor([1.5, 2.0]).view(2, 1, 1, 1, 1),
            unweighted_loss=torch.tensor(0.5),
            weighted_loss=torch.tensor(0.8),
        )

    def test_debug_env_var_detailed_log(self, simple_adapter, monkeypatch):
        monkeypatch.setenv("DEBUG_TRAINING", "1")
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            log_interval=1,
            summary_log_interval=1,
        )
        mock_model = MockModel()
        batch = {
            "video_latents": torch.randn(2, 16, 4, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
        }
        _, avg_loss, _, metrics = pipeline.step(
            mock_model, batch, torch.device("cpu"), torch.float32, global_step=0
        )
        assert not torch.isnan(avg_loss)

    def test_debug_env_var_summary_log(self, simple_adapter, monkeypatch):
        monkeypatch.setenv("DEBUG_TRAINING", "1")
        pipeline = FlowMatchingPipeline(
            model_adapter=simple_adapter,
            log_interval=1000,  # Not a detailed log step
            summary_log_interval=1,  # But is a summary log step
        )
        mock_model = MockModel()
        batch = {
            "video_latents": torch.randn(2, 16, 4, 8, 8),
            "text_embeddings": torch.randn(2, 77, 4096),
        }
        _, avg_loss, _, metrics = pipeline.step(
            mock_model, batch, torch.device("cpu"), torch.float32, global_step=1
        )
        assert not torch.isnan(avg_loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

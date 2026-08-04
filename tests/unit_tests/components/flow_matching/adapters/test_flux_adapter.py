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

"""Unit tests for FluxAdapter (flux.py): pack/unpack latents, prepare_inputs, forward."""

from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.adapters.flux import FluxAdapter


# =============================================================================
# TestFluxAdapterPackLatents
# =============================================================================


class TestFluxAdapterPackLatents:
    """Tests for FluxAdapter._pack_latents."""

    @pytest.mark.parametrize(
        "b, c, h, w",
        [
            (1, 16, 8, 8),
            (2, 16, 16, 16),
            (4, 4, 32, 64),
        ],
    )
    def test_pack_shape(self, b, c, h, w):
        adapter = FluxAdapter()
        latents = torch.randn(b, c, h, w)
        packed = adapter._pack_latents(latents)
        expected_patches = (h // 2) * (w // 2)
        expected_channels = c * 4
        assert packed.shape == (b, expected_patches, expected_channels)

    def test_pack_values_deterministic(self):
        adapter = FluxAdapter()
        torch.manual_seed(42)
        latents = torch.randn(1, 4, 4, 4)
        packed = adapter._pack_latents(latents)
        # Verify finite
        assert torch.isfinite(packed).all()
        assert packed.shape == (1, 4, 16)  # (4//2)*(4//2)=4, 4*4=16


# =============================================================================
# TestFluxAdapterUnpackLatents
# =============================================================================


class TestFluxAdapterUnpackLatents:
    """Tests for FluxAdapter._unpack_latents (static method)."""

    @pytest.mark.parametrize(
        "b, c, h, w",
        [
            (1, 16, 8, 8),
            (2, 16, 16, 16),
            (4, 4, 32, 64),
        ],
    )
    def test_unpack_shape(self, b, c, h, w):
        packed_patches = (h // 2) * (w // 2)
        packed_channels = c * 4
        packed = torch.randn(b, packed_patches, packed_channels)
        # unpack expects pixel-space height/width
        pixel_h = h * 8
        pixel_w = w * 8
        unpacked = FluxAdapter._unpack_latents(packed, pixel_h, pixel_w)
        assert unpacked.shape == (b, c, h, w)

    def test_pack_unpack_roundtrip(self):
        adapter = FluxAdapter()
        latents = torch.randn(2, 16, 8, 8)
        packed = adapter._pack_latents(latents)
        unpacked = FluxAdapter._unpack_latents(packed, 8 * 8, 8 * 8)
        assert torch.allclose(unpacked, latents, atol=1e-6)


# =============================================================================
# TestPrepareLatentImageIds
# =============================================================================


class TestPrepareLatentImageIds:
    """Tests for FluxAdapter._prepare_latent_image_ids."""

    def test_shape(self):
        adapter = FluxAdapter()
        ids = adapter._prepare_latent_image_ids(2, 8, 8, torch.device("cpu"), torch.float32)
        expected_patches = (8 // 2) * (8 // 2)
        assert ids.shape == (expected_patches, 3)

    def test_positional_values(self):
        adapter = FluxAdapter()
        ids = adapter._prepare_latent_image_ids(1, 4, 6, torch.device("cpu"), torch.float32)
        # Height: 4//2=2, Width: 6//2=3 -> 6 patches
        assert ids.shape == (6, 3)
        # First column should be zeros (batch idx placeholder)
        assert (ids[:, 0] == 0).all()
        # y positions: 0,0,0,1,1,1
        assert ids[0, 1] == 0.0
        assert ids[3, 1] == 1.0
        # x positions: 0,1,2,0,1,2
        assert ids[0, 2] == 0.0
        assert ids[1, 2] == 1.0
        assert ids[2, 2] == 2.0

    def test_dtype_and_device(self):
        adapter = FluxAdapter()
        ids = adapter._prepare_latent_image_ids(1, 8, 8, torch.device("cpu"), torch.float64)
        assert ids.dtype == torch.float64
        assert ids.device == torch.device("cpu")


# =============================================================================
# TestPrepareInputs
# =============================================================================


class TestPrepareInputs:
    """Tests for FluxAdapter.prepare_inputs."""

    def _make_context(self, noisy_latents, batch, **kwargs):
        defaults = dict(
            noisy_latents=noisy_latents,
            latents=torch.randn_like(noisy_latents),
            timesteps=torch.tensor([500.0, 500.0]),
            sigma=torch.tensor([0.5, 0.5]),
            task_type="t2v",
            data_type="image",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )
        defaults.update(kwargs)
        return FlowMatchingContext(**defaults)

    def test_4d_latent_accepted(self):
        adapter = FluxAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 77, 4096)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert "hidden_states" in inputs
        assert "encoder_hidden_states" in inputs
        assert "pooled_projections" in inputs
        assert "timestep" in inputs
        assert "img_ids" in inputs
        assert "txt_ids" in inputs

    def test_5d_latent_raises(self):
        adapter = FluxAdapter()
        noisy = torch.randn(2, 16, 4, 8, 8)  # 5D
        batch = {"text_embeddings": torch.randn(2, 77, 4096)}
        ctx = self._make_context(noisy, batch)
        with pytest.raises(ValueError, match="FluxAdapter expects 4D"):
            adapter.prepare_inputs(ctx)

    def test_pooled_prompt_embeds_key(self):
        adapter = FluxAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {
            "text_embeddings": torch.randn(2, 77, 4096),
            "pooled_prompt_embeds": torch.randn(2, 768),
        }
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["pooled_projections"].shape == (2, 768)

    def test_clip_pooled_key(self):
        adapter = FluxAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {
            "text_embeddings": torch.randn(2, 77, 4096),
            "clip_pooled": torch.randn(2, 768),
        }
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["pooled_projections"].shape == (2, 768)

    def test_zeros_fallback(self):
        adapter = FluxAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 77, 4096)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert (inputs["pooled_projections"] == 0).all()
        assert inputs["pooled_projections"].shape == (2, 768)

    def test_2d_text_unsqueeze(self):
        adapter = FluxAdapter()
        noisy = torch.randn(1, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(77, 4096)}  # 2D
        ctx = self._make_context(
            noisy, batch,
            timesteps=torch.tensor([500.0]),
            sigma=torch.tensor([0.5]),
        )
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["encoder_hidden_states"].ndim == 3

    def test_1d_pooled_unsqueeze(self):
        adapter = FluxAdapter()
        noisy = torch.randn(1, 16, 8, 8)
        batch = {
            "text_embeddings": torch.randn(1, 77, 4096),
            "pooled_prompt_embeds": torch.randn(768),  # 1D
        }
        ctx = self._make_context(
            noisy, batch,
            timesteps=torch.tensor([500.0]),
            sigma=torch.tensor([0.5]),
        )
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["pooled_projections"].ndim == 2

    def test_timestep_normalization(self):
        adapter = FluxAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 77, 4096)}
        timesteps = torch.tensor([500.0, 1000.0])
        ctx = self._make_context(noisy, batch, timesteps=timesteps)
        inputs = adapter.prepare_inputs(ctx)
        expected = timesteps / 1000.0
        assert torch.allclose(inputs["timestep"], expected)

    def test_cfg_dropout_zeroing(self):
        adapter = FluxAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {
            "text_embeddings": torch.randn(2, 77, 4096) + 1.0,  # Non-zero
            "pooled_prompt_embeds": torch.randn(2, 768) + 1.0,
        }
        ctx = self._make_context(noisy, batch, cfg_dropout_prob=1.0)
        inputs = adapter.prepare_inputs(ctx)
        assert (inputs["encoder_hidden_states"] == 0).all()
        assert (inputs["pooled_projections"] == 0).all()

    def test_guidance_embedding(self):
        adapter = FluxAdapter(guidance_scale=7.5)
        noisy = torch.randn(2, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 77, 4096)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["guidance"].shape == (2,)
        assert torch.allclose(inputs["guidance"], torch.tensor([7.5, 7.5]))


# =============================================================================
# TestForward
# =============================================================================


class TestForward:
    """Tests for FluxAdapter.forward."""

    def test_unpacked_output_shape(self):
        adapter = FluxAdapter()
        # Prepare mock inputs matching what prepare_inputs would return
        b, c, h, w = 2, 16, 8, 8
        packed_patches = (h // 2) * (w // 2)
        packed_channels = c * 4

        mock_model = MagicMock()
        # Model returns packed prediction
        mock_model.return_value = (torch.randn(b, packed_patches, packed_channels),)

        inputs = {
            "hidden_states": torch.randn(b, packed_patches, packed_channels),
            "encoder_hidden_states": torch.randn(b, 77, 4096),
            "pooled_projections": torch.randn(b, 768),
            "timestep": torch.tensor([0.5, 0.5]),
            "img_ids": torch.zeros(packed_patches, 3),
            "txt_ids": torch.zeros(b, 77, 3),
            "guidance": torch.tensor([3.5, 3.5]),
            "_original_shape": (b, c, h, w),
        }

        pred = adapter.forward(mock_model, inputs)
        assert pred.shape == (b, c, h, w)

    def test_tuple_output_handling(self):
        adapter = FluxAdapter()
        b, c, h, w = 1, 4, 4, 4
        packed_patches = (h // 2) * (w // 2)
        packed_channels = c * 4

        mock_model = MagicMock()
        # Model returns tuple with extra outputs
        mock_model.return_value = (
            torch.randn(b, packed_patches, packed_channels),
            "extra_output",
        )

        inputs = {
            "hidden_states": torch.randn(b, packed_patches, packed_channels),
            "encoder_hidden_states": torch.randn(b, 10, 4096),
            "pooled_projections": torch.randn(b, 768),
            "timestep": torch.tensor([0.5]),
            "img_ids": torch.zeros(packed_patches, 3),
            "txt_ids": torch.zeros(b, 10, 3),
            "guidance": torch.tensor([3.5]),
            "_original_shape": (b, c, h, w),
        }

        pred = adapter.forward(mock_model, inputs)
        assert pred.shape == (b, c, h, w)

    def test_forward_calls_model_with_correct_kwargs(self):
        adapter = FluxAdapter()
        b, c, h, w = 1, 4, 4, 4
        packed_patches = (h // 2) * (w // 2)
        packed_channels = c * 4

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, packed_patches, packed_channels),)

        inputs = {
            "hidden_states": torch.randn(b, packed_patches, packed_channels),
            "encoder_hidden_states": torch.randn(b, 10, 4096),
            "pooled_projections": torch.randn(b, 768),
            "timestep": torch.tensor([0.5]),
            "img_ids": torch.zeros(packed_patches, 3),
            "txt_ids": torch.zeros(b, 10, 3),
            "guidance": torch.tensor([3.5]),
            "_original_shape": (b, c, h, w),
        }

        adapter.forward(mock_model, inputs)
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "hidden_states" in call_kwargs
        assert "encoder_hidden_states" in call_kwargs
        assert "return_dict" in call_kwargs
        assert call_kwargs["return_dict"] is False

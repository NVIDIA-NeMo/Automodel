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

"""Unit tests for QwenImageAdapter: pack/unpack latents, prepare_inputs, forward."""

from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.adapters.qwen_image import (
    QwenImageAdapter,
)

# =============================================================================
# TestQwenImageAdapterPackLatents
# =============================================================================


class TestQwenImageAdapterPackLatents:
    """Tests for QwenImageAdapter._pack_latents."""

    @pytest.mark.parametrize(
        "b, c, h, w",
        [
            (1, 16, 8, 8),
            (2, 16, 16, 16),
            (4, 4, 32, 64),
        ],
    )
    def test_pack_shape(self, b, c, h, w):
        adapter = QwenImageAdapter()
        latents = torch.randn(b, c, h, w)
        packed = adapter._pack_latents(latents)
        expected_patches = (h // 2) * (w // 2)
        expected_channels = c * 4
        assert packed.shape == (b, expected_patches, expected_channels)

    def test_pack_values_deterministic(self):
        adapter = QwenImageAdapter()
        torch.manual_seed(42)
        latents = torch.randn(1, 4, 4, 4)
        packed = adapter._pack_latents(latents)
        assert torch.isfinite(packed).all()
        assert packed.shape == (1, 4, 16)  # (4//2)*(4//2)=4, 4*4=16


# =============================================================================
# TestQwenImageAdapterUnpackLatents
# =============================================================================


class TestQwenImageAdapterUnpackLatents:
    """Tests for QwenImageAdapter._unpack_latents (static method)."""

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
        pixel_h = h * 8
        pixel_w = w * 8
        unpacked = QwenImageAdapter._unpack_latents(packed, pixel_h, pixel_w)
        assert unpacked.shape == (b, c, h, w)

    def test_pack_unpack_roundtrip(self):
        adapter = QwenImageAdapter()
        latents = torch.randn(2, 16, 8, 8)
        packed = adapter._pack_latents(latents)
        unpacked = QwenImageAdapter._unpack_latents(packed, 8 * 8, 8 * 8)
        assert torch.allclose(unpacked, latents, atol=1e-6)


# =============================================================================
# TestPrepareInputs
# =============================================================================


class TestPrepareInputs:
    """Tests for QwenImageAdapter.prepare_inputs."""

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
        adapter = QwenImageAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 77, 2048)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert "hidden_states" in inputs
        assert "encoder_hidden_states" in inputs
        assert "timestep" in inputs
        assert "img_shapes" in inputs
        assert "guidance" in inputs
        assert "encoder_hidden_states_mask" in inputs

    def test_5d_latent_raises(self):
        adapter = QwenImageAdapter()
        noisy = torch.randn(2, 16, 4, 8, 8)  # 5D
        batch = {"text_embeddings": torch.randn(2, 77, 2048)}
        ctx = self._make_context(noisy, batch)
        with pytest.raises(ValueError, match="QwenImageAdapter expects 4D"):
            adapter.prepare_inputs(ctx)

    def test_packed_hidden_states_shape(self):
        adapter = QwenImageAdapter()
        b, c, h, w = 2, 16, 8, 8
        noisy = torch.randn(b, c, h, w)
        batch = {"text_embeddings": torch.randn(b, 77, 2048)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        expected_patches = (h // 2) * (w // 2)
        expected_channels = c * 4
        assert inputs["hidden_states"].shape == (b, expected_patches, expected_channels)

    def test_2d_text_unsqueeze(self):
        adapter = QwenImageAdapter()
        noisy = torch.randn(1, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(77, 2048)}  # 2D
        ctx = self._make_context(
            noisy,
            batch,
            timesteps=torch.tensor([500.0]),
            sigma=torch.tensor([0.5]),
        )
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["encoder_hidden_states"].ndim == 3

    def test_timestep_normalization(self):
        adapter = QwenImageAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 77, 2048)}
        timesteps = torch.tensor([500.0, 1000.0])
        ctx = self._make_context(noisy, batch, timesteps=timesteps)
        inputs = adapter.prepare_inputs(ctx)
        expected = timesteps / 1000.0
        assert torch.allclose(inputs["timestep"], expected)

    def test_cfg_dropout_zeroing(self):
        adapter = QwenImageAdapter()
        noisy = torch.randn(2, 16, 8, 8)
        batch = {
            "text_embeddings": torch.randn(2, 77, 2048) + 1.0,  # Non-zero
        }
        ctx = self._make_context(noisy, batch, cfg_dropout_prob=1.0)
        inputs = adapter.prepare_inputs(ctx)
        assert (inputs["encoder_hidden_states"] == 0).all()

    def test_img_shapes_value(self):
        adapter = QwenImageAdapter()
        b, c, h, w = 2, 16, 8, 8
        noisy = torch.randn(b, c, h, w)
        batch = {"text_embeddings": torch.randn(b, 77, 2048)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["img_shapes"] == [[(1, h // 2, w // 2)]] * b

    def test_encoder_hidden_states_mask_is_none(self):
        adapter = QwenImageAdapter()
        b, c, h, w = 2, 16, 8, 8
        noisy = torch.randn(b, c, h, w)
        batch = {"text_embeddings": torch.randn(b, 77, 2048)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["encoder_hidden_states_mask"] is None

    def test_guidance_none_when_disabled(self):
        adapter = QwenImageAdapter(use_guidance_embeds=False)
        noisy = torch.randn(2, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 77, 2048)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["guidance"] is None

    def test_guidance_embedding_when_enabled(self):
        adapter = QwenImageAdapter(guidance_scale=7.5, use_guidance_embeds=True)
        noisy = torch.randn(2, 16, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 77, 2048)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["guidance"].shape == (2,)
        assert torch.allclose(inputs["guidance"], torch.tensor([7.5, 7.5]))


# =============================================================================
# TestForward
# =============================================================================


class TestForward:
    """Tests for QwenImageAdapter.forward."""

    def test_unpacked_output_shape(self):
        adapter = QwenImageAdapter()
        b, c, h, w = 2, 16, 8, 8
        packed_patches = (h // 2) * (w // 2)
        packed_channels = c * 4

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, packed_patches, packed_channels),)

        inputs = {
            "hidden_states": torch.randn(b, packed_patches, packed_channels),
            "encoder_hidden_states": torch.randn(b, 77, 2048),
            "encoder_hidden_states_mask": None,
            "timestep": torch.tensor([0.5, 0.5]),
            "img_shapes": [[(1, h // 2, w // 2)]] * b,
            "guidance": None,
            "_original_shape": (b, c, h, w),
        }

        pred = adapter.forward(mock_model, inputs)
        assert pred.shape == (b, c, h, w)

    def test_tuple_output_handling(self):
        adapter = QwenImageAdapter()
        b, c, h, w = 1, 4, 4, 4
        packed_patches = (h // 2) * (w // 2)
        packed_channels = c * 4

        mock_model = MagicMock()
        mock_model.return_value = (
            torch.randn(b, packed_patches, packed_channels),
            "extra_output",
        )

        inputs = {
            "hidden_states": torch.randn(b, packed_patches, packed_channels),
            "encoder_hidden_states": torch.randn(b, 10, 2048),
            "encoder_hidden_states_mask": None,
            "timestep": torch.tensor([0.5]),
            "img_shapes": [[(1, h // 2, w // 2)]] * b,
            "guidance": None,
            "_original_shape": (b, c, h, w),
        }

        pred = adapter.forward(mock_model, inputs)
        assert pred.shape == (b, c, h, w)

    def test_forward_calls_model_with_correct_kwargs(self):
        adapter = QwenImageAdapter()
        b, c, h, w = 1, 4, 4, 4
        packed_patches = (h // 2) * (w // 2)
        packed_channels = c * 4

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, packed_patches, packed_channels),)

        inputs = {
            "hidden_states": torch.randn(b, packed_patches, packed_channels),
            "encoder_hidden_states": torch.randn(b, 10, 2048),
            "encoder_hidden_states_mask": None,
            "timestep": torch.tensor([0.5]),
            "img_shapes": [[(1, h // 2, w // 2)]] * b,
            "guidance": None,
            "_original_shape": (b, c, h, w),
        }

        adapter.forward(mock_model, inputs)
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "hidden_states" in call_kwargs
        assert "encoder_hidden_states" in call_kwargs
        assert "encoder_hidden_states_mask" in call_kwargs
        assert "timestep" in call_kwargs
        assert "img_shapes" in call_kwargs
        assert "guidance" in call_kwargs
        assert "return_dict" in call_kwargs
        assert call_kwargs["return_dict"] is False


# =============================================================================
# TestCreateAdapter
# =============================================================================


class TestCreateAdapter:
    """Tests for create_adapter factory function with qwen_image type."""

    def test_create_qwen_image_adapter(self):
        from nemo_automodel.components.flow_matching.pipeline import create_adapter

        adapter = create_adapter("qwen_image")
        assert isinstance(adapter, QwenImageAdapter)

    def test_create_qwen_image_adapter_with_kwargs(self):
        from nemo_automodel.components.flow_matching.pipeline import create_adapter

        adapter = create_adapter("qwen_image", guidance_scale=5.0)
        assert isinstance(adapter, QwenImageAdapter)
        assert adapter.guidance_scale == 5.0

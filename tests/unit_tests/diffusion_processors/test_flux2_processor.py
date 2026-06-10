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

from unittest.mock import MagicMock

import pytest
import torch

from tools.diffusion.processors.flux2 import Flux2Processor


@pytest.fixture
def processor():
    return Flux2Processor()


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
class TestFlux2Properties:
    def test_model_type(self, processor):
        assert processor.model_type == "flux2"

    def test_default_model_name(self, processor):
        assert processor.default_model_name == "black-forest-labs/FLUX.2-dev"


# ---------------------------------------------------------------------------
# Encode image (mocked VAE + BN stats)
# ---------------------------------------------------------------------------
class TestEncodeImage:
    def _make_mock_models(self, h=8, w=8):
        """Return mocked models dict with vae, bn_mean, bn_std."""
        vae = MagicMock()
        # VAE encodes [1, 3, H, W] → [1, 32, H/8, W/8], mode() called
        raw_latent = torch.randn(1, 32, h, w)
        mock_dist = MagicMock()
        mock_dist.latent_dist.mode.return_value = raw_latent
        vae.encode.return_value = mock_dist

        # BN statistics: shape [1, 128, 1, 1]
        bn_mean = torch.zeros(1, 128, 1, 1)
        bn_std = torch.ones(1, 128, 1, 1)

        return {"vae": vae}, raw_latent, bn_mean, bn_std

    def _make_models_with_bn(self, h=8, w=8, bn_mean_val=0.0, bn_std_val=1.0):
        models, raw_latent, _, _ = self._make_mock_models(h, w)
        bn_mean = torch.full((1, 128, 1, 1), bn_mean_val)
        bn_std = torch.full((1, 128, 1, 1), bn_std_val)
        models["bn_mean"] = bn_mean
        models["bn_std"] = bn_std
        return models, raw_latent, bn_mean, bn_std

    def test_encode_image_shape(self, processor):
        # encode_image with Flux2Pipeline._patchify_latents is mocked via module patch
        # We test the pipeline directly without diffusers by verifying shape contract
        vae = MagicMock()
        # Simulate patchified output [1, 128, H/16, W/16] returned from _patchify_latents
        # We'll mock at a higher level: vae encode returns raw_latent, patchify returns patchified
        import sys
        from unittest.mock import patch

        h_patch, w_patch = 4, 4
        raw_latent = torch.randn(1, 32, h_patch * 2, w_patch * 2).float()
        patchified = torch.randn(1, 128, h_patch, w_patch).float()

        mock_dist = MagicMock()
        mock_dist.latent_dist.mode.return_value = raw_latent
        vae.encode.return_value = mock_dist

        bn_mean = torch.zeros(1, 128, 1, 1)
        bn_std = torch.ones(1, 128, 1, 1)
        models = {"vae": vae, "bn_mean": bn_mean, "bn_std": bn_std}

        image = torch.randn(1, 3, 64, 64)

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls._patchify_latents.return_value = patchified

        with patch.dict(sys.modules, {"diffusers": MagicMock(Flux2Pipeline=mock_pipeline_cls)}):
            result = processor.encode_image(image, models, device="cpu")

        assert result.ndim == 3
        assert result.shape == (128, h_patch, w_patch)

    def test_encode_image_dtype(self, processor):
        import sys
        from unittest.mock import patch

        h_patch, w_patch = 4, 4
        raw_latent = torch.randn(1, 32, h_patch * 2, w_patch * 2).float()
        patchified = torch.randn(1, 128, h_patch, w_patch).float()

        mock_dist = MagicMock()
        mock_dist.latent_dist.mode.return_value = raw_latent
        vae = MagicMock()
        vae.encode.return_value = mock_dist

        bn_mean = torch.zeros(1, 128, 1, 1)
        bn_std = torch.ones(1, 128, 1, 1)
        models = {"vae": vae, "bn_mean": bn_mean, "bn_std": bn_std}

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls._patchify_latents.return_value = patchified

        with patch.dict(sys.modules, {"diffusers": MagicMock(Flux2Pipeline=mock_pipeline_cls)}):
            result = processor.encode_image(torch.randn(1, 3, 64, 64), models, device="cpu")

        assert result.dtype == torch.float16

    def test_encode_image_squeezed(self, processor):
        import sys
        from unittest.mock import patch

        h_patch, w_patch = 4, 4
        raw_latent = torch.randn(1, 32, h_patch * 2, w_patch * 2).float()
        patchified = torch.randn(1, 128, h_patch, w_patch).float()

        mock_dist = MagicMock()
        mock_dist.latent_dist.mode.return_value = raw_latent
        vae = MagicMock()
        vae.encode.return_value = mock_dist

        bn_mean = torch.zeros(1, 128, 1, 1)
        bn_std = torch.ones(1, 128, 1, 1)
        models = {"vae": vae, "bn_mean": bn_mean, "bn_std": bn_std}

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls._patchify_latents.return_value = patchified

        with patch.dict(sys.modules, {"diffusers": MagicMock(Flux2Pipeline=mock_pipeline_cls)}):
            result = processor.encode_image(torch.randn(1, 3, 64, 64), models, device="cpu")

        # Batch dimension must be squeezed
        assert result.ndim == 3

    def test_encode_image_applies_bn_normalization(self, processor):
        """Verify the result is (patchified - bn_mean) / bn_std cast to float16."""
        import sys
        from unittest.mock import patch

        h_patch, w_patch = 4, 4
        raw_latent = torch.randn(1, 32, h_patch * 2, w_patch * 2).float()
        patchified = torch.ones(1, 128, h_patch, w_patch).float() * 2.0

        mock_dist = MagicMock()
        mock_dist.latent_dist.mode.return_value = raw_latent
        vae = MagicMock()
        vae.encode.return_value = mock_dist

        bn_mean = torch.ones(1, 128, 1, 1) * 0.5
        bn_std = torch.ones(1, 128, 1, 1) * 0.5
        models = {"vae": vae, "bn_mean": bn_mean, "bn_std": bn_std}

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls._patchify_latents.return_value = patchified

        with patch.dict(sys.modules, {"diffusers": MagicMock(Flux2Pipeline=mock_pipeline_cls)}):
            result = processor.encode_image(torch.randn(1, 3, 64, 64), models, device="cpu")

        # Expected: (2.0 - 0.5) / 0.5 = 3.0
        expected = torch.full((128, h_patch, w_patch), 3.0, dtype=torch.float16)
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Encode text (mocked pipeline.encode_prompt)
# ---------------------------------------------------------------------------
class TestEncodeText:
    def _make_mock_models(self, seq_len=64):
        pipeline = MagicMock()
        prompt_embeds = torch.randn(1, seq_len, 15360)
        text_ids = torch.zeros(1, seq_len, 4, dtype=torch.long)
        pipeline.encode_prompt.return_value = (prompt_embeds, text_ids)
        return {"pipeline": pipeline}, prompt_embeds

    def test_encode_text_returns_expected_keys(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("a serene landscape", models, device="cpu")
        assert set(result.keys()) == {"prompt_embeds"}

    def test_encode_text_shape(self, processor):
        seq_len = 64
        models, prompt_embeds = self._make_mock_models(seq_len=seq_len)
        result = processor.encode_text("test", models, device="cpu")
        # Batch dim squeezed: [1, seq_len, 15360] -> [seq_len, 15360]
        assert result["prompt_embeds"].shape == (seq_len, 15360)

    def test_encode_text_dtype(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")
        assert result["prompt_embeds"].dtype == torch.float16

    def test_encode_text_on_cpu(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")
        assert result["prompt_embeds"].device == torch.device("cpu")

    def test_encode_text_passes_correct_args(self, processor):
        models, _ = self._make_mock_models()
        processor.encode_text("my prompt", models, device="cpu")
        models["pipeline"].encode_prompt.assert_called_once_with(
            prompt="my prompt",
            max_sequence_length=512,
            text_encoder_out_layers=(10, 20, 30),
        )

    def test_encode_text_no_clip_keys(self, processor):
        """Flux2 has no CLIP encoder — result must not contain clip-related keys."""
        models, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")
        for unexpected_key in ("clip_hidden", "pooled_prompt_embeds", "clip_tokens", "t5_tokens"):
            assert unexpected_key not in result


# ---------------------------------------------------------------------------
# Verify latent
# ---------------------------------------------------------------------------
class TestVerifyLatent:
    def test_valid_latent_passes(self, processor):
        latent = torch.randn(128, 8, 8)
        assert processor.verify_latent(latent, models={}, device="cpu") is True

    def test_wrong_ndim_fails(self, processor):
        latent = torch.randn(1, 128, 8, 8)  # 4D instead of 3D
        assert processor.verify_latent(latent, models={}, device="cpu") is False

    def test_wrong_channels_fails(self, processor):
        latent = torch.randn(16, 8, 8)  # 16 channels instead of 128
        assert processor.verify_latent(latent, models={}, device="cpu") is False

    def test_nan_latent_fails(self, processor):
        latent = torch.randn(128, 8, 8)
        latent[0, 0, 0] = float("nan")
        assert processor.verify_latent(latent, models={}, device="cpu") is False

    def test_inf_latent_fails(self, processor):
        latent = torch.randn(128, 8, 8)
        latent[0, 0, 0] = float("inf")
        assert processor.verify_latent(latent, models={}, device="cpu") is False

    def test_verify_does_not_require_models(self, processor):
        # verify_latent must work without models dict (no VAE decode in Flux2)
        latent = torch.randn(128, 8, 8)
        assert processor.verify_latent(latent, models={}, device="cpu") is True


# ---------------------------------------------------------------------------
# Cache data
# ---------------------------------------------------------------------------
class TestGetCacheData:
    def test_cache_data_structure(self, processor):
        latent = torch.randn(128, 8, 8)
        text_encodings = {
            "prompt_embeds": torch.randn(64, 15360),
        }
        metadata = {
            "original_resolution": (512, 512),
            "bucket_resolution": (512, 512),
            "crop_offset": (0, 0),
            "prompt": "a beautiful mountain",
            "image_path": "/path/to/img.png",
            "bucket_id": "512x512",
            "aspect_ratio": 1.0,
        }

        result = processor.get_cache_data(latent, text_encodings, metadata)

        assert result["latent"] is latent
        assert result["prompt_embeds"] is text_encodings["prompt_embeds"]
        assert result["original_resolution"] == (512, 512)
        assert result["bucket_resolution"] == (512, 512)
        assert result["crop_offset"] == (0, 0)
        assert result["prompt"] == "a beautiful mountain"
        assert result["image_path"] == "/path/to/img.png"
        assert result["bucket_id"] == "512x512"
        assert result["aspect_ratio"] == 1.0
        assert result["model_type"] == "flux2"

    def test_no_clip_keys_in_cache(self, processor):
        """Flux2 cache must not contain clip or t5 keys from Flux1."""
        latent = torch.randn(128, 8, 8)
        text_encodings = {"prompt_embeds": torch.randn(64, 15360)}
        metadata = {
            "original_resolution": (512, 512),
            "bucket_resolution": (512, 512),
            "crop_offset": (0, 0),
            "prompt": "test",
            "image_path": "/img.png",
            "bucket_id": "512x512",
            "aspect_ratio": 1.0,
        }
        result = processor.get_cache_data(latent, text_encodings, metadata)
        for flux1_key in ("clip_tokens", "clip_hidden", "pooled_prompt_embeds", "t5_tokens"):
            assert flux1_key not in result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestFlux2Registry:
    def test_registered_name(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert ProcessorRegistry.is_registered("flux2")

    def test_get_returns_correct_type(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        proc = ProcessorRegistry.get("flux2")
        assert isinstance(proc, Flux2Processor)

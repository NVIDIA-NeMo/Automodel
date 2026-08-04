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

from tools.diffusion.processors.qwen_image import QwenImageProcessor


@pytest.fixture
def processor():
    return QwenImageProcessor()


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
class TestQwenImageProperties:
    def test_model_type(self, processor):
        assert processor.model_type == "qwen_image"

    def test_default_model_name(self, processor):
        assert processor.default_model_name == "Qwen/Qwen-Image"


# ---------------------------------------------------------------------------
# Encode image (mocked VAE)
# ---------------------------------------------------------------------------
class TestEncodeImage:
    def _make_mock_models(self, latents_mean=None, latents_std=None):
        if latents_mean is None:
            latents_mean = [0.0] * 16
        if latents_std is None:
            latents_std = [1.0] * 16

        vae = MagicMock()
        vae.config.latents_mean = latents_mean
        vae.config.latents_std = latents_std

        # Qwen-Image VAE returns 5D: (B, C, T, H, W)
        latent = torch.randn(1, 16, 1, 8, 8)
        mock_dist = MagicMock()
        mock_dist.latent_dist.sample.return_value = latent
        vae.encode.return_value = mock_dist

        return {"vae": vae}, latent

    def test_encode_image_shape(self, processor):
        models, _ = self._make_mock_models()
        image = torch.randn(1, 3, 64, 64)

        result = processor.encode_image(image, models, device="cpu")

        # 5D (1, 16, 1, 8, 8) -> squeeze frame dim -> squeeze batch dim -> (16, 8, 8)
        assert result.ndim == 3
        assert result.shape == (16, 8, 8)

    def test_encode_image_dtype(self, processor):
        models, _ = self._make_mock_models()
        image = torch.randn(1, 3, 64, 64)

        result = processor.encode_image(image, models, device="cpu")
        assert result.dtype == torch.float16

    def test_encode_image_on_cpu(self, processor):
        models, _ = self._make_mock_models()
        image = torch.randn(1, 3, 64, 64)

        result = processor.encode_image(image, models, device="cpu")
        assert result.device == torch.device("cpu")

    def test_encode_image_applies_mean_std_normalization(self, processor):
        mean_vals = [0.5] * 16
        std_vals = [2.0] * 16
        models, raw_latent = self._make_mock_models(latents_mean=mean_vals, latents_std=std_vals)
        image = torch.randn(1, 3, 64, 64)

        result = processor.encode_image(image, models, device="cpu")

        latents_mean = torch.tensor(mean_vals).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(std_vals).view(1, -1, 1, 1, 1)
        expected = ((raw_latent - latents_mean) / latents_std).squeeze(2).squeeze(0).to(torch.float16)
        torch.testing.assert_close(result, expected)

    @pytest.mark.run_only_on("GPU")
    def test_encode_image_gpu(self, processor):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        latent = torch.randn(1, 16, 1, 8, 8, device="cuda")
        mock_dist = MagicMock()
        mock_dist.latent_dist.sample.return_value = latent

        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.encode.return_value = mock_dist

        models = {"vae": vae}
        image = torch.randn(1, 3, 64, 64)

        result = processor.encode_image(image, models, device="cuda")
        assert result.device == torch.device("cpu")
        assert result.dtype == torch.float16


# ---------------------------------------------------------------------------
# Encode text (mocked pipeline)
# ---------------------------------------------------------------------------
class TestEncodeText:
    def _make_mock_models(self):
        prompt_embeds = torch.randn(1, 1024, 2048)
        prompt_embeds_mask = torch.ones(1, 1024, dtype=torch.long)

        pipeline = MagicMock()
        pipeline.encode_prompt.return_value = (prompt_embeds, prompt_embeds_mask)

        return {"pipeline": pipeline}, prompt_embeds

    def test_encode_text_returns_expected_keys(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("a cat sitting on a table", models, device="cpu")

        assert set(result.keys()) == {"prompt_embeds"}

    def test_encode_text_shapes(self, processor):
        models, prompt_embeds = self._make_mock_models()
        result = processor.encode_text("hello world", models, device="cpu")

        assert result["prompt_embeds"].shape == prompt_embeds.shape

    def test_encode_text_prompt_embeds_dtype(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")

        assert result["prompt_embeds"].dtype == torch.bfloat16

    def test_encode_text_all_on_cpu(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")

        for v in result.values():
            assert v.device == torch.device("cpu")

    def test_encode_text_calls_pipeline_encode_prompt(self, processor):
        models, _ = self._make_mock_models()
        processor.encode_text("test prompt", models, device="cpu")

        models["pipeline"].encode_prompt.assert_called_once_with(
            prompt="test prompt",
            device="cpu",
        )


# ---------------------------------------------------------------------------
# Verify latent
# ---------------------------------------------------------------------------
class TestVerifyLatent:
    def _make_mock_models(self):
        # Qwen-Image VAE decode returns 5D: (B, C, T, H, W)
        decoded = torch.randn(1, 3, 1, 64, 64)
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.decode.return_value = decode_output

        return {"vae": vae}

    def test_valid_latent_passes(self, processor):
        models = self._make_mock_models()
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is True

    def test_nan_latent_fails(self, processor):
        decoded = torch.randn(1, 3, 1, 64, 64)
        decoded[0, 0, 0, 0, 0] = float("nan")
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.decode.return_value = decode_output

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is False

    def test_inf_latent_fails(self, processor):
        decoded = torch.randn(1, 3, 1, 64, 64)
        decoded[0, 0, 0, 0, 0] = float("inf")
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.decode.return_value = decode_output

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is False

    def test_wrong_channels_fails(self, processor):
        decoded = torch.randn(1, 4, 1, 64, 64)  # 4 channels instead of 3
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.decode.return_value = decode_output

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is False

    def test_decode_exception_returns_false(self, processor):
        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.decode.side_effect = RuntimeError("decode failed")

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is False

    @pytest.mark.run_only_on("GPU")
    def test_verify_latent_gpu(self, processor):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        decoded = torch.randn(1, 3, 1, 64, 64, device="cuda")
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.decode.return_value = decode_output

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cuda") is True


# ---------------------------------------------------------------------------
# Cache data
# ---------------------------------------------------------------------------
class TestGetCacheData:
    def test_cache_data_structure(self, processor):
        latent = torch.randn(16, 8, 8)
        text_encodings = {
            "prompt_embeds": torch.randn(1, 1024, 2048),
        }
        metadata = {
            "original_resolution": (512, 512),
            "bucket_resolution": (512, 512),
            "crop_offset": (0, 0),
            "prompt": "a scenic mountain view",
            "image_path": "/data/image.png",
            "bucket_id": "512x512",
            "aspect_ratio": 1.0,
        }

        result = processor.get_cache_data(latent, text_encodings, metadata)

        assert result["latent"] is latent
        assert result["prompt_embeds"] is text_encodings["prompt_embeds"]
        assert result["original_resolution"] == (512, 512)
        assert result["bucket_resolution"] == (512, 512)
        assert result["crop_offset"] == (0, 0)
        assert result["prompt"] == "a scenic mountain view"
        assert result["image_path"] == "/data/image.png"
        assert result["bucket_id"] == "512x512"
        assert result["aspect_ratio"] == 1.0
        assert result["model_type"] == "qwen_image"
        assert "text_tokens" not in result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestQwenImageRegistry:
    def test_registered_name(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert ProcessorRegistry.is_registered("qwen_image")

    def test_get_returns_correct_type(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        proc = ProcessorRegistry.get("qwen_image")
        assert isinstance(proc, QwenImageProcessor)

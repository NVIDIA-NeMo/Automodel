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

    def test_max_sequence_length(self, processor):
        assert processor.MAX_SEQUENCE_LENGTH == 256


# ---------------------------------------------------------------------------
# Encode image (mocked VAE)
# ---------------------------------------------------------------------------
class TestEncodeImage:
    def _make_mock_models(self, shift_factor=0.5, scaling_factor=0.7):
        vae = MagicMock()
        vae.config.shift_factor = shift_factor
        vae.config.scaling_factor = scaling_factor

        latent = torch.randn(1, 16, 8, 8)
        mock_dist = MagicMock()
        mock_dist.latent_dist.sample.return_value = latent
        vae.encode.return_value = mock_dist

        return {"vae": vae}, latent

    def test_encode_image_shape(self, processor):
        models, _ = self._make_mock_models()
        image = torch.randn(1, 3, 64, 64)

        result = processor.encode_image(image, models, device="cpu")

        # Batch dim squeezed: (1, 16, 8, 8) -> (16, 8, 8)
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

    def test_encode_image_applies_shift_and_scale(self, processor):
        shift = 0.3
        scale = 1.5
        models, raw_latent = self._make_mock_models(shift_factor=shift, scaling_factor=scale)
        image = torch.randn(1, 3, 64, 64)

        result = processor.encode_image(image, models, device="cpu")

        expected = ((raw_latent - shift) * scale).squeeze(0).to(torch.float16)
        torch.testing.assert_close(result, expected)

    @pytest.mark.run_only_on("GPU")
    def test_encode_image_gpu(self, processor):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        latent = torch.randn(1, 16, 8, 8, device="cuda")
        mock_dist = MagicMock()
        mock_dist.latent_dist.sample.return_value = latent

        vae = MagicMock()
        vae.config.shift_factor = 0.5
        vae.config.scaling_factor = 0.7
        vae.encode.return_value = mock_dist

        models = {"vae": vae}
        image = torch.randn(1, 3, 64, 64)

        result = processor.encode_image(image, models, device="cuda")
        assert result.device == torch.device("cpu")
        assert result.dtype == torch.float16


# ---------------------------------------------------------------------------
# Encode text (mocked tokenizer and Qwen2 encoder)
# ---------------------------------------------------------------------------
class TestEncodeText:
    @staticmethod
    def _make_tokenizer_output(input_ids):
        output = MagicMock()
        output.input_ids = input_ids
        output.__getitem__ = lambda self, key: getattr(self, key)
        return output

    def _make_mock_models(self):
        input_ids = torch.randint(0, 1000, (1, 256))
        tokenizer = MagicMock()
        tokenizer.return_value = self._make_tokenizer_output(input_ids)

        hidden_state = torch.randn(1, 256, 2048)
        encoder_output = MagicMock()
        encoder_output.last_hidden_state = hidden_state

        text_encoder = MagicMock()
        text_encoder.return_value = encoder_output

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}, input_ids, hidden_state

    def test_encode_text_returns_expected_keys(self, processor):
        models, _, _ = self._make_mock_models()
        result = processor.encode_text("a cat sitting on a table", models, device="cpu")

        assert set(result.keys()) == {"text_tokens", "prompt_embeds"}

    def test_encode_text_shapes(self, processor):
        models, input_ids, hidden_state = self._make_mock_models()
        result = processor.encode_text("hello world", models, device="cpu")

        assert result["text_tokens"].shape == (1, 256)
        assert result["prompt_embeds"].shape == hidden_state.shape

    def test_encode_text_prompt_embeds_dtype(self, processor):
        models, _, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")

        assert result["prompt_embeds"].dtype == torch.bfloat16

    def test_encode_text_all_on_cpu(self, processor):
        models, _, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")

        for v in result.values():
            assert v.device == torch.device("cpu")

    def test_encode_text_uses_correct_max_length(self, processor):
        models, _, _ = self._make_mock_models()
        processor.encode_text("test prompt", models, device="cpu")

        models["tokenizer"].assert_called_once_with(
            "test prompt",
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )


# ---------------------------------------------------------------------------
# Verify latent
# ---------------------------------------------------------------------------
class TestVerifyLatent:
    def _make_mock_models(self):
        decoded = torch.randn(1, 3, 64, 64)
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.scaling_factor = 0.7
        vae.decode.return_value = decode_output

        return {"vae": vae}

    def test_valid_latent_passes(self, processor):
        models = self._make_mock_models()
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is True

    def test_nan_latent_fails(self, processor):
        decoded = torch.randn(1, 3, 64, 64)
        decoded[0, 0, 0, 0] = float("nan")
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.scaling_factor = 0.7
        vae.decode.return_value = decode_output

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is False

    def test_inf_latent_fails(self, processor):
        decoded = torch.randn(1, 3, 64, 64)
        decoded[0, 0, 0, 0] = float("inf")
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.scaling_factor = 0.7
        vae.decode.return_value = decode_output

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is False

    def test_wrong_channels_fails(self, processor):
        decoded = torch.randn(1, 4, 64, 64)  # 4 channels instead of 3
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.scaling_factor = 0.7
        vae.decode.return_value = decode_output

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is False

    def test_decode_exception_returns_false(self, processor):
        vae = MagicMock()
        vae.config.scaling_factor = 0.7
        vae.decode.side_effect = RuntimeError("decode failed")

        models = {"vae": vae}
        latent = torch.randn(16, 8, 8)
        assert processor.verify_latent(latent, models, device="cpu") is False

    @pytest.mark.run_only_on("GPU")
    def test_verify_latent_gpu(self, processor):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        decoded = torch.randn(1, 3, 64, 64, device="cuda")
        decode_output = MagicMock()
        decode_output.sample = decoded

        vae = MagicMock()
        vae.config.scaling_factor = 0.7
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
            "text_tokens": torch.randint(0, 1000, (1, 256)),
            "prompt_embeds": torch.randn(1, 256, 2048),
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
        assert result["text_tokens"] is text_encodings["text_tokens"]
        assert result["prompt_embeds"] is text_encodings["prompt_embeds"]
        assert result["original_resolution"] == (512, 512)
        assert result["bucket_resolution"] == (512, 512)
        assert result["crop_offset"] == (0, 0)
        assert result["prompt"] == "a scenic mountain view"
        assert result["image_path"] == "/data/image.png"
        assert result["bucket_id"] == "512x512"
        assert result["aspect_ratio"] == 1.0
        assert result["model_type"] == "qwen_image"


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

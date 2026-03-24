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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tools.diffusion.processors.wan import (
    WanProcessor,
    _basic_clean,
    _prompt_clean,
    _whitespace_clean,
)


@pytest.fixture
def processor():
    return WanProcessor()


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
class TestWanProperties:
    def test_model_type(self, processor):
        assert processor.model_type == "wan"

    def test_default_model_name(self, processor):
        assert processor.default_model_name == "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    def test_supported_modes(self, processor):
        assert processor.supported_modes == ["video", "frames"]

    def test_quantization(self, processor):
        assert processor.quantization == 16

    def test_max_sequence_length(self, processor):
        assert processor.MAX_SEQUENCE_LENGTH == 226

    def test_frame_constraint_default(self, processor):
        # Wan doesn't override frame_constraint, inherits None from base
        assert processor.frame_constraint is None


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------
class TestTextCleaning:
    def test_whitespace_clean_multiple_spaces(self):
        assert _whitespace_clean("hello   world") == "hello world"

    def test_whitespace_clean_tabs_and_newlines(self):
        assert _whitespace_clean("hello\t\n  world") == "hello world"

    def test_whitespace_clean_leading_trailing(self):
        assert _whitespace_clean("  hello  ") == "hello"

    def test_basic_clean_html_entities(self):
        assert _basic_clean("&amp;") == "&"
        assert _basic_clean("&lt;tag&gt;") == "<tag>"

    def test_basic_clean_double_escaped_html(self):
        assert _basic_clean("&amp;amp;") == "&"

    def test_basic_clean_strips_whitespace(self):
        assert _basic_clean("  hello  ") == "hello"

    def test_prompt_clean_combined(self):
        assert _prompt_clean("  hello   &amp;   world  ") == "hello & world"

    def test_prompt_clean_empty_string(self):
        assert _prompt_clean("") == ""

    def test_prompt_clean_plain_text(self):
        assert _prompt_clean("A cat sitting on a mat") == "A cat sitting on a mat"

    def test_basic_clean_handles_missing_ftfy(self):
        """basic_clean should work even when ftfy/diffusers are not importable."""
        with patch("builtins.__import__", side_effect=ImportError):
            # Falls through to html.unescape path
            result = _basic_clean("&amp; test")
        assert "test" in result


# ---------------------------------------------------------------------------
# Encode video (mocked VAE)
# ---------------------------------------------------------------------------
class TestEncodeVideo:
    def _make_mock_models(self, latents_mean=None, latents_std=None):
        if latents_mean is None:
            latents_mean = [0.0] * 16
        if latents_std is None:
            latents_std = [1.0] * 16

        vae = MagicMock()
        vae.config.latents_mean = latents_mean
        vae.config.latents_std = latents_std

        latent_mean = torch.randn(1, 16, 3, 8, 8)
        latent_sample = torch.randn(1, 16, 3, 8, 8)

        mock_dist = MagicMock()
        mock_dist.latent_dist.mean = latent_mean
        mock_dist.latent_dist.sample.return_value = latent_sample
        vae.encode.return_value = mock_dist

        return {
            "vae": vae,
            "dtype": torch.float32,
        }, latent_mean, latent_sample

    def test_encode_video_deterministic(self, processor):
        models, latent_mean, _ = self._make_mock_models()
        video_tensor = torch.randn(1, 3, 9, 64, 64)

        result = processor.encode_video(video_tensor, models, device="cpu", deterministic=True)

        models["vae"].encode.assert_called_once()
        assert result.dtype == torch.float16
        assert result.device == torch.device("cpu")

    def test_encode_video_stochastic(self, processor):
        models, _, latent_sample = self._make_mock_models()
        video_tensor = torch.randn(1, 3, 9, 64, 64)

        result = processor.encode_video(video_tensor, models, device="cpu", deterministic=False)

        models["vae"].encode.assert_called_once()
        assert result.dtype == torch.float16

    def test_encode_video_applies_mean_std_normalization(self, processor):
        mean_vals = [0.5] * 16
        std_vals = [2.0] * 16
        models, latent_mean, _ = self._make_mock_models(
            latents_mean=mean_vals, latents_std=std_vals
        )
        video_tensor = torch.randn(1, 3, 5, 32, 32)

        result = processor.encode_video(video_tensor, models, device="cpu", deterministic=True)

        latents_mean_t = torch.tensor(mean_vals).view(1, -1, 1, 1, 1)
        latents_std_t = torch.tensor(std_vals).view(1, -1, 1, 1, 1)
        expected = (latent_mean - latents_mean_t) / latents_std_t

        torch.testing.assert_close(result.float(), expected.float(), atol=1e-3, rtol=1e-3)

    def test_encode_video_missing_latents_mean_raises(self, processor):
        vae = MagicMock()
        # Remove latents_mean attribute
        del vae.config.latents_mean
        vae.config.latents_std = [1.0] * 16

        latent_dist = MagicMock()
        latent_dist.latent_dist.mean = torch.randn(1, 16, 3, 8, 8)
        vae.encode.return_value = latent_dist

        models = {"vae": vae, "dtype": torch.float32}
        video_tensor = torch.randn(1, 3, 5, 32, 32)

        with pytest.raises(ValueError, match="latents_mean"):
            processor.encode_video(video_tensor, models, device="cpu")

    def test_encode_video_missing_latents_std_raises(self, processor):
        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        # Remove latents_std attribute
        del vae.config.latents_std

        latent_dist = MagicMock()
        latent_dist.latent_dist.mean = torch.randn(1, 16, 3, 8, 8)
        vae.encode.return_value = latent_dist

        models = {"vae": vae, "dtype": torch.float32}
        video_tensor = torch.randn(1, 3, 5, 32, 32)

        with pytest.raises(ValueError, match="latents_std"):
            processor.encode_video(video_tensor, models, device="cpu")

    @pytest.mark.run_only_on("GPU")
    def test_encode_video_gpu(self, processor):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        latent_mean = torch.randn(1, 16, 3, 8, 8, device="cuda")
        mock_dist = MagicMock()
        mock_dist.latent_dist.mean = latent_mean
        mock_dist.latent_dist.sample.return_value = latent_mean

        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 16
        vae.config.latents_std = [1.0] * 16
        vae.encode.return_value = mock_dist

        models = {"vae": vae, "dtype": torch.float16}
        video_tensor = torch.randn(1, 3, 5, 64, 64)

        result = processor.encode_video(video_tensor, models, device="cuda", deterministic=True)

        assert result.device == torch.device("cpu")
        assert result.dtype == torch.float16


# ---------------------------------------------------------------------------
# Encode text (mocked tokenizer and encoder)
# ---------------------------------------------------------------------------
class TestEncodeText:
    @staticmethod
    def _make_tokenizer_output(input_ids, attention_mask):
        """Create a mock tokenizer output supporting both dict and attribute access."""
        output = MagicMock()
        output.input_ids = input_ids
        output.attention_mask = attention_mask
        # Support dict-style iteration: {k: v.to(device) for k, v in inputs.items()}
        output.items.return_value = [
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
        ]
        output.__getitem__ = lambda self, key: getattr(self, key)
        return output

    def _make_mock_models(self, seq_len=10, hidden_dim=1024):
        """Build mock tokenizer and text_encoder for encode_text tests.

        Args:
            seq_len: Actual (non-padding) token length in the mock tokenizer output.
            hidden_dim: Hidden dimension of encoder output.
        """
        max_len = WanProcessor.MAX_SEQUENCE_LENGTH

        input_ids = torch.randint(0, 1000, (1, max_len))
        attention_mask = torch.zeros(1, max_len, dtype=torch.long)
        attention_mask[0, :seq_len] = 1

        tokenizer = MagicMock()
        tokenizer.return_value = self._make_tokenizer_output(input_ids, attention_mask)

        encoder_output = MagicMock()
        encoder_output.last_hidden_state = torch.randn(1, max_len, hidden_dim)

        text_encoder = MagicMock()
        text_encoder.return_value = encoder_output

        models = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
        }
        return models, hidden_dim

    def test_encode_text_returns_expected_keys(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("a test prompt", models, device="cpu")
        assert set(result.keys()) == {"text_embeddings"}

    def test_encode_text_output_shape(self, processor):
        models, hidden_dim = self._make_mock_models(seq_len=15)
        result = processor.encode_text("hello world", models, device="cpu")
        assert result["text_embeddings"].shape == (1, 226, hidden_dim)

    def test_encode_text_trims_and_repads(self, processor):
        """Verify the trim-and-repad logic: positions beyond seq_len should be zero."""
        seq_len = 8
        models, hidden_dim = self._make_mock_models(seq_len=seq_len)
        result = processor.encode_text("short", models, device="cpu")

        embeddings = result["text_embeddings"]
        # Positions beyond seq_len should be all zeros
        assert (embeddings[0, seq_len:, :] == 0).all()
        # Positions within seq_len should NOT all be zero (very unlikely for random data)
        assert not (embeddings[0, :seq_len, :] == 0).all()

    def test_encode_text_on_cpu(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")
        assert result["text_embeddings"].device == torch.device("cpu")

    def test_encode_text_cleans_prompt(self, processor):
        """Prompt cleaning should be applied before tokenization."""
        models, _ = self._make_mock_models()
        processor.encode_text("  hello   &amp;   world  ", models, device="cpu")

        # Verify the tokenizer was called with the cleaned prompt
        call_args = models["tokenizer"].call_args
        assert call_args[0][0] == "hello & world"

    def test_encode_text_tokenizer_args(self, processor):
        models, _ = self._make_mock_models()
        processor.encode_text("test prompt", models, device="cpu")

        models["tokenizer"].assert_called_once_with(
            "test prompt",
            max_length=226,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )


# ---------------------------------------------------------------------------
# Cache data
# ---------------------------------------------------------------------------
class TestGetCacheData:
    def test_cache_data_structure(self, processor):
        latent = torch.randn(1, 16, 3, 8, 8)
        text_encodings = {
            "text_embeddings": torch.randn(1, 226, 1024),
        }
        first_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        metadata = {
            "first_frame": first_frame,
            "original_resolution": (1920, 1080),
            "bucket_resolution": (1280, 720),
            "bucket_id": "1280x720",
            "aspect_ratio": 16 / 9,
            "num_frames": 25,
            "prompt": "test prompt",
            "video_path": "/path/to/video.mp4",
            "deterministic": True,
            "mode": "video",
        }

        result = processor.get_cache_data(latent, text_encodings, metadata)

        assert result["video_latents"] is latent
        assert result["text_embeddings"] is text_encodings["text_embeddings"]
        assert isinstance(result["first_frame"], torch.Tensor)
        torch.testing.assert_close(
            result["first_frame"], torch.from_numpy(first_frame)
        )
        assert result["model_type"] == "wan"
        assert result["model_version"] == "wan2.1"
        assert result["processing_mode"] == "video"
        assert result["deterministic_latents"] is True
        assert result["num_frames"] == 25
        assert result["prompt"] == "test prompt"
        assert result["original_resolution"] == (1920, 1080)
        assert result["bucket_resolution"] == (1280, 720)
        assert result["bucket_id"] == "1280x720"
        assert result["aspect_ratio"] == 16 / 9
        assert result["video_path"] == "/path/to/video.mp4"

    def test_cache_data_none_first_frame(self, processor):
        latent = torch.randn(1, 16, 3, 8, 8)
        text_encodings = {"text_embeddings": torch.randn(1, 226, 1024)}
        metadata = {"first_frame": None}

        result = processor.get_cache_data(latent, text_encodings, metadata)
        assert result["first_frame"] is None

    def test_cache_data_missing_optional_metadata(self, processor):
        latent = torch.randn(1, 16, 3, 8, 8)
        text_encodings = {"text_embeddings": torch.randn(1, 226, 1024)}
        # first_frame key missing entirely
        metadata = {}

        result = processor.get_cache_data(latent, text_encodings, metadata)

        assert result["first_frame"] is None
        assert result["original_resolution"] is None
        assert result["num_frames"] is None
        assert result["deterministic_latents"] is True  # default
        assert result["processing_mode"] == "video"  # default

    def test_cache_data_frames_mode(self, processor):
        latent = torch.randn(1, 16, 3, 8, 8)
        text_encodings = {"text_embeddings": torch.randn(1, 226, 1024)}
        metadata = {"mode": "frames", "first_frame": None}

        result = processor.get_cache_data(latent, text_encodings, metadata)
        assert result["processing_mode"] == "frames"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestWanRegistry:
    def test_registered_names(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert ProcessorRegistry.is_registered("wan")
        assert ProcessorRegistry.is_registered("wan2.1")

    def test_get_returns_correct_type(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        proc = ProcessorRegistry.get("wan")
        assert isinstance(proc, WanProcessor)

    def test_both_aliases_return_same_class(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert ProcessorRegistry.get_class("wan") is ProcessorRegistry.get_class("wan2.1")

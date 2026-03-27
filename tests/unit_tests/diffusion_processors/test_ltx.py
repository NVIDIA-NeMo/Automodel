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

import numpy as np
import pytest
import torch

from tools.diffusion.processors.ltx import LTXProcessor


@pytest.fixture
def processor():
    return LTXProcessor()


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
class TestLTXProperties:
    def test_model_type(self, processor):
        assert processor.model_type == "ltx"

    def test_default_model_name(self, processor):
        assert processor.default_model_name == "Lightricks/LTX-Video"

    def test_supported_modes(self, processor):
        assert processor.supported_modes == ["video", "frames"]

    def test_quantization(self, processor):
        assert processor.quantization == 32

    def test_max_sequence_length(self, processor):
        assert processor.MAX_SEQUENCE_LENGTH == 128

    def test_frame_constraint(self, processor):
        assert processor.frame_constraint == "8n+1"


# ---------------------------------------------------------------------------
# Frame count handling
# ---------------------------------------------------------------------------
class TestFrameHandling:
    def test_closest_valid_frame_count_exact(self, processor):
        # 8*1+1 = 9, 8*2+1 = 17, 8*3+1 = 25
        assert processor.get_closest_valid_frame_count(9) == 9
        assert processor.get_closest_valid_frame_count(17) == 17
        assert processor.get_closest_valid_frame_count(25) == 25

    def test_closest_valid_frame_count_rounds(self, processor):
        # round((10-1)/8) = round(1.125) = 1 -> 9
        assert processor.get_closest_valid_frame_count(10) == 9
        # round((13-1)/8) = round(1.5) = 2 -> 17
        assert processor.get_closest_valid_frame_count(13) == 17
        # round((14-1)/8) = round(1.625) = 2 -> 17
        assert processor.get_closest_valid_frame_count(14) == 17
        # round((20-1)/8) = round(2.375) = 2 -> 17
        assert processor.get_closest_valid_frame_count(20) == 17
        # round((22-1)/8) = round(2.625) = 3 -> 25
        assert processor.get_closest_valid_frame_count(22) == 25

    def test_closest_valid_frame_count_minimum(self, processor):
        # Minimum valid count is 9 (n=1)
        assert processor.get_closest_valid_frame_count(1) == 9
        assert processor.get_closest_valid_frame_count(5) == 9

    def test_adjust_frame_count_no_change(self, processor):
        frames = np.random.randint(0, 255, (9, 64, 64, 3), dtype=np.uint8)
        result = processor.adjust_frame_count(frames, 9)
        assert len(result) == 9

    def test_adjust_frame_count_downsample(self, processor):
        frames = np.random.randint(0, 255, (30, 64, 64, 3), dtype=np.uint8)
        result = processor.adjust_frame_count(frames, 25)
        assert len(result) == 25

    def test_adjust_frame_count_upsample(self, processor):
        frames = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)
        result = processor.adjust_frame_count(frames, 9)
        assert len(result) == 9


# ---------------------------------------------------------------------------
# Encode video (mocked VAE)
# ---------------------------------------------------------------------------
class TestEncodeVideo:
    def _make_mock_models(self):
        vae = MagicMock()
        # LTX VAE has latents_mean/latents_std as tensors on the model
        vae.latents_mean = torch.zeros(128)
        vae.latents_std = torch.ones(128)
        vae.config.scaling_factor = 1.0

        latent_mean = torch.randn(1, 128, 2, 8, 8)
        latent_sample = torch.randn(1, 128, 2, 8, 8)

        mock_dist = MagicMock()
        mock_dist.latent_dist.mean = latent_mean
        mock_dist.latent_dist.sample.return_value = latent_sample
        vae.encode.return_value = mock_dist

        return {"vae": vae, "dtype": torch.float32}, latent_mean, latent_sample

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

    def test_encode_video_applies_normalization(self, processor):
        vae = MagicMock()
        vae.latents_mean = torch.full((128,), 0.5)
        vae.latents_std = torch.full((128,), 2.0)
        vae.config.scaling_factor = 1.0

        latent_mean = torch.randn(1, 128, 2, 8, 8)
        mock_dist = MagicMock()
        mock_dist.latent_dist.mean = latent_mean
        vae.encode.return_value = mock_dist

        models = {"vae": vae, "dtype": torch.float32}
        video_tensor = torch.randn(1, 3, 9, 64, 64)

        result = processor.encode_video(video_tensor, models, device="cpu", deterministic=True)

        # Expected: (latent - mean) * scaling / std = (latent - 0.5) * 1.0 / 2.0
        mean_t = torch.full((1, 128, 1, 1, 1), 0.5)
        std_t = torch.full((1, 128, 1, 1, 1), 2.0)
        expected = (latent_mean - mean_t) * 1.0 / std_t

        torch.testing.assert_close(result.float(), expected.float(), atol=1e-3, rtol=1e-3)

    def test_encode_video_missing_latents_mean_raises(self, processor):
        vae = MagicMock()
        del vae.latents_mean
        vae.latents_std = torch.ones(128)

        mock_dist = MagicMock()
        mock_dist.latent_dist.mean = torch.randn(1, 128, 2, 8, 8)
        vae.encode.return_value = mock_dist

        models = {"vae": vae, "dtype": torch.float32}
        video_tensor = torch.randn(1, 3, 9, 64, 64)

        with pytest.raises(ValueError, match="latents_mean"):
            processor.encode_video(video_tensor, models, device="cpu")

    def test_encode_video_missing_latents_std_raises(self, processor):
        vae = MagicMock()
        vae.latents_mean = torch.zeros(128)
        del vae.latents_std

        mock_dist = MagicMock()
        mock_dist.latent_dist.mean = torch.randn(1, 128, 2, 8, 8)
        vae.encode.return_value = mock_dist

        models = {"vae": vae, "dtype": torch.float32}
        video_tensor = torch.randn(1, 3, 9, 64, 64)

        with pytest.raises(ValueError, match="latents_std"):
            processor.encode_video(video_tensor, models, device="cpu")


# ---------------------------------------------------------------------------
# Encode text (mocked tokenizer and encoder)
# ---------------------------------------------------------------------------
class TestEncodeText:
    @staticmethod
    def _make_tokenizer_output(input_ids, attention_mask):
        output = MagicMock()
        output.input_ids = input_ids
        output.attention_mask = attention_mask
        output.items.return_value = [
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
        ]
        output.__getitem__ = lambda self, key: getattr(self, key)
        return output

    def _make_mock_models(self, seq_len=10, hidden_dim=4096):
        max_len = LTXProcessor.MAX_SEQUENCE_LENGTH

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
        assert "text_embeddings" in result
        assert "text_mask" in result

    def test_encode_text_output_shape(self, processor):
        models, hidden_dim = self._make_mock_models(seq_len=15)
        result = processor.encode_text("hello world", models, device="cpu")
        assert result["text_embeddings"].shape == (1, 128, hidden_dim)
        assert result["text_mask"].shape == (1, 128)

    def test_encode_text_on_cpu(self, processor):
        models, _ = self._make_mock_models()
        result = processor.encode_text("test", models, device="cpu")
        assert result["text_embeddings"].device == torch.device("cpu")
        assert result["text_mask"].device == torch.device("cpu")

    def test_encode_text_tokenizer_args(self, processor):
        models, _ = self._make_mock_models()
        processor.encode_text("test prompt", models, device="cpu")

        models["tokenizer"].assert_called_once_with(
            "test prompt",
            max_length=128,
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
        latent = torch.randn(1, 128, 2, 8, 8)
        text_encodings = {
            "text_embeddings": torch.randn(1, 128, 4096),
            "text_mask": torch.ones(1, 128),
        }
        first_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        metadata = {
            "first_frame": first_frame,
            "original_resolution": (1920, 1080),
            "bucket_resolution": (672, 384),
            "bucket_id": "672x384",
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
        assert result["text_mask"] is text_encodings["text_mask"]
        assert isinstance(result["first_frame"], torch.Tensor)
        assert result["model_type"] == "ltx"
        assert result["model_version"] == "ltx-video"
        assert result["processing_mode"] == "video"
        assert result["num_frames"] == 25
        assert result["prompt"] == "test prompt"

    def test_cache_data_none_first_frame(self, processor):
        latent = torch.randn(1, 128, 2, 8, 8)
        text_encodings = {
            "text_embeddings": torch.randn(1, 128, 4096),
            "text_mask": torch.ones(1, 128),
        }
        metadata = {"first_frame": None}

        result = processor.get_cache_data(latent, text_encodings, metadata)
        assert result["first_frame"] is None

    def test_cache_data_missing_optional_metadata(self, processor):
        latent = torch.randn(1, 128, 2, 8, 8)
        text_encodings = {
            "text_embeddings": torch.randn(1, 128, 4096),
        }
        metadata = {}

        result = processor.get_cache_data(latent, text_encodings, metadata)
        assert result["first_frame"] is None
        assert result["original_resolution"] is None
        assert result["deterministic_latents"] is True
        assert result["processing_mode"] == "video"

    def test_cache_data_has_required_keys(self, processor):
        """Verify keys required by collate functions are present."""
        latent = torch.randn(1, 128, 2, 8, 8)
        text_encodings = {
            "text_embeddings": torch.randn(1, 128, 4096),
            "text_mask": torch.ones(1, 128),
        }
        metadata = {"first_frame": None}

        result = processor.get_cache_data(latent, text_encodings, metadata)

        # Required by collate_fn_video
        assert "video_latents" in result
        assert "text_embeddings" in result
        # Optional VIDEO_OPTIONAL_FIELDS
        assert "text_mask" in result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestLTXRegistry:
    def test_registered_names(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert ProcessorRegistry.is_registered("ltx")
        assert ProcessorRegistry.is_registered("ltx-video")

    def test_get_returns_correct_type(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        proc = ProcessorRegistry.get("ltx")
        assert isinstance(proc, LTXProcessor)

    def test_both_aliases_return_same_class(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert ProcessorRegistry.get_class("ltx") is ProcessorRegistry.get_class("ltx-video")

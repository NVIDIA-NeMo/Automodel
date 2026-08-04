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

from tools.diffusion.processors.hunyuan import HunyuanVideoProcessor


@pytest.fixture
def processor():
    return HunyuanVideoProcessor()


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
class TestHunyuanProperties:
    def test_model_type(self, processor):
        assert processor.model_type == "hunyuanvideo"

    def test_default_model_name(self, processor):
        assert "HunyuanVideo" in processor.default_model_name

    def test_supported_modes(self, processor):
        assert processor.supported_modes == ["video"]

    def test_frame_constraint(self, processor):
        assert processor.frame_constraint == "4n+1"

    def test_quantization(self, processor):
        assert processor.quantization == 16

    def test_default_image_embed_shape(self, processor):
        assert processor.DEFAULT_IMAGE_EMBED_SHAPE == (729, 1152)


# ---------------------------------------------------------------------------
# Frame count validation
# ---------------------------------------------------------------------------
class TestValidateFrameCount:
    @pytest.mark.parametrize("n", [1, 5, 9, 13, 17, 21, 61, 121])
    def test_valid_4n_plus_1(self, processor, n):
        assert processor.validate_frame_count(n) is True

    @pytest.mark.parametrize("n", [0, 2, 3, 4, 6, 7, 8, 10, 12, 20, 100])
    def test_invalid_frame_counts(self, processor, n):
        assert processor.validate_frame_count(n) is False


# ---------------------------------------------------------------------------
# Closest valid frame count
# ---------------------------------------------------------------------------
class TestGetClosestValidFrameCount:
    @pytest.mark.parametrize(
        "input_frames, expected",
        [
            (1, 1),
            (2, 1),
            (3, 1),  # equidistant, pick lower
            (4, 5),
            (5, 5),
            (6, 5),
            (7, 5),  # equidistant, pick lower
            (8, 9),
            (9, 9),
            (10, 9),
            (12, 13),
            (13, 13),
            (120, 121),
            (121, 121),
        ],
    )
    def test_closest_valid(self, processor, input_frames, expected):
        assert processor.get_closest_valid_frame_count(input_frames) == expected

    def test_result_always_satisfies_constraint(self, processor):
        for n in range(1, 200):
            result = processor.get_closest_valid_frame_count(n)
            assert processor.validate_frame_count(result), f"Failed for input {n}, got {result}"
            assert result >= 1


# ---------------------------------------------------------------------------
# Adjust frame count
# ---------------------------------------------------------------------------
class TestAdjustFrameCount:
    def test_same_count_returns_unchanged(self, processor):
        frames = np.random.randint(0, 255, (9, 64, 64, 3), dtype=np.uint8)
        result = processor.adjust_frame_count(frames, 9)
        np.testing.assert_array_equal(result, frames)

    def test_downsample(self, processor):
        frames = np.random.randint(0, 255, (20, 32, 32, 3), dtype=np.uint8)
        result = processor.adjust_frame_count(frames, 5)
        assert result.shape == (5, 32, 32, 3)

    def test_upsample(self, processor):
        frames = np.random.randint(0, 255, (3, 32, 32, 3), dtype=np.uint8)
        result = processor.adjust_frame_count(frames, 9)
        assert result.shape == (9, 32, 32, 3)

    def test_invalid_target_raises(self, processor):
        frames = np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="4n\\+1"):
            processor.adjust_frame_count(frames, 10)

    @pytest.mark.parametrize("target", [1, 5, 9, 13])
    def test_valid_targets_accepted(self, processor, target):
        frames = np.random.randint(0, 255, (20, 16, 16, 3), dtype=np.uint8)
        result = processor.adjust_frame_count(frames, target)
        assert len(result) == target

    def test_preserves_first_and_last_frame_on_downsample(self, processor):
        """Uniform sampling should include first and last frames."""
        frames = np.arange(20 * 4 * 4 * 3, dtype=np.uint8).reshape(20, 4, 4, 3)
        result = processor.adjust_frame_count(frames, 5)
        np.testing.assert_array_equal(result[0], frames[0])
        np.testing.assert_array_equal(result[-1], frames[-1])


# ---------------------------------------------------------------------------
# Encode video (mocked VAE)
# ---------------------------------------------------------------------------
class TestEncodeVideo:
    def _make_mock_models(self, shift_factor=0.5, scaling_factor=0.7):
        vae = MagicMock()
        vae.config.shift_factor = shift_factor
        vae.config.scaling_factor = scaling_factor

        latent_mean = torch.randn(1, 16, 3, 8, 8)
        latent_sample = torch.randn(1, 16, 3, 8, 8)

        mock_dist = MagicMock()
        mock_dist.latent_dist.mean = latent_mean
        mock_dist.latent_dist.sample.return_value = latent_sample
        vae.encode.return_value = mock_dist

        return (
            {
                "vae": vae,
                "dtype": torch.float32,
                "device": "cpu",
            },
            latent_mean,
            latent_sample,
        )

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

    def test_encode_video_applies_shift_and_scale(self, processor):
        shift = 0.5
        scale = 2.0
        models, latent_mean, _ = self._make_mock_models(shift_factor=shift, scaling_factor=scale)
        video_tensor = torch.randn(1, 3, 5, 32, 32)

        result = processor.encode_video(video_tensor, models, device="cpu", deterministic=True)

        expected = (latent_mean - shift) * scale
        torch.testing.assert_close(result.float(), expected.float(), atol=1e-3, rtol=1e-3)

    def test_encode_video_no_shift_factor(self, processor):
        models, latent_mean, _ = self._make_mock_models(shift_factor=None, scaling_factor=2.0)
        models["vae"].config.shift_factor = None
        video_tensor = torch.randn(1, 3, 5, 32, 32)

        result = processor.encode_video(video_tensor, models, device="cpu", deterministic=True)

        expected = latent_mean * 2.0
        torch.testing.assert_close(result.float(), expected.float(), atol=1e-3, rtol=1e-3)

    @pytest.mark.run_only_on("GPU")
    def test_encode_video_gpu(self, processor):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        latent_mean = torch.randn(1, 16, 3, 8, 8, device="cuda")
        mock_dist = MagicMock()
        mock_dist.latent_dist.mean = latent_mean
        mock_dist.latent_dist.sample.return_value = latent_mean

        vae = MagicMock()
        vae.config.shift_factor = 0.5
        vae.config.scaling_factor = 0.7
        vae.encode.return_value = mock_dist

        models = {"vae": vae, "dtype": torch.float16, "device": "cuda"}
        video_tensor = torch.randn(1, 3, 5, 64, 64)

        result = processor.encode_video(video_tensor, models, device="cuda", deterministic=True)

        assert result.device == torch.device("cpu")
        assert result.dtype == torch.float16


# ---------------------------------------------------------------------------
# Encode text (mocked pipeline)
# ---------------------------------------------------------------------------
class TestEncodeText:
    def test_encode_text_returns_expected_keys(self, processor):
        prompt_embeds = torch.randn(1, 77, 768)
        prompt_mask = torch.ones(1, 77)
        prompt_embeds_2 = torch.randn(1, 128, 1024)
        prompt_mask_2 = torch.ones(1, 128)

        pipeline = MagicMock()
        pipeline.encode_prompt.return_value = (
            prompt_embeds,
            prompt_mask,
            prompt_embeds_2,
            prompt_mask_2,
        )

        models = {"pipeline": pipeline, "dtype": torch.float16}

        result = processor.encode_text("a test prompt", models, device="cpu")

        assert set(result.keys()) == {
            "text_embeddings",
            "text_mask",
            "text_embeddings_2",
            "text_mask_2",
        }
        assert result["text_embeddings"].shape == prompt_embeds.shape
        assert result["text_mask"].shape == prompt_mask.shape
        assert result["text_embeddings_2"].shape == prompt_embeds_2.shape
        assert result["text_mask_2"].shape == prompt_mask_2.shape

    def test_encode_text_all_on_cpu(self, processor):
        pipeline = MagicMock()
        pipeline.encode_prompt.return_value = (
            torch.randn(1, 10, 64),
            torch.ones(1, 10),
            torch.randn(1, 20, 128),
            torch.ones(1, 20),
        )
        models = {"pipeline": pipeline, "dtype": torch.float32}
        result = processor.encode_text("hello", models, device="cpu")
        for v in result.values():
            assert v.device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Encode first frame (mocked pipeline)
# ---------------------------------------------------------------------------
class TestEncodeFirstFrame:
    def test_encode_first_frame_numpy(self, processor):
        image_embeds = torch.randn(1, 729, 1152)
        pipeline = MagicMock()
        pipeline.encode_image.return_value = image_embeds

        models = {"pipeline": pipeline, "dtype": torch.float16}
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = processor.encode_first_frame(frame, models, device="cpu")

        pipeline.encode_image.assert_called_once()
        assert result.device == torch.device("cpu")

    def test_encode_first_frame_pil(self, processor):
        from PIL import Image

        image_embeds = torch.randn(1, 729, 1152)
        pipeline = MagicMock()
        pipeline.encode_image.return_value = image_embeds

        models = {"pipeline": pipeline, "dtype": torch.float16}
        frame = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

        processor.encode_first_frame(frame, models, device="cpu")
        pipeline.encode_image.assert_called_once()
        # When input is PIL, it should be passed as-is
        call_kwargs = pipeline.encode_image.call_args
        assert call_kwargs.kwargs["image"] is frame


# ---------------------------------------------------------------------------
# Cache data
# ---------------------------------------------------------------------------
class TestGetCacheData:
    def test_cache_data_structure(self, processor):
        latent = torch.randn(1, 16, 3, 8, 8)
        text_encodings = {
            "text_embeddings": torch.randn(1, 77, 768),
            "text_mask": torch.ones(1, 77),
            "text_embeddings_2": torch.randn(1, 128, 1024),
            "text_mask_2": torch.ones(1, 128),
        }
        metadata = {
            "image_embeds": torch.randn(1, 729, 1152),
            "original_resolution": (1920, 1080),
            "bucket_resolution": (1280, 720),
            "bucket_id": "1280x720",
            "aspect_ratio": 16 / 9,
            "num_frames": 17,
            "prompt": "test prompt",
            "video_path": "/path/to/video.mp4",
            "deterministic": True,
            "mode": "video",
        }

        result = processor.get_cache_data(latent, text_encodings, metadata)

        assert result["video_latents"] is latent
        assert result["text_embeddings"] is text_encodings["text_embeddings"]
        assert result["text_mask"] is text_encodings["text_mask"]
        assert result["text_embeddings_2"] is text_encodings["text_embeddings_2"]
        assert result["text_mask_2"] is text_encodings["text_mask_2"]
        assert result["image_embeds"] is metadata["image_embeds"]
        assert result["model_type"] == "hunyuanvideo"
        assert result["model_version"] == "hunyuanvideo-1.5"
        assert result["processing_mode"] == "video"
        assert result["deterministic_latents"] is True
        assert result["num_frames"] == 17
        assert result["prompt"] == "test prompt"

    def test_cache_data_missing_optional_metadata(self, processor):
        latent = torch.randn(1, 16, 3, 8, 8)
        text_encodings = {
            "text_embeddings": torch.randn(1, 10, 64),
            "text_mask": torch.ones(1, 10),
            "text_embeddings_2": torch.randn(1, 20, 128),
            "text_mask_2": torch.ones(1, 20),
        }
        metadata = {}

        result = processor.get_cache_data(latent, text_encodings, metadata)

        assert result["image_embeds"] is None
        assert result["original_resolution"] is None
        assert result["num_frames"] is None
        assert result["deterministic_latents"] is True  # default
        assert result["processing_mode"] == "video"  # default


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestHunyuanRegistry:
    def test_registered_names(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert ProcessorRegistry.is_registered("hunyuan")
        assert ProcessorRegistry.is_registered("hunyuanvideo")
        assert ProcessorRegistry.is_registered("hunyuanvideo-1.5")

    def test_get_returns_correct_type(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        proc = ProcessorRegistry.get("hunyuan")
        assert isinstance(proc, HunyuanVideoProcessor)

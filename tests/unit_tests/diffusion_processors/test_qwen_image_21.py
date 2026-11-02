# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from tools.diffusion.processors.qwen_image_21 import QwenImage21Processor


@pytest.fixture
def processor():
    return QwenImage21Processor()


def _mock_vae(channels=64, latents_mean=None, latents_std=None):
    vae = MagicMock()
    vae.config.latents_mean = latents_mean if latents_mean is not None else [0.0] * channels
    vae.config.latents_std = latents_std if latents_std is not None else [1.0] * channels
    latent = torch.randn(1, channels, 1, 4, 4)
    vae.encode.return_value.latent_dist.sample.return_value = latent
    return vae, latent


class TestProperties:
    def test_model_type(self, processor):
        assert processor.model_type == "qwen_image_21"

    def test_default_model_name(self, processor):
        assert processor.default_model_name == "Qwen/Qwen-Image-2.1"

    def test_registered(self):
        from tools.diffusion.processors.registry import ProcessorRegistry

        assert isinstance(ProcessorRegistry.get("qwen_image_21"), QwenImage21Processor)


class TestEncodeImage:
    def test_adds_opaque_alpha_to_rgb(self, processor):
        vae, _ = _mock_vae()
        image = torch.rand(1, 3, 64, 64) * 2 - 1
        processor.encode_image(image, {"vae": vae}, device="cpu")

        vae_input = vae.encode.call_args[0][0]
        assert vae_input.shape == (1, 4, 1, 64, 64)
        torch.testing.assert_close(vae_input[:, :3, 0], image.to(torch.bfloat16))
        assert (vae_input[:, 3] == 1).all()

    def test_keeps_rgba_input(self, processor):
        vae, _ = _mock_vae()
        image = torch.rand(1, 4, 64, 64) * 2 - 1
        processor.encode_image(image, {"vae": vae}, device="cpu")
        torch.testing.assert_close(vae.encode.call_args[0][0][:, :, 0], image.to(torch.bfloat16))

    def test_normalizes_and_squeezes(self, processor):
        vae, latent = _mock_vae(latents_mean=[1.0] * 64, latents_std=[2.0] * 64)
        result = processor.encode_image(torch.zeros(1, 3, 64, 64), {"vae": vae}, device="cpu")

        assert result.shape == (64, 4, 4)
        assert result.dtype == torch.float16
        expected = ((latent - 1.0) / 2.0).squeeze(2).squeeze(0).to(torch.float16)
        torch.testing.assert_close(result, expected)


class TestEncodeText:
    def test_returns_prompt_embeds(self, processor):
        prompt_embeds = torch.randn(1, 12, 4096)
        pipeline = MagicMock()
        pipeline.encode_prompt.return_value = (prompt_embeds, None, torch.zeros(1, 12, dtype=torch.bool))

        result = processor.encode_text("a fox", {"pipeline": pipeline}, device="cpu")

        assert set(result) == {"prompt_embeds"}
        assert result["prompt_embeds"].dtype == torch.bfloat16
        torch.testing.assert_close(result["prompt_embeds"], prompt_embeds.to(torch.bfloat16))
        pipeline.encode_prompt.assert_called_once_with(prompt="a fox", device="cpu")

    def test_rejects_image_tokens(self, processor):
        pipeline = MagicMock()
        pipeline.encode_prompt.return_value = (torch.randn(1, 4, 8), None, torch.ones(1, 4, dtype=torch.bool))
        with pytest.raises(ValueError, match="image tokens"):
            processor.encode_text("a fox", {"pipeline": pipeline}, device="cpu")


class TestVerifyLatent:
    def _models(self, decoded):
        vae = MagicMock()
        vae.config.latents_mean = [0.0] * 64
        vae.config.latents_std = [1.0] * 64
        vae.dtype = torch.float32
        vae.decode.return_value.sample = decoded
        return {"vae": vae}

    def test_rgba_passes(self, processor):
        assert processor.verify_latent(torch.randn(64, 4, 4), self._models(torch.rand(1, 4, 1, 64, 64)), "cpu")

    def test_rgb_output_fails(self, processor):
        assert not processor.verify_latent(torch.randn(64, 4, 4), self._models(torch.rand(1, 3, 1, 64, 64)), "cpu")

    def test_nan_fails(self, processor):
        decoded = torch.rand(1, 4, 1, 64, 64)
        decoded[0, 0, 0, 0, 0] = float("nan")
        assert not processor.verify_latent(torch.randn(64, 4, 4), self._models(decoded), "cpu")


def test_get_cache_data(processor):
    metadata = {
        "original_resolution": (1024, 768),
        "bucket_resolution": (1024, 768),
        "crop_offset": (0, 0),
        "prompt": "a fox",
        "image_path": "/tmp/fox.png",
        "bucket_id": 3,
        "aspect_ratio": 1.33,
    }
    latent = torch.randn(64, 48, 64)
    prompt_embeds = torch.randn(1, 10, 4096)
    data = processor.get_cache_data(latent, {"prompt_embeds": prompt_embeds}, metadata)

    assert data["latent"] is latent
    assert data["prompt_embeds"] is prompt_embeds
    assert data["model_type"] == "qwen_image_21"
    for key, value in metadata.items():
        assert data[key] == value

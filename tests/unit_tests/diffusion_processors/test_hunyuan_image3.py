# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.datasets.diffusion.text_to_image_dataset import (
    PROMPT_IDS_KEY,
    PROMPT_SUFFIX_IDS_KEY,
    UNCOND_PROMPT_IDS_KEY,
)
from tools.diffusion.processors import hunyuan_image3
from tools.diffusion.processors.hunyuan_image3 import HunyuanImage3Processor

TIMESTEP, EOI = 13, 15


class _FakePromptTokenizer:
    """Stands in for ``HunyuanImage3PromptTokenizer`` (its own tests live with the model)."""

    def __init__(self):
        self.calls = []

    def target_size(self, width, height):
        return (1280, 768) if width > height else (1024, 1024)

    def __call__(self, prompt, height, width):
        self.calls.append((prompt, height, width))
        return {
            PROMPT_IDS_KEY: torch.tensor([1, 101, TIMESTEP]),
            UNCOND_PROMPT_IDS_KEY: torch.tensor([1, 16, TIMESTEP]),
            PROMPT_SUFFIX_IDS_KEY: torch.tensor([EOI]),
        }


def _processor() -> HunyuanImage3Processor:
    processor = HunyuanImage3Processor()
    processor._config = SimpleNamespace(vae={"latent_channels": 32})
    processor._prompt_tokenizer = _FakePromptTokenizer()
    return processor


def _metadata(width: int = 64, height: int = 32) -> dict:
    return {
        "bucket_resolution": (width, height),
        "original_resolution": (width, height),
        "crop_offset": (0, 0),
        "prompt": "a cat",
        "image_path": "/data/cat.png",
        "bucket_id": 3,
        "aspect_ratio": width / height,
    }


def test_properties():
    processor = HunyuanImage3Processor()
    assert processor.model_type == "hunyuan_image3"
    assert processor.default_model_name == "tencent/HunyuanImage-3.0"


def test_load_models_uses_the_release_assets(monkeypatch):
    config = SimpleNamespace(vae={"latent_channels": 32})
    prompt_tokenizer, vae = object(), object()
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda path, trust_remote_code: config)
    monkeypatch.setattr("nemo_automodel._diffusers._hf_cache.resolve_diffusion_model_dir", lambda name: "/ckpt")
    monkeypatch.setattr(
        hunyuan_image3.HunyuanImage3PromptTokenizer, "from_pretrained", classmethod(lambda cls, d, c: prompt_tokenizer)
    )
    monkeypatch.setattr(hunyuan_image3, "load_release_vae", lambda d, vae_config, device: (d, vae_config, device, vae))
    processor = HunyuanImage3Processor()
    models = processor.load_models("tencent/HunyuanImage-3.0", "cpu")
    assert models == {"vae": ("/ckpt", config.vae, "cpu", vae)}
    assert processor._config is config and processor._prompt_tokenizer is prompt_tokenizer


def test_get_cache_data():
    processor = _processor()
    latent = torch.randn(32, 2, 4)
    text = processor.encode_text("a cat", models={}, device="cpu")
    cache = processor.get_cache_data(latent, text, _metadata())
    assert processor._prompt_tokenizer.calls == [("a cat", 32, 64)]
    assert cache["latent"] is latent
    assert cache[PROMPT_IDS_KEY][-1].item() == TIMESTEP
    assert cache[PROMPT_SUFFIX_IDS_KEY].tolist() == [EOI]
    assert cache["model_type"] == "hunyuan_image3" and cache["bucket_id"] == 3
    with pytest.raises(RuntimeError, match="does not match bucket"):  # transposed latent, same token count
        processor.get_cache_data(torch.randn(32, 4, 2), text, _metadata())
    with pytest.raises(RuntimeError, match="does not match bucket"):
        processor.get_cache_data(torch.randn(32, 3, 3), text, _metadata())


def test_encode_image_samples_scales_and_drops_frame_axis():
    vae = MagicMock()
    vae.config.scaling_factor = 0.5
    latent = torch.ones(1, 32, 1, 2, 4)
    vae.encode.return_value.latent_dist.sample.return_value = latent
    out = HunyuanImage3Processor().encode_image(torch.zeros(1, 3, 32, 64), {"vae": vae}, device="cpu")
    assert out.shape == (32, 2, 4) and out.dtype == torch.float32
    assert torch.equal(out, torch.full((32, 2, 4), 0.5))
    vae.encode.return_value.latent_dist.sample.return_value = torch.ones(1, 32, 2, 2, 4)
    with pytest.raises(ValueError, match="one latent frame"):
        HunyuanImage3Processor().encode_image(torch.zeros(1, 3, 32, 64), {"vae": vae}, device="cpu")


def test_target_resolution_uses_release_group_not_generic_buckets():
    generic_bucket = {"resolution": (512, 384)}
    processor = _processor()
    assert processor.target_resolution(1920, 1080, generic_bucket) == (1280, 768)
    assert processor.target_resolution(500, 500, generic_bucket) == (1024, 1024)


def test_base_target_resolution_uses_bucket():
    from tools.diffusion.processors.qwen_image import QwenImageProcessor

    assert QwenImageProcessor().target_resolution(1000, 750, {"resolution": [512, 384]}) == (512, 384)


def test_verify_latent():
    processor = _processor()
    assert processor.verify_latent(torch.zeros(32, 2, 2), {}, "cpu")
    assert not processor.verify_latent(torch.full((32, 2, 2), float("nan")), {}, "cpu")
    assert not processor.verify_latent(torch.zeros(16, 2, 2), {}, "cpu")

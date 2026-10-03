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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from tools.diffusion.processors import hunyuan_image3
from tools.diffusion.processors.hunyuan_image3 import HunyuanImage3Processor

BOS, BOI, SIZE, RATIO, TIMESTEP, IMG, EOI, CFG = 1, 10, 11, 12, 13, 14, 15, 16


class _FakeResoGroup:
    def get_target_size(self, width, height):
        return (1280, 768) if width > height else (1024, 1024)


class _FakeImageProcessor:
    reso_group = _FakeResoGroup()

    def build_image_info(self, size: str):
        height, width = (int(v) for v in size.split("x"))
        return SimpleNamespace(token_height=height // 16, token_width=width // 16)


class _FakeWrapper:
    """Mimics TokenizerWrapper.apply_chat_template for one prompt in gen_image mode with CFG."""

    def __init__(self, timestep_offset: int = 1, uncond_shift: int = 0, image_tokens: int | None = None):
        self.timestep_offset = timestep_offset
        self.uncond_shift = uncond_shift
        self.image_tokens = image_tokens
        self.calls = []

    def apply_chat_template(self, **kwargs):
        self.calls.append(kwargs)
        info = kwargs["batch_gen_image_info"][0]
        num_image = self.image_tokens or info.token_height * info.token_width
        prompt = [101, 102, 103]
        cond = [BOS, *prompt, BOI, SIZE, RATIO, TIMESTEP] + [IMG] * num_image + [EOI]
        uncond = [BOS, *[CFG] * len(prompt), BOI, SIZE, RATIO, TIMESTEP] + [IMG] * num_image + [EOI]
        start = len(prompt) + 5
        span = slice(start, start + num_image)
        uncond_span = slice(start + self.uncond_shift, start + self.uncond_shift + num_image)
        output = SimpleNamespace(
            tokens=torch.tensor([cond, uncond]),
            gen_image_slices=[[span], [uncond_span]],
            gen_timestep_scatter_index=torch.tensor([[start - self.timestep_offset], [start - 1]]),
        )
        return {"output": output, "sections": None}


def _processor(**wrapper_kwargs) -> HunyuanImage3Processor:
    processor = HunyuanImage3Processor()
    processor._config = SimpleNamespace(image_base_size=1024)
    processor._wrapper = _FakeWrapper(**wrapper_kwargs)
    processor._image_processor = _FakeImageProcessor()
    processor._sequence_template = "pretrain"
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


def test_build_prompt_tokens_splits_around_the_image_span():
    processor = _processor()
    tokens = processor.build_prompt_tokens("a cat", height=32, width=64)
    assert tokens["prompt_input_ids"].tolist() == [BOS, 101, 102, 103, BOI, SIZE, RATIO, TIMESTEP]
    assert tokens["uncond_prompt_input_ids"].tolist() == [BOS, CFG, CFG, CFG, BOI, SIZE, RATIO, TIMESTEP]
    assert tokens["prompt_suffix_ids"].tolist() == [EOI]
    call = processor._wrapper.calls[0]
    assert call["mode"] == "gen_image" and call["cfg_factor"] == 2 and call["sequence_template"] == "pretrain"


@pytest.mark.parametrize(
    "wrapper_kwargs, message",
    [
        ({"uncond_shift": 1}, "different positions"),
        ({"image_tokens": 3}, "does not hold"),
        ({"timestep_offset": 2}, "<timestep> right before"),
    ],
)
def test_build_prompt_tokens_validates_release_layout(wrapper_kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        _processor(**wrapper_kwargs).build_prompt_tokens("a cat", height=32, width=64)


def test_get_cache_data():
    processor = _processor()
    latent = torch.randn(32, 2, 4)
    text = processor.encode_text("a cat", models={}, device="cpu")
    cache = processor.get_cache_data(latent, text, _metadata())
    assert cache["latent"] is latent
    assert cache["prompt_input_ids"][-1].item() == TIMESTEP
    assert cache["prompt_suffix_ids"].tolist() == [EOI]
    assert cache["model_type"] == "hunyuan_image3" and cache["bucket_id"] == 3
    with pytest.raises(RuntimeError, match="does not match bucket"):
        processor.get_cache_data(torch.randn(32, 3, 3), text, _metadata())


def test_encode_image_samples_scales_and_drops_frame_axis():
    vae = MagicMock()
    vae.config.scaling_factor = 0.5
    latent = torch.ones(1, 32, 1, 2, 4)
    vae.encode.return_value.latent_dist.sample.return_value = latent
    out = HunyuanImage3Processor().encode_image(torch.zeros(1, 3, 32, 64), {"vae": vae}, device="cpu")
    assert out.shape == (32, 2, 4) and out.dtype == torch.bfloat16
    assert torch.equal(out, torch.full((32, 2, 4), 0.5, dtype=torch.bfloat16))
    vae.encode.return_value.latent_dist.sample.return_value = torch.ones(1, 32, 2, 2, 4)
    with pytest.raises(ValueError, match="one latent frame"):
        HunyuanImage3Processor().encode_image(torch.zeros(1, 3, 32, 64), {"vae": vae}, device="cpu")


def test_target_resolution_uses_release_group_not_generic_buckets():
    calculator = MagicMock()
    processor = _processor()
    assert processor.target_resolution(1920, 1080, calculator) == (1280, 768)
    assert processor.target_resolution(500, 500, calculator) == (1024, 1024)
    calculator.get_bucket_for_image.assert_not_called()


def test_base_target_resolution_uses_calculator():
    from tools.diffusion.processors.qwen_image import QwenImageProcessor

    calculator = MagicMock()
    calculator.get_bucket_for_image.return_value = {"resolution": (512, 384)}
    assert QwenImageProcessor().target_resolution(1000, 750, calculator) == (512, 384)


def test_verify_latent():
    processor = HunyuanImage3Processor()
    assert processor.verify_latent(torch.zeros(32, 2, 2), {}, "cpu")
    assert not processor.verify_latent(torch.full((32, 2, 2), float("nan")), {}, "cpu")
    assert not processor.verify_latent(torch.zeros(16, 2, 2), {}, "cpu")


class _TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    @classmethod
    def from_config(cls, config):
        return cls()


def test_load_vae_reads_only_vae_weights(tmp_path, monkeypatch):
    weights = {"vae.conv.weight": torch.full((4, 3, 1, 1), 2.0), "vae.conv.bias": torch.ones(4)}
    save_file(weights, str(tmp_path / "a.safetensors"))
    save_file({"model.wte.weight": torch.zeros(2, 2)}, str(tmp_path / "b.safetensors"))
    weight_map = {
        "vae.conv.weight": "a.safetensors",
        "vae.conv.bias": "a.safetensors",
        "model.wte.weight": "b.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    monkeypatch.setattr("transformers.dynamic_module_utils.get_class_from_dynamic_module", lambda name, path: _TinyVAE)
    vae = hunyuan_image3._load_vae(str(tmp_path), SimpleNamespace(vae={}), device="cpu")
    assert torch.equal(vae.conv.weight, weights["vae.conv.weight"]) and not vae.training

    save_file({"vae.other.weight": torch.zeros(1)}, str(tmp_path / "a.safetensors"))
    weight_map = {"vae.other.weight": "a.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(RuntimeError, match="do not match"):
        hunyuan_image3._load_vae(str(tmp_path), SimpleNamespace(vae={}), device="cpu")

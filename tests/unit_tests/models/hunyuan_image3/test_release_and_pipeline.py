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

"""CPU tests of the HunyuanImage-3.0 release assets (VAE, prompt format) and the sampling pipeline."""

import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from nemo_automodel.components.datasets.diffusion.text_to_image_dataset import (
    PROMPT_IDS_KEY,
    PROMPT_SUFFIX_IDS_KEY,
    UNCOND_PROMPT_IDS_KEY,
)
from nemo_automodel.components.models.hunyuan_image3 import pipeline as pipeline_module
from nemo_automodel.components.models.hunyuan_image3 import release
from nemo_automodel.components.models.hunyuan_image3.pipeline import HunyuanImage3Pipeline, flow_sigmas
from nemo_automodel.components.models.hunyuan_image3.release import HunyuanImage3PromptTokenizer
from tests.unit_tests.models.hunyuan_image3.test_hunyuan_image3 import IMAGE_ID, TIMESTEP_ID, _model

BOS, BOI, SIZE, RATIO, TIMESTEP, IMG, EOI, CFG = 1, 10, 11, 12, 13, 14, 15, 16


# ---------------------------------------------------------------------------------------------------------------
# Release prompt format and VAE
# ---------------------------------------------------------------------------------------------------------------


class _FakeResoGroup:
    def get_target_size(self, width, height):
        return (1280, 768) if width > height else (1024, 1024)


class _FakeImageProcessor:
    reso_group = _FakeResoGroup()

    def __init__(self, config=None):
        self.config = config

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


def _prompt_tokenizer(**wrapper_kwargs) -> HunyuanImage3PromptTokenizer:
    return HunyuanImage3PromptTokenizer(_FakeWrapper(**wrapper_kwargs), _FakeImageProcessor(), 1024, "pretrain")


def test_prompt_tokenizer_splits_around_the_image_span():
    tokenizer = _prompt_tokenizer()
    tokens = tokenizer("a cat", height=32, width=64)
    assert tokens[PROMPT_IDS_KEY].tolist() == [BOS, 101, 102, 103, BOI, SIZE, RATIO, TIMESTEP]
    assert tokens[UNCOND_PROMPT_IDS_KEY].tolist() == [BOS, CFG, CFG, CFG, BOI, SIZE, RATIO, TIMESTEP]
    assert tokens[PROMPT_SUFFIX_IDS_KEY].tolist() == [EOI]
    call = tokenizer.wrapper.calls[0]
    assert call["mode"] == "gen_image" and call["cfg_factor"] == 2 and call["sequence_template"] == "pretrain"
    assert call["bot_task"] == "auto" and call["batch_system_prompt"] is None and call["image_base_size"] == 1024


@pytest.mark.parametrize(
    "wrapper_kwargs, message",
    [
        ({"uncond_shift": 1}, "different positions"),
        ({"image_tokens": 3}, "does not hold"),
        ({"timestep_offset": 2}, "<timestep> right before"),
    ],
)
def test_prompt_tokenizer_validates_release_layout(wrapper_kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        _prompt_tokenizer(**wrapper_kwargs)("a cat", height=32, width=64)


def test_prompt_tokenizer_target_size_uses_release_group():
    assert _prompt_tokenizer().target_size(1920, 1080) == (1280, 768)
    assert _prompt_tokenizer().target_size(500, 500) == (1024, 1024)


def test_prompt_tokenizer_from_pretrained_loads_the_wrapper_next_to_the_image_processor(tmp_path, monkeypatch):
    (tmp_path / "generation_config.json").write_text(json.dumps({"sequence_template": "instruct"}))
    image_processor_cls = type("HunyuanImage3ImageProcessor", (_FakeImageProcessor,), {"__module__": "fakepkg.ip"})
    wrapper_module = types.ModuleType("fakepkg.tokenizer_wrapper")
    wrapper_module.TokenizerWrapper = lambda tokenizer: ("wrapper", tokenizer)
    monkeypatch.setitem(sys.modules, "fakepkg.tokenizer_wrapper", wrapper_module)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda path, trust_remote_code: "tok")
    monkeypatch.setattr(
        "transformers.dynamic_module_utils.get_class_from_dynamic_module", lambda name, path: image_processor_cls
    )
    config = SimpleNamespace(image_base_size=1024)
    tokenizer = HunyuanImage3PromptTokenizer.from_pretrained(str(tmp_path), config)
    assert tokenizer.wrapper == ("wrapper", "tok")
    assert isinstance(tokenizer.image_processor, image_processor_cls) and tokenizer.image_processor.config is config
    assert tokenizer.sequence_template == "instruct" and tokenizer.image_base_size == 1024


class _TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    @classmethod
    def from_config(cls, config):
        return cls()


def test_load_release_vae_reads_only_vae_weights(tmp_path, monkeypatch):
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
    vae = release.load_release_vae(str(tmp_path), {}, device="cpu")
    assert torch.equal(vae.conv.weight, weights["vae.conv.weight"]) and not vae.training

    save_file({"vae.other.weight": torch.zeros(1)}, str(tmp_path / "a.safetensors"))
    weight_map = {"vae.other.weight": "a.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(RuntimeError, match="do not match"):
        release.load_release_vae(str(tmp_path), {}, device="cpu")


# ---------------------------------------------------------------------------------------------------------------
# Sampling pipeline
# ---------------------------------------------------------------------------------------------------------------


def test_flow_sigmas_follow_the_release_shift():
    assert torch.allclose(flow_sigmas(4, 1.0), torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
    sigmas = flow_sigmas(2, 3.0)
    assert torch.allclose(sigmas, torch.tensor([1.0, 0.75, 0.0]))  # 3 * 0.5 / (1 + 2 * 0.5)
    assert sigmas.dtype == torch.float32


class _ConstantTransformer(nn.Module):
    """Predicts ``cond_value`` for the prompt row and ``uncond_value`` for the ``<cfg>`` row; records its inputs."""

    def __init__(self, cond_value: float = 1.0, uncond_value: float = -1.0, channels: int = 4):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.config = SimpleNamespace(image_token_id=IMG, vae={"latent_channels": channels})
        self.values = torch.tensor([cond_value, uncond_value])
        self.calls = []

    def forward(self, input_ids, latents, timestep):
        self.calls.append((input_ids, latents, timestep))
        values = self.values[: latents.shape[0]].to(latents.dtype)
        return (values.view(-1, 1, 1, 1).expand_as(latents).clone(),)


class _RecordingVAE(nn.Module):
    def __init__(self, temporal: bool = False):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.5)
        if temporal:
            self.ffactor_temporal = 4
        self.decoded = None

    def decode(self, latents, return_dict, generator):
        self.decoded = latents
        image = torch.zeros(latents.shape[0], 3, *latents.shape[2:]) if latents.ndim == 4 else latents[:, :3]
        return (image,)


def _pipeline(transformer, vae=None, tokenizer=None, flow_shift=3.0):
    return HunyuanImage3Pipeline(
        transformer, vae or _RecordingVAE(), tokenizer or _prompt_tokenizer(), flow_shift, torch.device("cpu")
    )


def _initial_latents(seed: int, shape) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.bfloat16).float()


def test_pipeline_runs_guided_euler_steps_and_decodes():
    transformer, vae = _ConstantTransformer(cond_value=1.0, uncond_value=-1.0), _RecordingVAE()
    pipe = _pipeline(transformer, vae)
    out = pipe(
        "a cat", torch.Generator().manual_seed(0), num_inference_steps=3, guidance_scale=4.0, height=32, width=64
    )

    # The prompt is snapped to the release resolution group (here 1280x768, a 48x80 latent grid).
    input_ids, latents, _ = transformer.calls[0]
    assert input_ids.shape[0] == 2 and latents.shape == (2, 4, 48, 80) and latents.dtype == torch.bfloat16
    assert input_ids[0, :8].tolist() == [BOS, 101, 102, 103, BOI, SIZE, RATIO, TIMESTEP]
    assert input_ids[1, :8].tolist() == [BOS, CFG, CFG, CFG, BOI, SIZE, RATIO, TIMESTEP]
    assert (input_ids[:, 8:-1] == IMG).all() and input_ids.shape[1] == 8 + 48 * 80 + 1
    timesteps = torch.stack([call[2] for call in transformer.calls])
    assert torch.equal(timesteps, (flow_sigmas(3, 3.0)[:-1] * 1000)[:, None].expand(3, 2))

    # uncond + 4 * (cond - uncond) = 7, integrated over sigma from 1 to 0: x_0 = x_1 - 7.
    expected = _initial_latents(0, (1, 4, 48, 80)) - 7.0
    assert torch.allclose(vae.decoded, expected / 2.0 + 0.5, atol=1e-5)
    image = out.images[0]
    assert image.size == (80, 48) and image.getpixel((0, 0)) == (128, 128, 128)


def test_pipeline_without_guidance_runs_the_prompt_row_only():
    transformer, vae = _ConstantTransformer(cond_value=1.0), _RecordingVAE()
    _pipeline(transformer, vae)(
        "a cat", torch.Generator().manual_seed(1), 2, guidance_scale=1.0, height=1024, width=1024
    )
    assert all(call[0].shape[0] == 1 for call in transformer.calls)
    expected = _initial_latents(1, (1, 4, 64, 64)) - 1.0
    assert torch.allclose(vae.decoded, expected / 2.0 + 0.5, atol=1e-5)


def test_pipeline_decodes_through_the_temporal_vae_axis():
    vae = _RecordingVAE(temporal=True)
    out = _pipeline(_ConstantTransformer(), vae)("a cat", None, 1, guidance_scale=1.0, height=1024, width=1024)
    assert vae.decoded.shape == (1, 4, 1, 64, 64)
    assert out.images[0].size == (64, 64)


class _FixedTokenizer:
    """Fixed ids in the tiny model's vocabulary; keeps the requested size."""

    def target_size(self, width, height):
        return width, height

    def __call__(self, prompt, height, width):
        return {
            PROMPT_IDS_KEY: torch.tensor([5, 6, TIMESTEP_ID]),
            UNCOND_PROMPT_IDS_KEY: torch.tensor([7, 7, TIMESTEP_ID]),
            PROMPT_SUFFIX_IDS_KEY: torch.tensor([3]),
        }


def test_pipeline_samples_with_the_native_model():
    transformer, vae = _model(), _RecordingVAE()
    pipe = HunyuanImage3Pipeline(transformer, vae, _FixedTokenizer(), 3.0, torch.device("cpu"))
    out = pipe("a cat", torch.Generator().manual_seed(0), 2, guidance_scale=5.0, height=32, width=48)
    assert transformer.config.image_token_id == IMAGE_ID
    assert vae.decoded.shape == (1, 4, 2, 3) and torch.isfinite(vae.decoded).all()
    assert out.images[0].size == (3, 2)


def test_from_transformer_reads_the_release_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "generation_config.json").write_text(json.dumps({"flow_shift": 2.5}))
    config = SimpleNamespace(vae={"latent_channels": 4})
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda path, trust_remote_code: config)
    monkeypatch.setattr(pipeline_module, "load_release_vae", lambda d, vae_config, device: ("vae", d, vae_config))
    monkeypatch.setattr(
        pipeline_module.HunyuanImage3PromptTokenizer, "from_pretrained", classmethod(lambda cls, d, c: ("tok", d, c))
    )
    transformer = _ConstantTransformer()
    pipe = HunyuanImage3Pipeline.from_transformer(transformer, str(tmp_path))
    assert pipe.transformer is transformer and pipe.flow_shift == 2.5
    assert pipe.vae == ("vae", str(tmp_path), config.vae) and pipe.prompt_tokenizer == ("tok", str(tmp_path), config)


def test_from_transformer_imports_the_remote_code_first_rank_first(tmp_path, monkeypatch):
    (tmp_path / "generation_config.json").write_text("{}")
    events = []

    class _RecordingFirstRank:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *exc):
            events.append("exit")

    monkeypatch.setattr(pipeline_module, "FirstRankPerNode", _RecordingFirstRank)
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda path, trust_remote_code: events.append("config") or SimpleNamespace(vae={}),
    )
    monkeypatch.setattr(pipeline_module, "load_release_vae", lambda *args: events.append("vae"))
    monkeypatch.setattr(
        pipeline_module.HunyuanImage3PromptTokenizer, "from_pretrained", classmethod(lambda *a: events.append("tok"))
    )
    pipe = HunyuanImage3Pipeline.from_transformer(_ConstantTransformer(), str(tmp_path))
    assert events == ["enter", "config", "vae", "tok", "exit"] and pipe.flow_shift == 3.0

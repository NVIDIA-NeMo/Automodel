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

"""Every registered model class keeps the LM-head interface that FP32 logits rely on.

``logits_dtype: float32`` replaces the head that ``get_output_embeddings()`` returns
through ``set_output_embeddings()``. Each registered class with a vocabulary head is
built on the meta device and must expose a head that ``Float32LogitsLinear.from_head``
converts, owning the same weight, and a setter that installs it.
"""

import importlib
import sys
import typing
from collections.abc import Callable
from dataclasses import asdict

import pytest
import torch
from torch import nn
from transformers import Glm4MoeLiteConfig, MiniMaxM2Config, Mistral3Config, NemotronHConfig, PretrainedConfig

from nemo_automodel._transformers.infrastructure import _use_fp32_logits
from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_REGISTRATIONS, MODEL_ARCH_MAPPING
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_mt2.config import HyMT2Config
from nemo_automodel.shared.model_utils import Float32LogitsLinear

# Registered classes without a vocabulary head: embedding, decision and policy models.
_NO_LM_HEAD = {
    "D1OmniModel",
    "LlamaBidirectionalForSequenceClassification",
    "LlamaBidirectionalModel",
    "LlamaNemotronVLModel",
    "Ministral3BidirectionalModel",
    "Mistral3BidirectionalModel",
    "Mistral3VLBidirectionalForSequenceClassification",
    "PraxisPolicy",
}

# Construction needs an optional module.
_NEEDS_MODULE = {
    "BeamForCausalLM": "transformers.models.beam",
    "KimiK31ForConditionalGeneration": "fla",
    "KimiK3ForCausalLM": "fla",
    "KimiK3ForConditionalGeneration": "fla",
    "NemotronParseForConditionalGeneration": "timm",
}

# Construction needs the checkpoint's remote-code configuration (vision and audio encoders).
_NEEDS_CHECKPOINT = {"NemotronOmniForConditionalGeneration"}


def _helper(path: str) -> Callable[..., PretrainedConfig]:
    """Import a config helper from a model's own tests, named ``module:function``."""
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def _mistral_large4_vlm_config() -> PretrainedConfig:
    from nemo_automodel.components.models.mistral_large4.vision import MistralLarge4VisionConfig

    vision = MistralLarge4VisionConfig(
        hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, patch_size=2, image_size=16
    )
    return _helper("tests.unit_tests.models.mistral_large4.math_reference:tiny_config")(vision_config=asdict(vision))


def _registry_config(model_type: str, **kwargs: object) -> PretrainedConfig:
    module, name = _CUSTOM_CONFIG_REGISTRATIONS[model_type]
    return getattr(importlib.import_module(module), name)(**kwargs)


# Classes whose default config does not build. Prefer the tiny config of the model's own tests.
_CONFIGS: dict[str, Callable[[], PretrainedConfig]] = {
    "BagelForUnifiedMultimodal": lambda: _registry_config("bagel", visual_und=False, visual_gen=False),
    "DeepseekV41ForCausalLM": lambda: _helper("tests.unit_tests.models.deepseek_v41.test_model:_tiny_config")(),
    "DiffusionGemmaForBlockDiffusion": lambda: importlib.import_module(
        "transformers.models.diffusion_gemma.configuration_diffusion_gemma"
    ).DiffusionGemmaConfig(
        text_config=_helper(
            "tests.unit_tests.models.diffusion_gemma.test_diffusion_gemma_parity:_tiny_text_config_dict"
        )(),
        vision_config=None,
        canvas_length=4,
    ),
    "Gemma4DrafterForCausalLM": lambda: importlib.import_module(
        "transformers.models.gemma4_assistant.configuration_gemma4_assistant"
    ).Gemma4AssistantConfig(
        text_config=_helper(
            "tests.unit_tests.models.gemma4_drafter.test_drafter_wrapper:_make_tiny_drafter_text_config"
        )(),
        backbone_hidden_size=32,
    ),
    "Glm4MoeLiteForCausalLM": Glm4MoeLiteConfig,
    "HYV3ForCausalLM": lambda: _registry_config("hy_v3"),
    "HyMT2ForCausalLM": HyMT2Config,
    "KimiLinear48BForCausalLM": lambda: _helper("tests.unit_tests.models.kimi_linear.test_model:_tiny_kimi_config")(),
    "KimiVLForConditionalGeneration": lambda: _registry_config("kimi_vl"),
    "MiniMaxM2ForCausalLM": MiniMaxM2Config,
    "MiniMaxM3SparseForConditionalGeneration": lambda: _registry_config("minimax_m3_vl"),
    "Mistral3ForConditionalGeneration": lambda: Mistral3Config(text_config=_registry_config("mistral4").to_dict()),
    "Mistral4ForCausalLM": lambda: _registry_config("mistral4"),
    "MistralLarge4ForConditionalGeneration": _mistral_large4_vlm_config,
    "NemotronHForCausalLM": NemotronHConfig,
    "NemotronParseForConditionalGeneration": lambda: importlib.import_module(
        "nemo_automodel.components.models.nemotron_parse.model"
    ).NemotronParseConfig(
        encoder={"patch_size": 16, "max_resolution": 64},
        decoder={"vocab_size": 64, "d_model": 32, "encoder_attention_heads": 4, "decoder_attention_heads": 4},
        max_sequence_length=32,
    ),
    "Qwen3_8_FlashNextForConditionalGeneration": lambda: _helper(
        "tests.unit_tests.models.qwen3_8_flash_next.test_qsa_route_replay:_tiny_config"
    )(),
    "Step3p5ForCausalLM": lambda: _registry_config("step3p5", layer_types=["full_attention"] * 2, num_hidden_layers=2),
    "Step3p7ForConditionalGeneration": lambda: _helper("tests.unit_tests.models.step3p7.test_model:small_config")(),
}

# Constructor arguments for classes whose defaults need a checkpoint.
_BUILD_KWARGS: dict[str, Callable[[], dict[str, object]]] = {
    # The HF state-dict adapter expects the released checkpoint's single PLE layer.
    "Qwen3_8_FlashNextForConditionalGeneration": lambda: {"backend": BackendConfig(enable_hf_state_dict_adapter=False)},
}


def _model_class(name: str) -> type[nn.Module]:
    module = next(entry[0] for entry in MODEL_ARCH_MAPPING.values() if entry[1] == name)
    return getattr(importlib.import_module(module), name)


def _default_config(model_cls: type[nn.Module]) -> PretrainedConfig:
    """Build the config class a model declares, or the one its constructor annotates."""
    config_cls = getattr(model_cls, "config_class", None)
    if config_cls is None or config_cls is PretrainedConfig:
        hint = typing.get_type_hints(model_cls.__init__, globalns=vars(sys.modules[model_cls.__module__]))["config"]
        config_cls = next(arg for arg in (hint, *typing.get_args(hint)) if isinstance(arg, type))
    return config_cls()


_LM_CLASSES = sorted({entry[1] for entry in MODEL_ARCH_MAPPING.values()} - _NO_LM_HEAD)


@pytest.mark.parametrize("name", _LM_CLASSES)
def test_registered_lm_head_converts_to_fp32_logits(name, monkeypatch):
    # Some constructors record the current CUDA device for their rotary embedding.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    if name in _NEEDS_MODULE:
        pytest.importorskip(_NEEDS_MODULE[name])
    if name in _NEEDS_CHECKPOINT:
        pytest.skip("construction needs the checkpoint's remote-code configuration")
    model_cls = _model_class(name)
    config = _CONFIGS[name]() if name in _CONFIGS else _default_config(model_cls)
    kwargs = _BUILD_KWARGS[name]() if name in _BUILD_KWARGS else {}
    with torch.device("meta"):
        model = (
            model_cls.from_config(config, **kwargs)
            if hasattr(model_cls, "from_config")
            else model_cls(config, **kwargs)
        )
    head = model.get_output_embeddings()

    _use_fp32_logits(model, torch.float32, pipelined=False)

    fp32_head = model.get_output_embeddings()
    assert isinstance(fp32_head, Float32LogitsLinear)
    assert fp32_head.weight is head.weight

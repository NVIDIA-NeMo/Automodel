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

"""Every registered model class with a vocabulary head takes ``logits_dtype`` and gives it to its head.

The loader passes ``model.logits_dtype`` to the constructor of repo models without checking for the
argument, so each class is built on the meta device with it.
"""

import importlib
import inspect
import sys
import typing
from collections.abc import Callable
from dataclasses import asdict

import pytest
import torch
from torch import nn
from transformers import Glm4MoeLiteConfig, MiniMaxM2Config, Mistral3Config, NemotronHConfig, PretrainedConfig

from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_REGISTRATIONS, MODEL_ARCH_MAPPING
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_mt2.config import HyMT2Config

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
}

# Construction needs the checkpoint's remote-code vision and audio encoder configurations.
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


def _stub_radio_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for NemotronParse's RADIO vision encoder, whose configuration and model are Hub remote code."""
    module = importlib.import_module("nemo_automodel.components.models.nemotron_parse.model")
    encoder_config = PretrainedConfig(patch_size=16, max_resolution=64)
    monkeypatch.setattr(module.AutoConfig, "from_pretrained", lambda *args, **kwargs: encoder_config)
    monkeypatch.setattr(module.AutoModel, "from_config", lambda config, trust_remote_code: nn.Identity())


def _nemotron_parse_config() -> PretrainedConfig:
    module = importlib.import_module("nemo_automodel.components.models.nemotron_parse.model")
    decoder = {"vocab_size": 50, "d_model": 32, "decoder_attention_heads": 4, "decoder_ffn_dim": 64}
    return module.NemotronParseConfig(
        encoder={"patch_size": 16, "max_resolution": 64}, decoder=decoder, max_sequence_length=32
    )


def _eager_vision(config: PretrainedConfig) -> PretrainedConfig:
    """Select eager vision attention: FlashAttention 2, the default, raises at construction without CUDA."""
    config.vision_config._attn_implementation = "eager"
    return config


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
    "Ernie5_5_MoeForCausalLM": lambda: _helper("tests.unit_tests.models.ernie5_5.test_model:_tiny_config")(),
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
    "KimiK31ForConditionalGeneration": lambda: _eager_vision(_registry_config("kimi_k31")),
    "KimiK3ForConditionalGeneration": lambda: _eager_vision(_registry_config("kimi_k3")),
    "KimiLinear48BForCausalLM": lambda: _helper("tests.unit_tests.models.kimi_linear.test_model:_tiny_kimi_config")(),
    "KimiVLForConditionalGeneration": lambda: _registry_config("kimi_vl"),
    "MiniMaxM2ForCausalLM": MiniMaxM2Config,
    "MiniMaxM3SparseForConditionalGeneration": lambda: _registry_config("minimax_m3_vl"),
    "Mistral3ForConditionalGeneration": lambda: Mistral3Config(text_config=_registry_config("mistral4").to_dict()),
    "Mistral4ForCausalLM": lambda: _registry_config("mistral4"),
    "MistralLarge4ForConditionalGeneration": _mistral_large4_vlm_config,
    "NemotronHForCausalLM": NemotronHConfig,
    "NemotronParseForConditionalGeneration": _nemotron_parse_config,
    "Qwen3_8_FlashNextForConditionalGeneration": lambda: _helper(
        "tests.unit_tests.models.qwen3_8_flash_next.test_qsa_route_replay:_tiny_config"
    )(),
    "Step3p5ForCausalLM": lambda: _registry_config("step3p5", layer_types=["full_attention"] * 2, num_hidden_layers=2),
    "Step3p7ForConditionalGeneration": lambda: _helper("tests.unit_tests.models.step3p7.test_model:small_config")(),
}

# Constructor arguments for classes whose defaults need a checkpoint.
_BUILD_KWARGS: dict[str, Callable[[], dict[str, object]]] = {
    # ERNIE 5.5 implements the torch kernels only, and its default backend runs the DSA layers on cuDNN
    # kernels that need BF16 and the released dimensions.
    "Ernie5_5_MoeForCausalLM": lambda: {
        "backend": BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch", rope_fusion=False)
    },
    # The HF state-dict adapter expects the released checkpoint's single PLE layer.
    "Qwen3_8_FlashNextForConditionalGeneration": lambda: {"backend": BackendConfig(enable_hf_state_dict_adapter=False)},
}


# Stubs for remote-code components, applied before the config is built.
_STUBS: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "NemotronParseForConditionalGeneration": _stub_radio_encoder,
}


def _default_config(model_cls: type[nn.Module]) -> PretrainedConfig:
    """Build the config class a model declares, or the one its constructor annotates."""
    config_cls = getattr(model_cls, "config_class", None)
    if config_cls is None or config_cls is PretrainedConfig:
        hint = typing.get_type_hints(model_cls.__init__, globalns=vars(sys.modules[model_cls.__module__]))["config"]
        config_cls = next(arg for arg in (hint, *typing.get_args(hint)) if isinstance(arg, type))
    return config_cls()


# Two modules may register classes of the same name, so each case is a (module, class name) pair.
_LM_CLASSES = sorted({entry[:2] for entry in MODEL_ARCH_MAPPING.values() if entry[1] not in _NO_LM_HEAD})


@pytest.mark.parametrize(
    ("module", "name"), [pytest.param(*entry, id=f"{entry[0].split('.')[-2]}.{entry[1]}") for entry in _LM_CLASSES]
)
def test_registered_lm_class_takes_logits_dtype(module, name, monkeypatch):
    # Some constructors record the current CUDA device for their rotary embedding.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    if name in _NEEDS_MODULE:
        pytest.importorskip(_NEEDS_MODULE[name])
    if name in _NEEDS_CHECKPOINT:
        pytest.skip("construction needs the checkpoint's remote-code configuration")
    if name in _STUBS:
        _STUBS[name](monkeypatch)
    model_cls = getattr(importlib.import_module(module), name)
    config = _CONFIGS[name]() if name in _CONFIGS else _default_config(model_cls)
    kwargs = _BUILD_KWARGS[name]() if name in _BUILD_KWARGS else {}
    with torch.device("meta"):
        # The loader passes logits_dtype to the constructor. FP64 is no model's default,
        # so the head reports it only if the argument reached it.
        model = model_cls(config, logits_dtype=torch.float64, **kwargs)

    assert model.get_output_embeddings().logits_dtype == torch.float64


@pytest.mark.parametrize(
    ("module", "name"), [pytest.param(*entry, id=f"{entry[0].split('.')[-2]}.{entry[1]}") for entry in _LM_CLASSES]
)
def test_registered_lm_class_can_skip_its_logits(module, name):
    # Fused-linear losses call every forward with compute_logits=False, which spares the head's FSDP all-gather
    # where the forward is the repo's own. A class that inherits a Hugging Face forward wraps it with_compute_logits.
    if name in _NEEDS_MODULE:
        pytest.importorskip(_NEEDS_MODULE[name])
    forward = inspect.signature(getattr(importlib.import_module(module), name).forward).parameters
    if "logits_to_keep" not in forward:
        pytest.skip("the recipe uses MaskedCrossEntropy for models without logits_to_keep")
    assert forward["compute_logits"].default is True

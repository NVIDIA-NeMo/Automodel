# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""BAGEL import smoke test.

Only asserts that the top-level symbols are importable and are classes. It
does not instantiate the full model because BAGEL construction needs the
checkpoint-sized nested configs and optional GPU dependencies.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from transformers.configuration_utils import PretrainedConfig


def test_bagel_imports() -> None:
    from nemo_automodel.components.models.bagel import (
        BagelConfig,
        BagelForUnifiedMultimodal,
        BagelTextConfig,
        BagelVAEConfig,
    )
    from nemo_automodel.recipes.multimodal.finetune import FinetuneRecipeForMultimodal

    assert inspect.isclass(BagelConfig)
    assert inspect.isclass(BagelTextConfig)
    assert inspect.isclass(BagelVAEConfig)
    assert inspect.isclass(BagelForUnifiedMultimodal)
    assert inspect.isclass(FinetuneRecipeForMultimodal)


def test_bagel_config_uses_typed_nested_configs_and_parent_serializer(tmp_path) -> None:
    from nemo_automodel.components.models.bagel.configuration import BagelConfig, BagelTextConfig, BagelVAEConfig

    cfg = BagelConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "pad_token_id": None,
            "rope_theta": 10000.0,
        },
        vae_config={"z_channels": 16, "downsample": 8},
    )

    assert isinstance(cfg.text_config, BagelTextConfig)
    assert isinstance(cfg.llm_config, BagelTextConfig)
    assert cfg.text_config.qk_norm is True
    assert cfg.text_config.layer_module == "Qwen2DecoderLayer"
    assert cfg.text_config.freeze_und is False
    assert cfg.text_config.is_causal is True
    assert cfg.text_config.partial_rotary_factor == 1.0
    assert cfg.text_config.pad_token_id == 151643
    assert isinstance(cfg.vae_config, BagelVAEConfig)
    assert cfg.vae_config.z_channels == 16
    assert cfg.vae_config.downsample == 8
    assert BagelConfig.to_dict is PretrainedConfig.to_dict

    serialized = cfg.to_dict()

    assert serialized["text_config"]["qk_norm"] is True
    assert serialized["text_config"]["layer_module"] == "Qwen2DecoderLayer"
    assert serialized["text_config"]["freeze_und"] is False
    assert serialized["text_config"]["is_causal"] is True
    assert serialized["text_config"]["partial_rotary_factor"] == 1.0
    assert serialized["text_config"]["pad_token_id"] == 151643
    assert serialized["vae_config"]["z_channels"] == 16
    assert serialized["vae_config"]["downsample"] == 8

    cfg.save_pretrained(tmp_path)
    restored = BagelConfig.from_pretrained(tmp_path)

    assert isinstance(restored.text_config, BagelTextConfig)
    assert isinstance(restored.vae_config, BagelVAEConfig)
    assert restored.text_config.pad_token_id == 151643
    assert restored.vae_config.z_channels == 16
    assert restored.vae_config.downsample == 8


def test_bagel_config_rejects_stock_qwen2_config() -> None:
    from transformers import Qwen2Config

    from nemo_automodel.components.models.bagel.configuration import BagelConfig

    with pytest.raises(TypeError, match="BagelTextConfig"):
        BagelConfig(text_config=Qwen2Config())


def test_bagel_text_config_loads_vanilla_qwen_config(tmp_path) -> None:
    from transformers import Qwen2Config

    from nemo_automodel.components.models.bagel.configuration import BagelTextConfig

    Qwen2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    ).save_pretrained(tmp_path)

    cfg = BagelTextConfig.from_pretrained(tmp_path)

    assert isinstance(cfg, BagelTextConfig)
    assert cfg.qk_norm is True
    assert cfg.layer_module == "Qwen2DecoderLayer"
    assert cfg.freeze_und is False
    assert cfg.is_causal is True
    assert cfg.partial_rotary_factor == 1.0
    assert cfg.pad_token_id == 151643


def test_bagel_stage2_config_selects_mot_decoder() -> None:
    from nemo_automodel.components.models.bagel.configuration import BagelConfig
    from nemo_automodel.components.models.bagel.model import _prepare_config_for_stage

    cfg = BagelConfig(visual_gen=False, stage=2)

    _prepare_config_for_stage(cfg)

    assert cfg.visual_gen is True
    assert cfg.text_config.layer_module == "Qwen2MoTDecoderLayer"


def test_bagel_stage1_config_drops_generation_path() -> None:
    from nemo_automodel.components.models.bagel.configuration import BagelConfig
    from nemo_automodel.components.models.bagel.model import _prepare_config_for_stage

    cfg = BagelConfig(visual_gen=True, stage=1)
    cfg.text_config.layer_module = None

    _prepare_config_for_stage(cfg)

    assert cfg.visual_gen is False
    assert cfg.text_config.layer_module == "Qwen2DecoderLayer"


def test_bagel_model_construction_does_not_mutate_input_config(monkeypatch) -> None:
    import nemo_automodel.components.models.bagel.model as bagel_model

    cfg = bagel_model.BagelConfig(
        visual_und=False,
        visual_gen=False,
        stage=2,
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )

    class _FakeBagelModel:
        def __init__(self, config, backend=None) -> None:
            self.config = config
            self.backend = backend

    monkeypatch.setattr(bagel_model, "BagelModel", _FakeBagelModel)

    model = bagel_model.BagelForUnifiedMultimodal(cfg)

    assert cfg.stage == 2
    assert cfg.visual_gen is False
    assert cfg.text_config.layer_module == "Qwen2DecoderLayer"
    assert model.config.stage == 2
    assert model.config.visual_gen is True
    assert model.config.text_config.layer_module == "Qwen2MoTDecoderLayer"


def test_bagel_rejects_tied_word_embeddings() -> None:
    from nemo_automodel.components.models.bagel.configuration import BagelConfig
    from nemo_automodel.components.models.bagel.model import BagelForUnifiedMultimodal

    # UNTIED_ONLY: the guard reads the nested text_config tie flag and raises at the
    # top of __init__, before the (checkpoint-sized) model is constructed.
    cfg = BagelConfig(
        text_config=dict(
            tie_word_embeddings=True,
            hidden_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            vocab_size=32,
            intermediate_size=32,
        )
    )
    with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=True"):
        BagelForUnifiedMultimodal(cfg)


def test_bagel_from_pretrained_passes_backend_to_model(monkeypatch, tmp_path) -> None:
    import nemo_automodel.components.models.bagel.model as bagel_model

    config = SimpleNamespace(stage=None)
    captured = {}

    monkeypatch.setattr(
        bagel_model.BagelConfig,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: config),
    )

    def _fake_init(self, cfg, backend=None):
        self.config = cfg
        captured["backend"] = backend

    monkeypatch.setattr(bagel_model.BagelForUnifiedMultimodal, "__init__", _fake_init)
    monkeypatch.setattr(
        bagel_model.BagelForUnifiedMultimodal,
        "load_state_dict",
        lambda self, state_dict, strict: ([], []),
    )

    def _fake_load_checkpoint(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(bagel_model, "load_bagel_checkpoint_state_dict", _fake_load_checkpoint)

    backend = {"linear": "te"}
    bagel_model.BagelForUnifiedMultimodal.from_pretrained(tmp_path, stage=2, backend=backend)

    assert captured["backend"] is backend
    assert captured["stage"] == 2

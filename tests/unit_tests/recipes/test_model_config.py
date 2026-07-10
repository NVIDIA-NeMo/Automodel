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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from nemo_automodel._transformers import NeMoAutoModelForImageTextToText, NeMoAutoModelForSequenceClassification
from nemo_automodel._transformers.tokenizer_config import TokenizerConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.quantization import qlora
from nemo_automodel.components.quantization.fp8 import FP8Config
from nemo_automodel.components.quantization.qlora import BitsAndBytesQuantizationConfig
from nemo_automodel.components.utils.compile_utils import CompileConfig
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.llm.kd import _build_teacher_model
from nemo_automodel.recipes.model_config import ModelConfig, ModelTargetConfig
from nemo_automodel.recipes.vlm.config import VlmInputConfig
from transformers import AutoProcessor


class _BuiltModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(2, 2)


def test_recipe_config_resolves_complete_model_build_config_without_eager_targets():
    calls = []

    def model_factory(**kwargs):
        calls.append(("model", kwargs))
        return _BuiltModel()

    def nested_config_factory(**kwargs):
        calls.append(("nested", kwargs))
        return SimpleNamespace(**kwargs)

    def qat_factory(**kwargs):
        calls.append(("qat", kwargs))
        return SimpleNamespace(**kwargs)

    raw = ConfigNode(
        {
            "seed": 123,
            "model": {
                "_target_": model_factory,
                "config": {"_target_": nested_config_factory, "hidden_size": 16},
            },
            "packed_sequence": {"packed_sequence_size": 128},
            "fp8": {"enabled": True},
            "compile": {"enabled": True},
            "quantization": {"load_in_4bit": True},
            "qat": {"enabled": True, "qat_config": {"_target_": qat_factory, "groupsize": 64}},
            "sdpa_method": ["math"],
        }
    )

    config = RecipeConfig(raw).model

    assert calls == []
    assert config.model_factory is model_factory
    assert isinstance(config.model_kwargs["config"], ModelTargetConfig)
    assert config.seed == 123
    assert config.has_packed_sequence is None
    assert config.fp8_config.enabled is True
    assert config.compile_config.enabled is True
    assert config.quantization_config.load_in_4bit is True
    assert config.qat_enabled is True
    assert config.qat_config.factory is qat_factory
    assert config.sdpa_method == ("math",)


def test_recipe_config_keeps_sequence_classification_model_unpacked():
    raw = ConfigNode(
        {
            "model": {
                "_target_": NeMoAutoModelForSequenceClassification.from_pretrained,
                "pretrained_model_name_or_path": "org/model",
            },
            "packed_sequence": {"packed_sequence_size": 128},
            "fp8": {"enabled": True},
            "compile": {"enabled": True},
            "quantization": {"load_in_4bit": True},
            "qat": {"enabled": True},
            "sdpa_method": ["math"],
        }
    )

    config = RecipeConfig(raw).model

    assert config.has_packed_sequence is None
    assert config.fp8_config is None
    assert config.compile_config.enabled is True
    assert config.quantization_config.load_in_4bit is True
    assert config.qat_enabled is False
    assert config.sdpa_method is None


def test_recipe_config_exposes_one_model_tokenizer_and_dataloader_api_for_vlm():
    raw = ConfigNode(
        {
            "model": {
                "_target_": NeMoAutoModelForImageTextToText.from_pretrained,
                "pretrained_model_name_or_path": "org/vlm",
            },
            "tokenizer": {"_target_": AutoProcessor.from_pretrained},
            "freeze_config": {"freeze_vision_tower": True, "freeze_language_model": False},
            "dataset": {"_target_": lambda: [{}]},
            "dataloader": {},
            "packed_sequence": {"pack_size": 128},
        }
    )

    config = RecipeConfig(raw)

    assert config.model.model_name == "org/vlm"
    assert config.model.factory_applies_infrastructure is True
    assert config.model.freeze_config == {"freeze_vision_tower": True, "freeze_language_model": False}
    assert config.model.has_packed_sequence is True
    assert isinstance(config.tokenizer, TokenizerConfig)
    assert config.tokenizer.factory == AutoProcessor.from_pretrained
    assert isinstance(config.dataloader, VlmInputConfig)


def test_recipe_config_deprecates_processor_key_in_favor_of_tokenizer():
    raw = ConfigNode(
        {
            "model": {
                "_target_": NeMoAutoModelForImageTextToText.from_pretrained,
                "pretrained_model_name_or_path": "org/vlm",
            },
            "processor": {"_target_": AutoProcessor.from_pretrained},
        }
    )

    with pytest.warns(DeprecationWarning, match="rename it to tokenizer"):
        config = RecipeConfig(raw).tokenizer

    assert config.factory == AutoProcessor.from_pretrained


def test_tokenizer_config_retries_known_auto_processor_layer_validation_failure():
    processor = object()
    validation_error = ValueError("num_hidden_layers must match the number of layer types")

    with (
        patch.object(AutoProcessor, "from_pretrained", side_effect=[validation_error, processor]) as factory,
        patch(
            "nemo_automodel._transformers.v4_patches.layer_types.relax_layer_types_validator",
            return_value=True,
        ) as relax,
    ):
        config = TokenizerConfig(factory=factory, kwargs={"pretrained_model_name_or_path": "org/vlm"})
        result = config.build()

    assert result is processor
    assert factory.call_count == 2
    relax.assert_called_once_with()


@pytest.mark.parametrize(("cp_size", "prepares_cp_inputs", "expected_microbatches"), [(2, True, None), (1, True, 2)])
def test_vlm_dataloader_config_owns_pipeline_media_chunking_policy(
    cp_size,
    prepares_cp_inputs,
    expected_microbatches,
):
    calls = []

    class FakePipeline:
        def __init__(self):
            model_part = SimpleNamespace()
            if prepares_cp_inputs:
                model_part.prepare_model_inputs_for_cp = lambda: None
            self.parts = [model_part]
            self.pp_batch_size = 4
            self.pp_microbatch_size = 2

    train = SimpleNamespace(
        packing=None,
        resolve_packing_attn_implementation=lambda **kwargs: None,
        build=lambda **kwargs: calls.append(kwargs) or "train",
    )
    config = VlmInputConfig(train=train, batch_size=4)
    tokenizer = object()

    with (
        patch("nemo_automodel.recipes.vlm.config.AutoPipeline", FakePipeline),
        patch("nemo_automodel.recipes.vlm.config.ScopedRNG", return_value=nullcontext()),
        patch("nemo_automodel.recipes.vlm.config.FirstRankPerNode", return_value=nullcontext()),
    ):
        result = config.build(
            model=FakePipeline(),
            tokenizer=tokenizer,
            dp_rank=0,
            dp_world_size=1,
            pp_enabled=True,
            cp_size=cp_size,
        )

    assert result.train == "train"
    assert calls[0]["tokenizer"] is tokenizer
    assert calls[0]["pp_n_microbatches"] == expected_microbatches


def test_recipe_config_resolves_teacher_without_student_precision_features():
    teacher_factory = lambda **kwargs: _BuiltModel()
    raw = ConfigNode(
        {
            "seed": 99,
            "teacher_model": {"_target_": teacher_factory},
            "packed_sequence": {"packed_sequence_size": 128},
            "fp8": {"enabled": True},
            "compile": {"enabled": True},
            "quantization": {"load_in_4bit": True},
            "qat": {"enabled": True},
            "sdpa_method": ["math"],
        }
    )

    config = RecipeConfig(raw).teacher_model

    assert config.model_factory is teacher_factory
    assert config.seed == 99
    assert config.has_packed_sequence is None
    assert config.fp8_config is None
    assert config.compile_config is None
    assert config.quantization_config is None
    assert config.qat_enabled is False
    assert config.sdpa_method is None


def test_model_config_builds_nested_values_and_preserves_state_across_builds():
    nested_calls = []
    model_calls = []
    nested = ModelTargetConfig(
        factory=lambda **kwargs: nested_calls.append(kwargs) or SimpleNamespace(**kwargs),
        kwargs={"hidden_size": 16},
    )

    def model_factory(**kwargs):
        model_calls.append(kwargs)
        model = _BuiltModel()
        model.classifier.requires_grad_(False)
        return model

    config = ModelConfig(
        model_factory=model_factory,
        model_kwargs={"config": nested},
        factory_applies_infrastructure=True,
        seed=7,
        has_packed_sequence=True,
        sdpa_method=("math",),
    )
    peft_config = object()
    distributed_setup = DistributedSetup(mesh_context=SimpleNamespace(cp_size=1))

    with patch("nemo_automodel.recipes.model_config.ScopedRNG", return_value=nullcontext()):
        first = config.build(
            peft_config=peft_config,
            distributed_setup=distributed_setup,
            unfreeze_modules=["classifier"],
        )
        second = config.build(peft_config=peft_config, distributed_setup=distributed_setup)

    assert len(nested_calls) == 2
    assert len(model_calls) == 2
    assert model_calls[0]["config"].hidden_size == 16
    assert model_calls[0]["peft_config"] is peft_config
    assert model_calls[0]["distributed_setup"] is distributed_setup
    assert model_calls[0]["has_packed_sequence"] is True
    assert model_calls[0]["sdpa_method"] == ["math"]
    assert all(parameter.requires_grad for parameter in first.classifier.parameters())
    assert all(not parameter.requires_grad for parameter in second.classifier.parameters())
    assert config.model_kwargs["config"] is nested
    assert config.sdpa_method == ("math",)


def test_model_config_builds_optional_precision_and_quantization_configs():
    calls = []
    quantization = BitsAndBytesQuantizationConfig(load_in_4bit=True)
    qat = ModelTargetConfig(factory=lambda **kwargs: SimpleNamespace(**kwargs), kwargs={"groupsize": 64})
    fp8 = FP8Config(enabled=True)
    compile_config = CompileConfig(enabled=True)
    config = ModelConfig(
        model_factory=lambda **kwargs: calls.append(kwargs) or _BuiltModel(),
        factory_applies_infrastructure=True,
        fp8_config=fp8,
        compile_config=compile_config,
        quantization_config=quantization,
        qat_enabled=True,
        qat_config=qat,
    )
    quantization_runtime = object()

    with (
        patch("nemo_automodel.recipes.model_config.ScopedRNG", return_value=nullcontext()),
        patch.object(BitsAndBytesQuantizationConfig, "build", return_value=quantization_runtime),
    ):
        config.build()

    assert calls[0]["fp8_config"].enabled is True
    assert calls[0]["fp8_config"] is not fp8
    assert calls[0]["compile_config"].enabled is True
    assert calls[0]["compile_config"] is not compile_config
    assert calls[0]["quantization_config"] is quantization_runtime
    assert calls[0]["qat_config"].groupsize == 64


def test_model_config_rejects_qat_with_peft_before_construction():
    def model_factory(**kwargs):
        raise AssertionError("model construction must not run")

    config = ModelConfig(model_factory=model_factory, factory_applies_infrastructure=True, qat_enabled=True)

    with pytest.raises(ValueError, match="QAT with PEFT"):
        config.build(peft_config=object())


def test_model_config_does_not_build_disabled_qat_target():
    qat_factory = MagicMock(return_value=object())
    model_factory = MagicMock(return_value=_BuiltModel())
    config = ModelConfig(
        model_factory=model_factory,
        factory_applies_infrastructure=True,
        qat_enabled=False,
        qat_config=ModelTargetConfig(factory=qat_factory),
    )

    with patch("nemo_automodel.recipes.model_config.ScopedRNG", return_value=nullcontext()):
        config.build()

    qat_factory.assert_not_called()
    assert "qat_config" not in model_factory.call_args.kwargs


def test_model_config_disables_runtime_rope_fusion_for_context_parallelism():
    calls = []
    backend = ModelTargetConfig(factory=BackendConfig, kwargs={"rope_fusion": True})
    config = ModelConfig(
        model_factory=lambda **kwargs: calls.append(kwargs) or _BuiltModel(),
        model_kwargs={"backend": backend},
        factory_applies_infrastructure=True,
    )
    distributed_setup = DistributedSetup(mesh_context=SimpleNamespace(cp_size=2))

    with patch("nemo_automodel.recipes.model_config.ScopedRNG", return_value=nullcontext()):
        config.build(distributed_setup=distributed_setup)

    assert calls[0]["backend"].rope_fusion is False
    assert backend.kwargs["rope_fusion"] is True


def test_bitsandbytes_quantization_config_builds_declared_4bit_settings():
    runtime_config = object()
    bitsandbytes_config = MagicMock(return_value=runtime_config)
    config = BitsAndBytesQuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_quant_storage="float16",
    )

    with (
        patch.object(qlora, "HAS_BNB", True),
        patch.object(qlora, "HAS_TRANSFORMERS", True),
        patch.object(qlora, "transformers", SimpleNamespace(BitsAndBytesConfig=bitsandbytes_config)),
    ):
        result = config.build()

    assert result is runtime_config
    bitsandbytes_config.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_quant_storage="float16",
    )


def test_kd_teacher_uses_typed_model_config_and_freezes_runtime_model():
    config = ModelConfig(
        model_factory=lambda **kwargs: _BuiltModel(),
        factory_applies_infrastructure=True,
    )
    distributed_setup = DistributedSetup(mesh_context=SimpleNamespace(cp_size=1))

    with patch("nemo_automodel.recipes.model_config.ScopedRNG", return_value=nullcontext()):
        teacher = _build_teacher_model(
            model_config=config,
            distributed_setup=distributed_setup,
            device="cpu",
        )

    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

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

"""Unified QAT configuration, independent selectors, and legacy compatibility."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import importlib.util
import sys
from unittest.mock import patch

import pytest
import torch
import yaml
from torch import nn

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.quantization.qat import QATConfig, QATRule, get_quantizer_mode
from nemo_automodel.components.quantization.weight_qat import WeightQuantizationConfig

TARGET = "nemo_automodel.components.quantization.qat.QATConfig"
RULE = "nemo_automodel.components.quantization.qat.QATRule"
WEIGHT = "nemo_automodel.components.quantization.weight_qat.WeightQuantizationConfig"


def test_numerical_plan_does_not_require_optional_torchao(monkeypatch):
    import nemo_automodel.components.quantization.qat as original

    spec = importlib.util.spec_from_file_location("qat_without_torchao", original.__file__)
    isolated = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, isolated)
    with patch("nemo_automodel.shared.import_utils.safe_import", return_value=(False, None)):
        spec.loader.exec_module(isolated)
    config = isolated.QATConfig(target_modules=("projection",), weight=WeightQuantizationConfig("mxfp4"))
    assert config.build()[0].weight.block_size == (1, 32)
    node = ConfigNode({
        "_target_": isolated.QATConfig,
        "target_modules": ["projection"],
        "weight": {"format": "mxfp4"},
    })
    assert node.instantiate() == config
    with pytest.raises(ImportError, match="TorchAO QAT is required"):
        isolated.QATConfig().create_quantizer()


@pytest.mark.parametrize("form", ["simple", "rules", "direct_rule"])
@pytest.mark.parametrize("explicit_rule", [False, True])
@pytest.mark.parametrize("explicit_weight", [False, True])
@pytest.mark.parametrize("scale_format", [None, "e8m0", "float32"])
def test_plain_and_explicit_nested_yaml_share_typed_boundary(form, explicit_rule, explicit_weight, scale_format):
    weight = {"format": "fp8", "block_size": [32, 32]} if scale_format == "float32" else {
        "format": "mxfp4", "block_size": [1, 32]
    }
    if scale_format is not None:
        weight["scale_format"] = scale_format
    expected_weight = WeightQuantizationConfig(**weight)
    if explicit_weight:
        weight["_target_"] = WEIGHT
    rule = {"target_modules": ["*.q_proj"], "weight": weight}
    if form == "rules" and explicit_rule:
        rule["_target_"] = RULE
    raw = {"_target_": RULE if form == "direct_rule" else TARGET, **({"rules": [rule]} if form == "rules" else rule)}
    snapshot = deepcopy(raw)
    node = ConfigNode(raw)
    config = node.instantiate()
    plan = (config,) if form == "direct_rule" else config.build()
    assert plan == (QATRule(("*.q_proj",), expected_weight),)
    assert asdict(plan[0].weight)["scale_format"] == (scale_format or "e8m0")
    assert isinstance(plan, tuple)
    assert isinstance(plan[0].weight, WeightQuantizationConfig)
    assert config == node.instantiate()
    assert raw == node.raw_config == snapshot
    for serialized in (node.raw_config, node.to_yaml_dict()):
        restored = yaml.safe_load(yaml.safe_dump(serialized))
        assert restored == snapshot
        assert ConfigNode(restored).instantiate() == config
    with pytest.raises(FrozenInstanceError):
        plan[0].weight.block_size = (32, 32)


@pytest.mark.parametrize("settings, expected", [
    ({}, {"quantizer_type": "int8_dynact_int4weight"}),
    (
        {"quantizer_type": "int8_dynact_int4weight", "groupsize": 32},
        {"quantizer_type": "int8_dynact_int4weight", "groupsize": 32},
    ),
    (
        {"quantizer_type": "int4_weight_only", "padding_allowed": True, "precision": torch.float32},
        {"quantizer_type": "int4_weight_only", "padding_allowed": True, "precision": torch.float32},
    ),
])
def test_legacy_to_dict_preserves_exact_output(settings, expected):
    config = QATConfig(**settings)
    assert config.is_weight_qat is False
    assert config.to_dict() == expected


@pytest.mark.parametrize("form", ["simple", "rules"])
def test_legacy_to_dict_rejects_weight_rules(form):
    rule = QATRule(("*.q_proj",), WeightQuantizationConfig("mxfp4"))
    config = (
        QATConfig(target_modules=rule.target_modules, weight=rule.weight)
        if form == "simple" else QATConfig(rules=(rule,))
    )
    assert config.is_weight_qat is True
    with pytest.raises(ValueError, match="legacy only; serialize via ConfigNode/shared boundary"):
        config.to_dict()
    with pytest.raises(ValueError, match="TorchAO settings only"):
        config.create_quantizer()


@pytest.mark.parametrize("form", ["simple", "rules"])
def test_weight_qat_classification_is_read_only_and_not_cached(form):
    config = QATConfig()
    assert config.is_weight_qat is False
    rule = QATRule(("projection",), WeightQuantizationConfig("mxfp4"))
    if form == "simple":
        config.target_modules, config.weight = rule.target_modules, rule.weight
    else:
        config.rules = (rule,)
    assert config.is_weight_qat is True
    assert "is_weight_qat" not in asdict(config)
    assert hasattr(config, "is_weight_qat")
    with pytest.raises(AttributeError):
        setattr(config, "is_weight_qat", False)
    assert config.is_weight_qat is True
    config.target_modules, config.weight, config.rules = None, None, None
    assert config.is_weight_qat is False


@pytest.mark.parametrize("form", ["simple", "rules"])
@pytest.mark.parametrize("format, block_size", [("fp8", (128, 128)), ("mxfp4", (1, 32))])
def test_repeated_build_preserves_rule_plan_types_and_source_config(form, format, block_size):
    rule = QATRule(("q_proj",), WeightQuantizationConfig(format))
    config = (
        QATConfig(target_modules=rule.target_modules, weight=rule.weight)
        if form == "simple" else QATConfig(rules=(rule,))
    )
    snapshot = deepcopy(config)
    first, second = config.build(), config.build()
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert first == second and first is not second
    assert first == (QATRule(rule.target_modules, WeightQuantizationConfig(format, block_size)),)
    assert isinstance(first[0], QATRule)
    assert isinstance(first[0].weight, WeightQuantizationConfig)
    assert first[0] is not second[0] and first[0].weight is not second[0].weight
    assert config == snapshot and rule.weight.block_size is None
    with pytest.raises(FrozenInstanceError):
        first[0].weight.block_size = None


def test_default_blocks_resolve_without_mutating_declarative_config():
    config = QATConfig(rules=(
        QATRule(("q_proj",), WeightQuantizationConfig("fp8")),
        QATRule(("v_proj",), WeightQuantizationConfig("mxfp4")),
    ))
    first, second = config.build(), config.build()
    assert first == second and first is not second
    assert [rule.weight.block_size for rule in first] == [(128, 128), (1, 32)]
    assert all(rule.weight.block_size is None for rule in config.rules)
    assert all(a is not b for a, b in zip(first, second))


@pytest.mark.parametrize("settings", [
    {"rules": []},
    {"rules": "projection"},
    {"rules": [{"target_modules": ["projection"]}]},
    {"target_modules": [], "weight": WeightQuantizationConfig()},
    {"target_modules": "projection", "weight": WeightQuantizationConfig()},
    {"target_modules": [1], "weight": WeightQuantizationConfig()},
    {"target_modules": ["projection"]},
    {"weight": WeightQuantizationConfig()},
    {"target_modules": ["projection"], "weight": {"format": "fp8"}},
])
def test_python_constructor_rejects_missing_empty_or_untyped_new_schema(settings):
    with pytest.raises(TypeError):
        QATConfig(**settings)


@pytest.mark.parametrize("settings", [
    {"rules": [], "target_modules": ["projection"]},
    {"rules": [], "weight": WeightQuantizationConfig()},
    {"rules": [], "quantizer_type": "int4_weight_only"},
    {"rules": [], "groupsize": 32},
    {"target_modules": ["projection"], "weight": WeightQuantizationConfig(), "quantizer_type": "int8_dynact_int4weight"},
    {"target_modules": ["projection"], "weight": WeightQuantizationConfig(), "groupsize": 32},
])
def test_legacy_and_new_forms_cannot_silently_override_each_other(settings):
    with pytest.raises(ValueError, match="mutually exclusive"):
        QATConfig(**settings)


@pytest.mark.parametrize("field,value", [
    ("weight", {"format": "int4"}),
    ("weight", {"format": "mxfp4", "block_size": [1, 64]}),
    ("weight", {"format": "fp8", "unexpected": True}),
    ("weight", {"format": "fp8", "scale_format": "fp32"}),
    ("weight", {"format": "mxfp4", "scale_format": "float32"}),
    ("target_modules", []),
])
def test_plain_yaml_invalid_fields_fail_at_instantiation(field, value):
    raw = {"_target_": TARGET, "target_modules": ["projection"], "weight": {"format": "mxfp4"}}
    raw[field] = value
    with pytest.raises((TypeError, ValueError)):
        ConfigNode(raw).instantiate()


@pytest.mark.parametrize("kind,mode", [
    ("int8_dynact_int4weight", "8da4w-qat"),
    ("int4_weight_only", "4w-qat"),
])
@pytest.mark.parametrize("entry", ["build", "create_quantizer"])
def test_real_torchao_constructor_and_legacy_yaml_keep_defaults(kind, mode, entry):
    config = ConfigNode({"_target_": TARGET, "quantizer_type": kind, "groupsize": 32}).instantiate()
    assert config.is_weight_qat is False
    quantizer = getattr(config, entry)()
    assert get_quantizer_mode(quantizer) == mode
    assert quantizer.groupsize == 32
    assert quantizer.precision == torch.bfloat16
    assert quantizer.scales_precision == torch.bfloat16
    assert config.quantizer_kwargs == {"groupsize": 32}


def test_legacy_default_precision_overrides_and_unknown_type():
    config = QATConfig()
    assert config.quantizer_type == "int8_dynact_int4weight"
    assert get_quantizer_mode(config.build()) == "8da4w-qat"
    custom = QATConfig("int4_weight_only", groupsize=32, precision=torch.float32, scales_precision=torch.float32)
    assert custom.create_quantizer().precision == torch.float32
    assert custom.build().scales_precision == torch.float32
    unknown = QATConfig("unknown")
    assert unknown.is_weight_qat is False
    for entry in (unknown.build, unknown.create_quantizer):
        with pytest.raises(ValueError, match="Unknown quantizer_type: unknown"):
            entry()
    with pytest.raises(ValueError, match="TorchAO settings only"):
        QATConfig(target_modules=("projection",), weight=WeightQuantizationConfig()).create_quantizer()


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_qat_selectors_do_not_inherit_or_mutate_peft_selectors(device):
    with torch.device(device):
        model = nn.ModuleDict({"q_proj": nn.Linear(32, 32), "v_proj": nn.Linear(32, 32)})
        peft = PeftConfig(target_modules=["q_proj", "v_proj"], dim=4, use_memory_efficient_lora=False)
        apply_lora_to_linear_modules(model, peft)
    config = ConfigNode({"_target_": TARGET, "target_modules": ["v_proj"], "weight": {"format": "mxfp4"}}).instantiate()
    parameters = dict(model.named_parameters())
    QAT(config).prepare(model)
    assert model["q_proj"].weight_fake_quantizer is None
    assert model["v_proj"].weight_fake_quantizer.config.format == "mxfp4"
    assert peft.target_modules == ["q_proj", "v_proj"]
    assert config.target_modules == ("v_proj",)
    assert all(p is parameters[name] for name, p in model.named_parameters())


def test_torchao_peft_rejected_before_infrastructure_mutation():
    from nemo_automodel._transformers import infrastructure

    model = nn.Sequential(nn.Linear(32, 32, dtype=torch.bfloat16))
    peft = PeftConfig(target_modules=["0"])
    with patch.object(infrastructure, "apply_lora_to_linear_modules") as apply_peft:
        with pytest.raises(ValueError, match="INT4 QAT with PEFT"):
            infrastructure._apply_peft_and_lower_precision(model, 1, None, peft, None, None, QATConfig().build())
    apply_peft.assert_not_called()
    assert type(model[0]) is nn.Linear


def test_public_bridge_cannot_bypass_legacy_peft_guard():
    model = nn.Sequential(nn.Linear(32, 32, dtype=torch.bfloat16))
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["0"], use_memory_efficient_lora=False))
    with pytest.raises(ValueError, match="INT4 QAT with PEFT"):
        QAT(QATConfig()).prepare(model)


@pytest.mark.parametrize("form", ["simple", "rules"])
def test_recipe_rejects_full_parameter_weight_qat_before_model_build(form):
    from nemo_automodel.recipes.llm import train_ft

    rule = {"target_modules": ["projection"], "weight": {"format": "fp8", "block_size": [32, 32]}}
    node = ConfigNode({"enabled": True, "qat_config": {"_target_": TARGET, **(rule if form == "simple" else {"rules": [rule]})}})
    model = ConfigNode({"_target_": train_ft.NeMoAutoModelForCausalLM.from_config})
    with patch.object(model, "instantiate") as instantiate:
        with pytest.raises(ValueError, match="Full-parameter FP8/MXFP4 QAT is unsupported"):
            train_ft.build_model(model, None, seed=42, cfg_qat=node)
    instantiate.assert_not_called()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("architecture", ["deepseek_v4", "glm5_next"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("qat_enabled", [False, True])
def test_tiny_models_through_high_level_infrastructure(architecture, dtype, qat_enabled):
    """Real CPU construction/prepare/update, not distributed or pretrained parity.

    V4 traverses the public recipe/AutoModel route, redirecting CUDA device
    defaults to CPU. GLM starts from the production-initialized source model,
    as in the composed worker: its separate from_config initialization failure
    is outside this gate change. Neither path replaces PEFT or QAT preparation.
    Two-H100 FSDP2+EP2 numerical parity lives in the composed functional gate.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from nemo_automodel._transformers import infrastructure
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
    from nemo_automodel.components.quantization.weight_qat import WeightFakeQuantizer
    from nemo_automodel.recipes.llm import train_ft
    from tests.unit_tests._transformers.test_lora_qat_models import _tiny_model

    template = _tiny_model(architecture, dtype)
    # Checkpoint-style architecture metadata selects the registered Automodel
    # implementation; these directly constructed tiny configs omit it.
    template.config.architectures = [type(template).__name__]
    expert_path = (
        "model.language_model.layers.0.mlp.experts" if architecture == "glm5_next" else "model.layers.0.mlp.experts"
    )
    cfg_model = ConfigNode({
        "_target_": train_ft.NeMoAutoModelForCausalLM.from_config,
        "config": template.config,
        "backend": template.backend,
        "dtype": dtype,
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
    })
    weight = (
        {"format": "fp8", "block_size": [128, 128], "scale_format": "float32"}
        if architecture == "glm5_next" else {"format": "mxfp4"}
    )
    qat_node = ConfigNode({
        "enabled": qat_enabled,
        "fake_quant_after_n_steps": 0,
        "qat_config": {"_target_": TARGET, "target_modules": [expert_path], "weight": weight},
    })
    peft = PeftConfig(target_modules=[expert_path], dim=4, alpha=8, use_memory_efficient_lora=False)
    initialize = type(template).initialize_weights

    def initialize_on_cpu(model, buffer_device=None, dtype=torch.bfloat16):
        # V4's default buffer device is CUDA even on a CPU runner. Exercise
        # the original initializer with its explicit supported CPU argument.
        return initialize(model, buffer_device=torch.device("cpu"), dtype=dtype)

    # Explicit fullgraph kernels exist in these architectures; force_eager keeps
    # this CPU contract test independent of a CUDA/compiler installation.
    with (
        torch.compiler.set_stance("force_eager"),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.current_device", return_value=torch.device("cpu")),
        patch.object(type(template), "initialize_weights", initialize_on_cpu),
    ):
        if architecture == "deepseek_v4":
            model = train_ft.build_model(cfg_model, peft, seed=19, cfg_qat=qat_node)
        else:
            _, _, _, quantizer = infrastructure.instantiate_infrastructure(
                qat_config=qat_node.qat_config.instantiate() if qat_enabled else None,
            )
            model = infrastructure.apply_model_infrastructure(
                template, is_meta_device=False, device=torch.device("cpu"),
                peft_config=peft, qat_quantizer=quantizer,
            )
        # AutoModel adds its infrastructure mixin as a subclass.
        assert isinstance(model, type(template))
        assert model.backend.dispatcher == model.backend.experts == "torch"
        experts = model.get_submodule(expert_path)
        assert isinstance(experts, GroupedExpertsLoRA)
        assert not experts.use_torch_mm and not experts.use_mxfp8
        if qat_enabled:
            assert isinstance(experts.weight_fake_quantizer, WeightFakeQuantizer)
            assert experts.weight_fake_quantizer.config.format == weight["format"]
            assert experts.weight_fake_quantizer.config.scale_format == weight.get("scale_format", "e8m0")
        else:
            assert experts.weight_fake_quantizer is None
        assert all(p.dtype == dtype for p in experts.parameters())
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        assert len(trainable) == 4
        assert all(name.startswith(expert_path + ".lora_") for name in trainable)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        before = {name: p.detach().clone() for name, p in trainable.items()}
        optimizer = torch.optim.SGD(trainable.values(), lr=4.0)
        tokens = torch.tensor([[1, 5, 9, 13, 17, 21]])
        with sdpa_kernel(SDPBackend.MATH):
            logits = model(tokens).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, 64), tokens[:, 1:].reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
        assert all(model.get_parameter(name).grad is None for name in frozen)
        optimizer.step()
        assert any(not torch.equal(p, before[name]) for name, p in trainable.items())
        for name, original in frozen.items():
            torch.testing.assert_close(model.get_parameter(name), original, rtol=0, atol=0)
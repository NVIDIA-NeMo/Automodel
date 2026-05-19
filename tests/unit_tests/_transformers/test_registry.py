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

import types
from unittest.mock import patch

import pytest


def _new_registry_instance(registry_module):
    """Create a fresh registry with an empty auto_map for testing."""
    from nemo_automodel._transformers.registry import _LazyArchMapping

    mapping = _LazyArchMapping(auto_map={})
    return registry_module._ModelRegistry(model_arch_name_to_cls=mapping)


def test_register_single_class():
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class FakeModelA:
        pass

    inst.register("FakeModelA", FakeModelA)

    assert "FakeModelA" in inst.model_arch_name_to_cls
    assert inst.model_arch_name_to_cls["FakeModelA"] is FakeModelA


def test_register_multiple_classes():
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class FakeModelB:
        pass

    class FakeModelC:
        pass

    inst.register("FakeModelB", FakeModelB)
    inst.register("FakeModelC", FakeModelC)

    assert "FakeModelB" in inst.model_arch_name_to_cls
    assert "FakeModelC" in inst.model_arch_name_to_cls
    assert inst.model_arch_name_to_cls["FakeModelB"] is FakeModelB
    assert inst.model_arch_name_to_cls["FakeModelC"] is FakeModelC


def test_duplicate_register_raises():
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class DupClass:
        pass

    inst.register("DupClass", DupClass)

    with pytest.raises(ValueError, match="Duplicated model implementation"):
        inst.register("DupClass", DupClass)


def test_duplicate_register_exist_ok():
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class OrigClass:
        pass

    class ReplacementClass:
        pass

    inst.register("MyArch", OrigClass)
    inst.register("MyArch", ReplacementClass, exist_ok=True)

    assert inst.model_arch_name_to_cls["MyArch"] is ReplacementClass


def test_supported_models_and_getter():
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class A:
        pass

    inst.register("A", A)

    assert "A" in inst.supported_models
    assert inst.get_model_cls_from_model_arch("A") is A


def test_get_registry_is_cached():
    from nemo_automodel._transformers import registry as reg

    reg.get_registry.cache_clear()
    r1 = reg.get_registry()
    r2 = reg.get_registry()
    assert r1 is r2


def test_registry_reexports_split_modules():
    """The registry package remains the public import path while definitions live in split modules."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec
    from nemo_automodel._transformers.registry.model_registry import (
        _CUSTOM_CONFIG_SPECS,
        MODEL_ARCH_MAPPING,
        MODEL_PACKAGE_SPECS,
    )

    assert reg.ModelPackageSpec is ModelPackageSpec
    assert reg.MODEL_ARCH_MAPPING is MODEL_ARCH_MAPPING
    assert reg.MODEL_PACKAGE_SPECS is MODEL_PACKAGE_SPECS
    assert reg._CUSTOM_CONFIG_SPECS is _CUSTOM_CONFIG_SPECS


def test_lazy_arch_mapping_auto_map():
    """Static auto_map entries are lazily loaded on first access."""
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeClass:
        pass

    fake_module = types.SimpleNamespace(FakeClass=FakeClass)
    spec = ModelPackageSpec(package="fake", model_module="module", class_name="FakeClass")
    mapping = _LazyArchMapping({"FakeArch": spec})

    mapping._modules["fake.module"] = fake_module

    assert "FakeArch" in mapping
    assert mapping["FakeArch"] is FakeClass
    assert "FakeArch" in mapping._loaded

    with pytest.raises(KeyError):
        mapping["NonExistent"]


def test_lazy_arch_mapping_accepts_model_package_spec_entries():
    """Static auto_map entries may be declared as ModelPackageSpec metadata."""
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeClass:
        pass

    fake_module = types.SimpleNamespace(FakeClass=FakeClass)
    spec = ModelPackageSpec.from_module_path(
        "fake.package.model",
        "FakeClass",
        model_types=("fake",),
        optional_modules=frozenset({"patches"}),
    )
    mapping = _LazyArchMapping({"FakeArch": spec})
    mapping._modules["fake.package.model"] = fake_module

    assert mapping["FakeArch"] is FakeClass
    assert mapping.get_spec("FakeArch").package == "fake.package"
    assert mapping.get_spec("FakeArch").model_types == ("fake",)


def test_lazy_arch_mapping_extra_overrides_auto_map():
    """Dynamically registered entries take precedence over static entries."""
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class StaticClass:
        pass

    class DynamicClass:
        pass

    fake_module = types.SimpleNamespace(StaticClass=StaticClass)
    spec = ModelPackageSpec(package="fake", model_module="module", class_name="StaticClass")
    mapping = _LazyArchMapping({"MyArch": spec})
    mapping._modules["fake.module"] = fake_module

    assert mapping["MyArch"] is StaticClass

    mapping["MyArch"] = DynamicClass
    assert mapping["MyArch"] is DynamicClass


def test_lazy_arch_mapping_unavailable_model():
    """Auto_map entries whose imports fail are removed and excluded from containment."""
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    spec = ModelPackageSpec(package="nonexistent.module", model_module="path", class_name="BadClass")
    mapping = _LazyArchMapping({"BadArch": spec})

    assert "BadArch" not in mapping
    assert mapping.get_spec("BadArch") is None


def test_default_registry_has_static_entries():
    """The default registry is populated from MODEL_ARCH_MAPPING."""
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING, _ModelRegistry

    inst = _ModelRegistry()
    for arch_name in MODEL_ARCH_MAPPING:
        assert arch_name in inst.model_arch_name_to_cls.keys()


def test_registry_imports_optional_module_for_architecture():
    """Convention modules are imported through ModelPackageSpec package metadata."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    fake_module = types.SimpleNamespace(__name__="fake.package.patches")
    spec = ModelPackageSpec(
        package="fake.package",
        architectures=("FakeArch",),
        optional_modules=frozenset({"patches"}),
    )
    inst = reg._ModelRegistry(model_arch_name_to_cls=_LazyArchMapping(auto_map={}), package_specs=(spec,))

    with patch(
        "nemo_automodel._transformers.registry.base.importlib.import_module", return_value=fake_module
    ) as mock_import:
        modules = inst.iter_optional_modules_for_architectures(("FakeArch",), "patches")

    mock_import.assert_called_once_with("fake.package.patches")
    assert modules == (fake_module,)


def test_registry_filters_optional_modules_by_scope():
    """Global patch lookup imports only specs that advertise global patches."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    global_module = types.SimpleNamespace(__name__="fake.global_model.patches")
    specs = (
        ModelPackageSpec(
            package="fake.global_model",
            optional_modules=frozenset({"patches"}),
            global_patches=True,
        ),
        ModelPackageSpec(
            package="fake.runtime_model",
            optional_modules=frozenset({"patches"}),
            global_patches=False,
        ),
    )
    inst = reg._ModelRegistry(model_arch_name_to_cls=_LazyArchMapping(auto_map={}), package_specs=specs)

    with patch(
        "nemo_automodel._transformers.registry.base.importlib.import_module", return_value=global_module
    ) as mock_import:
        modules = inst.iter_optional_modules("patches", global_patches=True)

    mock_import.assert_called_once_with("fake.global_model.patches")
    assert modules == (global_module,)


def test_registry_discovers_config_class_from_config_module():
    """Config classes are imported from declared convention modules on demand."""
    from transformers import PretrainedConfig

    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeConfig(PretrainedConfig):
        model_type = "fake_model"

    FakeConfig.__module__ = "fake.package.config"
    fake_module = types.SimpleNamespace(__name__="fake.package.config", FakeConfig=FakeConfig)
    spec = ModelPackageSpec(package="fake.package", model_types=("fake_model",), config_module="config")
    inst = reg._ModelRegistry(model_arch_name_to_cls=_LazyArchMapping(auto_map={}), package_specs=(spec,))

    with (
        patch("transformers.AutoConfig.register") as mock_register,
        patch("nemo_automodel._transformers.registry.base.importlib.import_module", return_value=fake_module),
    ):
        assert inst.ensure_config_registered("fake_model") is True

    mock_register.assert_called_once_with("fake_model", FakeConfig)


def test_registry_registers_explicit_config_alias():
    """Explicit config metadata supports aliases whose key differs from class.model_type."""
    from transformers import PretrainedConfig

    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeConfig(PretrainedConfig):
        model_type = "canonical_model"

    fake_module = types.SimpleNamespace(__name__="fake.package.model", FakeConfig=FakeConfig)
    spec = ModelPackageSpec(
        package="fake.package",
        model_types=("canonical_model", "alias_model"),
        config_module="model",
        config_class_name="FakeConfig",
    )
    inst = reg._ModelRegistry(model_arch_name_to_cls=_LazyArchMapping(auto_map={}), package_specs=(spec,))

    with (
        patch("transformers.AutoConfig.register") as mock_register,
        patch("nemo_automodel._transformers.registry.base.importlib.import_module", return_value=fake_module),
    ):
        assert inst.ensure_config_registered("alias_model") is True

    mock_register.assert_called_once_with("alias_model", FakeConfig)


def test_registry_registers_config_alias_when_auto_config_rejects_mismatch():
    """HF validates ``config.model_type``; aliases fall back to CONFIG_MAPPING directly."""
    from transformers import PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry import _LazyArchMapping
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeConfig(PretrainedConfig):
        model_type = "canonical_model"

    fake_module = types.SimpleNamespace(__name__="fake.package.model", FakeConfig=FakeConfig)
    spec = ModelPackageSpec(
        package="fake.package",
        model_types=("alias_model_with_mismatch",),
        config_module="model",
        config_class_name="FakeConfig",
    )
    inst = reg._ModelRegistry(model_arch_name_to_cls=_LazyArchMapping(auto_map={}), package_specs=(spec,))

    with (
        patch("transformers.AutoConfig.register", side_effect=ValueError("model_type mismatch")),
        patch.object(CONFIG_MAPPING, "register") as mock_mapping_register,
        patch("nemo_automodel._transformers.registry.base.importlib.import_module", return_value=fake_module),
    ):
        assert inst.ensure_config_registered("alias_model_with_mismatch") is True

    mock_mapping_register.assert_called_once_with("alias_model_with_mismatch", FakeConfig)


def test_resolve_custom_model_cls_found():
    """resolve_custom_model_cls returns the class when it exists and has no supports_config."""
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class PlainModel:
        pass

    inst.register("PlainModel", PlainModel)
    assert inst.resolve_custom_model_cls("PlainModel", object()) is PlainModel


def test_resolve_custom_model_cls_not_found():
    """resolve_custom_model_cls returns None for unregistered architectures."""
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)
    assert inst.resolve_custom_model_cls("NonExistent", object()) is None


def test_resolve_custom_model_cls_supports_config_true():
    """resolve_custom_model_cls returns the class when supports_config returns True."""
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class SupportedModel:
        @classmethod
        def supports_config(cls, config):
            return True

    inst.register("SupportedModel", SupportedModel)
    assert inst.resolve_custom_model_cls("SupportedModel", object()) is SupportedModel


def test_resolve_custom_model_cls_supports_config_false():
    """resolve_custom_model_cls returns None when supports_config returns False."""
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class UnsupportedModel:
        @classmethod
        def supports_config(cls, config):
            return False

    inst.register("UnsupportedModel", UnsupportedModel)
    assert inst.resolve_custom_model_cls("UnsupportedModel", object()) is None


def test_resolve_custom_model_cls_passes_config_to_supports():
    """resolve_custom_model_cls passes the config to supports_config for inspection."""
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class ConfigAwareModel:
        @classmethod
        def supports_config(cls, config):
            return getattr(config, "ok", False)

    inst.register("ConfigAwareModel", ConfigAwareModel)

    good = types.SimpleNamespace(ok=True)
    bad = types.SimpleNamespace(ok=False)
    assert inst.resolve_custom_model_cls("ConfigAwareModel", good) is ConfigAwareModel
    assert inst.resolve_custom_model_cls("ConfigAwareModel", bad) is None


def test_custom_config_specs_are_metadata_only_until_requested():
    """Custom config entries should not require eager registration at registry import time."""
    from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_SPECS

    for spec in _CUSTOM_CONFIG_SPECS:
        assert spec.config_module_path
        assert spec.config_class_name


def test_kimi_k25_arch_alias_in_model_arch_mapping():
    """KimiK25ForConditionalGeneration (checkpoint arch) must map to KimiK25VLForConditionalGeneration."""
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    assert "KimiK25ForConditionalGeneration" in MODEL_ARCH_MAPPING, (
        "KimiK25ForConditionalGeneration missing from MODEL_ARCH_MAPPING. "
        "Kimi-K2.5 checkpoints use this architecture name and need it mapped "
        "to KimiK25VLForConditionalGeneration."
    )
    spec = MODEL_ARCH_MAPPING["KimiK25ForConditionalGeneration"]
    assert spec.class_name == "KimiK25VLForConditionalGeneration"


def test_deepseek_v4_registered_in_arch_mapping():
    """DeepseekV4ForCausalLM must be registered in MODEL_ARCH_MAPPING."""
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    assert "DeepseekV4ForCausalLM" in MODEL_ARCH_MAPPING, (
        "DeepseekV4ForCausalLM missing from MODEL_ARCH_MAPPING. "
        "DSV4 checkpoints declare this architecture and need it routed to the "
        "in-tree model implementation."
    )
    spec = MODEL_ARCH_MAPPING["DeepseekV4ForCausalLM"]
    assert spec.module_path == "nemo_automodel.components.models.deepseek_v4.model"
    assert spec.class_name == "DeepseekV4ForCausalLM"


def test_deepseek_v4_in_custom_config_specs():
    """deepseek_v4 model_type must be declared in _CUSTOM_CONFIG_SPECS."""
    from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_SPECS

    specs_by_model_type = {model_type: spec for spec in _CUSTOM_CONFIG_SPECS for model_type in spec.model_types}
    assert "deepseek_v4" in specs_by_model_type, (
        "deepseek_v4 must be in _CUSTOM_CONFIG_SPECS so ModelRegistry can register "
        "DSV4 configs on demand before AutoConfig.from_pretrained runs."
    )
    spec = specs_by_model_type["deepseek_v4"]
    assert spec.config_module_path == "nemo_automodel.components.models.deepseek_v4.config"
    assert spec.config_class_name == "DeepseekV4Config"


def test_all_model_folders_registered_in_auto_map():
    """Every model folder with a model.py must have at least one entry in MODEL_ARCH_MAPPING.

    This catches the case where a developer adds a new model directory under
    ``nemo_automodel/components/models/`` but forgets to add it to the static
    ``MODEL_ARCH_MAPPING`` in ``registry.py``.
    """
    import pathlib

    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    models_root = pathlib.Path(__file__).resolve().parents[3] / "nemo_automodel" / "components" / "models"

    # Collect the set of module paths referenced by the auto_map
    registered_module_paths = {spec.module_path for spec in MODEL_ARCH_MAPPING.values()}

    missing = []
    documentation_only_model_dirs = {"blueprint"}
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith(("_", ".")):
            continue
        if model_dir.name in documentation_only_model_dirs:
            continue
        model_file = model_dir / "model.py"
        if not model_file.exists():
            continue
        expected_module = f"nemo_automodel.components.models.{model_dir.name}.model"
        if expected_module not in registered_module_paths:
            missing.append(model_dir.name)

    assert not missing, (
        f"Model folder(s) {missing} contain a model.py but are not registered "
        f"in MODEL_ARCH_MAPPING (registry.py). Add an entry for each architecture "
        f"exported by these modules."
    )

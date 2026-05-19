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
    return registry_module._BaseModelRegistry(model_specs=())


def _model_arch_lookup(model_arch_mapping):
    """Return the architecture-keyed lookup derived from registry package specs."""
    from nemo_automodel._transformers.registry.base import _normalize_model_arch_mapping

    return _normalize_model_arch_mapping(model_arch_mapping)


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


def test_make_registry_is_cached():
    from nemo_automodel._transformers import registry as reg

    reg.make_registry.cache_clear()
    r1 = reg.make_registry()
    r2 = reg.make_registry()
    assert r1 is r2


def test_make_retrieval_registry_is_cached():
    from nemo_automodel._transformers import registry as reg

    reg.make_retrieval_registry.cache_clear()
    r1 = reg.make_retrieval_registry()
    r2 = reg.make_retrieval_registry()
    assert r1 is r2


def test_registry_reexports_split_modules():
    """The registry package remains the public import path while definitions live in split modules."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec
    from nemo_automodel._transformers.registry.model_registry import (
        MODEL_PACKAGE_SPECS,
        RETRIEVAL_MODEL_PACKAGE_SPECS,
        RetrievalModelRegistry,
        make_registry,
        make_retrieval_registry,
    )

    assert reg.ModelPackageSpec is ModelPackageSpec
    assert reg.MODEL_PACKAGE_SPECS is MODEL_PACKAGE_SPECS
    assert reg.RETRIEVAL_MODEL_PACKAGE_SPECS is RETRIEVAL_MODEL_PACKAGE_SPECS
    assert reg.RetrievalModelRegistry is RetrievalModelRegistry
    assert reg.make_registry is make_registry
    assert reg.make_retrieval_registry is make_retrieval_registry


def test_registry_mapping_loads_model_class_on_demand():
    """Static auto_map entries are lazily loaded on first access."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeClass:
        pass

    fake_module = types.SimpleNamespace(FakeClass=FakeClass)
    spec = ModelPackageSpec(package="fake", model_module="module", class_name="FakeClass", architectures=("FakeArch",))
    inst = reg._BaseModelRegistry(model_specs=(spec,))

    with patch("nemo_automodel._transformers.registry.base.importlib.import_module", return_value=fake_module):
        assert "FakeArch" in inst.model_arch_name_to_cls
    assert inst.model_arch_name_to_cls["FakeArch"] is FakeClass
    assert "FakeArch" in inst._loaded_model_classes

    with pytest.raises(KeyError):
        inst.model_arch_name_to_cls["NonExistent"]


def test_registry_accepts_model_package_spec_entries():
    """Static auto_map entries may be declared as ModelPackageSpec metadata."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeClass:
        pass

    fake_module = types.SimpleNamespace(FakeClass=FakeClass)
    spec = ModelPackageSpec.from_module_path(
        "fake.package.model",
        "FakeClass",
        architectures=["FakeArch"],
        model_types=("fake",),
    )
    inst = reg._BaseModelRegistry(model_specs={"FakeArch": spec})

    with patch("nemo_automodel._transformers.registry.base.importlib.import_module", return_value=fake_module):
        assert inst.model_arch_name_to_cls["FakeArch"] is FakeClass
    assert inst.model_arch_name_to_cls.keys() == {"FakeArch"}


def test_model_package_spec_from_model_class():
    """ModelPackageSpec can be derived directly from a model class object."""
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeClass:
        pass

    FakeClass.__module__ = "fake.package.model"

    spec = ModelPackageSpec.from_model_class(FakeClass, architectures=["FakeArch"])

    assert spec.module_path == "fake.package.model"
    assert spec.class_name == "FakeClass"
    assert spec.architectures == ("FakeArch",)


def test_registry_derives_keys_from_spec_architectures():
    """Tuple-based auto_map entries derive lookup keys from spec.architectures."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    spec = ModelPackageSpec(
        package="fake.package",
        class_name="FakeClass",
        architectures=("FakeArch", "FakeAlias"),
    )
    inst = reg._BaseModelRegistry(model_specs=(spec,))

    assert inst.model_arch_name_to_cls.keys() == {"FakeArch", "FakeAlias"}


def test_base_registry_owns_architecture_metadata():
    """Package metadata indexes live on the registry, not on the lazy class loader."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    spec = ModelPackageSpec(
        package="fake.package",
        class_name="FakeClass",
        architectures=("FakeArch", "FakeAlias"),
        model_types=("fake",),
    )
    inst = reg._BaseModelRegistry(model_specs=(spec,))

    assert inst.get_model_package_spec("FakeArch") is spec
    assert inst.get_model_package_spec("FakeAlias") is spec
    assert inst.get_model_package_specs_for_model_type("fake") == (spec,)


def test_registry_rejects_duplicate_architectures():
    """Duplicate architecture names across package specs fail during registry construction."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    specs = (
        ModelPackageSpec(package="fake.package_a", class_name="FakeClassA", architectures=("FakeArch",)),
        ModelPackageSpec(package="fake.package_b", class_name="FakeClassB", architectures=("FakeArch",)),
    )

    with pytest.raises(ValueError, match="Duplicated model architecture entry for 'FakeArch'"):
        reg._BaseModelRegistry(model_specs=specs)


def test_registry_dynamic_entry_overrides_static_entry():
    """Dynamically registered entries take precedence over static entries."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class StaticClass:
        pass

    class DynamicClass:
        pass

    fake_module = types.SimpleNamespace(StaticClass=StaticClass)
    spec = ModelPackageSpec(package="fake", model_module="module", class_name="StaticClass", architectures=("MyArch",))
    inst = reg._BaseModelRegistry(model_specs=(spec,))

    with patch("nemo_automodel._transformers.registry.base.importlib.import_module", return_value=fake_module):
        assert inst.model_arch_name_to_cls["MyArch"] is StaticClass

    inst.model_arch_name_to_cls["MyArch"] = DynamicClass
    assert inst.model_arch_name_to_cls["MyArch"] is DynamicClass
    assert inst.get_model_package_spec("MyArch").class_name == "DynamicClass"


def test_registry_unavailable_model():
    """Auto_map entries whose imports fail are removed and excluded from containment."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    spec = ModelPackageSpec(
        package="nonexistent.module",
        model_module="path",
        class_name="BadClass",
        architectures=("BadArch",),
    )
    inst = reg._BaseModelRegistry(model_specs=(spec,))

    assert "BadArch" not in inst.model_arch_name_to_cls
    assert "BadArch" not in inst.model_arch_name_to_cls.keys()


def test_registry_discards_failed_import_from_metadata_index():
    """Failed model imports remove the architecture from the registry-owned metadata index."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    spec = ModelPackageSpec(
        package="nonexistent.module",
        model_module="path",
        class_name="BadClass",
        architectures=("BadArch",),
    )
    inst = reg._BaseModelRegistry(model_specs=(spec,))

    assert "BadArch" not in inst.model_arch_name_to_cls
    assert inst.get_model_package_spec("BadArch") is None
    assert "BadArch" not in inst.supported_models

    class GoodClass:
        pass

    inst.register("GoodArch", GoodClass)
    assert "BadArch" not in inst.model_arch_name_to_cls.keys()


def test_direct_lazy_registration_updates_registry_metadata():
    """Direct mapping assignment still informs the registry about dynamic model metadata."""
    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    class DirectClass:
        pass

    DirectClass.__module__ = "fake.direct.model"
    inst.model_arch_name_to_cls["DirectArch"] = DirectClass

    spec = inst.get_model_package_spec("DirectArch")
    assert spec.module_path == "fake.direct.model"
    assert spec.class_name == "DirectClass"
    assert inst.model_arch_name_to_cls["DirectArch"] is DirectClass


def test_default_registry_has_static_entries():
    """The default registry is populated from MODEL_PACKAGE_SPECS."""
    from nemo_automodel._transformers.registry import MODEL_PACKAGE_SPECS, make_registry

    inst = make_registry()
    for spec in MODEL_PACKAGE_SPECS:
        for arch_name in spec.architectures:
            assert arch_name in inst.model_arch_name_to_cls.keys()


def test_retrieval_registry_has_separate_static_entries():
    """Retrieval architectures live in the retrieval registry, not the default registry."""
    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry import MODEL_PACKAGE_SPECS, RETRIEVAL_MODEL_PACKAGE_SPECS

    retrieval_arch = "LlamaBidirectionalModel"
    default_registry = reg._BaseModelRegistry(model_specs=MODEL_PACKAGE_SPECS)
    retrieval_registry = reg._BaseModelRegistry(model_specs=RETRIEVAL_MODEL_PACKAGE_SPECS)

    assert retrieval_arch not in _model_arch_lookup(MODEL_PACKAGE_SPECS)
    assert retrieval_arch in _model_arch_lookup(RETRIEVAL_MODEL_PACKAGE_SPECS)
    assert retrieval_arch not in default_registry.model_arch_name_to_cls.keys()
    assert retrieval_arch in retrieval_registry.model_arch_name_to_cls.keys()


def test_registry_discovers_config_class_from_config_module():
    """Config classes are imported from declared convention modules on demand."""
    from transformers import PretrainedConfig

    from nemo_automodel._transformers import registry as reg
    from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

    class FakeConfig(PretrainedConfig):
        model_type = "fake_model"

    FakeConfig.__module__ = "fake.package.config"
    fake_module = types.SimpleNamespace(__name__="fake.package.config", FakeConfig=FakeConfig)
    spec = ModelPackageSpec(package="fake.package", model_types=("fake_model",), config_module="config")
    inst = reg._BaseModelRegistry(model_specs=(spec,))

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
    inst = reg._BaseModelRegistry(model_specs=(spec,))

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
    inst = reg._BaseModelRegistry(model_specs=(spec,))

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


def test_config_metadata_is_metadata_only_until_requested():
    """Config metadata should not require eager registration at registry import time."""
    from nemo_automodel._transformers.registry import MODEL_PACKAGE_SPECS

    for spec in MODEL_PACKAGE_SPECS:
        if spec.config_module_path is None:
            continue
        assert spec.config_module_path
        assert spec.config_class_name


def test_kimi_k25_arch_alias_in_model_package_specs():
    """KimiK25ForConditionalGeneration (checkpoint arch) must map to KimiK25VLForConditionalGeneration."""
    from nemo_automodel._transformers.registry import MODEL_PACKAGE_SPECS

    mapping = _model_arch_lookup(MODEL_PACKAGE_SPECS)
    assert "KimiK25ForConditionalGeneration" in mapping, (
        "KimiK25ForConditionalGeneration missing from MODEL_PACKAGE_SPECS. "
        "Kimi-K2.5 checkpoints use this architecture name and need it mapped "
        "to KimiK25VLForConditionalGeneration."
    )
    spec = mapping["KimiK25ForConditionalGeneration"]
    assert spec.class_name == "KimiK25VLForConditionalGeneration"


def test_deepseek_v4_registered_in_model_package_specs():
    """DeepseekV4ForCausalLM must be registered in MODEL_PACKAGE_SPECS."""
    from nemo_automodel._transformers.registry import MODEL_PACKAGE_SPECS

    mapping = _model_arch_lookup(MODEL_PACKAGE_SPECS)
    assert "DeepseekV4ForCausalLM" in mapping, (
        "DeepseekV4ForCausalLM missing from MODEL_PACKAGE_SPECS. "
        "DSV4 checkpoints declare this architecture and need it routed to the "
        "in-tree model implementation."
    )
    spec = mapping["DeepseekV4ForCausalLM"]
    assert spec.module_path == "nemo_automodel.components.models.deepseek_v4.model"
    assert spec.class_name == "DeepseekV4ForCausalLM"


def test_deepseek_v4_in_model_package_specs():
    """deepseek_v4 model_type must be declared in MODEL_PACKAGE_SPECS."""
    from nemo_automodel._transformers.registry import MODEL_PACKAGE_SPECS

    specs_by_model_type = {model_type: spec for spec in MODEL_PACKAGE_SPECS for model_type in spec.model_types}
    assert "deepseek_v4" in specs_by_model_type, (
        "deepseek_v4 must be in MODEL_PACKAGE_SPECS so ModelRegistry can register "
        "DSV4 configs on demand before AutoConfig.from_pretrained runs."
    )
    spec = specs_by_model_type["deepseek_v4"]
    assert spec.config_module_path == "nemo_automodel.components.models.deepseek_v4.config"
    assert spec.config_class_name == "DeepseekV4Config"


def test_all_model_folders_registered_in_auto_map():
    """Every model folder with a model.py must have at least one registry package-spec entry.

    This catches the case where a developer adds a new model directory under
    ``nemo_automodel/components/models/`` but forgets to add it to the static
    registry package specs in ``registry.py``.
    """
    import pathlib

    from nemo_automodel._transformers.registry import MODEL_PACKAGE_SPECS, RETRIEVAL_MODEL_PACKAGE_SPECS

    models_root = pathlib.Path(__file__).resolve().parents[3] / "nemo_automodel" / "components" / "models"

    # Collect the set of module paths referenced by the registries.
    registered_module_paths = {spec.module_path for spec in (*MODEL_PACKAGE_SPECS, *RETRIEVAL_MODEL_PACKAGE_SPECS)}

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
        f"in the registry package specs (registry.py). Add an entry for each architecture "
        f"exported by these modules."
    )

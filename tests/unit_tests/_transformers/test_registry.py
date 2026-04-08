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


def test_lazy_arch_mapping_auto_map():
    """Static auto_map entries are lazily loaded on first access."""
    from nemo_automodel._transformers.registry import _LazyArchMapping

    class FakeClass:
        pass

    fake_module = types.SimpleNamespace(FakeClass=FakeClass)
    mapping = _LazyArchMapping({"FakeArch": ("fake.module", "FakeClass")})

    mapping._modules["fake.module"] = fake_module

    assert "FakeArch" in mapping
    assert mapping["FakeArch"] is FakeClass
    assert "FakeArch" in mapping._loaded

    with pytest.raises(KeyError):
        mapping["NonExistent"]


def test_lazy_arch_mapping_extra_overrides_auto_map():
    """Dynamically registered entries take precedence over static entries."""
    from nemo_automodel._transformers.registry import _LazyArchMapping

    class StaticClass:
        pass

    class DynamicClass:
        pass

    fake_module = types.SimpleNamespace(StaticClass=StaticClass)
    mapping = _LazyArchMapping({"MyArch": ("fake.module", "StaticClass")})
    mapping._modules["fake.module"] = fake_module

    assert mapping["MyArch"] is StaticClass

    mapping["MyArch"] = DynamicClass
    assert mapping["MyArch"] is DynamicClass


def test_lazy_arch_mapping_unavailable_model():
    """Auto_map entries whose imports fail are removed and excluded from containment."""
    from nemo_automodel._transformers.registry import _LazyArchMapping

    mapping = _LazyArchMapping({"BadArch": ("nonexistent.module.path", "BadClass")})

    assert "BadArch" not in mapping
    assert "BadArch" not in mapping._auto_map


def test_default_registry_has_static_entries():
    """The default registry is populated from MODEL_ARCH_MAPPING."""
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING, _ModelRegistry

    inst = _ModelRegistry()
    for arch_name in MODEL_ARCH_MAPPING:
        assert arch_name in inst.model_arch_name_to_cls.keys()


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


def test_custom_config_registrations_in_config_mapping():
    """Models in _CUSTOM_CONFIG_REGISTRATIONS must be registered in CONFIG_MAPPING after import.

    This ensures that AutoConfig.from_pretrained can resolve custom model types
    (e.g. kimi_k25, kimi_vl) from local checkpoints without trust_remote_code=True.
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    from nemo_automodel._transformers.registry import _CUSTOM_CONFIG_REGISTRATIONS

    missing = []
    for model_type in _CUSTOM_CONFIG_REGISTRATIONS:
        if model_type not in CONFIG_MAPPING:
            missing.append(model_type)

    assert not missing, (
        f"Model type(s) {missing} are in _CUSTOM_CONFIG_REGISTRATIONS but not in "
        f"CONFIG_MAPPING. The _register_custom_configs() call at module level may "
        f"have failed for these entries."
    )


def test_kimi_k25_arch_alias_in_model_arch_mapping():
    """KimiK25ForConditionalGeneration (checkpoint arch) must map to KimiK25VLForConditionalGeneration."""
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    assert "KimiK25ForConditionalGeneration" in MODEL_ARCH_MAPPING, (
        "KimiK25ForConditionalGeneration missing from MODEL_ARCH_MAPPING. "
        "Kimi-K2.5 checkpoints use this architecture name and need it mapped "
        "to KimiK25VLForConditionalGeneration."
    )
    module_path, cls_name = MODEL_ARCH_MAPPING["KimiK25ForConditionalGeneration"]
    assert cls_name == "KimiK25VLForConditionalGeneration"


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
    registered_module_paths = {v[0] for v in MODEL_ARCH_MAPPING.values()}

    missing = []
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith(("_", ".")):
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

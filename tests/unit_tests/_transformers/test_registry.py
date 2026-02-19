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

import importlib
import logging
import pkgutil
import types

import pytest


def _make_fake_package():
    return types.SimpleNamespace(__path__=["/dev/null"])


def _make_fake_walk(orig_walk, prefix_to_names):
    """
    Return a fake pkgutil.walk_packages that only intercepts specific prefixes.
    For any other prefix, it delegates to the original implementation.
    """

    def _fake_walk(path=None, prefix="", onerror=None):
        names = prefix_to_names.get(prefix)
        if names is not None:
            return [(None, n, False) for n in names]
        return orig_walk(path, prefix, onerror)

    return _fake_walk


def _make_fake_import(orig_import, package_to_modules):
    """
    Return a fake importlib.import_module that serves fake packages/modules for
    keys present in package_to_modules; delegates to original import otherwise.

    package_to_modules: dict[str, object]
        Keys are fully-qualified module names (e.g., "test.pkg", "test.pkg.mod").
        Values are either a fake package (with __path__) or a fake module object.
    """

    def _fake_import(name, package=None):
        if name in package_to_modules:
            return package_to_modules[name]
        return orig_import(name, package=package)

    return _fake_import


def _new_registry_instance(registry_module, monkeypatch):
    """Create a clean _ModelRegistry with the module-level dicts emptied."""
    monkeypatch.setattr(registry_module, "_MODEL_ARCH_TO_MODULE", {})
    monkeypatch.setattr(registry_module, "_ALIASES", {})
    return registry_module._ModelRegistry()


def test_walk_registers_single_class(monkeypatch):
    pkg_name = "test.pkg.single"
    mod_name = f"{pkg_name}.modA"

    class FakeModelA:
        pass

    fake_pkg = _make_fake_package()
    fake_mod = types.SimpleNamespace(ModelClass=FakeModelA)

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        _make_fake_walk(orig_walk, {f"{pkg_name}.": [mod_name]}),
    )
    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        _make_fake_import(orig_import, {pkg_name: fake_pkg, mod_name: fake_mod}),
    )

    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg, monkeypatch)
    inst.register_modeling_path(pkg_name)

    assert "FakeModelA" in inst.supported_models
    assert inst.get_model_cls_from_model_arch("FakeModelA") is FakeModelA


def test_walk_registers_list_of_classes(monkeypatch):
    pkg_name = "test.pkg.multi"
    mod_name = f"{pkg_name}.mod"

    class FakeModelB:
        pass

    class FakeModelC:
        pass

    fake_pkg = _make_fake_package()
    fake_mod = types.SimpleNamespace(ModelClass=[FakeModelB, FakeModelC])

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        _make_fake_walk(orig_walk, {f"{pkg_name}.": [mod_name]}),
    )
    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        _make_fake_import(orig_import, {pkg_name: fake_pkg, mod_name: fake_mod}),
    )

    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg, monkeypatch)
    inst.register_modeling_path(pkg_name)

    assert "FakeModelB" in inst.supported_models
    assert "FakeModelC" in inst.supported_models
    assert inst.get_model_cls_from_model_arch("FakeModelB") is FakeModelB
    assert inst.get_model_cls_from_model_arch("FakeModelC") is FakeModelC


def test_alias_applied_via_walk(monkeypatch):
    pkg_name = "test.pkg.override"
    mod_name = f"{pkg_name}.mod"

    Qwen3OmniMoeThinkerForConditionalGeneration = type(  # noqa: N806
        "Qwen3OmniMoeThinkerForConditionalGeneration", (), {}
    )

    fake_pkg = _make_fake_package()
    fake_mod = types.SimpleNamespace(ModelClass=Qwen3OmniMoeThinkerForConditionalGeneration)

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        _make_fake_walk(orig_walk, {f"{pkg_name}.": [mod_name]}),
    )
    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        _make_fake_import(orig_import, {pkg_name: fake_pkg, mod_name: fake_mod}),
    )

    from nemo_automodel._transformers import registry as reg

    monkeypatch.setattr(reg, "_MODEL_ARCH_TO_MODULE", {})
    monkeypatch.setattr(
        reg,
        "_ALIASES",
        {"Qwen3OmniMoeForConditionalGeneration": "Qwen3OmniMoeThinkerForConditionalGeneration"},
    )
    inst = reg._ModelRegistry()
    inst.register_modeling_path(pkg_name)

    assert "Qwen3OmniMoeForConditionalGeneration" in inst.supported_models
    assert (
        inst.get_model_cls_from_model_arch("Qwen3OmniMoeForConditionalGeneration")
        is Qwen3OmniMoeThinkerForConditionalGeneration
    )


def test_alias_applied_via_static_dict(monkeypatch):
    from nemo_automodel._transformers import registry as reg

    monkeypatch.setattr(
        reg,
        "_MODEL_ARCH_TO_MODULE",
        {"RealClass": "some.fake.module"},
    )
    monkeypatch.setattr(reg, "_ALIASES", {"AliasClass": "RealClass"})
    inst = reg._ModelRegistry()

    assert "RealClass" in inst.supported_models
    assert "AliasClass" in inst.supported_models
    assert inst.alias["AliasClass"] == "RealClass"


def test_lazy_import_resolves_class(monkeypatch):
    from nemo_automodel._transformers import registry as reg

    class LazyModel:
        pass

    fake_mod = types.SimpleNamespace(LazyModel=LazyModel)
    monkeypatch.setattr(
        reg,
        "_MODEL_ARCH_TO_MODULE",
        {"LazyModel": "test.lazy.module"},
    )
    monkeypatch.setattr(reg, "_ALIASES", {})

    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        _make_fake_import(orig_import, {"test.lazy.module": fake_mod}),
    )

    inst = reg._ModelRegistry()
    assert "LazyModel" in inst.supported_models
    assert "LazyModel" not in inst._cache

    cls = inst.get_model_cls_from_model_arch("LazyModel")
    assert cls is LazyModel
    assert inst._cache["LazyModel"] is LazyModel


def test_walk_skips_already_registered(monkeypatch):
    """Walking should skip models already present from the static dict."""
    pkg_name = "test.pkg.dup"
    mod_name = f"{pkg_name}.mod"

    class DupClass:
        pass

    fake_pkg = _make_fake_package()
    fake_mod = types.SimpleNamespace(ModelClass=DupClass)

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        _make_fake_walk(orig_walk, {f"{pkg_name}.": [mod_name]}),
    )
    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        _make_fake_import(orig_import, {pkg_name: fake_pkg, mod_name: fake_mod}),
    )

    from nemo_automodel._transformers import registry as reg

    monkeypatch.setattr(reg, "_MODEL_ARCH_TO_MODULE", {"DupClass": "pre.existing.module"})
    monkeypatch.setattr(reg, "_ALIASES", {})
    inst = reg._ModelRegistry()
    inst.register_modeling_path(pkg_name)

    assert "DupClass" not in inst._dynamic


def test_register_modeling_path_adds_and_registers(monkeypatch):
    pkg_name = "test.pkg.register"
    mod_name = f"{pkg_name}.mod"

    class ModelX:
        pass

    fake_pkg = _make_fake_package()
    fake_mod = types.SimpleNamespace(ModelClass=ModelX)

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        _make_fake_walk(orig_walk, {f"{pkg_name}.": [mod_name]}),
    )
    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        _make_fake_import(orig_import, {pkg_name: fake_pkg, mod_name: fake_mod}),
    )

    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg, monkeypatch)
    assert pkg_name not in inst._walked_paths
    inst.register_modeling_path(pkg_name)

    assert pkg_name in inst._walked_paths
    assert "ModelX" in inst.supported_models
    assert inst.get_model_cls_from_model_arch("ModelX") is ModelX


def test_supported_models_and_getter(monkeypatch):
    pkg_name = "test.pkg.getter"
    mod_name = f"{pkg_name}.mod"

    class A:
        pass

    fake_pkg = _make_fake_package()
    fake_mod = types.SimpleNamespace(ModelClass=A)

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        _make_fake_walk(orig_walk, {f"{pkg_name}.": [mod_name]}),
    )
    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        _make_fake_import(orig_import, {pkg_name: fake_pkg, mod_name: fake_mod}),
    )

    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg, monkeypatch)
    inst.register_modeling_path(pkg_name)

    assert isinstance(inst.supported_models, set)
    assert "A" in inst.supported_models
    assert inst.get_model_cls_from_model_arch("A") is A


def test_ignore_import_error_logs_warning(monkeypatch, caplog):
    pkg_name = "test.pkg.bad"
    bad_mod_name = f"{pkg_name}.badmod"

    fake_pkg = _make_fake_package()

    orig_import = importlib.import_module

    def _raising_import(name, package=None):
        if name == pkg_name:
            return fake_pkg
        if name == bad_mod_name:
            raise RuntimeError("boom")
        return orig_import(name, package=package)

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        _make_fake_walk(orig_walk, {f"{pkg_name}.": [bad_mod_name]}),
    )
    monkeypatch.setattr(importlib, "import_module", _raising_import)

    from nemo_automodel._transformers import registry as reg

    caplog.set_level(logging.WARNING, logger=reg.__name__)
    inst = _new_registry_instance(reg, monkeypatch)
    inst.register_modeling_path(pkg_name)

    assert any("Ignore import error when loading" in rec.message for rec in caplog.records)


def test_get_registry_is_cached(monkeypatch):
    from nemo_automodel._transformers import registry as reg

    reg.get_registry.cache_clear()
    r1 = reg.get_registry()
    r2 = reg.get_registry()
    assert r1 is r2


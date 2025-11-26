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


def _new_registry_instance(registry_module):
    # Use an empty modeling_path to avoid scanning the real codebase
    return registry_module._ModelRegistry(modeling_path=[])


def test_mapping_registers_single_class(monkeypatch):
    # Prepare fakes for our synthetic package
    pkg_name = "test.pkg.single"
    mod_name = f"{pkg_name}.modA"

    class FakeModelA:
        pass

    fake_pkg = _make_fake_package()
    fake_mod = types.SimpleNamespace(ModelClass=FakeModelA)

    # Patch discovery/import only for our synthetic prefix
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

    inst = _new_registry_instance(reg)
    inst._mapping_model_arch_name_to_cls(pkg_name)

    assert "FakeModelA" in inst.model_arch_name_to_cls
    assert inst.model_arch_name_to_cls["FakeModelA"] is FakeModelA


def test_mapping_registers_list_of_classes(monkeypatch):
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

    inst = _new_registry_instance(reg)
    inst._mapping_model_arch_name_to_cls(pkg_name)

    assert "FakeModelB" in inst.model_arch_name_to_cls
    assert "FakeModelC" in inst.model_arch_name_to_cls
    assert inst.model_arch_name_to_cls["FakeModelB"] is FakeModelB
    assert inst.model_arch_name_to_cls["FakeModelC"] is FakeModelC


def test_naming_override_applied(monkeypatch):
    pkg_name = "test.pkg.override"
    mod_name = f"{pkg_name}.mod"

    # Create a class with the exact name to be overridden
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

    inst = _new_registry_instance(reg)
    inst._mapping_model_arch_name_to_cls(pkg_name)

    # Ensure the override mapping key is used
    assert "Qwen3OmniMoeForConditionalGeneration" in inst.model_arch_name_to_cls
    assert (
        inst.model_arch_name_to_cls["Qwen3OmniMoeForConditionalGeneration"]
        is Qwen3OmniMoeThinkerForConditionalGeneration
    )


def test_duplicate_model_raises_assertion(monkeypatch):
    pkg_name = "test.pkg.dup"
    mod1 = f"{pkg_name}.mod1"
    mod2 = f"{pkg_name}.mod2"

    # Two different classes but same __name__ "DupClass"
    DupClass1 = type("DupClass", (), {})
    DupClass2 = type("DupClass", (), {})

    fake_pkg = _make_fake_package()
    fake_mod1 = types.SimpleNamespace(ModelClass=DupClass1)
    fake_mod2 = types.SimpleNamespace(ModelClass=DupClass2)

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(
        pkgutil,
        "walk_packages",
        _make_fake_walk(orig_walk, {f"{pkg_name}.": [mod1, mod2]}),
    )
    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        _make_fake_import(orig_import, {pkg_name: fake_pkg, mod1: fake_mod1, mod2: fake_mod2}),
    )

    from nemo_automodel._transformers import registry as reg

    inst = _new_registry_instance(reg)

    with pytest.raises(AssertionError):
        inst._mapping_model_arch_name_to_cls(pkg_name)


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

    inst = _new_registry_instance(reg)
    assert pkg_name not in inst.modeling_path
    inst.register_modeling_path(pkg_name)

    assert pkg_name in inst.modeling_path
    assert "ModelX" in inst.model_arch_name_to_cls
    assert inst.model_arch_name_to_cls["ModelX"] is ModelX


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

    inst = _new_registry_instance(reg)
    inst._mapping_model_arch_name_to_cls(pkg_name)

    keys_view_type = type({}.keys())
    assert isinstance(inst.supported_models, keys_view_type)
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
    inst = _new_registry_instance(reg)
    inst._mapping_model_arch_name_to_cls(pkg_name)

    assert any("Ignore import error when loading" in rec.message for rec in caplog.records)


def test_get_registry_is_cached(monkeypatch):
    # Avoid scanning the real codebase by overriding MODELING_PATH
    from nemo_automodel._transformers import registry as reg

    pkg_name = "test.pkg.cached"
    fake_pkg = _make_fake_package()

    orig_walk = pkgutil.walk_packages
    monkeypatch.setattr(pkgutil, "walk_packages", _make_fake_walk(orig_walk, {f"{pkg_name}.": []}))
    orig_import = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", _make_fake_import(orig_import, {pkg_name: fake_pkg})
    )

    monkeypatch.setattr(reg, "MODELING_PATH", [pkg_name])

    # Reset cache and verify memoization
    reg.get_registry.cache_clear()
    r1 = reg.get_registry()
    r2 = reg.get_registry()
    assert r1 is r2


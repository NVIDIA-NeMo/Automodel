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

"""Unit tests for ``nemo_automodel._transformers.v4_patches.layer_types``.

Tests are kept hermetic by swapping ``transformers.configuration_utils`` in
``sys.modules`` with a stand-in module, rather than touching the real
package (which may or may not be installed and may already be imported).
"""

from __future__ import annotations

import sys
import types

import pytest

from nemo_automodel._transformers.v4_patches import layer_types as lt_mod


@pytest.fixture
def isolated_layer_types_state():
    """Reset module globals and meta-path / sys.modules mutations after each test."""
    saved_patched = lt_mod._PATCHED
    saved_hook_installed = lt_mod._HOOK_INSTALLED
    saved_validator_relaxed = lt_mod._VALIDATOR_RELAXED
    saved_meta_path = list(sys.meta_path)
    saved_transformers = sys.modules.get("transformers")
    saved_cu = sys.modules.get(lt_mod._TARGET_MODULE)

    lt_mod._PATCHED = False
    lt_mod._HOOK_INSTALLED = False
    lt_mod._VALIDATOR_RELAXED = False
    # The package installs a finder at import time; strip it so each test can
    # reason about the finders it inserts in isolation. The original list is
    # restored verbatim on teardown.
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, lt_mod._LayerTypesPatchFinder)]

    try:
        yield
    finally:
        lt_mod._PATCHED = saved_patched
        lt_mod._HOOK_INSTALLED = saved_hook_installed
        lt_mod._VALIDATOR_RELAXED = saved_validator_relaxed
        sys.meta_path[:] = saved_meta_path
        _restore(sys.modules, "transformers", saved_transformers)
        _restore(sys.modules, lt_mod._TARGET_MODULE, saved_cu)


def _restore(mapping, key, value):
    if value is None:
        mapping.pop(key, None)
    else:
        mapping[key] = value


def _install_fake_transformers(initial_types=("sliding_attention", "full_attention")):
    """Register a minimal transformers package exposing ``configuration_utils``."""
    fake_pkg = types.ModuleType("transformers")
    fake_pkg.__path__ = []  # mark as package so submodule imports resolve

    fake_cu = types.ModuleType("transformers.configuration_utils")
    fake_cu.ALLOWED_LAYER_TYPES = tuple(initial_types)

    fake_pkg.configuration_utils = fake_cu

    sys.modules["transformers"] = fake_pkg
    sys.modules["transformers.configuration_utils"] = fake_cu
    return fake_cu


class TestPatchAllowedLayerTypes:
    def test_extends_allowed_layer_types(self, isolated_layer_types_state):
        fake_cu = _install_fake_transformers(initial_types=("sliding_attention",))

        modified = lt_mod.patch_allowed_layer_types()

        assert modified is True
        assert "sliding_attention" in fake_cu.ALLOWED_LAYER_TYPES  # preserved
        for extra in lt_mod.DEFAULT_EXTRA_LAYER_TYPES:
            assert extra in fake_cu.ALLOWED_LAYER_TYPES

    def test_idempotent_second_call_noop(self, isolated_layer_types_state):
        fake_cu = _install_fake_transformers(initial_types=("sliding_attention",))

        assert lt_mod.patch_allowed_layer_types() is True
        after_first = fake_cu.ALLOWED_LAYER_TYPES

        # _PATCHED should short-circuit; tuple must be identical.
        assert lt_mod.patch_allowed_layer_types() is False
        assert fake_cu.ALLOWED_LAYER_TYPES == after_first

    def test_no_duplicates_when_guard_reset(self, isolated_layer_types_state):
        """Even if the _PATCHED guard is bypassed, entries aren't duplicated."""
        fake_cu = _install_fake_transformers(initial_types=("sliding_attention",))

        lt_mod.patch_allowed_layer_types()
        lt_mod._PATCHED = False  # simulate a stale/unguarded re-call
        lt_mod.patch_allowed_layer_types()

        for extra in lt_mod.DEFAULT_EXTRA_LAYER_TYPES:
            assert fake_cu.ALLOWED_LAYER_TYPES.count(extra) == 1

    def test_custom_extra_argument(self, isolated_layer_types_state):
        fake_cu = _install_fake_transformers(initial_types=("existing",))

        assert lt_mod.patch_allowed_layer_types(extra=("foo", "bar")) is True
        assert "foo" in fake_cu.ALLOWED_LAYER_TYPES
        assert "bar" in fake_cu.ALLOWED_LAYER_TYPES
        # Defaults should NOT have been added when a custom extra is supplied.
        for default_extra in lt_mod.DEFAULT_EXTRA_LAYER_TYPES:
            assert default_extra not in fake_cu.ALLOWED_LAYER_TYPES

    def test_skips_when_attribute_missing(self, isolated_layer_types_state):
        fake_pkg = types.ModuleType("transformers")
        fake_pkg.__path__ = []
        fake_cu = types.ModuleType("transformers.configuration_utils")
        # Intentionally no ALLOWED_LAYER_TYPES attribute.
        fake_pkg.configuration_utils = fake_cu
        sys.modules["transformers"] = fake_pkg
        sys.modules["transformers.configuration_utils"] = fake_cu

        assert lt_mod.patch_allowed_layer_types() is False
        assert not hasattr(fake_cu, "ALLOWED_LAYER_TYPES")


class TestInstallLayerTypesPatchHook:
    def test_defers_when_transformers_not_loaded(self, isolated_layer_types_state):
        sys.modules.pop(lt_mod._TARGET_MODULE, None)
        sys.modules.pop("transformers", None)

        assert lt_mod.install_layer_types_patch_hook() is True

        finders = [f for f in sys.meta_path if isinstance(f, lt_mod._LayerTypesPatchFinder)]
        assert len(finders) == 1
        # Patch must NOT have run yet — no configuration_utils was present.
        assert lt_mod._PATCHED is False

    def test_patches_immediately_when_already_loaded(self, isolated_layer_types_state):
        fake_cu = _install_fake_transformers(initial_types=("sliding_attention",))

        assert lt_mod.install_layer_types_patch_hook() is True

        # Already-imported branch: patch is applied directly, no finder added.
        assert lt_mod._PATCHED is True
        for extra in lt_mod.DEFAULT_EXTRA_LAYER_TYPES:
            assert extra in fake_cu.ALLOWED_LAYER_TYPES
        finders = [f for f in sys.meta_path if isinstance(f, lt_mod._LayerTypesPatchFinder)]
        assert finders == []

    def test_hook_install_is_idempotent(self, isolated_layer_types_state):
        sys.modules.pop(lt_mod._TARGET_MODULE, None)
        sys.modules.pop("transformers", None)

        assert lt_mod.install_layer_types_patch_hook() is True
        assert lt_mod.install_layer_types_patch_hook() is False

        finders = [f for f in sys.meta_path if isinstance(f, lt_mod._LayerTypesPatchFinder)]
        assert len(finders) == 1

    def test_finder_triggers_patch_on_import(self, isolated_layer_types_state):
        """End-to-end: installing the hook, then loading the target module,
        should cause ``ALLOWED_LAYER_TYPES`` to be extended post-exec."""
        sys.modules.pop(lt_mod._TARGET_MODULE, None)
        sys.modules.pop("transformers", None)

        assert lt_mod.install_layer_types_patch_hook() is True
        assert lt_mod._PATCHED is False

        # Simulate transformers.configuration_utils landing in sys.modules *after*
        # the hook was installed, then manually drive the finder's wrapped loader
        # flow the way importlib would.
        finder = next(f for f in sys.meta_path if isinstance(f, lt_mod._LayerTypesPatchFinder))

        fake_pkg = types.ModuleType("transformers")
        fake_pkg.__path__ = []
        sys.modules["transformers"] = fake_pkg

        fake_cu = types.ModuleType("transformers.configuration_utils")
        fake_cu.ALLOWED_LAYER_TYPES = ("sliding_attention",)

        class _StubLoader:
            def exec_module(self, module):
                # Populate the module the way the real loader would.
                module.ALLOWED_LAYER_TYPES = fake_cu.ALLOWED_LAYER_TYPES

        class _StubSpec:
            def __init__(self):
                self.loader = _StubLoader()

        class _StubFinder:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == lt_mod._TARGET_MODULE:
                    return _StubSpec()
                return None

        # Place the stub finder *after* ours so our finder delegates to it.
        sys.meta_path.append(_StubFinder())

        spec = finder.find_spec(lt_mod._TARGET_MODULE)
        assert spec is not None and spec.loader is not None

        # Driving the wrapped exec_module should apply the patch.
        sys.modules[lt_mod._TARGET_MODULE] = fake_cu
        spec.loader.exec_module(fake_cu)

        assert lt_mod._PATCHED is True
        for extra in lt_mod.DEFAULT_EXTRA_LAYER_TYPES:
            assert extra in fake_cu.ALLOWED_LAYER_TYPES


def _install_fake_pretrained_config_tree():
    """Register a minimal transformers.PretrainedConfig + subclass with a fake validator."""

    def _boom(self):
        raise ValueError("`num_hidden_layers` must equal number of layer types")

    _boom.__name__ = "validate_layer_type"

    fake_pkg = types.ModuleType("transformers")
    fake_pkg.__path__ = []

    class _FakePretrainedConfig:
        validate_layer_type = _boom
        __class_validators__ = [_boom]

    class _FakeChildConfig(_FakePretrainedConfig):
        __class_validators__ = [_boom]

    fake_pkg.PretrainedConfig = _FakePretrainedConfig
    sys.modules["transformers"] = fake_pkg
    return _FakePretrainedConfig, _FakeChildConfig, _boom


class TestRelaxLayerTypesValidator:
    def test_replaces_validator_on_base_and_subclass(self, isolated_layer_types_state):
        base, child, _original = _install_fake_pretrained_config_tree()

        assert lt_mod.relax_layer_types_validator() is True

        assert base.validate_layer_type is lt_mod._noop_validate_layer_type
        assert base.__class_validators__[0] is lt_mod._noop_validate_layer_type
        assert child.__class_validators__[0] is lt_mod._noop_validate_layer_type

        instance = child.__new__(child)
        for v in child.__class_validators__:
            v(instance)  # must not raise

    def test_idempotent(self, isolated_layer_types_state):
        _install_fake_pretrained_config_tree()

        assert lt_mod.relax_layer_types_validator() is True
        assert lt_mod.relax_layer_types_validator() is False

    def test_missing_attribute_is_tolerated(self, isolated_layer_types_state):
        fake_pkg = types.ModuleType("transformers")
        fake_pkg.__path__ = []

        class _Bare:
            pass

        fake_pkg.PretrainedConfig = _Bare
        sys.modules["transformers"] = fake_pkg

        assert lt_mod.relax_layer_types_validator() is False

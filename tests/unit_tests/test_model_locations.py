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

"""Tests for the transformers/diffusers/retrieval model split.

Models are split between the generic ``transformers`` bridge, the
flow-matching/diffusion bridge, and retrieval-specific task ownership. Covers:

1. The :mod:`nemo_automodel._model_locations` routing table matches what is
   actually on disk (drift guard).
2. The deprecated pre-split ``nemo_automodel.components.models.*`` path still
   resolves, warns, and yields the *same* module objects as the new location.
"""

import importlib
import pathlib
import re
import subprocess
import sys

import pytest

from nemo_automodel._model_locations import (
    DIFFUSERS_MODELS,
    DIFFUSERS_MODELS_PACKAGE,
    RETRIEVAL_MODELS,
    RETRIEVAL_MODELS_PACKAGE,
    TRANSFORMERS_MODELS_PACKAGE,
    models_package_for,
    resolve_model_module,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
# Derived from the package constants rather than spelled out as path segments,
# so a future package rename cannot leave these pointing at a stale directory.
_TRANSFORMERS_DIR = _REPO_ROOT.joinpath(*TRANSFORMERS_MODELS_PACKAGE.split("."))
_DIFFUSERS_DIR = _REPO_ROOT.joinpath(*DIFFUSERS_MODELS_PACKAGE.split("."))
_RETRIEVAL_DIR = _REPO_ROOT.joinpath(*RETRIEVAL_MODELS_PACKAGE.split("."))


def _model_dirs(root: pathlib.Path) -> set[str]:
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(("_", ".")) and (p / "__init__.py").exists()
    }


def _collect_moved_warnings(module: str) -> list[str]:
    """Import ``module`` in a fresh interpreter and return its "has moved" warnings.

    A subprocess is used because the relocation warning is emitted once, on
    first load; poking at ``sys.modules`` in-process would leak rebuilt module
    objects into the rest of the test session.
    """
    code = (
        "import warnings, importlib\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        f"    importlib.import_module({module!r})\n"
        "for c in caught:\n"
        "    if issubclass(c.category, DeprecationWarning) and 'has moved to' in str(c.message):\n"
        "        print(c.message)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return [line for line in result.stdout.splitlines() if "has moved to" in line]


class TestRoutingTableMatchesDisk:
    """Model-location tables must stay in sync with the directory contents."""

    def test_package_constants_map_to_real_directories(self):
        """Guards the three derived paths above against a silent rename."""
        assert _TRANSFORMERS_DIR.is_dir(), _TRANSFORMERS_DIR
        assert _DIFFUSERS_DIR.is_dir(), _DIFFUSERS_DIR
        assert _RETRIEVAL_DIR.is_dir(), _RETRIEVAL_DIR

    def test_diffusers_table_matches_directory(self):
        assert DIFFUSERS_MODELS == _model_dirs(_DIFFUSERS_DIR)

    def test_retrieval_table_matches_directory(self):
        assert RETRIEVAL_MODELS == _model_dirs(_RETRIEVAL_DIR)

    def test_no_model_lives_in_multiple_packages(self):
        package_models = [
            _model_dirs(_TRANSFORMERS_DIR),
            _model_dirs(_DIFFUSERS_DIR),
            _model_dirs(_RETRIEVAL_DIR),
        ]
        for index, models in enumerate(package_models):
            for other in package_models[index + 1 :]:
                assert not (models & other)

    def test_diffusers_side_models_are_flow_matching_adapters(self):
        """The invariant that actually separates the two sides.

        "Imports upstream ``diffusers``" does *not* work as the criterion: only
        ``qwen_image_edit`` does, because ``NeMoAutoDiffusionPipeline`` builds
        the pipeline and hands the adapter tensors. What every diffusers-side
        package does have is a concrete ``ModelAdapter``.

        Asserted against the ABC rather than the presence of an ``adapter.py``,
        so a stub file or a half-implemented adapter fails here.
        """
        from nemo_automodel.components.flow_matching.adapters.base import ModelAdapter

        for name in sorted(_model_dirs(_DIFFUSERS_DIR)):
            module = importlib.import_module(f"{DIFFUSERS_MODELS_PACKAGE}.{name}.adapter")
            concrete = [
                obj
                for obj in vars(module).values()
                if isinstance(obj, type)
                and issubclass(obj, ModelAdapter)
                and obj is not ModelAdapter
                and not obj.__abstractmethods__
            ]
            assert concrete, f"{name} exposes no concrete ModelAdapter subclass"

    def test_non_diffusers_models_are_not_adapters(self):
        """Converse direction: a ``ModelAdapter`` belongs on the diffusers side."""
        offenders = [
            f"{root.name}/{d}"
            for root in (_TRANSFORMERS_DIR, _RETRIEVAL_DIR)
            for d in sorted(_model_dirs(root))
            if any("ModelAdapter" in p.read_text(encoding="utf-8", errors="ignore") for p in (root / d).rglob("*.py"))
        ]
        assert offenders == [], f"non-diffusers packages defining a ModelAdapter: {offenders}"


class TestResolveModelModule:
    """Legacy flat paths map onto the split packages."""

    def test_empty_suffix_is_transformers_root(self):
        assert resolve_model_module("") == TRANSFORMERS_MODELS_PACKAGE

    def test_transformers_side_model(self):
        assert resolve_model_module("llama.model") == f"{TRANSFORMERS_MODELS_PACKAGE}.llama.model"

    def test_diffusers_side_model(self):
        assert resolve_model_module("qwen_image_edit.adapter") == f"{DIFFUSERS_MODELS_PACKAGE}.qwen_image_edit.adapter"

    def test_retrieval_side_model(self):
        assert resolve_model_module("llama_bidirectional.model") == (
            f"{RETRIEVAL_MODELS_PACKAGE}.llama_bidirectional.model"
        )

    def test_retrieval_helpers_route_out_of_transformers_common(self):
        assert resolve_model_module("common.bidirectional") == "nemo_automodel.retrieval.state_dict_adapter"
        assert resolve_model_module("common.inbatch_neg_utils") == "nemo_automodel.retrieval.inbatch_negatives"

    def test_non_model_members_route_to_transformers(self):
        for suffix in ("gpt2", "common.utils", "deprecation"):
            assert resolve_model_module(suffix).startswith(TRANSFORMERS_MODELS_PACKAGE)

    def test_models_package_for(self):
        assert models_package_for("qwen_image_edit") == DIFFUSERS_MODELS_PACKAGE
        assert models_package_for("llama_bidirectional") == RETRIEVAL_MODELS_PACKAGE
        assert models_package_for("llama") == TRANSFORMERS_MODELS_PACKAGE


class TestLegacyComponentsModelsAlias:
    """``nemo_automodel.components.models.*`` still works, but is deprecated."""

    @pytest.mark.parametrize(
        "legacy, canonical",
        [
            (
                "nemo_automodel.components.models.llama.model",
                "nemo_automodel._transformers.models.llama.model",
            ),
            (
                "nemo_automodel.components.models.qwen_image_edit.adapter",
                "nemo_automodel._diffusers.models.qwen_image_edit.adapter",
            ),
            (
                "nemo_automodel.components.models.llama_bidirectional.model",
                "nemo_automodel.retrieval.models.llama_bidirectional.model",
            ),
            # Shared helpers, not a model -- reached by both downstream code and
            # YAML _target_ values.
            (
                "nemo_automodel.components.models.common",
                "nemo_automodel._transformers.models.common",
            ),
            (
                "nemo_automodel.components.models.common.utils",
                "nemo_automodel._transformers.models.common.utils",
            ),
            (
                "nemo_automodel.components.models.common.bidirectional",
                "nemo_automodel.retrieval.state_dict_adapter",
            ),
            (
                "nemo_automodel.components.models.common.inbatch_neg_utils",
                "nemo_automodel.retrieval.inbatch_negatives",
            ),
        ],
    )
    def test_legacy_path_is_same_module_object(self, legacy, canonical):
        assert importlib.import_module(legacy) is importlib.import_module(canonical)

    @pytest.mark.parametrize(
        "target",
        [
            # Both spellings appear in the wild: re-exported from the package,
            # and from the module that defines it.
            "nemo_automodel.components.models.common.BackendConfig",
            "nemo_automodel.components.models.common.utils.BackendConfig",
        ],
    )
    def test_legacy_backend_config_target_resolves(self, target):
        """``BackendConfig`` is the most-referenced symbol under the old namespace."""
        from nemo_automodel._transformers.models.common import BackendConfig
        from nemo_automodel.components.config.loader import _resolve_target

        assert _resolve_target(target) is BackendConfig

    def test_legacy_import_emits_deprecation_warning(self):
        """The warning only fires on first load, so import in a clean interpreter."""
        moved = _collect_moved_warnings("nemo_automodel.components.models.qwen2.model")
        assert any("transformers.models" in m for m in moved), moved

    def test_legacy_namespace_rejects_names_that_never_lived_there(self):
        """flux/wan/... were never under components.models; don't invent them."""
        for name in ("flux", "wan", "hunyuan", "ltx2", "qwen_image"):
            assert resolve_model_module(name, legacy=True).startswith(TRANSFORMERS_MODELS_PACKAGE)
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(f"nemo_automodel.components.models.{name}")

    def test_legacy_namespace_keeps_the_one_name_that_did(self):
        assert resolve_model_module("qwen_image_edit.adapter", legacy=True).startswith(DIFFUSERS_MODELS_PACKAGE)

    def test_public_models_alias_routes_to_all_model_packages(self):
        """The live alias reflects current locations, unlike the legacy one."""
        assert resolve_model_module("flux").startswith(DIFFUSERS_MODELS_PACKAGE)
        assert resolve_model_module("llama_bidirectional").startswith(RETRIEVAL_MODELS_PACKAGE)
        assert resolve_model_module("llama").startswith(TRANSFORMERS_MODELS_PACKAGE)

    def test_public_models_alias_does_not_warn(self):
        """``nemo_automodel.models.*`` is supported, not deprecated."""
        assert _collect_moved_warnings("nemo_automodel.models.qwen3.model") == []
        assert _collect_moved_warnings("nemo_automodel.models.llama_bidirectional.model") == []


class TestRelocatedModules:
    """Individual module aliases resolve to one canonical module object."""

    def test_rename_table_targets_exist(self):
        from nemo_automodel import _RENAMED_MODULES

        for old, new in _RENAMED_MODULES.items():
            assert importlib.import_module(new).__name__ == new
            assert not (_REPO_ROOT / pathlib.Path(*old.split("."))).with_suffix(".py").exists(), (
                f"{old} still exists on disk; the alias would be shadowed by the real module"
            )

    def test_legacy_module_path_is_same_object(self):
        from nemo_automodel import _RENAMED_MODULES

        for old, new in _RENAMED_MODULES.items():
            assert importlib.import_module(old) is importlib.import_module(new)


class TestRelocatedRetrievalPackages:
    """Retrieval families moved as packages without duplicating module state."""

    def test_old_package_paths_are_identity_preserving_aliases(self):
        from nemo_automodel import _RENAMED_PACKAGES

        for old, new in _RENAMED_PACKAGES.items():
            assert not _REPO_ROOT.joinpath(*old.split(".")).exists()
            assert importlib.import_module(old) is importlib.import_module(new)

    @pytest.mark.parametrize(
        "old, canonical",
        [
            (
                "nemo_automodel._transformers.models.llama_bidirectional.model",
                "nemo_automodel.retrieval.models.llama_bidirectional.model",
            ),
            (
                "nemo_automodel._transformers.models.ministral_bidirectional.model",
                "nemo_automodel.retrieval.models.ministral_bidirectional.model",
            ),
            (
                "nemo_automodel._transformers.models.llama_nemotron_vl.processor",
                "nemo_automodel.retrieval.models.llama_nemotron_vl.processor",
            ),
            (
                "nemo_automodel.models.llama_bidirectional.model",
                "nemo_automodel.retrieval.models.llama_bidirectional.model",
            ),
            (
                "nemo_automodel.models.common.bidirectional",
                "nemo_automodel.retrieval.state_dict_adapter",
            ),
        ],
    )
    def test_old_and_public_children_are_same_module_object(self, old, canonical):
        assert importlib.import_module(old) is importlib.import_module(canonical)

    def test_retrieval_auto_model_exports_are_same_classes(self):
        import nemo_automodel

        canonical = importlib.import_module("nemo_automodel.retrieval.auto_model")
        legacy = importlib.import_module("nemo_automodel._transformers.auto_model")
        for name in ("NeMoAutoModelBiEncoder", "NeMoAutoModelCrossEncoder"):
            cls = getattr(canonical, name)
            assert getattr(legacy, name) is cls
            assert getattr(nemo_automodel, name) is cls

    def test_retrieval_package_is_lazy(self):
        code = (
            "import sys\n"
            "import nemo_automodel.retrieval\n"
            "assert 'nemo_automodel.retrieval.auto_model' not in sys.modules\n"
            "assert 'nemo_automodel.retrieval.modeling' not in sys.modules\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize(
        "code",
        [
            ("import nemo_automodel._transformers.registry\nimport nemo_automodel.retrieval.modeling\n"),
            ("import nemo_automodel.retrieval.modeling\nimport nemo_automodel._transformers.registry\n"),
            (
                "from nemo_automodel._transformers.auto_model import NeMoAutoModelBiEncoder as old\n"
                "from nemo_automodel.retrieval.auto_model import NeMoAutoModelBiEncoder as new\n"
                "assert old is new\n"
            ),
            (
                "from nemo_automodel.retrieval.auto_model import NeMoAutoModelCrossEncoder as new\n"
                "from nemo_automodel._transformers.auto_model import NeMoAutoModelCrossEncoder as old\n"
                "assert old is new\n"
            ),
        ],
    )
    def test_retrieval_import_orders_are_cycle_free(self, code):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stderr


class TestRelocatedFlowMatchingAdapters:
    """Concrete adapters moved, while the flow-matching component stays put."""

    def test_flow_matching_remains_a_component(self):
        assert (_REPO_ROOT / "nemo_automodel/components/flow_matching").is_dir()
        assert not (_REPO_ROOT / "nemo_automodel/retrieval/flow_matching").exists()

    def test_legacy_package_attribute_still_resolves(self):
        """``from ...flow_matching.adapters import FluxAdapter`` must keep working."""
        adapters = importlib.import_module("nemo_automodel.components.flow_matching.adapters")
        for name in (
            "FluxAdapter",
            "Flux2Adapter",
            "HunyuanAdapter",
            "LTX2Adapter",
            "QwenImageAdapter",
            "SimpleAdapter",
        ):
            assert getattr(adapters, name).__module__.startswith(DIFFUSERS_MODELS_PACKAGE)

    def test_every_configured_adapter_type_is_dispatchable(self):
        """Each ``adapter_type`` used by a shipped recipe must still build."""
        from nemo_automodel.components.flow_matching.pipeline import create_adapter

        pattern = re.compile(r"adapter_type:\s*\"?([a-z_0-9]+)")
        configured = {
            m.group(1)
            for yaml in (_REPO_ROOT / "examples").rglob("*.yaml")
            for m in pattern.finditer(yaml.read_text(encoding="utf-8", errors="ignore"))
        }
        assert configured, "no adapter_type found in examples/ -- the probe is broken"
        for adapter_type in sorted(configured):
            built = create_adapter(adapter_type)
            assert type(built).__module__.startswith(DIFFUSERS_MODELS_PACKAGE), (
                f"adapter_type '{adapter_type}' built {type(built).__module__}, expected a diffusers-side model"
            )

    def test_importing_adapters_package_does_not_load_models(self):
        """The contract must stay cheap to import."""
        code = (
            "import importlib, sys\n"
            "importlib.import_module('nemo_automodel.components.flow_matching.adapters')\n"
            "loaded = [m for m in sys.modules if m.startswith('nemo_automodel._diffusers.models.')]\n"
            "print(loaded)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[]", f"eagerly loaded model code: {result.stdout.strip()}"


class TestUpstreamPackagesNotShadowed:
    """``nemo_automodel._transformers`` must not shadow the HuggingFace package.

    Python 3 absolute imports make this safe, but the names now collide, so
    pin the behaviour down.
    """

    @pytest.mark.parametrize("package", ["transformers", "diffusers"])
    def test_bare_import_resolves_to_site_packages(self, package):
        code = (
            "import nemo_automodel\n"
            f"import {package} as p\n"
            f"assert p.__name__ == '{package}', p.__name__\n"
            "assert 'nemo_automodel' not in (getattr(p, '__file__', '') or ''), p.__file__\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT)
        if "ModuleNotFoundError" in result.stderr:
            pytest.skip(f"upstream {package} not installed")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"


class TestFromImportForms:
    """``from <models pkg> import <model>`` must work for every entry point.

    Regression guard: the package ``__getattr__`` raises ``ModuleNotFoundError``
    for unknown models, but ``hasattr`` only swallows ``AttributeError`` -- so
    raising it for a model that *does* exist breaks the from-import form.
    """

    @pytest.mark.parametrize(
        "statement, expected",
        [
            (
                "from nemo_automodel._transformers.models import llama as p",
                "nemo_automodel._transformers.models.llama",
            ),
            (
                "from nemo_automodel._diffusers.models import qwen_image_edit as p",
                "nemo_automodel._diffusers.models.qwen_image_edit",
            ),
            (
                "from nemo_automodel.models import llama as p",
                "nemo_automodel._transformers.models.llama",
            ),
            (
                "from nemo_automodel.components.models import llama as p",
                "nemo_automodel._transformers.models.llama",
            ),
            (
                "from nemo_automodel.retrieval.models import llama_bidirectional as p",
                "nemo_automodel.retrieval.models.llama_bidirectional",
            ),
            (
                "from nemo_automodel._transformers.models import llama_bidirectional as p",
                "nemo_automodel.retrieval.models.llama_bidirectional",
            ),
            (
                "from nemo_automodel.models import llama_bidirectional as p",
                "nemo_automodel.retrieval.models.llama_bidirectional",
            ),
            (
                "from nemo_automodel.components.models import llama_bidirectional as p",
                "nemo_automodel.retrieval.models.llama_bidirectional",
            ),
        ],
    )
    def test_from_import_resolves(self, statement, expected):
        code = f"import warnings; warnings.simplefilter('ignore'); {statement}; print(p.__name__)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected

    def test_unknown_model_still_raises(self):
        with pytest.raises(ModuleNotFoundError, match="has no submodule 'not_a_model'"):
            importlib.import_module(f"{TRANSFORMERS_MODELS_PACKAGE}.not_a_model")


class TestWrongPackageErrorMessages:
    """Importing a model from the wrong side points at the right package."""

    def test_diffusers_model_from_transformers_package(self):
        with pytest.raises(ModuleNotFoundError, match=r"trained through the diffusion bridge"):
            importlib.import_module(f"{TRANSFORMERS_MODELS_PACKAGE}.qwen_image_edit")

    def test_transformers_model_from_diffusers_package(self):
        with pytest.raises(ModuleNotFoundError, match=TRANSFORMERS_MODELS_PACKAGE):
            importlib.import_module(f"{DIFFUSERS_MODELS_PACKAGE}.llama")

    def test_retrieval_model_from_diffusers_package(self):
        with pytest.raises(ModuleNotFoundError, match=RETRIEVAL_MODELS_PACKAGE):
            importlib.import_module(f"{DIFFUSERS_MODELS_PACKAGE}.llama_bidirectional")

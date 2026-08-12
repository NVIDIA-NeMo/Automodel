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

"""Tests for the transformers/diffusers model split.

Models are split by the upstream HuggingFace package they are written against.
Covers:

1. The :mod:`nemo_automodel._model_locations` routing table matches what is
   actually on disk (drift guard).
2. The deprecated pre-split ``nemo_automodel.components.models.*`` path still
   resolves, warns, and yields the *same* module objects as the new location.
"""

import importlib
import pathlib
import subprocess
import sys

import pytest

from nemo_automodel._model_locations import (
    DIFFUSERS_MODELS,
    DIFFUSERS_MODELS_PACKAGE,
    TRANSFORMERS_MODELS_PACKAGE,
    models_package_for,
    resolve_model_module,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TRANSFORMERS_DIR = _REPO_ROOT / "nemo_automodel" / "_transformers" / "models"
_DIFFUSERS_DIR = _REPO_ROOT / "nemo_automodel" / "_diffusers" / "models"


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
    """``DIFFUSERS_MODELS`` must stay in sync with the directory contents."""

    def test_diffusers_table_matches_directory(self):
        assert DIFFUSERS_MODELS == _model_dirs(_DIFFUSERS_DIR)

    def test_no_model_lives_in_both_packages(self):
        assert not (_model_dirs(_TRANSFORMERS_DIR) & _model_dirs(_DIFFUSERS_DIR))

    def test_only_diffusers_side_models_import_diffusers(self):
        """A model importing upstream ``diffusers`` belongs under ``_diffusers``."""
        offenders = {
            d
            for d in _model_dirs(_TRANSFORMERS_DIR)
            if any("diffusers" in p.read_text(encoding="utf-8") for p in (_TRANSFORMERS_DIR / d).rglob("*.py"))
        }
        assert offenders == set(), f"transformers-side models referencing diffusers: {sorted(offenders)}"


class TestResolveModelModule:
    """Legacy flat paths map onto the split packages."""

    def test_empty_suffix_is_transformers_root(self):
        assert resolve_model_module("") == TRANSFORMERS_MODELS_PACKAGE

    def test_transformers_side_model(self):
        assert resolve_model_module("llama.model") == f"{TRANSFORMERS_MODELS_PACKAGE}.llama.model"

    def test_diffusers_side_model(self):
        assert resolve_model_module("qwen_image_edit.adapter") == f"{DIFFUSERS_MODELS_PACKAGE}.qwen_image_edit.adapter"

    def test_non_model_members_route_to_transformers(self):
        for suffix in ("gpt2", "common.utils", "deprecation"):
            assert resolve_model_module(suffix).startswith(TRANSFORMERS_MODELS_PACKAGE)

    def test_models_package_for(self):
        assert models_package_for("qwen_image_edit") == DIFFUSERS_MODELS_PACKAGE
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
        ],
    )
    def test_legacy_path_is_same_module_object(self, legacy, canonical):
        assert importlib.import_module(legacy) is importlib.import_module(canonical)

    def test_legacy_import_emits_deprecation_warning(self):
        """The warning only fires on first load, so import in a clean interpreter."""
        moved = _collect_moved_warnings("nemo_automodel.components.models.qwen2.model")
        assert any("_transformers.models" in m for m in moved), moved

    def test_public_models_alias_does_not_warn(self):
        """``nemo_automodel.models.*`` is supported, not deprecated."""
        assert _collect_moved_warnings("nemo_automodel.models.qwen3.model") == []


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
        with pytest.raises(ModuleNotFoundError, match=r"built on the 'diffusers' package"):
            importlib.import_module(f"{TRANSFORMERS_MODELS_PACKAGE}.qwen_image_edit")

    def test_transformers_model_from_diffusers_package(self):
        with pytest.raises(ModuleNotFoundError, match=TRANSFORMERS_MODELS_PACKAGE):
            importlib.import_module(f"{DIFFUSERS_MODELS_PACKAGE}.llama")

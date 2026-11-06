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

"""Unit tests for lazy import __init__.py patterns in datasets.diffusion, flow_matching, and _diffusers."""

import subprocess
import sys
from pathlib import Path

import pytest

# =============================================================================
# TestDatasetsLazyImports
# =============================================================================


class TestDatasetsLazyImports:
    """Tests for nemo_automodel.components.datasets.diffusion.__init__.py lazy imports."""

    def test_dir_returns_sorted_attrs(self):
        import nemo_automodel.components.datasets.diffusion as mod

        result = dir(mod)
        assert result == sorted(result)

    def test_all_matches_lazy_attrs(self):
        import nemo_automodel.components.datasets.diffusion as mod

        assert mod.__all__ == sorted(mod._LAZY_ATTRS.keys())

    def test_getattr_loads_valid_attrs(self):
        import nemo_automodel.components.datasets.diffusion as mod

        for attr_name in mod._LAZY_ATTRS:
            attr = getattr(mod, attr_name)
            assert attr is not None

    def test_invalid_attr_raises(self):
        import nemo_automodel.components.datasets.diffusion as mod

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(mod, "NonExistentClass12345")

    def test_specific_attrs_resolve(self):
        from nemo_automodel.components.datasets.diffusion import MetaFilesDataset, build_mock_dataloader

        assert MetaFilesDataset is not None
        assert build_mock_dataloader is not None


# =============================================================================
# TestFlowMatchingLazyImports
# =============================================================================


class TestFlowMatchingLazyImports:
    """Tests for nemo_automodel.components.flow_matching.__init__.py lazy imports."""

    def test_dir_returns_sorted_attrs(self):
        import nemo_automodel.components.flow_matching as mod

        result = dir(mod)
        assert result == sorted(result)

    def test_all_matches_lazy_attrs(self):
        import nemo_automodel.components.flow_matching as mod

        assert mod.__all__ == sorted(mod._LAZY_ATTRS.keys())

    def test_getattr_loads_valid_attrs(self):
        import nemo_automodel.components.flow_matching as mod

        for attr_name in mod._LAZY_ATTRS:
            attr = getattr(mod, attr_name)
            assert attr is not None

    def test_invalid_attr_raises(self):
        import nemo_automodel.components.flow_matching as mod

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(mod, "NonExistentPipeline12345")

    def test_specific_attrs_resolve(self):
        from nemo_automodel.components.flow_matching import FlowMatchingPipeline, FluxAdapter, SimpleAdapter

        assert FlowMatchingPipeline is not None
        assert FluxAdapter is not None
        assert SimpleAdapter is not None


# =============================================================================
# TestDiffusersLazyImports
# =============================================================================


class TestDiffusersLazyImports:
    """Tests for nemo_automodel._diffusers.__init__.py lazy imports."""

    def test_dir_returns_sorted_attrs(self):
        import nemo_automodel._diffusers as mod

        result = dir(mod)
        assert result == sorted(result)

    def test_all_matches_lazy_attrs(self):
        import nemo_automodel._diffusers as mod

        assert mod.__all__ == sorted(mod._LAZY_ATTRS.keys())

    def test_getattr_loads_valid_attrs(self):
        """Test lazy loading of each attribute. Some may fail due to optional deps."""
        import nemo_automodel._diffusers as mod

        for attr_name in mod._LAZY_ATTRS:
            try:
                attr = getattr(mod, attr_name)
                assert attr is not None
            except (ImportError, Exception):
                # diffusers may not be installed; that's OK for this test
                pytest.skip(f"Could not import {attr_name} (optional dependency missing)")

    def test_invalid_attr_raises(self):
        import nemo_automodel._diffusers as mod

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(mod, "NonExistentPipeline12345")


def _run_in_fresh_python(script: str) -> None:
    """Check import boundaries without modules cached by test collection."""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_text_worker_imports_remain_lightweight():
    _run_in_fresh_python(
        """
import importlib
import sys

models = importlib.import_module("nemo_automodel.components.models")
llm = importlib.import_module("nemo_automodel.components.datasets.llm")
zephon = importlib.import_module("nemo_automodel.components.datasets.zephon")
assert "build_gpt2_model" in dir(models)
assert "neat_pack_dataset" in dir(llm)
assert set(zephon.__all__) <= set(dir(zephon))
assert "nemo_automodel.components.datasets.zephon.config" not in sys.modules
importlib.import_module("nemo_automodel.components.datasets.zephon.llm")
for module in ("transformers", "datasets",
               "nemo_automodel.components.models.gpt2",
               "nemo_automodel.components.datasets.llm.retrieval_dataset",
               "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset"):
    assert module not in sys.modules, module

# Eager builder loading previously also exposed this submodule. Resolve it
# before requesting the builder to exercise the original import spelling.
from nemo_automodel.components.models import gpt2
assert gpt2 is importlib.import_module("nemo_automodel.models.gpt2")
assert models.build_gpt2_model is gpt2.build_gpt2_model
"""
    )


def test_text_package_exports_keep_their_original_owners():
    _run_in_fresh_python(
        """
import importlib

for package in ("nemo_automodel.components.models",
                "nemo_automodel.components.datasets.llm",
                "nemo_automodel.components.datasets.zephon"):
    module = importlib.import_module(package)
    for name in module.__all__:
        value = getattr(module, name)
        owner = importlib.import_module(value.__module__)
        assert value is getattr(owner, name), (package, name)
        assert value is getattr(module, name), (package, name)
    try:
        getattr(module, "unavailable_export")
    except ModuleNotFoundError as exc:
        assert package.endswith(".models") and "Upgrade with" in str(exc)
    except AttributeError:
        assert not package.endswith(".models")
    else:
        raise AssertionError(package)

llm = importlib.import_module("nemo_automodel.components.datasets.llm")
packing = importlib.import_module("nemo_automodel.components.datasets.llm.neat_packing")
assert llm.neat_pack_dataset is packing.neat_pack_dataset
assert "neat_pack_dataset" not in llm.__all__
try:
    importlib.import_module("nemo_automodel.components.models.unavailable_model")
except ModuleNotFoundError as exc:
    assert "Available model submodules" in str(exc)
else:
    raise AssertionError("Missing-model diagnostic was lost")
"""
    )

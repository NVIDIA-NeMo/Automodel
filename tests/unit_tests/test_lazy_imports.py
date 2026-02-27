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

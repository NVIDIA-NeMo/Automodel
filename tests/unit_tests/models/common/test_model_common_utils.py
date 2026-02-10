# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components.models.common.utils import (
    BackendConfig,
    TEFp8Config,
    build_fp8_recipe,
    maybe_te_fp8_autocast,
)


class TestTEFp8Config:
    def test_default_recipe(self):
        cfg = TEFp8Config()
        assert cfg.recipe == "current"

    def test_block_recipe(self):
        cfg = TEFp8Config(recipe="block")
        assert cfg.recipe == "block"

    def test_passthrough_recipe_object(self):
        """Non-string recipe objects are passed through directly."""
        sentinel = object()
        cfg = TEFp8Config(recipe=sentinel)
        assert build_fp8_recipe(cfg) is sentinel

    def test_maybe_te_autocast_without_te(self):
        """Without TE installed, maybe_te_fp8_autocast returns nullcontext."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            ctx = maybe_te_fp8_autocast(cfg)
            assert isinstance(ctx, nullcontext)

    def test_maybe_te_autocast_none_config(self):
        """None config returns nullcontext."""
        ctx = maybe_te_fp8_autocast(None)
        assert isinstance(ctx, nullcontext)

    def test_build_recipe_without_te(self):
        """Without TE installed, build_fp8_recipe returns None."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            assert build_fp8_recipe(cfg) is None


class TestBackendConfigTeFp8:
    def test_te_fp8_default_is_none(self):
        """BackendConfig.te_fp8 defaults to None."""
        cfg = BackendConfig()
        assert cfg.te_fp8 is None

    def test_te_fp8_dict_normalized(self):
        """A dict te_fp8 is converted to TEFp8Config."""
        cfg = BackendConfig(te_fp8={"recipe": "block"}, experts="te", dispatcher="deepep")
        assert isinstance(cfg.te_fp8, TEFp8Config)
        assert cfg.te_fp8.recipe == "block"

    def test_te_fp8_requires_te_backend(self):
        """te_fp8 requires linear='te' or experts='te'."""
        with pytest.raises(ValueError, match="te_fp8 requires at least one TE backend"):
            BackendConfig(te_fp8=TEFp8Config(), linear="torch", experts="torch")


# ---------------------------------------------------------------------------
#  FP8 weight-caching subclass tests
# ---------------------------------------------------------------------------


class _FakeLinear(torch.nn.Module):
    """Minimal stand-in for TE Linear so tests run without transformer_engine."""

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x, is_first_microbatch=None, **kwargs):
        return x @ self.weight.T


class _FakeGroupedLinear(torch.nn.Module):
    """Minimal stand-in for TE GroupedLinear."""

    def __init__(self, num_gemms, in_features, out_features, **kwargs):
        super().__init__()
        for i in range(num_gemms):
            setattr(self, f"weight{i}", torch.nn.Parameter(torch.randn(out_features, in_features)))
        self.num_gemms = num_gemms

    def forward(self, inp, m_splits, is_first_microbatch=None):
        return inp @ self.weight0.T


class TestFP8CachingLinear:
    """Test version-tracking logic in FP8CachingLinear."""

    def _make_cls(self):
        """Build FP8CachingLinear using _FakeLinear as the base."""
        with patch(
            "nemo_automodel.components.models.common.utils.importlib.util.find_spec",
            return_value=True,
        ), patch.dict(
            "sys.modules",
            {"transformer_engine.pytorch.module.linear": MagicMock(Linear=_FakeLinear)},
        ):
            from nemo_automodel.components.models.common.utils import _make_fp8_caching_linear

            return _make_fp8_caching_linear()

    def test_first_forward_is_first_microbatch(self):
        cls = self._make_cls()
        m = cls(in_features=4, out_features=4)
        x = torch.randn(2, 4)

        # Capture the is_first_microbatch passed to super
        calls = []
        original_forward = _FakeLinear.forward

        def spy(self_, x_, is_first_microbatch=None, **kw):
            calls.append(is_first_microbatch)
            return original_forward(self_, x_, is_first_microbatch=is_first_microbatch, **kw)

        with patch.object(_FakeLinear, "forward", spy):
            m(x)
        assert calls[-1] is True

    def test_second_forward_reuses_cache(self):
        cls = self._make_cls()
        m = cls(in_features=4, out_features=4)
        x = torch.randn(2, 4)

        calls = []
        original_forward = _FakeLinear.forward

        def spy(self_, x_, is_first_microbatch=None, **kw):
            calls.append(is_first_microbatch)
            return original_forward(self_, x_, is_first_microbatch=is_first_microbatch, **kw)

        with patch.object(_FakeLinear, "forward", spy):
            m(x)
            m(x)
        assert calls[0] is True
        assert calls[1] is False

    def test_weight_mutation_invalidates_cache(self):
        cls = self._make_cls()
        m = cls(in_features=4, out_features=4)
        x = torch.randn(2, 4)

        calls = []
        original_forward = _FakeLinear.forward

        def spy(self_, x_, is_first_microbatch=None, **kw):
            calls.append(is_first_microbatch)
            return original_forward(self_, x_, is_first_microbatch=is_first_microbatch, **kw)

        with patch.object(_FakeLinear, "forward", spy):
            m(x)  # first → True
            m(x)  # second → False
            with torch.no_grad():
                m.weight.add_(1)  # simulate optimizer step (in-place mutation)
            m(x)  # should detect version change → True
        assert calls == [True, False, True]

    def test_explicit_override_respected(self):
        cls = self._make_cls()
        m = cls(in_features=4, out_features=4)
        x = torch.randn(2, 4)

        calls = []
        original_forward = _FakeLinear.forward

        def spy(self_, x_, is_first_microbatch=None, **kw):
            calls.append(is_first_microbatch)
            return original_forward(self_, x_, is_first_microbatch=is_first_microbatch, **kw)

        with patch.object(_FakeLinear, "forward", spy):
            m(x, is_first_microbatch=False)
        assert calls[-1] is False


class TestFP8CachingGroupedLinear:
    """Test version-tracking logic in FP8CachingGroupedLinear."""

    def _make_cls(self):
        """Build FP8CachingGroupedLinear using _FakeGroupedLinear as the base."""
        with patch(
            "nemo_automodel.components.models.common.utils.importlib.util.find_spec",
            return_value=True,
        ), patch.dict(
            "sys.modules",
            {"transformer_engine.pytorch.module.grouped_linear": MagicMock(GroupedLinear=_FakeGroupedLinear)},
        ):
            from nemo_automodel.components.models.common.utils import _make_fp8_caching_grouped_linear

            return _make_fp8_caching_grouped_linear()

    def test_first_forward_is_first_microbatch(self):
        cls = self._make_cls()
        m = cls(num_gemms=2, in_features=4, out_features=4)
        x = torch.randn(2, 4)

        calls = []
        original_forward = _FakeGroupedLinear.forward

        def spy(self_, inp, m_splits, is_first_microbatch=None):
            calls.append(is_first_microbatch)
            return original_forward(self_, inp, m_splits, is_first_microbatch=is_first_microbatch)

        with patch.object(_FakeGroupedLinear, "forward", spy):
            m(x, [1, 1])
        assert calls[-1] is True

    def test_second_forward_reuses_cache(self):
        cls = self._make_cls()
        m = cls(num_gemms=2, in_features=4, out_features=4)
        x = torch.randn(2, 4)

        calls = []
        original_forward = _FakeGroupedLinear.forward

        def spy(self_, inp, m_splits, is_first_microbatch=None):
            calls.append(is_first_microbatch)
            return original_forward(self_, inp, m_splits, is_first_microbatch=is_first_microbatch)

        with patch.object(_FakeGroupedLinear, "forward", spy):
            m(x, [1, 1])
            m(x, [1, 1])
        assert calls == [True, False]

    def test_weight0_mutation_invalidates_cache(self):
        cls = self._make_cls()
        m = cls(num_gemms=2, in_features=4, out_features=4)
        x = torch.randn(2, 4)

        calls = []
        original_forward = _FakeGroupedLinear.forward

        def spy(self_, inp, m_splits, is_first_microbatch=None):
            calls.append(is_first_microbatch)
            return original_forward(self_, inp, m_splits, is_first_microbatch=is_first_microbatch)

        with patch.object(_FakeGroupedLinear, "forward", spy):
            m(x, [1, 1])  # first → True
            m(x, [1, 1])  # second → False
            with torch.no_grad():
                m.weight0.add_(1)  # simulate optimizer step (in-place mutation)
            m(x, [1, 1])  # should detect version change → True
        assert calls == [True, False, True]

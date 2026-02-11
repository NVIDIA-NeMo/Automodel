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
from unittest.mock import patch

import pytest

from nemo_automodel.components.models.common.utils import (
    BackendConfig,
    TEFp8Config,
    get_is_first_microbatch,
    get_is_optim_step,
    set_is_first_microbatch,
    set_is_optim_step,
)


class TestIsOptimStep:
    def teardown_method(self):
        set_is_optim_step(False)

    def test_default_is_false(self):
        set_is_optim_step(False)
        assert get_is_optim_step() is False

    def test_set_true(self):
        set_is_optim_step(True)
        assert get_is_optim_step() is True

    def test_set_false(self):
        set_is_optim_step(True)
        set_is_optim_step(False)
        assert get_is_optim_step() is False


class TestIsFirstMicrobatch:
    def teardown_method(self):
        set_is_first_microbatch(None)

    def test_default_is_none(self):
        set_is_first_microbatch(None)
        assert get_is_first_microbatch() is None

    def test_set_true(self):
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

    def test_set_false(self):
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False

    def test_set_none(self):
        set_is_first_microbatch(True)
        set_is_first_microbatch(None)
        assert get_is_first_microbatch() is None

    def test_grad_accumulation_lifecycle(self):
        """Simulate the typical GA lifecycle: True -> False -> True -> False."""
        # Start of optimizer step: first microbatch
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

        # After first microbatch
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False

        # Next optimizer step: first microbatch again
        set_is_first_microbatch(True)
        assert get_is_first_microbatch() is True

        # After first microbatch
        set_is_first_microbatch(False)
        assert get_is_first_microbatch() is False


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
        assert cfg.build_recipe() is sentinel

    def test_maybe_te_autocast_without_te(self):
        """Without TE installed, maybe_te_autocast returns nullcontext."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            ctx = cfg.maybe_te_autocast()
            assert isinstance(ctx, nullcontext)

    def test_build_recipe_without_te(self):
        """Without TE installed, build_recipe returns None."""
        cfg = TEFp8Config()
        with patch("nemo_automodel.components.models.common.utils.HAVE_TE", False):
            assert cfg.build_recipe() is None


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

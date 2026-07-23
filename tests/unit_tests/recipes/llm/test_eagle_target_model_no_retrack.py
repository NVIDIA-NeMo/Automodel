# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Regression tests for the EAGLE recipe ``target_model`` re-track fix.

``TrainEagle1Recipe.setup`` and ``TrainEagle3Recipe.setup`` used to do::

    self.target_model = self.target_model.to(self.device)

on the unsharded path.  ``nn.Module.to`` is in-place, so the reassignment is
redundant -- and it re-triggers ``BaseRecipe.__setattr__`` state tracking,
which raises ``RuntimeError: State key 'target_model' is already tracked``
since the duplicate-key guard was hardened from ``assert`` to ``raise``.

The fix drops the reassignment and keeps the in-place ``.to(self.device)``
call.  These tests pin that behavior on both EAGLE-1 (which EAGLE-2
inherits from) and EAGLE-3 by driving ``setup()`` up to the patched line
with stubs and asserting (1) no ``RuntimeError`` from ``__setattr__`` and
(2) ``target_model`` is tracked exactly once.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn


class _SetupReachedTarget(Exception):
    """Sentinel raised by the wrapper stub once ``setup()`` has executed
    past the patched ``self.target_model.to(self.device)`` line."""


def _tiny_target_module() -> nn.Module:
    """A real ``nn.Module`` so ``BaseRecipe.is_model`` flags it for tracking
    and so ``.to(device)`` is a real in-place no-op rather than a Mock call."""

    class _T(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)

        def requires_grad_(self, mode: bool = True):  # noqa: D401 - mirror nn.Module API
            for p in self.parameters():
                p.requires_grad_(mode)
            return self

    return _T()


def _fake_dist_env() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        world_size=1,
        is_main=True,
        rank=0,
        local_rank=0,
    )


def _fake_target_config() -> MagicMock:
    cfg = MagicMock()
    cfg.architectures = ["LlamaForCausalLM"]
    return cfg


def _make_cfg() -> MagicMock:
    """Build a minimal ``cfg`` mock matching the attribute access in
    ``setup()`` up to and including the patched line.

    No ``distributed:`` block -> ``self.dist_setup`` stays ``None`` ->
    the unsharded branch (the line under test) executes.
    """

    cfg = MagicMock()
    # ``self.cfg.get("distributed", None)`` -> None drives the unsharded path.
    cfg.get = MagicMock(side_effect=lambda key, default=None: default)
    cfg.recipe_args = SimpleNamespace(
        target_model_name_or_path="ignored/path",
        get=lambda key, default=None: default,
    )
    # ``recipe_cfg.get("trust_remote_code", False)`` style calls go through
    # ``recipe_args.get``; SimpleNamespace doesn't expose ``get`` by default,
    # so we attach one explicitly below.
    cfg.recipe_args.get = lambda key, default=None: "sdpa" if key == "target_attn_implementation" else default
    return cfg


@pytest.mark.parametrize(
    "recipe_module, recipe_cls_name, target_wrapper_attr, draft_spec_attr",
    [
        (
            "nemo_automodel.recipes.llm.train_eagle1",
            "TrainEagle1Recipe",
            "HFEagleTargetModel",
            "resolve_eagle1_draft_spec",
        ),
        (
            "nemo_automodel.recipes.llm.train_eagle3",
            "TrainEagle3Recipe",
            "HFEagle3TargetModel",
            "resolve_eagle3_draft_spec",
        ),
    ],
)
def test_setup_unsharded_does_not_retrack_target_model(
    recipe_module, recipe_cls_name, target_wrapper_attr, draft_spec_attr
):
    """Regression: ``setup()`` on the unsharded path must run
    ``self.target_model.to(self.device)`` in place, without re-triggering
    ``BaseRecipe.__setattr__`` state tracking on ``target_model``.
    """
    import importlib

    mod = importlib.import_module(recipe_module)
    recipe_cls = getattr(mod, recipe_cls_name)

    target_module = _tiny_target_module()

    # The wrapper is the first call site that runs *after* the patched line.
    # Raising a sentinel here both (a) confirms we got past the patched line
    # and (b) short-circuits the rest of ``setup()`` (which depends on real
    # data files / HF downloads we don't want in a unit test).
    def _wrapper_sentinel(*_args, **_kwargs):
        raise _SetupReachedTarget()

    cfg = _make_cfg()
    recipe = recipe_cls(cfg)

    with (
        patch.object(mod, "initialize_distributed", return_value=_fake_dist_env()),
        patch.object(mod, "setup_logging"),
        patch.object(mod, "AutoConfig") as mock_auto_config,
        patch.object(mod, "NeMoAutoTokenizer") as mock_tok,
        patch.object(mod, draft_spec_attr, return_value=MagicMock()),
        patch.object(mod, "NeMoAutoModelForCausalLM") as mock_auto_model,
        patch.object(mod, target_wrapper_attr, side_effect=_wrapper_sentinel),
    ):
        mock_auto_config.from_pretrained.return_value = _fake_target_config()
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_auto_model.from_pretrained.return_value = target_module

        with pytest.raises(_SetupReachedTarget):
            recipe.setup()

    # Recipe must hold the same module object (in-place ``.to``, no rebind).
    assert recipe.target_model is target_module

    # ``target_model`` must be tracked exactly once. The buggy reassignment
    # would have raised ``RuntimeError`` before reaching the wrapper sentinel.
    tracked = recipe.__dict__["__state_tracked"]
    assert "target_model" in tracked

    # EAGLE-3 forwards ``target_attn_implementation`` to the target load; EAGLE-1 ignores it.
    expected_attn = "sdpa" if recipe_cls_name == "TrainEagle3Recipe" else None
    assert mock_auto_model.from_pretrained.call_args.kwargs.get("attn_implementation") == expected_attn


def test_eagle2_inherits_eagle1_fix():
    """``TrainEagle2Recipe`` inherits ``setup`` from ``TrainEagle1Recipe``,
    so the same fix applies. Pin the inheritance and method identity so a
    future override has to consciously re-establish the contract.
    """
    from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe
    from nemo_automodel.recipes.llm.train_eagle2 import TrainEagle2Recipe

    assert issubclass(TrainEagle2Recipe, TrainEagle1Recipe)
    assert TrainEagle2Recipe.setup is TrainEagle1Recipe.setup

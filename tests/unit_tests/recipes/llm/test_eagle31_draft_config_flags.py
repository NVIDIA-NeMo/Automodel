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

"""Regression tests: ``recipe_args.fc_norm`` / ``norm_output`` must reach the draft config.

``TrainEagle3Recipe.setup`` used to assemble ``draft_config`` from the target
config plus a fixed set of explicit overrides that did not include the two
EAGLE-3.1 drafter toggles. The draft reads them via
``getattr(config, "fc_norm", False)`` / ``getattr(config, "norm_output",
False)``, so setting them in ``recipe_args`` (as both
``examples/speculative/eagle3_1`` YAMLs do) was silently ignored: the run
trained a plain EAGLE-3 draft and the saved ``config.json`` lacked the flags
at serve time too.

These tests drive ``setup()`` up to the draft-model build with stubs (the
same pattern as ``test_eagle_target_model_no_retrack``) and assert the flags
land on the draft config object exactly as set in ``recipe_args``, that they
survive serialization, and that a real draft built from that config registers
the EAGLE-3.1 modules.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel


class _DraftBuilt(Exception):
    """Sentinel raised by the draft-class stub once ``setup()`` has assembled
    the draft config; carries the config and short-circuits the rest of
    ``setup()`` (which needs real data files / a live target)."""

    def __init__(self, config):
        super().__init__("draft build reached")
        self.config = config


def _tiny_target_config() -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    return config


def _fake_dist_env() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        world_size=1,
        is_main=True,
        rank=0,
        local_rank=0,
    )


def _make_cfg(recipe_overrides: dict) -> MagicMock:
    """Minimal ``cfg`` mock matching the attribute access in ``setup()`` up to
    the draft build. ``recipe_overrides`` backs ``recipe_args.get``."""
    cfg = MagicMock()
    cfg.get = MagicMock(side_effect=lambda key, default=None: default)
    cfg.recipe_args = SimpleNamespace(target_model_name_or_path="ignored/path")
    cfg.recipe_args.get = lambda key, default=None: recipe_overrides.get(key, default)
    return cfg


def _run_setup_to_draft_build(recipe_overrides: dict) -> LlamaConfig:
    """Drive ``TrainEagle3Recipe.setup()`` through the draft-config assembly
    and return the config object handed to the draft class."""
    import nemo_automodel.recipes.llm.train_eagle3 as mod

    def _draft_cls(config):
        raise _DraftBuilt(config)

    recipe = mod.TrainEagle3Recipe(_make_cfg(recipe_overrides))
    with (
        patch.object(mod, "initialize_distributed", return_value=_fake_dist_env()),
        patch.object(mod, "setup_logging"),
        patch.object(mod, "AutoConfig") as mock_auto_config,
        patch.object(mod, "NeMoAutoTokenizer") as mock_tok,
        patch.object(mod, "resolve_eagle3_draft_spec", return_value=SimpleNamespace(draft_cls=_draft_cls)),
        patch.object(
            mod.TrainEagle3Recipe,
            "_setup_online_target",
            return_value=(torch.arange(16), torch.ones(128, dtype=torch.bool)),
        ),
    ):
        mock_auto_config.from_pretrained.return_value = _tiny_target_config()
        mock_tok.from_pretrained.return_value = MagicMock()
        with pytest.raises(_DraftBuilt) as excinfo:
            recipe.setup()
    return excinfo.value.config


@pytest.mark.parametrize("fc_norm, norm_output", [(True, False), (False, True)])
def test_recipe_args_eagle31_flags_reach_draft_config(fc_norm, norm_output):
    """Each toggle must land on the draft config exactly as set in
    ``recipe_args``, independently of the other (no cross-wiring)."""
    config = _run_setup_to_draft_build({"fc_norm": fc_norm, "norm_output": norm_output})

    assert config.fc_norm is fc_norm
    assert config.norm_output is norm_output


def test_eagle31_flags_absent_default_off():
    """Configs that never mention the toggles keep the plain EAGLE-3 draft:
    both flags must be explicitly False (not merely missing) so the saved
    ``config.json`` states the architecture unambiguously."""
    config = _run_setup_to_draft_build({})

    assert config.fc_norm is False
    assert config.norm_output is False


def test_eagle31_flags_build_the_eagle31_modules():
    """End-to-end through the recipe assembly: the flags must survive
    ``to_dict()`` (what ``save_pretrained`` writes to the serve-time
    ``config.json``), and a draft built from the recipe-produced config must
    register the per-chunk ``model.fc_norm`` RMSNorms (the observable
    EAGLE-3.1 architecture change)."""
    config = _run_setup_to_draft_build({"fc_norm": True, "norm_output": True})

    serialized = config.to_dict()
    assert serialized["fc_norm"] is True
    assert serialized["norm_output"] is True

    draft = LlamaEagle3DraftModel(config)
    assert any(k.startswith("model.fc_norm.") for k in draft.state_dict())

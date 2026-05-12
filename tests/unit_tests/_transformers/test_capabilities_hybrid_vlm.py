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

"""Tests for ``_is_hybrid`` drilling into VLM wrapper's ``language_model.config``.

The capability gate originally checked only the outer ``model.config`` for
hybrid attributes (e.g. NemotronH's ``layers_block_type``). VLM wrappers like
NemotronOmniForConditionalGeneration carry the LLM as a sub-attribute, so the
hybrid markers live on ``model.language_model.config``. Without this drill-down,
``validate_for_mesh`` rejected ``cp_size>1`` + ``attn=sdpa`` for Nemotron-Omni
with the misleading "requires the TE attention backend" error.
"""

from __future__ import annotations

from types import SimpleNamespace

from nemo_automodel._transformers.capabilities import _is_hybrid


class _DummyModel:
    def __init__(self, config=None, language_model=None):
        self.config = config
        if language_model is not None:
            self.language_model = language_model


def test_is_hybrid_via_outer_config_layers_block_type():
    """Plain hybrid LLM with NemotronH-style attribute on outer config."""
    cfg = SimpleNamespace(layers_block_type=["M", "*", "M", "-", "M"])
    model = _DummyModel(config=cfg)
    assert _is_hybrid(model) is True


def test_is_hybrid_via_outer_config_hybrid_override_pattern():
    """HF hybrid models expose ``hybrid_override_pattern``."""
    cfg = SimpleNamespace(hybrid_override_pattern="M*M-M")
    model = _DummyModel(config=cfg)
    assert _is_hybrid(model) is True


def test_is_hybrid_via_outer_config_is_hybrid_model_flag():
    cfg = SimpleNamespace(is_hybrid_model=True)
    model = _DummyModel(config=cfg)
    assert _is_hybrid(model) is True


def test_is_hybrid_drills_into_language_model_config():
    """VLM wrapper: outer config has no hybrid markers but language_model.config
    does. ``_is_hybrid`` must drill in and return True."""
    outer_cfg = SimpleNamespace()  # no hybrid attrs
    inner_cfg = SimpleNamespace(layers_block_type=["M", "*", "M"])
    inner_lm = SimpleNamespace(config=inner_cfg)
    model = _DummyModel(config=outer_cfg, language_model=inner_lm)
    assert _is_hybrid(model) is True


def test_is_hybrid_drills_into_language_model_via_hybrid_override_pattern():
    outer_cfg = SimpleNamespace()
    inner_cfg = SimpleNamespace(hybrid_override_pattern="M*M")
    inner_lm = SimpleNamespace(config=inner_cfg)
    model = _DummyModel(config=outer_cfg, language_model=inner_lm)
    assert _is_hybrid(model) is True


def test_is_hybrid_drills_into_language_model_via_is_hybrid_flag():
    outer_cfg = SimpleNamespace()
    inner_cfg = SimpleNamespace(is_hybrid_model=True)
    inner_lm = SimpleNamespace(config=inner_cfg)
    model = _DummyModel(config=outer_cfg, language_model=inner_lm)
    assert _is_hybrid(model) is True


def test_not_hybrid_when_neither_config_has_marker():
    """Plain non-hybrid VLM (e.g. Gemma4 dense): outer + inner both lack
    hybrid markers. Must return False."""
    outer_cfg = SimpleNamespace()
    inner_cfg = SimpleNamespace()
    inner_lm = SimpleNamespace(config=inner_cfg)
    model = _DummyModel(config=outer_cfg, language_model=inner_lm)
    assert _is_hybrid(model) is False


def test_not_hybrid_when_pattern_has_no_M():
    """``layers_block_type`` exists but contains only attention layers ("*"):
    not hybrid (no Mamba/SSM)."""
    cfg = SimpleNamespace(layers_block_type=["*", "*", "*"])
    model = _DummyModel(config=cfg)
    assert _is_hybrid(model) is False


def test_is_hybrid_handles_none_config():
    """Models without ``.config`` should not crash."""
    model = SimpleNamespace()  # no .config attr at all
    assert _is_hybrid(model) is False


def test_is_hybrid_handles_empty_pattern():
    """Empty pattern is not hybrid (avoids any('M' in '') edge case)."""
    cfg = SimpleNamespace(layers_block_type=[])
    model = _DummyModel(config=cfg)
    assert _is_hybrid(model) is False


def test_is_hybrid_outer_takes_priority_when_marked():
    """If outer is marked, no need to look at inner — short-circuit True."""
    outer_cfg = SimpleNamespace(layers_block_type=["M", "*"])
    # inner has nothing — outer alone should be enough
    inner_lm = SimpleNamespace(config=SimpleNamespace())
    model = _DummyModel(config=outer_cfg, language_model=inner_lm)
    assert _is_hybrid(model) is True


def test_is_hybrid_case_insensitive_M_match():
    """Pattern matching uses ``str.upper()`` so lowercase 'm' counts."""
    cfg = SimpleNamespace(layers_block_type=["m", "*", "m"])
    model = _DummyModel(config=cfg)
    assert _is_hybrid(model) is True


def test_is_hybrid_inner_language_model_with_no_config():
    """``language_model`` exists but its ``config`` is None — must not crash."""
    outer_cfg = SimpleNamespace()
    inner_lm = SimpleNamespace(config=None)
    model = _DummyModel(config=outer_cfg, language_model=inner_lm)
    assert _is_hybrid(model) is False

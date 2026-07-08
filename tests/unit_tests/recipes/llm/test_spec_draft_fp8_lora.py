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

"""Unit tests for FP8 draft training and LoRA draft adaptation in the spec-decode recipes.

Covers the shared helpers in ``_spec_train_utils`` (``apply_draft_fp8``,
``raise_if_peft_configured``), the EAGLE-3 recipe's ``_load_draft_weights`` /
``_apply_draft_peft_and_fp8`` wiring, the checkpointer ``is_peft`` flip, and
DSpark's post-step FP8 scale precompute hook.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.recipes.llm._spec_train_utils import (
    apply_draft_compile,
    apply_draft_fp8,
    raise_if_peft_configured,
)
from nemo_automodel.recipes.llm.train_dspark import TrainDSparkRecipe
from nemo_automodel.recipes.llm.train_eagle3 import (
    _apply_draft_peft_and_fp8,
    _load_draft_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.fc = nn.Linear(8, 8)


class _Cfg:
    """Duck-typed recipe config node: dict-backed ``get``."""

    def __init__(self, d=None):
        self._d = d or {}

    def get(self, key, default=None):
        return self._d.get(key, default)


def _peft_node(**overrides):
    kwargs = dict(target_modules=["q_proj"], dim=4, use_triton=False, use_memory_efficient_lora=False)
    kwargs.update(overrides)
    return SimpleNamespace(instantiate=lambda: PeftConfig(**kwargs))


# ---------------------------------------------------------------------------
# apply_draft_fp8
# ---------------------------------------------------------------------------


def test_apply_draft_fp8_none_cfg_is_noop():
    with patch("nemo_automodel.recipes.llm._spec_train_utils.apply_fp8_to_model") as mock_apply:
        apply_draft_fp8(_TinyDraft(), None)
    mock_apply.assert_not_called()


def test_apply_draft_fp8_disabled_is_noop():
    model = _TinyDraft()
    apply_draft_fp8(model, {"enabled": False, "recipe_name": "tensorwise"})
    assert type(model.q_proj) is nn.Linear


def test_apply_draft_fp8_enabled_builds_config_and_converts():
    model = _TinyDraft()
    with patch("nemo_automodel.recipes.llm._spec_train_utils.apply_fp8_to_model") as mock_apply:
        cfg_fp8 = {"enabled": True, "recipe_name": "tensorwise", "filter_fqns": ["lm_head"], "emulate": True}
        apply_draft_fp8(model, cfg_fp8)
    mock_apply.assert_called_once()
    assert mock_apply.call_args.args[0] is model
    fp8_config = mock_apply.call_args.kwargs["config"]
    assert fp8_config.enabled is True
    assert fp8_config.recipe_name == "tensorwise"
    assert fp8_config.filter_fqns == ["lm_head"]
    assert fp8_config.emulate is True


# ---------------------------------------------------------------------------
# apply_draft_compile
# ---------------------------------------------------------------------------


def test_apply_draft_compile_none_cfg_is_noop():
    with patch("nemo_automodel.recipes.llm._spec_train_utils.compile_module_inplace") as mock_compile:
        apply_draft_compile(_TinyDraft(), None)
    mock_compile.assert_not_called()


def test_apply_draft_compile_disabled_is_noop():
    model = _TinyDraft()
    apply_draft_compile(model, {"enabled": False})
    assert getattr(model, "_compiled_call_impl", None) is None


def test_apply_draft_compile_enabled_compiles_in_place():
    model = _TinyDraft()
    apply_draft_compile(model, {"enabled": True, "mode": "default"})
    assert model._compiled_call_impl is not None


# ---------------------------------------------------------------------------
# raise_if_peft_configured
# ---------------------------------------------------------------------------


def test_raise_if_peft_configured_passes_without_peft():
    raise_if_peft_configured(_Cfg(), "TrainDFlashRecipe")


def test_raise_if_peft_configured_rejects_peft():
    with pytest.raises(ValueError, match="TrainDSparkRecipe"):
        raise_if_peft_configured(_Cfg({"peft": _peft_node()}), "TrainDSparkRecipe")


# ---------------------------------------------------------------------------
# _apply_draft_peft_and_fp8 (EAGLE-3)
# ---------------------------------------------------------------------------


def test_peft_and_fp8_absent_returns_model_unchanged():
    model = _TinyDraft()
    peft_config = _apply_draft_peft_and_fp8(model, _Cfg(), parallel_drafting=False)
    assert peft_config is None
    assert all(p.requires_grad for p in model.parameters())


def test_peft_rejected_with_parallel_drafting():
    cfg = _Cfg({"peft": _peft_node()})
    with pytest.raises(ValueError, match="parallel_drafting"):
        _apply_draft_peft_and_fp8(_TinyDraft(), cfg, parallel_drafting=True)


def test_peft_rejected_with_fp8():
    cfg = _Cfg({"peft": _peft_node(), "fp8": {"enabled": True}})
    with pytest.raises(ValueError, match="cannot be combined"):
        _apply_draft_peft_and_fp8(_TinyDraft(), cfg, parallel_drafting=False)


def test_peft_allows_disabled_fp8_block():
    cfg = _Cfg({"peft": _peft_node(), "fp8": {"enabled": False}})
    peft_config = _apply_draft_peft_and_fp8(_TinyDraft(), cfg, parallel_drafting=False)
    assert peft_config is not None


def test_peft_applies_lora_and_freezes_base():
    model = _TinyDraft()
    peft_config = _apply_draft_peft_and_fp8(model, _Cfg({"peft": _peft_node()}), parallel_drafting=False)
    assert isinstance(peft_config, PeftConfig)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable, "LoRA adapters must be trainable"
    assert all("lora_" in name for name in trainable)
    # The un-matched linear stays frozen entirely.
    assert not model.fc.weight.requires_grad
    assert not model.q_proj.weight.requires_grad


def test_peft_no_matched_modules_raises():
    cfg = _Cfg({"peft": _peft_node(target_modules=["zz_proj"])})
    with pytest.raises(ValueError, match="matched no draft modules"):
        _apply_draft_peft_and_fp8(_TinyDraft(), cfg, parallel_drafting=False)


# ---------------------------------------------------------------------------
# _load_draft_weights
# ---------------------------------------------------------------------------


def _save_tiny_state(path: Path, state_dict):
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(path))


def test_load_draft_weights_from_directory(tmp_path):
    src = _TinyDraft()
    _save_tiny_state(tmp_path / "model.safetensors", src.state_dict())
    dst = _TinyDraft()
    _load_draft_weights(dst, str(tmp_path))
    torch.testing.assert_close(dst.q_proj.weight, src.q_proj.weight)
    torch.testing.assert_close(dst.fc.bias, src.fc.bias)


def test_load_draft_weights_from_single_file(tmp_path):
    src = _TinyDraft()
    file_path = tmp_path / "draft.safetensors"
    _save_tiny_state(file_path, src.state_dict())
    dst = _TinyDraft()
    _load_draft_weights(dst, str(file_path))
    torch.testing.assert_close(dst.q_proj.weight, src.q_proj.weight)


def test_load_draft_weights_partial_overlap_warns_but_loads(tmp_path, caplog):
    src = _TinyDraft()
    state = dict(src.state_dict())
    state.pop("fc.weight")  # missing on disk
    state["extra.weight"] = torch.zeros(2, 2)  # unexpected on disk
    _save_tiny_state(tmp_path / "model.safetensors", state)
    dst = _TinyDraft()
    with caplog.at_level("WARNING"):
        _load_draft_weights(dst, str(tmp_path))
    torch.testing.assert_close(dst.q_proj.weight, src.q_proj.weight)
    assert any("not found" in r.message for r in caplog.records)
    assert any("unused" in r.message for r in caplog.records)


def test_load_draft_weights_no_key_overlap_raises(tmp_path):
    _save_tiny_state(tmp_path / "model.safetensors", {"foreign.weight": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="shares no keys"):
        _load_draft_weights(_TinyDraft(), str(tmp_path))


def test_load_draft_weights_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_draft_weights(_TinyDraft(), str(tmp_path))


# ---------------------------------------------------------------------------
# DSpark _maybe_precompute_fp8_scales
# ---------------------------------------------------------------------------


def _bare_dspark_recipe():
    recipe = TrainDSparkRecipe.__new__(TrainDSparkRecipe)
    recipe.trainer_module = _TinyDraft()
    return recipe


def test_dspark_precompute_noop_when_disabled():
    recipe = _bare_dspark_recipe()
    recipe._precompute_fp8_scales = False
    with patch("nemo_automodel.recipes.llm.train_dspark.precompute_float8_dynamic_scale_for_fsdp") as mock_precompute:
        recipe._maybe_precompute_fp8_scales()
    mock_precompute.assert_not_called()


def test_dspark_precompute_noop_when_flag_unset():
    recipe = _bare_dspark_recipe()
    with patch("nemo_automodel.recipes.llm.train_dspark.precompute_float8_dynamic_scale_for_fsdp") as mock_precompute:
        recipe._maybe_precompute_fp8_scales()
    mock_precompute.assert_not_called()


def test_dspark_precompute_runs_when_enabled():
    recipe = _bare_dspark_recipe()
    recipe._precompute_fp8_scales = True
    with patch("nemo_automodel.recipes.llm.train_dspark.precompute_float8_dynamic_scale_for_fsdp") as mock_precompute:
        recipe._maybe_precompute_fp8_scales()
    mock_precompute.assert_called_once_with(recipe.trainer_module)

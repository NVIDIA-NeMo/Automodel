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


def test_raise_if_peft_configured_rejects_draft_weights_path():
    cfg = _Cfg({"recipe_args": _Cfg({"draft_weights_path": "/some/draft"})})
    with pytest.raises(ValueError, match="draft_weights_path"):
        raise_if_peft_configured(cfg, "TrainDFlashRecipe")


def test_raise_if_peft_configured_allows_empty_recipe_args():
    raise_if_peft_configured(_Cfg({"recipe_args": _Cfg()}), "TrainDFlashRecipe")


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


def test_peft_rejected_with_trainable_embeddings():
    """An explicit freeze_embeddings: false cannot be honored under the LoRA global freeze."""
    cfg = _Cfg({"peft": _peft_node()})
    with pytest.raises(ValueError, match="freeze_embeddings"):
        _apply_draft_peft_and_fp8(_TinyDraft(), cfg, parallel_drafting=False, freeze_embeddings=False)


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


def test_load_draft_weights_missing_keys_raise(tmp_path):
    """An incomplete warm-start checkpoint must be a hard error, not a warning."""
    src = _TinyDraft()
    state = dict(src.state_dict())
    state.pop("fc.weight")  # missing on disk
    _save_tiny_state(tmp_path / "model.safetensors", state)
    with pytest.raises(ValueError, match="missing"):
        _load_draft_weights(_TinyDraft(), str(tmp_path))


def test_load_draft_weights_unexpected_keys_warn_but_load(tmp_path, caplog):
    src = _TinyDraft()
    state = dict(src.state_dict())
    state["extra.weight"] = torch.zeros(2, 2)  # unexpected on disk
    _save_tiny_state(tmp_path / "model.safetensors", state)
    dst = _TinyDraft()
    with caplog.at_level("WARNING"):
        _load_draft_weights(dst, str(tmp_path))
    torch.testing.assert_close(dst.q_proj.weight, src.q_proj.weight)
    assert any("unused" in r.message for r in caplog.records)


def test_load_draft_weights_no_key_overlap_raises(tmp_path):
    _save_tiny_state(tmp_path / "model.safetensors", {"foreign.weight": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match="shares no keys"):
        _load_draft_weights(_TinyDraft(), str(tmp_path))


def test_load_draft_weights_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_draft_weights(_TinyDraft(), str(tmp_path))


# ---------------------------------------------------------------------------
# _load_draft_weights: draft-vocab mapping guard
# ---------------------------------------------------------------------------


class _TinyMappedDraft(nn.Module):
    """Tiny draft stand-in carrying the persistent d2t/t2d vocab-mapping buffers."""

    def __init__(self, d2t: torch.Tensor):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", torch.zeros(16, dtype=torch.bool))


def test_load_draft_weights_accepts_matching_vocab_mapping(tmp_path):
    d2t = torch.arange(4, dtype=torch.long)
    src = _TinyMappedDraft(d2t)
    _save_tiny_state(tmp_path / "model.safetensors", src.state_dict())
    dst = _TinyMappedDraft(d2t.clone())
    _load_draft_weights(dst, str(tmp_path))
    torch.testing.assert_close(dst.q_proj.weight, src.q_proj.weight)


def test_load_draft_weights_rejects_mismatched_vocab_mapping(tmp_path):
    """A checkpoint trained for a different draft-vocab mapping must fail, not silently overwrite."""
    src = _TinyMappedDraft(torch.arange(4, dtype=torch.long))
    _save_tiny_state(tmp_path / "model.safetensors", src.state_dict())
    dst = _TinyMappedDraft(torch.arange(4, dtype=torch.long) + 7)
    with pytest.raises(ValueError, match="different draft-vocab mapping"):
        _load_draft_weights(dst, str(tmp_path))


def test_load_draft_weights_rejects_one_sided_mapping(tmp_path):
    """Checkpoint has mapping buffers but the current draft does not (or vice versa): reject."""
    src = _TinyMappedDraft(torch.arange(4, dtype=torch.long))
    _save_tiny_state(tmp_path / "model.safetensors", src.state_dict())
    with pytest.raises(ValueError, match="different draft-vocab mapping"):
        _load_draft_weights(_TinyDraft(), str(tmp_path))


# ---------------------------------------------------------------------------
# save_checkpoint forwards peft_config; adapter-only save round-trip
# ---------------------------------------------------------------------------


def _saving_recipe(tmp_path):
    """A TrainEagle3Recipe stub with just enough state to run the real save_checkpoint."""
    from unittest.mock import MagicMock

    from nemo_automodel.recipes.llm.train_eagle3 import TrainEagle3Recipe

    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.checkpointer = MagicMock()
    recipe.checkpointer.config = SimpleNamespace(enabled=True, is_async=False)
    recipe.checkpoint_config = SimpleNamespace(checkpoint_dir=str(tmp_path))
    recipe.tokenizer = object()
    recipe.optimizer = MagicMock()
    recipe.lr_scheduler = MagicMock()
    recipe.rng = MagicMock()
    recipe._module = lambda: SimpleNamespace(draft_model=nn.Linear(2, 2))
    recipe._save_extra_state = lambda path, epoch: None
    recipe._update_latest_symlink = lambda path: None
    recipe.cfg = SimpleNamespace()
    return recipe


def test_save_checkpoint_forwards_peft_config(tmp_path):
    """Regression: save_model must receive the recipe's PeftConfig or the PeftAddon crashes on None."""
    recipe = _saving_recipe(tmp_path)
    recipe.peft_config = PeftConfig(target_modules=["q_proj"])
    recipe.save_checkpoint(epoch=0, step=1)
    assert recipe.checkpointer.save_model.call_args.kwargs["peft_config"] is recipe.peft_config


def test_save_checkpoint_without_peft_forwards_none(tmp_path):
    recipe = _saving_recipe(tmp_path)
    recipe.save_checkpoint(epoch=0, step=1)
    assert recipe.checkpointer.save_model.call_args.kwargs["peft_config"] is None


def test_merged_lora_state_dict_folds_adapters():
    """Merged weights equal base + (alpha/dim) * B @ A, with lora_* keys dropped."""
    from nemo_automodel.recipes.llm.train_eagle3 import _merged_lora_state_dict

    model = _TinyDraft()
    base_weight = model.q_proj.weight.detach().clone()
    _apply_draft_peft_and_fp8(model, _Cfg({"peft": _peft_node()}), parallel_drafting=False)
    # lora_B initializes to zeros (merge would be a no-op); set both factors explicitly.
    with torch.no_grad():
        model.q_proj.lora_A.weight.normal_()
        model.q_proj.lora_B.weight.normal_()

    merged = _merged_lora_state_dict(model)

    assert all("lora_" not in k for k in merged)
    expected = base_weight + (model.q_proj.lora_B.weight @ model.q_proj.lora_A.weight) * model.q_proj.scale
    torch.testing.assert_close(merged["q_proj.weight"], expected)
    torch.testing.assert_close(merged["fc.weight"], model.fc.weight)


def test_save_checkpoint_final_lora_exports_merged_draft(tmp_path):
    """The final checkpoint of a LoRA run must leave a serve-ready merged export."""
    recipe = _saving_recipe(tmp_path)
    recipe.peft_config = PeftConfig(target_modules=["q_proj"])
    with patch("nemo_automodel.recipes.llm.train_eagle3._export_merged_lora_draft") as mock_export:
        recipe.save_checkpoint(epoch=0, step=1, is_final_checkpoint=False)
        mock_export.assert_not_called()
        recipe.save_checkpoint(epoch=0, step=2, is_final_checkpoint=True)
        mock_export.assert_called_once()


def test_lora_draft_adapter_save_writes_adapter_files(tmp_path):
    """End to end: a LoRA-patched draft saved through the real Checkpointer writes adapter-only files."""
    import os

    from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig

    model = _TinyDraft()
    peft_config = _apply_draft_peft_and_fp8(model, _Cfg({"peft": _peft_node()}), parallel_drafting=False)
    checkpoint_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path),
        model_save_format="safetensors",
        model_repo_id="fake/tiny",
        model_cache_dir=str(tmp_path / "hf_cache"),
        save_consolidated=False,
        is_peft=True,
        model_state_dict_keys=list(model.state_dict().keys()),
    )
    checkpointer = checkpoint_config.build(dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)
    step_dir = tmp_path / "epoch_0_step_1"
    os.makedirs(step_dir)

    checkpointer.save_model(model, str(step_dir), peft_config=peft_config)

    adapter_file = step_dir / "model" / "adapter_model.safetensors"
    assert adapter_file.exists()
    from safetensors.torch import load_file

    adapter_keys = list(load_file(str(adapter_file)).keys())
    assert adapter_keys and all("lora_" in k for k in adapter_keys)


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

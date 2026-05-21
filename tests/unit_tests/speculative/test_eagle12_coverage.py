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

"""Additional unit tests for EAGLE-1/2 coverage: draft helpers, target accessors,
recipe utilities, trainer module edge cases, and entrypoint validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.speculative.eagle.core_v12 import EagleStepMetrics, EagleTrainerModule
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import (
    LlamaEagleDraftModel,
    _build_causal_mask,
)
from nemo_automodel.components.speculative.eagle.target_v12 import (
    EagleTargetBatch,
    HFEagleTargetModel,
    _shift_left_with_zero,
)
from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe, _all_reduce_mean
from nemo_automodel.recipes.llm.train_eagle1 import main as eagle1_main
from nemo_automodel.recipes.llm.train_eagle2 import TrainEagle2Recipe
from nemo_automodel.recipes.llm.train_eagle2 import main as eagle2_main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_llama_config(**overrides) -> LlamaConfig:
    defaults = dict(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=32,
    )
    defaults.update(overrides)
    cfg = LlamaConfig(**defaults)
    cfg.torch_dtype = torch.float32
    return cfg


def _tiny_target() -> LlamaForCausalLM:
    return LlamaForCausalLM(_tiny_llama_config()).to(torch.float32).eval()


def _tiny_draft() -> LlamaEagleDraftModel:
    cfg = _tiny_llama_config()
    cfg.draft_num_hidden_layers = 1
    return LlamaEagleDraftModel(cfg).to(torch.float32)


# ---------------------------------------------------------------------------
# draft_llama_v12: _build_causal_mask
# ---------------------------------------------------------------------------


class TestBuildCausalMask:
    def test_shape_and_dtype(self):
        mask = torch.ones(2, 4, dtype=torch.long)
        result = _build_causal_mask(mask, torch.float32)
        assert result.shape == (2, 1, 4, 4)
        assert result.dtype == torch.float32

    def test_causal_upper_triangle_is_masked(self):
        mask = torch.ones(1, 3, dtype=torch.long)
        result = _build_causal_mask(mask, torch.float32)
        result = result.squeeze()
        assert result[0, 0].item() == 0.0
        assert result[0, 1] < -1e30
        assert result[1, 0].item() == 0.0
        assert result[1, 1].item() == 0.0
        assert result[1, 2] < -1e30

    def test_padding_positions_are_masked(self):
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
        result = _build_causal_mask(mask, torch.float32).squeeze()
        assert result[0, 2] < -1e30
        assert result[1, 2] < -1e30


# ---------------------------------------------------------------------------
# draft_llama_v12: copy_embeddings / freeze_embeddings / _repeat_kv
# ---------------------------------------------------------------------------


class TestDraftModelHelpers:
    def test_copy_embeddings_from_target(self):
        target = _tiny_target()
        draft = _tiny_draft()
        draft.copy_embeddings_from_target(target.get_input_embeddings())
        torch.testing.assert_close(
            draft.embed_tokens.weight.data,
            target.get_input_embeddings().weight.data,
        )

    def test_freeze_embeddings(self):
        draft = _tiny_draft()
        assert draft.embed_tokens.weight.requires_grad
        draft.freeze_embeddings()
        assert not draft.embed_tokens.weight.requires_grad

    def test_repeat_kv_noop_when_groups_equal_one(self):
        cfg = _tiny_llama_config(num_attention_heads=2, num_key_value_heads=2)
        cfg.draft_num_hidden_layers = 1
        draft = LlamaEagleDraftModel(cfg)
        attn = draft.layers[0].self_attn
        assert attn.num_key_value_groups == 1
        t = torch.randn(1, 2, 4, 8)
        out = attn._repeat_kv(t)
        assert out is t

    def test_repeat_kv_when_groups_greater_than_one(self):
        cfg = _tiny_llama_config(num_attention_heads=4, num_key_value_heads=2)
        cfg.draft_num_hidden_layers = 1
        draft = LlamaEagleDraftModel(cfg)
        attn = draft.layers[0].self_attn
        assert attn.num_key_value_groups == 2
        t = torch.randn(1, 2, 4, 8)
        out = attn._repeat_kv(t)
        assert out.shape == (1, 4, 4, 8)

    def test_draft_model_respects_draft_num_hidden_layers(self):
        cfg = _tiny_llama_config(num_hidden_layers=4)
        cfg.draft_num_hidden_layers = 2
        model = LlamaEagleDraftModel(cfg)
        assert len(model.layers) == 2


# ---------------------------------------------------------------------------
# target_v12: accessors
# ---------------------------------------------------------------------------


class TestTargetAccessors:
    def test_get_input_embeddings(self):
        target = _tiny_target()
        wrapper = HFEagleTargetModel(target)
        assert wrapper.get_input_embeddings() is target.get_input_embeddings()

    def test_get_lm_head(self):
        target = _tiny_target()
        wrapper = HFEagleTargetModel(target)
        assert wrapper.get_lm_head() is target.lm_head


# ---------------------------------------------------------------------------
# target_v12: _shift_left_with_zero edge cases
# ---------------------------------------------------------------------------


class TestShiftLeftWithZero:
    def test_single_element_sequence(self):
        t = torch.tensor([[42]])
        result = _shift_left_with_zero(t)
        assert result.shape == (1, 1)
        assert result[0, 0].item() == 0

    def test_3d_tensor(self):
        t = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = _shift_left_with_zero(t)
        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[0, 1], torch.tensor([0.0, 0.0]))


# ---------------------------------------------------------------------------
# target_v12: EagleTargetBatch dataclass
# ---------------------------------------------------------------------------


def test_eagle_target_batch_fields():
    dummy = torch.zeros(1)
    batch = EagleTargetBatch(
        input_hidden_states=dummy,
        target_hidden_states=dummy,
        target_logits=dummy,
        input_ids=dummy,
        attention_mask=dummy,
        loss_mask=dummy,
    )
    assert hasattr(batch, "input_hidden_states")
    assert hasattr(batch, "target_logits")


# ---------------------------------------------------------------------------
# core_v12: EagleStepMetrics dataclass
# ---------------------------------------------------------------------------


def test_eagle_step_metrics_fields():
    dummy = torch.zeros(1)
    m = EagleStepMetrics(loss=dummy, hidden_loss=dummy, token_loss=dummy, accuracy=dummy, valid_tokens=dummy)
    assert m.loss is dummy


# ---------------------------------------------------------------------------
# core_v12: compute_logits
# ---------------------------------------------------------------------------


def test_compute_logits_shape():
    target = _tiny_target()
    draft = _tiny_draft()
    trainer = EagleTrainerModule(draft, target_lm_head=target.lm_head)
    hidden = torch.randn(2, 4, 16)
    logits = trainer.compute_logits(hidden)
    assert logits.shape == (2, 4, 64)


# ---------------------------------------------------------------------------
# core_v12: forward with all-zero loss_mask (edge case)
# ---------------------------------------------------------------------------


def test_trainer_forward_with_zero_loss_mask():
    torch.manual_seed(0)
    target = _tiny_target()
    target.requires_grad_(False)
    draft = _tiny_draft()
    trainer = EagleTrainerModule(draft, target_lm_head=target.lm_head)

    bs, sl = 2, 4
    metrics = trainer(
        input_ids=torch.randint(0, 64, (bs, sl)),
        attention_mask=torch.ones(bs, sl, dtype=torch.long),
        loss_mask=torch.zeros(bs, sl, dtype=torch.long),
        input_hidden_states=torch.randn(bs, sl, 16),
        target_hidden_states=torch.randn(bs, sl, 16),
        target_logits=torch.randn(bs, sl, 64),
    )
    assert metrics.valid_tokens.item() == 0
    assert torch.isfinite(metrics.loss)


# ---------------------------------------------------------------------------
# core_v12: custom loss weights
# ---------------------------------------------------------------------------


def test_trainer_custom_loss_weights():
    torch.manual_seed(0)
    target = _tiny_target()
    target.requires_grad_(False)
    draft = _tiny_draft()

    trainer_h = EagleTrainerModule(draft, target_lm_head=target.lm_head, hidden_loss_weight=10.0, token_loss_weight=0.0)
    bs, sl = 1, 4
    kwargs = dict(
        input_ids=torch.randint(0, 64, (bs, sl)),
        attention_mask=torch.ones(bs, sl, dtype=torch.long),
        loss_mask=torch.ones(bs, sl, dtype=torch.long),
        input_hidden_states=torch.randn(bs, sl, 16),
        target_hidden_states=torch.randn(bs, sl, 16),
        target_logits=torch.randn(bs, sl, 64),
    )
    metrics = trainer_h(**kwargs)
    torch.testing.assert_close(metrics.loss, 10.0 * metrics.hidden_loss, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# train_eagle1: _all_reduce_mean (non-distributed path)
# ---------------------------------------------------------------------------


def test_all_reduce_mean_non_distributed():
    t = torch.tensor(4.0)
    result = _all_reduce_mean(t)
    assert result.item() == 4.0


# ---------------------------------------------------------------------------
# train_eagle1: main() / train_eagle2: main() ValueError guards
# ---------------------------------------------------------------------------


def test_eagle1_main_requires_config():
    with pytest.raises(ValueError, match="config_path is required"):
        eagle1_main(config_path=None)


def test_eagle2_main_requires_config():
    with pytest.raises(ValueError, match="config_path is required"):
        eagle2_main(config_path=None)


# ---------------------------------------------------------------------------
# train_eagle1: TrainEagle1Recipe construction
# ---------------------------------------------------------------------------


def test_eagle1_recipe_init():
    cfg = MagicMock()
    recipe = TrainEagle1Recipe(cfg)
    assert recipe.cfg is cfg


# ---------------------------------------------------------------------------
# train_eagle2: TrainEagle2Recipe is subclass of TrainEagle1Recipe
# ---------------------------------------------------------------------------


def test_eagle2_is_subclass():
    assert issubclass(TrainEagle2Recipe, TrainEagle1Recipe)


# ---------------------------------------------------------------------------
# train_eagle1: _module(), _save_checkpoint(), _run_eval() via manual wiring
# ---------------------------------------------------------------------------


class TestRecipeInternals:
    def _build_recipe(self, tmp_path):
        """Wire up a recipe with tiny models, no distributed init needed."""
        target = _tiny_target()
        target.requires_grad_(False)
        draft = _tiny_draft()

        trainer_module = EagleTrainerModule(draft, target_lm_head=target.lm_head)

        recipe = TrainEagle1Recipe.__new__(TrainEagle1Recipe)
        recipe.device = torch.device("cpu")
        recipe.dist_env = SimpleNamespace(is_main=True, world_size=1)
        recipe.trainer_module = trainer_module
        recipe.target_wrapper = HFEagleTargetModel(target)
        recipe.output_dir = tmp_path
        recipe.runtime = SimpleNamespace(global_step=5)
        recipe.grad_accumulation_steps = 1
        recipe.max_grad_norm = 1.0
        recipe.num_epochs = 1
        recipe.log_every_steps = 1
        recipe.peak_lr = 1e-4

        recipe.optimizer = torch.optim.AdamW([p for p in trainer_module.parameters() if p.requires_grad], lr=1e-4)
        recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(recipe.optimizer, lambda s: 1.0)

        return recipe

    def test_module_returns_unwrapped(self, tmp_path):
        recipe = self._build_recipe(tmp_path)
        assert recipe._module() is recipe.trainer_module

    def test_save_extra_state_round_trip(self, tmp_path):
        """``_save_extra_state`` persists ``global_step`` and ``epoch``; the
        matching ``_load_extra_state`` restores them. Full ``save_checkpoint``
        (model + optimizer via DCP) is covered separately by the dedicated
        checkpoint-resume test file."""
        recipe = self._build_recipe(tmp_path)
        recipe.runtime.global_step = 11
        save_dir = tmp_path / "extra"
        save_dir.mkdir()
        recipe._save_extra_state(str(save_dir), epoch=2)
        assert (save_dir / "eagle_meta.pt").exists()

        fresh = self._build_recipe(tmp_path)
        fresh._resume_epoch = 0
        fresh.runtime.global_step = 0
        fresh._load_extra_state(str(save_dir))
        assert fresh.runtime.global_step == 11
        assert fresh._resume_epoch == 2

    def test_run_eval_returns_none_without_val_loader(self, tmp_path):
        recipe = self._build_recipe(tmp_path)
        recipe.val_dataloader = None
        assert recipe._run_eval() is None

    def test_run_eval_returns_metrics_with_val_loader(self, tmp_path):
        recipe = self._build_recipe(tmp_path)
        bs, sl = 2, 6
        val_data = TensorDataset(
            torch.randint(0, 64, (4, sl)),
            torch.ones(4, sl, dtype=torch.long),
            torch.ones(4, sl, dtype=torch.long),
        )
        val_loader = DataLoader(val_data, batch_size=bs)

        class _KeyedLoader:
            def __init__(self, loader):
                self._loader = loader

            def __iter__(self):
                for ids, attn, lm in self._loader:
                    yield {"input_ids": ids, "attention_mask": attn, "loss_mask": lm}

        recipe.val_dataloader = _KeyedLoader(val_loader)
        result = recipe._run_eval()
        assert result is not None
        assert "val_loss" in result
        assert "val_accuracy" in result

    def test_run_train_validation_loop_single_epoch(self, tmp_path):
        recipe = self._build_recipe(tmp_path)
        recipe.val_dataloader = None
        bs, sl = 2, 6
        train_data = TensorDataset(
            torch.randint(0, 64, (4, sl)),
            torch.ones(4, sl, dtype=torch.long),
            torch.ones(4, sl, dtype=torch.long),
        )
        train_loader = DataLoader(train_data, batch_size=bs)

        class _KeyedLoader:
            def __init__(self, loader):
                self._loader = loader

            def __iter__(self):
                for ids, attn, lm in self._loader:
                    yield {"input_ids": ids, "attention_mask": attn, "loss_mask": lm}

            def __len__(self):
                return len(self._loader)

        recipe.train_dataloader = _KeyedLoader(train_loader)
        recipe.num_epochs = 1
        recipe.log_every_steps = 1
        recipe.run_train_validation_loop()
        assert recipe.runtime.global_step >= 1
        # End-of-epoch save is a no-op here because the test fixture skips
        # ``setup()`` and never builds a checkpointer; the dedicated
        # ``test_eagle_checkpoint_resume`` file exercises the save/load path.

    def test_run_train_loop_with_grad_accumulation(self, tmp_path):
        recipe = self._build_recipe(tmp_path)
        recipe.val_dataloader = None
        recipe.grad_accumulation_steps = 3
        bs, sl = 1, 6
        train_data = TensorDataset(
            torch.randint(0, 64, (5, sl)),
            torch.ones(5, sl, dtype=torch.long),
            torch.ones(5, sl, dtype=torch.long),
        )
        train_loader = DataLoader(train_data, batch_size=bs)

        class _KeyedLoader:
            def __init__(self, loader):
                self._loader = loader

            def __iter__(self):
                for ids, attn, lm in self._loader:
                    yield {"input_ids": ids, "attention_mask": attn, "loss_mask": lm}

            def __len__(self):
                return len(self._loader)

        recipe.train_dataloader = _KeyedLoader(train_loader)
        recipe.num_epochs = 1
        initial_step = recipe.runtime.global_step
        recipe.run_train_validation_loop()
        assert recipe.runtime.global_step - initial_step == 2

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

"""Tests for the DSpine recipe seams over the DFlash recipe.

The recipe is constructed via ``__new__`` (bypassing ``setup()``), so only the
overridden seams are exercised: the draft-class swap, the ``dflash_config``
extension, the target-logit capture, the codebook initialization and trainer
swap, the step schedule, and the extra-metrics plumbing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from nemo_automodel.components.speculative.dflash.draft_qwen3_dspine import Qwen3DSpineDraftModel
from nemo_automodel.components.speculative.dflash.dspine_core import (
    DSpineSchedule,
    DSpineStepMetrics,
    DSpineTrainerModule,
)
from nemo_automodel.components.speculative.dflash.registry import resolve_dflash_draft_spec
from nemo_automodel.recipes.llm.train_dspine import TrainDSpineRecipe
from tests.unit_tests.speculative.test_dspine_draft import BLOCK_SIZE, HIDDEN, MASK_ID, VOCAB, draft_cfg

TARGET_LAYER_IDS = [1, 9, 17, 25, 33]

# ``dflash_config`` of the released tencent/Qwen3-8B-DSpine-b16 drafter, minus the
# ``selected_prefix_code_bridge.kappa`` entry that no loader reads.
RELEASED_DSPINE_KEYS = {
    "draft_block_causal": True,
    "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-06, "rank": 512},
    "selected_prefix_decode_mode": "serial",
    "selected_prefix_feature_gt_training": True,
    "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
    "selected_prefix_feature_teacher_layers": [1, 2, 3],
    "selected_prefix_feature_teacher_schedule": {
        "hold_fraction": 0.16666666666666666,
        "kind": "block_bernoulli",
        "peak_probability": 0.5,
        "zero_fraction": 0.3333333333333333,
    },
    "selected_prefix_feature_write_mode": "independent",
    "selected_prefix_feature_writes": True,
    "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
    "selected_prefix_terminal": False,
}


def _recipe():
    recipe = TrainDSpineRecipe.__new__(TrainDSpineRecipe)
    recipe.mask_token_id = MASK_ID
    recipe.block_size = BLOCK_SIZE
    return recipe


def _recipe_with_draft(head=None):
    recipe = _recipe()
    torch.manual_seed(0)
    recipe.draft_model = Qwen3DSpineDraftModel(draft_cfg())
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    if head is not None:
        with torch.no_grad():
            lm_head.weight.copy_(head)
    recipe.target_model = SimpleNamespace(
        get_output_embeddings=lambda: lm_head,
        get_input_embeddings=lambda: torch.nn.Embedding(VOCAB, HIDDEN),
    )
    recipe.draft_sliding_window = None
    return recipe


def test_draft_cls_selects_the_dspine_draft():
    spec = resolve_dflash_draft_spec(["Qwen3ForCausalLM"])
    assert _recipe()._draft_cls(spec) is Qwen3DSpineDraftModel


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3MoeForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
def test_draft_cls_selects_the_dspine_draft_for_every_qwen3_family_target(architecture):
    assert _recipe()._draft_cls(resolve_dflash_draft_spec([architecture])) is Qwen3DSpineDraftModel


def test_draft_cls_rejects_a_family_without_a_dspine_draft():
    with pytest.raises(ValueError, match="No DSpine draft"):
        _recipe()._draft_cls(resolve_dflash_draft_spec(["KimiK3ForCausalLM"]))


def test_qwen3_5_target_draft_config_matches_the_released_qwen3_8_27b_drafter():
    """A Qwen3.5-family target (Qwen3.8-27B's) must yield a plain, causal Qwen3 draft.

    The released tencent/Qwen3.8-27B-DSpine-b16 config is ``model_type: qwen3`` with
    ``is_causal: true``, sliding-attention layers, a draft attention shape of its
    own, and no partial-rotary fields -- SGLang's DSPINE loader rejects
    ``partial_rotary_factor != 1``.
    """
    recipe = _recipe()
    recipe.block_size = 16
    recipe.mask_token_id = 248070
    recipe.draft_sliding_window = 8192
    recipe_cfg = {
        "draft_num_hidden_layers": 5,
        "draft_num_attention_heads": 32,
        "draft_num_key_value_heads": 8,
        "draft_head_dim": 128,
    }
    target = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        partial_rotary_factor=0.25,
    )
    built = recipe._build_qwen3_draft_config(
        recipe_cfg,
        target_text_config=target,
        draft_cls=Qwen3DSpineDraftModel,
        draft_num_hidden_layers=5,
        num_target_layers=64,
        target_layer_ids=[1, 16, 31, 46, 61],
        attention_backend="sdpa",
    )
    saved = built.to_dict()
    assert saved["model_type"] == "qwen3"
    assert saved["is_causal"] is True
    assert (saved["num_attention_heads"], saved["num_key_value_heads"], saved["head_dim"]) == (32, 8, 128)
    assert saved["layer_types"] == ["sliding_attention"] * 5 and saved["sliding_window"] == 8192
    assert "partial_rotary_factor" not in saved
    assert "partial_rotary_factor" not in (saved.get("rope_parameters") or {})
    assert saved["dflash_config"]["draft_block_causal"] is True

    built.dflash_config["selected_prefix_code_bridge"]["code_size"] = 64
    draft = Qwen3DSpineDraftModel(built)
    assert built.architectures == ["DSpineDraftModel"]
    assert draft.config.is_causal is True


def test_dflash_config_warns_on_a_message_size_the_serving_kernels_cannot_run(caplog):
    with caplog.at_level("WARNING"):
        _recipe()._build_dflash_config({"message_size": 192}, TARGET_LAYER_IDS)
    assert "multiple of 128" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        _recipe()._build_dflash_config({}, TARGET_LAYER_IDS)
    assert "multiple of 128" not in caplog.text


def test_default_dflash_config_reproduces_the_released_drafter():
    out = _recipe()._build_dflash_config({}, TARGET_LAYER_IDS)
    assert {key: out[key] for key in RELEASED_DSPINE_KEYS} == RELEASED_DSPINE_KEYS
    assert out["mask_token_id"] == MASK_ID
    assert out["target_layer_ids"] == TARGET_LAYER_IDS
    assert out["block_size"] == BLOCK_SIZE


def test_dflash_config_threads_the_recipe_knobs():
    out = _recipe()._build_dflash_config(
        {
            "draft_num_hidden_layers": 3,
            "code_size": 64,
            "message_size": 32,
            "refine_top_k": 8,
            "teacher_forcing_layers": 1,
            "teacher_forcing_probability": 0.25,
            "teacher_forcing_hold_fraction": 0.1,
            "teacher_forcing_zero_fraction": 0.2,
        },
        TARGET_LAYER_IDS,
    )
    assert out["selected_prefix_code_bridge"] == {"code_size": 64, "rank": 32, "eps": 1e-6}
    assert out["selected_prefix_feature_layers"] == [1, 2, 3]
    assert out["selected_prefix_feature_teacher_layers"] == [1]
    assert out["selected_prefix_last_write_refine"]["head_topk"] == 8
    assert out["selected_prefix_feature_teacher_schedule"] == {
        "kind": "block_bernoulli",
        "peak_probability": 0.25,
        "hold_fraction": 0.1,
        "zero_fraction": 0.2,
    }


def test_dflash_config_rejects_more_teacher_layers_than_draft_layers():
    with pytest.raises(ValueError, match="teacher_forcing_layers"):
        _recipe()._build_dflash_config({"draft_num_hidden_layers": 2, "teacher_forcing_layers": 3}, TARGET_LAYER_IDS)


def test_recipe_dflash_config_builds_a_dspine_draft():
    """The config the recipe stamps must be one the DSpine draft (and so the
    serving loader, which checks the same keys) accepts."""
    cfg = draft_cfg()
    recipe = _recipe()
    cfg.dflash_config = recipe._build_dflash_config(
        {"draft_num_hidden_layers": 3, "code_size": 16, "message_size": 8, "refine_top_k": 4}, [1, 3, 5]
    )
    draft = Qwen3DSpineDraftModel(cfg)
    assert draft.head_topk == 4
    assert draft.teacher_forcing_layers == {0, 1, 2}


def test_target_wrapper_captures_the_target_logits(monkeypatch):
    recipe = _recipe()
    recipe.target_model = object()
    captured = {}
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_dspine.HFDFlashTargetModel",
        lambda model, **kwargs: captured.update(model=model, **kwargs),
    )
    recipe._build_target_wrapper([1, 3])
    assert captured["capture_logits"] is True
    assert captured["target_layer_ids"] == [1, 3]


def test_build_trainer_module_initializes_the_codebook_from_the_target_head():
    head = torch.randn(VOCAB, HIDDEN)
    recipe = _recipe_with_draft(head)
    bridge = recipe.draft_model.selected_prefix_bridge
    assert not bridge.codebook.any()  # placeholder until the production initializer runs

    module = recipe._build_trainer_module(
        "sdpa", {"num_anchors": 7, "loss_decay_gamma": 5.0, "distribution_ce_weight": 0.2, "objective_chunk_blocks": 3}
    )

    assert isinstance(module, DSpineTrainerModule)
    assert (module.num_anchors, module.loss_decay_gamma) == (7, 5.0)
    assert (module.distribution_ce_weight, module.objective_chunk_blocks) == (0.2, 3)
    torch.testing.assert_close(bridge.codebook.norm(dim=-1), torch.ones(VOCAB))
    assert all(torch.isfinite(p).all() for p in recipe.draft_model.parameters())
    reference = type(bridge)(hidden_size=HIDDEN, num_layers=3, vocab_size=VOCAB, code_size=16, rank=8)
    reference.initialize_transfer_space(head)
    torch.testing.assert_close(bridge.codebook, reference.codebook)


def test_build_trainer_module_defaults():
    module = _recipe_with_draft()._build_trainer_module("sdpa", {})
    assert module.loss_decay_gamma == 7.0
    assert module.num_anchors == 512
    assert module.distribution_ce_weight == 0.1
    assert module.candidate_mass_threshold == 1e-4
    assert module.objective_chunk_blocks == 128


def test_build_trainer_module_rejects_a_dflash_loss_type():
    with pytest.raises(ValueError, match="loss_type"):
        _recipe_with_draft()._build_trainer_module("sdpa", {"loss_type": "dpace"})


def test_setup_builds_the_schedule_from_the_stamped_config(monkeypatch):
    recipe = _recipe_with_draft()
    recipe.draft_model.config.dflash_config = recipe._build_dflash_config({"draft_num_hidden_layers": 3}, [1, 3, 5])
    recipe.cfg = SimpleNamespace(recipe_args={"alignment_weight": 0.4})
    recipe.warmup_steps = 12
    recipe.total_optim_steps = 600
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_dflash.TrainDFlashRecipe.setup", lambda self: None)
    recipe.setup()
    assert recipe.dspine_schedule == DSpineSchedule(
        alignment_weight=0.4,
        alignment_warmup_steps=12,  # unset -> the LR warmup
        teacher_forcing_probability=0.5,
        teacher_forcing_hold_fraction=1 / 6,
        teacher_forcing_zero_fraction=1 / 3,
        total_steps=600,
    )
    assert recipe._last_dspine_metrics is None

    recipe.cfg = SimpleNamespace(recipe_args={"alignment_warmup_steps": 146})
    recipe.setup()
    assert recipe.dspine_schedule.alignment_warmup_steps == 146


def _metrics(**overrides):
    fields = dict(
        loss=torch.tensor(30.0),
        loss_weight=torch.tensor(2.0),
        accuracy=torch.tensor(0.5),
        valid_tokens=torch.tensor(10.0),
        correct_tokens=torch.tensor(5.0),
        accept_len=torch.tensor(3.5),
        accept_len_sum=torch.tensor(7.0),
        valid_blocks=torch.tensor(2.0),
        weight_sum=torch.tensor(8.0),
        backbone_ce_sum=torch.tensor(16.0),
        backbone_l1_sum=torch.tensor(4.0),
        refine_ce=torch.tensor(0.5),
        refine_ce_weight=torch.tensor(6.0),
        refine_l1=torch.tensor(0.25),
        refine_l1_weight=torch.tensor(4.0),
        alignment_sum=torch.tensor(2.0),
        candidate_hits=torch.tensor(9.0),
        base_correct_tokens=torch.tensor(4.0),
        base_accept_len=torch.tensor(3.0),
        base_accept_len_sum=torch.tensor(6.0),
    )
    fields.update(overrides)
    return DSpineStepMetrics(**fields)


def test_run_trainer_step_passes_logits_and_the_schedule():
    recipe = _recipe()
    recipe.runtime = SimpleNamespace(global_step=15)
    recipe.dspine_schedule = DSpineSchedule(
        alignment_weight=0.5,
        alignment_warmup_steps=10,
        teacher_forcing_probability=0.5,
        teacher_forcing_hold_fraction=1 / 6,
        teacher_forcing_zero_fraction=1 / 3,
        total_steps=60,
    )
    seen = {}
    recipe.trainer_module = lambda **kwargs: seen.update(kwargs) or _metrics()
    batch = SimpleNamespace(
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        hidden_states=torch.zeros(1, 4, 8),
        loss_mask=torch.ones(1, 4),
        logits=torch.zeros(1, 4, VOCAB),
        position_ids=None,
        seq_lens=None,
        doc_remaining=None,
    )
    out = recipe._run_trainer_step(batch)
    assert seen["target_logits"] is batch.logits
    assert seen["alignment_beta"] == 0.5
    assert seen["teacher_forcing"] == pytest.approx(0.25)
    assert recipe._last_dspine_metrics is out


def test_train_sums_are_window_averageable():
    values = _recipe()._extra_train_metric_sums(_metrics())
    assert values == {
        "train/accept_len": (pytest.approx(7.0), pytest.approx(2.0)),
        "train/backbone_ce": (pytest.approx(16.0), pytest.approx(8.0)),
        "train/backbone_l1": (pytest.approx(4.0), pytest.approx(8.0)),
        "train/refine_ce": (pytest.approx(3.0), pytest.approx(6.0)),
        "train/refine_l1": (pytest.approx(1.0), pytest.approx(4.0)),
        "train/alignment": (pytest.approx(2.0), pytest.approx(8.0)),
        "train/candidate_coverage": (pytest.approx(9.0), pytest.approx(10.0)),
        "train/base_accuracy": (pytest.approx(4.0), pytest.approx(10.0)),
        "train/base_accept_len": (pytest.approx(6.0), pytest.approx(2.0)),
    }


def test_run_eval_weights_each_term_by_its_own_denominator():
    recipe = _recipe()
    recipe.device = torch.device("cpu")
    recipe.trainer_module = torch.nn.Identity()
    recipe.target_wrapper = SimpleNamespace(generate_batch=lambda **kwargs: None)
    recipe.val_dataloader = [
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4),
            "loss_mask": torch.ones(1, 4),
        }
    ] * 2
    recipe._run_trainer_step = Mock(
        side_effect=[
            _metrics(loss=torch.tensor(30.0), refine_ce=torch.tensor(1.0), refine_ce_weight=torch.tensor(1.0)),
            _metrics(loss=torch.tensor(10.0), refine_ce=torch.tensor(0.0), refine_ce_weight=torch.tensor(0.0)),
        ]
    )
    result = recipe._run_eval()
    assert result["val_loss"] == pytest.approx(20.0)  # sum(numerator) / sum(sequences)
    assert result["val_refine_ce"] == pytest.approx(1.0)  # the empty batch adds no weight
    assert result["val_backbone_ce"] == pytest.approx(2.0)
    assert result["val_candidate_coverage"] == pytest.approx(0.9)
    assert result["val_base_accept_len"] == pytest.approx(3.0)
    assert recipe.trainer_module.training


def test_empty_eval_sums_have_stable_collective_order():
    recipe = _recipe()
    recipe.device = torch.device("cpu")
    sums = recipe._empty_extra_eval_metric_sums()
    assert list(sums) == list(recipe._extra_eval_metric_sums(_metrics()))
    assert all(n.item() == 0.0 and d.item() == 0.0 for n, d in sums.values())


def test_log_extra_train_metrics(caplog):
    recipe = _recipe()
    recipe._last_dspine_metrics = _metrics()
    with caplog.at_level("INFO"):
        recipe._log_extra_train_metrics(epoch_idx=0)
    assert "dspine:" in caplog.text
    assert "backbone_ce=2.0000" in caplog.text
    assert "candidate_coverage=0.9000" in caplog.text
    recipe._last_dspine_metrics = None
    recipe._log_extra_train_metrics(epoch_idx=0)  # no step yet: must not raise


def test_main_runs_setup_then_loop(monkeypatch):
    from nemo_automodel.recipes.llm import train_dspine

    calls = []
    monkeypatch.setattr(train_dspine, "parse_args_and_load_config", lambda p: SimpleNamespace())
    monkeypatch.setattr(TrainDSpineRecipe, "setup", lambda self: calls.append("setup"))
    monkeypatch.setattr(TrainDSpineRecipe, "run_train_validation_loop", lambda self: calls.append("loop"))
    train_dspine.main("cfg.yaml")
    assert calls == ["setup", "loop"]

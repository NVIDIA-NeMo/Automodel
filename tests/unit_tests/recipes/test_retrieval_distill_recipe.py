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
"""Unit tests for the embedding-distillation recipe helpers and setup guards."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from nemo_automodel._transformers.retrieval import RetrieverStudentWithProjection
from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.recipes.retrieval import distill_bi_encoder as mod
from nemo_automodel.recipes.retrieval.distill_bi_encoder import (
    _build_or_none,
    _cfg_get_path,
    _dp_group_src_rank,
    _move_to_device,
    _repeat_queries_to_match_docs,
    _split_grouped_passages,
    _unpack_qpn,
)
from nemo_automodel.recipes.retrieval.train_bi_encoder import (
    TrainBiEncoderRecipe,
    _configure_sentence_transformer_export,
)


class _DictLikeConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class _FakeEncoder(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.model = SimpleNamespace(config=self.config)
        self.name_or_path = ""
        self.source_model_path = None
        self.source_repository_path = None
        self.state_dict_adapter = EncoderStateDictAdapter()
        self.sentence_transformer_export_config = SimpleNamespace(query_prompt="", document_prompt="")

    def forward(self, inputs):
        return torch.zeros(inputs["input_ids"].shape[0], self.model.config.hidden_size)

    def configure_sentence_transformer_prompts(
        self,
        query_prompt: str,
        document_prompt: str,
        *,
        tokenizer: PreTrainedTokenizerBase | ProcessorMixin | None = None,
    ) -> None:
        self.sentence_transformer_export_config.query_prompt = query_prompt
        self.sentence_transformer_export_config.document_prompt = document_prompt
        self.prompt_tokenizer = tokenizer

    def disable_sentence_transformer_export(self):
        self.sentence_transformer_export_config = None


class _OptimizerGroupingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Module()
        self.vision_tower.linear = nn.Linear(4, 4)
        self.vision_tower.norm = nn.LayerNorm(4)
        self.multi_modal_projector = nn.Linear(4, 4)
        self.language_model = nn.Linear(4, 4)
        self.projection = nn.Linear(4, 6)


class _TrainableEncoder(_FakeEncoder):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__(hidden_size=4)
        self.embedding = nn.Embedding(8, 4, dtype=dtype)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pool tiny token embeddings for gradient comparisons.

        Args:
            inputs: input_ids tensor of shape [batch, sequence].

        Returns:
            Tensor of shape [batch, hidden].
        """
        return self.embedding(inputs["input_ids"]).mean(dim=1)


def _param_group_by_id(optimizer):
    return {id(param): group for group in optimizer.param_groups for param in group["params"]}


# ---------------------------------------------------------------------------
# RetrieverStudentWithProjection projection restore
# ---------------------------------------------------------------------------
@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_retriever_student_without_projection_matches_direct_training(dtype: torch.dtype) -> None:
    student = _TrainableEncoder(dtype)
    reference = deepcopy(student)
    model = RetrieverStudentWithProjection(student=student, teacher_hidden_size=4, use_projection=False)
    inputs = {"input_ids": torch.tensor([[0, 1, 2], [3, 4, 5]])}
    target = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 8

    recipe = mod.EmbeddingDistillRecipe.__new__(mod.EmbeddingDistillRecipe)
    recipe.cfg = _DictLikeConfig(optimizer={"lr": 0.1, "distill_projection_lr": 1.0})
    recipe.model_parts = [model]
    optimizer = torch.optim.SGD(recipe._build_optimizer_param_groups(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)

    pooled, distill_embeddings, captures = model(inputs)
    expected = reference(inputs).float()
    assert model.projection is None
    assert distill_embeddings is pooled
    assert distill_embeddings.dtype == torch.float32
    assert captures == {}
    assert recipe._projection_parameters() == []
    assert set(model.state_dict()) == {f"student.{key}" for key in student.state_dict()}
    assert {id(param) for group in optimizer.param_groups for param in group["params"]} == {
        id(param) for param in student.parameters()
    }
    torch.testing.assert_close(distill_embeddings, expected, rtol=0, atol=0)

    loss = torch.nn.functional.mse_loss(distill_embeddings, target)
    reference_loss = torch.nn.functional.mse_loss(expected, target)
    loss.backward()
    reference_loss.backward()
    for param, reference_param in zip(student.parameters(), reference.parameters(), strict=True):
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
        torch.testing.assert_close(param.grad, reference_param.grad, rtol=0, atol=0)

    optimizer.step()
    reference_optimizer.step()
    for param, reference_param in zip(student.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(param, reference_param, rtol=0, atol=0)

    restored = RetrieverStudentWithProjection(
        student=_TrainableEncoder(dtype), teacher_hidden_size=4, use_projection=False
    )
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(inputs)[1], model(inputs)[1], rtol=0, atol=0)


@pytest.mark.cpu_only
def test_retriever_student_without_projection_requires_equal_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="student_hidden_size=4, teacher_hidden_size=6"):
        RetrieverStudentWithProjection(student=_FakeEncoder(), teacher_hidden_size=6, use_projection=False)


@pytest.mark.cpu_only
def test_retriever_student_without_projection_rejects_projection_restore(tmp_path: Path) -> None:
    model = RetrieverStudentWithProjection(student=_FakeEncoder(), teacher_hidden_size=4, use_projection=False)

    with pytest.raises(ValueError, match="Cannot restore.*use_projection=False"):
        model.restore_projection_from_checkpoint(str(tmp_path))


@pytest.mark.cpu_only
@pytest.mark.parametrize("use_projection", [False, True])
def test_retriever_student_build_selects_projection(monkeypatch: pytest.MonkeyPatch, use_projection: bool) -> None:
    from nemo_automodel import NeMoAutoModelBiEncoder

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(NeMoAutoModelBiEncoder, "from_pretrained", lambda **kwargs: _FakeEncoder())

    model = RetrieverStudentWithProjection.build("unused", teacher_hidden_size=4, use_projection=use_projection)

    assert isinstance(model.projection, nn.Linear) if use_projection else model.projection is None


@pytest.mark.cpu_only
def test_retriever_student_without_projection_checkpoint_omits_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint_dir = tmp_path / "epoch_0_step_1"
    checkpoint_dir.mkdir()
    monkeypatch.setattr(TrainBiEncoderRecipe, "save_checkpoint", lambda *args, **kwargs: None)
    recipe = mod.EmbeddingDistillRecipe.__new__(mod.EmbeddingDistillRecipe)
    recipe.checkpointer = SimpleNamespace(config=SimpleNamespace(enabled=True, checkpoint_dir=str(tmp_path)))
    recipe.dist_env = SimpleNamespace(is_main=True)
    recipe.model_parts = [
        RetrieverStudentWithProjection(student=_FakeEncoder(), teacher_hidden_size=4, use_projection=False)
    ]

    recipe.save_checkpoint(epoch=0, step=1, train_loss=0.0)

    assert not (checkpoint_dir / "projection.pt").exists()


def test_retriever_student_restores_projection_from_epoch_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "epoch_0_step_10"
    consolidated_dir = checkpoint_dir / "model" / "consolidated"
    consolidated_dir.mkdir(parents=True)
    weight = torch.full((6, 4), 0.25)
    bias = torch.full((6,), -0.5)
    torch.save({"weight": weight, "bias": bias, "in_features": 4, "out_features": 6}, checkpoint_dir / "projection.pt")

    model = RetrieverStudentWithProjection(student=_FakeEncoder(hidden_size=4), teacher_hidden_size=6)
    model.restore_projection_from_checkpoint(str(consolidated_dir))

    assert torch.equal(model.projection.weight.detach().cpu(), weight)
    assert torch.equal(model.projection.bias.detach().cpu(), bias)


def test_retriever_student_projection_restore_checks_shape(tmp_path):
    student_dir = tmp_path / "step_10" / "student"
    student_dir.mkdir(parents=True)
    torch.save({"weight": torch.ones(5, 4), "bias": torch.zeros(5)}, student_dir / "projection.pt")

    model = RetrieverStudentWithProjection(student=_FakeEncoder(hidden_size=4), teacher_hidden_size=6)
    with pytest.raises(RuntimeError, match="Projection checkpoint shape mismatch"):
        model.restore_projection_from_checkpoint(str(student_dir), projection_path=str(student_dir / "projection.pt"))


def test_retriever_student_projection_restore_requires_consolidated_path(tmp_path):
    student_dir = tmp_path / "step_10" / "student"
    student_dir.mkdir(parents=True)

    model = RetrieverStudentWithProjection(student=_FakeEncoder(hidden_size=4), teacher_hidden_size=6)
    with pytest.raises(ValueError, match="epoch_0_step_x/model/consolidated"):
        model.restore_projection_from_checkpoint(str(student_dir))


def test_retriever_student_projection_export_strips_wrapper_and_projection():
    model = RetrieverStudentWithProjection(student=_FakeEncoder(hidden_size=4), teacher_hidden_size=6)
    state_dict = {
        "student.model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(1),
        "student.model.multi_modal_projector.linear_1.weight": torch.ones(1),
        "projection.weight": torch.ones(6, 4),
        "projection.bias": torch.ones(6),
    }

    hf_state_dict = model.state_dict_adapter.to_hf(state_dict)

    assert sorted(hf_state_dict) == [
        "language_model.layers.0.self_attn.q_proj.weight",
        "multi_modal_projector.linear_1.weight",
    ]


def test_retriever_student_projection_delegates_sentence_transformer_export_config():
    model = RetrieverStudentWithProjection(student=_FakeEncoder(hidden_size=4), teacher_hidden_size=6)

    model.configure_sentence_transformer_prompts(query_prompt="query:", document_prompt="passage:")

    assert model.sentence_transformer_export_config.query_prompt == "query:"
    assert model.sentence_transformer_export_config.document_prompt == "passage:"
    assert model.student.sentence_transformer_export_config.query_prompt == "query:"
    assert model.student.sentence_transformer_export_config.document_prompt == "passage:"


@pytest.mark.cpu_only
@pytest.mark.parametrize("with_tokenizer", [False, True])
def test_retriever_student_projection_configures_export_during_recipe_setup(
    monkeypatch: pytest.MonkeyPatch, with_tokenizer: bool
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    student = _FakeEncoder(hidden_size=4)
    model = RetrieverStudentWithProjection(student=student, teacher_hidden_size=6)
    tokenizer = PreTrainedTokenizerBase() if with_tokenizer else None
    collator = SimpleNamespace(query_prefix="query:", passage_prefix="passage:")

    _configure_sentence_transformer_export(model, collator, tokenizer=tokenizer)

    assert model.sentence_transformer_export_config is student.sentence_transformer_export_config
    assert model.sentence_transformer_export_config.query_prompt == "query: "
    assert model.sentence_transformer_export_config.document_prompt == "passage: "
    assert student.prompt_tokenizer is tokenizer


# ---------------------------------------------------------------------------
# _build_or_none
# ---------------------------------------------------------------------------
def test_build_or_none_returns_none_for_none():
    assert _build_or_none(None) is None


def test_build_or_none_calls_instantiate():
    sentinel = object()
    cfg = SimpleNamespace(instantiate=lambda: sentinel)

    assert _build_or_none(cfg) is sentinel


def test_build_or_none_returns_plain_object():
    obj = object()

    assert _build_or_none(obj) is obj


# ---------------------------------------------------------------------------
# _cfg_get_path
# ---------------------------------------------------------------------------
def test_cfg_get_path_flat_key():
    cfg = _DictLikeConfig(temperature=0.05)

    assert _cfg_get_path(cfg, "temperature") == 0.05


def test_cfg_get_path_nested_dict():
    cfg = {"dataloader": {"collate_fn": {"teacher_embeddings_cache": "/cache"}}}

    assert _cfg_get_path(cfg, "dataloader.collate_fn.teacher_embeddings_cache") == "/cache"


def test_cfg_get_path_attribute_fallback():
    cfg = SimpleNamespace(outer=SimpleNamespace(inner="value"))

    assert _cfg_get_path(cfg, "outer.inner") == "value"


def test_cfg_get_path_returns_default_when_missing():
    cfg = {"a": {"b": 1}}

    assert _cfg_get_path(cfg, "a.c.d", default="fallback") == "fallback"


# ---------------------------------------------------------------------------
# _move_to_device
# ---------------------------------------------------------------------------
def test_move_to_device_moves_tensors_and_keeps_others():
    batch = {"ids": torch.arange(4), "meta": "keep", "n": 3}

    out = _move_to_device(batch, torch.device("cpu"))

    assert torch.equal(out["ids"], torch.arange(4))
    assert out["meta"] == "keep"
    assert out["n"] == 3


# ---------------------------------------------------------------------------
# _unpack_qpn
# ---------------------------------------------------------------------------
def _base_qpn_batch():
    return {
        "q_input_ids": torch.randint(0, 10, (2, 5)),
        "q_attention_mask": torch.ones(2, 5, dtype=torch.long),
        "d_input_ids": torch.randint(0, 10, (2, 5)),
        "d_attention_mask": torch.ones(2, 5, dtype=torch.long),
    }


def test_unpack_qpn_without_negatives():
    q, d, n, mask = _unpack_qpn(_base_qpn_batch())

    assert set(q) == {"input_ids", "attention_mask"}
    assert set(d) == {"input_ids", "attention_mask"}
    assert n is None
    assert mask is None


def test_unpack_qpn_preserves_vlm_fields():
    batch = _base_qpn_batch()
    batch["d_pixel_values"] = torch.randn(8, 3, 16, 16)
    batch["d_image_sizes"] = torch.tensor([[16, 16]] * 8)
    batch["d_pixel_attention_mask"] = torch.ones(8, 16, 16, dtype=torch.bool)

    _q, d, n, mask = _unpack_qpn(batch)

    assert set(d) == {"input_ids", "attention_mask", "pixel_values", "image_sizes", "pixel_attention_mask"}
    assert d["pixel_values"].shape == (8, 3, 16, 16)
    assert n is None
    assert mask is None


def test_unpack_qpn_with_valid_negatives():
    batch = _base_qpn_batch()
    batch["n_input_ids"] = torch.randint(0, 10, (2, 3, 5))
    batch["n_attention_mask"] = torch.ones(2, 3, 5, dtype=torch.long)
    batch["n_mask"] = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

    _q, _d, n, mask = _unpack_qpn(batch)

    assert n["input_ids"].shape == (6, 5)
    assert n["attention_mask"].shape == (6, 5)
    assert torch.equal(mask, batch["n_mask"])


def test_unpack_qpn_flattens_negative_vlm_fields():
    batch = _base_qpn_batch()
    batch["n_input_ids"] = torch.randint(0, 10, (2, 3, 5))
    batch["n_attention_mask"] = torch.ones(2, 3, 5, dtype=torch.long)
    batch["n_pixel_values"] = torch.randn(2, 3, 3, 16, 16)
    batch["n_image_sizes"] = torch.tensor([[[16, 16], [16, 16], [16, 16]]] * 2)
    batch["n_mask"] = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

    _q, _d, n, mask = _unpack_qpn(batch)

    assert n["input_ids"].shape == (6, 5)
    assert n["attention_mask"].shape == (6, 5)
    assert n["pixel_values"].shape == (6, 3, 16, 16)
    assert n["image_sizes"].shape == (6, 2)
    assert torch.equal(mask, batch["n_mask"])


def test_unpack_qpn_non_3d_negatives_returns_none():
    batch = _base_qpn_batch()
    batch["n_input_ids"] = torch.randint(0, 10, (2, 5))
    batch["n_attention_mask"] = torch.ones(2, 5, dtype=torch.long)
    batch["n_mask"] = torch.ones(2, 1, dtype=torch.long)

    _q, _d, n, mask = _unpack_qpn(batch)

    assert n is None and mask is None


def test_unpack_qpn_zero_negatives_returns_none():
    batch = _base_qpn_batch()
    batch["n_input_ids"] = torch.zeros(2, 0, 0, dtype=torch.long)
    batch["n_attention_mask"] = torch.zeros(2, 0, 0, dtype=torch.long)
    batch["n_mask"] = torch.zeros(2, 0, dtype=torch.long)

    _q, _d, n, mask = _unpack_qpn(batch)

    assert n is None and mask is None


def test_unpack_qpn_all_masked_negatives_returns_none():
    batch = _base_qpn_batch()
    batch["n_input_ids"] = torch.randint(0, 10, (2, 3, 5))
    batch["n_attention_mask"] = torch.ones(2, 3, 5, dtype=torch.long)
    batch["n_mask"] = torch.zeros(2, 3, dtype=torch.long)

    _q, _d, n, mask = _unpack_qpn(batch)

    assert n is None and mask is None


# ---------------------------------------------------------------------------
# grouped-passage helpers
# ---------------------------------------------------------------------------
def test_repeat_queries_to_match_grouped_documents():
    s_q = torch.randn(2, 4)
    t_q = torch.randn(2, 8)
    s_d = torch.randn(6, 8)

    s_q_rep, t_q_rep = _repeat_queries_to_match_docs(s_q, t_q, s_d)

    assert s_q_rep.shape == (6, 4)
    assert t_q_rep.shape == (6, 8)
    assert torch.equal(s_q_rep[0], s_q[0])
    assert torch.equal(s_q_rep[2], s_q[0])
    assert torch.equal(s_q_rep[3], s_q[1])


def test_split_grouped_passages_returns_positive_and_negatives():
    reps = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(8, 3)

    pos, neg = _split_grouped_passages(reps, batch_size=2, n_passages=4, label="test")

    assert pos.shape == (2, 3)
    assert neg.shape == (2, 3, 3)
    assert torch.equal(pos[0], reps[0])
    assert torch.equal(neg[0, 0], reps[1])


def test_sync_projection_gradients_allreduces_missing_grads(monkeypatch):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 3)

    model = _Model()
    model.projection.weight.grad = torch.ones_like(model.projection.weight)
    model.projection.bias.grad = torch.ones_like(model.projection.bias)

    calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        calls.append(tensor)
        tensor.mul_(2.0)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    recipe = mod.EmbeddingDistillRecipe.__new__(mod.EmbeddingDistillRecipe)
    recipe.model_parts = [model]
    recipe._get_dp_group = lambda include_cp: None
    recipe._get_dp_group_size = lambda include_cp: 2

    recipe._sync_projection_gradients()

    assert len(calls) == 2


# ---------------------------------------------------------------------------
# optimizer parameter groups
# ---------------------------------------------------------------------------
def test_student_lr_overrides_preserve_decay_and_no_decay_groups():
    base_lr = 1.0e-5
    base_wd = 0.1
    vision_lr = 1.0e-6
    projector_lr = 3.0e-4
    distill_projection_lr = 3.0e-3

    model = _OptimizerGroupingModel()
    recipe = mod.EmbeddingDistillRecipe.__new__(mod.EmbeddingDistillRecipe)
    recipe.cfg = _DictLikeConfig(
        optimizer={
            "lr": base_lr,
            "distill_projection_lr": distill_projection_lr,
            "vision_tower_lr": vision_lr,
            "multi_modal_projector_lr": projector_lr,
        },
    )
    recipe.model_parts = [model]

    optimizer = torch.optim.AdamW(recipe._build_optimizer_param_groups(), lr=base_lr, weight_decay=base_wd)
    assert "distill_projection_lr" not in recipe.cfg.optimizer
    assert "vision_tower_lr" not in recipe.cfg.optimizer
    assert "multi_modal_projector_lr" not in recipe.cfg.optimizer
    recipe.optimizer = [optimizer]
    recipe._apply_student_lr_overrides()

    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=base_lr,
        max_lr=base_lr,
        min_lr=base_lr,
        lr_warmup_steps=0,
        lr_decay_steps=100,
        lr_decay_style="constant",
        start_wd=base_wd,
        end_wd=base_wd,
        wd_incr_steps=100,
        wd_incr_style="constant",
    )
    scheduler.step(1)

    group_by_param = _param_group_by_id(optimizer)
    vision_linear = model.vision_tower.linear
    vision_norm = model.vision_tower.norm

    assert group_by_param[id(vision_linear.weight)]["lr"] == pytest.approx(vision_lr)
    assert group_by_param[id(vision_linear.weight)]["weight_decay"] == pytest.approx(base_wd)
    assert group_by_param[id(vision_linear.bias)]["lr"] == pytest.approx(vision_lr)
    assert group_by_param[id(vision_linear.bias)]["weight_decay"] == pytest.approx(0.0)
    assert group_by_param[id(vision_norm.weight)]["lr"] == pytest.approx(vision_lr)
    assert group_by_param[id(vision_norm.weight)]["weight_decay"] == pytest.approx(0.0)

    assert group_by_param[id(model.multi_modal_projector.weight)]["lr"] == pytest.approx(projector_lr)
    assert group_by_param[id(model.multi_modal_projector.weight)]["weight_decay"] == pytest.approx(base_wd)
    assert group_by_param[id(model.multi_modal_projector.bias)]["lr"] == pytest.approx(projector_lr)
    assert group_by_param[id(model.multi_modal_projector.bias)]["weight_decay"] == pytest.approx(0.0)

    assert group_by_param[id(model.language_model.weight)]["lr"] == pytest.approx(base_lr)
    assert group_by_param[id(model.language_model.weight)]["weight_decay"] == pytest.approx(base_wd)
    assert group_by_param[id(model.projection.weight)]["lr"] == pytest.approx(distill_projection_lr)
    assert group_by_param[id(model.projection.weight)]["weight_decay"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _dp_group_src_rank
# ---------------------------------------------------------------------------
def test_dp_group_src_rank_none_group():
    assert _dp_group_src_rank(None) == 0


def test_dp_group_src_rank_error_returns_zero():
    # A bogus group object makes torch.distributed raise, which is swallowed.
    assert _dp_group_src_rank(object()) == 0


# ---------------------------------------------------------------------------
# setup() intermediate-loss guards
# ---------------------------------------------------------------------------
def _make_recipe_with_cfg(cfg):
    recipe = mod.EmbeddingDistillRecipe.__new__(mod.EmbeddingDistillRecipe)
    recipe.cfg = cfg
    recipe.optimizer = []
    return recipe


def test_setup_default_intermediate_loss_inherits_layer_pairs(monkeypatch):
    monkeypatch.setattr(mod.TrainBiEncoderRecipe, "setup", lambda self: None)
    monkeypatch.setattr(mod.EmbeddingDistillRecipe, "_sync_projection_parameters", lambda self: None)
    monkeypatch.setattr(mod.EmbeddingDistillRecipe, "_log_trainable_parameter_summary", lambda self: None)
    cfg = _DictLikeConfig(
        intermediate_loss_weight=0.0,
        layer_pairs=[[1, 2], [3, 4]],
        teacher_embeddings_cache="/tmp/cache",
    )
    recipe = _make_recipe_with_cfg(cfg)

    recipe.setup()

    assert recipe.teacher_model is None
    assert recipe.intermediate_loss.layer_pairs == [(1, 2), (3, 4)]


def test_setup_rejects_empty_layer_pairs_on_configured_loss(monkeypatch):
    monkeypatch.setattr(mod.TrainBiEncoderRecipe, "setup", lambda self: None)
    cfg = _DictLikeConfig(
        intermediate_loss_weight=1.0,
        layer_pairs=[[1, 2]],
        intermediate_loss=mod.IntermediateDistillLoss(layer_pairs=[]),
        teacher_embeddings_cache="/tmp/cache",
    )
    recipe = _make_recipe_with_cfg(cfg)

    with pytest.raises(ValueError, match="empty layer_pairs"):
        recipe.setup()


def test_setup_rejects_uncaptured_layer_pairs(monkeypatch):
    monkeypatch.setattr(mod.TrainBiEncoderRecipe, "setup", lambda self: None)
    cfg = _DictLikeConfig(
        intermediate_loss_weight=1.0,
        layer_pairs=[[1, 2]],
        intermediate_loss=mod.IntermediateDistillLoss(layer_pairs=[(5, 9)]),
        teacher_embeddings_cache="/tmp/cache",
    )
    recipe = _make_recipe_with_cfg(cfg)

    with pytest.raises(ValueError, match="do not capture"):
        recipe.setup()


def test_setup_requires_layer_pairs_when_weight_positive(monkeypatch):
    monkeypatch.setattr(mod.TrainBiEncoderRecipe, "setup", lambda self: None)
    cfg = _DictLikeConfig(
        intermediate_loss_weight=1.0,
        layer_pairs=[],
        teacher_embeddings_cache="/tmp/cache",
    )
    recipe = _make_recipe_with_cfg(cfg)

    with pytest.raises(ValueError, match="requires non-empty layer_pairs"):
        recipe.setup()


# ---------------------------------------------------------------------------
# validation scoring reps (_extract_scoring_reps)
# ---------------------------------------------------------------------------
def test_extract_scoring_reps_base_is_identity():
    recipe = TrainBiEncoderRecipe.__new__(TrainBiEncoderRecipe)
    reps = torch.randn(2, 4)

    assert recipe._extract_scoring_reps(reps) is reps


def test_extract_scoring_reps_distill_unpacks_pooled():
    recipe = mod.EmbeddingDistillRecipe.__new__(mod.EmbeddingDistillRecipe)
    pooled = torch.randn(2, 4)
    projected = torch.randn(2, 8)
    output = (pooled, projected, {0: torch.randn(2, 3, 4)})

    assert recipe._extract_scoring_reps(output) is pooled
    # A plain tensor (e.g. if forward is ever simplified) still passes through.
    plain = torch.randn(2, 4)
    assert recipe._extract_scoring_reps(plain) is plain


class _TupleStudent(nn.Module):
    """Student whose forward mimics RetrieverStudentWithProjection's 3-tuple output."""

    pooling = "avg"
    l2_normalize = False

    def __init__(self, hidden=4):
        super().__init__()
        self.hidden = hidden

    def forward(self, inputs):
        ids = inputs["input_ids"].float()
        n = ids.shape[0]
        pooled = ids.mean(dim=1, keepdim=True).expand(n, self.hidden).contiguous()
        projected = pooled.clone()
        return pooled, projected, {}


def test_validation_epoch_runs_with_tuple_student_output(monkeypatch):
    """Regression: inherited validation must handle the student's tuple forward output."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    recipe = mod.EmbeddingDistillRecipe.__new__(mod.EmbeddingDistillRecipe)
    recipe.model_parts = [_TupleStudent(hidden=4)]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.step_scheduler = SimpleNamespace(step=1, epoch=0)
    recipe.val_n_passages = 2
    recipe.temperature = 0.05

    batch = {
        "q_input_ids": torch.randint(1, 10, (2, 3)),
        "q_attention_mask": torch.ones(2, 3, dtype=torch.long),
        "d_input_ids": torch.randint(1, 10, (4, 3)),
        "d_attention_mask": torch.ones(4, 3, dtype=torch.long),
    }

    result = recipe._run_validation_epoch([batch])

    assert set(result.metrics) == {"val_loss", "val_acc1", "val_mrr"}
    assert all(torch.isfinite(torch.tensor(v)) for v in result.metrics.values())

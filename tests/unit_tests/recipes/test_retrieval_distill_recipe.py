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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.recipes.retrieval import distill_bi_encoder as mod
from nemo_automodel.recipes.retrieval.distill_bi_encoder import (
    _build_or_none,
    _cfg_get_path,
    _clean_path,
    _copy_checkpoint_metadata,
    _dp_group_src_rank,
    _mirror_hf_metadata,
    _move_to_device,
    _strip_student_prefix,
    _unpack_qpn,
)
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe


class _DictLikeConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


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


def test_unpack_qpn_with_valid_negatives():
    batch = _base_qpn_batch()
    batch["n_input_ids"] = torch.randint(0, 10, (2, 3, 5))
    batch["n_attention_mask"] = torch.ones(2, 3, 5, dtype=torch.long)
    batch["n_mask"] = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)

    _q, _d, n, mask = _unpack_qpn(batch)

    assert n["input_ids"].shape == (6, 5)
    assert n["attention_mask"].shape == (6, 5)
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
# _strip_student_prefix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prefix", ["student.model.", "module.student.model.", "model."])
def test_strip_student_prefix_variants(prefix):
    t = torch.randn(2)
    state = {f"{prefix}layers.0.weight": t, "projection.weight": torch.randn(2)}

    stripped = _strip_student_prefix(state)

    assert set(stripped) == {"layers.0.weight"}
    assert torch.equal(stripped["layers.0.weight"], t)


def test_strip_student_prefix_already_backbone_drops_projection():
    t = torch.randn(2)
    state = {"encoder.weight": t, "projection.weight": torch.randn(2), "projection.bias": torch.randn(2)}

    stripped = _strip_student_prefix(state)

    assert set(stripped) == {"encoder.weight"}


# ---------------------------------------------------------------------------
# _dp_group_src_rank
# ---------------------------------------------------------------------------
def test_dp_group_src_rank_none_group():
    assert _dp_group_src_rank(None) == 0


def test_dp_group_src_rank_error_returns_zero():
    # A bogus group object makes torch.distributed raise, which is swallowed.
    assert _dp_group_src_rank(object()) == 0


# ---------------------------------------------------------------------------
# filesystem helpers
# ---------------------------------------------------------------------------
def test_clean_path_file_dir_and_symlink(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    _clean_path(f)
    assert not f.exists()

    d = tmp_path / "d"
    d.mkdir()
    (d / "inner.txt").write_text("y")
    _clean_path(d)
    assert not d.exists()

    target = tmp_path / "target.txt"
    target.write_text("z")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    _clean_path(link)
    assert not link.exists()
    assert target.exists()


def test_mirror_hf_metadata_copies_non_weight_files(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "tokenizer.json").write_text("{}")
    (src / "model.safetensors").write_text("weights")
    (src / "pytorch_model.bin").write_text("weights")
    (src / "model.safetensors.index.json").write_text("{}")

    dst = tmp_path / "dst"
    _mirror_hf_metadata(src, dst)

    names = {p.name for p in dst.iterdir()}
    assert names == {"config.json", "tokenizer.json"}


def test_mirror_hf_metadata_overwrite_behavior(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("new")

    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "config.json").write_text("old")

    _mirror_hf_metadata(src, dst, overwrite=False)
    assert (dst / "config.json").read_text() == "old"

    _mirror_hf_metadata(src, dst, overwrite=True)
    assert (dst / "config.json").read_text() == "new"


def test_mirror_hf_metadata_missing_src_is_noop(tmp_path):
    dst = tmp_path / "dst"
    _mirror_hf_metadata(tmp_path / "does_not_exist", dst)

    assert not dst.exists()


def test_copy_checkpoint_metadata_skips_safetensors(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "pytorch_model.bin").write_text("weights")
    (src / "model.safetensors").write_text("weights")
    (src / "model.safetensors.index.json").write_text("{}")

    dst = tmp_path / "dst"
    dst.mkdir()
    _copy_checkpoint_metadata(src, dst)

    names = {p.name for p in dst.iterdir()}
    assert names == {"config.json", "pytorch_model.bin"}


# ---------------------------------------------------------------------------
# setup() intermediate-loss guards
# ---------------------------------------------------------------------------
def _make_recipe_with_cfg(cfg):
    recipe = mod.EmbeddingDistillRecipe.__new__(mod.EmbeddingDistillRecipe)
    recipe.cfg = cfg
    return recipe


def test_setup_default_intermediate_loss_inherits_layer_pairs(monkeypatch):
    monkeypatch.setattr(mod.TrainBiEncoderRecipe, "setup", lambda self: None)
    monkeypatch.setattr(mod.EmbeddingDistillRecipe, "_sync_projection_parameters", lambda self: None)
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

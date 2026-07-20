# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Qwen3.5-MoE CP pre-embedding (PR #2432).

``prepare_model_inputs_for_cp`` builds full-sequence multimodal embeddings and
mRoPE positions *before* context-parallel sharding, plus a dense ``seq_index``
the linear-attention layers need. Instantiating the full VL model is expensive,
so we build a barebones instance via ``__new__`` and stub the heavy submodules
(visual encoder, rope-index helper, embedding table).
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.models.qwen3_5_moe")

from nemo_automodel.components.models.qwen3_5_moe.model import (
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
)


def _build_model(*, rope_index=None, image_token_id=None, video_token_id=None):
    """Build a barebones Qwen3_5MoeForConditionalGeneration with stubbed deps."""
    model = Qwen3_5MoeForConditionalGeneration.__new__(Qwen3_5MoeForConditionalGeneration)
    nn.Module.__init__(model)

    hidden = 4

    def _embed(input_ids):
        # Deterministic embeddings: [B, S, H] where each value mirrors the token id.
        return input_ids.unsqueeze(-1).expand(*input_ids.shape, hidden).float()

    # Instance attribute shadows the class method.
    model.get_input_embeddings = lambda: _embed

    inner = types.SimpleNamespace()
    inner.visual = None  # no rotary_pos_emb attribute -> hasattr() is False

    def _default_rope_index(input_ids, **kwargs):
        # [3, B, S] mRoPE positions + a rope_delta per row.
        b, s = input_ids.shape
        pos = torch.arange(s).view(1, 1, s).expand(3, b, s).contiguous()
        return pos, torch.zeros(b, 1)

    inner.get_rope_index = rope_index or _default_rope_index
    inner.rope_deltas = None
    model.model = inner

    model.config = types.SimpleNamespace(
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )
    return model


class TestPrepareModelInputsForCP:
    def test_requires_input_ids(self):
        model = _build_model()
        with pytest.raises(ValueError, match="requires input_ids"):
            model.prepare_model_inputs_for_cp({"input_ids": None})

    def test_returns_sharder_and_positions_only(self):
        """Sharder-only hook: no inputs_embeds (the forward embeds), full mRoPE
        positions returned for the aux shard, mm_token_type_ids consumed."""
        from nemo_automodel.components.distributed.cp_sharder import (
            ContextParallelismSharder,
            round_robin_local_indices,
            shard_batch_aux_only,
        )

        model = _build_model()
        out = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[5, 6, 7, 8]])})

        assert "inputs_embeds" not in out  # embedding happens in forward now
        sharder = out["cp_sharder"]
        assert isinstance(sharder, ContextParallelismSharder)
        assert sharder.shard_batch is shard_batch_aux_only
        assert sharder.local_token_global_indices is round_robin_local_indices
        assert out["position_ids"].shape == (3, 1, 4)  # mRoPE [3, B, S]
        assert out["mm_token_type_ids"] is None
        assert model.model.rope_deltas is not None

    def test_input_ids_not_consumed(self):
        """input_ids stays in the batch for the forward's in-forward embed+splice."""
        model = _build_model()
        out = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[5, 6, 7, 8]])})
        assert "input_ids" not in out

    def test_existing_position_ids_not_recomputed(self):
        called = {"count": 0}

        def _rope(input_ids, **kwargs):
            called["count"] += 1
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope)
        pos = torch.arange(4).view(1, 4)
        out = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[5, 6, 7, 8]]), "position_ids": pos})

        assert called["count"] == 0, "get_rope_index must not run when position_ids provided"
        assert out["position_ids"] is pos

    def test_image_grid_hws_promoted_to_thw(self):
        """image_grid_hws of shape [N, 2] is promoted to [N, 3] and written back for the forward."""
        captured = {}

        def _rope(input_ids, **kwargs):
            captured.update(kwargs)
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope)
        image_grid_hws = torch.tensor([[2, 2]])  # [N, 2]
        out = model.prepare_model_inputs_for_cp(
            {
                "input_ids": torch.tensor([[5, 6, 7, 8]]),
                "image_grid_hws": image_grid_hws,
            }
        )
        assert captured["image_grid_thw"].tolist() == [[1, 2, 2]]
        assert out["image_grid_thw"].tolist() == [[1, 2, 2]]
        assert out["image_grid_hws"] is None

    def test_mm_token_type_ids_synthesized_from_token_ids(self):
        """When get_rope_index accepts mm_token_type_ids, it is built from image/video token ids."""
        captured = {}

        def _rope(input_ids, *, image_grid_thw=None, video_grid_thw=None, attention_mask=None, mm_token_type_ids=None):
            captured["mm_token_type_ids"] = mm_token_type_ids
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope, image_token_id=6, video_token_id=8)
        model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[5, 6, 7, 8]])})

        # token 6 -> image (1), token 8 -> video (2), others 0.
        assert captured["mm_token_type_ids"].tolist() == [[0, 1, 0, 2]]


class TestEmbedAndSpliceForCP:
    """The in-forward embed + vision splice (moved out of the CP hook)."""

    def test_image_features_scattered_into_embeds(self):
        model = _build_model(image_token_id=99)
        model.model.visual = types.SimpleNamespace(rotary_pos_emb=types.SimpleNamespace(to=lambda dev: None))
        feat = torch.full((1, 4), 8.0)
        model.model.get_image_features = lambda pixel_values, image_grid_thw=None, return_dict=True: (
            types.SimpleNamespace(pooler_output=[feat])
        )

        def _mask(input_ids, *, inputs_embeds=None, image_features=None, video_features=None):
            image_mask = (input_ids == 99).unsqueeze(-1).expand_as(inputs_embeds)
            return image_mask, torch.zeros_like(image_mask)

        model.model.get_placeholder_mask = _mask
        emb = model._embed_and_splice_for_cp(
            torch.tensor([[5, 99, 7]]),
            pixel_values=torch.zeros(1, 3, 2, 2),
            pixel_values_videos=None,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            video_grid_thw=None,
        )
        assert torch.allclose(emb[0, 1], torch.full((4,), 8.0))  # image token overwritten
        assert torch.allclose(emb[0, 0], torch.full((4,), 5.0))  # text token untouched


class _FakeCPMesh:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class TestPipelineStageMetas:
    def _model(self, *, cp_size, lm_head):
        model = Qwen3_5MoeForConditionalGeneration.__new__(Qwen3_5MoeForConditionalGeneration)
        nn.Module.__init__(model)
        model.config = types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=8, vocab_size=32))
        model.lm_head = nn.Linear(8, 32, bias=False) if lm_head else None
        model.cp_mesh = _FakeCPMesh(cp_size) if cp_size > 1 else None
        return model

    def test_cp_shards_stage_outputs(self):
        model = self._model(cp_size=2, lm_head=True)
        ins, outs = model.get_pipeline_stage_metas(is_first=True, microbatch_size=1, seq_len=6, dtype=torch.float32)
        assert ins[0].shape == (1, 6) and ins[0].dtype == torch.long  # full token ids in
        assert outs[0].shape == (1, 4, 32)  # local (pad 6->8, //2) logits out

    def test_cp1_symmetric(self):
        model = self._model(cp_size=1, lm_head=True)
        ins, outs = model.get_pipeline_stage_metas(is_first=True, microbatch_size=2, seq_len=5, dtype=torch.float32)
        assert ins[0].shape == (2, 5) and outs[0].shape == (2, 5, 32)


def _build_inner_model():
    """Barebones Qwen3_5MoeModel with stubbed language_model and no vision encoder."""
    model = Qwen3_5MoeModel.__new__(Qwen3_5MoeModel)
    nn.Module.__init__(model)
    model.visual = None  # forces the text-only path

    captured = {}

    def _language_model(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(logits=torch.zeros(1, 4, 8))

    model.language_model = _language_model
    return model, captured


class TestTextOnlyForward:
    def test_int_input_ids_passed_through(self):
        model, captured = _build_inner_model()
        input_ids = torch.tensor([[1, 2, 3, 4]])

        model.forward(input_ids=input_ids)

        assert captured["input_ids"] is input_ids
        assert captured["inputs_embeds"] is None

    def test_float_input_ids_treated_as_embeds(self):
        """Pipeline-parallel: float input_ids are already embeddings."""
        model, captured = _build_inner_model()
        embeds = torch.randn(1, 4, 8)

        model.forward(input_ids=embeds)

        assert captured["input_ids"] is None
        assert captured["inputs_embeds"] is embeds

    def test_inputs_embeds_passed_through(self):
        model, captured = _build_inner_model()
        embeds = torch.randn(1, 4, 8)

        model.forward(inputs_embeds=embeds)

        assert captured["inputs_embeds"] is embeds
        assert captured["input_ids"] is None

    def test_raises_when_neither_provided(self):
        model, _ = _build_inner_model()
        with pytest.raises(ValueError, match="Either input_ids or inputs_embeds"):
            model.forward()

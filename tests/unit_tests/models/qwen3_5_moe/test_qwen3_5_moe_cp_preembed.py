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
            model.prepare_model_inputs_for_cp(None)

    def test_text_only_builds_embeds_and_positions(self):
        model = _build_model()
        input_ids = torch.tensor([[5, 6, 7, 8]])

        out = model.prepare_model_inputs_for_cp(input_ids)

        # seq_index is derived inside the CP linear-attn layer, not here.
        assert {k for k, v in out.items() if v is not None} == {"inputs_embeds", "position_ids"}
        assert out["inputs_embeds"].shape == (1, 4, 4)
        # position_ids came from get_rope_index (mRoPE [3, B, S]).
        assert out["position_ids"].shape == (3, 1, 4)
        # rope_deltas stashed back onto the inner model.
        assert model.model.rope_deltas is not None

    def test_existing_position_ids_not_recomputed(self):
        called = {"count": 0}

        def _rope(input_ids, **kwargs):
            called["count"] += 1
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope)
        pos = torch.arange(4).view(1, 4)
        out = model.prepare_model_inputs_for_cp(torch.tensor([[5, 6, 7, 8]]), position_ids=pos)

        assert called["count"] == 0, "get_rope_index must not run when position_ids provided"
        assert out["position_ids"] is pos

    def test_image_grid_hws_promoted_to_thw(self):
        """image_grid_hws of shape [N, 2] is promoted to [N, 3] by prepending a temporal=1 column."""
        captured = {}

        def _rope(input_ids, **kwargs):
            captured.update(kwargs)
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope)
        image_grid_hws = torch.tensor([[2, 2]])  # [N, 2]
        model.prepare_model_inputs_for_cp(torch.tensor([[5, 6, 7, 8]]), image_grid_hws=image_grid_hws)
        assert captured["image_grid_thw"].tolist() == [[1, 2, 2]]

    def test_mm_token_type_ids_synthesized_from_token_ids(self):
        """When get_rope_index accepts mm_token_type_ids, it is built from image/video token ids."""
        captured = {}

        def _rope(input_ids, *, image_grid_thw=None, video_grid_thw=None, attention_mask=None, mm_token_type_ids=None):
            captured["mm_token_type_ids"] = mm_token_type_ids
            return torch.zeros(3, 1, input_ids.shape[1]), torch.zeros(1, 1)

        model = _build_model(rope_index=_rope, image_token_id=6, video_token_id=8)
        model.prepare_model_inputs_for_cp(torch.tensor([[5, 6, 7, 8]]))

        # token 6 -> image (1), token 8 -> video (2), others 0.
        assert captured["mm_token_type_ids"].tolist() == [[0, 1, 0, 2]]

    def test_consumed_keys_removed_from_batch(self):
        """Consumed keys come back as None markers for the dispatcher to remove."""
        model = _build_model()
        position_ids = torch.zeros(3, 1, 4, dtype=torch.long)
        out = model.prepare_model_inputs_for_cp(
            torch.tensor([[5, 6, 7, 8]]),
            mm_token_type_ids=torch.zeros(1, 4, dtype=torch.long),
            position_ids=position_ids,
        )
        assert out["input_ids"] is None
        assert out["mm_token_type_ids"] is None
        # position_ids is not consumed into embeds and must survive.
        assert out["position_ids"] is position_ids


class TestForwardPreEmbedDispatch:
    def test_pre_embed_only_dispatches_to_prepare(self):
        model = _build_model()
        sentinel = {"inputs_embeds": torch.zeros(1, 4, 4)}

        captured = {}

        def _fake_prepare(input_ids, **kwargs):
            captured["input_ids"] = input_ids
            captured["kwargs"] = kwargs
            return sentinel

        model.prepare_model_inputs_for_cp = _fake_prepare

        input_ids = torch.tensor([[5, 6, 7, 8]])
        pixel_values = torch.randn(4, 8)
        # The dispatcher hands the batch opaquely to forward; forward restores
        # the hook's explicit tensor arguments.
        out = model.forward(
            _pre_embed_only=True,
            _cp_batch={"input_ids": input_ids, "pixel_values": pixel_values},
        )

        assert out is sentinel
        assert torch.equal(captured["input_ids"], input_ids)
        assert "pixel_values" in captured["kwargs"]


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

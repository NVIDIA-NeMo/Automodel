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

"""Tests for the CP enablement on ``NemotronOmniForConditionalGeneration``.

Covers:
  - ``prepare_model_inputs_for_cp`` returns a dict containing ``inputs_embeds``
    with the expected shape and image/video/sound token positions filled.
  - ``prepare_inputs_embeds_for_cp`` is a thin Tensor-returning wrapper.
  - ``forward(_pre_embed_only=True)`` delegates to ``prepare_model_inputs_for_cp``
    without entering the LLM body (so FSDP2 forward pre-hooks fire on
    ``__call__`` while the LLM forward is skipped).
  - ``forward(inputs_embeds=...)`` skips the multimodal scatter block.

The model is constructed via ``object.__new__`` + minimal stub submodules so we
do not load the 30B real model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniForConditionalGeneration,
)


IMG_TOKEN_ID = 18
SOUND_TOKEN_ID = 27
HIDDEN = 8


class _StubEmbedding(nn.Module):
    """nn.Embedding-equivalent that always returns a small constant per id."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.hidden = hidden

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Deterministic: each id maps to a vector of (id+1)/100.
        out = (ids.float().unsqueeze(-1) + 1.0) / 100.0
        return out.expand(*ids.shape, self.hidden).clone()


class _StubLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._embed = _StubEmbedding()

    def get_input_embeddings(self):
        return self._embed


def _make_omni_stub(*, with_sound_encoder: bool = True):
    """Construct a NemotronOmniForConditionalGeneration with only the attrs the
    CP path touches. Vision/video/sound encoders are stubs producing constants."""
    self = object.__new__(NemotronOmniForConditionalGeneration)
    nn.Module.__init__(self)
    self.img_context_token_id = IMG_TOKEN_ID
    self.sound_context_token_id = SOUND_TOKEN_ID
    self.language_model = _StubLanguageModel()

    # extract_feature returns a tensor of constant 9.0 with shape [N_tiles, K, H]
    # where the model will reshape to (-1, H) and scatter onto img positions.
    def _extract_feature(pixel_values):
        # pixel_values: [N_tiles, C, H, W] -> emit one feature per tile.
        n_tiles = pixel_values.shape[0]
        return torch.full((n_tiles, 1, HIDDEN), 9.0)

    def _extract_feature_dynamic(pixel_values, imgs_sizes):
        n = pixel_values.shape[0]
        return torch.full((1, n, HIDDEN), 7.0)

    def _extract_video_feature(pixel_values_videos):
        n = pixel_values_videos.shape[0]
        return torch.full((1, n, HIDDEN), 5.0)

    def _extract_sound_feature(features, attention_mask):
        n = features.shape[0]
        return torch.full((n, 1, HIDDEN), 3.0)

    self.extract_feature = _extract_feature
    self.extract_feature_dynamic = _extract_feature_dynamic
    self.extract_video_feature = _extract_video_feature
    self.extract_sound_feature = _extract_sound_feature
    self.sound_encoder = nn.Identity() if with_sound_encoder else None
    return self


# -----------------------------------------------------------------------------
# prepare_model_inputs_for_cp
# -----------------------------------------------------------------------------


def test_prepare_model_inputs_for_cp_returns_dict():
    model = _make_omni_stub()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    out = model.prepare_model_inputs_for_cp(input_ids=input_ids)
    assert isinstance(out, dict)
    assert "inputs_embeds" in out
    assert out["inputs_embeds"].shape == (1, 4, HIDDEN)


def test_prepare_model_inputs_for_cp_text_only_returns_pure_embeds():
    """No multimodal inputs -> embeds are just embed_tokens(input_ids)."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[5, 6, 7]])
    out = model.prepare_model_inputs_for_cp(input_ids=input_ids)["inputs_embeds"]
    expected = model.language_model.get_input_embeddings()(input_ids)
    assert torch.equal(out, expected)


def test_prepare_model_inputs_for_cp_image_scatter_at_placeholder_positions():
    """Image positions in input_ids must receive the vit feature value (9.0)."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[1, IMG_TOKEN_ID, IMG_TOKEN_ID, 4]])
    pixel_values = torch.zeros(2, 3, 4, 4)  # 2 tiles
    image_flags = torch.tensor([[1], [1]])
    out = model.prepare_model_inputs_for_cp(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_flags=image_flags,
    )["inputs_embeds"]
    assert out.shape == (1, 4, HIDDEN)
    # Positions 1 and 2 are image tokens => value 9.0
    assert torch.allclose(out[0, 1], torch.full((HIDDEN,), 9.0))
    assert torch.allclose(out[0, 2], torch.full((HIDDEN,), 9.0))
    # Positions 0 and 3 are text tokens => preserved from embed lookup
    expected_pos0 = (torch.tensor([1.0]) + 1.0) / 100.0
    expected_pos3 = (torch.tensor([4.0]) + 1.0) / 100.0
    assert torch.allclose(out[0, 0], expected_pos0.expand(HIDDEN))
    assert torch.allclose(out[0, 3], expected_pos3.expand(HIDDEN))


def test_prepare_model_inputs_for_cp_dynamic_res_takes_priority_over_static():
    """When imgs_sizes is provided, the dynamic-res branch (extract_feature_dynamic)
    handles vision; the static path (extract_feature) is skipped."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[1, IMG_TOKEN_ID, 3]])
    pixel_values = torch.zeros(1, 3, 8, 8)
    imgs_sizes = torch.tensor([[8, 8]])
    out = model.prepare_model_inputs_for_cp(
        input_ids=input_ids,
        pixel_values=pixel_values,
        imgs_sizes=imgs_sizes,
    )["inputs_embeds"]
    # extract_feature_dynamic stub returns 7.0; extract_feature returns 9.0
    assert torch.allclose(out[0, 1], torch.full((HIDDEN,), 7.0))


def test_prepare_model_inputs_for_cp_video_scatter_at_img_token_positions():
    """Video features scatter at the same img_context_token_id positions
    (image and video are mutually exclusive on Nemotron-Omni)."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[1, IMG_TOKEN_ID, IMG_TOKEN_ID, 4]])
    pixel_values_videos = torch.zeros(2, 3, 4, 4)
    out = model.prepare_model_inputs_for_cp(
        input_ids=input_ids,
        pixel_values_videos=pixel_values_videos,
    )["inputs_embeds"]
    assert torch.allclose(out[0, 1], torch.full((HIDDEN,), 5.0))
    assert torch.allclose(out[0, 2], torch.full((HIDDEN,), 5.0))


def test_prepare_model_inputs_for_cp_sound_scatter_at_sound_token():
    model = _make_omni_stub()
    input_ids = torch.tensor([[SOUND_TOKEN_ID, 2, SOUND_TOKEN_ID]])
    sound_features = torch.zeros(2, 4, 16)  # 2 sound chunks
    sound_attention_mask = torch.ones(2, 4)
    out = model.prepare_model_inputs_for_cp(
        input_ids=input_ids,
        sound_features=sound_features,
        sound_attention_mask=sound_attention_mask,
    )["inputs_embeds"]
    assert torch.allclose(out[0, 0], torch.full((HIDDEN,), 3.0))
    assert torch.allclose(out[0, 2], torch.full((HIDDEN,), 3.0))
    expected_pos1 = (torch.tensor([2.0]) + 1.0) / 100.0
    assert torch.allclose(out[0, 1], expected_pos1.expand(HIDDEN))


def test_prepare_model_inputs_for_cp_sound_skipped_when_no_sound_encoder():
    """If model.sound_encoder is None (sound disabled), sound branch must be a no-op."""
    model = _make_omni_stub(with_sound_encoder=False)
    input_ids = torch.tensor([[SOUND_TOKEN_ID, 2, SOUND_TOKEN_ID]])
    sound_features = torch.zeros(2, 4, 16)
    out = model.prepare_model_inputs_for_cp(
        input_ids=input_ids,
        sound_features=sound_features,
    )["inputs_embeds"]
    # Should equal pure embed lookup (sound positions unchanged)
    expected = model.language_model.get_input_embeddings()(input_ids)
    assert torch.equal(out, expected)


# -----------------------------------------------------------------------------
# prepare_inputs_embeds_for_cp (thin wrapper)
# -----------------------------------------------------------------------------


def test_prepare_inputs_embeds_for_cp_returns_tensor_not_dict():
    """Thin wrapper: returns ``Tensor`` (just inputs_embeds), not a dict."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[1, 2, 3]])
    out = model.prepare_inputs_embeds_for_cp(input_ids=input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, HIDDEN)


def test_prepare_inputs_embeds_for_cp_matches_prepare_model_inputs_for_cp():
    """Wrapper output equals ``prepare_model_inputs_for_cp(...)["inputs_embeds"]``."""
    model = _make_omni_stub()
    input_ids = torch.tensor([[1, IMG_TOKEN_ID, 3]])
    pixel_values = torch.zeros(1, 3, 4, 4)
    image_flags = torch.tensor([[1]])
    a = model.prepare_inputs_embeds_for_cp(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_flags=image_flags,
    )
    b = model.prepare_model_inputs_for_cp(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_flags=image_flags,
    )["inputs_embeds"]
    assert torch.equal(a, b)


# -----------------------------------------------------------------------------
# forward(_pre_embed_only=True)
# -----------------------------------------------------------------------------


def test_forward_pre_embed_only_returns_dict_from_prepare_model_inputs_for_cp(monkeypatch):
    """forward(_pre_embed_only=True) must early-return prepare_model_inputs_for_cp's dict
    WITHOUT entering the LLM forward."""
    model = _make_omni_stub()

    # Sentinel: if the LLM body is invoked, language_model.__call__ would be hit.
    # We assert it is NOT by mocking it to raise.
    def _llm_must_not_run(*args, **kwargs):
        raise AssertionError("language_model should NOT be called under _pre_embed_only=True")

    model.language_model.forward = _llm_must_not_run  # would also catch __call__

    out = model.forward(
        input_ids=torch.tensor([[1, IMG_TOKEN_ID, 3]]),
        pixel_values=torch.zeros(1, 3, 4, 4),
        image_flags=torch.tensor([[1]]),
        _pre_embed_only=True,
    )
    assert isinstance(out, dict)
    assert "inputs_embeds" in out
    assert out["inputs_embeds"].shape == (1, 3, HIDDEN)


def test_forward_pre_embed_only_default_false_does_not_short_circuit(monkeypatch):
    """Default ``_pre_embed_only=False`` must NOT take the early-return path."""
    model = _make_omni_stub()

    # Prove the early branch isn't triggered by patching prepare_model_inputs_for_cp
    # to a sentinel and asserting it's NOT what gets returned. (The full forward
    # would hit the LLM, which we mock to raise — so we expect AssertionError.)
    def _llm_raises(*args, **kwargs):
        raise AssertionError("LM was reached past the multimodal block")

    model.language_model.forward = _llm_raises

    with pytest.raises(AssertionError, match="LM was reached"):
        model.forward(
            input_ids=torch.tensor([[1, 2, 3]]),
            # _pre_embed_only defaults to False
        )


def test_forward_inputs_embeds_skips_multimodal_scatter_block():
    """If caller passes inputs_embeds (the post-CP-shard path), forward should
    NOT call extract_feature etc. — the embeds are already correct."""
    model = _make_omni_stub()
    sentinel = []
    orig_extract = model.extract_feature

    def _spy(pixel_values):
        sentinel.append(pixel_values)
        return orig_extract(pixel_values)

    model.extract_feature = _spy

    # Mock LM to swallow the call so forward can complete
    def _fake_llm(input_ids=None, inputs_embeds=None, **kw):
        return SimpleNamespace(logits=inputs_embeds, loss=None, hidden_states=None)

    model.language_model.forward = _fake_llm
    model.language_model.__call__ = _fake_llm  # nn.Module.__call__ wraps forward

    pre_built = torch.randn(1, 3, HIDDEN)
    out = model.forward(
        inputs_embeds=pre_built,
        pixel_values=torch.zeros(1, 3, 4, 4),  # provided but should be IGNORED
        image_flags=torch.tensor([[1]]),
    )
    # extract_feature must NOT have been called
    assert sentinel == [], "extract_feature should be skipped when inputs_embeds is supplied"

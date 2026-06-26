# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU-only tests for Qwen3.5 VLM media routing."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.models.qwen3_5")

from nemo_automodel.components.models.qwen3_5.model import (
    HFQwen3_5Model,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)


def _build_conditional_model(*, image_token_id=None, video_token_id=None, vision_start_token_id=None):
    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = types.SimpleNamespace(
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
    )
    return model


class TestPopStagedVlmMedia:
    def test_drops_orphaned_image_media(self):
        model = _build_conditional_model(image_token_id=99, video_token_id=98, vision_start_token_id=97)
        kwargs = {
            "pixel_values": torch.randn(4, 8),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }

        pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw = model._pop_staged_vlm_media(
            torch.tensor([[10, 11, 12]]),
            kwargs,
        )

        assert pixel_values is None
        assert pixel_values_videos is None
        assert image_grid_thw is None
        assert video_grid_thw is None

    def test_keeps_image_media_when_placeholder_exists(self):
        model = _build_conditional_model(image_token_id=99, video_token_id=98, vision_start_token_id=97)
        pixel_values_in = torch.randn(4, 8)
        image_grid_in = torch.tensor([[1, 4, 4]])
        kwargs = {
            "pixel_values": pixel_values_in,
            "image_grid_thw": image_grid_in,
        }

        pixel_values, _, image_grid_thw, _ = model._pop_staged_vlm_media(
            torch.tensor([[10, 99, 12]]),
            kwargs,
        )

        assert pixel_values is pixel_values_in
        assert image_grid_thw is image_grid_in

    def test_drops_orphaned_video_media(self):
        model = _build_conditional_model(image_token_id=99, video_token_id=98, vision_start_token_id=97)
        kwargs = {
            "pixel_values_videos": torch.randn(4, 8),
            "video_grid_thw": torch.tensor([[1, 4, 4]]),
        }

        pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw = model._pop_staged_vlm_media(
            torch.tensor([[10, 11, 12]]),
            kwargs,
        )

        assert pixel_values is None
        assert pixel_values_videos is None
        assert image_grid_thw is None
        assert video_grid_thw is None


class TestQwen3_5ModelForward:
    def _build_inner_model(self):
        model = Qwen3_5Model.__new__(Qwen3_5Model)
        nn.Module.__init__(model)
        model.visual = types.SimpleNamespace(rotary_pos_emb=types.SimpleNamespace(to=lambda device: None))
        model.get_input_embeddings = lambda: lambda input_ids: pytest.fail("media forward should keep input_ids")
        return model

    def test_media_forward_keeps_input_ids_for_hf_placeholder_mask(self, monkeypatch):
        model = self._build_inner_model()
        captured = {}
        sentinel = object()

        def _fake_hf_forward(self, **kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(HFQwen3_5Model, "forward", _fake_hf_forward)

        input_ids = torch.tensor([[5, 99, 7]])
        pixel_values = torch.randn(4, 8)
        out = model.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )

        assert out is sentinel
        assert captured["input_ids"] is input_ids
        assert captured["inputs_embeds"] is None
        assert captured["pixel_values"] is pixel_values

    def test_media_forward_accepts_hidden_states_as_input_ids(self, monkeypatch):
        model = self._build_inner_model()
        captured = {}
        sentinel = object()

        def _fake_hf_forward(self, **kwargs):
            captured.update(kwargs)
            return sentinel

        monkeypatch.setattr(HFQwen3_5Model, "forward", _fake_hf_forward)

        hidden_states = torch.randn(1, 3, 4)
        out = model.forward(
            input_ids=hidden_states,
            pixel_values=torch.randn(4, 8),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )

        assert out is sentinel
        assert captured["input_ids"] is None
        assert captured["inputs_embeds"] is hidden_states

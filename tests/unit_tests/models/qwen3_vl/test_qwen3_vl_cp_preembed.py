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

"""CPU tests for dense Qwen3-VL context-parallel pre-embedding."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
)

import nemo_automodel.components.models.qwen3_vl.cp_batch as cp_batch_module
import nemo_automodel.components.models.qwen3_vl.model as qwen3_vl_model_module
from nemo_automodel.components.models.qwen3_vl.cp_batch import make_qwen3_vl_cp_batch
from nemo_automodel.components.models.qwen3_vl.model import Qwen3VLForConditionalGeneration


def _bare_model() -> Qwen3VLForConditionalGeneration:
    """Build a lightweight Qwen3-VL instance with deterministic stubbed modules."""
    model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
    nn.Module.__init__(model)
    hidden = 4
    model.get_input_embeddings = lambda: (
        lambda input_ids: input_ids.unsqueeze(-1).expand(*input_ids.shape, hidden).float()
    )
    model.config = types.SimpleNamespace(
        image_token_id=91,
        video_token_id=92,
        text_config=types.SimpleNamespace(vocab_size=8),
    )
    model.model = types.SimpleNamespace(
        visual=types.SimpleNamespace(dtype=torch.float32, spatial_merge_size=2),
        rope_deltas=None,
    )

    def _placeholder_mask(input_ids, *, inputs_embeds, image_features=None, video_features=None):
        """Return image/video placeholder masks.

        Args:
            input_ids: Token ids of shape ``[batch, sequence]``.
            inputs_embeds: Token embeddings of shape ``[batch, sequence, hidden]``.
            image_features: Optional image features of shape ``[image_tokens, hidden]``.
            video_features: Optional video features of shape ``[video_tokens, hidden]``.

        Returns:
            Image and video masks, each shaped ``[batch, sequence, hidden]``.
        """
        image_mask = (input_ids == 91).unsqueeze(-1).expand_as(inputs_embeds)
        video_mask = (input_ids == 92).unsqueeze(-1).expand_as(inputs_embeds)
        return image_mask, video_mask

    def _positions(
        *,
        input_ids,
        inputs_embeds,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        mm_token_type_ids=None,
    ):
        """Return deterministic mRoPE positions.

        Args:
            input_ids: Token ids of shape ``[batch, sequence]``.
            inputs_embeds: Token embeddings of shape ``[batch, sequence, hidden]``.
            image_grid_thw: Optional image grids of shape ``[num_images, 3]``.
            video_grid_thw: Optional video grids of shape ``[num_videos, 3]``.
            attention_mask: Optional mask of shape ``[batch, sequence]``.
            mm_token_type_ids: Optional modality ids of shape ``[batch, sequence]``.

        Returns:
            Position ids of shape ``[3, batch, sequence]``.
        """
        del inputs_embeds, image_grid_thw, video_grid_thw, attention_mask, mm_token_type_ids
        batch, sequence = input_ids.shape
        return torch.arange(sequence).view(1, 1, sequence).expand(3, batch, sequence).clone()

    model.model.get_placeholder_mask = _placeholder_mask
    model.model.compute_3d_position_ids = _positions
    return model


def _tiny_config(*, tie_word_embeddings: bool = False) -> Qwen3VLConfig:
    """Create a tiny dense Qwen3-VL config suitable for CPU parity tests."""
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "use_cache": False,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "out_hidden_size": 16,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=30,
        video_token_id=31,
        tie_word_embeddings=tie_word_embeddings,
    )
    config.architectures = ["Qwen3VLForConditionalGeneration"]
    return config


def test_non_cp_text_forward_matches_hugging_face() -> None:
    """The registered model keeps the stock HF forward numerics outside CP."""
    torch.manual_seed(7)
    config = _tiny_config()
    reference = HFQwen3VLForConditionalGeneration(config).eval()
    actual = Qwen3VLForConditionalGeneration(config).eval()
    actual.load_state_dict(reference.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    position_ids = torch.arange(4).view(1, 1, 4).expand(3, 1, 4)

    with torch.no_grad():
        expected_logits = reference(input_ids=input_ids, position_ids=position_ids).logits
        actual_logits = actual(input_ids=input_ids, position_ids=position_ids).logits

    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-6, rtol=1e-6)


def test_real_vision_preembed_and_deepstack_forward_match_hugging_face() -> None:
    """The complete image pre-embed and DeepStack branch preserve HF logits."""
    torch.manual_seed(11)
    config = _tiny_config()
    reference = HFQwen3VLForConditionalGeneration(config).eval()
    actual = Qwen3VLForConditionalGeneration(config).eval()
    actual.load_state_dict(reference.state_dict())
    input_ids = torch.tensor([[1, config.image_token_id, 2, 3]], dtype=torch.long)
    position_ids = torch.arange(4).view(1, 1, 4).expand(3, 1, 4)
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    patch_dim = (
        config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
    )
    pixel_values = torch.randn(4, patch_dim)

    with torch.no_grad():
        expected_logits = reference(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits
        prepared = actual.prepare_model_inputs_for_cp(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        prepared.pop("_cp_make_batch_fn")
        actual_logits = actual(**prepared).logits

    torch.testing.assert_close(actual_logits, expected_logits, atol=1e-5, rtol=1e-5)


def test_tied_config_aliases_lm_head_and_text_embedding() -> None:
    """The BOTH tie policy preserves the HF Qwen3-VL tied checkpoint variant."""
    model = Qwen3VLForConditionalGeneration(_tiny_config(tie_word_embeddings=True))
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight


def test_mixed_image_video_preembed_uses_one_visual_forward(monkeypatch) -> None:
    """Mixed media is encoded once and returns sequence-ordered DeepStack inputs."""
    model = _bare_model()
    calls = []

    def _visual(visual, pixel_values, grid_thw):
        """Return deterministic pooled and DeepStack tensors.

        Args:
            visual: Stub vision tower receiving the call.
            pixel_values: Patch rows of shape ``[patch_rows, patch_dim]``.
            grid_thw: Media grids of shape ``[num_entries, 3]``.

        Returns:
            Output whose pooled and DeepStack tensors have shape
            ``[visual_tokens, hidden]``.
        """
        calls.append((visual, pixel_values, grid_thw))
        pooled = torch.arange(16, dtype=torch.float32).view(4, 4)
        return types.SimpleNamespace(
            pooler_output=pooled,
            deepstack_features=[pooled + 100, pooled + 200],
        )

    monkeypatch.setattr(qwen3_vl_model_module, "maybe_distribute_visual", _visual)
    input_ids = torch.tensor([[5, 91, 91, 7, 92, 92]])
    out = model.prepare_model_inputs_for_cp(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=torch.tensor([[0, 1, 1, 0, 2, 2]]),
        pixel_values=torch.randn(8, 6),
        pixel_values_videos=torch.randn(8, 6),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
        video_grid_thw=torch.tensor([[1, 2, 4]]),
    )

    assert len(calls) == 1
    assert calls[0][1].shape == (16, 6)
    assert calls[0][2].tolist() == [[1, 2, 4], [1, 2, 4]]
    assert out["visual_pos_masks"].tolist() == [[False, True, True, False, True, True]]
    torch.testing.assert_close(out["inputs_embeds"][0, 1:3], torch.arange(8).view(2, 4).float())
    torch.testing.assert_close(out["inputs_embeds"][0, 4:6], torch.arange(8, 16).view(2, 4).float())
    torch.testing.assert_close(out["_deepstack_visual_embeds"][0], torch.arange(16).view(4, 4).float() + 100)
    assert out["_cp_make_batch_fn"] is make_qwen3_vl_cp_batch
    assert out["position_ids"].shape == (3, 1, 6)


def test_preembed_requires_media_grid_pairs() -> None:
    """Incomplete patch/grid inputs fail before entering the vision tower."""
    model = _bare_model()
    with pytest.raises(ValueError, match="provided together"):
        model.prepare_model_inputs_for_cp(
            input_ids=torch.tensor([[1, 2]]),
            pixel_values=torch.randn(4, 6),
        )


class _FakeCpMesh:
    """Minimal CP-mesh surface used by the model-owned batch sharder."""

    mesh_dim_names = ("cp",)

    def __init__(self, rank: int) -> None:
        self.rank = rank

    def size(self) -> int:
        return 2

    def get_local_rank(self) -> int:
        return self.rank


def test_cp_batch_shards_deepstack_in_token_order_with_gradients(monkeypatch) -> None:
    """Both CP ranks together select every DeepStack row exactly once with autograd."""
    captured = []

    def _capture(_mesh, batch, **_kwargs):
        """Capture the already transformed batch instead of invoking CP collectives.

        Args:
            _mesh: Unused context-parallel mesh.
            batch: Batch containing sequence and DeepStack tensors.
            **_kwargs: Unused shared-sharder options.

        Returns:
            A placeholder context plus the unchanged batch mapping.
        """
        captured.append(batch)
        return types.SimpleNamespace(), batch

    monkeypatch.setattr(cp_batch_module, "make_cp_batch_and_ctx", _capture)
    deepstack = torch.tensor([[10.0], [20.0], [30.0]], requires_grad=True)
    losses = []
    expected_masks = [
        [[True, False, False, False]],
        [[True, False, True, False]],
    ]
    expected_values = [[10.0], [20.0, 30.0]]

    for rank in range(2):
        batch = {
            "inputs_embeds": torch.zeros(1, 5, 2, requires_grad=True),
            "labels": torch.zeros(1, 5, dtype=torch.long),
            "position_ids": torch.arange(5).view(1, 5),
            "visual_pos_masks": torch.tensor([[True, False, True, False, True]]),
            "_deepstack_visual_embeds": [deepstack],
        }
        _, local_batch = make_qwen3_vl_cp_batch(_FakeCpMesh(rank), None, batch)
        assert local_batch["visual_pos_masks"].tolist() == expected_masks[rank]
        assert local_batch["_deepstack_visual_embeds"][0].flatten().tolist() == expected_values[rank]
        losses.append(local_batch["_deepstack_visual_embeds"][0].sum())

    sum(losses).backward()
    torch.testing.assert_close(deepstack.grad, torch.ones_like(deepstack))
    assert len(captured) == 2


def test_cp_batch_rejects_a_visual_mask_that_does_not_match_the_sequence(monkeypatch) -> None:
    """Mismatched mask and embedding sequence extents fail before shared CP setup."""
    monkeypatch.setattr(cp_batch_module, "make_cp_batch_and_ctx", lambda *_args, **_kwargs: None)
    batch = {
        "inputs_embeds": torch.zeros(1, 5, 2),
        "labels": torch.zeros(1, 5, dtype=torch.long),
        "visual_pos_masks": torch.zeros(1, 4, dtype=torch.bool),
        "_deepstack_visual_embeds": [],
    }

    with pytest.raises(ValueError, match="must match inputs_embeds"):
        make_qwen3_vl_cp_batch(_FakeCpMesh(0), None, batch)


def test_cp_forward_passes_local_deepstack_to_language_model() -> None:
    """The post-shard branch bypasses vision and injects local DeepStack features."""
    model = _bare_model()
    captured = {}

    class _LanguageModel(nn.Module):
        def forward(self, *, inputs_embeds, visual_pos_masks, deepstack_visual_embeds, **kwargs):
            """Capture local DeepStack inputs and return hidden states.

            Args:
                inputs_embeds: Embeddings of shape ``[batch, local_sequence, hidden]``.
                visual_pos_masks: Visual mask of shape ``[batch, local_sequence]``.
                deepstack_visual_embeds: Per-layer tensors of shape
                    ``[local_visual_tokens, hidden]``.
                **kwargs: Remaining language-model forward arguments.

            Returns:
                Output with hidden states of shape ``[batch, local_sequence, hidden]``.
            """
            captured.update(
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                kwargs=kwargs,
            )
            return BaseModelOutputWithPast(last_hidden_state=inputs_embeds + 1)

    model.model.language_model = _LanguageModel()
    model.lm_head = nn.Linear(4, 8, bias=False)
    inputs_embeds = torch.randn(1, 4, 4)
    visual_mask = torch.tensor([[True, False, True, False]])
    deepstack = [torch.randn(2, 4)]
    out = model.forward(
        inputs_embeds=inputs_embeds,
        position_ids=torch.arange(4).view(1, 4),
        _qwen3_vl_cp_preembedded=True,
        visual_pos_masks=visual_mask,
        _deepstack_visual_embeds=deepstack,
    )

    assert captured["visual_pos_masks"] is visual_mask
    assert captured["deepstack_visual_embeds"] is deepstack
    assert out.logits.shape == (1, 4, 8)

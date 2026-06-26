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

from types import SimpleNamespace

import torch

from nemo_automodel.components.models.gemma4_moe.cp_attention import _boundary_allowed_mask
from nemo_automodel.components.models.gemma4_moe.cp_batch import _prepare_manual_cp_batch
from nemo_automodel.components.models.gemma4_moe.model import Gemma4UnifiedForConditionalGeneration


def test_gemma4_unified_capabilities_are_cp_only():
    caps = Gemma4UnifiedForConditionalGeneration.get_capabilities(SimpleNamespace())

    assert caps.supports_cp is True
    assert caps.supports_tp is False
    assert caps.supports_pp is False
    assert caps.supports_ep is False


def test_gemma4_unified_prepare_model_inputs_for_cp_binds_batch_sharder_without_preembedding():
    sentinel = object()
    fake_self = SimpleNamespace(_cp_shard_batch=sentinel)
    input_ids = torch.arange(8).view(1, 8)

    out = Gemma4UnifiedForConditionalGeneration.prepare_model_inputs_for_cp(
        fake_self,
        input_ids=input_ids,
        num_chunks=2,
    )

    assert out == {"input_ids": input_ids, "_cp_make_batch_fn": sentinel}


def test_gemma4_unified_prepare_model_inputs_for_cp_carries_vision_metadata():
    sentinel = object()
    fake_self = SimpleNamespace(_cp_shard_batch=sentinel)
    mm = torch.tensor([[0, 1, 1, 0, 2, 2]])

    out = Gemma4UnifiedForConditionalGeneration.prepare_model_inputs_for_cp(
        fake_self,
        input_ids=torch.arange(6).view(1, 6),
        mm_token_type_ids=mm,
    )

    assert out["_cp_make_batch_fn"] is sentinel
    assert out["mm_token_type_ids"] is mm
    assert out["_gemma4_vision_group_ids"].tolist() == [[-1, 0, 0, -1, 1, 1]]
    assert out["_cp_metadata_seq_dims"] == {"_gemma4_vision_group_ids": 1}
    assert out["_cp_metadata_pad_values"] == {"_gemma4_vision_group_ids": -1}


def test_gemma4_unified_prepare_model_inputs_for_cp_merges_image_features():
    sentinel = object()
    embedding = torch.nn.Embedding(100, 4)
    image_features = torch.tensor([[10.0, 11.0, 12.0, 13.0]])
    fake_self = SimpleNamespace(
        _cp_shard_batch=sentinel,
        config=SimpleNamespace(image_token_id=42),
        model=SimpleNamespace(get_input_embeddings=lambda: embedding),
        _get_text_pad_token_id=lambda: 0,
        get_image_features=lambda *args, **kwargs: SimpleNamespace(pooler_output=image_features),
    )
    input_ids = torch.tensor([[1, 42, 3]])
    pixel_values = torch.randn(1, 3, 8, 8)
    base_embeds = embedding(torch.tensor([[1, 0, 3]]))

    out = Gemma4UnifiedForConditionalGeneration.prepare_model_inputs_for_cp(
        fake_self,
        input_ids=input_ids,
        pixel_values=pixel_values,
    )

    assert "input_ids" not in out
    assert out["_cp_make_batch_fn"] is sentinel
    torch.testing.assert_close(out["inputs_embeds"][:, 0, :], base_embeds[:, 0, :])
    torch.testing.assert_close(out["inputs_embeds"][:, 1, :], image_features)
    torch.testing.assert_close(out["inputs_embeds"][:, 2, :], base_embeds[:, 2, :])
    assert out["mm_token_type_ids"].tolist() == [[0, 1, 0]]
    assert out["_gemma4_vision_group_ids"].tolist() == [[-1, 0, -1]]


def test_gemma4_unified_prepare_model_inputs_for_cp_requires_input_ids():
    fake_self = SimpleNamespace(_cp_shard_batch=object())

    try:
        Gemma4UnifiedForConditionalGeneration.prepare_model_inputs_for_cp(fake_self, input_ids=None)
    except ValueError as exc:
        assert "requires input_ids" in str(exc)
    else:
        raise AssertionError("expected prepare_model_inputs_for_cp to require input_ids")


def test_gemma4_unified_packed_attention_mask_mapping_applies_sliding_window():
    fake_self = SimpleNamespace(
        config=SimpleNamespace(image_token_id=42),
        _get_text_config=lambda: SimpleNamespace(sliding_window=2, _attn_implementation="sdpa"),
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)),
    )
    attention_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    packed_seq_ids = torch.tensor([[1, 1, 1, 1]])
    mm_token_type_ids = torch.zeros_like(packed_seq_ids)

    masks = Gemma4UnifiedForConditionalGeneration._prepare_packed_attention_mask_mapping(
        fake_self,
        attention_mask,
        packed_seq_ids,
        mm_token_type_ids,
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        inputs_embeds=None,
    )

    assert set(masks) == {"full_attention", "sliding_attention"}
    assert masks["full_attention"].dtype == torch.bool
    assert masks["full_attention"][0, 0, 3, 0]
    assert not masks["sliding_attention"][0, 0, 3, 0]
    assert masks["sliding_attention"][0, 0, 3, 2]


def test_gemma4_unified_packed_attention_mask_mapping_keeps_indexed_2d_mask():
    fake_self = SimpleNamespace()
    indexed_mask = torch.tensor([[1, 1, 2, 2]])

    out = Gemma4UnifiedForConditionalGeneration._prepare_packed_attention_mask_mapping(
        fake_self,
        indexed_mask,
        packed_seq_ids=indexed_mask,
        mm_token_type_ids=torch.zeros_like(indexed_mask),
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        inputs_embeds=None,
    )

    assert out is indexed_mask


def test_gemma4_unified_cp_boundary_mask_blocks_cross_document_attention():
    attention = SimpleNamespace(sliding_window=None)
    query = torch.empty(1, 1, 4, 1)
    ctx = SimpleNamespace(seq_global_start=4, seq_local=4, query=query, is_causal=True)
    q_ids = torch.tensor([[3, 3, 4, 4]])
    kv_ids = torch.tensor([[1, 1, 2, 2]])

    allowed = _boundary_allowed_mask(
        attention,
        ctx,
        packed_seq_ids_q=q_ids,
        packed_seq_ids_kv=kv_ids,
        padding_mask_q=None,
        padding_mask_kv=None,
        vision_group_ids_q=None,
        vision_group_ids_kv=None,
        kv_global_start=0,
        use_vision_bidirectional=False,
    )

    # The remote KV chunk is entirely in the causal past, so only document
    # boundaries can mask it. Since IDs do not match, no pair is visible.
    assert allowed.shape == (1, 4, 4)
    assert allowed.sum().item() == 0


def test_gemma4_unified_cp_batch_accepts_attention_mask_mapping_for_padding():
    full_attention = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    full_attention[..., 3, :] = False
    full_attention[..., :, 3] = False
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "labels": torch.tensor([[2, 3, 4, -100]]),
        "position_ids": torch.tensor([[0, 1, 2, 0]]),
        "attention_mask": {
            "full_attention": full_attention,
            "sliding_attention": full_attention.clone(),
        },
    }

    _prepare_manual_cp_batch(None, None, batch, loss_mask=None)

    assert batch["padding_mask"].tolist() == [[False, False, False, True]]

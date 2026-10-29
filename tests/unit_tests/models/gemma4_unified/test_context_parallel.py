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

"""Gemma4 Unified model-owned context parallelism, on CPU.

With ``cp_mesh`` unset the contiguous shard is the identity and no ring is installed. The CP forward
leaves all masking to the ring, so on a model whose sliding window covers the sequence it must
reproduce the Hugging Face forward exactly; the ring itself is covered by the GPU parity test in
``tests/functional_tests/context_parallel/run_gemma4_unified_cp.py``.
"""

import logging

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedVisionConfig,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextAttention

import nemo_automodel.components.models.gemma4_moe.model as gemma4_model
from nemo_automodel._transformers.capabilities import ModelSupports
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.gemma4_unified.model import Gemma4UnifiedForConditionalGeneration

IMAGE_TOKEN_ID = 100


def _model(attn_implementation="sdpa", **text_overrides):
    text_kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=8,
        attention_k_eq_v=True,
        use_bidirectional_attention="vision",
        num_kv_shared_layers=0,
        hidden_size_per_layer_input=0,
        pad_token_id=0,
    )
    text = Gemma4UnifiedTextConfig(**{**text_kwargs, **text_overrides})
    vision = Gemma4UnifiedVisionConfig(
        mm_embed_dim=32, output_proj_dims=32, patch_size=2, pooling_kernel_size=2, model_patch_size=4, mm_posemb_size=16
    )
    config = Gemma4UnifiedConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), audio_config=None)
    config.image_token_id = IMAGE_TOKEN_ID
    config._attn_implementation = attn_implementation
    config.get_text_config()._attn_implementation = attn_implementation
    torch.manual_seed(0)
    return Gemma4UnifiedForConditionalGeneration(config).eval()


def _attention(model):
    return [m for m in model.modules() if isinstance(m, Gemma4UnifiedTextAttention)]


def _image_inputs():
    ids = torch.randint(1, IMAGE_TOKEN_ID, (1, 24))
    ids[0, 6:10] = IMAGE_TOKEN_ID
    grid = torch.arange(4)
    return {
        "input_ids": ids,
        "pixel_values": torch.randn(1, 4, 48),
        "image_position_ids": torch.stack([grid % 2, grid // 2], dim=-1).unsqueeze(0),
    }


def test_capabilities_and_ring_attached_at_construction():
    model = _model()
    attention = _attention(model)
    assert len(attention) == 4
    assert all(callable(getattr(m, "setup_cp_attention", None)) for m in attention)
    assert not any(getattr(m, "_cp_uses_attention_hook", False) for m in attention)
    supports = ModelSupports(model)
    assert supports.supports_cp and supports.supports_cp_with_sequence_packing


def test_cp_attention_backends_follow_the_text_config():
    attention = _attention(_model(cp_sliding_attn_backend="fa", cp_full_attn_backend="ffpa"))
    assert all(m._gemma4_cp_sliding_backend == "fa" and m._gemma4_cp_use_ffpa for m in attention)
    with pytest.raises(ValueError, match="cp_sliding_attn_backend"):
        _model(cp_sliding_attn_backend="sdpa")
    with pytest.raises(ValueError, match="cp_full_attn_backend"):
        _model(cp_full_attn_backend="sdpa")


def test_cp_ffpa_backend_warns_when_the_kernel_is_unavailable(monkeypatch, caplog):
    from nemo_automodel.components.models.gemma4_moe import cp_attention

    monkeypatch.setattr(cp_attention, "_ffpa_varlen_ring_available", lambda: False)
    with caplog.at_level(logging.WARNING, logger=cp_attention.__name__):
        _model(cp_full_attn_backend="ffpa")
    assert "fall back to Flex" in caplog.text
    caplog.clear()
    monkeypatch.setattr(cp_attention, "_ffpa_varlen_ring_available", lambda: True)
    with caplog.at_level(logging.WARNING, logger=cp_attention.__name__):
        _model(cp_full_attn_backend="ffpa")
    assert "fall back to Flex" not in caplog.text


@pytest.mark.parametrize("with_image", [False, True])
def test_cp_forward_without_a_mesh_matches_the_hf_forward(with_image):
    model = _model(sliding_window=32)  # covers the sequence: without the ring every layer is causal
    inputs = _image_inputs() if with_image else {"input_ids": torch.randint(1, IMAGE_TOKEN_ID, (1, 24))}
    masks_seen = []
    model.model.language_model.register_forward_pre_hook(
        lambda module, args, kwargs: masks_seen.append(kwargs.get("attention_mask")), with_kwargs=True
    )
    with torch.no_grad():
        expected = model(**inputs).logits
        expected_last = model(**inputs, logits_to_keep=1).logits
        model._cp_enabled = True  # cp_mesh stays None: identity shard, no ring
        out = model(**inputs, labels=inputs["input_ids"], use_cache=True)
        last = model(**inputs, logits_to_keep=1).logits
    torch.testing.assert_close(out.logits, expected)
    torch.testing.assert_close(last, expected_last)
    assert out.loss is None  # the recipe computes the loss on the sharded labels
    # The ring owns the masks, so Hugging Face builds none for the shard.
    assert masks_seen[-1] == {"full_attention": None, "sliding_attention": None}


def test_non_cp_forward_builds_the_vision_masks_against_the_cache(monkeypatch):
    model = _model()
    inputs = _image_inputs()
    mm = (inputs["input_ids"] == IMAGE_TOKEN_ID).long()
    builder = model.create_masks_for_generate
    caches = []

    def spy(config, inputs_embeds, attention_mask, past_key_values, position_ids, **kwargs):
        caches.append(past_key_values)
        return builder(config, inputs_embeds, attention_mask, past_key_values, position_ids, **kwargs)

    monkeypatch.setattr(model, "create_masks_for_generate", spy)
    cache = DynamicCache(config=model.config.get_text_config())
    with torch.no_grad():
        model(**inputs, mm_token_type_ids=mm, past_key_values=cache, use_cache=True)
    assert caches == [cache]


def test_non_cp_forward_vision_masks_from_inputs_embeds():
    model = _model()
    # Hugging Face looks up every placeholder embedding on this path; keep them inside the tiny vocab.
    model.config.video_token_id, model.config.audio_token_id = IMAGE_TOKEN_ID + 1, IMAGE_TOKEN_ID + 2
    ids = torch.randint(1, IMAGE_TOKEN_ID, (1, 24))
    mm = torch.zeros_like(ids)
    mm[0, 6:10] = 1
    hf_forward = super(Gemma4UnifiedForConditionalGeneration, model).forward
    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids)
        masks = model.create_masks_for_generate(model.config, embeds, None, None, None, mm_token_type_ids=mm)
        expected = hf_forward(inputs_embeds=embeds, mm_token_type_ids=mm, attention_mask=masks).logits
        out = model(inputs_embeds=embeds, mm_token_type_ids=mm).logits
    torch.testing.assert_close(out, expected)


def test_non_cp_forward_uses_the_generation_vision_masks():
    model = _model()
    inputs = _image_inputs()
    mm = (inputs["input_ids"] == IMAGE_TOKEN_ID).long()
    hf_forward = super(Gemma4UnifiedForConditionalGeneration, model).forward
    with torch.no_grad():
        embeds = model.get_input_embeddings()(inputs["input_ids"].masked_fill(mm.bool(), 0))
        masks = model.create_masks_for_generate(model.config, embeds, None, None, None, mm_token_type_ids=mm)
        expected = hf_forward(**inputs, mm_token_type_ids=mm, attention_mask=masks).logits
        hf_training = hf_forward(**inputs, mm_token_type_ids=mm).logits
        out = model(**inputs, mm_token_type_ids=mm).logits
    torch.testing.assert_close(out, expected)
    # Hugging Face's training forward also opens the image block on the full-attention layers.
    assert not torch.allclose(out, hf_training)


def test_cp_forward_stashes_ring_metadata_on_hooked_modules():
    model = _model()
    attention = _attention(model)
    for module in attention:
        module._cp_uses_attention_hook = True
    model._cp_enabled = True
    inputs = _image_inputs()
    mm = (inputs["input_ids"] == IMAGE_TOKEN_ID).long()
    padding_mask = torch.zeros_like(mm, dtype=torch.bool)
    doc_ids = torch.ones_like(mm)
    with torch.no_grad():
        model(**inputs, mm_token_type_ids=mm, padding_mask=padding_mask, _packed_seq_ids=doc_ids)
    meta = attention[0]._cp_dense_metadata
    assert all(m._cp_dense_metadata is meta for m in attention)
    assert meta["_gemma4_has_vision_tokens"] is True
    torch.testing.assert_close(meta["mm_token_type_ids"], mm)
    expected_groups = torch.full_like(mm, -1)
    expected_groups[0, 6:10] = 0
    torch.testing.assert_close(meta["_gemma4_vision_group_ids"], expected_groups)
    assert meta["padding_mask"] is padding_mask and meta["_packed_seq_ids"] is doc_ids

    with torch.no_grad():
        model(input_ids=inputs["input_ids"])
    meta = attention[0]._cp_dense_metadata
    assert meta["_gemma4_has_vision_tokens"] is False
    assert meta["mm_token_type_ids"] is None and meta["_gemma4_vision_group_ids"] is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"inputs_embeds": torch.zeros(1, 4, 64)}, "requires input_ids"),
        (
            {"input_ids": torch.ones(1, 4, dtype=torch.long), "inputs_embeds": torch.zeros(1, 4, 64)},
            "requires input_ids",
        ),
        ({"input_ids": torch.ones(1, 4, dtype=torch.long), "mm_encoder_outputs": {}}, "mm_encoder_outputs"),
        (
            {"input_ids": torch.ones(1, 4, dtype=torch.long), "pixel_values_videos": torch.zeros(1)},
            "not video or audio",
        ),
        ({"input_ids": torch.ones(1, 4, dtype=torch.long), "input_features": torch.zeros(1)}, "not video or audio"),
    ],
)
def test_cp_forward_rejects_unsupported_inputs(kwargs, match):
    model = _model()
    model._cp_enabled = True
    with pytest.raises(ValueError, match=match):
        model(**kwargs)


def test_setup_cp_attention_installs_the_ring_once(monkeypatch):
    model = _model()
    calls = []
    for module in _attention(model):
        monkeypatch.setattr(module, "setup_cp_attention", lambda mesh, m=module: calls.append((m, mesh)))
    model.setup_cp_attention("mesh")
    model.setup_cp_attention("mesh")
    assert len(calls) == 4 and all(mesh == "mesh" for _, mesh in calls)
    assert model._cp_enabled


def test_setup_cp_attention_requires_sdpa():
    model = _model(attn_implementation="eager")
    with pytest.raises(ValueError, match="attn_implementation='sdpa'"):
        model.setup_cp_attention("mesh")
    assert not getattr(model, "_cp_enabled", False)


def test_setup_cp_attention_rejects_fully_bidirectional_attention():
    model = _model(use_bidirectional_attention="all")
    with pytest.raises(ValueError, match="use_bidirectional_attention='all'"):
        model.setup_cp_attention("mesh")
    assert not getattr(model, "_cp_enabled", False)


def test_cp_forward_rejects_image_tokens_that_do_not_match_the_features():
    model = _model()
    model._cp_enabled = True
    inputs = _image_inputs()
    inputs["input_ids"][0, 9] = 1  # a truncated image keeps 3 of its 4 placeholder tokens
    with pytest.raises(ValueError, match="Image features and image tokens do not match"):
        model(**inputs)


def test_cp_sharder_records_the_mesh_and_shards_aux_streams(monkeypatch):
    model = _model()
    monkeypatch.setattr(model, "setup_cp_attention", lambda mesh: setattr(model, "installed", mesh))
    seen = {}

    def fake_shard(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0):
        seen.update(cp_mesh=cp_mesh, tp_mesh=tp_mesh, batch=batch, loss_mask=loss_mask, pad=padding_token_id)
        return "ctx", batch

    monkeypatch.setattr(gemma4_model, "make_contiguous_aux_only_shard_cp_batch_and_ctx", fake_shard)
    batch = {"input_ids": torch.ones(1, 4, dtype=torch.long)}
    prepared = model.prepare_model_inputs_for_cp(batch, num_chunks=3)
    sharder = prepared["cp_sharder"]
    assert set(prepared) == {"cp_sharder"} and isinstance(sharder, ContextParallelSharder)
    assert sharder.shard_batch("cp", "tp", batch, loss_mask="mask", padding_token_id=7) == ("ctx", batch)
    assert model.installed == "cp" and model.cp_mesh == "cp"
    assert seen == {"cp_mesh": "cp", "tp_mesh": "tp", "batch": batch, "loss_mask": "mask", "pad": 7}


def test_prepare_model_inputs_for_cp_requires_input_ids():
    with pytest.raises(ValueError, match="requires input_ids"):
        _model().prepare_model_inputs_for_cp({"inputs_embeds": torch.zeros(1, 4, 64)})

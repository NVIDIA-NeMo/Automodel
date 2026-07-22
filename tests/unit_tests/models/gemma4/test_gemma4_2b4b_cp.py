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

"""Coverage for the small dense Gemma4 (E2B/E4B) context-parallel paths.

E2B/E4B differ from plain-dense 31B in two ways that must flow through the
model-owned CP ring: KV-sharing (the last N decoder layers reuse an earlier
layer's keys/values) and per-layer inputs (``hidden_size_per_layer_input``).
These tests exercise the CPU-testable seams that those two features touch:
contiguous batch sharding of the 4D ``per_layer_inputs`` tensor, the
capability flip that turns CP on for the dense+audio variant, and the
per-layer-input threading through ``prepare_model_inputs_for_cp``.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel._strategy import shard_batch_contiguous
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
)


@pytest.fixture(autouse=True)
def _force_non_distributed(monkeypatch):
    """Pin the single-process (non-distributed) contiguous-shard path.

    These are CPU unit tests driven by a fake CP mesh. When the full L0 suite
    leaves ``torch.distributed`` initialized (test-order pollution), the
    contiguous-shard code would take its distributed branch and call
    ``cp_mesh.get_group()`` on the fake mesh. Forcing ``is_initialized`` False
    keeps the intended non-distributed branch regardless of suite ordering.
    """
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


# ---------------------------------------------------------------------------
# Config + model helpers (small dense E2B/E4B-like)
# ---------------------------------------------------------------------------
def _text_config(**overrides):
    """A tiny dense Gemma4 text config with KV-sharing + per-layer inputs."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=1,  # extreme GQA like E2B/E4B
        head_dim=16,
        num_hidden_layers=6,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        enable_moe_block=False,
        layer_types=["sliding_attention", "full_attention"] * 3,
        sliding_window=32,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="float32",
        # E2B/E4B-specific: KV-sharing on the trailing layers + per-layer inputs.
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=256,
        use_double_wide_mlp=False,
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def _cfg(audio=True, **text_overrides):
    cfg = Gemma4Config(text_config=_text_config(**text_overrides))
    # The dense+audio variant (E2B/E4B) is keyed off a non-None audio_config.
    cfg.audio_config = {} if audio else None
    return cfg


def _backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class _FakeCPMesh:
    """Minimal CP mesh for the non-distributed contiguous-shard path."""

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank


# ---------------------------------------------------------------------------
# get_capabilities: dense+audio (E2B/E4B) now supports CP
# ---------------------------------------------------------------------------
def test_get_capabilities_dense_audio_variant_enables_cp():
    caps = Gemma4ForConditionalGeneration.get_capabilities(_cfg(audio=True))
    assert caps.supports_cp is True
    # TP/PP/EP stay off for the dense+audio variant.
    assert caps.supports_tp is False
    assert caps.supports_pp is False
    assert caps.supports_ep is False


def test_get_capabilities_plain_dense_unchanged():
    # No audio_config -> plain dense 31B path keeps full TP/CP/PP.
    caps = Gemma4ForConditionalGeneration.get_capabilities(_cfg(audio=False))
    assert caps.supports_cp is True
    assert caps.supports_tp is True
    assert caps.supports_pp is True


# ---------------------------------------------------------------------------
# cp_batch: per_layer_inputs (4D [B, S, L, D]) is sharded on the seq dim
# ---------------------------------------------------------------------------
def _shard_batch(batch, cp_size, cp_rank, seq_len):
    mesh = _FakeCPMesh(cp_size, cp_rank)
    # Exercise the public contiguous shard directly on the 4D per_layer_inputs key
    # (the production entry the model wraps), the same slice the sunk forward applies.
    prepared = shard_batch_contiguous(
        mesh,
        None,
        batch,
        loss_mask=None,
        padding_token_id=0,
        extra_seq_keys={"per_layer_inputs": 1},
        extra_pad_values={"per_layer_inputs": 0},
    )
    return prepared.batch


def test_cp_batch_shards_per_layer_inputs_on_seq_dim():
    cp_size, seq, n_layers, pli = 2, 8, 6, 8
    base = {
        "input_ids": torch.arange(seq).view(1, seq) + 1,
        "labels": torch.arange(seq).view(1, seq),
        # per_layer_inputs is the distinguishing E2B/E4B 4D tensor.
        "per_layer_inputs": torch.arange(seq * n_layers * pli, dtype=torch.float32).view(1, seq, n_layers, pli),
    }
    local = seq // cp_size
    shards = [_shard_batch({k: v.clone() for k, v in base.items()}, cp_size, r, seq) for r in range(cp_size)]

    for r, sh in enumerate(shards):
        pli_t = sh["per_layer_inputs"]
        # seq dim (dim 1) is sharded; the layer + feature dims are untouched.
        assert pli_t.shape == (1, local, n_layers, pli)
        expected = base["per_layer_inputs"][:, r * local : (r + 1) * local]
        assert torch.equal(pli_t, expected)


def test_cp_batch_per_layer_inputs_concatenates_back_to_full_sequence():
    # seq divisible by 2*cp_size so no padding is inserted before sharding.
    cp_size, seq, n_layers, pli = 4, 16, 6, 8
    full = torch.randn(1, seq, n_layers, pli)
    base = {
        "input_ids": torch.arange(seq).view(1, seq) + 1,
        "labels": torch.arange(seq).view(1, seq),
        "per_layer_inputs": full,
    }
    pieces = [
        _shard_batch({k: v.clone() for k, v in base.items()}, cp_size, r, seq)["per_layer_inputs"]
        for r in range(cp_size)
    ]
    # Reassembling the contiguous shards recovers the original full tensor.
    assert torch.equal(torch.cat(pieces, dim=1), full)


def test_cp_batch_pads_per_layer_inputs_for_indivisible_seq():
    # seq=6, cp_size=4 -> pad to multiple of 2*cp_size=8 before sharding.
    cp_size, seq, n_layers, pli = 4, 6, 6, 8
    base = {
        "input_ids": torch.arange(seq).view(1, seq) + 1,
        "labels": torch.arange(seq).view(1, seq),
        "per_layer_inputs": torch.randn(1, seq, n_layers, pli),
    }
    sh = _shard_batch(base, cp_size, 0, seq)
    # padded length 8 / cp_size 4 = 2 local tokens per rank.
    assert sh["per_layer_inputs"].shape == (1, 2, n_layers, pli)


# ---------------------------------------------------------------------------
# KV-sharing structure: the trailing layers are flagged as shared and resolve
# their source layer (mirrors HF's first_kv_shared_layer_idx math).
# ---------------------------------------------------------------------------
def test_kv_shared_layers_resolve_source_layer():
    model = Gemma4ForConditionalGeneration(_cfg(audio=False), backend=_backend())
    tc = model.config.text_config
    first_shared = tc.num_hidden_layers - tc.num_kv_shared_layers
    layers = model.model.language_model.layers
    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        if idx >= first_shared:
            assert attn.is_kv_shared_layer is True
            # The source-layer index is an HF-internal attribute whose name/presence
            # varies across transformers versions (exposed as `kv_shared_layer_index`
            # in 5.5, absent in 5.8). When present, verify it points back to an
            # earlier same-type layer.
            shared_idx = getattr(attn, "kv_shared_layer_index", None)
            if shared_idx is not None:
                assert shared_idx < first_shared
                assert tc.layer_types[shared_idx] == tc.layer_types[idx]
        else:
            assert attn.is_kv_shared_layer is False


def test_dense_init_attaches_ring_to_all_self_attention_incl_shared():
    model = Gemma4ForConditionalGeneration(_cfg(audio=False), backend=_backend())
    hooked = [m for m in model.modules() if getattr(m, "_cp_manual_metadata_keys", None)]
    # Every decoder layer's attention -- shared and non-shared alike -- gets the ring,
    # so shared layers still rotate their reused K/V through the CP transport.
    assert len(hooked) == model.config.text_config.num_hidden_layers


# ---------------------------------------------------------------------------
# per_layer_inputs threading through prepare_model_inputs_for_cp
# ---------------------------------------------------------------------------
def test_prepare_model_inputs_is_sharder_only_for_e_series():
    # Sunk CP: the E-series hook is sharder-only too (per_layer_inputs is built +
    # sliced in the forward, not pre-embedded here). audio=False avoids building
    # the audio tower; the per-layer-input machinery lives in text_config.
    cfg = _cfg(audio=False)
    cfg.image_token_id = 99
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.float32)
    prepared = model.prepare_model_inputs_for_cp({"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])})
    assert prepared.shard_batch == model._cp_shard_batch_aux_only


def test_cp_sunk_prepare_inputs_slices_per_layer_inputs_on_seq_dim():
    # The E-series 4D per_layer_inputs is built on the full sequence in-forward and
    # contiguously sliced on the seq dim alongside inputs_embeds (Megatron-style).
    cfg = _cfg(audio=False)
    cfg.image_token_id = 99
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.float32)
    tc = cfg.text_config
    cp_size, seq = 2, 8  # divisor 2*cp = 4 -> no pad; local = 4
    model.cp_mesh = _FakeCPMesh(cp_size, 0)
    prepared = model._cp_sunk_prepare_inputs(
        input_ids=torch.arange(seq).view(1, seq) + 1,
        pixel_values=None,
        image_position_ids=None,
        mm_token_type_ids=None,
    )
    local = seq // cp_size
    assert prepared["inputs_embeds"].shape == (1, local, tc.hidden_size)
    assert prepared["per_layer_inputs"].shape == (1, local, tc.num_hidden_layers, tc.hidden_size_per_layer_input)
    # ring metadata is sliced to the same local length
    assert prepared["mm_token_type_ids"].shape == (1, local)
    assert prepared["_gemma4_vision_group_ids"].shape == (1, local)


def test_prepare_per_layer_inputs_masks_image_tokens():
    cfg = _cfg(audio=False)
    cfg.image_token_id = 99
    model = Gemma4ForConditionalGeneration(cfg, backend=_backend()).to(torch.float32)
    ids = torch.tensor([[1, 99, 3, 99, 5, 6, 7, 8]])
    special = ids == 99
    captured = {}

    real_get = model.model.language_model.get_per_layer_inputs

    def spy(llm_ids, embeds):
        captured["ids"] = llm_ids.clone()
        return real_get(llm_ids, embeds)

    with mock.patch.object(model.model.language_model, "get_per_layer_inputs", side_effect=spy):
        model._prepare_per_layer_inputs_for_cp(ids, special)
    # image-token positions are replaced by the pad id before computing per-layer inputs
    pad = model._get_text_pad_token_id()
    assert torch.equal(captured["ids"], ids.masked_fill(special, pad))


def test_prepare_per_layer_inputs_none_without_feature():
    fake = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(hidden_size_per_layer_input=0)))
    out = Gemma4ForConditionalGeneration._prepare_per_layer_inputs_for_cp(
        fake, torch.tensor([[1, 2, 3, 4]]), torch.zeros(1, 4, dtype=torch.bool)
    )
    assert out is None

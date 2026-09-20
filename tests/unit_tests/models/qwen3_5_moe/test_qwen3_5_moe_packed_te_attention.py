# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the packed TransformerEngine attention route.

TransformerEngine has no mask representation for per-document causality, so a
packed batch must reach it as ``qkv_format="thd"`` plus ``cu_seqlens`` over the
unpadded token stream. These tests stub the backend attention function to assert
the tensor contract without needing TE or a GPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.models.qwen3_5_moe")

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5.packing import prepare_gated_delta_packed_metadata
from nemo_automodel.components.models.qwen3_5_moe.model import (
    Qwen3_5MoeBlock,
    _packed_document_ids,
    _Qwen3_5MoeAttention,
    _rolled_embed_inputs,
)


def _build_attention(recorded: dict) -> _Qwen3_5MoeAttention:
    """Build a bare attention module whose backend call records its inputs."""
    attn = _Qwen3_5MoeAttention.__new__(_Qwen3_5MoeAttention)
    nn.Module.__init__(attn)
    attn.backend = BackendConfig(attn="te")
    attn._packed_te_metadata = None

    def _record(query, key, value, **kwargs):
        recorded["q"] = query
        recorded["k"] = key
        recorded["v"] = value
        recorded["kwargs"] = kwargs
        # TE returns [tokens, heads, head_dim] for qkv_format="thd"; make the values
        # position-dependent so a mis-scatter is visible in the repadded output.
        return query.clone()

    attn._base_attn_func = _record
    attn.attn_func = attn._dispatch_attention
    return attn


class TestPackedTEAttention:
    """Cover the unpad -> cu_seqlens -> repad contract."""

    def test_unpads_and_builds_cu_seqlens(self):
        """Two documents plus padding collapse to a dense stream with cu_seqlens."""
        recorded: dict = {}
        attn = _build_attention(recorded)
        # Row holds document 1 (2 tokens), document 2 (3 tokens), then 1 padding token.
        document_ids = torch.tensor([[1, 1, 2, 2, 2, 0]], dtype=torch.long)
        metadata = prepare_gated_delta_packed_metadata(document_ids, None)

        query = torch.arange(1 * 6 * 2 * 4, dtype=torch.float32).reshape(1, 6, 2, 4)
        out = attn._packed_te_attention(query, query.clone(), query.clone(), metadata)

        assert recorded["kwargs"]["qkv_format"] == "thd"
        assert recorded["kwargs"]["attn_mask_type"] == "padding_causal"
        assert recorded["kwargs"]["cu_seqlens_q"].tolist() == [0, 2, 5]
        assert recorded["kwargs"]["cu_seqlens_kv"].tolist() == [0, 2, 5]
        # Longest document is 3 tokens; TE reads out of bounds if this is too small.
        assert recorded["kwargs"]["max_seqlen_q"] == 3
        assert recorded["kwargs"]["max_seqlen_kv"] == 3
        # Padding is gone: 5 valid tokens, heads and head_dim preserved.
        assert tuple(recorded["q"].shape) == (5, 2, 4)
        assert out.shape == query.shape

    def test_repad_restores_valid_positions_and_zeroes_padding(self):
        """Valid positions round-trip unchanged; the padded tail stays zero."""
        recorded: dict = {}
        attn = _build_attention(recorded)
        document_ids = torch.tensor([[1, 1, 2, 2, 2, 0]], dtype=torch.long)
        metadata = prepare_gated_delta_packed_metadata(document_ids, None)

        query = torch.arange(1 * 6 * 2 * 4, dtype=torch.float32).reshape(1, 6, 2, 4)
        out = attn._packed_te_attention(query, query.clone(), query.clone(), metadata)

        # The stub is the identity, so every valid position must survive the
        # unpad/repad round trip bit-for-bit.
        assert torch.equal(out[0, :5], query[0, :5])
        assert torch.all(out[0, 5] == 0)

    def test_padding_is_dropped_between_documents(self):
        """Interior padding is removed, so cu_seqlens describes a dense stream."""
        recorded: dict = {}
        attn = _build_attention(recorded)
        # Document 1 (2 tokens), padding, document 2 (2 tokens).
        document_ids = torch.tensor([[1, 1, 0, 2, 2]], dtype=torch.long)
        metadata = prepare_gated_delta_packed_metadata(document_ids, None)

        query = torch.arange(1 * 5 * 1 * 2, dtype=torch.float32).reshape(1, 5, 1, 2)
        out = attn._packed_te_attention(query, query.clone(), query.clone(), metadata)

        assert tuple(recorded["q"].shape) == (4, 1, 2)
        assert recorded["kwargs"]["cu_seqlens_q"].tolist() == [0, 2, 4]
        # The interior padding slot must come back zeroed, not filled from a neighbour.
        assert torch.all(out[0, 2] == 0)
        assert torch.equal(out[0, 3:], query[0, 3:])

    def test_explicit_mask_is_rejected(self):
        """An explicit mask alongside cu_seqlens is ambiguous and must raise."""
        recorded: dict = {}
        attn = _build_attention(recorded)
        document_ids = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
        metadata = prepare_gated_delta_packed_metadata(document_ids, None)
        query = torch.zeros(1, 4, 1, 2)

        with pytest.raises(ValueError, match="cu_seqlens"):
            attn._packed_te_attention(query, query.clone(), query.clone(), metadata, attention_mask=torch.ones(1, 4))

    def test_unpacked_batch_bypasses_the_thd_route(self):
        """Without packed metadata the dispatch falls through to the plain backend call."""
        recorded: dict = {}
        attn = _build_attention(recorded)
        query = torch.zeros(1, 4, 1, 2)

        attn._dispatch_attention(query, query.clone(), query.clone(), window_size=(-1, 0))

        assert "qkv_format" not in recorded["kwargs"]
        assert tuple(recorded["q"].shape) == (1, 4, 1, 2)


class TestBlockRoutesPackedTE:
    """The block must never hand TE an indexed document map."""

    def _block(self, attn_backend: str):
        block = Qwen3_5MoeBlock.__new__(Qwen3_5MoeBlock)
        nn.Module.__init__(block)
        block.layer_type = "full_attention"
        block.input_layernorm = nn.Identity()
        block.post_attention_layernorm = nn.Identity()
        recorded: dict = {}

        class _SelfAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.backend = BackendConfig(attn=attn_backend)

            def forward(self, **kwargs):
                recorded["kwargs"] = kwargs
                return kwargs["x"]

        block.self_attn = _SelfAttn()
        block.linear_attn = nn.Identity()
        block._mlp = lambda *, x, padding_mask: torch.zeros_like(x)
        return block, recorded

    def test_te_packed_batch_swaps_mask_for_metadata(self):
        """TE gets packed_te_metadata and a None mask, not the indexed map."""
        block, recorded = self._block("te")
        indexed = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)

        block(
            torch.zeros(1, 5, 4),
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=indexed,
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
        )

        assert recorded["kwargs"]["attention_mask"] is None
        metadata = recorded["kwargs"]["packed_te_metadata"]
        assert metadata is not None
        assert metadata.cu_seqlens.tolist() == [0, 2, 4]

    def test_te_unpacked_batch_keeps_its_padding_mask(self):
        """A plain 0/1 mask is not a packing map and must reach TE unchanged."""
        block, recorded = self._block("te")
        mask = torch.ones(1, 5, dtype=torch.long)

        block(
            torch.zeros(1, 5, 4),
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=mask,
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
        )

        assert recorded["kwargs"]["attention_mask"] is mask
        assert "packed_te_metadata" not in recorded["kwargs"]

    def test_sdpa_packed_batch_is_untouched(self):
        """SDPA consumes the explicit mask, so the TE rerouting must not apply."""
        block, recorded = self._block("sdpa")
        sdpa_mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()

        block(
            torch.zeros(1, 5, 4),
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=sdpa_mask,
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
        )

        assert recorded["kwargs"]["attention_mask"] is sdpa_mask
        assert "packed_te_metadata" not in recorded["kwargs"]


class TestPackedMTPEmbeds:
    """MTP future-token embeddings must not cross a packed document boundary."""

    def test_rolled_embeds_stop_at_document_boundaries(self):
        """Depth-1 embeddings are zero at each document's last token."""
        document_ids = torch.tensor([[1, 1, 2, 2, 2]], dtype=torch.long)
        # Distinct per-position values so a boundary leak is visible.
        embeds = torch.arange(1 * 5 * 1, dtype=torch.float32).reshape(1, 5, 1) + 1.0

        (depth1,) = _rolled_embed_inputs(embeds, 1, document_ids)

        # Position 1 is the last token of document 1: its depth-1 source would be
        # position 2, the first token of document 2 -> must be zeroed.
        assert depth1[0, 1, 0].item() == 0.0
        # Position 4 is the last token of document 2 and of the row.
        assert depth1[0, 4, 0].item() == 0.0
        # Within a document the shift is the plain next token.
        assert depth1[0, 0, 0].item() == embeds[0, 1, 0].item()
        assert depth1[0, 2, 0].item() == embeds[0, 3, 0].item()

    def test_unpacked_rolled_embeds_are_unchanged(self):
        """Without a document map the original cumulative roll is preserved."""
        embeds = torch.arange(1 * 4 * 1, dtype=torch.float32).reshape(1, 4, 1) + 1.0

        (depth1,) = _rolled_embed_inputs(embeds, 1, None)

        assert depth1[0, 0, 0].item() == embeds[0, 1, 0].item()
        assert depth1[0, 3, 0].item() == 0.0

    def test_packed_document_ids_detection(self):
        """Only an indexed multi-document map counts as packed."""
        indexed = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
        assert _packed_document_ids(indexed, None) is indexed
        # A 4-D backend mask carries no document ids; they arrive alongside it.
        sdpa_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
        assert _packed_document_ids(sdpa_mask, indexed) is indexed
        # A plain validity mask is not a packing map.
        assert _packed_document_ids(torch.ones(1, 4, dtype=torch.long), None) is None
        assert _packed_document_ids(None, None) is None

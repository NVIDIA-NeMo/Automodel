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

"""Packed rows of ERNIE 5.5: no attention layer reads across a document boundary.

The models are small FP32 models on the eager attention path. Their indexers have 32 heads, so that index scores
have no ties and the selected keys of a document do not depend on where the document sits in a row.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate
from nemo_automodel.components.models.ernie5_5 import hca as hca_module
from nemo_automodel.components.models.ernie5_5.dsa import cudnn_dsa_layout
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from tests.unit_tests.models.ernie5_5.test_model import VOCAB, _tiny_model

pytestmark = pytest.mark.cpu_only

RATIO, WINDOW = 128, 8
# One document with a complete compression window, one shorter than the attention window allows to fill, and one
# shorter than a compression window. With every document moved to a multiple of 128 tokens the row has 512 tokens.
LENGTHS = (130, 37, 100)
STARTS = (0, 256, 384)
TIE_FREE_INDEXER = {"index_n_heads": 32, "index_head_dim": 16, "index_topk": 3}
# Largest difference measured between a packed row and its documents run one by one: 6.0e-8.
ATOL = 1e-6


def _model(ratios: tuple[int, ...]) -> Ernie5_5_MoeForCausalLM:
    """Build a seeded model whose sinks and virtual-head factors are nonzero, so that every path matters."""
    model = _tiny_model(ratios, sliding_window=WINDOW, **TIE_FREE_INDEXER)
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(("vha_postmix_V", "sinks_param.weight", "softmax_offset")):
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)
    return model.eval()


def _documents(seed: int = 0) -> list[torch.Tensor]:
    """Return one token tensor of shape [1, length] per entry of ``LENGTHS``, without the padding token."""
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(1, VOCAB, (1, length), generator=generator) for length in LENGTHS]


def _one_by_one(model: Ernie5_5_MoeForCausalLM, documents: list[torch.Tensor]) -> torch.Tensor:
    """Return the logits of every document run alone, concatenated to shape [tokens, vocab]."""
    return torch.cat([model(document).logits[0] for document in documents])


@pytest.mark.parametrize("ratios", [(128, 128), (-2, -2), (128, 128, -2, 128, -2)])
def test_packed_row_matches_its_documents_run_one_by_one(ratios):
    model, documents = _model(ratios), _documents()
    with torch.no_grad():
        packed = model(torch.cat(documents, dim=1), seq_lens=torch.tensor([LENGTHS])).logits[0]
        torch.testing.assert_close(packed, _one_by_one(model, documents), rtol=0, atol=ATOL)


def test_padding_between_documents_and_unused_length_entries_are_ignored():
    model, documents = _model((128, 128, -2)), _documents()
    pads = (3, 0, 9)
    row = torch.cat(
        [torch.cat((document, document.new_zeros(1, pad)), dim=1) for document, pad in zip(documents, pads)], dim=1
    )
    spans = [length + pad for length, pad in zip(LENGTHS, pads)]
    with torch.no_grad():
        packed = model(
            row, seq_lens=torch.tensor([[*LENGTHS, -1000]]), seq_lens_padded=torch.tensor([[*spans, -1000]])
        ).logits[0]
        expected = _one_by_one(model, documents)
    offset = expected_offset = 0
    for length, span in zip(LENGTHS, spans):
        torch.testing.assert_close(
            packed[offset : offset + length], expected[expected_offset : expected_offset + length], rtol=0, atol=ATOL
        )
        assert not packed[offset + length : offset + span].any()
        offset, expected_offset = offset + span, expected_offset + length


@pytest.mark.parametrize("ratios", [(128, 128), (-2, -2), (128, 128, -2, 128, -2)])
@pytest.mark.parametrize("changed", [0, 1, 2])
def test_changing_one_document_leaves_the_logits_of_the_others_unchanged(ratios, changed):
    model, documents = _model(ratios), _documents()
    other = _documents(seed=1)
    replaced = [other[index] if index == changed else document for index, document in enumerate(documents)]
    seq_lens = torch.tensor([LENGTHS])
    with torch.no_grad():
        before = model(torch.cat(documents, dim=1), seq_lens=seq_lens).logits[0].split(LENGTHS)
        after = model(torch.cat(replaced, dim=1), seq_lens=seq_lens).logits[0].split(LENGTHS)
    for index in range(len(LENGTHS)):
        assert torch.equal(before[index], after[index]) == (index != changed)


@pytest.mark.parametrize("ratios", [(128, 128), (-2, -2)])
def test_a_row_without_document_boundaries_changes_the_later_documents(ratios):
    """Control of the comparisons above: read as one document, the same row gives other logits after the first one."""
    model, documents = _model(ratios), _documents()
    with torch.no_grad():
        merged = model(torch.cat(documents, dim=1)).logits[0].split(LENGTHS)
        expected = _one_by_one(model, documents).split(LENGTHS)
    torch.testing.assert_close(merged[0], expected[0], rtol=0, atol=ATOL)
    for index in (1, 2):
        assert (merged[index] - expected[index]).abs().max() > 1e-3


def test_gradients_of_a_packed_row_equal_the_sum_over_its_documents():
    model, documents = _model((128, 128, -2, 128, -2)), _documents()
    labels = [torch.randint(1, VOCAB, (length,), generator=torch.Generator().manual_seed(3)) for length in LENGTHS]
    for document, target in zip(documents, labels):
        F.cross_entropy(model(document).logits[0], target, reduction="sum").backward()
    expected = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    model.zero_grad()
    logits = model(torch.cat(documents, dim=1), seq_lens=torch.tensor([LENGTHS])).logits[0]
    F.cross_entropy(logits, torch.cat(labels), reduction="sum").backward()
    assert len(expected) > 100
    for name, parameter in model.named_parameters():
        if name in expected:
            # Largest difference measured: 9.0e-5 of the largest entry of a gradient, in a tensor whose entries
            # cancel to 1e-5.
            assert (parameter.grad - expected[name]).abs().max() <= 1e-3 * expected[name].abs().max(), name
        else:
            assert parameter.grad is None, name


def test_hca_reads_the_window_and_the_complete_windows_of_the_query_document(monkeypatch):
    """Rows that name their own offset show which keys the sparse attention of a packed row receives."""
    attention = _model((128,)).model.layers[0].attention
    sequence, head_dim = STARTS[-1] + RATIO, attention.head_dim
    position_ids = torch.zeros(1, sequence, dtype=torch.int64)
    packed_seq_ids = torch.zeros(1, sequence, dtype=torch.int64)
    for document, (start, length) in enumerate(zip(STARTS, LENGTHS), start=1):
        position_ids[0, start : start + length] = torch.arange(length)
        packed_seq_ids[0, start : start + length] = document
    offsets = torch.arange(sequence, dtype=torch.float32)
    # Key i holds i in its first channel. The compressor pools (offset, 1) of every token uniformly, so the ratio of
    # the first two channels of a compressed key is the mean offset of its 128 tokens.
    keys = torch.zeros(1, sequence, head_dim)
    keys[0, :, 0] = offsets
    hidden = torch.zeros(1, sequence, attention.config.hidden_size)
    hidden[0, :, 0], hidden[0, :, 1] = offsets, 1.0
    with torch.no_grad():
        attention.compressor.wgate.weight.zero_()
        attention.compressor.wkv.weight.zero_()
        attention.compressor.wkv.weight[0, 0] = attention.compressor.wkv.weight[1, 1] = 1.0
    seen = {}

    def record(query, kv, sinks, indices, scale, **kwargs):
        seen.update(kv=kv, indices=indices, kwargs=kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(hca_module, "dsv4_sparse_attention", record)
    rotary = _model((128,)).model._hca_rotary(torch.device("cpu"))
    rope = rotary(torch.empty(1, sequence, 1), position_ids)
    query = torch.zeros(1, attention.num_heads, sequence, head_dim)
    q_residual = torch.zeros(1, sequence, attention.config.q_lora_rank)
    with torch.no_grad():
        attention._attend_packed(hidden, q_residual, query, keys, rope, rotary, position_ids, packed_seq_ids)

    kv, indices = seen["kv"][0], seen["indices"][0]
    assert seen["kwargs"] == {"backend": "torch", "all_rows_nonempty": True}
    names = torch.where(kv[:, 1] != 0, kv[:, 0] / kv[:, 1], kv[:, 0])
    for start, length in zip(STARTS, LENGTHS):
        for position in (0, 1, WINDOW - 1, WINDOW, RATIO - 2, RATIO - 1, RATIO, length - 1):
            if position >= length:
                continue
            read = sorted(names[index].item() for index in indices[start + position] if index >= 0)
            window = [float(offset) for offset in range(start + max(position - WINDOW + 1, 0), start + position + 1)]
            pooled = [start + RATIO * index + (RATIO - 1) / 2 for index in range((position + 1) // RATIO)]
            assert read == pytest.approx(sorted(window + pooled), abs=1e-3), (start, position)
    # A padding token reads itself only.
    padding = STARTS[1] - 1
    assert [index.item() for index in indices[padding] if index >= 0] == [padding]


def test_cudnn_layout_of_a_packed_row_and_of_one_shard():
    # Two documents of 3 and 2 tokens, each in a span of 4 tokens.
    document_ids = torch.tensor([0, 0, 0, -1, 1, 1, -1, -1])
    cu_seqlens, row = cudnn_dsa_layout(document_ids, 0, 8)
    assert cu_seqlens.tolist() == [0, 3, 5] and cu_seqlens.dtype == torch.int32
    assert row.starts.tolist() == [0, 0, 0, 0, 4, 4, 4, 4]
    assert row.causal_lengths.tolist() == [1, 2, 3, 0, 1, 2, 0, 0]
    assert row.query_valid.tolist() == [True, True, True, False, True, True, False, False]
    assert row.key_source_indices is None and row.total_key_tokens == 8 and not row.all_rows_nonempty

    # The second half of the row as one context-parallel shard: its queries read keys of the whole row.
    cu_seqlens, shard = cudnn_dsa_layout(document_ids, 4, 4)
    assert cu_seqlens.tolist() == [0, 3, 5]
    assert shard.starts.tolist() == [4, 4, 4, 4]
    assert shard.causal_lengths.tolist() == [1, 2, 0, 0]
    assert shard.key_source_indices.tolist() == [4, 5, 6, 7] and shard.total_key_tokens == 8

    # A shard that starts inside a document reads the keys of that document on the preceding rank.
    cu_seqlens, shard = cudnn_dsa_layout(torch.tensor([0, 0, 0, 0, 0, 0, -1, -1]), 4, 4)
    assert cu_seqlens.tolist() == [0, 6]
    assert shard.starts.tolist() == [0, 0, 0, 0]
    assert shard.causal_lengths.tolist() == [5, 6, 0, 0]
    assert shard.key_source_indices.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


class _Mesh:
    """Stand-in for a context-parallel mesh of two ranks outside a distributed run."""

    def __init__(self, rank: int) -> None:
        self.rank = rank

    def size(self) -> int:
        return 2

    def get_local_rank(self) -> int:
        return self.rank

    def get_group(self) -> None:
        return None


def test_sharder_starts_documents_at_window_multiples_and_gives_each_rank_one_contiguous_shard():
    model, documents = _model((128, 128, -2)), _documents()
    row = torch.cat(documents, dim=1)
    shards = []
    for rank in range(2):
        batch = {"input_ids": row, "labels": row.clone(), "seq_lens": torch.tensor([LENGTHS]), "qkv_format": "thd"}
        sharder = model.prepare_model_inputs_for_cp(batch)["cp_sharder"]
        _, local, layout = sharder.shard_batch(_Mesh(rank), None, batch)
        assert layout.padded_seq_len == 512 and "seq_lens" not in local
        shards.append(local)
    assert all(shard["input_ids"].shape == (1, 256) for shard in shards)
    ids = torch.cat([shard["packed_seq_ids"] for shard in shards], dim=1)[0]
    positions = torch.cat([shard["position_ids"] for shard in shards], dim=1)[0]
    tokens = torch.cat([shard["input_ids"] for shard in shards], dim=1)[0]
    labels = torch.cat([shard["labels"] for shard in shards], dim=1)[0]
    valid = torch.cat([shard["attention_mask"] for shard in shards], dim=1)[0]
    for document, (start, length) in enumerate(zip(STARTS, LENGTHS)):
        assert torch.equal(ids[start : start + length], torch.full((length,), document + 1))
        assert torch.equal(positions[start : start + length], torch.arange(length))
        assert torch.equal(tokens[start : start + length], documents[document][0])
        assert torch.equal(labels[start : start + length], documents[document][0])
    assert int((ids > 0).sum()) == sum(LENGTHS) and torch.equal(valid, ids > 0)
    assert torch.equal(labels[ids == 0], torch.full((512 - sum(LENGTHS),), -100))


def test_prepared_shard_of_the_whole_row_matches_the_packed_row():
    """``packed_seq_ids`` and positions of the sharder give the logits of ``seq_lens`` in the sharder's layout."""
    model, documents = _model((128, 128, -2)), _documents()
    row = torch.cat(documents, dim=1)
    batch = {"input_ids": row, "labels": row.clone(), "seq_lens": torch.tensor([LENGTHS]), "qkv_format": "thd"}
    _, local, _ = model.prepare_model_inputs_for_cp(batch)["cp_sharder"].shard_batch(None, None, dict(batch))
    with torch.no_grad():
        expected = model(row, seq_lens=torch.tensor([LENGTHS])).logits[0].split(LENGTHS)
        prepared = model(
            local["input_ids"],
            attention_mask=local["attention_mask"],
            position_ids=local["position_ids"],
            packed_seq_ids=local["packed_seq_ids"],
        ).logits[0]
    for start, logits in zip(STARTS, expected):
        assert torch.equal(prepared[start : start + logits.shape[0]], logits)


def test_packed_and_context_parallel_inputs_that_are_incomplete_are_rejected():
    model = _model((128,))
    tokens = _documents()[0]
    ones = torch.ones_like(tokens)
    with pytest.raises(ValueError, match="seq_lens describes a whole packed row"):
        model(tokens, seq_lens=torch.tensor([[130]]), packed_seq_ids=ones, position_ids=ones)
    with pytest.raises(ValueError, match="requires position_ids, and packed_seq_ids or attention_mask"):
        model(tokens, packed_seq_ids=ones)
    model.cp_mesh = SimpleNamespace(get_group=object)
    with pytest.raises(ValueError, match="seq_lens describes a whole packed row"):
        model(tokens, seq_lens=torch.tensor([[130]]))
    with pytest.raises(ValueError, match="requires position_ids, and packed_seq_ids or attention_mask"):
        model(tokens, position_ids=ones)


def test_document_spans_that_do_not_cover_the_row_are_rejected():
    model, documents = _model((128,)), _documents()
    padded = torch.cat((documents[0], documents[0].new_zeros(1, 126), documents[1]), dim=1)
    with pytest.raises(ValueError, match="must add up to the row length 293"):
        model(padded, seq_lens=torch.tensor([[130, 37]]))
    with pytest.raises(ValueError, match="must add up to the row length 267"):
        model(torch.cat(documents, dim=1), seq_lens=torch.tensor([[130, 37]]))
    with pytest.raises(ValueError, match="must add up to the row length 293"):
        model(padded, seq_lens=torch.tensor([[130, 37]]), seq_lens_padded=torch.tensor([[256, 30]]))


def test_positions_that_do_not_restart_in_every_aligned_document_are_rejected():
    model, documents = _model((128, -2)), _documents()
    row = torch.cat((documents[0], documents[0].new_zeros(1, 126), documents[1], documents[0].new_zeros(1, 91)), dim=1)
    ids = torch.tensor([[1] * 130 + [0] * 126 + [2] * 37 + [0] * 91])
    positions = torch.cat((torch.arange(130), torch.zeros(126, dtype=torch.int64), torch.arange(37)))
    positions = F.pad(positions, (0, 91))[None]
    with torch.no_grad():
        laid_out = model(row, position_ids=positions, packed_seq_ids=ids).logits[0]
        packed = model(row, seq_lens=torch.tensor([[130, 37]]), seq_lens_padded=torch.tensor([[256, 128]])).logits[0]
    torch.testing.assert_close(laid_out[ids[0] > 0], packed[ids[0] > 0], rtol=0, atol=ATOL)
    with pytest.raises(ValueError, match="must count up from zero in every document"):
        model(row, position_ids=torch.arange(384)[None], packed_seq_ids=ids)
    with pytest.raises(ValueError, match="must count up from zero in every document"):
        model(row, position_ids=positions + (ids == 2), packed_seq_ids=ids)
    with pytest.raises(ValueError, match="must start at a multiple of 128 tokens"):
        model(
            torch.cat((documents[0], documents[1]), dim=1),
            position_ids=torch.cat((torch.arange(130), torch.arange(37)))[None],
            packed_seq_ids=torch.tensor([[1] * 130 + [2] * 37]),
        )
    with pytest.raises(ValueError, match="must be numbered in row order"):
        model(row, position_ids=positions, packed_seq_ids=ids.clamp_max(1))
    # The first token of a later shard may continue a document; the first shard starts one.
    with pytest.raises(ValueError, match="must count up from zero in every document"):
        model._validate_packed_layout(positions + 1, ids, first=0)
    model._validate_packed_layout(torch.arange(128, 512)[None], torch.ones(1, 384, dtype=torch.int64), first=384)


def test_capabilities_admit_context_parallelism_with_thd_packing():
    model = _model((128, 128, -2))
    model = attach_capabilities_and_validate(model, SimpleNamespace(tp_size=1, cp_size=4, pp_size=1, ep_size=4))
    assert model.supports.supports_cp and model.supports.supports_sequence_packing
    assert model.supports.supports_cp_with_sequence_packing

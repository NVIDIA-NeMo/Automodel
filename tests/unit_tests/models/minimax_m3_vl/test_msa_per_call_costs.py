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

"""CPU contracts for the MSA per-call costs that are hoisted out of the hot path."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl import _msa as msa
from nemo_automodel.components.models.minimax_m3_vl import model as m3_model
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig


def test_align_backward_tensor_scatters_in_place_without_a_copy() -> None:
    compact = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    positions = torch.tensor([5, 0, 3, 1], dtype=torch.int64)

    aligned = msa._align_backward_tensor(compact, positions, 8)

    assert aligned.shape == (8, 2, 3)
    assert aligned.is_contiguous()
    torch.testing.assert_close(aligned[positions], compact, rtol=0, atol=0)
    unwritten = torch.ones(8, dtype=torch.bool).index_fill_(0, positions, False)
    assert torch.count_nonzero(aligned[unwritten]) == 0


def _doc_ids() -> torch.Tensor:
    """Return int64 packed ids[1, 262] with two documents and a padding tail."""
    ids = torch.zeros(1, 262, dtype=torch.int64)
    ids[0, :128] = 1
    ids[0, 128:200] = 2
    return ids


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the _MSAPackedLayout.build calls model._memoized_msa_layout makes."""
    counter = [0]
    build = msa._MSAPackedLayout.build

    def counting(doc_ids: torch.Tensor) -> msa._MSAPackedLayout:
        counter[0] += 1
        return build(doc_ids)

    monkeypatch.setattr(m3_model._MSAPackedLayout, "build", staticmethod(counting))
    return counter


def test_layout_is_built_once_per_microbatch_and_released_with_it(builds: list[int]) -> None:
    packed_seq_ids = _doc_ids()
    # Seven virtual pipeline stages of one microbatch resolve their own doc_ids from the same batch
    # tensor and must share one layout.
    layouts = [m3_model._memoized_msa_layout(packed_seq_ids.clone(), packed_seq_ids) for _ in range(7)]
    assert builds[0] == 1
    assert all(layout is layouts[0] for layout in layouts)

    # A second microbatch is a different tensor, so it gets its own layout.
    other_ids = _doc_ids()
    assert m3_model._memoized_msa_layout(other_ids.clone(), other_ids) is not layouts[0]
    assert builds[0] == 2

    # The memo holds the batch weakly: dropping it drops the layout entry.
    assert len(m3_model._MSA_LAYOUT_MEMO) == 2
    del packed_seq_ids, other_ids, layouts
    assert len(m3_model._MSA_LAYOUT_MEMO) == 0


def test_layout_is_rebuilt_when_the_batch_carries_no_packed_ids(builds: list[int]) -> None:
    # A single-document pack takes this path (the collator emits _packed_seq_ids only when some row
    # holds two documents), so the stages of that microbatch must still agree on the geometry.
    doc_ids = _doc_ids()
    first = m3_model._memoized_msa_layout(doc_ids, None)
    second = m3_model._memoized_msa_layout(doc_ids, None)

    assert builds[0] == 2 and first is not second
    assert len(m3_model._MSA_LAYOUT_MEMO) == 0
    assert torch.equal(first.cu_seqlens, second.cu_seqlens) and first.max_seqlen == second.max_seqlen


def test_layout_exposes_the_packed_document_geometry() -> None:
    doc_ids = torch.zeros(2, 8, dtype=torch.int64)
    doc_ids[0, :3] = 1
    doc_ids[0, 3:5] = 2
    doc_ids[1, 2:6] = 1
    layout = msa._MSAPackedLayout.build(doc_ids)

    cu_seqlens = layout.cu_seqlens
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous()
    assert cu_seqlens.tolist() == [0, 3, 5, 9]
    assert layout.max_seqlen == 4
    # Every cu_seqlens slice must cover exactly one document, in pack row order.
    packed_ids = layout.pack(doc_ids.unsqueeze(-1)).squeeze(-1)
    assert int(cu_seqlens[-1]) == packed_ids.shape[0]
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
        assert packed_ids[start:end].unique().numel() == 1


def _text_model(attn: str, sparse_attn: str) -> m3_model.MiniMaxM3TextModel:
    """Construct the smallest MiniMaxM3TextModel (1 dense + 1 sparse layer) on the meta device."""
    config = MiniMaxM3VLTextConfig(
        hidden_size=32,
        intermediate_size=32,
        dense_intermediate_size=48,
        shared_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=64,
        rotary_dim=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=0,
        moe_layer_freq=[0, 0],
        num_mtp_modules=0,
        attention_dropout=0.0,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_block_size": 128,
            "sparse_topk_blocks": 16,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
            "sparse_attention_freq": [0, 1],
            "sparse_disable_index_value": [0, 1],
        },
    )
    backend = BackendConfig(attn=attn, sparse_attn=sparse_attn, linear="torch", rms_norm="torch", rope_fusion=False)
    with torch.device("meta"):
        return m3_model.MiniMaxM3TextModel(config, backend)


def test_msa_with_dense_layers_requires_a_varlen_attention_backend() -> None:
    # Dense layers are packed to [tokens, hidden] once MSA is on, so they isolate documents with
    # cu_seqlens; sdpa drops cu_seqlens on the floor (attention/utils.py:207-212).
    with pytest.raises(NotImplementedError, match="backend.attn='te'"):
        _text_model("sdpa", "msa")
    # _msa_model_has_dense_layers is vacuously true with zero MSA layers, so a guard missing the
    # _msa_layer_ids conjunct would stop every sparse_attn='generic' model from constructing.
    assert _text_model("sdpa", "generic") is not None

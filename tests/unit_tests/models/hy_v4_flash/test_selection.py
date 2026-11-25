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

"""Key lists handed to the sparse attention kernel: layout contract and agreement with a brute-force selection."""

import pytest
import torch

from nemo_automodel.components.models.hy_v4_flash.attention import (
    HY4FlashCompressedIndexer,
    HY4FlashRope,
    causal_prefix_plan,
)
from nemo_automodel.components.models.hy_v4_flash.layers import inverse_frequencies, mrope_cos_sin
from tests.unit_tests.models.hy_v4_flash.tiny import (
    TOPK,
    build_model,
    complex_rotation,
    pack,
    random_documents,
    tiny_text_config,
)


def _brute_force_logits(
    indexer: HY4FlashCompressedIndexer,
    q_resid: torch.Tensor,
    hidden: torch.Tensor,
    group_keys: torch.Tensor,
    slots: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Group logits of every query from the full [heads, groups] score block, one query at a time.

    Args:
        indexer: The indexer under test (its projections are the only shared code).
        q_resid: Tensor of shape [queries, q_lora_rank].
        hidden: Tensor of shape [queries, hidden].
        group_keys: Tensor of shape [groups, index_head_dim], unrotated.
        slots: Integer tensor of shape [queries], document-local positions of the queries.
        theta: Rotary base of the sparse layers.

    Returns:
        Tensor of shape [queries, groups].
    """
    kpool, heads, dim, rope = indexer.kpool, indexer.n_heads, indexer.head_dim, indexer.rope_dim
    keys = group_keys.clone()
    keys[:, dim - rope :] = complex_rotation(keys[:, dim - rope :], torch.arange(keys.shape[0]) * kpool, theta)
    query = indexer.query_projection(q_resid).view(-1, heads, dim).clone()
    for head in range(heads):
        query[:, head, dim - rope :] = complex_rotation(query[:, head, dim - rope :], slots, theta)
    weights = indexer.head_weight_projection(hidden) * (heads**-0.5 * dim**-0.5)
    return torch.stack([(weights[row][:, None] * torch.relu(query[row] @ keys.T)).sum(0) for row in range(len(slots))])


@pytest.mark.parametrize("first_slot", [0, 5])
@pytest.mark.parametrize("query_chunk", [3, 256, None])  # queries scored at a time; None: the default budget
def test_chunked_selection_is_a_brute_force_topk_and_a_sorted_prefix(first_slot: int, query_chunk: int | None) -> None:
    torch.manual_seed(0)
    config = tiny_text_config()  # index_topk 8, kpool 2: three recalled groups plus the query's own group
    theta = config.layer_rope_parameters["indexed_attention"]["rope_theta"]
    section = config.layer_rope_parameters["indexed_attention"]["mrope_section"]
    indexer = HY4FlashCompressedIndexer(config).float()
    for parameter in indexer.parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    kpool, recall, document = indexer.kpool, indexer.group_topk - 1, 41
    slots = torch.arange(first_slot, document)
    q_resid = torch.randn(len(slots), config.q_lora_rank)
    hidden = torch.randn(len(slots), config.hidden_size)
    group_keys = indexer.pool_groups(torch.randn(document, config.index_head_dim))
    if query_chunk is not None:  # the fp32 score block of ``query_chunk`` queries: [queries, heads, groups]
        indexer.score_block_bytes = query_chunk * 4 * indexer.n_heads * group_keys.shape[0]
    cos, sin = mrope_cos_sin(
        torch.arange(document).expand(3, 1, -1),
        inverse_frequencies(config.qk_rope_head_dim, theta),
        section,
        torch.float32,
    )
    plan = indexer.select(
        q_resid,
        hidden,
        HY4FlashRope(cos[0, first_slot:], sin[0, first_slot:]),
        group_keys,
        HY4FlashRope(cos[0, : group_keys.shape[0] * kpool : kpool], sin[0, : group_keys.shape[0] * kpool : kpool]),
        first_slot,
    )
    logits = _brute_force_logits(indexer, q_resid, hidden, group_keys, slots, theta)
    assert plan.indices.shape == (len(slots), config.index_topk) and plan.indices.dtype == torch.int32
    dropped = 0
    for row, slot in enumerate(slots.tolist()):
        own = slot // kpool
        length = int(plan.lengths[row])
        keys = plan.indices[row, :length].tolist()
        assert keys == sorted(set(keys)) and (plan.indices[row, length:] == -1).all()  # ascending valid prefix
        own_tokens = list(range(own * kpool, slot + 1))
        assert keys[len(keys) - len(own_tokens) :] == own_tokens
        recalled = keys[: len(keys) - len(own_tokens)]
        groups = sorted({token // kpool for token in recalled})
        assert recalled == [group * kpool + offset for group in groups for offset in range(kpool)]
        assert len(groups) == min(recall, own) and all(group < own for group in groups)
        # the brute-force top-k under the total order (score, then lowest group first)
        ranked = sorted(range(own), key=lambda group: (-float(logits[row, group]), group))
        assert groups == sorted(ranked[:recall])
        dropped += len(keys) < slot + 1
    assert dropped > 0, "no query dropped a visible key: sparse selection is not exercised"


def test_causal_prefix_plan_windows_stay_inside_documents_and_padding_has_no_keys() -> None:
    # documents [0, 5) and [5, 9), then one padding token whose document "starts" after itself
    position = torch.arange(10)
    doc_start = torch.tensor([0, 0, 0, 0, 0, 5, 5, 5, 5, 10])
    plan = causal_prefix_plan(position, doc_start, 3)
    assert plan.lengths.tolist() == [1, 2, 3, 3, 3, 1, 2, 3, 3, 0]
    assert plan.indices.tolist() == [
        [0, -1, -1],
        [0, 1, -1],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [5, -1, -1],
        [5, 6, -1],
        [5, 6, 7],
        [6, 7, 8],
        [-1, -1, -1],
    ]


def test_pooled_key_norm_stays_off_the_compiled_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pooling runs once per document with that document's group count, so it must not reach the compiled norm core.

    The compiled core specializes on the input shape: called with a new group count it compiles again, which costs
    seconds per new document layout on packed data. The stand-in below records every shape the compiled core would see.
    """
    from nemo_automodel.components.models.hy_v4_flash import layers

    seen: list[tuple[int, ...]] = []

    def recording_core(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        seen.append(tuple(hidden_states.shape))
        return layers._rms_norm_core(hidden_states, weight, eps)

    cfg = tiny_text_config()
    indexer = HY4FlashCompressedIndexer(cfg).float()
    documents = [torch.randn(length, indexer.head_dim) for length in (3 * indexer.kpool, 7 * indexer.kpool + 1)]

    monkeypatch.setattr(layers, "_rms_norm", recording_core)
    pooled = [indexer.pool_groups(document) for document in documents]
    assert seen == []
    assert [tuple(p.shape) for p in pooled] == [(3, indexer.head_dim), (7, indexer.head_dim)]

    # Control: a norm with a fixed input shape does go through the swappable core.
    indexer.token_keys(torch.randn(5, cfg.hidden_size))
    assert seen == [(5, indexer.head_dim)]


@pytest.mark.parametrize("lengths", [(5, 9, 3), (5, 41, 3)])
def test_pooling_modules_are_entered_whether_or_not_a_document_needs_selection(lengths: tuple[int, ...]) -> None:
    """FSDP turns the first call of a sharded unit into a collective, so every rank must issue the same calls.

    The fp32 group-pooling modules only do work for documents longer than ``index_topk``. A rank whose pack holds no
    such document must still enter them, at the same place, or its collective sequence differs from a rank that does
    (a deadlock under FSDP). Both packs must therefore enter each module in every sparse layer.
    """
    model = build_model()
    assert (max(lengths) > TOPK) == (lengths == (5, 41, 3))
    sparse_layers = [layer for layer in model.model.layers if layer.self_attn.compressed_indexer is not None]
    assert sparse_layers
    entered = []
    for index, layer in enumerate(sparse_layers):
        indexer = layer.self_attn.compressed_indexer
        for name in ("key_value_projection", "key_gate_projection", "intra_group_ape", "key_norm"):
            getattr(indexer, name).register_forward_pre_hook(
                lambda module, args, index=index, name=name: entered.append((index, name))
            )
    with torch.no_grad():
        model(**pack(random_documents(lengths)))
    first_entries = list(dict.fromkeys(entered))
    assert first_entries == [
        (index, name)
        for index in range(len(sparse_layers))
        for name in ("key_value_projection", "key_gate_projection", "intra_group_ape", "key_norm")
    ]

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

"""Compact visibility must produce exactly the legacy dense-mask route IDs."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4.cp import build_packed_seq_ids
from nemo_automodel.components.models.deepseek_v4.layers import (
    _build_indexer_topk_compressed_mask,
    build_causal_padding_mask,
    build_packed_causal_padding_mask,
)
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import build_dsv4_sparse_topk_indices
from nemo_automodel.components.models.deepseek_v4.sparse_mask import DSV4SparseMask, build_dsv4_sparse_mask


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("local_queries", [False, True])
def test_compact_indices_exactly_match_dense(ratio, packed, padding, local_queries):
    torch.manual_seed(82)
    batch, keys, window = 2, 265, 17
    device = torch.device("cpu")
    slots = torch.arange(keys).expand(batch, -1)
    positions = slots.clone()
    lengths = torch.tensor([[128, 132], [132, 128]])
    docs = build_packed_seq_ids(lengths, seq_len=keys, device=device) if packed else None
    if packed:
        positions[0, 128:] -= 128
        positions[1, 132:] -= 132
        dense = build_packed_causal_padding_mask(lengths, keys, torch.float32, device, window)
    else:
        dense = build_causal_padding_mask(None, keys, torch.float32, device, batch, window)
    pad = torch.zeros(batch, keys, dtype=torch.bool)
    if padding:
        pad[:, 4:8] = True
        dense.masked_fill_(pad[:, None, None, :], torch.finfo(torch.float32).min)
    pooled = keys // ratio if ratio else 0
    pooled_docs = pooled_pos = chosen = None
    if pooled:
        if packed:
            starts = torch.arange(pooled) * ratio
            pooled_docs = docs[:, starts]
            pooled_pos = positions[:, starts] // ratio
        if ratio == 4:
            chosen = torch.randint(-1, pooled + 3, (batch, keys, 11))
            chosen[..., 1] = chosen[..., 0]  # duplicate selection slots
            extension = _build_indexer_topk_compressed_mask(dense, chosen, pooled)
        else:
            valid_pool = torch.arange(pooled)[None, None] < ((positions + 1) // ratio)[..., None]
            extension = torch.where(valid_pool, 0.0, torch.finfo(torch.float32).min)
        if packed:
            if ratio != 4:
                extension = torch.zeros_like(extension)
            allowed = (
                (docs[..., None] > 0)
                & (docs[..., None] == pooled_docs[:, None])
                & (pooled_pos[:, None] < ((positions + 1) // ratio)[..., None])
            )
            extension = extension.masked_fill(~allowed, torch.finfo(torch.float32).min)
        dense = torch.cat((dense, extension[:, None]), -1)
    rows = torch.tensor([0, 7, 127, 128, 131, 132, 255, 264]) if local_queries else torch.arange(keys)
    compact = DSV4SparseMask(slots[:, rows], slots, docs[:, rows] if packed else None, docs, pad, window)
    kwargs = dict(
        batch_size=batch,
        seq_len=len(rows),
        key_len=keys + pooled,
        window_size=window,
        device=device,
        compress_ratio=ratio,
        compressed_topk=chosen[:, rows] if chosen is not None else None,
        n_pooled=pooled,
        vanilla_key_len=keys,
        q_positions=slots[:, rows],
    )
    expected = build_dsv4_sparse_topk_indices(**kwargs, attention_mask=dense[:, :, rows])
    candidates = build_dsv4_sparse_topk_indices(**kwargs)
    actual = compact.filter_indices(
        candidates,
        vanilla_key_len=keys,
        compress_ratio=ratio,
        query_positions=positions[:, rows],
        query_seq_ids=docs[:, rows] if packed else None,
        pooled_seq_ids=pooled_docs,
        pooled_positions=pooled_pos,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_builder_keeps_only_linear_metadata():
    positions = torch.arange(65536)[None]
    metadata = build_dsv4_sparse_mask(
        positions, key_len=65536, packed_seq_ids=None, padding_mask=None, sliding_window=128, cp_group=None
    )
    assert metadata.query_positions.shape == metadata.key_positions.shape == (1, 65536)
    assert metadata.query_seq_ids is metadata.key_seq_ids is metadata.key_padding is None

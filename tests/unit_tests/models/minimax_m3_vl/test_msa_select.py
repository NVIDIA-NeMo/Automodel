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

"""The MSA torch selection reference, and the geometric invariant the fused rule leans on."""

import torch

from nemo_automodel.components.models.minimax_m3_vl.msa import MSAMicrobatch
from tests.unit_tests.models.minimax_m3_vl._msa_select_reference import select_blocks_reference

_BLOCK, _HEADS, _DIM, _TOPK = 8, 4, 16, 3


def _microbatch(lengths: tuple[int, ...]) -> MSAMicrobatch:
    """Build a single-row packed microbatch holding ``lengths`` back-to-back documents."""
    documents = torch.zeros(1, sum(lengths), dtype=torch.int64)
    position = 0
    for document, length in enumerate(lengths, start=1):
        documents[0, position : position + length] = document
        position += length
    return MSAMicrobatch.from_document_map(documents, forced_blocks=(0, 1))


def test_selection_is_document_local_and_causal() -> None:
    microbatch = _microbatch((37, 21))
    generator = torch.Generator().manual_seed(20260907)
    tokens = int(microbatch.cu_seqlens[-1])
    index_q = torch.randn(tokens, _HEADS, _DIM, generator=generator)
    index_k = torch.randn(tokens, 1, _DIM, generator=generator)
    selected = select_blocks_reference(
        microbatch,
        index_q,
        index_k,
        block_size=_BLOCK,
        topk_blocks=_TOPK,
        init_blocks=0,
        local_blocks=1,
        score_type="max",
    )

    assert selected.shape == (_HEADS, tokens, _TOPK)
    assert selected.dtype == torch.int32 and selected.is_contiguous()
    own_block = microbatch.document_positions // _BLOCK
    # -1 pads a short row; every real pick is a block of this query's own document, at or before it.
    assert (selected <= own_block[:, None]).all()
    assert (selected >= -1).all()
    # The local block is forced in, so no row is entirely padding and every row holds its own block.
    assert (selected == own_block[:, None]).any(dim=-1).all()
    # Rows whose document offers fewer than topk blocks pad the remainder rather than repeating.
    short = own_block < _TOPK - 1
    assert (selected[:, short].eq(-1).sum(dim=-1) > 0).all()

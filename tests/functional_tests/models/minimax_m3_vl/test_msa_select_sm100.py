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

"""The fused MSA scorer against the Torch reference it replaces, on real SM100 hardware.

Both produce the same canonical support, so they are compared as block *sets* per (head, query):
slot order is not part of that contract, and top-k is discontinuous at a tie, so rows whose 16th and
17th candidate scores sit inside the scorer's own error are exempt. Everything else must agree.
"""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl import msa, msa_bindings
from nemo_automodel.shared.import_utils import UnavailableError
from tests.unit_tests.models.minimax_m3_vl._msa_select_reference import (
    block_scores_reference,
    select_blocks_reference,
)

_BLOCK, _HEADS, _DIM, _TOPK = 128, 4, 128, 16
_INIT, _LOCAL = 0, 1
_LENGTHS = (1503, 2048, 900, 128, 1, 129, 4096)
# The 16th and 17th candidate are ordered by the last ulp when they are this close; the fused pass
# accumulates in a different order, so those rows may legitimately pick either.
_TIE_MARGIN = 1e-5


def _unavailable() -> str | None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        return "requires an SM100 GPU"
    try:
        msa_bindings.kernels()
    except UnavailableError:
        return "requires uv sync --extra msa"
    return None


_SKIP_REASON = _unavailable()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "SM100 available")


def _fixture() -> tuple[msa.MSAMicrobatch, torch.Tensor, torch.Tensor]:
    """Build a packed microbatch of _LENGTHS documents with seeded bf16 index projections."""
    device = torch.device("cuda", torch.cuda.current_device())
    documents = torch.zeros(1, sum(_LENGTHS), dtype=torch.int64, device=device)
    position = 0
    for document, length in enumerate(_LENGTHS, start=1):
        documents[0, position : position + length] = document
        position += length
    microbatch = msa.MSAMicrobatch.from_document_map(documents, forced_blocks=(_INIT, _LOCAL))
    generator = torch.Generator(device=device).manual_seed(20260907)
    tokens = sum(_LENGTHS)
    index_q = torch.randn(tokens, _HEADS, _DIM, device=device, dtype=torch.bfloat16, generator=generator)
    index_k = torch.randn(tokens, 1, _DIM, device=device, dtype=torch.bfloat16, generator=generator)
    return microbatch, index_q, index_k


def test_the_fused_scorer_selects_the_same_blocks_as_the_reference() -> None:
    microbatch, index_q, index_k = _fixture()
    fused = microbatch.select_blocks(index_q, index_k)
    reference = select_blocks_reference(
        microbatch,
        index_q,
        index_k,
        block_size=_BLOCK,
        topk_blocks=_TOPK,
        init_blocks=_INIT,
        local_blocks=_LOCAL,
        score_type="max",
    )

    assert fused.shape == reference.shape == (_HEADS, index_q.shape[0], _TOPK)
    assert fused.dtype == torch.int32 and fused.is_contiguous()
    # Sets, not slots: build_k2q_csr reorders by query, so slot order is not part of the contract.
    disagree = (fused.sort(-1).values != reference.sort(-1).values).any(-1)
    # Exempt the rows the reference itself cannot order: its own 16th and 17th candidates are closer
    # together than the fused pass's accumulation error, so either answer is the canonical one.
    ranked = block_scores_reference(
        microbatch, index_q, index_k, block_size=_BLOCK, init_blocks=_INIT, local_blocks=_LOCAL, score_type="max"
    )
    top = ranked.topk(_TOPK + 1, dim=-1).values
    gap = top[..., _TOPK - 1] - top[..., _TOPK]
    exempt = torch.isfinite(top[..., _TOPK]) & (gap <= _TIE_MARGIN * top[..., _TOPK - 1].abs().clamp_min(1.0))
    assert not bool((disagree & ~exempt).any()), (
        f"{int((disagree & ~exempt).sum())} of {disagree.numel()} rows disagree outside a numerical tie"
    )


def test_reusing_the_score_buffer_cannot_change_the_selection() -> None:
    # One buffer is shared by every layer of a microbatch, so tiles this call does not write still
    # hold the previous layer's scores; the selection rule must already reject every one of them.
    microbatch, index_q, index_k = _fixture()
    expected = microbatch.select_blocks(index_q, index_k)
    msa._score_scratch(index_q.device, microbatch._plan.score_shape).fill_(1e30)
    assert torch.equal(microbatch.select_blocks(index_q, index_k), expected)

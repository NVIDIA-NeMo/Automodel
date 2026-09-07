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

"""Block selection for MSA: which key blocks of its own document each query attends to.

``select_blocks`` is the entry point, and the only thing a caller learns. It maps one layer's index
projections to the canonical support ``int32 [index_heads, tokens, topk_blocks]`` of document-local
block ids, padded with ``-1``. That tensor is the sole semantic seam between selection and the rest
of MSA -- ``build_k2q_csr`` and ``sparse_atten_func`` consume it unchanged. Everything behind it is
private: the per-microbatch ``fmha_sm100`` plan and where it is shared from, the score buffer it
writes into, the per-token document geometry, and the forced/candidate/top-k rule.

Selection stops after the scorer's OnlyScore pass. The top-k rule stays in Torch rather than calling
the package's ``sparse_topk_select``, which cannot express M3's per-query forced block and drops
causality silently in two other ways; the Torch definition it is checked against lives with the
tests.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.weak import WeakIdKeyDictionary

from nemo_automodel.components.models.minimax_m3_vl._msa import _MSAPackedLayout, _require_msa
from nemo_automodel.components.models.minimax_m3_vl.kernels import require_sm100


def _apply_selection_rule(
    block_score: torch.Tensor, candidate: torch.Tensor, forced: torch.Tensor, *, topk_blocks: int
) -> torch.Tensor:
    """Turn document-local block scores into canonical support.

    Args:
        block_score: Scores [index_heads, tokens, blocks]; entries outside ``candidate`` are never
            read, so they may hold anything.
        candidate: The [tokens, blocks] mask of blocks each query may pick.
        forced: The [tokens, blocks] subset each query must pick.
        topk_blocks: Blocks each query keeps.

    Returns:
        Document-local block ids [index_heads, tokens, topk_blocks], int32, padded with -1.
    """
    negative_infinity = float("-inf")
    block_score = block_score.masked_fill(~candidate, negative_infinity).masked_fill_(forced, float("inf"))
    if block_score.shape[-1] < topk_blocks:
        block_score = F.pad(block_score, (0, topk_blocks - block_score.shape[-1]), value=negative_infinity)
    values, indices = block_score.topk(topk_blocks, dim=-1)
    return torch.where(values == negative_infinity, -1, indices).to(torch.int32).contiguous()


@dataclass(frozen=True, slots=True)
class _MSASelectionPlan:
    """One packed microbatch's layer-invariant block-selection state.

    Scoring a layer costs an ``fmha_sm100`` OnlyScore pass plus the top-k rule. Everything that
    depends only on the microbatch -- the FMHA execution plan, the shape of the score buffer it
    writes into, and the per-token document geometry -- is built once here and reused by all 57 MSA
    layers.

    The plan is pinned to ``split_prefill_decode=False`` and ``num_kv_splits=1``: neither is a
    tuning knob. The first splits a batch whose first document is short into two sub-plans, which
    costs 2.9x on the score pass and adds two host syncs per call; the second lets the planner pick
    a variant from an SM-count estimate, making the compiled variant data-dependent.
    """

    _plan: Any
    _score_shape: tuple[int, int, int]
    _num_blocks: int
    _topk_blocks: int
    _candidate: torch.Tensor
    _forced: torch.Tensor
    # The arguments this plan was built from, in ``select_blocks``' keyword order. The plan is
    # shared per microbatch on a key that cannot see them, so it carries them to be checked against.
    _parameters: tuple[int, int, int, int, int]

    @classmethod
    def build(
        cls,
        layout: _MSAPackedLayout,
        *,
        index_heads: int,
        block_size: int,
        topk_blocks: int,
        init_blocks: int,
        local_blocks: int,
    ) -> "_MSASelectionPlan":
        """Plan the scoring pass for one packed microbatch.

        Args:
            layout: The packed-microbatch layout whose documents will be scored.
            index_heads: Index heads, used as the scorer's query heads against one shared key head.
            block_size: Keys per block.
            topk_blocks: Blocks each query keeps.
            init_blocks: Leading blocks of the document forced in.
            local_blocks: Non-zero forces the query's own block in.

        Returns:
            A plan valid for every layer of this microbatch.

        Raises:
            AssertionError: If the planner split the batch, or silently disabled the score output
                because ``index_heads * max_k_tiles * tokens`` overflowed int32.
        """
        require_sm100(layout.cu_seqlens.device)
        kernels = _require_msa()
        # fmha_sm100_plan needs host lengths; this is the microbatch's second and last sync.
        lengths = layout.cu_seqlens.diff().cpu()
        plan = kernels.fmha_sm100_plan(
            lengths,
            lengths,
            index_heads,
            num_kv_heads=1,
            causal=True,
            output_maxscore=True,
            num_kv_splits=1,
            split_prefill_decode=False,
        )
        split_batch, _, _, score_plan, _ = plan
        assert split_batch is False, "the planner split the batch into decode and prefill sub-plans"
        max_k_tiles = int(score_plan["max_k_tiles"])
        tokens = int(layout.cu_seqlens[-1])
        # api.py:616-619 prints and sets max_k_tiles = -1, then falls back to dense attention with
        # max_score=None, which would read as "no blocks selected" rather than as a failure.
        assert max_k_tiles > 0, f"scorer disabled maxscore: {index_heads} * max_k_tiles * {tokens} > 2**31"

        own_block = (layout._workspace_positions - layout._query_doc_starts) // block_size
        num_blocks = -(-layout.max_seqlen // block_size)
        blocks = torch.arange(num_blocks, device=own_block.device)
        # A query's document-local position lies inside its own document, so its own block is never
        # past that document's last block: "at or before my own block" is the whole permission. The
        # second bound this used to carry, against the document's block count, could never exclude
        # anything, and cost a host repeat_interleave over every token plus a transfer per
        # microbatch.
        candidate = blocks <= own_block[:, None]
        forced = blocks < init_blocks
        if local_blocks > 0:
            forced = forced | (blocks == own_block[:, None])
        return cls(
            _plan=plan,
            _score_shape=(index_heads, max_k_tiles, tokens),
            _num_blocks=num_blocks,
            _topk_blocks=topk_blocks,
            _candidate=candidate,
            _forced=forced & candidate,
            _parameters=(index_heads, block_size, topk_blocks, init_blocks, local_blocks),
        )

    def select(self, index_q: torch.Tensor, index_k: torch.Tensor) -> torch.Tensor:
        """Score this layer's index projections and reduce them to canonical support.

        Args:
            index_q: Index queries [tokens, index_heads, index_dim], bf16, post norm and RoPE.
            index_k: Shared index key [tokens, 1, index_dim], bf16, post norm and RoPE.

        Returns:
            Document-local block ids [index_heads, tokens, topk_blocks], int32, padded with -1.
        """
        kernels = _require_msa()
        # The score buffer is reused across layers, so tiles the kernel does not write hold the
        # previous layer's values; the selection rule rejects exactly those (a query only reads
        # blocks of its own document, up to its own). v is never read with output_o=False.
        _, max_score = kernels.fmha_sm100(
            index_q,
            index_k,
            index_k,
            self._plan,
            max_score=_score_scratch(index_q.device, self._score_shape),
            output_o=False,
            output_maxscore=True,
        )
        return _apply_selection_rule(
            max_score[:, : self._num_blocks].permute(0, 2, 1),
            self._candidate,
            self._forced,
            topk_blocks=self._topk_blocks,
        )


_SCORE_SCRATCH: dict[torch.device, torch.Tensor] = {}


def _score_scratch(device: torch.device, shape: tuple[int, int, int]) -> torch.Tensor:
    """Return the shared score buffer, grown to hold ``shape``.

    The scorer stores rather than accumulates, and every tile the selection rule reads is written by
    the same call, so nothing carries over between layers or microbatches and one buffer serves them
    all. This relies on scoring passes never overlapping, which holds because MSA is single-stream
    and rejects CUDA-graph capture. Sharing is what makes the buffer affordable: ``max_k_tiles`` is
    rounded up to 128 tiles whatever the documents are, so most of each buffer is never written, and
    measured over the seven plans one microbatch builds without ``_packed_seq_ids``, per-plan buffers
    cost 224 MiB against this buffer's 11.28 MiB.

    Args:
        device: CUDA device the scorer runs on.
        shape: ``[index_heads, max_k_tiles, tokens]`` the scorer will write.

    Returns:
        A contiguous float32 view of that shape.
    """
    elements = shape[0] * shape[1] * shape[2]
    buffer = _SCORE_SCRATCH.get(device)
    if buffer is None or buffer.numel() < elements:
        buffer = torch.empty(elements, dtype=torch.float32, device=device)
        _SCORE_SCRATCH[device] = buffer
    return buffer[:elements].view(shape)


_PLAN_MEMO: WeakIdKeyDictionary = WeakIdKeyDictionary()


def select_blocks(
    layout: _MSAPackedLayout,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    index_heads: int,
    block_size: int,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
) -> torch.Tensor:
    """Choose each query's key blocks within its own document, for one layer of one microbatch.

    Selection is not differentiable: the score is a hard top-k over unnormalized QK maxima, so the
    index projections reach this through ``torch.no_grad``.

    The layer-invariant half of the work -- planning the scoring pass and deriving each query's
    document geometry -- is built on the first layer that asks and shared with the rest of the
    microbatch. ``_MSAPackedLayout`` has slots and so cannot be weakly referenced, but the
    ``cu_seqlens`` tensor it owns can, and lives exactly as long; keying on it gives the shared state
    the microbatch's lifetime without the model having to thread it through every layer. That key
    cannot see the arguments below, so a plan carries the ones it was built from and they are
    checked on reuse.

    Args:
        layout: The packed-microbatch layout the tokens came from, shared by every stage.
        index_q: Index queries [tokens, index_heads, index_dim], bf16, post norm and RoPE.
        index_k: Shared index key [tokens, 1, index_dim], bf16, post norm and RoPE.
        index_heads: Index heads, used as the scorer's query heads against one shared key head.
        block_size: Keys per block.
        topk_blocks: Blocks each query keeps.
        init_blocks: Leading blocks of the document forced in.
        local_blocks: Non-zero forces the query's own block in.

    Returns:
        Document-local block ids [index_heads, tokens, topk_blocks], int32, padded with -1.

    Raises:
        AssertionError: If this microbatch was already scored with different selection arguments.
    """
    parameters = (index_heads, block_size, topk_blocks, init_blocks, local_blocks)
    plan = _PLAN_MEMO.get(layout.cu_seqlens)
    if plan is None:
        plan = _MSASelectionPlan.build(
            layout,
            index_heads=index_heads,
            block_size=block_size,
            topk_blocks=topk_blocks,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
        _PLAN_MEMO[layout.cu_seqlens] = plan
    assert plan._parameters == parameters, (
        f"this microbatch is already being scored with {plan._parameters}, not {parameters}"
    )
    return plan.select(index_q, index_k)


# The compiled scorer variant is keyed on (dtype, qo_tile, single_wg, sparse_mode, page_size,
# split_kv, pack_factor); with everything else pinned it depends only on the longest document, and
# these five values are the measured boundaries between the reachable variants.
_WARMUP_MAX_DOCS = (16, 32, 64, 128, 256)


@lru_cache(maxsize=None)
def warm_selection_variants(
    device: torch.device,
    *,
    index_heads: int,
    index_dim: int,
    block_size: int,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Compile every reachable scorer variant once per process.

    Each variant costs about 44 s on a cold cache. Without this, a microbatch whose longest document
    crosses a boundary stalls training mid-run -- and under pipeline parallelism every stage waits.

    Warming runs the production path over a synthetic one-document microbatch rather than reassembling
    the planner call, so the variant compiled here is the variant production reaches by construction.
    Take the selection arguments from the same place the layer's own calls take them.

    Args:
        device: The CUDA device the scorer will run on.
        index_heads: Index heads, used as the scorer's query heads.
        index_dim: Index head dimension.
        block_size: Keys per block.
        topk_blocks: Blocks each query keeps.
        init_blocks: Leading blocks of the document forced in.
        local_blocks: Non-zero forces the query's own block in.
    """
    require_sm100(device)
    for max_doc in _WARMUP_MAX_DOCS:
        layout = _MSAPackedLayout.build(torch.ones((1, max_doc), dtype=torch.int64, device=device))
        plan = _MSASelectionPlan.build(
            layout,
            index_heads=index_heads,
            block_size=block_size,
            topk_blocks=topk_blocks,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )
        plan.select(
            torch.zeros((max_doc, index_heads, index_dim), dtype=torch.bfloat16, device=device),
            torch.zeros((max_doc, 1, index_dim), dtype=torch.bfloat16, device=device),
        )

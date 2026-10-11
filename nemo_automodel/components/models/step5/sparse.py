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

"""CSA proxy compression and learned region selection for Step-5."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.kernels import gather_region_payload, indexer_score_tile
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding


@dataclass(frozen=True)
class CausalDocumentLayout:
    """Native varlen layout for the valid prefix of one single-batch shard.

    Attributes:
        key_start: Physical start of the first queried document's keys.
        key_stop: Exclusive last queried token; future keys must be omitted.
        cu_seqlens_q: Int32 cumulative query counts for the intersected documents.
        cu_seqlens_k: Int32 cumulative key counts for those same documents.
        max_seqlen_q: Largest local document query segment.
        max_seqlen_k: Largest causal key prefix for one queried document.
        valid_queries: Query prefix evaluated by the native varlen kernel. Zero
            identifies a fully padded batch row with a proven padding_key.
        padding_key: Last document's first token, if a trailing padding-query
            suffix must retain the original singleton-key attention semantics.
        leading_padding: Native SWA dummy keys before the received halo, used
            to keep document-relative 96-token K tile origins aligned across CP.
            They are outside every valid query window.
    """

    key_start: int
    key_stop: int
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    valid_queries: int
    padding_key: int | None = None
    leading_padding: int = 0


@dataclass(frozen=True)
class IndexedRegionLayout:
    """Physical bounds shared across indexed CSA layers in one model call.

    Attributes:
        starts: Int32 tensor [batch, regions], first physical key in each region.
            Missing regions use global_keys. Allocated independently of Q/K/V.
        ends: Int32 tensor [batch, regions], exclusive physical key endpoint.
            Missing regions use zero. Trailing padding may span more than eight
            physical keys, but contains no valid key beyond the first eight.
    """

    starts: torch.Tensor
    ends: torch.Tensor


@dataclass(frozen=True)
class RegionSelection:
    """Sparse selection tensors, all local to one CP rank's queries.

    Attributes:
        region_ids: Sorted int32 tensor [batch, groups, local_queries, topk + 1].
            Unused slots contain a sentinel larger than any valid region.
        token_regions: Int64 tensor of shape [batch, global_keys].
        query_documents: Int64 tensor of shape [batch, local_queries].
        key_documents: Int64 tensor of shape [batch, global_keys].
        region_count: Number of global regions; also the unused-route sentinel.
        all_regions_selected: Every query in this local shard selects all causal
            regions in its own document; membership adds no restriction.
        causal_layout: Optional compact varlen layout for fully selected,
            single-batch shards with contiguous document positions.
        causal_batch_layouts: Proven prefix or all-padding layout for both rows
            of a fully selected B2 shard; None if either row needs the fallback.
        indexed_layout: Optional physical region bounds for long sparse shards.
        all_padding_key: The only allowed global key for a fully padded shard,
            after proving standard document positions and trailing padding.
    """

    region_ids: torch.Tensor
    token_regions: torch.Tensor
    query_documents: torch.Tensor
    key_documents: torch.Tensor
    region_count: int
    all_regions_selected: bool = False
    causal_layout: CausalDocumentLayout | None = None
    all_padding_key: int | None = None
    indexed_layout: IndexedRegionLayout | None = None
    causal_batch_layouts: tuple[CausalDocumentLayout, ...] | None = None


@dataclass(frozen=True)
class RegionMetadata:
    """Layer-independent CSA layout for one contiguous CP query shard.

    Attributes:
        token_regions: Int64 global region IDs [batch, global_keys].
        documents: Int64 global document IDs [batch, global_keys].
        region_documents: Int64 document IDs [batch, regions].
        current: Int64 current region IDs [batch, local_queries].
        query_documents: Int64 document IDs [batch, local_queries].
        dense_region_ids: Int32 ascending full-prefix IDs [batch, local_queries,
            min(topk, regions) + 1]; unused entries contain region_count.
        sparse_queries: Boolean [batch, local_queries], true only when more than
            topk historical regions in the same document are visible.
        score_tiles: Host tuples (batch, query_start, query_stop, first_region,
            past_last_region), bounded to 128 local queries each. Only tiles
            containing a sparse query are included.
        summary_ranges: Merged (batch, first_region, past_last_region) intervals
            needed by score tiles. Empty on prefix-only or irregular layouts.
        score_range_indices: Index into summary_ranges for each score tile.
        region_sources: Optional int32 source token indices [batch, regions,
            region_size], shared across layers. Missing slots contain -1.
        region_source_valid: Boolean shape matching region_sources, excluding
            padding tokens as well as missing slots.
        region_count: Number of global regions, also the unused-route sentinel.
        needs_projection: Whether any query on any CP rank needs learned scores.
            All ranks derive it from the same global layout before collectives.
        causal_layout: Optional native varlen metadata for this query shard.
        window_layout: Optional native SWA layout, including at most window - 1
            preceding keys. Independent of whether CSA selection is sparse.
        causal_batch_layouts: Prefix or all-padding layouts when both B2 rows
            satisfy native causal attention semantics.
        indexed_layout: Optional shared physical bounds for indexed attention.
        all_padding_key: Singleton global key for a fully padded, fully selected
            local CSA shard, or None when the shortcut cannot be established.
    """

    token_regions: torch.Tensor
    documents: torch.Tensor
    region_documents: torch.Tensor
    current: torch.Tensor
    query_documents: torch.Tensor
    dense_region_ids: torch.Tensor
    sparse_queries: torch.Tensor
    score_tiles: tuple[tuple[int, int, int, int, int], ...]
    region_count: int
    needs_projection: bool
    causal_layout: CausalDocumentLayout | None = None
    window_layout: CausalDocumentLayout | None = None
    summary_ranges: tuple[tuple[int, int, int], ...] = ()
    score_range_indices: tuple[int, ...] = ()
    region_sources: torch.Tensor | None = None
    region_source_valid: torch.Tensor | None = None
    all_padding_key: int | None = None
    indexed_layout: IndexedRegionLayout | None = None
    causal_batch_layouts: tuple[CausalDocumentLayout, ...] | None = None


def _summary_ranges(
    score_tiles: list[tuple[int, int, int, int, int]],
) -> tuple[tuple[tuple[int, int, int], ...], tuple[int, ...]]:
    """Merge each batch's needed historical intervals without inspecting GPU data."""
    ranges: list[tuple[int, int, int]] = []
    for batch, first, stop in sorted((b, first, stop) for b, _, _, first, stop in score_tiles):
        if ranges and batch == ranges[-1][0] and first <= ranges[-1][2]:
            ranges[-1] = (batch, ranges[-1][1], max(stop, ranges[-1][2]))
        else:
            ranges.append((batch, first, stop))
    indices = tuple(
        next(i for i, (rb, lo, hi) in enumerate(ranges) if rb == b and lo <= first and stop <= hi)
        for b, _, _, first, stop in score_tiles
    )
    return tuple(ranges), indices


def _summarize_region_payload(packed_keys: torch.Tensor, packed_z: torch.Tensor) -> torch.Tensor:
    """Preserve eager FP32 arithmetic and both required quantization roundings.

    Inductor can remove the intermediate BF16 rounding when this arithmetic
    is compiled. Even emulate_precision_casts changes a few FP8 summaries
    through FP32 arithmetic, so only the integer gather is compiled.
    """
    shift = packed_z.float().amax(dim=(-1, -2), keepdim=True)
    probabilities = torch.exp(packed_z.float() - shift).nan_to_num(0.0)
    summary = (probabilities * packed_keys.float()).sum(2) / probabilities.sum(2).clamp_min(1e-20)
    return summary.to(torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)


def region_metadata(
    key_positions: torch.Tensor,
    key_padding: torch.Tensor,
    *,
    query_start: int,
    queries: int,
    region_size: int,
    topk: int,
    window: int | None = None,
    indexed_csa: bool = False,
) -> RegionMetadata:
    """Prepare document/region metadata once per model forward.

    Args:
        key_positions: Int64 document-local positions [batch, global_keys].
        key_padding: Boolean padding flags [batch, global_keys].
        query_start: Physical start of this rank's contiguous query shard.
        queries: Number of local queries.
        region_size: Tokens per region, reset at every document boundary.
        topk: Maximum number of selected historical regions.
        indexed_csa: Prepare physical bounds only for regular sparse shards
            whose query-weighted historical region span reaches 32K tokens.
        window: Optional native SWA layout width. A globally trailing padded
            suffix keeps its custom-mask path. No additional host synchronization.

    Returns:
        Shared immutable metadata. Only linear token/region storage and a bounded
        [batch, local_queries, topk + 1] integer prefix template are allocated.
        One host transfer resolves allocation sizes and sparse tile extents.
    """
    documents = ((key_positions == 0) & ~key_padding).long().cumsum(1)
    region_position = key_positions // region_size
    new_region = torch.ones_like(key_positions, dtype=torch.bool)
    new_region[:, 1:] = (region_position[:, 1:] != region_position[:, :-1]) | (documents[:, 1:] != documents[:, :-1])
    token_regions = new_region.long().cumsum(1) - 1
    starts = torch.ones_like(new_region)
    starts[:, 1:] = documents[:, 1:] != documents[:, :-1]
    first_region = torch.where(starts, token_regions, 0).cummax(1).values
    all_sparse = token_regions - first_region > topk
    current = token_regions[:, query_start : query_start + queries]
    first = first_region[:, query_start : query_start + queries]
    sparse_queries = all_sparse[:, query_start : query_start + queries]
    batch, keys = key_positions.shape
    # The direct source map must have unique valid document slots. Preserve
    # the original scatter for irregular positions or internal padding.
    physical = torch.arange(keys, device=key_positions.device)[None]
    valid_key_counts = (~key_padding).sum(1, keepdim=True)
    trailing_padding = (key_padding == (physical >= valid_key_counts)).all()
    consecutive_positions = (key_positions[:, 1:] == 0) | (key_positions[:, 1:] == key_positions[:, :-1] + 1)
    summary_layout_valid = (
        trailing_padding
        & (consecutive_positions | key_padding[:, 1:]).all()
        & (key_positions >= 0).all()
        & ((key_positions == 0) | ~key_padding).all()
    )
    padding = (-queries) % 128
    lower = F.pad(torch.where(sparse_queries, first, keys), (0, padding), value=keys)
    upper = F.pad(torch.where(sparse_queries, current, 0), (0, padding))
    lower = lower.reshape(batch, -1, 128).amin(-1)
    upper = upper.reshape(batch, -1, 128).amax(-1)
    bounds = torch.stack((lower, upper), -1)
    # Transfer both B2 rows in the same metadata synchronization as B1.
    # Larger batches retain the existing conservative fallback.
    prefix_fields = key_positions.new_zeros((batch, 4))
    all_padding_fields = key_positions.new_full((batch,), -1)
    if batch in (1, 2) and queries:
        local_positions = key_positions[:, query_start : query_start + queries]
        local_valid = ~key_padding[:, query_start : query_start + queries]
        valid_queries = local_valid.sum(1)
        key_start = query_start - local_positions[:, 0]
        key_stop = query_start + valid_queries
        local_physical = physical[:, query_start : query_start + queries]
        in_prefix = (physical >= key_start[:, None]) & (physical < key_stop[:, None])
        query_prefix = (local_valid == (local_physical < key_stop[:, None])).all(1)
        padding_positions_zero = ((local_positions == 0) | local_valid).all(1)
        padding_key = torch.where(local_valid, local_physical - local_positions, -1).amax(1)
        last_valid = (valid_key_counts - 1).clamp_min(0)
        last_document_start = (last_valid - key_positions.gather(1, last_valid))[:, 0]
        all_padding_valid = (
            summary_layout_valid
            & (key_positions[:, 0] == 0)
            & (valid_queries == 0)
            & padding_positions_zero
            & (valid_key_counts[:, 0] > 0)
            & (last_document_start >= 0)
            & ~sparse_queries.any(1)
        )
        all_padding_fields = torch.where(all_padding_valid, last_document_start, -1)
        prefix_valid = (
            ~sparse_queries.any(1)
            & (valid_queries > 0)
            & query_prefix
            & ((valid_queries == queries) | (trailing_padding & padding_positions_zero))
            & (local_positions[:, 0] >= 0)
            & (key_start >= 0)
            & ~(key_padding & in_prefix).any(1)
            & (consecutive_positions | ~in_prefix[:, 1:]).all(1)
        )
        window_valid = key_positions.new_zeros(batch)
        if window is not None and batch == 1:
            window_start = torch.maximum(key_start, key_start.new_tensor(max(0, query_start - window + 1)))
            in_window = (physical >= window_start[:, None]) & (physical < key_stop[:, None])
            window_valid = (
                (valid_queries > 0)
                & query_prefix
                & ((valid_queries == queries) | (trailing_padding & padding_positions_zero))
                & (local_positions[:, 0] >= 0)
                & (key_start >= 0)
                & ~(key_padding & in_window).any(1)
                & (consecutive_positions | ~in_window[:, 1:]).all(1)
            ).long()
        prefix_fields = torch.cat(
            (torch.stack((prefix_valid.long(), valid_queries, padding_key, window_valid), dim=1), local_positions),
            dim=1,
        )
    # One synchronization for the entire model call, never one per layer/tile.
    layout = (
        torch.cat(
            (
                (token_regions.amax() + 1).reshape(1),
                all_sparse.any().long().reshape(1),
                bounds.reshape(-1),
                prefix_fields.reshape(-1),
                all_padding_fields,
                summary_layout_valid.long().reshape(1),
            )
        )
        .cpu()
        .tolist()
    )
    region_count, needs_projection = int(layout[0]), bool(layout[1])
    all_padding_key = int(layout[-2]) if batch == 1 and layout[-2] >= 0 else None
    tiles_per_batch = lower.shape[1]
    score_tiles = []
    for index in range(batch * tiles_per_batch):
        lo, hi = int(layout[2 + 2 * index]), int(layout[3 + 2 * index])
        if lo < hi:
            start = (index % tiles_per_batch) * 128
            score_tiles.append((index // tiles_per_batch, start, min(start + 128, queries), lo, hi))
    summary_ranges: tuple[tuple[int, int, int], ...] = ()
    score_range_indices: tuple[int, ...] = ()
    region_sources = None
    region_source_valid = None
    if score_tiles and bool(layout[-1]):
        summary_ranges, score_range_indices = _summary_ranges(score_tiles)
        # Shared integer metadata only; K/Z remain in their original global
        # buffers. A trailing padded short document can collide at slot zero,
        # but its region is never a sparse query's legal historical candidate.
        region_sources = torch.full(
            (batch, region_count, region_size), -1, dtype=torch.int32, device=key_positions.device
        )
        slots = token_regions * region_size + key_positions.remainder(region_size)
        source_tokens = torch.arange(keys, dtype=torch.int32, device=key_positions.device).expand(batch, -1)
        region_sources.view(batch, -1).scatter_(1, slots, source_tokens)
        source_padding = key_padding.gather(1, region_sources.long().clamp_min(0).view(batch, -1))
        region_source_valid = (region_sources >= 0) & ~source_padding.view_as(region_sources)
    indexed_layout = None
    history_work = (
        sum((stop - start) * (last - first) * region_size for _, start, stop, first, last in score_tiles)
        if indexed_csa
        else 0
    )
    if (
        indexed_csa
        and score_tiles
        and bool(layout[-1])
        and region_size == 8
        and history_work >= 32768 * batch * queries
    ):
        # One integer map per model call, shared by every local CSA layer.
        # Unlike compression slots, physical bounds cannot lose a document's
        # first valid key when trailing padding also has position zero.
        region_starts = torch.full((batch, region_count), keys, device=key_positions.device, dtype=torch.int32)
        region_ends = torch.zeros_like(region_starts)
        physical_indices = torch.arange(keys, device=key_positions.device, dtype=torch.int32).expand(batch, -1)
        region_starts.scatter_reduce_(1, token_regions, physical_indices, reduce="amin")
        region_ends.scatter_reduce_(1, token_regions, physical_indices + 1, reduce="amax")
        indexed_layout = IndexedRegionLayout(region_starts, region_ends)
    causal_layout = None
    causal_batch_layouts = None
    window_layout = None
    prefix_offset = 2 + 2 * batch * tiles_per_batch
    if batch in (1, 2) and queries:
        row_layouts: list[CausalDocumentLayout] = []
        for row in range(batch):
            offset = prefix_offset + row * prefix_fields.shape[1]
            if bool(layout[offset]):
                valid_queries, padding_key = layout[offset + 1 : offset + 3]
                local_positions = layout[offset + 4 : offset + 4 + valid_queries]
                query_boundaries = (
                    [0] + [i for i in range(1, valid_queries) if local_positions[i] == 0] + [valid_queries]
                )
                key_boundaries = [0] + [end + local_positions[0] for end in query_boundaries[1:]]
                cumulative = torch.tensor(
                    query_boundaries + key_boundaries, dtype=torch.int32, device=key_positions.device
                )
                cu_q, cu_k = cumulative.chunk(2)
                row_layouts.append(
                    CausalDocumentLayout(
                        query_start - local_positions[0],
                        query_start + valid_queries,
                        cu_q,
                        cu_k,
                        max(stop - start for start, stop in zip(query_boundaries, query_boundaries[1:])),
                        max(stop - start for start, stop in zip(key_boundaries, key_boundaries[1:])),
                        valid_queries,
                        padding_key if valid_queries < queries else None,
                    )
                )
            elif batch == 2 and layout[-batch - 1 + row] >= 0:
                padding_key = int(layout[-batch - 1 + row])
                cumulative = torch.tensor([0, 0, 0, 1], dtype=torch.int32, device=key_positions.device)
                cu_q, cu_k = cumulative.chunk(2)
                row_layouts.append(
                    CausalDocumentLayout(padding_key, padding_key + 1, cu_q, cu_k, 0, 1, 0, padding_key=padding_key)
                )
        if batch == 1 and row_layouts:
            causal_layout = row_layouts[0]
        elif batch == 2 and len(row_layouts) == batch:
            causal_batch_layouts = tuple(row_layouts)
    if batch == 1 and window is not None and bool(layout[prefix_offset + 3]):
        valid_queries, padding_key = layout[prefix_offset + 1 : prefix_offset + 3]
        local_positions = layout[prefix_offset + 4 : prefix_offset + 4 + valid_queries]
        prefix_keys = min(local_positions[0], window - 1)
        # FA4 SM90 D192 uses 96-token K tiles for native local attention.
        # Its first segment can start mid-document on a CP rank. Dummy keys
        # before the halo restore the same tile origins as CP1; bottom-right
        # window masking excludes them from every real query.
        leading_padding = (local_positions[0] - prefix_keys) % 96
        prefix_keys += leading_padding
        query_boundaries = [0] + [i for i in range(1, valid_queries) if local_positions[i] == 0] + [valid_queries]
        key_boundaries = [0] + [end + prefix_keys for end in query_boundaries[1:]]
        cumulative = torch.tensor(query_boundaries + key_boundaries, dtype=torch.int32, device=key_positions.device)
        cu_q, cu_k = cumulative.chunk(2)
        window_layout = CausalDocumentLayout(
            query_start - prefix_keys,
            query_start + valid_queries,
            cu_q,
            cu_k,
            max(stop - start for start, stop in zip(query_boundaries, query_boundaries[1:])),
            max(stop - start for start, stop in zip(key_boundaries, key_boundaries[1:])),
            valid_queries,
            padding_key=padding_key if valid_queries < queries else None,
            leading_padding=leading_padding,
        )
    region_documents = torch.zeros(batch, region_count, device=key_positions.device, dtype=torch.long)
    region_documents.scatter_reduce_(1, token_regions, documents, reduce="amax")
    budget = min(topk, region_count)
    prefix = first.unsqueeze(-1) + torch.arange(budget + 1, device=key_positions.device)
    dense_ids = torch.where(prefix <= current.unsqueeze(-1), prefix, region_count).to(torch.int32)
    return RegionMetadata(
        token_regions,
        documents,
        region_documents,
        current,
        documents[:, query_start : query_start + queries],
        dense_ids,
        sparse_queries,
        tuple(score_tiles),
        region_count,
        needs_projection,
        causal_layout,
        window_layout,
        summary_ranges,
        score_range_indices,
        region_sources,
        region_source_valid,
        all_padding_key,
        indexed_layout,
        causal_batch_layouts,
    )


class Step5IndexerRMSNorm(Step3p5RMSNorm):
    """Indexer RMSNorm with source-required activation-dtype parameter rounding."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Normalize [..., proxy_dim] activations and preserve their dtype."""
        raw = hidden.float()
        scale = self.weight.to(hidden.dtype).float() + 1
        return (raw * torch.rsqrt(raw.square().mean(-1, keepdim=True) + self.variance_epsilon) * scale).to(hidden.dtype)


class Step5IndexerLayerNorm(nn.LayerNorm):
    """Indexer variance E[x²] - E[x]², evaluated inside the module's FSDP hooks."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Normalize [..., proxy_dim] with BF16-rounded scale/bias and FP32 math."""
        raw = hidden.float()
        mean = raw.mean(-1, keepdim=True)
        variance = raw.square().mean(-1, keepdim=True) - mean.square()
        scale = self.weight.to(hidden.dtype).float()
        bias = self.bias.to(hidden.dtype).float()
        return ((raw - mean) * torch.rsqrt(variance + self.eps) * scale + bias).to(hidden.dtype)


class Step5IndexerWeightProjection(nn.Linear):
    """Keep FP32 source weights while matching the reference BF16 projection."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project [..., hidden_size] to signed FP32 indexer head weights."""
        return F.linear(hidden, self.weight.to(hidden.dtype)).float()


class Step5SparseIndexer(nn.Module):
    """Checkpoint-backed, discrete CSA selector; LM loss does not train top-k indices."""

    def __init__(self, config: Step5Config, layer_idx: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config.sparse_config
        width = self.config.proxy_dim
        self.q_proj = nn.Linear(
            config.hidden_size, self.config.sparse_indexer_num_heads * width, bias=False, dtype=dtype
        )
        self.k_proj = nn.Linear(config.hidden_size, width, bias=False, dtype=dtype)
        self.z_proj = nn.Linear(config.hidden_size, width, bias=False, dtype=dtype)
        self.w_proj = Step5IndexerWeightProjection(
            config.hidden_size, self.config.sparse_indexer_num_heads, bias=False, dtype=torch.float32
        )
        self.q_norm = Step5IndexerRMSNorm(width, config.rms_norm_eps)
        self.k_norm = Step5IndexerLayerNorm(width, eps=config.rms_norm_eps, dtype=torch.float32)
        self.register_buffer("ssmax_s", torch.zeros(config.num_attention_heads, dtype=torch.float32))
        span = self.config.sparse_indexer_rope_dim if self.config.sparse_indexer_use_rope else 0
        self.rotary_emb = Step5RotaryEmbedding(config, layer_idx, span=span)

    def project_keys(self, hidden: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute normalized index keys and per-dimension compression logits.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden].
            positions: Tensor of shape [batch, sequence].

        Returns:
            Two tensors of shape [batch, sequence, proxy_dim], in the hidden dtype.
        """
        key = self.k_proj(hidden)
        key = self.k_norm(key)
        key = self.rotary_emb(key.unsqueeze(2), positions).squeeze(2)
        return key, self.z_proj(hidden)

    @torch.no_grad()
    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        keys: torch.Tensor | None,
        compression_logits: torch.Tensor | None,
        key_positions: torch.Tensor,
        key_padding: torch.Tensor,
        *,
        query_start: int = 0,
        metadata: RegionMetadata | None = None,
    ) -> RegionSelection:
        """Select historical CSA regions, scoring only genuinely sparse queries.

        Args:
            hidden: Tensor of shape [batch, local_queries, hidden].
            positions: Int64 document-local positions [batch, local_queries].
            keys: Projected global index keys [batch, global_keys, proxy_dim], or
                None when every global query sees at most topk historical regions.
            compression_logits: Same shape as keys, or None on that full-prefix path.
            key_positions: Int64 document-local positions [batch, global_keys].
            key_padding: Boolean padding flags [batch, global_keys].
            query_start: Physical start of the contiguous local query shard.
            metadata: Shared document/region layout for the current model call.
                Direct component callers may omit it.

        Returns:
            Ascending selected region IDs [batch, groups, local_queries, topk + 1]
            with the shared global token/document mapping. Fully selected prefixes
            do not depend on scores, signed head weights, or tie breaking.
        """
        cfg = self.config
        batch, queries = hidden.shape[:2]
        groups = cfg.num_provider_groups
        heads = cfg.sparse_indexer_num_heads // groups
        dim = cfg.proxy_dim
        if metadata is None:
            metadata = region_metadata(
                key_positions,
                key_padding,
                query_start=query_start,
                queries=queries,
                region_size=cfg.region_block_size,
                topk=cfg.topk,
            )
        region_count = metadata.region_count
        ids = metadata.dense_region_ids[:, None].expand(-1, groups, -1, -1)
        # FP32 norm/projection modules can own FSDP collectives. Every rank
        # invokes their hooks in the same order, including zero-token calls
        # when this shard has no sparse queries; no projection GEMM is needed.
        projected_hidden = hidden if metadata.score_tiles else hidden[:, :0]
        projected_positions = positions if metadata.score_tiles else positions[:, :0]
        projected_queries = projected_hidden.shape[1]
        query = self.q_proj(projected_hidden).view(batch, projected_queries, groups * heads, dim)
        query = self.q_norm(query)
        query = self.rotary_emb(query, projected_positions).view(batch, projected_queries, groups, heads, dim)
        weights = self.w_proj(projected_hidden).view(batch, projected_queries, groups, heads)
        weights = weights * heads**-0.5
        if not metadata.score_tiles:
            return RegionSelection(
                ids,
                metadata.token_regions,
                metadata.query_documents,
                metadata.documents,
                region_count,
                all_regions_selected=True,
                causal_layout=metadata.causal_layout,
                causal_batch_layouts=metadata.causal_batch_layouts,
                all_padding_key=metadata.all_padding_key,
            )
        if keys is None or compression_logits is None:
            raise ValueError("Sparse CSA queries require the global indexer projections")
        if metadata.region_sources is None:
            slot = key_positions % cfg.region_block_size
            rows = torch.arange(batch, device=keys.device).unsqueeze(1)
            packed_keys = keys.new_zeros(batch, region_count, cfg.region_block_size, dim)
            packed_z = torch.full_like(packed_keys, float("-inf"))
            packed_keys[rows, metadata.token_regions, slot] = keys
            packed_z[rows, metadata.token_regions, slot] = compression_logits.masked_fill(
                key_padding.unsqueeze(-1), float("-inf")
            )
            summaries = (_summarize_region_payload(packed_keys, packed_z),)
        else:
            assert metadata.region_source_valid is not None
            summaries = tuple(
                _summarize_region_payload(
                    *gather_region_payload(
                        keys[b : b + 1],
                        compression_logits[b : b + 1],
                        metadata.region_sources[b : b + 1, first:last],
                        metadata.region_source_valid[b : b + 1, first:last],
                    )
                )
                for b, first, last in metadata.summary_ranges
            )
        query = query.to(torch.float8_e4m3fn).to(torch.bfloat16)
        budget = min(cfg.topk, region_count)
        region_ids = ids.clone()
        for tile, (b, start, stop, first, last) in enumerate(metadata.score_tiles):
            # Only this tile's documents and visible history can affect top-k.
            if metadata.region_sources is None:
                summary = summaries[0][b : b + 1, first:last]
            else:
                range_index = metadata.score_range_indices[tile]
                base = metadata.summary_ranges[range_index][1]
                summary = summaries[range_index][:, first - base : last - base]
            scores = indexer_score_tile(query[b : b + 1], weights[b : b + 1], summary, start, stop)
            region_range = torch.arange(first, last, device=hidden.device)
            current = metadata.current[b : b + 1, start:stop]
            candidates = (region_range < current.unsqueeze(-1)) & (
                metadata.region_documents[b : b + 1, None, first:last]
                == metadata.query_documents[b : b + 1, start:stop, None]
            )
            scores.masked_fill_(~candidates[:, None], float("-inf"))
            cutoff = scores.topk(budget, dim=-1, sorted=False).values.amin(-1, keepdim=True)
            greater = (scores > cutoff) & candidates[:, None]
            tied = (scores == cutoff) & candidates[:, None]
            remaining = budget - greater.sum(-1, keepdim=True)
            selected = greater | (tied & (tied.cumsum(-1) <= remaining))
            selected_ids = torch.where(selected, region_range, region_count)
            historical = selected_ids.topk(budget, dim=-1, largest=False, sorted=True).values
            selected_ids = torch.cat((historical, current[:, None, :, None].expand(-1, groups, -1, -1)), -1)
            # Sparse rows have exactly budget historical entries before current;
            # dense rows retain the independently constructed prefix template.
            region_ids[b : b + 1, :, start:stop] = torch.where(
                metadata.sparse_queries[b : b + 1, None, start:stop, None],
                selected_ids,
                region_ids[b : b + 1, :, start:stop],
            )
        return RegionSelection(
            region_ids,
            metadata.token_regions,
            metadata.query_documents,
            metadata.documents,
            region_count,
            indexed_layout=metadata.indexed_layout,
        )

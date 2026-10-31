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

"""Differentiable CSA2 attention from the official DeepSeek V4.1 inference model.

Full layers publish compressed KV, index keys, and selected positions. Reindex
layers replace only the selection; Reuse layers consume it unchanged. Immutable
per-forward state keeps source gradients intact through activation recomputation.
Weights are dequantized for training while the released FP8 window KV and FP4
compressed KV/indexer representations retain their quantize/dequantize boundaries. The released indexer
weights are frozen because the hard top-k operation provides no language-model
gradient and the inference release does not implement indexer distillation.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F
from torch.utils._python_dispatch import _disable_current_modes

from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.common.rope_triton import apply_rope_triton
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4FP32Parameter, DeepseekV4GroupedLinear
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import (
    SparseIndexPlan,
    dsv4_sparse_attention,
    plan_dsv4_sparse_indices,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.cp import gather_sequence, gather_window
from nemo_automodel.components.models.deepseek_v41.indexer import indexer_scores
from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41RMSNorm
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache
from nemo_automodel.shared.recompute_replay import RecomputeReplay
from nemo_automodel.shared.utils import dtype_from_str


@dataclass(frozen=True)
class DeepseekV41AttentionState:
    """CSA2 state owned by one full-sequence model forward.

    Attributes:
        global_valid: Optional batch validity [batch, global_sequence], shared across layers.
        global_seq_ids: Optional global document IDs [batch, global_sequence].
        compressed_kv: Rotated KV tensor of shape [batch, compressed, head_dim].
        index_keys: Rotated index keys of shape [batch, compressed, index_head_dim].
        topk_indices: Compressed-relative indices of shape [batch, sequence, topk],
            with -1 denoting absent positions.
        candidates: Optional block mask [batch, sequence, ceil(compressed / candidate_block_size)].
            For packed input each query uses its own document's block coordinates.
        compressed_valid: Optional valid-group mask of shape [batch, compressed].
        compressed_seq_ids: Optional document IDs [batch, compressed], zero for padding.
        compression_ratio: Number of tokens represented by a compressed position.
        sparse_plan: Optional :class:`SparseIndexPlan` of the current selection for the TileLang /
            cuDNN backends, built by the first layer that consumes the selection and reused by every
            later layer of the same lineage; value-only, dropped whenever the selection changes.

    Under CP, compressed and index-key axes span the global sequence while
    sequence axes in topk_indices and candidates contain only local queries.

    Tensor fields retain autograd history and are never modified by consumers.
    A model creates an empty state for every forward, including every microbatch.
    """

    global_valid: torch.Tensor | None = None
    global_seq_ids: torch.Tensor | None = None
    compressed_kv: torch.Tensor | None = None
    index_keys: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    compressed_valid: torch.Tensor | None = None
    compression_ratio: int = 0
    compressed_seq_ids: torch.Tensor | None = None
    sparse_plan: SparseIndexPlan | None = None


@dataclass(frozen=True)
class DeepseekV41AttentionOutput:
    """Attention result and the shared state for the next layer.

    Attributes:
        hidden_states: Tensor of shape [batch, sequence, hidden].
        state: Shared tensor layouts documented by DeepseekV41AttentionState.
    """

    hidden_states: torch.Tensor
    state: DeepseekV41AttentionState


@dataclass(frozen=True)
class _SelectionRecord:
    """The value-only state fields one layer produced, replayed to its activation-checkpoint recompute.

    Attributes:
        index_keys: Rotated index keys of shape [batch, compressed, index_head_dim], or ``None``.
        topk_indices: Compressed-relative selection of shape [batch, sequence, topk], ``-1`` absent, or ``None``.
        candidates: Optional block mask [batch, sequence, ceil(compressed / candidate_block_size)].
            For packed input each query uses its own document's block coordinates.
        sparse_plan: The lineage's :class:`SparseIndexPlan`, or ``None`` for the dense backends.
    """

    index_keys: torch.Tensor | None
    topk_indices: torch.Tensor | None
    candidates: torch.Tensor | None
    sparse_plan: SparseIndexPlan | None


csa2_selection_replay: RecomputeReplay[_SelectionRecord] = RecomputeReplay("CSA2 selection")
"""Record/replay channel for the frozen indexer's selection and the sparse-index plan across AC recompute.

``DeepseekV41Block.nemo_checkpoint_context_fn`` binds it around each checkpointed block call when
``DeepseekV41TextConfig.reuse_sparse_selection_on_recompute`` is set: the checkpoint forward records what a
layer adds to the state, the recompute takes it back instead of re-running the indexer and the index
prologue. Both are integer / frozen-score work autograd never reads, so gradients are unchanged.
"""


@contextmanager
def csa2_selection_region() -> Iterator[None]:
    """Run the frozen indexer and the sparse-index planning outside any active TorchDispatch mode.

    Selective activation checkpointing records every op of the checkpoint forward through a TorchDispatch
    mode and indexes the recompute against that record. Replaying the recorded selection skips these ops
    in the recompute, so they must be invisible to the mode in the forward as well (as Qwen3.8-Flash-Next's
    ``qsa_route_selection_region`` does). Everything inside runs under ``torch.no_grad``; hiding the
    region only affects checkpoint bookkeeping.
    """
    with _disable_current_modes():
        yield


class _RotaryEmbedding(nn.Module):
    """Adjacent-pair RoPE with the official frequency-only YaRN adjustment."""

    def __init__(self, config: DeepseekV41TextConfig, *, compressed: bool) -> None:
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.theta = config.compress_rope_theta if compressed else config.rope_theta
        scaling = config.rope_scaling if compressed else None
        self.factor = float(scaling.get("factor", 1.0)) if scaling else 1.0
        self.original_length = int(scaling.get("original_max_position_embeddings", 0)) if scaling else 0
        self.beta_fast = float(scaling.get("beta_fast", 32)) if scaling else 32.0
        self.beta_slow = float(scaling.get("beta_slow", 1)) if scaling else 1.0

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Construct FP32 phase angles without a cast-sensitive frequency buffer.

        Args:
            positions: Integer tensor of shape [batch, sequence].

        Returns:
            FP32 angles of shape [batch, sequence, rotary_pairs], where
            rotary_pairs is half the rotary head dimension.
        """
        frequencies = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=positions.device, dtype=torch.float32) / self.dim)
        )
        if self.original_length > 0:
            low = max(
                math.floor(
                    self.dim
                    * math.log(self.original_length / (self.beta_fast * 2 * math.pi))
                    / (2 * math.log(self.theta))
                ),
                0,
            )
            high = min(
                math.ceil(
                    self.dim
                    * math.log(self.original_length / (self.beta_slow * 2 * math.pi))
                    / (2 * math.log(self.theta))
                ),
                self.dim - 1,
            )
            ramp = (
                (torch.arange(self.dim // 2, device=positions.device, dtype=torch.float32) - low)
                / max(high - low, 1e-3)
            ).clamp(0, 1)
            frequencies = frequencies / self.factor * ramp + frequencies * (1 - ramp)
        return positions.float().unsqueeze(-1) * frequencies


_ROPE_IMPL = "torch"


def use_triton_rope(enabled: bool) -> None:
    """Route ``_apply_rope`` through the fused Triton kernel.

    The Triton path is opt-in and off by default; the eager path stays the reference.

    Args:
        enabled: Use :func:`apply_rope_triton` when True, the eager implementation when False.
    """
    global _ROPE_IMPL
    _ROPE_IMPL = "triton" if enabled else "torch"


def _apply_rope(values: torch.Tensor, angles: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Rotate the final channels without changing the input storage.

    Args:
        values: Tensor of shape [batch, sequence, channels] or
            [batch, sequence, heads, channels].
        angles: FP32 tensor of shape [batch, sequence, rotary_pairs]. The last
            2 * rotary_pairs channels of values use adjacent-pair rotation.
        inverse: Conjugate the rotation for the attention output.

    Returns:
        Tensor with the shape and dtype of values, in independent storage.
    """
    if _ROPE_IMPL == "triton":
        return apply_rope_triton(values, angles, inverse=inverse)
    rotary_dim = angles.shape[-1] * 2
    pairs = torch.view_as_complex(values[..., -rotary_dim:].float().unflatten(-1, (-1, 2)).contiguous())
    # Preserve the released complex-multiply rounding. Separate real-valued
    # multiplies change some BF16 values and can cross subsequent FP4/FP8 bins.
    frequencies = torch.polar(torch.ones_like(angles), angles)
    if values.ndim == 4:
        frequencies = frequencies.unsqueeze(2)
    if inverse:
        frequencies = frequencies.conj()
    rotated = torch.view_as_real(pairs * frequencies).flatten(-2).to(values.dtype)
    # Assemble the result as a contiguous copy of the input plus one narrow slice write. The values are
    # bitwise those of ``torch.cat((values[..., :-rotary_dim], rotated), dim=-1)``; the batched-concat
    # kernel that cat launches for a [batch, sequence, heads, 512] query reads and writes the whole tensor
    # through 128-byte chunks (1.9 ms per call on GB200), the clone streams it (0.4 ms).
    output = values.clone(memory_format=torch.contiguous_format)
    output[..., -rotary_dim:] = rotated
    return output


class _CompressorLinear(nn.Linear):
    """Keep pooling weights in FP32 while selecting compute from the input dtype."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Project using the input's explicit compute dtype.

        Args:
            values: Tensor of shape [..., input] with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., output] in the input dtype.
        """
        with torch.autocast(device_type=values.device.type, enabled=False):
            return F.linear(values, self.weight.to(values.dtype))


class _Compressor(nn.Module):
    """Non-overlapping channelwise softmax pooling; ratio one is a projection."""

    def __init__(
        self, config: DeepseekV41TextConfig, *, ratio: int, dtype: torch.dtype, rms_norm: str = "torch_fp32"
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.wkv = _CompressorLinear(config.hidden_size, config.head_dim, bias=False, dtype=torch.float32)
        norm = (
            partial(initialize_rms_norm_module, "te", device=self.wkv.weight.device)
            if rms_norm == "te"
            else DeepseekV41RMSNorm
        )
        self.norm = norm(config.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        if ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return complete compressed groups before rotary embedding.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, floor(sequence / ratio), head_dim] in the
            input dtype. Incomplete trailing groups contribute no output.
        """
        if self.ratio == 1:
            return self.norm(self.wkv(hidden_states))
        length = hidden_states.shape[1] // self.ratio * self.ratio
        values = self.wkv(hidden_states[:, :length].float())
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            scores = self.wgate(hidden_states[:, :length].float())
        values = values.unflatten(1, (-1, self.ratio))
        scores = scores.unflatten(1, (-1, self.ratio))
        pooled = (values * scores.softmax(dim=2)).sum(dim=2).to(hidden_states.dtype)
        return self.norm(pooled)


def _select_candidate_blocks(
    scores: torch.Tensor, visible_lengths: torch.Tensor, *, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Keep high-scoring blocks and always retain the latest visible block.

    Args:
        scores: Causally masked scores of shape [batch, sequence, compressed].
        visible_lengths: Integer tensor of shape [batch, sequence, 1] counting
            visible compressed positions for each query.
        topk_blocks: Maximum retained blocks per query.
        block_size: Compressed positions per block.

    Returns:
        Boolean candidate-block mask [batch, sequence, ceil(compressed / block_size)].
    """
    width = scores.shape[-1]
    block_scores = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    block_scores = block_scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last_block = (visible_lengths - 1) // block_size
    block_ids = torch.arange(block_scores.shape[-1], device=scores.device)
    block_scores = block_scores.masked_fill(block_ids == last_block, torch.inf)
    selected = block_scores.topk(min(topk_blocks, block_scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter(-1, selected.indices, selected.values > -torch.inf)
    return keep


class _Indexer(nn.Module):
    """Frozen released CSA2 indexer with shared keys and hierarchical selection."""

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        *,
        layer_idx: int,
        dtype: torch.dtype,
        rms_norm: str = "torch_fp32",
        attn_backend: str = "eager",
    ) -> None:
        super().__init__()
        self.attn_backend = attn_backend
        self.owns_keys = layer_idx in config.kv_source_layer_ids
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False, dtype=dtype)
        if self.owns_keys:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False, dtype=dtype)
            norm = (
                partial(initialize_rms_norm_module, "te", device=self.wk.weight.device)
                if rms_norm == "te"
                else DeepseekV41RMSNorm
            )
            self.k_norm = norm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        query_latent: torch.Tensor,
        latent: torch.Tensor | None,
        angles: torch.Tensor,
        compressed_angles: torch.Tensor,
        state: DeepseekV41AttentionState,
        position_ids: torch.Tensor | None = None,
        cp_group: dist.ProcessGroup | None = None,
        packed_seq_ids: torch.Tensor | None = None,
    ) -> DeepseekV41AttentionState:
        """Produce index keys when owned, then replace the current selection.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            query_latent: Tensor of shape [batch, sequence, q_lora_rank].
            latent: Optional unrotated KV of shape [batch, compressed, head_dim].
            angles: FP32 rotary angles of shape [batch, sequence, rotary_pairs].
            compressed_angles: FP32 angles of shape [batch, compressed, rotary_pairs].
            state: Shared tensors with layouts in DeepseekV41AttentionState.
            position_ids: Optional physical global positions [batch, local_sequence].
            packed_seq_ids: Optional document IDs [batch, local_sequence], zero for padding.
            cp_group: Optional CP group; latent and compressed_angles contain
                local complete groups, while state index keys span all ranks.

        Returns:
            New state with index_keys [batch, compressed, index_head_dim],
            topk_indices [batch, sequence, topk], and optional candidates
            [batch, sequence, candidate_blocks]. Compressed KV storage is preserved;
            the sparse-index plan is dropped because a new selection needs a new plan.
        """
        keys = state.index_keys
        if self.owns_keys:
            if latent is None:
                raise ValueError("A Full CSA2 layer must provide its unrotated compressed latent")
            keys = quantize_cache(
                _apply_rope(self.k_norm(self.wk(latent)), compressed_angles), format="mxfp4", block_size=32
            )
            keys = gather_sequence(keys, cp_group)
        if keys is None:
            raise ValueError("A Reindex CSA2 layer requires index keys from a preceding Full layer")
        batch, sequence, _ = hidden_states.shape
        width = keys.shape[1]
        if width == 0:
            return replace(
                state,
                index_keys=keys,
                topk_indices=torch.empty(batch, sequence, 0, dtype=torch.long, device=hidden_states.device),
                candidates=(
                    torch.empty(batch, sequence, 0, dtype=torch.bool, device=hidden_states.device)
                    if self.is_candidate_source
                    else state.candidates
                ),
                sparse_plan=None,
            )
        queries = self.wq_b(query_latent).unflatten(-1, (self.num_heads, self.head_dim))
        queries = quantize_cache(_apply_rope(queries, angles), format="mxfp4", block_size=32)
        weights = self.weights_proj(hidden_states) * (self.head_dim**-0.5 * self.num_heads**-0.5)
        if position_ids is None:
            position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
        return self._select_queries(
            queries, keys, weights, position_ids.expand(batch, -1), packed_seq_ids, replace(state, index_keys=keys)
        )

    def _select_queries(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        sequence_ids: torch.Tensor | None,
        state: DeepseekV41AttentionState,
    ) -> DeepseekV41AttentionState:
        """Score query tiles against only their document's complete key interval.

        Args:
            queries: Rotated BF16 index queries [batch, local_sequence, heads, index_dim].
            keys: Global rotated index keys [batch, compressed, index_dim].
            weights: BF16 per-head weights [batch, local_sequence, heads].
            positions: Global physical query positions [batch, local_sequence].
            sequence_ids: Optional local document IDs [batch, local_sequence].
            state: Shared tensors documented by DeepseekV41AttentionState.

        Returns:
            Immutable state with top-k global key indices [batch, local_sequence, topk]
            and candidate-block masks [batch, local_sequence, ceil(compressed / block_size)].
            No global query-by-key scores or tokenwise candidate mask are retained.
        """
        batch, sequence = queries.shape[:2]
        width = keys.shape[1]
        indices = torch.full((batch, sequence, min(self.topk, width)), -1, dtype=torch.long, device=queries.device)
        block_count = (width + self.candidate_block_size - 1) // self.candidate_block_size
        candidates = (
            torch.zeros(batch, sequence, block_count, dtype=torch.bool, device=queries.device)
            if self.is_candidate_source
            else state.candidates
        )
        if self.uses_candidates and (candidates is None or candidates.shape != (batch, sequence, block_count)):
            raise ValueError("A hierarchical Reindex layer requires candidates from its source layer")
        if sequence_ids is not None and (state.compressed_seq_ids is None or state.compressed_valid is None):
            raise ValueError("Packed CSA2 state requires compressed document IDs and validity")
        for row in range(batch):
            documents = [None] if sequence_ids is None else sequence_ids[row].unique().tolist()
            for document in documents:
                if document == 0:
                    continue
                query_ids = (
                    torch.arange(sequence, device=queries.device)
                    if document is None
                    else torch.where(sequence_ids[row] == document)[0]
                )
                key_ids = (
                    torch.arange(width, device=keys.device)
                    if document is None
                    else torch.where((state.compressed_seq_ids[row] == document) & state.compressed_valid[row])[0]
                )
                if key_ids.numel() == 0:
                    continue
                document_keys = keys[row : row + 1, key_ids]
                # Keep the entire document key width: topk tie-breaking depends on
                # that width. Only the independent query axis is tiled.
                # Small packed documents fit in one tile; long documents keep
                # bounded score storage instead of paying one launch per 128 queries.
                query_tile_size = max(128, 1024 * 1024 // key_ids.numel())
                for query_tile in query_ids.split(query_tile_size):
                    visible = (positions[row : row + 1, query_tile] + 1) // state.compression_ratio
                    allowed = key_ids.view(1, 1, -1) < visible.unsqueeze(-1)
                    if state.compressed_valid is not None:
                        allowed = allowed & state.compressed_valid[row : row + 1, key_ids].unsqueeze(1)
                    q = queries[row : row + 1, query_tile]
                    w = weights[row : row + 1, query_tile]
                    if self.attn_backend in ("tilelang", "cudnn"):
                        scores = indexer_scores(q, document_keys, w, allowed)
                    else:
                        # Preserve the reference BF16 matmul and product rounding.
                        scores = torch.einsum("bshd,btd->bsht", q, document_keys)
                        scores = (scores.relu() * w.unsqueeze(-1)).sum(dim=2)
                        scores = scores.masked_fill(~allowed, -torch.inf)
                    if self.is_candidate_source:
                        block_visible = (
                            torch.isfinite(scores).sum(-1, keepdim=True)
                            if document is not None
                            else visible.unsqueeze(-1)
                        )
                        keep = _select_candidate_blocks(
                            scores,
                            block_visible,
                            topk_blocks=self.candidate_topk_blocks,
                            block_size=self.candidate_block_size,
                        )
                        candidates[row, query_tile, : keep.shape[-1]] = keep[0]
                    elif self.uses_candidates:
                        block_ids = torch.arange(key_ids.numel(), device=keys.device) // self.candidate_block_size
                        keep = candidates[row, query_tile][:, block_ids]
                        scores = scores.masked_fill(~keep.unsqueeze(0), -torch.inf)
                    selected = scores.topk(min(self.topk, key_ids.numel()), dim=-1, sorted=False)
                    ordered = selected.indices.sort(-1).values
                    valid = torch.isfinite(scores.gather(-1, ordered))
                    indices[row, query_tile, : ordered.shape[-1]] = torch.where(valid, key_ids[ordered], -1)[0]
        return replace(state, topk_indices=indices, candidates=candidates, sparse_plan=None)


def validate_attention_inputs(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    packed_seq_ids: torch.Tensor | None,
    *,
    sequence: int,
    cp_group: dist.ProcessGroup | None = None,
) -> None:
    """Run the host-synchronising checks on the per-forward inputs once.

    Each check reads a device tensor on the host (``torch.equal`` / ``torch.all`` / ``torch.any``), which drains the
    CUDA launch queue. The model calls this once per forward and passes ``inputs_validated=True`` to every layer;
    a layer called directly (``inputs_validated=False``) still validates its own inputs. Shape checks stay in the
    layer because they cost nothing.
    """
    cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else dist.get_rank(cp_group)
    positions = torch.arange(sequence, device=position_ids.device) + cp_rank * sequence
    if packed_seq_ids is None and not torch.equal(position_ids, positions.expand_as(position_ids)):
        raise ValueError("DeepSeek V4.1 attention supports only contiguous zero-based full-sequence position_ids")
    if attention_mask is not None:
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("DeepSeek V4.1 attention_mask must contain only zero and one")
        valid_tokens = attention_mask.bool()
        if packed_seq_ids is None and torch.any(valid_tokens[:, 1:] & ~valid_tokens[:, :-1]):
            raise ValueError("DeepSeek V4.1 compression supports right padding only")
        if packed_seq_ids is None and cp_size > 1:
            global_valid = gather_sequence(valid_tokens, cp_group)
            if torch.any(global_valid[:, 1:] & ~global_valid[:, :-1]):
                raise ValueError("DeepSeek V4.1 compression supports right padding only across CP ranks")


class DeepseekV41Attention(nn.Module):
    """Full-sequence CSA2 with local KV, shared compressed KV, and an attention sink.

    The training implementation supports eager, SDPA and TileLang attention with
    torch linear layers and eager FP32 or TE RMSNorm. Left padding, KV-cache
    decoding and tensor sharding remain unsupported. Context parallelism keeps
    contiguous local queries and exchanges window KV and shared compressed KV.
    Every local sequence must contain complete compression groups.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        if backend.attn not in ("eager", "sdpa", "tilelang", "cudnn"):
            raise ValueError("DeepSeek V4.1 attention supports backend.attn='eager', 'sdpa', 'tilelang', or 'cudnn'")
        if backend.attn in ("tilelang", "cudnn") and config.attention_dropout:
            raise ValueError("The sparse attention kernels (tilelang, cudnn) require attention_dropout=0")
        if backend.linear != "torch" or backend.rms_norm not in ("torch_fp32", "te"):
            raise ValueError("DeepSeek V4.1 attention requires torch linear layers and torch_fp32 or te RMSNorm")
        self.backend = backend
        self.cp_group: dist.ProcessGroup | None = None
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_groups = config.o_groups
        self.window_size = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        self.reuse_selection_on_recompute = bool(config.reuse_sparse_selection_on_recompute)
        dtype = dtype_from_str(config.torch_dtype, torch.bfloat16)
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False, dtype=dtype)
        # TE defaults to CUDA when device is omitted, including under a meta
        # construction context. Use the projection's actual device explicitly.
        norm = (
            partial(initialize_rms_norm_module, "te", device=self.wq_a.weight.device)
            if backend.rms_norm == "te"
            else DeepseekV41RMSNorm
        )
        self.q_norm = norm(config.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.kv_norm = norm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // self.num_groups,
            self.num_groups * config.o_lora_rank,
            self.num_groups,
        ).to(dtype=dtype)
        self.wo_b = nn.Linear(self.num_groups * config.o_lora_rank, config.hidden_size, bias=False, dtype=dtype)
        self.sinks_param = DeepseekV4FP32Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.rotary_emb = _RotaryEmbedding(config, compressed=bool(self.compress_ratio))
        self.compressor = (
            _Compressor(config, ratio=self.compress_ratio, dtype=dtype, rms_norm=backend.rms_norm)
            if self.is_kv_source
            else None
        )
        self.indexer = (
            _Indexer(config, layer_idx=layer_idx, dtype=dtype, rms_norm=backend.rms_norm, attn_backend=backend.attn)
            if self.is_index_source
            else None
        )

    def setup_cp_attention(self, cp_mesh: DeviceMesh) -> None:
        """Configure the model-owned KV transport through the shared CP hook."""
        self.cp_group = cp_mesh.get_group()

    @property
    def attn_sink(self) -> nn.Parameter:
        """Return the checkpoint's per-head FP32 attention sink parameter."""
        return self.sinks_param.weight

    def reset_parameters(self, init_std: float = 0.02) -> None:
        """Initialize every attention parameter after construction or meta materialization.

        Args:
            init_std: Standard deviation of projection weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.kv_norm):
            nn.init.ones_(norm.weight)
        if self.compressor is not None:
            nn.init.ones_(self.compressor.norm.weight)
        if self.indexer is not None and self.indexer.owns_keys:
            nn.init.ones_(self.indexer.k_norm.weight)
        nn.init.zeros_(self.attn_sink)

    def _project_output(self, attended: torch.Tensor, angles: torch.Tensor, valid_tokens: torch.Tensor) -> torch.Tensor:
        """Undo RoPE on [batch, sequence, heads, head_dim] and project to hidden width.

        ``angles`` has shape [batch, sequence, rotary_pairs]; ``valid_tokens``
        is boolean [batch, sequence]. Return [batch, sequence, hidden] with
        padded queries zeroed, preserving the attention output dtype.
        """
        attended = _apply_rope(attended, angles, inverse=True)
        grouped = attended.reshape(*attended.shape[:2], self.num_groups, -1)
        output = self.wo_b(self.wo_a(grouped).flatten(2))
        return output.masked_fill(~valid_tokens.unsqueeze(-1), 0)

    def _plan_sparse_indices(
        self,
        next_state: DeepseekV41AttentionState,
        kv_sequence: int,
        *,
        positions: torch.Tensor,
        position_ids: torch.Tensor,
        valid_tokens: torch.Tensor,
        global_valid: torch.Tensor,
        global_seq_ids: torch.Tensor | None,
        packed_seq_ids: torch.Tensor | None,
        global_sequence: int,
        window_start: int,
        all_rows_nonempty: bool,
    ) -> SparseIndexPlan:
        """Build the sparse slot indices of one selection lineage and prepare them for the backend.

        Slots keep the released order: the sliding window first, then the selected compressed positions
        offset past the window keys. The result depends only on per-forward constants and on the current
        selection, so every layer that consumes the same selection shares it.

        Args:
            next_state: State after this layer's compressor and indexer; ``topk_indices`` is read on
                compressed layers.
            kv_sequence: Number of key positions per batch row of the K/V this plan is for.
            positions: Integer tensor of shape [sequence] with this rank's global query positions.
            position_ids: Integer tensor of shape [batch, sequence] or [1, sequence], document-local when packed.
            valid_tokens: Boolean tensor of shape [batch, sequence]; padded queries get no slots.
            global_valid: Boolean tensor of shape [batch, global_sequence] over the window keys.
            global_seq_ids: Optional document IDs [batch, global_sequence], zero for padding.
            packed_seq_ids: Optional document IDs [batch, sequence], zero for padding.
            global_sequence: Full physical token count per batch row.
            window_start: Global index of the first locally retained window key.
            all_rows_nonempty: Whether every query has at least one valid slot.

        Returns:
            The backend plan built from slot indices [batch, sequence, padded_slots] (window slots, then
            compressed slots offset by the retained window length, ``-1`` masked, padded to a multiple of 64).
        """
        batch = valid_tokens.shape[0]
        starts = (positions - self.window_size + 1).clamp_min(0).expand(batch, -1)
        if packed_seq_ids is not None:
            starts = torch.maximum(starts, positions.unsqueeze(0) - position_ids)
        slots = starts.unsqueeze(-1) + torch.arange(min(global_sequence, self.window_size), device=positions.device)
        visible = (slots <= positions.view(1, -1, 1)) & (slots < global_sequence)
        safe_slots = slots.clamp_max(global_sequence - 1)
        visible = visible & global_valid.gather(1, safe_slots.flatten(1)).view_as(slots)
        if packed_seq_ids is not None:
            key_seq_ids = global_seq_ids.gather(1, safe_slots.flatten(1)).view_as(slots)
            visible = visible & (key_seq_ids == packed_seq_ids.unsqueeze(-1))
        indices = torch.where(visible, slots - window_start, -1)
        if self.compress_ratio:
            selected = next_state.topk_indices
            indices = torch.cat(
                (indices, torch.where(selected >= 0, selected + kv_sequence - next_state.compressed_kv.shape[1], -1)),
                dim=-1,
            )
        indices = indices.masked_fill(~valid_tokens.unsqueeze(-1), -1)
        indices = F.pad(indices, (0, -indices.shape[-1] % 64), value=-1)
        return plan_dsv4_sparse_indices(
            indices, kv_sequence, backend=self.backend.attn, all_rows_nonempty=all_rows_nonempty
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        state: DeepseekV41AttentionState,
        attention_mask: torch.Tensor | None = None,
        cp_group: dist.ProcessGroup | None = None,
        packed_seq_ids: torch.Tensor | None = None,
        inputs_validated: bool = False,
    ) -> DeepseekV41AttentionOutput:
        """Apply attention and publish immutable state for the next layer.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            position_ids: Integer tensor of shape [batch, sequence] or [1, sequence],
                containing contiguous global positions for unpacked input and
                document-local positions for packed input.
            packed_seq_ids: Optional document IDs [batch, local_sequence], zero for padding.
                Document starts must be aligned to compression-group boundaries.
            state: Per-forward tensors documented in DeepseekV41AttentionState.
                Consumers must receive the state from their preceding layer.
            attention_mask: Optional binary right-padding mask of shape
                [batch, local_sequence], with one for tokens and zero for padding.
            cp_group: Optional CP group, overriding setup_cp_attention. Hidden
                states and positions are local shards; shared KV state is global.
            inputs_validated: True when the caller already ran validate_attention_inputs
                on these position_ids / attention_mask / packed_seq_ids (the model does, once per
                forward); the layer then skips its own host-synchronising checks.

        Returns:
            Output with hidden_states [batch, sequence, hidden] and the new state,
            whose tensors use the layouts in DeepseekV41AttentionState. Input
            tensors and the input state are never mutated.
        """
        batch, sequence, _ = hidden_states.shape
        if sequence == 0:
            raise ValueError("DeepSeek V4.1 attention requires a nonempty full sequence")
        cp_group = self.cp_group if cp_group is None else cp_group
        cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
        cp_rank = 0 if cp_group is None else dist.get_rank(cp_group)
        if cp_size > 1 and self.compress_ratio and sequence % self.compress_ratio:
            raise ValueError("DeepSeek V4.1 CP shards must contain complete compression groups")
        global_sequence = sequence * cp_size
        positions = torch.arange(sequence, device=hidden_states.device) + cp_rank * sequence
        if position_ids.shape not in ((1, sequence), (batch, sequence)):
            raise ValueError("DeepSeek V4.1 position_ids must have shape [batch, sequence] or [1, sequence]")
        if packed_seq_ids is not None and packed_seq_ids.shape != (batch, sequence):
            raise ValueError("packed_seq_ids must have shape [batch, local_sequence]")
        if attention_mask is not None and attention_mask.shape != (batch, sequence):
            raise ValueError("DeepSeek V4.1 attention_mask must have shape [batch, sequence]")
        if not inputs_validated:
            # Host-synchronising checks; the model runs them once per forward for all layers.
            validate_attention_inputs(
                position_ids, attention_mask, packed_seq_ids, sequence=sequence, cp_group=cp_group
            )
        valid_tokens = (
            torch.ones(batch, sequence, dtype=torch.bool, device=hidden_states.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        global_valid = state.global_valid
        if global_valid is None:
            global_valid = gather_sequence(valid_tokens, cp_group)
        global_seq_ids = state.global_seq_ids
        if global_seq_ids is None and packed_seq_ids is not None:
            global_seq_ids = gather_sequence(packed_seq_ids, cp_group)
        angles = self.rotary_emb(position_ids)
        query_latent = self.q_norm(self.wq_a(hidden_states))
        query = _apply_rope(self.wq_b(query_latent).unflatten(-1, (self.num_heads, self.head_dim)), angles)
        kv = quantize_cache(_apply_rope(self.kv_norm(self.wkv(hidden_states)), angles), format="fp8", block_size=32)
        kv, window_start = gather_window(kv, cp_group, window_size=self.window_size)
        window_sequence = kv.shape[1]
        replay = csa2_selection_replay.current() if self.reuse_selection_on_recompute else None
        record: _SelectionRecord | None = None
        produces = False  # this layer adds a selection or a sparse-index plan to the state
        next_state = (
            state
            if state.global_valid is not None
            else replace(state, global_valid=global_valid, global_seq_ids=global_seq_ids)
        )
        if self.compress_ratio:
            width = global_sequence // self.compress_ratio
            local_width = sequence // self.compress_ratio
            compressed_angles = self.rotary_emb(
                position_ids[:, : local_width * self.compress_ratio : self.compress_ratio]
            )
            latent = None
            if self.compressor is not None:
                latent = self.compressor(hidden_states)
                compressed_valid = (
                    valid_tokens[:, : local_width * self.compress_ratio]
                    .unflatten(1, (local_width, self.compress_ratio))
                    .all(dim=-1)
                )
                next_state = DeepseekV41AttentionState(
                    global_valid=global_valid,
                    global_seq_ids=global_seq_ids,
                    compressed_kv=gather_sequence(
                        quantize_cache(_apply_rope(latent, compressed_angles), format="nvfp4", block_size=16), cp_group
                    ),
                    compressed_valid=gather_sequence(compressed_valid, cp_group),
                    compression_ratio=self.compress_ratio,
                    compressed_seq_ids=(
                        None
                        if global_seq_ids is None
                        else global_seq_ids[:, : width * self.compress_ratio : self.compress_ratio]
                    ),
                )
            elif next_state.compressed_kv is None or next_state.compression_ratio != self.compress_ratio:
                raise ValueError("A CSA2 consumer requires a preceding Full layer with the same compression ratio")
            if self.indexer is not None:
                produces = True
                if replay is not None and replay[1] == "replay":
                    record = replay[0].take()
                if record is not None:
                    # The checkpoint forward's selection; the compressed KV stays the recomputed tensor so
                    # the compressor's gradient path belongs to the recompute.
                    next_state = replace(
                        next_state,
                        index_keys=record.index_keys,
                        topk_indices=record.topk_indices,
                        candidates=record.candidates,
                        sparse_plan=None,
                    )
                else:
                    with csa2_selection_region():
                        next_state = self.indexer(
                            hidden_states,
                            query_latent=query_latent,
                            latent=latent,
                            angles=angles,
                            compressed_angles=compressed_angles,
                            state=next_state,
                            position_ids=positions.unsqueeze(0).expand(batch, -1),
                            cp_group=cp_group,
                            packed_seq_ids=packed_seq_ids,
                        )
            if next_state.topk_indices is None or next_state.compressed_kv is None:
                raise ValueError("A Reuse CSA2 layer requires compressed KV and indices from its source")
            if next_state.compressed_kv.shape[:2] != (batch, width) or next_state.topk_indices.shape[:2] != (
                batch,
                sequence,
            ):
                raise ValueError("CSA2 state belongs to a different batch or sequence")
            kv = torch.cat((kv, next_state.compressed_kv), dim=1)
        if self.backend.attn in ("tilelang", "cudnn"):
            # Preserve the released sparse slot order and reuse V4's trainable
            # online-softmax kernel (TileLang), or the cuDNN Frontend DSA kernels
            # with the same indices, sink and scale (backend.attn="cudnn"). The slot
            # indices are identical for every layer that consumes one selection, so
            # the first consumer plans them once and the state carries the plan.
            plan = next_state.sparse_plan
            if plan is not None and (plan.backend != self.backend.attn or plan.kv_sequence != kv.shape[1]):
                plan = None  # another lineage's plan (a window-only layer after a compressed one)
            if plan is None:
                if not produces:
                    produces = True
                    if replay is not None and replay[1] == "replay":
                        record = replay[0].take()
                if record is not None and record.sparse_plan is not None:
                    plan = record.sparse_plan
                else:
                    with csa2_selection_region(), torch.no_grad():
                        plan = self._plan_sparse_indices(
                            next_state,
                            kv.shape[1],
                            positions=positions,
                            position_ids=position_ids,
                            valid_tokens=valid_tokens,
                            global_valid=global_valid,
                            global_seq_ids=global_seq_ids,
                            packed_seq_ids=packed_seq_ids,
                            global_sequence=global_sequence,
                            window_start=window_start,
                            # Without a padding mask every query sees at least its own slot; the cuDNN backward
                            # then skips its empty-row scan (a torch.nonzero host synchronisation).
                            all_rows_nonempty=attention_mask is None,
                        )
                next_state = replace(next_state, sparse_plan=plan)
            attended = dsv4_sparse_attention(
                query,
                kv,
                self.sinks_param(query),
                None,
                self.head_dim**-0.5,
                backend=self.backend.attn,
                reference_rounding=True,
                all_rows_nonempty=attention_mask is None,
                plan=plan,
            )
        else:
            # Dense masks are needed only by the eager/SDPA fallback; TileLang uses sparse indices.
            key_positions = torch.arange(window_sequence, device=positions.device) + window_start
            local_allowed = (key_positions.unsqueeze(0) <= positions.unsqueeze(1)) & (
                key_positions.unsqueeze(0) > positions.unsqueeze(1) - self.window_size
            )
            window_valid = global_valid[:, window_start : window_start + window_sequence]
            allowed = local_allowed.unsqueeze(0) & window_valid.unsqueeze(1)
            if packed_seq_ids is not None:
                allowed = allowed & (
                    packed_seq_ids.unsqueeze(-1)
                    == global_seq_ids[:, window_start : window_start + window_sequence].unsqueeze(1)
                )
            if self.compress_ratio:
                # A separate sentinel column makes -1 masking safe when valid index 0
                # also occurs in the row; boolean scatter must never overwrite it.
                compressed_allowed = torch.zeros(
                    batch, sequence, width + 1, dtype=torch.bool, device=hidden_states.device
                )
                indices = next_state.topk_indices
                compressed_allowed = compressed_allowed.scatter(-1, torch.where(indices >= 0, indices, width), True)
                allowed = torch.cat((allowed, compressed_allowed[..., :width]), dim=-1)
            # A zero-valued extra key contributes exp(attn_sink) only to the softmax
            # denominator. It keeps fully padded query rows numerically well-defined.
            kv = torch.cat((kv, kv.new_zeros(batch, 1, self.head_dim)), dim=1)
            bias = torch.zeros(batch, 1, sequence, allowed.shape[-1], device=hidden_states.device, dtype=torch.float32)
            bias = bias.masked_fill(~allowed.unsqueeze(1), -torch.inf).expand(-1, self.num_heads, -1, -1)
            sink = self.sinks_param(query).view(1, self.num_heads, 1, 1).expand(batch, -1, sequence, -1)
            bias = torch.cat((bias, sink), dim=-1)
            if self.backend.attn == "sdpa":
                attended = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    kv.unsqueeze(1),
                    kv.unsqueeze(1),
                    attn_mask=bias,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    scale=self.head_dim**-0.5,
                ).transpose(1, 2)
            else:
                with torch.autocast(device_type=query.device.type, enabled=False):
                    logits = torch.einsum("bshd,btd->bhst", query.float(), kv.float()) * self.head_dim**-0.5
                    probabilities = (logits + bias).softmax(dim=-1)
                    probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
                    attended = torch.einsum("bhst,btd->bshd", probabilities, kv.float()).to(query.dtype)
        if produces and replay is not None and replay[1] == "record":
            replay[0].record(
                _SelectionRecord(
                    next_state.index_keys, next_state.topk_indices, next_state.candidates, next_state.sparse_plan
                )
            )
        return DeepseekV41AttentionOutput(self._project_output(attended, angles, valid_tokens), next_state)

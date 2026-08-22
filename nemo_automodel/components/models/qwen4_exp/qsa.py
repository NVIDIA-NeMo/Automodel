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

"""Qwen4-Exp QSA routing, fused sparse GQA, and its PyTorch oracle.

The supplied Qwen4-Exp reference is an inference implementation: its indexer
returns integer top-k IDs and defines neither an auxiliary indexer loss nor a
straight-through gradient.  This module therefore freezes the indexer weights
explicitly.  Routing is still recomputed from the current hidden states on
every forward, while gradients flow through the main attention Q/K/V path.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm
from nemo_automodel.components.models.qwen4_exp.cp import Qwen4ExpCPContext, qwen4_cp_all_gather
from nemo_automodel.components.models.qwen4_exp.kernels._tilelang import HAS_TILELANG
from nemo_automodel.components.models.qwen4_exp.kernels.sparse_attention import tilelang_sparse_gqa_attention
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

# The gathered implementation is a numerical oracle and CPU fallback, not the
# production CUDA path. Keep its workspace bounded without exposing a model or
# recipe knob that could be mistaken for part of the QSA architecture.
_PYTORCH_ORACLE_QUERY_CHUNK_SIZE = 16


def right_padded_sequence_lengths(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate a non-packed right-tail mask and return logical lengths.

    Args:
        attention_mask: ``None`` or a binary tensor ``[B, S]``.  Each row must
            be exactly ``1 ** L + 0 ** (S - L)``; left/interior padding and
            packed document-ID masks are rejected.
        batch_size: Expected batch dimension ``B``.
        sequence_length: Expected physical sequence dimension ``S``.
        device: Device on which to return the lengths.

    Returns:
        Logical sequence lengths as int64 ``[B]``.
    """
    if attention_mask is None:
        return torch.full((batch_size,), sequence_length, dtype=torch.long, device=device)
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch_size, sequence_length):
        raise NotImplementedError(
            "Qwen4-Exp QSA currently requires a non-packed [batch, sequence] right-tail attention mask; "
            f"got {tuple(attention_mask.shape)}"
        )
    mask = attention_mask.to(device=device)
    if mask.dtype != torch.bool:
        binary = (mask == 0) | (mask == 1)
        if not bool(binary.all()):
            raise ValueError("Qwen4-Exp QSA attention_mask must contain only binary 0/1 values")
    valid = mask.bool()
    lengths = valid.sum(dim=-1, dtype=torch.long)
    expected = torch.arange(sequence_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    if not bool(torch.equal(valid, expected)):
        raise NotImplementedError(
            "Qwen4-Exp QSA currently supports only right-tail padding (1...10...0); "
            "left padding, interior padding, and packed masks are not supported"
        )
    return lengths


def apply_qsa_rope(states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply the Qwen4 attention RoPE to indexer states.

    Args:
        states: Index query or compressed-key states ``[B, N, H, D_index]``.
        freqs_cis: Model-composed rotary values ``[B, N, D_rope]`` laid out as
            ``cat(cos[..., D_rope/2], sin[..., D_rope/2])``.  ``D_rope`` may be
            smaller than ``D_index``; the remaining index dimensions pass
            through unchanged.

    Returns:
        Rotated states in the same ``[B, N, H, D_index]`` layout and dtype.
    """
    if states.ndim != 4:
        raise ValueError(f"QSA RoPE states must be [B, N, H, D], got {tuple(states.shape)}")
    if freqs_cis.ndim != 3 or freqs_cis.shape[:2] != states.shape[:2]:
        raise ValueError(
            "QSA RoPE frequencies must be [B, N, D_rope] and match the state token axes; "
            f"got states={tuple(states.shape)}, freqs={tuple(freqs_cis.shape)}"
        )
    rotary_width = freqs_cis.shape[-1]
    if rotary_width <= 0 or rotary_width % 2 != 0 or rotary_width > states.shape[-1]:
        raise ValueError(
            "QSA rotary width must be positive, even, and no wider than the index head; "
            f"got rotary_width={rotary_width}, index_head_dim={states.shape[-1]}"
        )
    cos, sin = freqs_cis.split(rotary_width // 2, dim=-1)
    return apply_rotary_emb(states, cos, sin)


@torch.no_grad()
def select_qsa_token_ids(
    index_queries: torch.Tensor,
    compressed_keys: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
    query_chunk_size: int = 128,
    query_position_offset: int = 0,
    global_sequence_length: int | None = None,
) -> torch.Tensor:
    """Score compressed blocks and expand gold QSA top-k IDs.

    For query position ``t``, only ``floor((t + 1) / compress_ratio)`` complete
    blocks are visible.  Each block score is
    ``sum_h(relu(dot(q[t,h], k[block,0]))) / sqrt(D)``.  The best
    ``token_budget / compress_ratio`` blocks are expanded to token IDs, then
    the 0--``compress_ratio - 1`` tokens in the current incomplete causal tail
    are appended.  Invalid slots are ``-1``.

    Args:
        index_queries: Normalized and rotated local index queries
            ``[B, S_query, H_index, D_index]``.
        compressed_keys: FP32-mean-pooled, normalized, rotated index keys
            ``[B, floor(S_global / compress_ratio), 1, D_index]``.
        sequence_lengths: Right-padded logical lengths ``[B]``.
        token_budget: Maximum number of tokens contributed by complete blocks.
        compress_ratio: Number of consecutive tokens represented by one block.
        query_chunk_size: Query rows scored together.  This bounds the temporary
            FP32 score tensor without changing top-k semantics.
        query_position_offset: Global position represented by local query row
            zero. It is zero without CP and ``cp_rank * S_query`` for the
            contiguous CP layout.
        global_sequence_length: Padded global physical sequence length. It
            defaults to the local query length for the non-CP path.

    Returns:
        Global logical token IDs ``[B, S_query, token_budget +
        compress_ratio - 1]`` in int32. Valid IDs are contiguous at the start
        of each row.
    """
    if index_queries.ndim != 4:
        raise ValueError(f"index_queries must be [B, S, H, D], got {tuple(index_queries.shape)}")
    if compressed_keys.ndim != 4 or compressed_keys.shape[2] != 1:
        raise ValueError(f"compressed_keys must be [B, P, 1, D], got {tuple(compressed_keys.shape)}")
    batch_size, query_sequence_length, num_heads, head_dim = index_queries.shape
    if compressed_keys.shape[0] != batch_size or compressed_keys.shape[-1] != head_dim:
        raise ValueError("QSA query/key batch and head dimensions must match")
    if sequence_lengths.shape != (batch_size,):
        raise ValueError(f"sequence_lengths must be [{batch_size}], got {tuple(sequence_lengths.shape)}")
    if token_budget <= 0 or compress_ratio <= 1 or token_budget % compress_ratio != 0:
        raise ValueError(
            "QSA requires a positive token_budget divisible by compress_ratio > 1; "
            f"got token_budget={token_budget}, compress_ratio={compress_ratio}"
        )
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError("QSA index queries require positive head count and head dimension")
    if query_position_offset < 0:
        raise ValueError(f"query_position_offset must be non-negative, got {query_position_offset}")
    if global_sequence_length is None:
        global_sequence_length = query_sequence_length
    if global_sequence_length < query_position_offset + query_sequence_length:
        raise ValueError(
            "QSA global sequence does not cover every local query; "
            f"got global={global_sequence_length}, offset={query_position_offset}, local={query_sequence_length}"
        )

    lengths = sequence_lengths.to(device=index_queries.device, dtype=torch.long)
    if bool(((lengths < 0) | (lengths > global_sequence_length)).any()):
        raise ValueError(f"QSA logical lengths must lie in [0, {global_sequence_length}]")
    required_blocks = torch.div(lengths, compress_ratio, rounding_mode="floor")
    num_blocks = compressed_keys.shape[1]
    if bool((required_blocks > num_blocks).any()):
        raise ValueError(
            "compressed_keys do not contain every complete logical block; "
            f"need at least {int(required_blocks.max())}, got {num_blocks}"
        )

    block_budget = token_budget // compress_ratio
    final_width = token_budget + compress_ratio - 1
    selected_tokens = torch.full(
        (batch_size, query_sequence_length, final_width),
        -1,
        dtype=torch.int32,
        device=index_queries.device,
    )
    block_offsets = torch.arange(compress_ratio, device=index_queries.device, dtype=torch.long)
    tail_offsets = torch.arange(compress_ratio - 1, device=index_queries.device, dtype=torch.long)
    score_scale = math.sqrt(head_dim)

    for batch_idx in range(batch_size):
        logical_length = int(lengths[batch_idx])
        available_blocks = logical_length // compress_ratio
        keys = compressed_keys[batch_idx, :available_blocks, 0].float()
        local_valid_length = min(max(logical_length - query_position_offset, 0), query_sequence_length)
        for query_start in range(0, local_valid_length, query_chunk_size):
            query_end = min(query_start + query_chunk_size, local_valid_length)
            query_positions = torch.arange(
                query_position_offset + query_start,
                query_position_offset + query_end,
                device=index_queries.device,
            )
            visible_blocks = torch.div(query_positions + 1, compress_ratio, rounding_mode="floor")
            rows = query_end - query_start
            result = torch.full((rows, final_width), -1, dtype=torch.int32, device=index_queries.device)

            topk_width = min(block_budget, available_blocks)
            if topk_width:
                # Gold fast_topk preserves causal block order while all visible
                # blocks fit the budget.  It starts score-ordered top-k only on
                # the first genuinely sparse row (t=2051 for c4/budget=2048).
                candidate_blocks = torch.arange(topk_width, device=index_queries.device)
                top_blocks = candidate_blocks.unsqueeze(0).expand(rows, -1).clone()
                valid_blocks = candidate_blocks.unsqueeze(0) < visible_blocks.unsqueeze(1)
                sparse_rows = visible_blocks > block_budget
                if bool(sparse_rows.any()):
                    sparse_queries = index_queries[batch_idx, query_start:query_end][sparse_rows].float()
                    scores = torch.einsum("qhd,pd->qhp", sparse_queries, keys)
                    scores = torch.relu(scores).sum(dim=1) / score_scale
                    block_ids = torch.arange(available_blocks, device=index_queries.device)
                    sparse_visible = visible_blocks[sparse_rows]
                    scores = scores.masked_fill(block_ids.unsqueeze(0) >= sparse_visible.unsqueeze(1), -torch.inf)
                    top_blocks[sparse_rows] = torch.topk(scores, k=block_budget, dim=-1).indices
                    valid_blocks[sparse_rows] = True
                expanded = top_blocks.unsqueeze(-1) * compress_ratio + block_offsets
                expanded = torch.where(valid_blocks.unsqueeze(-1), expanded, -torch.ones_like(expanded))
                result[:, : topk_width * compress_ratio] = expanded.reshape(rows, -1).to(torch.int32)

            tail_start = visible_blocks * compress_ratio
            tail_count = query_positions + 1 - tail_start
            valid_block_count = torch.minimum(visible_blocks, torch.full_like(visible_blocks, block_budget))
            tail_values = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
            tail_valid = tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)
            if bool(tail_valid.any()):
                destination = valid_block_count.unsqueeze(1) * compress_ratio + tail_offsets.unsqueeze(0)
                row_ids = torch.arange(rows, device=index_queries.device).unsqueeze(1).expand_as(tail_valid)
                result[row_ids[tail_valid], destination[tail_valid]] = tail_values[tail_valid].to(torch.int32)

            selected_tokens[batch_idx, query_start:query_end] = result

    return selected_tokens


def _gathered_qsa_gqa_attention_chunk(
    grouped_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Evaluate one bounded-workspace chunk of the PyTorch QSA oracle.

    ``grouped_query`` uses ``[B, Q, Hkv, G, D]``; K/V remain
    ``[B, S_global, Hkv, D]`` and IDs use ``[B, Q, K]``.

    Args:
        grouped_query: Local grouped queries ``[B, Q, Hkv, G, D]``.
        key: Global keys ``[B, S_global, Hkv, D]``.
        value: Global values ``[B, S_global, Hkv, D]``.
        selected_token_ids: Global logical IDs ``[B, Q, K]``; ``-1`` marks
            invalid fixed-width slots.
        softmax_scale: Score multiplier.

    Returns:
        Chunk output ``[B, Q, Hq, D]`` in the query dtype.
    """
    batch_size = grouped_query.shape[0]
    token_ids = selected_token_ids.long()
    valid = token_ids >= 0
    safe_ids = token_ids.clamp_min(0)
    batch_ids = torch.arange(batch_size, device=grouped_query.device).view(batch_size, 1, 1)

    # Advanced indexing keeps the physical KV-head axis intact:
    # [B, Q, K, Hkv, D] -> [B, Q, Hkv, K, D].
    gathered_key = key[batch_ids, safe_ids].permute(0, 1, 3, 2, 4)
    gathered_value = value[batch_ids, safe_ids].permute(0, 1, 3, 2, 4)
    scores = torch.einsum(
        "bqhgd,bqhkd->bqhgk",
        grouped_query.float(),
        gathered_key.float(),
    )
    scores = scores * softmax_scale
    score_valid = valid[:, :, None, None, :]
    scores = scores.masked_fill(~score_valid, -torch.inf)

    # Padding queries have no IDs. Give softmax a finite row, then erase every
    # probability so both output and Q/K/V gradient are exactly zero.
    has_tokens = valid.any(dim=-1)
    scores = torch.where(has_tokens[:, :, None, None, None], scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1).masked_fill(~score_valid, 0.0)
    chunk_output = torch.einsum(
        "bqhgk,bqhkd->bqhgd",
        probabilities,
        gathered_value.float(),
    )
    return chunk_output.flatten(2, 3).to(grouped_query.dtype)


def gathered_qsa_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the differentiable PyTorch QSA oracle without expanding K/V heads.

    This implementation is retained for CPU execution and numerical parity.
    CUDA training with ``backend.attn='tilelang'`` dispatches to the fused
    kernel instead. The oracle uses a private fixed query chunk solely to bound
    temporary gathered K/V storage; it has no public model configuration.

    Args:
        query: Main normalized/rotated local queries ``[B, S_query, Hq, D]``.
        key: Main normalized/rotated global keys ``[B, S_global, Hkv, D]``.
        value: Main global values ``[B, S_global, Hkv, D]``.
        selected_token_ids: Indexer output ``[B, S_query, K]``. IDs are global
            logical positions in the same batch row; ``-1`` marks fixed-width
            padding.
        softmax_scale: Score multiplier, defaulting to ``1 / sqrt(D)``.

    Returns:
        Sparse attention output ``[B, S_query, Hq, D]``. Rows with no selected
        IDs (right-tail padding queries) are exactly zero.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("QSA query, key, and value must use [B, S, H, D] layout")
    if key.shape != value.shape:
        raise ValueError(f"QSA key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")
    batch_size, query_sequence_length, num_query_heads, head_dim = query.shape
    kv_sequence_length = key.shape[1]
    if key.shape[0] != batch_size or key.shape[-1] != head_dim:
        raise ValueError("QSA main Q/K/V batch and head dimensions must match")
    num_kv_heads = key.shape[2]
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError(f"QSA requires Hq divisible by Hkv, got Hq={num_query_heads}, Hkv={num_kv_heads}")
    if selected_token_ids.ndim != 3 or selected_token_ids.shape[:2] != (batch_size, query_sequence_length):
        raise ValueError(
            f"selected_token_ids must be [B, S, K] matching the queries; got {tuple(selected_token_ids.shape)}"
        )
    if query_sequence_length == 0:
        return torch.empty_like(query)
    if bool(((selected_token_ids < -1) | (selected_token_ids >= kv_sequence_length)).any()):
        raise ValueError("QSA selected token IDs must be -1 or a valid logical sequence position")

    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    query_groups = num_query_heads // num_kv_heads
    grouped_query = query.unflatten(2, (num_kv_heads, query_groups))
    outputs: list[torch.Tensor] = []

    for query_start in range(0, query_sequence_length, _PYTORCH_ORACLE_QUERY_CHUNK_SIZE):
        query_end = min(query_start + _PYTORCH_ORACLE_QUERY_CHUNK_SIZE, query_sequence_length)
        chunk_query = grouped_query[:, query_start:query_end]
        chunk_token_ids = selected_token_ids[:, query_start:query_end]
        chunk_output = _gathered_qsa_gqa_attention_chunk(
            chunk_query,
            key,
            value,
            chunk_token_ids,
            softmax_scale=scale,
        )
        outputs.append(chunk_output)

    return torch.cat(outputs, dim=1)


def qsa_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    backend: str,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Dispatch QSA to fused TileLang on CUDA or the PyTorch oracle elsewhere.

    CPU execution always uses the oracle so model construction, checkpoint
    inspection, and distributed CPU parity tests do not require TileLang.
    CUDA execution is strict: Qwen4-Exp sparse GQA requires TileLang, and
    missing dependencies or unsupported backends/dtypes are reported rather
    than silently falling back to the much larger gathered implementation.
    """
    if not query.is_cuda:
        return gathered_qsa_gqa_attention(
            query,
            key,
            value,
            selected_token_ids,
            softmax_scale=softmax_scale,
        )
    if backend != "tilelang":
        raise RuntimeError(
            "Qwen4-Exp CUDA QSA requires backend.attn='tilelang'; "
            "call gathered_qsa_gqa_attention directly for a numerical oracle"
        )
    if not HAS_TILELANG:
        raise RuntimeError(
            "Qwen4-Exp TileLang QSA was requested on CUDA, but tilelang is not installed. "
            "Install tilelang; the PyTorch oracle is available only through "
            "gathered_qsa_gqa_attention for explicit numerical tests."
        )
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise RuntimeError("Qwen4-Exp TileLang QSA requires CUDA BF16 query, key, and value tensors")
    return tilelang_sparse_gqa_attention(
        query,
        key,
        value,
        selected_token_ids,
        softmax_scale=softmax_scale,
    )


class Qwen4ExpQSAIndexer(nn.Module):
    """Frozen, hookable Qwen4-Exp compressed-block indexer.

    ``forward`` has no cache or mutable routing state and returns the complete
    logical-ID tensor ``[B, S, indexer_budget + compress_ratio - 1]``.  A normal
    PyTorch forward hook can therefore capture the exact routing artifact used
    by the subsequent sparse attention.

    The fused projection produces raw query/key layout
    ``[B, S, (H_index + 1) * D_index]``.  Queries become
    ``[B, S, H_index, D_index]``.  Raw keys become ``[B, S, 1, D_index]`` and
    complete consecutive groups are averaged in FP32 into
    ``[B, floor(S / c), 1, D_index]`` before K RMSNorm and group-start RoPE.
    """

    def __init__(self, config: object, backend: BackendConfig) -> None:
        super().__init__()
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.num_query_heads = int(getattr(config, "indexer_n_heads"))
        self.num_key_heads = int(getattr(config, "indexer_kv_heads"))
        self.head_dim = int(getattr(config, "indexer_head_dim"))
        self.token_budget = int(getattr(config, "indexer_budget"))
        self.compress_ratio = int(getattr(config, "indexer_compress_ratio"))
        self.query_chunk_size = int(getattr(config, "qsa_indexer_query_chunk_size", 128))
        if self.num_query_heads <= 0 or self.head_dim <= 0:
            raise ValueError("QSA index head count and dimension must be positive")
        if self.num_key_heads != 1:
            raise ValueError(f"Qwen4-Exp QSA requires one index KV head, got {self.num_key_heads}")
        if self.compress_ratio <= 1 or self.token_budget <= 0 or self.token_budget % self.compress_ratio != 0:
            raise ValueError("QSA indexer_budget must be positive and divisible by indexer_compress_ratio > 1")
        if self.query_chunk_size <= 0:
            raise ValueError("qsa_indexer_query_chunk_size must be positive")

        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.index_qk_proj = initialize_linear_module(
            backend.linear,
            self.hidden_size,
            (self.num_query_heads + self.num_key_heads) * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        eps = float(getattr(config, "rms_norm_eps"))
        self.q_layernorm = Qwen3NextRMSNorm(self.head_dim, eps=eps)
        self.k_layernorm = Qwen3NextRMSNorm(self.head_dim, eps=eps)

        # The supplied gold path emits discrete top-k IDs and contains no
        # indexer auxiliary loss or STE.  Make that training contract explicit
        # instead of silently leaving parameters trainable with grad=None.
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cp_context: Qwen4ExpCPContext | None = None,
    ) -> torch.Tensor:
        """Return selected logical token IDs for every physical query row.

        Args:
            hidden_states: Decoder block input ``[B, S, hidden_size]``.
            freqs_cis: Composed Qwen4 rotary values ``[B, S, D_rope]`` stored
                as concatenated cosine/sine halves.
            attention_mask: Optional binary right-tail mask ``[B, S]``.
            cp_context: Optional contiguous CP metadata. When present,
                ``hidden_states`` and ``freqs_cis`` contain the local query
                shard while raw logical lengths and selected IDs use global
                sequence coordinates.

        Returns:
            int32 logical IDs ``[B, S, token_budget + compress_ratio - 1]``.
            Padding-query rows contain only ``-1``.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"QSA hidden_states must be [B, S, {self.hidden_size}], got {tuple(hidden_states.shape)}")
        batch_size, sequence_length, _ = hidden_states.shape
        if cp_context is None:
            sequence_lengths = right_padded_sequence_lengths(
                attention_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=hidden_states.device,
            )
            query_position_offset = 0
            global_sequence_length = sequence_length
        else:
            if cp_context.global_input_ids.shape[0] != batch_size:
                raise ValueError(
                    "QSA CP batch size disagrees with the global raw-ID context; "
                    f"got local={batch_size}, global={cp_context.global_input_ids.shape[0]}"
                )
            if sequence_length != cp_context.local_sequence_length:
                raise ValueError(
                    "QSA local sequence length disagrees with its CP context; "
                    f"got local={sequence_length}, context={cp_context.local_sequence_length}"
                )
            sequence_lengths = cp_context.global_sequence_lengths.to(hidden_states.device)
            query_position_offset = cp_context.local_sequence_start
            global_sequence_length = cp_context.global_sequence_length
            if sequence_length % self.compress_ratio != 0:
                raise ValueError(
                    "QSA contiguous CP requires every local sequence shard to be divisible by the compression ratio; "
                    f"got local={sequence_length}, ratio={self.compress_ratio}"
                )

        projected = self.index_qk_proj(hidden_states)
        query_width = self.num_query_heads * self.head_dim
        raw_query = projected[..., :query_width].unflatten(-1, (self.num_query_heads, self.head_dim))
        raw_key = projected[..., query_width:].unflatten(-1, (self.num_key_heads, self.head_dim))
        index_query = apply_qsa_rope(self.q_layernorm(raw_query), freqs_cis)

        num_blocks = sequence_length // self.compress_ratio
        grouped_key = raw_key[:, : num_blocks * self.compress_ratio].unflatten(1, (num_blocks, self.compress_ratio))
        compressed_key = grouped_key.float().mean(dim=2).to(raw_key.dtype)
        compressed_freqs = freqs_cis[:, : num_blocks * self.compress_ratio : self.compress_ratio]
        compressed_key = apply_qsa_rope(self.k_layernorm(compressed_key), compressed_freqs)
        if cp_context is not None:
            compressed_key = qwen4_cp_all_gather(
                compressed_key,
                cp_context,
                sequence_dim=1,
                differentiable=False,
            )

        return select_qsa_token_ids(
            index_query,
            compressed_key,
            sequence_lengths,
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            query_chunk_size=self.query_chunk_size,
            query_position_offset=query_position_offset,
            global_sequence_length=global_sequence_length,
        )

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize frozen indexer parameters for scratch-model construction."""
        nn.init.trunc_normal_(self.index_qk_proj.weight, mean=0.0, std=init_std)
        self.q_layernorm.reset_parameters()
        self.k_layernorm.reset_parameters()


__all__ = [
    "Qwen4ExpQSAIndexer",
    "apply_qsa_rope",
    "gathered_qsa_gqa_attention",
    "qsa_gqa_attention",
    "right_padded_sequence_lengths",
    "select_qsa_token_ids",
]

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

"""Trainable Step-5 decoder layers and context-parallel attention."""

import contextlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch.utils._python_dispatch import _disable_current_modes

from nemo_automodel.components.attention.flex_attention import FlexAttention
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.cp import gather_halo, gather_metadata, gather_sequence
from nemo_automodel.components.models.step5.fa4 import fa4_attention
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding
from nemo_automodel.components.models.step5.sparse import (
    RegionMetadata,
    RegionSelection,
    Step5SparseIndexer,
    pack_region_bitmap,
    region_metadata,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.recompute_replay import RecomputeReplay


def _rms_norm_with_rstd(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize in FP32 and retain the row statistic needed by backward."""
    raw = hidden.float()
    rstd = torch.rsqrt(raw.square().mean(-1, keepdim=True) + eps)
    normalized = raw * rstd
    return (normalized * (weight.float() + 1)).to(output_dtype), rstd


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype) -> torch.Tensor:
    """Normalize [..., width] in FP32 before rounding once to the output dtype."""
    return _rms_norm_with_rstd(hidden, weight, eps, output_dtype)[0]


# Residual and activation inputs have different precision and layout families.
# Give them separate Dynamo entries: mixing FP32/BF16, grad/no_grad, output
# casts, and strided QKV in one entry exhausts the default specialization cache.
@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True},
)
def _residual_rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype
) -> torch.Tensor:
    return _rms_norm(hidden, weight, eps, output_dtype)


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True},
)
def _activation_rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the same reduction outputs in grad/no_grad. Otherwise Inductor can
    # pick a different summation tree when the backward statistic is dead,
    # changing BF16 rounding at the final norm during checkpoint replay.
    return _rms_norm_with_rstd(hidden, weight, eps, output_dtype)


@torch.compile(
    fullgraph=True,
    dynamic=True,
    options={"emulate_precision_casts": True},
)
def _grouped_rms_norm(
    hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize grouped query rows without copying interleaved QKV gaps.

    Args:
        hidden: Query tensor [token_groups, heads_per_group, channels], where
            token_groups flattens batch, sequence and KV group. The outer stride
            includes the interleaved K/V rows, and channels has unit stride.
        weight: FP32 additive scale [channels].
        eps: Additive variance epsilon.
        output_dtype: Final normalized activation dtype.

    Returns:
        Normalized query [token_groups, heads_per_group, channels] and FP32
        inverse RMS [token_groups, heads_per_group, 1]. Neither aliases input.
    """
    return _rms_norm_with_rstd(hidden, weight, eps, output_dtype)


class Step5RMSNorm(Step3p5RMSNorm):
    """Fuse the main-stream norm while retaining FP32 additive scale parameters.

    Indexer normalization deliberately uses its separate implementation: its
    checkpoint requires rounding the scale to the activation dtype first.

    Args:
        hidden_size: Normalized feature width.
        eps: Additive variance epsilon.
        compile_norm: Enable the compiled CUDA core. CPU always uses eager math.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5, *, compile_norm: bool = False) -> None:
        super().__init__(hidden_size, eps)
        self.compile_norm = compile_norm

    def forward(self, hidden: torch.Tensor, *, output_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Normalize [..., hidden_size] before the optional projection-input cast.

        Args:
            hidden: Activations [..., hidden_size] with arbitrary leading axes.
                Grouped queries use [batch, sequence, groups, heads_per_group,
                hidden_size] and may be a strided slice of interleaved QKV.
                Decoder residual inputs remain FP32 through normalization.
            output_dtype: Final output dtype; defaults to the input dtype.

        Returns:
            Normalized activations with the input shape. The learned FP32 scale
            and every intermediate retain their original precision.
        """
        dtype = hidden.dtype if output_dtype is None else output_dtype
        if self.compile_norm and hidden.is_cuda:
            # Under no_grad, AOTAutograd compiles a separate inference graph whose
            # reductions may round differently. Reuse the training graph so grad,
            # no_grad and checkpoint replay produce identical outputs.
            training_graph = (
                not torch.is_grad_enabled() and not torch.is_inference_mode_enabled() and self.weight.requires_grad
            )
            with torch.enable_grad() if training_graph else contextlib.nullcontext():
                output = self._compiled_forward(hidden, dtype)
            return output.detach() if training_graph else output
        return _rms_norm(hidden, self.weight, self.variance_epsilon, dtype)

    def _compiled_forward(self, hidden: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Run the compiled CUDA norm on [..., hidden_size] and restore the input shape."""
        if hidden.ndim == 5:
            # Preserve each contiguous run of query heads. Flattening all
            # rows would copy across the interleaved K/V gap in each group.
            rows = hidden.reshape(-1, hidden.shape[-2], hidden.shape[-1])
            output, _ = _grouped_rms_norm(rows, self.weight, self.variance_epsilon, dtype)
            return output.reshape(hidden.shape)
        # Normalize the leading layout so decoder, Q/K, and final norms
        # share a compiled rank. Interleaved key projections stay strided
        # views: only the normalized feature axis must remain last.
        rows = hidden.reshape(-1, hidden.shape[-1])
        if hidden.dtype == torch.float32:
            output = _residual_rms_norm(rows, self.weight, self.variance_epsilon, dtype)
        else:
            output, _ = _activation_rms_norm(rows, self.weight, self.variance_epsilon, dtype)
        return output.reshape(hidden.shape)


def fused_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    key_positions: torch.Tensor,
    key_padding: torch.Tensor,
    documents: torch.Tensor,
    *,
    query_start: int,
    window: int | None,
    selection: RegionSelection | None,
) -> torch.Tensor:
    """Run the shared compiled FlexAttention kernel without a dense token mask.

    Args:
        query: CUDA tensor [batch, local_queries, query_heads, head_dim].
        key: CUDA tensor [batch, global_keys, kv_heads, head_dim].
        value: CUDA tensor with the same shape and dtype as key.
        positions: Document-local positions [batch, local_queries].
        key_positions: Document-local positions [batch, global_keys].
        key_padding: Boolean tensor [batch, global_keys], true for padding.
        documents: Document ids [batch, global_keys].
        query_start: Global offset of the contiguous query shard.
        window: Sliding window size, or None for CSA layers.
        selection: Compact sorted CSA region ids, or None for sliding attention.

    Returns:
        Attention output [batch, local_queries, query_heads, head_dim].
    """
    batch, queries, heads, dim = query.shape
    keys, kv_heads = key.shape[1:3]
    heads_per_group = heads // kv_heads
    query_documents = documents[:, query_start : query_start + queries]
    ids = selection.region_ids if selection is not None else None
    token_regions = selection.token_regions if selection is not None else None
    bitmap = None
    if ids is not None:
        regions = int(token_regions.max().item()) + 1
        words = (regions + 31) // 32
        if batch * kv_heads * queries * words * 4 > 128 * 1024**2:
            raise ValueError("Step5 Flex comparison bitmap exceeds 128 MiB; use BackendConfig(attn='fa4')")
        bitmap = pack_region_bitmap(ids, regions)

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        safe_q = q.clamp(max=queries - 1)
        safe_k = k.clamp(max=keys - 1)
        distance = positions[b, safe_q] - key_positions[b, safe_k]
        keep = (
            (q < queries)
            & (k < keys)
            & (distance >= 0)
            & ~key_padding[b, safe_k]
            & (query_documents[b, safe_q] == documents[b, safe_k])
        )
        if window is not None:
            keep = keep & (distance < window)
        if bitmap is not None:
            target = token_regions[b, safe_k]
            word = bitmap[b, h // heads_per_group, safe_q, target // 32]
            keep = keep & (((word >> (target % 32)) & 1) != 0)
        return keep

    # Coarse metadata is O((Q / 128) * (K / 128)), shared across heads and
    # batches. The fine predicate runs inside the fused attention kernel.
    block = 128
    q_blocks = (queries + block - 1) // block
    k_blocks = (keys + block - 1) // block
    rows = torch.arange(q_blocks, device=query.device)
    last = ((query_start + (rows + 1) * block - 1) // block).clamp(max=k_blocks - 1)
    first = (
        torch.zeros_like(last) if window is None else ((query_start + rows * block - window + 1) // block).clamp(min=0)
    )
    width = k_blocks if window is None else min(k_blocks, (window + 2 * block - 2) // block + 1)
    indices = first[:, None] + torch.arange(width, device=query.device)
    counts = (last - first + 1).clamp(min=0, max=width)
    indices = indices.clamp(max=k_blocks - 1).to(torch.int32)[None, None]
    # Construct reverse metadata analytically. from_kv_blocks otherwise expands
    # to a dense block grid, and assumes every row has K_blocks storage slots.
    columns = torch.arange(k_blocks, device=query.device)
    q_first = ((columns * block - query_start) // block).clamp(min=0)
    q_last = (
        torch.full_like(columns, q_blocks - 1)
        if window is None
        else ((((columns + 1) * block + window - 2 - query_start) // block).clamp(max=q_blocks - 1))
    )
    q_width = q_blocks if window is None else min(q_blocks, (window + 2 * block - 2) // block + 1)
    q_counts = (q_last - q_first + 1).clamp(min=0, max=q_width)
    q_indices = (q_first[:, None] + torch.arange(q_width, device=query.device)).clamp(max=q_blocks - 1)
    block_mask = BlockMask(
        seq_lengths=(queries, keys),
        kv_num_blocks=counts.to(torch.int32)[None, None],
        kv_indices=indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=q_counts.to(torch.int32)[None, None],
        q_indices=q_indices.to(torch.int32)[None, None],
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(block, block),
        mask_mod=mask_mod,
    )
    # FlexAttention needs power-of-two head dimensions. Preserve Step-5's
    # original scale when zero-padding 192 to 256, then discard padded outputs.
    padded = 1 << (dim - 1).bit_length()
    q = F.pad(query.transpose(1, 2), (0, padded - dim))
    k = F.pad(key.transpose(1, 2), (0, padded - dim))
    v = F.pad(value.transpose(1, 2), (0, padded - dim))
    out = FlexAttention.flex_attn(q, k, v, block_mask=block_mask, scale=dim**-0.5, enable_gqa=True)
    return out[..., :dim].transpose(1, 2)


region_replay: RecomputeReplay[RegionSelection] = RecomputeReplay("Step5 CSA route")


@torch.compile(fullgraph=True)
def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Fuse FP32 SwiGLU and its derivative without FP32 activation temporaries.

    Args:
        gate: Projection tensor [batch, sequence, intermediate].
        up: Projection tensor with the same shape and dtype.
        limit: Positive clamp bound; non-positive values disable clamping.

    Returns:
        Tensor with the same shape and dtype as gate.
    """
    activated, values = F.silu(gate.float()), up.float()
    if limit > 0:
        activated = activated.clamp(max=limit)
        values = values.clamp(-limit, limit)
    return (activated * values).to(gate.dtype)


class Step5MLP(nn.Module):
    """SwiGLU with separate gate/up projections and FP32 asymmetric clamping."""

    def __init__(
        self, config: Step5Config, backend: BackendConfig, *, width: int, limit: float, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.limit = limit
        self.gate_proj = initialize_linear_module(backend.linear, config.hidden_size, width, bias=False, dtype=dtype)
        self.up_proj = initialize_linear_module(backend.linear, config.hidden_size, width, bias=False, dtype=dtype)
        self.down_proj = initialize_linear_module(backend.linear, width, config.hidden_size, bias=False, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Evaluate the dense or shared feed-forward block.

        Args:
            hidden: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden] in the input dtype.
        """
        return self.down_proj(clamped_swiglu(self.gate_proj(hidden), self.up_proj(hidden), self.limit))


class Step5Attention(nn.Module):
    """GQA with windowed or CSA-selected keys and contiguous query-shard CP."""

    def __init__(self, config: Step5Config, layer_idx: int, backend: BackendConfig, *, dtype: torch.dtype) -> None:
        super().__init__()
        if backend.attn not in ("fa4", "flex", "sdpa", "eager", "torch"):
            raise ValueError("Step5 attention supports fa4, flex, sdpa and eager backends")
        self.backend = backend
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_groups
        self.heads_per_group = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self.cp_group: dist.ProcessGroup | None = None
        self.qkv_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, bias=False, dtype=dtype
        )
        self.g_proj = (
            initialize_linear_module(backend.linear, config.hidden_size, self.num_heads, bias=False, dtype=dtype)
            if config.use_head_wise_attn_gate
            else None
        )
        self.q_norm = Step5RMSNorm(self.head_dim, config.rms_norm_eps, compile_norm=backend.compile_norm)
        self.k_norm = Step5RMSNorm(self.head_dim, config.rms_norm_eps, compile_norm=backend.compile_norm)
        span = int(self.head_dim * config.partial_rotary_factors[layer_idx])
        self.rotary_emb = Step5RotaryEmbedding(config, layer_idx, span=span)
        sparse = (
            config.sparse_config.enabled and config.layer_types[layer_idx] in config.sparse_config.apply_to_layer_types
        )
        self.indexer = Step5SparseIndexer(config, layer_idx, dtype=dtype) if sparse else None

    def setup_cp_attention(self, cp_mesh: DeviceMesh) -> None:
        """Attach the CP group through AutoModel's model-owned attention hook."""
        self.cp_group = cp_mesh.get_group() if cp_mesh.size() > 1 else None

    def _select_regions(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        key_positions: torch.Tensor,
        key_padding: torch.Tensor,
        *,
        query_start: int,
        sparse_metadata: RegionMetadata | None,
    ) -> RegionSelection:
        """Select once or replay integer routes during activation recomputation.

        Args:
            hidden: Local normalized states [batch, local_queries, hidden].
            positions: Int64 local document positions [batch, local_queries].
            key_positions: Int64 global document positions [batch, global_keys].
            key_padding: Boolean global padding flags [batch, global_keys].
            query_start: Physical start of the local query shard.
            sparse_metadata: Layer-independent global/local region layout.

        Returns:
            Integer-only region selection; no projected K/V or activations are
            retained. The native checkpoint context owns the record lifetime.
        """
        replay = region_replay.current()
        if replay is not None and replay[1] == "replay":
            cached = replay[0].take()
            if cached is not None:
                return cached
        # The entire no-grad decision is invisible to selective checkpointing,
        # so its omitted recompute cannot shift MoE's per-op replay indexes.
        with _disable_current_modes(), torch.no_grad():
            if sparse_metadata is None:
                sparse_metadata = region_metadata(
                    key_positions,
                    key_padding,
                    query_start=query_start,
                    queries=hidden.shape[1],
                    region_size=self.indexer.config.region_block_size,
                    topk=self.indexer.config.topk,
                )
            index_k, index_z = None, None
            # This global predicate is identical across all CP ranks. A rank
            # with only dense-prefix queries still contributes keys when
            # another rank owns sparse queries.
            if sparse_metadata.needs_projection:
                index_k, index_z = self.indexer.project_keys(hidden, positions)
                index_kz = gather_metadata(torch.stack((index_k, index_z), dim=2), self.cp_group)
                index_k, index_z = index_kz.unbind(dim=2)
            else:
                # Keep FSDP norm hooks symmetric with DP peers that may own
                # long documents, while avoiding any nonempty projection.
                self.indexer.project_keys(hidden[:, :0], positions[:, :0])
            selection = self.indexer(
                hidden,
                positions,
                index_k,
                index_z,
                key_positions,
                key_padding,
                query_start=query_start,
                metadata=sparse_metadata,
            )
        if replay is not None and replay[1] == "record":
            replay[0].record(selection)
        return selection

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
        global_positions: torch.Tensor,
        global_padding: torch.Tensor,
        *,
        sparse_metadata: RegionMetadata | None = None,
    ) -> torch.Tensor:
        """Attend local queries to valid global keys, preserving document boundaries.

        Args:
            hidden: Tensor of shape [batch, local_sequence, hidden].
            positions: Tensor of shape [batch, local_sequence], document-local positions.
            padding_mask: Boolean tensor of shape [batch, local_sequence], true for padding.
            global_positions: Int tensor [batch, global_sequence], gathered once per model call.
            global_padding: Boolean tensor [batch, global_sequence], true for padding.

        Returns:
            Tensor of shape [batch, local_sequence, hidden].
        """
        batch, sequence = hidden.shape[:2]
        groups, heads, dim = self.num_kv_heads, self.heads_per_group, self.head_dim
        # Interleaved group layout: [group, query_heads + key_head + value_head, dim].
        qkv = self.qkv_proj(hidden).view(batch, sequence, groups, heads + 2, dim)
        query = qkv[:, :, :, :heads]
        key, value = qkv[:, :, :, heads], qkv[:, :, :, heads + 1]
        # Preserve the QKV slice through normalization; merging group/head axes
        # beforehand would materialize the otherwise strided query input.
        query = self.q_norm(query).flatten(2, 3)
        query, key = self.rotary_emb.forward_pair(query, self.k_norm(key), positions)
        query_start = 0 if self.cp_group is None else dist.get_rank(self.cp_group) * sequence
        key_positions, key_padding = global_positions, global_padding
        if self.cp_group is not None:
            local_kv = torch.stack((key, value), dim=2)
            if self.window is None:
                exchanged = gather_sequence(local_kv, self.cp_group)
            else:
                first_key = max(0, query_start - self.window + 1)
                # Align FA4's online-softmax tiles across CP sizes. Otherwise BF16
                # rounding can change the downstream discrete CSA/MoE routes.
                # Padding is local: transport still exchanges exactly window - 1.
                tile_width = (
                    128
                    if hidden.is_cuda and torch.cuda.get_device_capability(hidden.device) == (10, 0)
                    else (112 if self.head_dim == 192 else 80)
                )
                left_padding = first_key % tile_width if self.backend.attn == "fa4" else 0
                if (
                    tile_width == 112
                    and hidden.is_cuda
                    and sparse_metadata is not None
                    and sparse_metadata.window_layout is not None
                ):
                    # Reuse the existing single padded-halo concatenation. The
                    # 511 communicated tokens are unchanged; these extra zeros
                    # align native FA4's document-relative 96-token K tiles.
                    left_padding = max(left_padding, sparse_metadata.window_layout.leading_padding)
                exchanged = gather_halo(local_kv, self.cp_group, self.window - 1, left_padding=left_padding)
                key_positions = F.pad(global_positions[:, first_key : query_start + sequence], (left_padding, 0))
                key_padding = F.pad(
                    global_padding[:, first_key : query_start + sequence], (left_padding, 0), value=True
                )
                query_start += left_padding - first_key
            key, value = exchanged.unbind(dim=2)
        documents = ((key_positions == 0) & ~key_padding).long().cumsum(1)
        query_documents = documents[:, query_start : query_start + sequence]
        selection = None
        if self.indexer is not None:
            # Selection is discrete. No language-model-loss gradient crosses top-k;
            # differentiable K/V transport above carries attention gradients.
            selection = self._select_regions(
                hidden,
                positions,
                key_positions,
                key_padding,
                query_start=query_start,
                sparse_metadata=sparse_metadata,
            )
        if self.backend.attn == "fa4" and hidden.is_cuda:
            attended = fa4_attention(
                query,
                key,
                value,
                positions,
                key_positions,
                key_padding,
                documents,
                query_start=query_start,
                window=self.window,
                selection=selection,
                window_layout=None if sparse_metadata is None else sparse_metadata.window_layout,
            )
        elif self.backend.attn == "flex" and hidden.is_cuda:
            attended = fused_attention(
                query,
                key,
                value,
                positions,
                key_positions,
                key_padding,
                documents,
                query_start=query_start,
                window=self.window,
                selection=selection,
            )
        else:
            if sequence * key.shape[1] > 2048**2:
                raise ValueError("Use backend.attn='fa4' for long Step5 sequences; eager/SDPA are small-input oracles")
            key_length = key.shape[1]
            key = key.permute(0, 2, 1, 3).reshape(batch * groups, 1, key_length, dim)
            value = value.permute(0, 2, 1, 3).reshape(batch * groups, 1, key_length, dim)
            outputs = []
            for start in range(0, sequence, 256):
                stop = min(start + 256, sequence)
                local_q = query[:, start:stop].view(batch, stop - start, groups, heads, dim)
                local_q = local_q.permute(0, 2, 3, 1, 4).reshape(batch * groups, heads, stop - start, dim)
                distance = positions[:, start:stop, None] - key_positions[:, None, :]
                allowed = (
                    (distance >= 0)
                    & ~key_padding[:, None, :]
                    & (query_documents[:, start:stop, None] == documents[:, None, :])
                )
                if self.window is not None:
                    allowed = allowed & (distance < self.window)
                allowed = allowed[:, None].expand(-1, groups, -1, -1)
                if selection is not None:
                    targets = (
                        selection.token_regions[:, None, None, :].expand(-1, groups, stop - start, -1).contiguous()
                    )
                    ids = selection.region_ids[:, :, start:stop].contiguous()
                    slots = torch.searchsorted(ids, targets).clamp(max=ids.shape[-1] - 1)
                    allowed = allowed & (ids.gather(-1, slots) == targets)
                mask = allowed.reshape(batch * groups, 1, stop - start, key_length)
                if self.backend.attn == "eager":
                    logits = torch.matmul(local_q.float(), key.float().transpose(-1, -2)) * dim**-0.5
                    probabilities = logits.masked_fill(~mask, float("-inf")).softmax(-1).nan_to_num(0.0)
                    out = torch.matmul(probabilities, value.float()).to(hidden.dtype)
                else:
                    out = F.scaled_dot_product_attention(local_q, key, value, attn_mask=mask, enable_gqa=True)
                out = out.view(batch, groups, heads, stop - start, dim).permute(0, 3, 1, 2, 4)
                outputs.append(out.reshape(batch, stop - start, self.num_heads, dim))
            attended = torch.cat(outputs, dim=1)
        if self.g_proj is not None:
            attended = attended * self.g_proj(hidden).sigmoid().unsqueeze(-1)
        return self.o_proj(attended.flatten(-2))


class Step5DecoderLayer(nn.Module):
    """Pre-norm decoder with FP32 residual accumulation and optional shared MoE."""

    def __init__(
        self, config: Step5Config, layer_idx: int, backend: BackendConfig, moe_config: MoEConfig, *, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.self_attn = Step5Attention(config, layer_idx, backend, dtype=dtype)
        self.input_layernorm = Step5RMSNorm(config.hidden_size, config.rms_norm_eps, compile_norm=backend.compile_norm)
        self.post_attention_layernorm = Step5RMSNorm(
            config.hidden_size, config.rms_norm_eps, compile_norm=backend.compile_norm
        )
        limit = config.swiglu_limits[layer_idx]
        if config.use_moe and layer_idx in config.moe_layer_list:
            self.moe = MoE(replace(moe_config, swiglu_limit=limit), backend)
            self.share_expert = Step5MLP(
                config,
                backend,
                width=config.share_expert_dim,
                limit=config.swiglu_limits_shared[layer_idx],
                dtype=dtype,
            )
            self.mlp = None
        else:
            self.moe = None
            self.share_expert = None
            self.mlp = Step5MLP(config, backend, width=config.intermediate_size, limit=limit, dtype=dtype)

    def nemo_checkpoint_context_fn(
        self,
        context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None,
    ) -> Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None:
        """Compose the native AC context with one integer-route log per call.

        Args:
            context_fn: Existing plain/selective checkpoint context factory.

        Returns:
            A factory with a distinct recorder for each live pipeline microbatch,
            or the unchanged factory for a layer without CSA.
        """
        if self.self_attn.indexer is None:
            return context_fn
        return region_replay.checkpoint_context_fn(context_fn)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        padding_mask: torch.Tensor,
        global_positions: torch.Tensor,
        global_padding: torch.Tensor,
        *,
        compute_dtype: torch.dtype,
        sparse_metadata: RegionMetadata | None = None,
    ) -> torch.Tensor:
        """Apply attention and feed-forward residual branches.

        Args:
            hidden: FP32 tensor of shape [batch, local_sequence, hidden].
            positions: Tensor of shape [batch, local_sequence].
            padding_mask: Boolean tensor of shape [batch, local_sequence], true for padding.
            global_positions: Int tensor [batch, global_sequence], gathered once per model call.
            global_padding: Boolean tensor [batch, global_sequence], true for padding.
            compute_dtype: Activation dtype from the completed embedding projection.
            sparse_metadata: Shared global region layout and local query metadata.

        Returns:
            FP32 tensor of shape [batch, local_sequence, hidden].
        """
        normed = self.input_layernorm(hidden, output_dtype=compute_dtype)
        hidden = (
            hidden
            + self.self_attn(
                normed, positions, padding_mask, global_positions, global_padding, sparse_metadata=sparse_metadata
            ).float()
        )
        normed = self.post_attention_layernorm(hidden, output_dtype=compute_dtype)
        if self.moe is not None:
            combined = self.moe(normed, padding_mask).float() + self.share_expert(normed).float()
            return hidden + combined
        return hidden + self.mlp(normed).float()

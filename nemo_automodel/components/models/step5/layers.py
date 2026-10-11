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

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F
from torch.utils._python_dispatch import _disable_current_modes

from nemo_automodel.components.distributed.recompute_replay import RecomputeReplay
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5.attention import fused_attention
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.cp import gather_halo, gather_metadata, gather_sequence
from nemo_automodel.components.models.step5.fa4 import fa4_attention
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding
from nemo_automodel.components.models.step5.sparse import (
    RegionMetadata,
    RegionSelection,
    Step5SparseIndexer,
    region_metadata,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE

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
        self.q_norm = Step3p5RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Step3p5RMSNorm(self.head_dim, config.rms_norm_eps)
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
        query = qkv[:, :, :, :heads].reshape(batch, sequence, self.num_heads, dim)
        key, value = qkv[:, :, :, heads], qkv[:, :, :, heads + 1]
        query = self.rotary_emb(self.q_norm(query), positions)
        key = self.rotary_emb(self.k_norm(key), positions)
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
        self.input_layernorm = Step3p5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Step3p5RMSNorm(config.hidden_size, config.rms_norm_eps)
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
        normed = self.input_layernorm(hidden).to(compute_dtype)
        hidden = (
            hidden
            + self.self_attn(
                normed, positions, padding_mask, global_positions, global_padding, sparse_metadata=sparse_metadata
            ).float()
        )
        normed = self.post_attention_layernorm(hidden).to(compute_dtype)
        if self.moe is not None:
            combined = self.moe(normed, padding_mask).float() + self.share_expert(normed).float()
            return hidden + combined
        return hidden + self.mlp(normed).float()

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

"""Gated MLA with learnable sinks for Hy4-flash: sliding-window and compressed-indexer sparse layers.

Both layer types run the same *absorbed* latent attention. The per-head key up-projection is folded into the
query (``q_nope @ W_k``), every head then attends over one shared ``[kv_lora_rank + rope]`` latent per token, and
the value up-projection is applied to the latent output. This is mathematically the reference's expanded
attention, and it is what the shared FlashMLA-forward / cuDNN-backward sparse kernel
(``models/common/cudnn_sparse_attention``) consumes, so the kernel handles the learnable sink and per-document
index lists for both layer types:

* ``sliding_attention``: the indices of the last ``sliding_window`` tokens of the query's document.
* ``deepseek_sparse_attention``: dense causal attention while a document is not longer than ``index_topk``;
  otherwise the tokens of the ``index_topk // index_kpool - 1`` best-scoring complete groups plus the query's own
  group, chosen by :class:`HY4FlashCompressedIndexer`.

Packed documents are isolated: indices only ever point into the query's own document and groups are
document-local.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.cudnn_sparse_attention import cudnn_sparse_attention
from nemo_automodel.components.models.deepseek_v41.cp import gather_sequence
from nemo_automodel.components.models.hy_v4_flash.config import AttentionType, HY4FlashTextConfig
from nemo_automodel.components.models.hy_v4_flash.layers import HY4FlashRMSNorm, apply_interleaved_rope


@dataclass(frozen=True)
class HY4FlashDocLayout:
    """Document boundaries of one packed row, built once per forward pass and shared by every layer.

    Attributes:
        doc_ranges: ``(start, end)`` token intervals of the real documents in the *global* sequence
            (``end`` exclusive); padding is not listed.
        local_start: First global token held by this rank (0 without context parallelism).
        local_end: One past the last global token held by this rank.
        doc_start: Integer tensor of shape [local_tokens], the global index of the first token of each local token's
            document; ``position + 1`` for padding, which makes every key of a padding query invalid.
        doc_end: Integer tensor of shape [local_tokens], one past the last token of each local token's document.
    """

    doc_ranges: tuple[tuple[int, int], ...]
    local_start: int
    local_end: int
    doc_start: torch.Tensor
    doc_end: torch.Tensor


@dataclass(frozen=True)
class HY4FlashRope:
    """Interleaved M-RoPE tables of one attention type over the *global* sequence.

    Attributes:
        cos: Tensor of shape [global_tokens, rope_dim // 2].
        sin: Tensor of shape [global_tokens, rope_dim // 2].
    """

    cos: torch.Tensor
    sin: torch.Tensor


class HY4FlashFP32Parameter(nn.Module):
    """FP32 parameter holder whose forward returns its weight and ignores the input.

    A bare parameter on a bf16 module would be cast by the FSDP mixed-precision policy. Holding it in its own module lets
    the parallelizer give it a dedicated fp32 unit (and one cross-block unit for the sinks); the ignored input argument
    ties the call to the activation graph so FSDP's pre/post hooks fire.
    """

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(value.to(torch.float32))

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        """Return the weight (``reference`` only anchors the FSDP hooks)."""
        del reference
        return self.weight


class HY4FlashCompressedIndexer(nn.Module):
    """Group-pooled lightning indexer. Selection is discrete, so the indexer receives no loss gradient.

    Token keys are pooled ``index_kpool`` at a time through a softmax over fp32 value / gate projections
    (``key_value_projection`` / ``key_gate_projection`` biased by ``intra_group_ape``), normalised, and rotated
    with the sparse layer's RoPE at the first token of each group. Queries come from the attention's
    ``q_a_layernorm`` output.
    """

    def __init__(self, config: HY4FlashTextConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.kpool = config.index_kpool
        self.group_topk = config.index_topk // config.index_kpool
        self.rope_dim = config.qk_rope_head_dim
        self.query_projection = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.head_weight_projection = nn.Linear(config.hidden_size, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.token_key_projection = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.bfloat16)
        self.token_key_norm = HY4FlashRMSNorm(self.head_dim, config.rms_norm_eps, torch.bfloat16)
        self.key_value_projection = nn.Linear(self.head_dim, self.head_dim, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(self.head_dim, self.head_dim, bias=False, dtype=torch.float32)
        # The released key_norm weight is fp32; keep it exact instead of rounding it to bf16 (the output is cast back below).
        self.key_norm = HY4FlashRMSNorm(self.head_dim, config.rms_norm_eps, torch.float32)
        self.intra_group_ape = HY4FlashFP32Parameter(torch.empty(self.kpool, self.head_dim))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def token_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalised per-token index keys.

        Args:
            hidden_states: Tensor of shape [tokens, hidden].

        Returns:
            Tensor of shape [tokens, index_head_dim].
        """
        return self.token_key_norm(self.token_key_projection(hidden_states))

    @torch.no_grad()
    def pool_groups(self, token_keys: torch.Tensor) -> torch.Tensor:
        """Pool one document's token keys into complete-group keys.

        Args:
            token_keys: Tensor of shape [document_tokens, index_head_dim].

        Returns:
            Tensor of shape [document_tokens // index_kpool, index_head_dim] (a trailing partial group has no key).
        """
        groups = token_keys.shape[0] // self.kpool
        grouped = token_keys[: groups * self.kpool].reshape(groups, self.kpool, self.head_dim)
        with torch.autocast(device_type=token_keys.device.type, enabled=False):
            grouped = grouped.to(self.key_value_projection.weight.dtype)
            values = self.key_value_projection(grouped)
            gates = self.key_gate_projection(grouped) + self.intra_group_ape(grouped)
            pooled = (values * gates.softmax(dim=1)).sum(dim=1)
        return self.key_norm(pooled).to(token_keys.dtype)

    @torch.no_grad()
    def select(
        self,
        q_resid: torch.Tensor,
        hidden_states: torch.Tensor,
        query_rope: HY4FlashRope,
        group_keys: torch.Tensor,
        group_rope: HY4FlashRope,
        query_slots: torch.Tensor,
        document_length: int,
    ) -> torch.Tensor:
        """Select the document-local token indices every query attends to.

        Args:
            q_resid: Tensor of shape [queries, q_lora_rank].
            hidden_states: Tensor of shape [queries, hidden].
            query_rope: Tables of shape [queries, rope_dim // 2] at the queries' positions.
            group_keys: Tensor of shape [groups, index_head_dim], unrotated pooled keys of the document.
            group_rope: Tables of shape [groups, rope_dim // 2] at the first token of each group.
            query_slots: Integer tensor of shape [queries], document-local token index of each query.
            document_length: Number of tokens of the document.

        Returns:
            Int32 tensor of shape [queries, index_topk] of document-local token indices, ``-1`` where unused.
        """
        queries, groups = q_resid.shape[0], group_keys.shape[0]
        width = self.group_topk * self.kpool
        offsets = torch.arange(self.kpool, device=q_resid.device)
        own_group = query_slots // self.kpool
        complete_topk = min(self.group_topk - 1, groups)
        if groups:
            nope = self.head_dim - self.rope_dim
            query = self.query_projection(q_resid).view(queries, self.n_heads, self.head_dim)
            query = torch.cat(
                (
                    query[..., :nope],
                    apply_interleaved_rope(query[..., nope:].unsqueeze(0), query_rope.cos[None], query_rope.sin[None])[
                        0
                    ],
                ),
                dim=-1,
            )
            keys = torch.cat(
                (
                    group_keys[:, :nope],
                    apply_interleaved_rope(
                        group_keys[None, :, None, nope:], group_rope.cos[None], group_rope.sin[None]
                    )[0, :, 0],
                ),
                dim=-1,
            )
            head_weights = self.head_weight_projection(hidden_states).float() * (
                self.n_heads**-0.5 * self.head_dim**-0.5
            )
            scores = F.relu(torch.einsum("qhd,gd->qhg", query.float(), keys.float()))
            logits = torch.einsum("qh,qhg->qg", head_weights, scores)
            # A query sees only groups that closed before its own group.
            logits = logits.masked_fill(torch.arange(groups, device=logits.device) >= own_group[:, None], float("-inf"))
            recalled = logits.topk(complete_topk, dim=-1).indices
            valid_recalled = torch.gather(logits, 1, recalled) > float("-inf")
        else:
            recalled = own_group.new_empty((queries, 0))
            valid_recalled = torch.zeros((queries, 0), dtype=torch.bool, device=q_resid.device)
        tokens = recalled.unsqueeze(-1) * self.kpool + offsets
        tokens = tokens.masked_fill(~valid_recalled.unsqueeze(-1), -1).flatten(-2)
        own = own_group.unsqueeze(-1) * self.kpool + offsets
        own = own.masked_fill((own > query_slots[:, None]) | (own >= document_length), -1)
        selected = torch.cat((tokens, own), dim=-1)
        return F.pad(selected, (0, width - selected.shape[-1]), value=-1).to(torch.int32)


def sliding_window_indices(position: torch.Tensor, doc_start: torch.Tensor, window: int) -> torch.Tensor:
    """Global indices of the last ``window`` tokens (including the query) of each query, inside its document.

    Args:
        position: Integer tensor of shape [queries], global token index of each query.
        doc_start: Integer tensor of shape [queries], global index of each query's document start.
        window: Number of keys per query.

    Returns:
        Int32 tensor of shape [queries, window] of global key indices, ``-1`` before the document start.
    """
    indices = position[:, None] - torch.arange(window - 1, -1, -1, device=position.device)
    return torch.where(indices >= doc_start[:, None], indices, -1).to(torch.int32)


def dense_causal_indices(
    position: torch.Tensor, doc_start: torch.Tensor, doc_end: torch.Tensor, width: int
) -> torch.Tensor:
    """Global indices of every key of the query's document up to the query, padded with ``-1`` to ``width``.

    Args:
        position: Integer tensor of shape [queries], global token index of each query.
        doc_start: Integer tensor of shape [queries], global start of each query's document.
        doc_end: Integer tensor of shape [queries], global end (exclusive) of each query's document.
        width: Output width; documents longer than ``width`` must not be passed.

    Returns:
        Int32 tensor of shape [queries, width].
    """
    indices = doc_start[:, None] + torch.arange(width, device=position.device)
    valid = (indices <= position[:, None]) & (indices < doc_end[:, None])
    return torch.where(valid, indices, -1).to(torch.int32)


def reference_sparse_latent_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    value_dim: int,
    chunk: int = 64,
) -> torch.Tensor:
    """PyTorch implementation of the shared sparse latent attention kernel (CPU/CUDA, any dtype).

    Args:
        q: Tensor of shape [queries, heads, latent_dim + rope_dim].
        kv: Tensor of shape [keys, latent_dim + rope_dim], shared by all heads.
        indices: Integer tensor of shape [queries, width] into ``kv`` rows, ``-1`` marking unused slots.
        sink: fp32 tensor of shape [heads], learned sink logits (scaled-logit space).
        scale: Softmax scale applied to ``q . k``.
        value_dim: Leading channels of ``kv`` used as the value (the latent width).
        chunk: Queries processed at a time.

    Returns:
        Tensor of shape [queries, heads, value_dim] in the dtype of ``q``.
    """
    outputs = []
    for start in range(0, q.shape[0], chunk):
        idx = indices[start : start + chunk].long()
        valid = idx >= 0
        gathered = kv[idx.clamp_min(0)].float()  # [c, width, dim]
        logits = torch.einsum("chd,cwd->chw", q[start : start + chunk].float(), gathered) * scale
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        sink_col = sink.float()[None, :, None].expand(logits.shape[0], -1, 1)
        probs = torch.cat((logits, sink_col), dim=-1).softmax(dim=-1)[..., :-1]
        outputs.append(torch.einsum("chw,cwd->chd", probs, gathered[..., :value_dim]).to(q.dtype))
    return torch.cat(outputs, dim=0)


class HY4FlashAttention(nn.Module):
    """Gated MLA of one decoder layer (``attention_type`` selects heads, indices and rotary theta)."""

    def __init__(self, config: HY4FlashTextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type: AttentionType = config.layer_types[layer_idx]
        self.backend = backend
        self.num_heads = config.attention_heads(self.attention_type)
        self.nope_dim, self.rope_dim, self.v_dim = config.attention_dims(self.attention_type)
        self.kv_lora_rank = config.kv_lora_rank
        self.window = config.sliding_window
        self.index_topk = config.index_topk
        self.scale = (self.nope_dim + self.rope_dim) ** -0.5
        hidden, dtype = config.hidden_size, torch.bfloat16
        linear = backend.linear
        self.q_a_proj = initialize_linear_module(linear, hidden, config.q_lora_rank, config.attention_bias, dtype=dtype)
        self.q_a_layernorm = HY4FlashRMSNorm(config.q_lora_rank, config.rms_norm_eps, dtype)
        self.q_b_proj = initialize_linear_module(
            linear, config.q_lora_rank, self.num_heads * (self.nope_dim + self.rope_dim), False, dtype=dtype
        )
        self.kv_a_proj_with_mqa = initialize_linear_module(
            linear, hidden, config.kv_lora_rank + self.rope_dim, config.attention_bias, dtype=dtype
        )
        self.kv_a_layernorm = HY4FlashRMSNorm(config.kv_lora_rank, config.rms_norm_eps, dtype)
        self.kv_b_proj = initialize_linear_module(
            linear, config.kv_lora_rank, self.num_heads * (self.nope_dim + self.v_dim), False, dtype=dtype
        )
        self.o_proj = initialize_linear_module(
            linear, self.num_heads * self.v_dim, hidden, config.attention_bias, dtype=dtype
        )
        self.gate_proj = initialize_linear_module(linear, hidden, self.num_heads * self.v_dim, False, dtype=dtype)
        self.sinks_param = HY4FlashFP32Parameter(torch.full((self.num_heads,), config.learnable_sink_init))
        self.compressed_indexer = (
            HY4FlashCompressedIndexer(config) if self.attention_type == "deepseek_sparse_attention" else None
        )
        self._cp_group: dist.ProcessGroup | None = None

    def setup_cp_attention(self, cp_mesh: DeviceMesh) -> None:
        """Attach the context-parallel group (called by the shared parallelizer): keys are gathered, queries stay local."""
        self._cp_group = cp_mesh.get_group()

    def _absorbed_weights(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head key / value up-projections of ``kv_b_proj`` as ``[heads, dim, latent]`` tensors.

        Evaluating the module on an identity keeps the result differentiable through LoRA adapters and valid
        when FSDP shards ``kv_b_proj`` (a direct ``.weight`` read would see the sharded DTensor).
        """
        eye = torch.eye(self.kv_lora_rank, dtype=dtype, device=device)
        weight = self.kv_b_proj(eye).T.reshape(self.num_heads, self.nope_dim + self.v_dim, self.kv_lora_rank)
        return weight[:, : self.nope_dim], weight[:, self.nope_dim :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        layout: HY4FlashDocLayout,
        rope: HY4FlashRope,
        indexer_rope: HY4FlashRope | None = None,
    ) -> torch.Tensor:
        """Attend within documents.

        Args:
            hidden_states: Tensor of shape [1, local_tokens, hidden] (the local interval of the packed row).
            layout: Document boundaries of the packed row.
            rope: Rotary tables of this layer's attention type over the global sequence.
            indexer_rope: For sparse layers, the same tables (the indexer shares the layer's theta).

        Returns:
            Tensor of shape [1, local_tokens, hidden]; positions outside every document are zero.
        """
        x = hidden_states[0]
        tokens = x.shape[0]
        lo, hi = layout.local_start, layout.local_end
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).view(tokens, self.num_heads, self.nope_dim + self.rope_dim)
        q_nope, q_rot = q[..., : self.nope_dim], q[..., self.nope_dim :]
        latent_rot = self.kv_a_proj_with_mqa(x)
        latent = self.kv_a_layernorm(latent_rot[..., : self.kv_lora_rank])
        local_rope = HY4FlashRope(rope.cos[lo:hi], rope.sin[lo:hi])
        q_rot = apply_interleaved_rope(q_rot[None], local_rope.cos[None], local_rope.sin[None])[0]
        k_rot = apply_interleaved_rope(
            latent_rot[None, :, None, self.kv_lora_rank :], local_rope.cos[None], local_rope.sin[None]
        )[0, :, 0]
        gate = torch.sigmoid(self.gate_proj(x)).view(tokens, self.num_heads, self.v_dim)

        w_k, w_v = self._absorbed_weights(x.dtype, x.device)
        q_abs = torch.einsum("qhd,hdc->qhc", q_nope, w_k)
        query = torch.cat((q_abs, q_rot), dim=-1)
        kv_local = torch.cat((latent, k_rot), dim=-1)
        token_keys = self.compressed_indexer.token_keys(x) if self.compressed_indexer is not None else None
        kv, keys = self._gather_keys(kv_local, token_keys)

        position = torch.arange(lo, hi, device=x.device)
        doc_start, doc_end = layout.doc_start, layout.doc_end
        if self.compressed_indexer is None:
            indices = sliding_window_indices(position, doc_start, self.window)
        else:
            # Documents that fit inside index_topk attend densely (vectorised over all such queries); longer ones
            # select groups through the indexer, one document at a time.
            indices = dense_causal_indices(position, doc_start, doc_end, self.index_topk)
            for start, end in layout.doc_ranges:
                q_start, q_end = max(start, lo), min(end, hi)
                if end - start <= self.index_topk or q_end <= q_start:
                    continue
                group_keys = self.compressed_indexer.pool_groups(keys[start:end])
                kpool = self.compressed_indexer.kpool
                first_tokens = slice(start, start + group_keys.shape[0] * kpool, kpool)
                local = self.compressed_indexer.select(
                    q_resid[q_start - lo : q_end - lo],
                    x[q_start - lo : q_end - lo],
                    HY4FlashRope(rope.cos[q_start:q_end], rope.sin[q_start:q_end]),
                    group_keys,
                    HY4FlashRope(rope.cos[first_tokens], rope.sin[first_tokens]),
                    position[q_start - lo : q_end - lo] - start,
                    end - start,
                )
                indices[q_start - lo : q_end - lo] = torch.where(local >= 0, local + start, -1)

        out_latent = self._attend(query, kv, indices, self.sinks_param(query))
        out = torch.einsum("qhc,hvc->qhv", out_latent, w_v) * gate
        return self.o_proj(out.reshape(tokens, -1))[None]

    def _gather_keys(
        self, kv_local: torch.Tensor, token_keys: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the latent keys (and index token keys) over the global sequence.

        Without context parallelism the local tensors already are the global ones. With it, only the 576-wide latent
        keys (differentiable: backward reduce-scatters their gradients to the owning rank) and the 128-wide index token
        keys (no gradient) travel, instead of the 3072-wide hidden states.

        Args:
            kv_local: Tensor of shape [local_tokens, kv_lora_rank + rope_dim].
            token_keys: Optional tensor of shape [local_tokens, index_head_dim].

        Returns:
            The same tensors over the global sequence.
        """
        if self._cp_group is None or dist.get_world_size(self._cp_group) == 1:
            return kv_local, token_keys
        kv = gather_sequence(kv_local[None], self._cp_group)[0]
        if token_keys is None:
            return kv, None
        with torch.no_grad():
            keys = gather_sequence(token_keys[None], self._cp_group)[0]
        return kv, keys

    def _attend(
        self, query: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sinks: torch.Tensor
    ) -> torch.Tensor:
        """Run the sparse latent attention kernel (or its PyTorch twin).

        Args:
            query: Tensor of shape [queries, heads, kv_lora_rank + rope_dim].
            kv: Tensor of shape [keys, kv_lora_rank + rope_dim].
            indices: Int32 tensor of shape [queries, width] into ``kv`` rows, ``-1`` unused.
            sinks: fp32 tensor of shape [heads], the learned sink logits.

        Returns:
            Tensor of shape [queries, heads, kv_lora_rank].
        """
        if self.backend.attn == "cudnn":
            return cudnn_sparse_attention(
                query.contiguous(),
                kv.unsqueeze(1).contiguous(),
                indices.unsqueeze(1).contiguous(),
                self.scale,
                attn_sink=sinks,
            )
        return reference_sparse_latent_attention(query, kv, indices, sinks, self.scale, self.kv_lora_rank)

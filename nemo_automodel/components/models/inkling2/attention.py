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

"""Learned-relative GQA with eager reference and compiled training attention."""

import weakref
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from .config import Inkling2TextConfig
from .context_parallel import convolution_prefix, gather_preceding
from .flex_attention import Inkling2FlexAttention
from .layers import L2Clip, ShortConvolution, gate_context, value_norm
from .triton_attention import relative_attention

# Key layout of the most recent token layout, per number of exchanged keys; every layer with that number reuses it.
_KEY_LAYOUTS: dict[int, tuple[weakref.ref, weakref.ref, torch.Tensor, torch.Tensor]] = {}
# Preceding-key reach of the most recent token layout; every full layer reuses it.
_REACH: dict[str, tuple[weakref.ref, weakref.ref, int]] = {}


def _preceding_reach(valid: torch.Tensor, positions: torch.Tensor, mesh: DeviceMesh) -> int:
    """Return how many keys before a shard a query of this batch can see, the largest over the CP ranks.

    A query sees its own document only. A shard's first document may have started on the
    preceding ranks, and the logical position of its first valid token counts the tokens
    of that document before it.

    Args:
        valid: Boolean [batch, local_sequence] of this rank's shard.
        positions: Logical positions [batch, local_sequence] of this rank's shard.
        mesh: One-dimensional CP mesh with more than one rank.

    Returns:
        The same number on every rank, from one collective per token layout.
    """
    cached = _REACH.get("reach")
    if cached is not None and cached[0]() is valid and cached[1]() is positions:
        return cached[2]
    first = valid.long().argmax(dim=1, keepdim=True)
    start = torch.where(valid.gather(1, first), positions.gather(1, first), 0).max()
    dist.all_reduce(start, op=dist.ReduceOp.MAX, group=mesh.get_group())
    reach = int(start)
    _REACH["reach"] = (weakref.ref(valid), weakref.ref(positions), reach)
    return reach


def _key_layout(
    valid: torch.Tensor, positions: torch.Tensor, mesh: DeviceMesh, halo: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the positions and validity of the keys a rank attends under context parallelism.

    Args:
        valid: Boolean [batch, local_sequence] of this rank's shard.
        positions: Logical positions [batch, local_sequence] of this rank's shard.
        mesh: One-dimensional CP mesh with more than one rank.
        halo: Number of preceding keys the layer exchanges.

    Returns:
        Integer positions and Boolean validity [batch, key_sequence]. The layout is the same
        for every layer with this halo and for a checkpoint recomputation, so it is gathered once per batch.
    """
    cached = _KEY_LAYOUTS.get(halo)
    if cached is not None and cached[0]() is valid and cached[1]() is positions:
        return cached[2], cached[3]
    layout = torch.stack((positions, valid.long()), dim=-1)
    count = min(halo, layout.shape[1])
    prefix = gather_preceding(layout[:, -count:], mesh, halo, differentiable=False)
    layout = torch.cat((prefix, layout), dim=1)
    key_positions, key_valid = layout[..., 0], layout[..., 1].bool()
    for stale in [key for key, entry in _KEY_LAYOUTS.items() if entry[0]() is not valid or entry[1]() is not positions]:
        del _KEY_LAYOUTS[stale]
    _KEY_LAYOUTS[halo] = (weakref.ref(valid), weakref.ref(positions), key_positions, key_valid)
    return key_positions, key_valid


class AttentionSources(NamedTuple):
    """Q/K/V inputs, each [batch, sequence, hidden], after its respective norm."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


class Inkling2Attention(nn.Module):
    """Local or full GQA with relative features, convolutions, gate and sink."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        *,
        dtype: torch.dtype,
        backend: Literal["eager", "flex", "triton"] = "eager",
    ) -> None:
        super().__init__()
        self.cp_mesh: DeviceMesh | None = None
        self.is_local = layer_idx in config.local_layer_ids
        self.o_bias = config.o_bias
        self.heads = config.swa_num_attention_heads if self.is_local else config.num_attention_heads
        self.kv_heads = config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if self.is_local else config.head_dim
        self.d_rel = config.d_rel
        self.rel_extent = config.sliding_window_size if self.is_local else config.rel_extent
        self.window = config.sliding_window_size
        self.triton = backend == "triton"
        self.flex = (
            Inkling2FlexAttention(head_dim=self.head_dim, window=self.window if self.is_local else None)
            if backend == "flex"
            else None
        )
        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha
        self.value_norm = config.use_value_norm_no_gamma
        self.eps = config.rms_norm_eps
        q_size, kv_size = self.heads * self.head_dim, self.kv_heads * self.head_dim
        self.split_sizes = (
            q_size,
            kv_size,
            kv_size,
            self.heads * config.d_rel,
            q_size,
            self.heads,
        )
        # Preserve the source's fused [Q, K, V, R, gate, sink, padding] row layout.
        padded_rows = sum(self.split_sizes) + (-sum(self.split_sizes) % 8)
        self.qkvr = nn.Linear(config.hidden_size, padded_rows, bias=config.q_bias, dtype=dtype)
        self.wo_ud = nn.Linear(q_size, config.hidden_size, bias=config.o_bias, dtype=dtype)
        self.relative_table = nn.Parameter(torch.empty(config.d_rel, self.rel_extent, dtype=dtype))
        self.q_l2_clip = L2Clip(self.heads, self.head_dim, config.qk_l2_clip_threshold, learned_scale=True)
        self.k_l2_clip = L2Clip(
            self.kv_heads,
            self.head_dim,
            config.qk_l2_clip_threshold,
            learned_scale=False,
        )
        self.rel_l2_clip = L2Clip(self.heads, config.d_rel, config.rel_l2_clip_threshold, learned_scale=True)
        self.k_sconv = ShortConvolution(
            kv_size,
            config.sconv_kernel_size,
            dtype=dtype,
            heads=self.kv_heads,
            dynamic_input_size=0 if self.is_local else config.hidden_size,
        )
        self.v_sconv = ShortConvolution(
            kv_size,
            config.sconv_kernel_size,
            dtype=dtype,
            heads=self.kv_heads,
            dynamic_input_size=0 if self.is_local else config.hidden_size,
        )

    def setup_cp_attention(self, cp_mesh: DeviceMesh) -> None:
        """Configure model-owned KV transport through the shared MoE CP hook."""
        self.cp_mesh = cp_mesh
        self.k_sconv.cp_mesh = self.v_sconv.cp_mesh = cp_mesh

    @torch.no_grad()
    def reset_parameters(self, init_std: float) -> None:
        """Initialize fused projections, relative table, scales and convolution weights."""
        nn.init.normal_(self.qkvr.weight, std=init_std)
        self.qkvr.weight[sum(self.split_sizes) :].zero_()
        nn.init.normal_(self.wo_ud.weight, std=init_std)
        if self.qkvr.bias is not None:
            self.qkvr.bias.zero_()
        if self.wo_ud.bias is not None:
            self.wo_ud.bias.zero_()
        nn.init.normal_(self.relative_table, std=init_std)
        assert self.q_l2_clip.scale is not None and self.rel_l2_clip.scale is not None
        self.q_l2_clip.scale.fill_(1)
        self.rel_l2_clip.scale.fill_(1)
        self.k_sconv.reset_parameters(init_std)
        self.v_sconv.reset_parameters(init_std)

    def forward(self, sources: AttentionSources, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Evaluate the configured attention backend with a zero-valued virtual sink.

        Args:
            sources: Q/K/V tensors of shape [batch, sequence, hidden].
            valid: Boolean tensor of shape [batch, sequence], true for real tokens.
            positions: Integer tensor of shape [batch, sequence], logical token positions.

        Returns:
            Tensor of shape [batch, sequence, hidden] in sources.q's dtype.
            No input is mutated; future and padding keys contribute zero.
        """
        batch, sequence, _ = sources.q.shape
        projections = self.qkvr(sources.q)[..., : sum(self.split_sizes)].split(self.split_sizes, -1)
        q, k, v, r, gate, sink = projections
        q_end = self.split_sizes[0]
        k_end = q_end + self.split_sizes[1]
        v_end = k_end + self.split_sizes[2]
        # Full-attention layers, and local layers in later loop passes, give K and V
        # their own inputs. Invoke the module so LoRA updates and FSDP gather hooks
        # also apply to those rows.
        if sources.k is not sources.q:
            k = self._project_rows(sources.k, q_end, k_end)
        if sources.v is not sources.q:
            v = self._project_rows(sources.v, k_end, v_end)
        # K and V share one exchange of their convolution histories under CP.
        count = min(self.k_sconv.width - 1, sequence)
        tails = torch.cat((k[:, -count:], v[:, -count:]), dim=-1)
        k_history, v_history = convolution_prefix(tails, valid[:, -count:], self.cp_mesh, self.k_sconv.width).split(
            (k.shape[-1], v.shape[-1]), dim=-1
        )
        k = self.k_sconv(
            k, valid, dynamic_input=None if self.is_local else sources.k, positions=positions, prefix=k_history
        )
        v = self.v_sconv(
            v, valid, dynamic_input=None if self.is_local else sources.v, positions=positions, prefix=v_history
        )
        tau = (
            None
            if self.is_local
            else 1
            + self.log_scaling_alpha * torch.log(((positions.float() + 1) / self.log_scaling_n_floor).clamp(min=1))
        )
        q = self.q_l2_clip(q.view(batch, sequence, self.heads, self.head_dim), tau=tau)
        k = self.k_l2_clip(k.view(batch, sequence, self.kv_heads, self.head_dim))
        r = self.rel_l2_clip(r.view(batch, sequence, self.heads, self.d_rel), tau=tau)
        v = v.view(batch, sequence, self.kv_heads, self.head_dim)
        if self.value_norm:
            v = value_norm(v, self.eps)
        key_valid, key_positions = valid, positions
        query_offset = 0
        if self.cp_mesh is not None:
            # A query sees keys of its own document only, and a local query at most window - 1
            # of them, so only that many keys of the preceding ranks are exchanged, in whole
            # windows to keep the query offset block-aligned.
            reach = self.window - 1 if self.is_local else _preceding_reach(valid, positions, self.cp_mesh)
            halo = self.window * max(1, -(-reach // self.window))
            count = min(halo, sequence)
            kv_tail = torch.stack((k[:, -count:], v[:, -count:]), dim=2)
            kv_prefix = gather_preceding(kv_tail, self.cp_mesh, halo, differentiable=True)
            # A strided prefix makes the concatenation copy all of K or V through the slow strided path;
            # copying the short prefix first keeps the long copy on the contiguous one.
            k = torch.cat((kv_prefix[:, :, 0].contiguous(), k), dim=1)
            v = torch.cat((kv_prefix[:, :, 1].contiguous(), v), dim=1)
            key_positions, key_valid = _key_layout(valid, positions, self.cp_mesh, halo)
            query_offset = halo
        if self.triton and q.is_cuda:
            context = relative_attention(
                q,
                k,
                v,
                relative=r,
                table=self.relative_table,
                sink=sink,
                valid=valid,
                positions=positions,
                key_valid=key_valid,
                key_positions=key_positions,
                query_offset=query_offset,
                window=self.window if self.is_local else None,
            )
            return self._project_context(context, gate, valid)
        profiles = (r @ self.relative_table).transpose(1, 2)
        if self.flex is None:
            context = self._eager(
                q,
                k,
                v,
                profiles=profiles,
                sink=sink,
                valid=valid,
                positions=positions,
                key_valid=key_valid,
                key_positions=key_positions,
                query_offset=query_offset,
            )
        else:
            context = self.flex(
                q,
                k,
                v,
                profiles=profiles,
                sink=sink,
                valid=valid,
                key_valid=key_valid,
                positions=positions,
                key_positions=key_positions,
                query_offset=query_offset,
            )
        return self._project_context(context, gate, valid)

    def _project_context(self, context: torch.Tensor, gate: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Gate the attended heads and project them to the hidden size.

        Args:
            context: FP32 tensor of shape [batch, sequence, heads, head_dim].
            gate: Gate logits of shape [batch, sequence, heads * head_dim].
            valid: Boolean tensor of shape [batch, sequence], true for real tokens.

        Returns:
            Tensor of shape [batch, sequence, hidden]; padded queries give zero rows.
        """
        projected = self.wo_ud(gate_context(context, gate, valid))
        # The gated heads of padded queries are zero; only a projection bias would move them.
        return projected * valid.unsqueeze(-1) if self.o_bias else projected

    def _project_rows(self, source: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """Project a separate key or value input through its own rows of the fused weight.

        Args:
            source: Tensor of shape [batch, sequence, hidden].
            start: First row of the key or value block in the fused projection.
            end: Row after the block.

        Returns:
            Tensor of shape [batch, sequence, end - start]. A plain linear layer computes
            only these rows; a wrapped one (for example LoRA) runs whole and is sliced.
        """
        if type(self.qkvr) is not nn.Linear:
            return self.qkvr(source)[..., start:end]
        bias = None if self.qkvr.bias is None else self.qkvr.bias[start:end]
        return F.linear(source, self.qkvr.weight[start:end], bias)

    def _eager(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        profiles: torch.Tensor,
        sink: torch.Tensor,
        valid: torch.Tensor,
        positions: torch.Tensor,
        key_valid: torch.Tensor,
        key_positions: torch.Tensor,
        query_offset: int = 0,
    ) -> torch.Tensor:
        """Evaluate the quadratic reference without mutating inputs.

        Args:
            q: Tensor of shape [batch, sequence, heads, head_dim].
            k: Tensor of shape [batch, key_sequence, kv_heads, head_dim].
            v: Tensor of shape [batch, key_sequence, kv_heads, head_dim].
            profiles: Tensor of shape [batch, heads, sequence, relative_extent].
            sink: Tensor of shape [batch, sequence, heads].
            valid: Boolean tensor of shape [batch, sequence].
            positions: Integer logical positions [batch, local_sequence].
            key_valid: Boolean validity [batch, key_sequence], global under CP.
            key_positions: Integer logical positions [batch, key_sequence].
            query_offset: First local query's column in the global token stream.

        Returns:
            FP32 context of shape [batch, sequence, heads, head_dim].
        """
        batch, sequence = valid.shape
        # Broadcast GQA groups rather than gather repeated trainable projection rows.
        groups = self.heads // self.kv_heads
        k = k[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(batch, k.shape[1], self.heads, self.head_dim)
        v = v[:, :, :, None].expand(-1, -1, -1, groups, -1).reshape(batch, v.shape[1], self.heads, self.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float())
        distance = positions[:, :, None] - key_positions[:, None, :]
        profiles = profiles.float()
        offsets = distance.clamp(0, self.rel_extent - 1)[:, None].expand(-1, self.heads, -1, -1)
        relative = profiles.gather(-1, offsets)
        relative = relative.masked_fill((distance[:, None] < 0) | (distance[:, None] >= self.rel_extent), 0)
        # Only the QK product is scaled; the relative logits are already in the
        # softmax domain.
        scores = scores * self.head_dim**-0.5 + relative
        causal = torch.arange(k.shape[1], device=q.device)[None, :] <= (
            torch.arange(sequence, device=q.device)[:, None] + query_offset
        )
        query_origin = torch.arange(sequence, device=q.device)[None] + query_offset - positions
        key_origin = torch.arange(k.shape[1], device=q.device)[None] - key_positions
        same_document = query_origin[:, :, None] == key_origin[:, None, :]
        allowed = causal[None] & key_valid[:, None, :] & same_document
        if self.is_local:
            allowed = allowed & (distance >= 0) & (distance < self.window)
        scores = scores.masked_fill(~allowed[:, None], float("-inf"))
        # Softmax over real keys + a zero-value sink is equivalent to
        # softmax(scores) * sigmoid(logsumexp(scores) - sink), and is finite
        # even when a padded query has no valid keys.
        scores_with_sink = torch.cat((scores, sink.float().transpose(1, 2).unsqueeze(-1)), -1)
        probability = scores_with_sink.softmax(-1)[..., :-1]
        return torch.einsum("bhts,bshd->bthd", probability, v.float())

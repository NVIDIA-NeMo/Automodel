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

from typing import Literal, NamedTuple

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from .config import Inkling2TextConfig
from .context_parallel import gather_sequence
from .flex_attention import Inkling2FlexAttention
from .layers import L2Clip, ShortConvolution


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
        backend: Literal["eager", "flex"] = "eager",
    ) -> None:
        super().__init__()
        self.cp_mesh: DeviceMesh | None = None
        self.is_local = layer_idx in config.local_layer_ids
        self.heads = config.swa_num_attention_heads if self.is_local else config.num_attention_heads
        self.kv_heads = config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads
        self.head_dim = config.swa_head_dim if self.is_local else config.head_dim
        self.d_rel = config.d_rel
        self.rel_extent = config.sliding_window_size if self.is_local else config.rel_extent
        self.window = config.sliding_window_size
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
        if not self.is_local:
            # Only K/V rows use their independent depth-mixture inputs.
            q_end = self.split_sizes[0]
            k_end = q_end + self.split_sizes[1]
            v_end = k_end + self.split_sizes[2]
            # Invoke the module so LoRA updates and FSDP gather hooks also apply
            # to K/V, whose residual mixtures differ from the query input.
            k = self.qkvr(sources.k)[..., q_end:k_end]
            v = self.qkvr(sources.v)[..., k_end:v_end]
        k = self.k_sconv(k, valid, dynamic_input=None if self.is_local else sources.k, positions=positions)
        v = self.v_sconv(v, valid, dynamic_input=None if self.is_local else sources.v, positions=positions)
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
            value = v.float()
            v = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)).to(v.dtype)
        key_valid, key_positions = valid, positions
        query_offset = 0
        if self.cp_mesh is not None:
            k, v = gather_sequence(torch.stack((k, v), dim=2), self.cp_mesh, differentiable=True).unbind(2)
            layout = gather_sequence(torch.stack((positions, valid.long()), dim=-1), self.cp_mesh, differentiable=False)
            key_positions, key_valid = layout[..., 0], layout[..., 1].bool()
            query_offset = self.cp_mesh.get_local_rank() * sequence
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
        attended = context.flatten(2)
        attended = (attended * gate.float().sigmoid()).to(sources.q.dtype)
        return self.wo_ud(attended) * valid.unsqueeze(-1)

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

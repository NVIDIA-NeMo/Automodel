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

"""Native Inkling 2 text backbone for training.

The equations, including where values are computed in FP32 and cast back, follow the
Transformers ``inkling2`` reference (huggingface/transformers commit
109a7d19bf667168837f32929e00b791e195d14d, ``src/transformers/models/inkling2/modeling_inkling2.py``,
Apache-2.0). This module implements the training forward only: there is no decoding cache.

Sequence-position contract: the relative-position bias and the causal masks use the
token index inside each row, and the short convolutions mix neighbouring tokens of a row,
as in the reference. ``position_ids`` only drive the log scaling of the full-attention
layers. Packed rows holding several documents are therefore not separated.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import initialize_linear_module
from nemo_automodel.components.models.inkling.text import InklingRelativeLogits
from nemo_automodel.components.moe.config import MoEConfig

from .configuration import Inkling2TextConfig
from .layers import Inkling2MoE

_SUPPORTED_ATTENTION_BACKENDS = ("eager", "sdpa")


class Inkling2RMSNorm(nn.Module):
    """Weighted RMS normalization computed in FP32.

    The weight multiplies the FP32 normalized value before the cast back to the input
    dtype, as in the reference ``Inkling2RMSNorm``.
    """

    def __init__(self, hidden_size: int, eps: float, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last axis.

        Args:
            hidden_states: Tensor of shape ``[..., hidden]`` with arbitrary leading dimensions.

        Returns:
            Tensor with the shape and dtype of ``hidden_states``.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the learned scale to one."""
        nn.init.ones_(self.weight)


class Inkling2NormedEmbedding(nn.Embedding):
    """Token embedding followed by a weighted RMS normalization."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        norm_eps: float,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, dtype=dtype)
        self.embed_norm = Inkling2RMSNorm(embedding_dim, norm_eps, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed and normalize tokens.

        Args:
            input_ids: Long tensor of shape ``[batch, sequence]``.

        Returns:
            Tensor of shape ``[batch, sequence, hidden]`` in the embedding dtype.
        """
        return self.embed_norm(super().forward(input_ids))

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize the embedding table and its normalization."""
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        if self.padding_idx is not None:
            self.weight[self.padding_idx].zero_()
        self.embed_norm.init_weights()


def _unweighted_rms_norm(hidden_states: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS-normalize the last axis without a learned scale.

    Args:
        hidden_states: Tensor of shape ``[..., hidden]`` with arbitrary leading dimensions.
        eps: Variance epsilon.

    Returns:
        Tensor with the shape and dtype of ``hidden_states``; the statistic is FP32.
    """
    scale = torch.rsqrt(hidden_states.float().square().mean(-1, keepdim=True) + eps)
    return hidden_states * scale.to(hidden_states.dtype)


def _l2_clip(hidden_states: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scale each last-axis vector down to ``threshold`` when its L2 norm is larger.

    Args:
        hidden_states: FP32 tensor of shape ``[..., head_dim]`` with arbitrary leading dimensions.
        threshold: Maximum L2 norm.

    Returns:
        FP32 tensor with the shape of ``hidden_states``.
    """
    norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
    return hidden_states * (threshold / (norm + 1e-6)).clamp(max=1.0)


class Inkling2ScaledL2Clip(nn.Module):
    """Per-head learned scale followed by an L2-norm clip, computed in FP32."""

    def __init__(self, num_heads: int, head_dim: int, threshold: float, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.threshold = threshold
        self.scale = nn.Parameter(torch.ones(num_heads, head_dim, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Scale and clip head vectors.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, heads, head_dim]``.

        Returns:
            Tensor with the shape and dtype of ``hidden_states``.
        """
        input_dtype = hidden_states.dtype
        return _l2_clip(hidden_states.float() * self.scale.float(), self.threshold).to(input_dtype)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the scale to one."""
        nn.init.ones_(self.scale)


def _mask_padding_states(hidden_states: torch.Tensor, conv_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero padded tokens before a short convolution.

    Args:
        hidden_states: Tensor of shape ``[batch, sequence, channels]``.
        conv_mask: Optional tensor of shape ``[batch, sequence]``, nonzero for valid tokens.

    Returns:
        Tensor of shape ``[batch, sequence, channels]`` in the input dtype; aliases the input
        when ``conv_mask`` is ``None``.
    """
    if conv_mask is None:
        return hidden_states
    return (hidden_states * conv_mask[:, :, None]).to(hidden_states.dtype)


class Inkling2ShortConvolution(nn.Module):
    """Causal depthwise convolution with a scaled residual, computed in FP32.

    Output is ``(conv(x) + x) / sqrt(kernel + 1)``. ``conv1d.weight`` (shape
    ``[channels, 1, kernel]``) stays FP32.
    """

    def __init__(self, hidden_size: int, conv_kernel_size: int) -> None:
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=conv_kernel_size,
            groups=hidden_size,
            padding=conv_kernel_size - 1,
            bias=False,
            dtype=torch.float32,
        )

    def forward(self, hidden_states: torch.Tensor, conv_mask: torch.Tensor | None) -> torch.Tensor:
        """Apply the convolution over the sequence axis.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, channels]``.
            conv_mask: Optional tensor of shape ``[batch, sequence]``, nonzero for valid tokens;
                padded tokens are zeroed before the convolution but kept in the residual.

        Returns:
            Tensor of shape ``[batch, sequence, channels]`` in the input dtype.
        """
        input_dtype = hidden_states.dtype
        residual = hidden_states.float()
        sequence = residual.shape[1]
        context = _mask_padding_states(residual, conv_mask).transpose(1, 2)
        convolved = self.conv1d(context)[:, :, :sequence].transpose(1, 2)
        return ((convolved + residual) / math.sqrt(self.conv_kernel_size + 1)).to(input_dtype)

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize the depthwise taps."""
        nn.init.normal_(self.conv1d.weight, mean=0.0, std=init_std)


class Inkling2DynamicShortConvolution(nn.Module):
    """Causal convolution whose taps add static per-channel and per-token head weights.

    Tap ``lag`` weights the token ``lag`` steps back. The taps are ``static_weight[lag]``
    (per channel) plus ``weight_proj(dynamic_input)`` (per token, shared within a key/value
    head). Output is ``(conv(x) + x) / sqrt(2 * kernel + 1)`` in FP32. Used for keys and
    values of the full-attention layers.
    """

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        num_heads: int,
        conv_kernel_size: int,
        backend: BackendConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.conv_kernel_size = conv_kernel_size
        self.num_heads = num_heads
        self.static_weight = nn.Parameter(torch.zeros(conv_kernel_size, hidden_size, dtype=dtype))
        self.weight_proj = initialize_linear_module(
            backend.linear, input_size, num_heads * conv_kernel_size, bias=False, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        dynamic_input: torch.Tensor,
        conv_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the dynamic convolution over the sequence axis.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, channels]``, with
                ``channels = num_heads * head_dim``.
            dynamic_input: Tensor of shape ``[batch, sequence, input_size]`` that produces
                the per-token taps.
            conv_mask: Optional tensor of shape ``[batch, sequence]``, nonzero for valid tokens.

        Returns:
            Tensor of shape ``[batch, sequence, channels]`` in the dtype of ``hidden_states``.
        """
        input_dtype = hidden_states.dtype
        batch, sequence, channels = hidden_states.shape
        kernel = self.conv_kernel_size
        dynamic_weights = self.weight_proj(dynamic_input).float()
        dynamic_weights = dynamic_weights.view(batch, sequence, self.num_heads, 1, kernel)
        static_weights = self.static_weight.float().T.view(self.num_heads, -1, kernel)
        weights = static_weights + dynamic_weights

        residual = hidden_states.float()
        context = _mask_padding_states(residual, conv_mask).transpose(1, 2)
        context = F.pad(context, (kernel - 1, 0))
        # windows[..., lag] is the token ``lag`` steps back: [batch, sequence, channels, kernel].
        windows = context.unfold(-1, kernel, 1).flip(-1).transpose(1, 2)
        windows = windows.reshape(batch, sequence, self.num_heads, -1, kernel)
        convolved = (windows * weights).sum(-1).reshape(batch, sequence, channels)
        return ((convolved + residual) / math.sqrt(2 * kernel + 1)).to(input_dtype)

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Zero the static taps and initialize the tap projection."""
        nn.init.zeros_(self.static_weight)
        nn.init.normal_(self.weight_proj.weight, mean=0.0, std=init_std)


class Inkling2BlockAttentionResidual(nn.Module):
    """Attention over depth: learned queries mix the block states and the running state.

    Keys and values are the unweighted-RMS-normalized states, all in FP32. ``query`` (FP32,
    shape ``[hidden]`` or ``[num_queries, hidden]``) is the same at every position.
    """

    def __init__(self, hidden_size: int, num_queries: int) -> None:
        super().__init__()
        shape = (hidden_size,) if num_queries == 1 else (num_queries, hidden_size)
        self.query = nn.Parameter(torch.zeros(shape, dtype=torch.float32))

    def forward(self, block_states: torch.Tensor, partial_block_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Mix the depth axis.

        Args:
            block_states: Tensor of shape ``[batch, sequence, blocks, hidden]`` holding completed
                block states; ``blocks`` may be zero.
            partial_block_states: Tensor of shape ``[batch, sequence, hidden]``, the running state.

        Returns:
            ``num_queries`` FP32 tensors of shape ``[batch, sequence, hidden]``.
        """
        stacked_states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
        stacked_states = _unweighted_rms_norm(stacked_states, 1e-6)
        hidden_size = stacked_states.shape[-1]
        query_states = self.query.float().view(-1, hidden_size).expand(*stacked_states.shape[:-2], -1, -1)
        mixed = F.scaled_dot_product_attention(query_states, stacked_states, stacked_states, scale=hidden_size**-0.5)
        return mixed.unbind(-2)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Zero the queries, so every state starts with equal weight."""
        nn.init.zeros_(self.query)


def _repeat_key_value(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat grouped key/value heads to match the query heads.

    Args:
        hidden_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
        repeats: Number of query heads sharing each key/value head.

    Returns:
        Tensor of shape ``[batch, key_value_heads * repeats, sequence, head_dim]``.
    """
    if repeats == 1:
        return hidden_states
    batch, key_value_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, key_value_heads, repeats, sequence, head_dim)
    return expanded.reshape(batch, key_value_heads * repeats, sequence, head_dim)


class Inkling2Attention(nn.Module):
    """Inkling 2 grouped-query attention for sliding-window and full-attention layers.

    Queries and relative states get a learned scale and an L2 clip, keys an L2 clip, values an
    unweighted RMS norm. Keys and values pass through short convolutions (static on sliding
    layers, dynamic on full layers). The logits add a token-conditioned relative-position bias
    and one zero-value sink logit per query; the output is gated by ``sigmoid(gate_proj(x))``.
    Full layers also scale queries and relative states by ``1 + alpha * log(max(1, (p + 1) / n_floor))``.
    """

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        backend: BackendConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if backend.attn not in _SUPPORTED_ATTENTION_BACKENDS:
            raise ValueError(
                f"Inkling 2 attention supports backend.attn in {_SUPPORTED_ATTENTION_BACKENDS}, got {backend.attn!r}"
            )
        self.attention_backend = backend.attn
        self.config = config
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        self.num_key_value_heads = config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rel_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        hidden_size = config.hidden_size
        query_size = self.num_heads * self.head_dim
        key_value_size = self.num_key_value_heads * self.head_dim

        def linear(in_features: int, out_features: int) -> nn.Module:
            return initialize_linear_module(backend.linear, in_features, out_features, bias=False, dtype=dtype)

        self.q_proj = linear(hidden_size, query_size)
        self.k_proj = linear(hidden_size, key_value_size)
        self.v_proj = linear(hidden_size, key_value_size)
        self.r_proj = linear(hidden_size, self.num_heads * config.d_rel)
        self.o_proj = linear(query_size, hidden_size)
        self.gate_proj = linear(hidden_size, query_size)
        self.sink_proj = linear(hidden_size, self.num_heads)
        if self.is_sliding:
            self.k_sconv = Inkling2ShortConvolution(key_value_size, config.conv_kernel_size)
            self.v_sconv = Inkling2ShortConvolution(key_value_size, config.conv_kernel_size)
        else:
            self.k_sconv = Inkling2DynamicShortConvolution(
                key_value_size, hidden_size, self.num_key_value_heads, config.conv_kernel_size, backend, dtype=dtype
            )
            self.v_sconv = Inkling2DynamicShortConvolution(
                key_value_size, hidden_size, self.num_key_value_heads, config.conv_kernel_size, backend, dtype=dtype
            )
        self.rel_logits_proj = InklingRelativeLogits(config.d_rel, self.rel_extent, dtype=dtype)
        self.q_l2_clip = Inkling2ScaledL2Clip(self.num_heads, self.head_dim, config.qk_l2_clip_threshold, dtype=dtype)
        self.rel_l2_clip = Inkling2ScaledL2Clip(self.num_heads, config.d_rel, config.rel_l2_clip_threshold, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_hidden_states: torch.Tensor,
        value_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conv_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply self-attention.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]`` feeding the query,
                relative-state, gate and sink projections.
            key_hidden_states: Tensor of shape ``[batch, sequence, hidden]`` feeding the key
                projection (and the key convolution taps on full layers). Sliding layers pass
                ``hidden_states``.
            value_hidden_states: Tensor of shape ``[batch, sequence, hidden]`` for the values,
                as for keys.
            attention_mask: Boolean tensor of shape ``[batch or 1, 1, sequence, sequence]``,
                true where the query (axis 2) may attend to the key (axis 3).
            conv_mask: Optional tensor of shape ``[batch, sequence]``, nonzero for valid tokens.
            position_ids: Long tensor of shape ``[batch or 1, sequence]`` for the log scaling
                of full layers.

        Returns:
            Tensor of shape ``[batch, sequence, hidden]``.
        """
        batch, sequence = hidden_states.shape[:2]
        head_shape = (batch, sequence, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        if self.is_sliding:
            # The reference rounds the convolution output to the projection dtype.
            key_states = self.k_sconv(self.k_proj(key_hidden_states), conv_mask).to(query_states.dtype)
            value_states = self.v_sconv(self.v_proj(value_hidden_states), conv_mask).to(query_states.dtype)
        else:
            key_states = self.k_sconv(self.k_proj(key_hidden_states), key_hidden_states, conv_mask)
            value_states = self.v_sconv(self.v_proj(value_hidden_states), value_hidden_states, conv_mask)
        relative_states = self.r_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        sinks = self.sink_proj(hidden_states)

        query_states = self.q_l2_clip(query_states.view(head_shape))
        relative_states = self.rel_l2_clip(relative_states.view(batch, sequence, self.num_heads, -1))
        if not self.is_sliding:
            effective_positions = (position_ids + 1).float() / self.config.log_scaling_n_floor
            tau = 1.0 + self.config.log_scaling_alpha * torch.log(effective_positions.clamp(min=1.0))
            tau = tau[..., None, None]
            query_states = (query_states.float() * tau).to(query_states.dtype)
            relative_states = (relative_states.float() * tau).to(relative_states.dtype)
        query_states = query_states.transpose(1, 2)
        # Keys and values return to the projection dtype. FP32 FSDP units around the static
        # convolutions cast their inputs to FP32, so the convolution output dtype alone is not it.
        compute_dtype = query_states.dtype
        key_states = _l2_clip(key_states.view(head_shape).float(), self.config.qk_l2_clip_threshold)
        key_states = key_states.to(compute_dtype).transpose(1, 2)
        value_states = _unweighted_rms_norm(value_states.view(head_shape).float(), self.config.rms_norm_eps)
        value_states = value_states.to(compute_dtype).transpose(1, 2)

        positions = torch.arange(sequence, device=hidden_states.device)
        position_bias = self.rel_logits_proj(relative_states, positions, positions)
        # One extra logit per query and head: [batch, heads, sequence, 1].
        sink_logits = sinks.transpose(1, 2).unsqueeze(-1)
        dropout = self.attention_dropout if self.training else 0.0
        if self.attention_backend == "sdpa":
            attention_output = self._sdpa_attention(
                query_states, key_states, value_states, position_bias, sink_logits, attention_mask, dropout
            )
        else:
            attention_output = self._eager_attention(
                query_states, key_states, value_states, position_bias, sink_logits, attention_mask, dropout
            )

        attention_output = attention_output.transpose(1, 2).reshape(batch, sequence, -1)
        attention_output = (attention_output.float() * torch.sigmoid(gate.float())).to(attention_output.dtype)
        return self.o_proj(attention_output)

    def _eager_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_bias: torch.Tensor,
        sink_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        dropout: float,
    ) -> torch.Tensor:
        """Dense attention matching the reference eager path.

        Args:
            query_states: Tensor of shape ``[batch, heads, sequence, head_dim]``.
            key_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
            value_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
            position_bias: Tensor of shape ``[batch, heads, sequence, sequence]``.
            sink_logits: Tensor of shape ``[batch, heads, sequence, 1]``.
            attention_mask: Boolean tensor of shape ``[batch or 1, 1, sequence, sequence]``.
            dropout: Attention-probability dropout rate.

        Returns:
            Tensor of shape ``[batch, heads, sequence, head_dim]``.
        """
        key_states = _repeat_key_value(key_states, self.num_key_value_groups)
        value_states = _repeat_key_value(value_states, self.num_key_value_groups)
        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling + position_bias
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
        # The sink is one extra logit with a zero value: it only absorbs probability mass.
        logits = torch.cat([scores, sink_logits.to(scores.dtype)], dim=-1).float()
        probabilities = F.softmax(logits, dim=-1)[..., :-1].to(query_states.dtype)
        probabilities = F.dropout(probabilities, p=dropout, training=self.training)
        return torch.matmul(probabilities, value_states)

    def _sdpa_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_bias: torch.Tensor,
        sink_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        dropout: float,
    ) -> torch.Tensor:
        """SDPA with the sink as an extra all-zero key/value slot.

        A zero key gives a zero dot product, so the appended mask column makes that slot's logit
        equal the sink logit; its zero value adds nothing to the output.

        Args:
            query_states: Tensor of shape ``[batch, heads, sequence, head_dim]``.
            key_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
            value_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
            position_bias: Tensor of shape ``[batch, heads, sequence, sequence]``.
            sink_logits: Tensor of shape ``[batch, heads, sequence, 1]``.
            attention_mask: Boolean tensor of shape ``[batch or 1, 1, sequence, sequence]``.
            dropout: Attention-probability dropout rate.

        Returns:
            Tensor of shape ``[batch, heads, sequence, head_dim]``.
        """
        batch, key_value_heads, _, head_dim = key_states.shape
        sink_slot = key_states.new_zeros(batch, key_value_heads, 1, head_dim)
        key_states = _repeat_key_value(torch.cat([key_states, sink_slot], dim=2), self.num_key_value_groups)
        value_states = _repeat_key_value(torch.cat([value_states, sink_slot], dim=2), self.num_key_value_groups)
        bias = position_bias.to(query_states.dtype).masked_fill(~attention_mask, float("-inf"))
        bias = torch.cat([bias, sink_logits.to(bias.dtype)], dim=-1)
        return F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=bias, dropout_p=dropout, scale=self.scaling
        )

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize projections, clips, relative logits and convolutions."""
        for projection in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.r_proj,
            self.o_proj,
            self.gate_proj,
            self.sink_proj,
        ):
            nn.init.normal_(projection.weight, mean=0.0, std=init_std)
        self.rel_logits_proj.init_weights(init_std)
        self.q_l2_clip.init_weights()
        self.rel_l2_clip.init_weights()
        self.k_sconv.init_weights(init_std)
        self.v_sconv.init_weights(init_std)


class _Inkling2DecoderLayer(nn.Module):
    """Shared attention-residual and MoE half of the Inkling 2 decoder layers."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_block_boundary = layer_idx % config.n_layers_per_attn_res_block == 0
        self.mlp_input = Inkling2BlockAttentionResidual(config.hidden_size, num_queries=1)
        self.self_attn = Inkling2Attention(config, layer_idx, backend, dtype=dtype)
        self.mlp = Inkling2MoE(config, backend, moe_config)
        self.post_attention_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype)
        self.post_attention_convolution = Inkling2ShortConvolution(config.hidden_size, config.conv_kernel_size)
        self.post_mlp_convolution = Inkling2ShortConvolution(config.hidden_size, config.conv_kernel_size)

    def _mlp_block(
        self,
        attention_output: torch.Tensor,
        block_residual: torch.Tensor,
        block_states: torch.Tensor,
        conv_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Close the attention residual and run the MoE half of the layer.

        Args:
            attention_output: Tensor of shape ``[batch, sequence, hidden]``.
            block_residual: Layer input of shape ``[batch, sequence, hidden]`` in the model dtype.
            block_states: Tensor of shape ``[batch, sequence, blocks, hidden]``; not modified.
            conv_mask: Optional tensor of shape ``[batch, sequence]``, nonzero for valid tokens.
            padding_mask: Optional boolean tensor of shape ``[batch, sequence]``, true for
                padding tokens, which skip expert dispatch.

        Returns:
            Tensor of shape ``[batch, sequence, hidden]`` in the dtype of ``block_residual``.
        """
        dtype = block_residual.dtype
        # The convolutions round to the model dtype as in the reference, also when an FP32 FSDP
        # unit around them has cast their inputs to FP32.
        hidden_states = self.post_attention_convolution(attention_output, conv_mask).to(dtype)
        residual = hidden_states = hidden_states.float()
        if self.is_block_boundary:
            # A block boundary stores the layer input and restarts the running residual.
            block_states = torch.cat([block_states, block_residual.unsqueeze(-2)], dim=-2)
        else:
            residual = hidden_states = block_residual.float() + hidden_states
        hidden_states = self.mlp_input(block_states, hidden_states)[0]
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states).to(dtype), padding_mask)
        hidden_states = self.post_mlp_convolution(hidden_states, conv_mask).to(dtype)
        return (residual + hidden_states).to(dtype)

    @torch.no_grad()
    def _init_shared_weights(self, init_std: float, buffer_device: torch.device) -> None:
        """Initialize the modules shared by both layer kinds."""
        self.mlp_input.init_weights()
        self.self_attn.init_weights(init_std)
        self.post_attention_layernorm.init_weights()
        self.post_attention_convolution.init_weights(init_std)
        self.post_mlp_convolution.init_weights(init_std)
        self.mlp.init_weights(buffer_device, init_std)


class Inkling2SlidingDecoderLayer(_Inkling2DecoderLayer):
    """Decoder layer with sliding-window attention on one mixed input."""

    attention_type = "sliding_attention"

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config, layer_idx, backend, moe_config, dtype=dtype)
        self.attention_input = (
            Inkling2BlockAttentionResidual(config.hidden_size, num_queries=1) if layer_idx > 0 else None
        )
        self.input_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conv_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the decoder layer.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]`` in the model dtype.
            block_states: Tensor of shape ``[batch, sequence, blocks, hidden]``; not modified.
            attention_mask: Boolean sliding-window causal mask of shape
                ``[batch or 1, 1, sequence, sequence]``.
            conv_mask: Optional tensor of shape ``[batch, sequence]``, nonzero for valid tokens.
            padding_mask: Optional boolean tensor of shape ``[batch, sequence]``, true for padding.
            position_ids: Long tensor of shape ``[batch or 1, sequence]``.

        Returns:
            Tensor of shape ``[batch, sequence, hidden]`` in the model dtype.
        """
        dtype = hidden_states.dtype
        block_residual = hidden_states
        if self.attention_input is not None:
            hidden_states = self.attention_input(block_states, hidden_states)[0]
        normalized = self.input_layernorm(hidden_states).to(dtype)
        attention_output = self.self_attn(normalized, normalized, normalized, attention_mask, conv_mask, position_ids)
        return self._mlp_block(attention_output, block_residual, block_states, conv_mask, padding_mask)

    @torch.no_grad()
    def init_weights(self, init_std: float, buffer_device: torch.device) -> None:
        """Initialize every parameter of the layer."""
        self._init_shared_weights(init_std, buffer_device)
        if self.attention_input is not None:
            self.attention_input.init_weights()
        self.input_layernorm.init_weights()


class Inkling2FullDecoderLayer(_Inkling2DecoderLayer):
    """Decoder layer with full attention whose query, key and value inputs are mixed separately."""

    attention_type = "full_attention"

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(config, layer_idx, backend, moe_config, dtype=dtype)
        self.attention_input = (
            Inkling2BlockAttentionResidual(config.hidden_size, num_queries=3) if layer_idx > 0 else None
        )
        self.q_input_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype)
        self.k_input_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype)
        self.v_input_layernorm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conv_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the decoder layer.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]`` in the model dtype.
            block_states: Tensor of shape ``[batch, sequence, blocks, hidden]``; not modified.
            attention_mask: Boolean causal mask of shape ``[batch or 1, 1, sequence, sequence]``.
            conv_mask: Optional tensor of shape ``[batch, sequence]``, nonzero for valid tokens.
            padding_mask: Optional boolean tensor of shape ``[batch, sequence]``, true for padding.
            position_ids: Long tensor of shape ``[batch or 1, sequence]``.

        Returns:
            Tensor of shape ``[batch, sequence, hidden]`` in the model dtype.
        """
        dtype = hidden_states.dtype
        block_residual = hidden_states
        if self.attention_input is None:
            query_inputs = key_inputs = value_inputs = hidden_states
        else:
            query_inputs, key_inputs, value_inputs = self.attention_input(block_states, hidden_states)
        attention_output = self.self_attn(
            self.q_input_layernorm(query_inputs).to(dtype),
            self.k_input_layernorm(key_inputs).to(dtype),
            self.v_input_layernorm(value_inputs).to(dtype),
            attention_mask,
            conv_mask,
            position_ids,
        )
        return self._mlp_block(attention_output, block_residual, block_states, conv_mask, padding_mask)

    @torch.no_grad()
    def init_weights(self, init_std: float, buffer_device: torch.device) -> None:
        """Initialize every parameter of the layer."""
        self._init_shared_weights(init_std, buffer_device)
        if self.attention_input is not None:
            self.attention_input.init_weights()
        self.q_input_layernorm.init_weights()
        self.k_input_layernorm.init_weights()
        self.v_input_layernorm.init_weights()


def _causal_masks(
    attention_mask: torch.Tensor | None,
    sequence: int,
    sliding_window: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build boolean full and sliding-window causal masks.

    Args:
        attention_mask: Optional tensor of shape ``[batch, sequence]``, nonzero for valid keys.
        sequence: Sequence length.
        sliding_window: Number of visible tokens, the query included, in sliding layers.
        device: Target device.

    Returns:
        Full and sliding masks, boolean tensors of shape ``[batch or 1, 1, sequence, sequence]``,
        true where the query (axis 2) may attend to the key (axis 3).
    """
    positions = torch.arange(sequence, device=device)
    full = positions[None, :] <= positions[:, None]
    sliding = full & (positions[None, :] > positions[:, None] - sliding_window)
    full, sliding = full[None, None], sliding[None, None]
    if attention_mask is not None:
        key_valid = attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :]
        full, sliding = full & key_valid, sliding & key_valid
    return full, sliding


class Inkling2TextModel(nn.Module):
    """Native Inkling 2 text backbone (embedding, decoder layers, final mixing and norm)."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        backend: BackendConfig,
        moe_config: MoEConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = Inkling2NormedEmbedding(
            config.vocab_size, config.hidden_size, config.pad_token_id, config.rms_norm_eps, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                Inkling2SlidingDecoderLayer(config, layer_idx, backend, moe_config, dtype=dtype)
                if layer_type == "hybrid_sliding"
                else Inkling2FullDecoderLayer(config, layer_idx, backend, moe_config, dtype=dtype)
                for layer_idx, layer_type in enumerate(config.layer_types)
            ]
        )
        self.final_input = Inkling2BlockAttentionResidual(config.hidden_size, num_queries=1)
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype)

    def get_input_embeddings(self) -> nn.Module:
        """Return the normalized token embedding."""
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> BaseModelOutput:
        """Run the text backbone.

        Args:
            input_ids: Optional long tensor of shape ``[batch, sequence]``.
            inputs_embeds: Optional tensor of shape ``[batch, sequence, hidden]``, already
                normalized like the output of ``embed_tokens``. Exactly one of ``input_ids``
                and ``inputs_embeds`` is required.
            attention_mask: Optional padding tensor of shape ``[batch, sequence]``, nonzero for
                valid tokens.
            position_ids: Optional long tensor of shape ``[batch or 1, sequence]``; defaults to
                ``0 .. sequence - 1``. Used only by the log scaling of full layers.

        Returns:
            Output whose ``last_hidden_state`` has shape ``[batch, sequence, hidden]`` in the
            dtype of the embeddings.

        Raises:
            ValueError: If both or neither of ``input_ids`` and ``inputs_embeds`` are given, or
                ``attention_mask`` is not a two-dimensional padding mask.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, sequence = inputs_embeds.shape[:2]
        if attention_mask is not None and attention_mask.shape != (batch, sequence):
            raise ValueError(
                f"Inkling 2 expects a [batch, sequence] padding mask, got shape {tuple(attention_mask.shape)}"
            )
        if position_ids is None:
            position_ids = torch.arange(sequence, device=inputs_embeds.device)[None, :]

        full_mask, sliding_mask = _causal_masks(
            attention_mask, sequence, self.config.sliding_window_size, inputs_embeds.device
        )
        padding_mask = None if attention_mask is None else attention_mask == 0

        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(batch, sequence, 0, inputs_embeds.shape[-1])
        for decoder_layer in self.layers:
            layer_mask = sliding_mask if decoder_layer.attention_type == "sliding_attention" else full_mask
            new_hidden_states = decoder_layer(
                hidden_states, block_states, layer_mask, attention_mask, padding_mask, position_ids
            )
            if decoder_layer.is_block_boundary:
                block_states = torch.cat([block_states, hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_hidden_states

        hidden_states = self.final_input(block_states, hidden_states)[0]
        hidden_states = self.norm(hidden_states).to(inputs_embeds.dtype)
        return BaseModelOutput(last_hidden_state=hidden_states)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        """Initialize the backbone for checkpoint-free construction."""
        init_std = self.config.initializer_range
        self.embed_tokens.init_weights(init_std)
        for layer in self.layers:
            layer.init_weights(init_std, buffer_device)
        self.final_input.init_weights()
        self.norm.init_weights()


__all__ = [
    "Inkling2Attention",
    "Inkling2FullDecoderLayer",
    "Inkling2RMSNorm",
    "Inkling2SlidingDecoderLayer",
    "Inkling2TextModel",
]

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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import (
    _has_dtensor_params,
    cast_model_to_dtype,
    compute_lm_head_logits,
)
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2Config, MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.cp import (
    _MIMO_GLOBAL_IMAGE_MASK,
    _MIMO_GLOBAL_VIDEO_MASK,
    _MIMO_THD_LOCAL_INDICES,
    make_mimo_te_cp_sharder,
)
from nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter import MiMoV2FlashStateDictAdapter
from nemo_automodel.components.models.mimo_v2_flash.vision import MiMoVisionTransformer
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _convert_bool_4d_mask_to_additive(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.ndim != 4 or attention_mask.dtype != torch.bool:
        return attention_mask
    additive = torch.zeros(attention_mask.shape, dtype=dtype, device=attention_mask.device)
    return additive.masked_fill(~attention_mask, torch.finfo(dtype).min)


def _derive_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim == 2:
        return attention_mask == 0
    if attention_mask.ndim == 4:
        diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
        if attention_mask.dtype == torch.bool:
            return diagonal.logical_not()
        return diagonal != 0
    return attention_mask.bool().logical_not()


def _intersect_with_sliding_window(
    attention_mask: torch.Tensor,
    *,
    sliding_window: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply causal sliding-window bounds to an existing 4D document mask."""
    mask = _convert_bool_4d_mask_to_additive(attention_mask, dtype)
    if not mask.dtype.is_floating_point:
        mask = _convert_bool_4d_mask_to_additive(mask.bool(), dtype)
    query_len, key_len = mask.shape[-2:]
    query_positions = torch.arange(key_len - query_len, key_len, device=mask.device)
    key_positions = torch.arange(key_len, device=mask.device)
    outside_window = (key_positions[None, :] > query_positions[:, None]) | (
        (query_positions[:, None] - key_positions[None, :]) >= sliding_window
    )
    return mask.masked_fill(
        outside_window.unsqueeze(0).unsqueeze(0),
        torch.finfo(dtype).min,
    )


def _replace_modal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    token_id: int | None,
    modal_embeds: torch.Tensor | None,
) -> torch.Tensor:
    """Replace one kind of multimodal placeholder while preserving gradients."""
    if token_id is None or modal_embeds is None:
        return inputs_embeds
    if modal_embeds.ndim != 2:
        raise ValueError(f"modal_embeds must be rank two [N, H], got {tuple(modal_embeds.shape)}")
    mask = input_ids.eq(token_id)
    num_slots = int(mask.sum().item())
    if num_slots != modal_embeds.shape[0]:
        raise ValueError(
            f"Modal embedding count mismatch for token_id={token_id}: "
            f"found {num_slots} placeholders but got {modal_embeds.shape[0]} embeddings"
        )
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[mask] = modal_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return inputs_embeds


def _validate_te_thd_sink_cudnn(head_dim: int) -> None:
    """Reject the known pre-9.26 cuDNN ragged dSink kernel bug."""
    if head_dim in (64, 128, 256):
        return
    cudnn_version = torch.backends.cudnn.version()
    if cudnn_version is None or cudnn_version < 92600:
        detected = "unavailable" if cudnn_version is None else str(cudnn_version)
        raise RuntimeError(
            "MiMo TE THD training with learnable attention sinks requires cuDNN >= 9.26 "
            "when the a2a-compatible padded value head uses a generic kernel "
            f"(head_dim={head_dim}); detected cuDNN {detected}. Older cuDNN versions "
            "can illegally access memory in fused-attention backward "
            "(NVIDIA/TransformerEngine#3249)."
        )


def _normalize_image_grid(grid: torch.Tensor | None) -> torch.Tensor | None:
    """Accept the THW grids emitted by Qwen processors and PP's HW alias."""
    if grid is None or grid.shape[-1] == 3:
        return grid
    if grid.shape[-1] != 2:
        raise ValueError(f"Image grid must have two or three columns, got {tuple(grid.shape)}")
    temporal = torch.ones((grid.shape[0], 1), dtype=grid.dtype, device=grid.device)
    return torch.cat((temporal, grid), dim=-1)


def _fallback_additive_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    idx = torch.arange(seq_len, device=device)
    masked = idx.unsqueeze(0) > idx.unsqueeze(1)
    if sliding_window is not None and sliding_window > 0:
        masked = masked | ((idx.unsqueeze(1) - idx.unsqueeze(0)) >= sliding_window)
    additive = torch.zeros((seq_len, seq_len), dtype=dtype, device=device).masked_fill(masked, min_value)
    additive = additive.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).contiguous()
    if attention_mask is not None and attention_mask.ndim == 2:
        pad_add = (1.0 - attention_mask.to(dtype=dtype, device=device)).unsqueeze(1).unsqueeze(2) * min_value
        additive = additive + pad_add
    return additive


def _ensure_additive_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None,
    sliding_window: int | None,
) -> torch.Tensor:
    if mask is None or not isinstance(mask, torch.Tensor):
        return _fallback_additive_mask(batch_size, seq_len, dtype, device, attention_mask, sliding_window)
    return _convert_bool_4d_mask_to_additive(mask, dtype)


def _eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    if module.attention_sink_bias is not None:
        sinks = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        attn_weights = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)

    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if module.attention_sink_bias is not None:
        probs = probs[..., :-1]

    probs = F.dropout(probs, p=dropout, training=module.training)
    attn_output = torch.matmul(probs, value_states)
    return attn_output.transpose(1, 2).contiguous(), probs


class MiMoV2FlashRotaryEmbedding(nn.Module):
    """Rotary embedding module matching MiMo-V2-Flash partial-RoPE behavior."""

    inv_freq: torch.Tensor

    def __init__(
        self,
        *,
        rope_theta: float,
        head_dim: int,
        partial_rotary_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        rotary_dim = int(head_dim * partial_rotary_factor)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        if rotary_dim <= 0:
            raise ValueError(f"Invalid rotary_dim={rotary_dim} for head_dim={head_dim}")
        self.rope_theta = rope_theta
        self.rotary_dim = rotary_dim
        inv_freq = self._build_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._inv_freq_initialized = not inv_freq.is_meta
        self.attention_scaling = 1.0

    def _build_inv_freq(self, device: torch.device | None = None) -> torch.Tensor:
        """Build the FP32 inverse frequencies on the requested device."""
        positions = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device)
        return 1.0 / (self.rope_theta ** (positions / self.rotary_dim))

    @torch.no_grad()
    def _materialize_inv_freq(self, device: torch.device) -> None:
        """Restore a non-persistent RoPE buffer after meta-model materialization."""
        self.inv_freq = self._build_inv_freq(device)
        self._inv_freq_initialized = True

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._inv_freq_initialized or self.inv_freq.is_meta or self.inv_freq.device != x.device:
            self._materialize_inv_freq(x.device)
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MiMoV2RMSNorm(nn.Module):
    """RMSNorm used by MiMo-V2-Flash decoder blocks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _initialize_mimo_rms_norm(
    backend: BackendConfig,
    hidden_size: int,
    *,
    eps: float,
    dtype: torch.dtype,
) -> nn.Module:
    """Construct a MiMo RMSNorm from the configured backend."""
    if backend.rms_norm == "torch_fp32":
        return MiMoV2RMSNorm(hidden_size, eps=eps, dtype=dtype)
    return initialize_rms_norm_module(backend.rms_norm, hidden_size, eps=eps, dtype=dtype)


class MiMoV2FlashAttention(nn.Module):
    """MiMo-V2 attention with a Transformer Engine THD context-parallel path."""

    def __init__(self, config: MiMoV2FlashConfig, backend: BackendConfig, is_swa: bool, layer_idx: int):
        super().__init__()
        self.config = config
        self.backend = backend
        self.layer_idx = layer_idx
        self.is_swa = is_swa

        if is_swa:
            self.head_dim = int(config.swa_head_dim)
            self.v_head_dim = int(config.swa_v_head_dim)
            self.num_attention_heads = int(config.swa_num_attention_heads)
            self.num_key_value_heads = int(config.swa_num_key_value_heads)
        else:
            self.head_dim = int(config.head_dim)
            self.v_head_dim = int(config.v_head_dim)
            self.num_attention_heads = int(config.num_attention_heads)
            self.num_key_value_heads = int(config.num_key_value_heads)

        self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rope_dim -= self.rope_dim % 2
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = float(config.attention_dropout or 0.0)
        self.scaling = self.head_dim**-0.5
        self.v_scale = getattr(config, "attention_value_scale", None)
        self.sliding_window = int(config.sliding_window) if is_swa else None

        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.q_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.k_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.v_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_key_value_heads * self.v_head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            backend.linear,
            self.num_attention_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            dtype=dtype,
        )

        has_sink = (config.add_full_attention_sink_bias and not is_swa) or (
            config.add_swa_attention_sink_bias and is_swa
        )
        self.attn_module = None
        self.attn_func = None
        if backend.attn == "te":
            if self.v_head_dim > self.head_dim:
                raise ValueError(
                    "MiMo Transformer Engine attention requires v_head_dim <= qk head_dim so V can be "
                    f"zero-padded for a2a CP, got {self.v_head_dim} > {self.head_dim} at layer {layer_idx}."
                )
            self.attn_module, self.attn_func = initialize_attn_module_and_func(
                attn_impl="te",
                num_attention_heads=self.num_attention_heads,
                num_qk_channels=self.head_dim,
                # TE a2a CP cannot combine sliding/sink attention with unequal QK/V
                # dimensions. Padding V is mathematically exact because the padded
                # output channels are cropped before o_proj.
                num_v_channels=self.head_dim,
                softmax_scale=self.scaling,
                attn_mask_type="causal",
                qkv_format="bshd",
                num_gqa_groups=self.num_key_value_heads,
                attention_dropout=self.attention_dropout,
                softmax_type="learnable" if has_sink else "vanilla",
            )
            assert self.attn_module is not None
            if has_sink:
                # MiMo checkpoints freeze the sink. TE constructs it as a
                # Parameter, but FSDP resharding that Parameter invalidates the
                # pointer saved by TE's custom backward. Keep the identical
                # named tensor as a persistent FP32 buffer instead.
                softmax_offset = self.attn_module._parameters.pop("softmax_offset")
                self.attn_module.register_buffer(
                    "softmax_offset",
                    softmax_offset.detach().to(torch.float32),
                    persistent=True,
                )
            self.attention_sink_bias = None
        else:
            if has_sink:
                self.register_buffer("attention_sink_bias", torch.empty(self.num_attention_heads, dtype=torch.float32))
            else:
                self.attention_sink_bias = None

    def _validate_a2a_cp_size(self, cp_size: int) -> None:
        """Validate the head partition required by TE's a2a CP transport."""
        if cp_size <= 1:
            return
        if self.num_attention_heads % cp_size or self.num_key_value_heads % cp_size:
            raise ValueError(
                "MiMo TE context parallelism uses cp_comm_type='a2a', which requires both query and "
                "key/value head counts to be divisible by cp_size: "
                f"layer={self.layer_idx}, query_heads={self.num_attention_heads}, "
                f"kv_heads={self.num_key_value_heads}, cp_size={cp_size}."
            )

    def setup_cp_attention(self, cp_mesh, *, cp_stream: torch.cuda.Stream | None = None) -> None:
        """Validate and attach TE all-to-all context parallelism."""
        if self.backend.attn != "te" or self.attn_module is None:
            raise ValueError(
                "MiMo context parallelism requires backend.attn='te' with THD packed sequences; "
                f"got backend.attn={self.backend.attn!r}."
            )
        cp_size = int(cp_mesh.size())
        self._validate_a2a_cp_size(cp_size)
        if cp_size <= 1:
            return
        cp_stream = cp_stream if cp_stream is not None else torch.cuda.Stream()
        cp_group = cp_mesh.get_group()
        cp_ranks = torch.distributed.get_process_group_ranks(cp_group)
        self.attn_module.set_context_parallel_group(
            cp_group,
            cp_ranks,
            cp_stream,
            cp_comm_type="a2a",
        )

    @staticmethod
    def _apply_te_rope(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        qkv_format: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply partial RoPE to BSHD or flattened THD query/key tensors."""
        if qkv_format == "thd":
            if cos.ndim == 3:
                if cos.shape[0] != 1:
                    raise ValueError(f"THD RoPE expects one token stream, got cos shape {tuple(cos.shape)}")
                cos = cos.squeeze(0)
                sin = sin.squeeze(0)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)
        return (query_states * cos) + (_rotate_half(query_states) * sin), (key_states * cos) + (
            _rotate_half(key_states) * sin
        )

    def _forward_te(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run TE attention, zero-padding V for the exact MiMo a2a kernel path."""
        explicit_thd = kwargs.get("qkv_format") == "thd" or kwargs.get("cu_seqlens") is not None
        qkv_format = "thd" if explicit_thd else "bshd"
        restore_batch_dim = False
        if qkv_format == "thd" and hidden_states.ndim == 3:
            if hidden_states.shape[0] != 1:
                raise ValueError(
                    "MiMo TE THD attention expects a singleton token-stream dimension, "
                    f"got hidden_states shape {tuple(hidden_states.shape)}."
                )
            hidden_states = hidden_states.squeeze(0)
            restore_batch_dim = True
        if qkv_format == "thd" and hidden_states.ndim != 2:
            raise ValueError(f"MiMo TE THD attention expects [T,H], got {tuple(hidden_states.shape)}")
        if qkv_format == "bshd" and hidden_states.ndim != 3:
            raise ValueError(f"MiMo TE BSHD attention expects [B,S,H], got {tuple(hidden_states.shape)}")

        prefix = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states).view(*prefix, self.num_attention_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(*prefix, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(*prefix, self.num_key_value_heads, self.v_head_dim)
        if self.v_scale is not None:
            value_states = value_states * self.v_scale

        cos, sin = position_embeddings
        query_rope, query_nope = query_states.split((self.rope_dim, self.head_dim - self.rope_dim), dim=-1)
        key_rope, key_nope = key_states.split((self.rope_dim, self.head_dim - self.rope_dim), dim=-1)
        query_rope, key_rope = self._apply_te_rope(query_rope, key_rope, cos, sin, qkv_format=qkv_format)
        query_states = torch.cat((query_rope, query_nope), dim=-1)
        key_states = torch.cat((key_rope, key_nope), dim=-1)
        value_states = F.pad(value_states, (0, self.head_dim - self.v_head_dim))

        if (
            qkv_format == "thd"
            and self.training
            and self.attn_module is not None
            and getattr(self.attn_module, "softmax_offset", None) is not None
        ):
            _validate_te_thd_sink_cudnn(self.head_dim)

        te_kwargs = dict(kwargs)
        te_kwargs["qkv_format"] = qkv_format
        te_kwargs["window_size"] = (self.sliding_window - 1, 0) if self.sliding_window is not None else (-1, 0)
        query_states, key_states, value_states, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            query_states,
            key_states,
            value_states,
            None if qkv_format == "thd" else attention_mask,
            "te",
            **te_kwargs,
        )
        assert self.attn_func is not None
        output = self.attn_func(query_states, key_states, value_states, **attn_kwargs)
        output = postprocess_output_for_attn(output, "te")
        output = output.view(*prefix, self.num_attention_heads, self.head_dim)[..., : self.v_head_dim]
        output = self.o_proj(output.flatten(-2))
        return output.unsqueeze(0) if restore_batch_dim else output

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.backend.attn == "te":
            return self._forward_te(hidden_states, position_embeddings, attention_mask, **kwargs), None

        if kwargs.get("qkv_format") == "thd" or kwargs.get("cu_seqlens") is not None:
            raise ValueError(
                "MiMo THD packed attention requires backend.attn='te'; dense-mask SDPA/eager attention "
                "cannot safely recover THD document boundaries."
            )
        del kwargs
        batch, seq_len = hidden_states.shape[:2]
        q_shape = (batch, seq_len, self.num_attention_heads, self.head_dim)
        k_shape = (batch, seq_len, self.num_key_value_heads, self.head_dim)
        v_shape = (batch, seq_len, self.num_key_value_heads, self.v_head_dim)

        query_states = self.q_proj(hidden_states).view(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(k_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(v_shape).transpose(1, 2)
        if self.v_scale is not None:
            value_states = value_states * self.v_scale

        cos, sin = position_embeddings
        query_rope, query_nope = query_states.split((self.rope_dim, self.head_dim - self.rope_dim), dim=-1)
        key_rope, key_nope = key_states.split((self.rope_dim, self.head_dim - self.rope_dim), dim=-1)
        query_rope, key_rope = _apply_rotary_pos_emb(query_rope, key_rope, cos, sin)
        query_states = torch.cat((query_rope, query_nope), dim=-1)
        key_states = torch.cat((key_rope, key_nope), dim=-1)

        attn_output, attn_weights = _eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        del buffer_device
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)
            if getattr(linear, "bias", None) is not None:
                nn.init.zeros_(linear.bias)
        if self.backend.attn == "te" and self.attn_module is not None:
            if self.attn_module.softmax_offset is not None:
                nn.init.zeros_(self.attn_module.softmax_offset)
                self.attn_module.softmax_offset.requires_grad_(False)
        elif self.attention_sink_bias is not None:
            nn.init.zeros_(self.attention_sink_bias)


class MiMoV2FlashBlock(nn.Module):
    """Decoder block that alternates dense MLP and routed-MoE layers."""

    def __init__(self, layer_idx: int, config: MiMoV2FlashConfig, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        is_swa = config.hybrid_layer_pattern[layer_idx] == 1
        self.attention_type = "sliding_attention" if is_swa else "full_attention"
        self.self_attn = MiMoV2FlashAttention(config, backend, is_swa=is_swa, layer_idx=layer_idx)

        is_moe_layer = getattr(config, "n_routed_experts", None) is not None and bool(config.moe_layer_freq[layer_idx])
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        if is_moe_layer:
            self.mlp = MoE(moe_config, backend)
        else:
            self.mlp = MLP(
                config.hidden_size,
                config.intermediate_size,
                backend.linear,
                dtype=dtype,
                activation="swiglu",
                bias=False,
            )

        self.input_layernorm = _initialize_mimo_rms_norm(
            backend, config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype
        )
        self.post_attention_layernorm = _initialize_mimo_rms_norm(
            backend, config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, MoE):
            hidden_states = self.mlp(hidden_states, padding_mask)
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def init_weights(self, buffer_device: torch.device) -> None:
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device, init_std=0.02)
        self.mlp.init_weights(buffer_device)


class MiMoV2FlashModel(nn.Module):
    """Backbone model for Xiaomi MiMo-V2-Flash."""

    def __init__(
        self,
        config: MiMoV2FlashConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.backend = backend
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")

        # Route the gate compute in fp32 even when activations are bf16 to keep
        # MiMo's routing decisions stable (mirrors step3p5 and nemotron_v3).
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=int(config.n_routed_experts or 0),
            n_shared_experts=int(config.n_shared_experts or 0),
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="sigmoid_with_bias" if config.scoring_func == "sigmoid" else config.scoring_func,
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0.0,
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            apply_router_weight_after_down=config.apply_router_weight_after_down,
            softmax_before_topk=False,
            force_e_score_correction_bias=True,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleDict(
            {
                str(layer_id): MiMoV2FlashBlock(layer_id, config, self.moe_config, backend)
                for layer_id in range(config.num_hidden_layers)
            }
        )
        self.norm = _initialize_mimo_rms_norm(backend, config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.rotary_emb = MiMoV2FlashRotaryEmbedding(
            rope_theta=float(config.rope_theta),
            head_dim=int(config.head_dim),
            partial_rotary_factor=float(config.partial_rotary_factor),
            dtype=dtype,
        )
        self.swa_rotary_emb = MiMoV2FlashRotaryEmbedding(
            rope_theta=float(config.swa_rope_theta),
            head_dim=int(config.swa_head_dim),
            partial_rotary_factor=float(config.partial_rotary_factor),
            dtype=dtype,
        )
        self.has_sliding_layers = any(pattern == 1 for pattern in config.hybrid_layer_pattern)

    def _build_causal_mask_mapping(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = inputs_embeds.shape[:2]
        if isinstance(attention_mask, dict):
            full = attention_mask.get("full_attention")
            sliding = attention_mask.get("sliding_attention")
            if sliding is None:
                sliding = attention_mask.get("sliding_window_attention")
            return {
                "full_attention": _ensure_additive_mask(
                    full,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    attention_mask=None,
                    sliding_window=None,
                ),
                "sliding_attention": _ensure_additive_mask(
                    sliding,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    attention_mask=None,
                    sliding_window=self.config.sliding_window,
                ),
            }

        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            full = _ensure_additive_mask(
                attention_mask,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                attention_mask=None,
                sliding_window=None,
            )
            sliding_window = getattr(self.config, "sliding_window", None) if self.has_sliding_layers else None
            sliding = (
                _intersect_with_sliding_window(
                    full,
                    sliding_window=int(sliding_window),
                    dtype=inputs_embeds.dtype,
                )
                if sliding_window is not None
                else full
            )
            return {"full_attention": full, "sliding_attention": sliding}

        # Released MiMo configs leave _attn_implementation unset. The custom
        # eager attention owns mask application, so build ordinary masks here
        # instead of asking transformers to select an attention backend.
        if getattr(self.config, "_attn_implementation", None) is None:
            tensor_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else None
            full = _fallback_additive_mask(
                batch_size,
                seq_len,
                inputs_embeds.dtype,
                inputs_embeds.device,
                tensor_mask,
                None,
            )
            sliding_window = getattr(self.config, "sliding_window", None) if self.has_sliding_layers else None
            sliding = (
                _fallback_additive_mask(
                    batch_size,
                    seq_len,
                    inputs_embeds.dtype,
                    inputs_embeds.device,
                    tensor_mask,
                    int(sliding_window),
                )
                if sliding_window is not None
                else full
            )
            return {"full_attention": full, "sliding_attention": sliding}

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        full = create_causal_mask(**mask_kwargs)
        sliding = create_sliding_window_causal_mask(**mask_kwargs) if self.has_sliding_layers else None
        return {
            "full_attention": _ensure_additive_mask(
                full,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                sliding_window=None,
            ),
            "sliding_attention": _ensure_additive_mask(
                sliding,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                sliding_window=self.config.sliding_window,
            ),
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        _packed_seq_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        seq_lens_padded: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_padded: torch.Tensor | None = None,
        max_seqlen: int | torch.Tensor | None = None,
        qkv_format: str | None = None,
        cp_size: int = 1,
        cp_rank: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the text backbone in BSHD or genuine TE THD layout.

        THD batches must already have passed through the framework TE sharder.
        Their cumulative lengths delimit packed documents and are forwarded to
        every attention layer, including non-first pipeline stages.
        """
        del seq_lens, seq_lens_padded, kwargs
        is_thd = qkv_format == "thd"
        indexed_attention = (
            isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and attention_mask.numel() > 0
            and bool((attention_mask > 1).any().item())
        )
        if self.backend.attn == "te" and not is_thd and (_packed_seq_ids is not None or indexed_attention):
            raise ValueError(
                "MiMo TE packed attention requires THD packing with seq_lens/seq_lens_padded; "
                "NEAT document IDs cannot be represented safely by TE context parallelism."
            )
        if is_thd and self.backend.attn != "te":
            raise ValueError(
                "MiMo THD packed attention requires backend.attn='te'; select THD packing and TE together."
            )

        if is_thd and input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if is_thd and inputs_embeds is not None and inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if inputs_embeds is None:
            if self.embed_tokens is not None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds must be provided")
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                inputs_embeds = input_ids
        if inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided")
        if is_thd and inputs_embeds.ndim != 3:
            raise ValueError(
                f"MiMo THD model input must be [1,T,H] after pipeline chunking, got {tuple(inputs_embeds.shape)}."
            )
        if is_thd and inputs_embeds.shape[0] != 1:
            raise ValueError(
                "MiMo THD attention represents one flattened token stream per model call; "
                f"got batch dimension {inputs_embeds.shape[0]}."
            )

        if is_thd:
            if position_ids is not None and position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0)
            if padding_mask is not None and padding_mask.ndim == 1:
                padding_mask = padding_mask.unsqueeze(0)
            if cu_seqlens is None:
                raise ValueError(
                    "MiMo TE THD attention requires cu_seqlens from make_cp_batch_for_te; "
                    "call the model through its ContextParallelSharder."
                )
            if cu_seqlens.ndim == 2:
                if cu_seqlens.shape[0] != 1:
                    raise ValueError(f"Expected one THD cu_seqlens row, got {tuple(cu_seqlens.shape)}")
                cu_seqlens = cu_seqlens.squeeze(0)
            cu_seqlens = cu_seqlens[cu_seqlens >= 0].to(torch.int32).contiguous()
            if cu_seqlens_padded is not None:
                if cu_seqlens_padded.ndim == 2:
                    if cu_seqlens_padded.shape[0] != 1:
                        raise ValueError(
                            f"Expected one THD cu_seqlens_padded row, got {tuple(cu_seqlens_padded.shape)}"
                        )
                    cu_seqlens_padded = cu_seqlens_padded.squeeze(0)
                cu_seqlens_padded = cu_seqlens_padded[cu_seqlens_padded >= 0].to(torch.int32).contiguous()
                # MiMo packs each document as real tokens followed only by tail
                # padding. Use its physical boundary directly: causal real-token
                # outputs are unchanged, while this avoids TE/cuDNN's unstable
                # separate padded-cu backward path and matches CP>1 sharding.
                cu_seqlens = cu_seqlens_padded
                cu_seqlens_padded = None
            if isinstance(max_seqlen, torch.Tensor) and max_seqlen.numel() != 1:
                raise ValueError(f"Expected scalar max_seqlen, got {tuple(max_seqlen.shape)}")
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            attention_mask = None

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if padding_mask is None and isinstance(attention_mask, torch.Tensor):
            padding_mask = _derive_padding_mask(attention_mask)

        if is_thd:
            causal_mask_mapping = {"full_attention": None, "sliding_attention": None}
        elif self.backend.attn == "te":
            if isinstance(attention_mask, dict) or (
                isinstance(attention_mask, torch.Tensor) and attention_mask.ndim != 2
            ):
                raise ValueError(
                    "MiMo TE BSHD attention expects a 2D padding mask [batch, sequence], "
                    f"got {type(attention_mask).__name__}"
                    + (f" with shape {tuple(attention_mask.shape)}" if isinstance(attention_mask, torch.Tensor) else "")
                )
            # TE constructs the causal mask internally and applies each layer's
            # window_size. Passing the original padding mask avoids expanding an
            # additive [B,1,S,S] mask through TE's 2D padding-mask interface.
            causal_mask_mapping = {
                "full_attention": attention_mask,
                "sliding_attention": attention_mask,
            }
        else:
            causal_mask_mapping = self._build_causal_mask_mapping(
                inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        swa_position_embeddings = self.swa_rotary_emb(hidden_states, position_ids)
        attention_kwargs = {
            "qkv_format": qkv_format,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_padded": cu_seqlens_padded,
            "max_seqlen": max_seqlen,
            "cp_size": cp_size,
            "cp_rank": cp_rank,
        }
        attention_kwargs = {key: value for key, value in attention_kwargs.items() if value is not None}

        for decoder_layer in self.layers.values():
            layer_position_embeddings = (
                swa_position_embeddings if decoder_layer.attention_type == "sliding_attention" else position_embeddings
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=layer_position_embeddings,
                padding_mask=padding_mask,
                **attention_kwargs,
            )

        return self.norm(hidden_states) if self.norm is not None else hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device)


class MiMoV2FlashForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Causal LM wrapper for MiMo-V2-Flash with Automodel checkpoint adapters."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    # "rotary_emb" (matches self.rotary_emb + self.swa_rotary_emb) pins their inv_freq
    # buffers in fp32: cast_model_to_dtype's bf16 cast would otherwise round inv_freq and
    # degrade RoPE precision vs HF (see llama/rope_utils.py).
    _keep_in_fp32_modules_strict = [
        "mlp.gate.e_score_correction_bias",
        "attention_sink_bias",
        "attn_module.softmax_offset",
        "rotary_emb",
    ]
    _pp_keep_self_forward = True
    _pp_return_hidden_states_supported = True
    _skip_init_weights_on_load = True
    _owns_cp_attention = True
    _owns_packed_attention = True
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: MiMoV2FlashConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = MiMoV2FlashConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: MiMoV2FlashConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = MiMoV2FlashModel(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
        )
        self.visual = None
        if config.vision_config is not None:
            self.visual = MiMoVisionTransformer(
                config.vision_config,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiMoV2FlashStateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _get_multimodal_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        image_feature_indices: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        video_feature_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode global media and splice only this TE THD shard's feature rows."""
        has_image = pixel_values is not None
        has_video = pixel_values_videos is not None
        if not has_image and not has_video:
            return inputs_embeds
        if self.visual is None:
            raise ValueError("Image or video inputs require a non-empty vision_config")

        if has_image:
            image_grid_thw = _normalize_image_grid(image_grid_thw)
            image_embeds = self.visual(pixel_values, image_grid_thw)
            if image_feature_indices is not None:
                image_embeds = image_embeds.index_select(0, image_feature_indices.to(image_embeds.device))
            inputs_embeds = _replace_modal_embeddings(
                input_ids,
                inputs_embeds,
                getattr(self.config, "image_token_id", None),
                image_embeds,
            )
        if has_video:
            video_grid_thw = _normalize_image_grid(video_grid_thw)
            video_embeds = self.visual(pixel_values_videos, video_grid_thw)
            if video_feature_indices is not None:
                video_embeds = video_embeds.index_select(0, video_feature_indices.to(video_embeds.device))
            inputs_embeds = _replace_modal_embeddings(
                input_ids,
                inputs_embeds,
                getattr(self.config, "video_token_id", None),
                video_embeds,
            )
        return inputs_embeds

    @staticmethod
    def _local_modal_feature_indices(
        global_mask: torch.Tensor | None,
        local_indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Map local TE THD placeholders to rows in globally ordered media features."""
        if global_mask is None:
            return None
        if local_indices is None:
            raise ValueError("MiMo VLM THD media mapping requires TE local-token global indices")
        global_mask = global_mask.reshape(-1).bool()
        local_indices = local_indices.reshape(-1).to(device=global_mask.device, dtype=torch.long)
        if local_indices.numel() and int(local_indices.max().item()) >= global_mask.numel():
            raise ValueError(
                "MiMo VLM THD local token index exceeds the global placeholder mask: "
                f"max_index={int(local_indices.max().item())}, mask_tokens={global_mask.numel()}."
            )
        feature_index_by_token = global_mask.long().cumsum(0) - 1
        local_mask = global_mask.index_select(0, local_indices)
        return feature_index_by_token.index_select(0, local_indices)[local_mask].to(torch.long)

    def _pull_pipeline_media(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        *,
        global_image_mask: torch.Tensor | None = None,
        global_video_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Retrieve PP media using global masks so every CP rank runs the same vision work."""
        if input_ids is None or torch.is_floating_point(input_ids):
            return pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw

        chunk_idx = int(getattr(self, "_vlm_chunk_idx", 0) or 0)
        image_chunks = getattr(self, "_vlm_pixel_values_chunks", None)
        video_chunks = getattr(self, "_vlm_pixel_values_videos_chunks", None)
        has_staged_media = image_chunks is not None or video_chunks is not None
        if not has_staged_media:
            return pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw

        if image_chunks is not None and chunk_idx >= len(image_chunks):
            raise RuntimeError(f"MiMo PP image-media cursor {chunk_idx} exceeds {len(image_chunks)} staged chunks")
        if video_chunks is not None and chunk_idx >= len(video_chunks):
            raise RuntimeError(f"MiMo PP video-media cursor {chunk_idx} exceeds {len(video_chunks)} staged chunks")

        image_token_id = getattr(self.config, "image_token_id", None)
        image_placeholder_mask = global_image_mask
        if image_placeholder_mask is None and image_token_id is not None:
            image_placeholder_mask = input_ids.eq(image_token_id)
        if (
            pixel_values is None
            and image_chunks is not None
            and image_placeholder_mask is not None
            and bool(image_placeholder_mask.any())
        ):
            pixel_values = image_chunks[chunk_idx]
            grid_chunks = getattr(self, "_vlm_image_grid_hws_chunks", None)
            if grid_chunks is not None and chunk_idx < len(grid_chunks):
                image_grid_thw = _normalize_image_grid(grid_chunks[chunk_idx])

        video_token_id = getattr(self.config, "video_token_id", None)
        video_placeholder_mask = global_video_mask
        if video_placeholder_mask is None and video_token_id is not None:
            video_placeholder_mask = input_ids.eq(video_token_id)
        if (
            pixel_values_videos is None
            and video_chunks is not None
            and video_placeholder_mask is not None
            and bool(video_placeholder_mask.any())
        ):
            pixel_values_videos = video_chunks[chunk_idx]
            grid_chunks = getattr(self, "_vlm_video_grid_thw_chunks", None)
            if grid_chunks is not None and chunk_idx < len(grid_chunks):
                video_grid_thw = grid_chunks[chunk_idx]
        # Every integer-token microbatch consumes exactly one staged slot,
        # including text-only slots whose media chunks are empty.
        self._vlm_chunk_idx = chunk_idx + 1
        return pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | torch.Tensor:
        """Run text or VLM forward with TE-local packed media placement."""
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        is_thd = kwargs.get("qkv_format") == "thd"
        local_thd_indices = kwargs.pop(_MIMO_THD_LOCAL_INDICES, None)
        global_image_mask = kwargs.pop(_MIMO_GLOBAL_IMAGE_MASK, None)
        global_video_mask = kwargs.pop(_MIMO_GLOBAL_VIDEO_MASK, None)
        cp_size = int(kwargs.get("cp_size", 1))
        if cp_size > 1 and self.config.vision_config is not None and is_thd and local_thd_indices is None:
            raise ValueError("MiMo VLM TE context parallelism requires the local THD index map from its model sharder.")

        pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw = self._pull_pipeline_media(
            input_ids,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            global_image_mask=global_image_mask,
            global_video_mask=global_video_mask,
        )

        integer_tokens = input_ids is not None and not torch.is_floating_point(input_ids)
        has_media = pixel_values is not None or pixel_values_videos is not None
        if inputs_embeds is None and integer_tokens and has_media:
            if self.model.embed_tokens is None:
                raise ValueError("The first pipeline stage must own embed_tokens")
            image_feature_indices = self._local_modal_feature_indices(global_image_mask, local_thd_indices)
            video_feature_indices = self._local_modal_feature_indices(global_video_mask, local_thd_indices)
            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = self._get_multimodal_embeds(
                input_ids,
                inputs_embeds,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_feature_indices=image_feature_indices,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                video_feature_indices=video_feature_indices,
            )
            input_ids = None

        hidden = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **kwargs,
        )
        if getattr(self, "_pp_return_hidden_states", False) is True:
            return hidden
        if self.lm_head is None:
            return CausalLMOutputWithPast(
                logits=hidden,
                hidden_states=hidden if output_hidden_states else None,
            )

        output = compute_lm_head_logits(
            self.lm_head,
            hidden,
            logits_to_keep,
            is_thd=is_thd,
            output_hidden_states=output_hidden_states,
        )
        if (
            is_thd
            and output.hidden_states is not None
            and output.hidden_states.ndim == 3
            and output.hidden_states.shape[0] == 1
        ):
            # The TE THD sharder flattens labels to [T]. FusedLinearCrossEntropy
            # therefore consumes [T,H], while logits retain the public [1,T,V]
            # shape. PP returns above and keeps [microbatch,T,H] for its schedule.
            output.hidden_states = output.hidden_states.squeeze(0)
        return output

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Describe PP tensors using the already TE-sharded local token length."""
        hidden_size = int(self.config.hidden_size)
        vocab_size = int(self.config.vocab_size)
        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = (torch.empty(microbatch_size, seq_len, hidden_size, device="meta", dtype=dtype),)

        has_lm_head = self.lm_head is not None
        emits_hidden_states = getattr(self, "_pp_return_hidden_states", False) is True
        output_width = vocab_size if has_lm_head and not emits_hidden_states else hidden_size
        outputs_meta = (torch.empty(microbatch_size, seq_len, output_width, device="meta", dtype=dtype),)
        return inputs_meta, outputs_meta

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep both rotary embeddings per stage and the vision tower on stage zero."""
        text_model = text_model or self.model
        stage_modules = [list(modules) for modules in module_names_per_stage]
        for stage_idx, modules in enumerate(stage_modules):
            if "model.visual" in modules:
                modules.remove("model.visual")
            if stage_idx == 0 and self.visual is not None and "visual" not in modules:
                modules.append("visual")
            if stage_idx > 0 and "visual" in modules:
                modules.remove("visual")
        if getattr(text_model, "swa_rotary_emb", None) is not None:
            fqn = f"{layers_prefix}swa_rotary_emb"
            for modules in stage_modules:
                if fqn not in modules:
                    modules.append(fqn)
        return stage_modules

    def prepare_model_inputs_for_cp(self, batch: dict[str, Any], *, num_chunks: int = 1) -> dict[str, Any]:
        """Install MiMo's model-local adapter around framework TE THD sharding."""
        if self.backend.attn != "te":
            raise ValueError(f"MiMo context parallelism requires model.backend.attn='te'; got {self.backend.attn!r}.")
        if batch.get("qkv_format") != "thd":
            raise ValueError(
                "MiMo context parallelism requires packed THD input. NEAT/dense packed masks cannot be "
                "mapped to Transformer Engine document boundaries; set packing_strategy='thd'."
            )
        return {
            "cp_sharder": make_mimo_te_cp_sharder(
                model=self,
                num_chunks=num_chunks,
                image_token_id=getattr(self.config, "image_token_id", None),
                video_token_id=getattr(self.config, "video_token_id", None),
            )
        }

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device)
            if self.visual is not None:
                self.visual.init_weights()
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
        if _has_dtensor_params(self):
            return
        cast_model_to_dtype(self, dtype)


class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):
    """MiMo-V2.5/V2.6 wrapper using the ``mimo_v2`` checkpoint config."""

    # Registry validation intentionally requires every registered architecture
    # to declare its capability contract on the concrete class.
    ModelCapabilities = MiMoV2FlashForCausalLM.ModelCapabilities

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "MiMoV2ForCausalLM":
        """Resolve a MiMo-V2 config while checkpoint loading remains framework-owned."""
        config = MiMoV2Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)


ModelClass = MiMoV2FlashForCausalLM

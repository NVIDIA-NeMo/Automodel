# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Native Automodel support for Moonshot Kimi Linear causal LM checkpoints."""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.kimi_linear.config import KimiLinearConfig
from nemo_automodel.components.models.kimi_linear.state_dict_adapter import KimiLinearStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.import_utils import UnavailableError, safe_import_from
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

_FLA_MSG = "Kimi Linear requires the flash-linear-attention/fla extra. Install with `uv sync --extra fla`."
_SHORT_CONV_OK, ShortConvolution = safe_import_from("fla.modules", "ShortConvolution", msg=_FLA_MSG)
_FUSED_RMSNORM_GATED_OK, FusedRMSNormGated = safe_import_from(
    "fla.modules",
    "FusedRMSNormGated",
    msg=_FLA_MSG,
)
_CHUNK_KDA_OK, chunk_kda = safe_import_from("fla.ops.kda", "chunk_kda", msg=_FLA_MSG)
_RECURRENT_KDA_OK, fused_recurrent_kda = safe_import_from("fla.ops.kda", "fused_recurrent_kda", msg=_FLA_MSG)
_KDA_GATE_OK, fused_kda_gate = safe_import_from("fla.ops.kda.gate", "fused_kda_gate", msg=_FLA_MSG)
try:
    _FUSED_KDA_GATE_HAS_G_BIAS = _KDA_GATE_OK and "g_bias" in inspect.signature(fused_kda_gate).parameters
except (TypeError, ValueError):
    _FUSED_KDA_GATE_HAS_G_BIAS = False


def _require_fla() -> None:
    if not all((_SHORT_CONV_OK, _FUSED_RMSNORM_GATED_OK, _CHUNK_KDA_OK, _RECURRENT_KDA_OK, _KDA_GATE_OK)):
        raise UnavailableError(_FLA_MSG)


def _fused_kda_gate(g: torch.Tensor, a_log: torch.Tensor, head_dim: int, dt_bias: torch.Tensor) -> torch.Tensor:
    """Call FLA fused KDA gate across FLA versions.

    Args:
        g: Tensor of shape [batch, sequence, heads * head_dim].
        a_log: Tensor of shape [1, 1, heads, 1].
        head_dim: Per-head KDA dimension.
        dt_bias: Tensor of shape [heads * head_dim].

    Returns:
        Tensor of shape [batch, sequence, heads, head_dim].
    """
    if _FUSED_KDA_GATE_HAS_G_BIAS:
        return fused_kda_gate(g, a_log, head_dim, g_bias=dt_bias)
    gate_input = g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)
    return fused_kda_gate(gate_input, a_log, dt_bias=dt_bias)


def _torch_kda_gate(g: torch.Tensor, a_log: torch.Tensor, head_dim: int, dt_bias: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of FLA's KDA gate.

    Args:
        g: Tensor of shape [batch, sequence, heads * head_dim] or [batch, sequence, heads, head_dim].
        a_log: Tensor of shape [1, 1, heads, 1].
        head_dim: Per-head KDA dimension.
        dt_bias: Tensor of shape [heads * head_dim].

    Returns:
        Tensor of shape [batch, sequence, heads, head_dim].
    """
    gate = g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)
    num_heads = gate.shape[-2]
    gate = gate.float() + dt_bias.float().view(num_heads, head_dim)
    return -a_log.float().view(num_heads, 1).exp() * F.softplus(gate)


def _index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather rows from the first axis while preserving trailing tensor layout.

    Args:
        x: Tensor of shape [tokens, ...], with arbitrary trailing axes.
        indices: Tensor of shape [selected_tokens] containing first-axis row indices.

    Returns:
        Tensor of shape [selected_tokens, ...], with the same trailing axes as ``x``.
    """
    other_shape = x.shape[1:]
    return (
        x.reshape(x.shape[0], -1)
        .gather(0, indices[:, None].expand(-1, math.prod(other_shape)))
        .reshape(-1, *other_shape)
    )


def _index_put_first_axis(x: torch.Tensor, indices: torch.Tensor, first_axis_dim: int) -> torch.Tensor:
    """Scatter rows into the first axis while preserving trailing tensor layout.

    Args:
        x: Tensor of shape [selected_tokens, ...], with arbitrary trailing axes.
        indices: Tensor of shape [selected_tokens] containing destination row indices.
        first_axis_dim: Size of the output first axis.

    Returns:
        Tensor of shape [first_axis_dim, ...], with the same trailing axes as ``x``.
    """
    y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
    y[indices] = x
    return y


def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build metadata for converting padded batches to flattened valid tokens.

    Args:
        attention_mask: Binary mask tensor of shape [batch, sequence] where 1 marks valid tokens.

    Returns:
        Tuple containing ``indices`` of shape [total_valid_tokens], ``cu_seqlens`` of shape [batch + 1],
        and ``max_seqlen`` for the longest unpadded sequence.
    """
    mask = attention_mask.bool()
    lengths = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(lengths.max().item())
    cu_seqlens = F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen


def _pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Restore flattened valid tokens to padded batch layout.

    Args:
        hidden_states: Tensor of shape [total_valid_tokens, ...], with arbitrary trailing axes.
        indices: Tensor of shape [total_valid_tokens] containing flattened padded-batch row indices.
        batch_size: Number of sequences in the padded output batch.
        seq_len: Sequence length in the padded output batch.

    Returns:
        Tensor of shape [batch, sequence, ...], with the same trailing axes as ``hidden_states``.
    """
    output = _index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return output.reshape(batch_size, seq_len, *hidden_states.shape[1:])


def _make_causal_mask(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Create the additive causal attention mask for padded full attention.

    Args:
        inputs_embeds: Tensor of shape [batch, sequence, hidden].
        attention_mask: Optional binary mask tensor of shape [batch, sequence] where 1 marks valid tokens.
        dtype: Floating-point dtype used for the additive mask values.

    Returns:
        Additive causal mask tensor of shape [batch, 1, sequence, sequence], or None.
    """
    batch_size, seq_len = inputs_embeds.shape[:2]
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, device=inputs_embeds.device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    mask = mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None and attention_mask.ndim == 2:
        padding = attention_mask[:, None, None, :].to(torch.bool)
        mask = mask.masked_fill(~padding, min_value)
    return mask


class KimiRMSNorm(nn.Module):
    """Kimi RMSNorm with fp32 variance computation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize hidden states.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


class KimiMLAAttention(nn.Module):
    """Kimi MLA full-attention layer copied from the HF reference math."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        if config.q_lora_rank is not None:
            raise ValueError("Kimi Linear Automodel support currently expects q_lora_rank=None.")
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if not config.mla_use_nope:
            raise ValueError("Kimi Linear Automodel support expects mla_use_nope=True.")
        self.scaling = self.q_head_dim**-0.5

        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False, dtype=dtype)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank, dtype=dtype)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run MLA full attention.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            attention_mask: Optional additive attention mask of shape [batch, 1, sequence, sequence].
            **kwargs: Extra attention options accepted for HF compatibility.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        del kwargs
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        q_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if self.num_key_value_groups != 1:
            key_states = key_states[:, :, None, :, :].expand(
                batch_size,
                self.num_key_value_heads,
                self.num_key_value_groups,
                seq_length,
                self.q_head_dim,
            )
            value_states = value_states[:, :, None, :, :].expand(
                batch_size,
                self.num_key_value_heads,
                self.num_key_value_groups,
                seq_length,
                self.v_head_dim,
            )
            key_states = key_states.reshape(batch_size, self.num_heads, seq_length, self.q_head_dim)
            value_states = value_states.reshape(batch_size, self.num_heads, seq_length, self.v_head_dim)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
        return self.o_proj(attn_output)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        with buffer_device:
            for module in (self.q_proj, self.kv_a_proj_with_mqa, self.kv_b_proj, self.o_proj):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            self.kv_a_layernorm.reset_parameters()


class _KimiKDAFp32Param:
    """Descriptor exposing a KDA fp32 parameter from the ``_fp32_params`` holder."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(
        self, obj: nn.Module | None, owner: type[nn.Module] | None = None
    ) -> nn.Parameter | "_KimiKDAFp32Param":
        del owner
        if obj is None:
            return self
        holder = obj._modules.get("_fp32_params")
        if holder is not None:
            return getattr(holder, self.name)
        param = obj._parameters.get(self.name)
        if param is not None:
            return param
        raise AttributeError(f"{type(obj).__name__} has no KDA fp32 parameter {self.name!r}.")


class KimiKDAFp32Params(nn.Module):
    """Owns Kimi KDA fp32 recurrent-decay parameters and computes the gate."""

    def __init__(self, num_heads: int, projection_size: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32).view(1, 1, num_heads, 1))
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

    def forward(self, g: torch.Tensor, head_dim: int, use_fused_gate: bool = True) -> torch.Tensor:
        """Compute KDA decay gate while holder params are unsharded by FSDP.

        Args:
            g: Tensor of shape [batch, sequence, heads * head_dim].
            head_dim: Per-head KDA dimension.
            use_fused_gate: Whether to use FLA's fused KDA gate kernel.

        Returns:
            Tensor of shape [batch, sequence, heads, head_dim].
        """
        a_log = self.A_log.contiguous()
        dt_bias = self.dt_bias.contiguous()
        if use_fused_gate:
            return _fused_kda_gate(g, a_log, head_dim, dt_bias)
        return _torch_kda_gate(g, a_log, head_dim, dt_bias)


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention backed by FLA KDA kernels."""

    A_log = _KimiKDAFp32Param("A_log")
    dt_bias = _KimiKDAFp32Param("dt_bias")

    def __init__(self, config: KimiLinearConfig, layer_idx: int) -> None:
        _require_fla()
        super().__init__()
        self.config = config
        self.mode = getattr(config, "kda_mode", "chunk")
        if self.mode not in ("chunk", "fused_recurrent"):
            raise ValueError(f"Unsupported Kimi KDA mode {self.mode!r}.")
        self.hidden_size = config.hidden_size
        self.conv_size = config.linear_attn_config["short_conv_kernel_size"]
        self.head_dim = config.linear_attn_config["head_dim"]
        self.num_heads = config.linear_attn_config["num_heads"]
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads
        self.layer_idx = layer_idx

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.q_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, projection_k_size, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False, dtype=dtype)
        self.q_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
            dtype=dtype,
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_k_size,
            kernel_size=self.conv_size,
            activation="silu",
            dtype=dtype,
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
            dtype=dtype,
        )

        self._fp32_params = KimiKDAFp32Params(self.num_heads, projection_size)
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False, dtype=dtype)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False, dtype=dtype)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False, dtype=dtype)
        self.o_norm = FusedRMSNormGated(self.head_dim, eps=config.rms_norm_eps, activation="sigmoid", dtype=dtype)
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run KDA linear attention.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            attention_mask: Optional binary padding mask of shape [batch, sequence] where 1 marks valid tokens.
            **kwargs: Optional KDA kwargs, including ``cu_seqlens`` for packed sequences.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")
            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError("Kimi KDA attention_mask must have shape [batch, sequence].")

        batch_size, q_len, _ = hidden_states.shape

        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None
        if attention_mask is not None and getattr(self.config, "kda_unpad_inputs", True):
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = _index_first_axis(hidden_states.reshape(batch_size * q_len, -1), indices).unsqueeze(0)
        effective_q_len = hidden_states.shape[1]
        mode = "fused_recurrent" if effective_q_len <= 64 else self.mode

        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        k, _ = self.k_conv1d(x=self.k_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        v, _ = self.v_conv1d(x=self.v_proj(hidden_states), cache=None, output_final_state=False, cu_seqlens=cu_seqlens)
        g = self.f_b_proj(self.f_a_proj(hidden_states)).contiguous()
        beta = self.b_proj(hidden_states).float().sigmoid()

        q = q.reshape(*q.shape[:-1], self.num_k_heads, self.head_k_dim).contiguous()
        k = k.reshape(*k.shape[:-1], self.num_k_heads, self.head_k_dim).contiguous()
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).contiguous()
        beta = beta.contiguous()
        g = self._fp32_params(g, self.head_dim, getattr(self.config, "kda_use_fused_gate", True)).contiguous()
        use_qk_l2norm_in_kernel = getattr(self.config, "kda_use_qk_l2norm_in_kernel", True)
        if not use_qk_l2norm_in_kernel:
            q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(q.dtype)
            k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(k.dtype)

        if mode == "chunk":
            o, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )

        gate = self.g_b_proj(self.g_a_proj(hidden_states)).reshape(*hidden_states.shape[:-1], self.num_heads, -1)
        o = self.o_norm(o, gate)
        o = o.reshape(o.shape[0], o.shape[1], -1).contiguous()
        o = self.o_proj(o)
        if indices is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        with buffer_device:
            self.A_log.uniform_(1, 16).log_()
            self.dt_bias.zero_()
            for module in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.f_a_proj,
                self.f_b_proj,
                self.b_proj,
                self.g_a_proj,
                self.g_b_proj,
                self.o_proj,
            ):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
                conv.reset_parameters()
            if hasattr(self.o_norm, "reset_parameters"):
                self.o_norm.reset_parameters()


class KimiDecoderLayer(nn.Module):
    """Kimi decoder block with KDA/MLA attention and dense or MoE MLP."""

    def __init__(self, config: KimiLinearConfig, layer_idx: int, moe_config: MoEConfig, backend: BackendConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.is_linear_attn = config.is_kda_layer(layer_idx)
        self.is_moe_layer = (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % getattr(config, "moe_layer_freq", 1) == 0
        )
        self.self_attn = (
            KimiDeltaAttention(config, layer_idx) if self.is_linear_attn else KimiMLAAttention(config, layer_idx)
        )

        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        if self.is_moe_layer:
            self.block_sparse_moe = MoE(moe_config, backend)
        else:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Run one Kimi decoder layer.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            attention_mask: KDA layers receive a binary mask [batch, sequence]; MLA layers receive an additive
                causal mask [batch, 1, sequence, sequence].
            padding_mask: Optional boolean tensor of shape [batch, sequence], where true marks padding tokens.
            **attn_kwargs: Extra attention kwargs forwarded to KDA/MLA.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, **attn_kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe_layer:
            if attention_mask is not None and attention_mask.ndim == 2 and padding_mask is None:
                padding_mask = attention_mask.bool().logical_not()
            hidden_states = self._moe(hidden_states, padding_mask)
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def _moe(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Run a Kimi MoE layer.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            padding_mask: Optional boolean tensor of shape [batch, sequence], where true marks padding tokens.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        moe = unwrap_checkpoint_wrapper(self.block_sparse_moe)
        if not isinstance(moe, MoE):
            raise TypeError(f"Expected Kimi MoE layer, got {type(moe).__name__}.")
        experts = unwrap_checkpoint_wrapper(moe.experts)
        if (
            not self.training
            and padding_mask is None
            and isinstance(experts, GroupedExperts)
            and not self._has_dtensor_expert_params(experts)
        ):
            return self._moe_infer_hf_order(moe, experts, hidden_states)
        return self.block_sparse_moe(hidden_states, padding_mask)

    @staticmethod
    def _has_dtensor_expert_params(experts: GroupedExperts) -> bool:
        return hasattr(experts.gate_and_up_projs, "to_local") or hasattr(experts.down_projs, "to_local")

    def _moe_infer_hf_order(
        self,
        moe: MoE,
        experts: GroupedExperts,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Run Kimi inference MoE in the same expert-ordered loop as the HF reference.

        Args:
            moe: MoE module containing the router and optional shared experts.
            experts: Grouped routed experts for the layer.
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        token_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        weights, indices, _ = moe.gate(x, token_mask, moe.cp_mesh)

        counts = indices.new_zeros((indices.shape[0], experts.n_routed_experts))
        counts.scatter_(1, indices, 1)
        tokens_per_expert = counts.sum(dim=0).cpu().tolist()
        sorted_ids = indices.reshape(-1).argsort()
        sorted_tokens = x[sorted_ids // indices.shape[1]]

        gate_and_up_projs = experts.gate_and_up_projs.to(x.dtype)
        down_projs = experts.down_projs.to(x.dtype)
        gate_up_bias = experts.gate_up_proj_bias.to(x.dtype) if experts.gate_up_proj_bias is not None else None
        down_bias = experts.down_proj_bias.to(x.dtype) if experts.down_proj_bias is not None else None

        outputs = []
        start = 0
        for expert_idx, num_tokens in enumerate(tokens_per_expert):
            end = start + num_tokens
            if num_tokens == 0:
                continue
            tokens = sorted_tokens[start:end]
            gate_up = tokens @ gate_and_up_projs[expert_idx]
            if gate_up_bias is not None:
                gate_up = gate_up + gate_up_bias[expert_idx]
            gate, up = torch.chunk(gate_up, 2, dim=-1)
            expert_out = F.silu(gate) * up
            expert_out = expert_out @ down_projs[expert_idx]
            if down_bias is not None:
                expert_out = expert_out + down_bias[expert_idx]
            outputs.append(expert_out)
            start = end

        if outputs:
            routed = torch.cat(outputs, dim=0)
        else:
            routed = sorted_tokens.new_empty((0, x.shape[-1]))
        unpermuted = torch.empty_like(routed)
        unpermuted[sorted_ids] = routed
        y = (
            unpermuted.view(*indices.shape, -1)
            .type(weights.dtype)
            .mul_(weights.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(routed.dtype)
        )

        if moe.shared_experts is not None:
            shared = moe.shared_experts(x)
            if moe.shared_expert_gate is not None:
                shared = torch.sigmoid(moe.shared_expert_gate(x)) * shared
            y = y + shared
        return y.view(orig_shape)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device, init_std)
        if self.is_moe_layer:
            self.block_sparse_moe.init_weights(buffer_device, init_std)
        else:
            self.mlp.init_weights(buffer_device, init_std)


def _build_moe_config(
    config: KimiLinearConfig,
    model_dtype: torch.dtype,
    moe_overrides: dict[str, Any] | None,
) -> MoEConfig:
    moe_defaults = dict(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=config.num_shared_experts or 0,
        n_activated_experts=config.num_experts_per_token,
        n_expert_groups=config.num_expert_group,
        n_limited_groups=config.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func=config.moe_router_activation_func,
        route_scale=config.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=config.moe_renormalize,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        router_topk_sorted=False,
        router_weights_fp32=True,
        router_weight_uses_score_correction_bias=True,
        route_weight_after_down_proj=True,
        dtype=model_dtype,
        shared_expert_gate=False,
        shared_expert_inter_dim=config.moe_intermediate_size,
        force_e_score_correction_bias=True,
    )
    if moe_overrides:
        moe_defaults.update(moe_overrides)
    return MoEConfig(**moe_defaults)


class KimiLinearModel(nn.Module):
    """Kimi Linear decoder backbone with trainable Automodel MoE layers."""

    def __init__(
        self,
        config: KimiLinearConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.moe_config = moe_config or _build_moe_config(config, model_dtype, moe_overrides)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=model_dtype)
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): KimiDecoderLayer(config, layer_idx, self.moe_config, backend)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype)

    def _update_linear_attn_mask(
        self,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
    ) -> torch.Tensor | None:
        """Select the padding mask passed to KDA layers.

        Args:
            attention_mask: Optional binary padding mask tensor of shape [batch, sequence].
            cache_position: Tensor of shape [sequence] containing current token positions.

        Returns:
            Binary padding mask tensor of shape [batch, sequence], or None when no KDA mask is needed.
        """
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            return None
        return attention_mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Run the Kimi Linear decoder.

        Args:
            input_ids: Optional token ids of shape [batch, sequence].
            inputs_embeds: Optional embeddings of shape [batch, sequence, hidden].
            attention_mask: Optional binary padding mask of shape [batch, sequence].
            position_ids: Optional positions of shape [batch, sequence]; accepted for HF compatibility.
            padding_mask: Optional boolean tensor of shape [batch, sequence], where true marks padding tokens.
            cache_position: Optional position vector of shape [sequence].
            **attn_kwargs: Additional attention kwargs used by packed or THD execution.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        del position_ids
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)
        causal_mask = _make_causal_mask(inputs_embeds, attention_mask, dtype=inputs_embeds.dtype)
        hidden_states = inputs_embeds

        for decoder_layer in self.layers.values():
            layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask
            layer_padding_mask = padding_mask
            if decoder_layer.is_linear_attn and layer_mask is not None:
                layer_padding_mask = layer_mask.bool().logical_not()
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                padding_mask=layer_padding_mask,
                **attn_kwargs,
            )

        return self.norm(hidden_states)

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.layers.values():
                if block.is_moe_layer and block.block_sparse_moe.gate.bias_update_factor > 0:
                    block.block_sparse_moe.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or (
            torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
        )
        init_std = self.config.initializer_range
        with buffer_device:
            nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=init_std)
            if self.padding_idx is not None:
                self.embed_tokens.weight[self.padding_idx].zero_()
            self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device, init_std)


class KimiLinearForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Kimi Linear causal LM with native trainable MoE layers."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules = ["_fp32_params", "e_score_correction_bias"]
    _keep_in_fp32_modules_strict = ["_fp32_params", "e_score_correction_bias"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for Kimi Linear."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: KimiLinearConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "KimiLinearForCausalLM":
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "KimiLinearForCausalLM":
        config = KimiLinearConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: KimiLinearConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.config = config
        self.backend = copy.copy(backend) if backend is not None else BackendConfig()
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = KimiLinearModel(config, self.backend, moe_config=moe_config, moe_overrides=moe_overrides)
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=model_dtype,
        )
        self.vocab_size = config.vocab_size
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = KimiLinearStateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=model_dtype,
            )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run Kimi Linear causal LM.

        Args:
            input_ids: Optional token ids of shape [batch, sequence].
            attention_mask: Optional binary padding mask of shape [batch, sequence].
            position_ids: Optional positions of shape [batch, sequence].
            inputs_embeds: Optional embeddings of shape [batch, sequence, hidden].
            padding_mask: Optional boolean tensor of shape [batch, sequence], where true marks padding tokens.
            logits_to_keep: Number of trailing sequence logits to compute, or tensor indices.
            output_hidden_states: Whether to include hidden states in the output.
            **attn_kwargs: Additional attention kwargs used by packed or THD execution.

        Returns:
            Causal LM output whose logits have shape [batch, sequence, vocab] unless ``logits_to_keep`` trims sequence.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        is_thd = attn_kwargs.get("qkv_format") == "thd"
        if is_thd:
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids,
                position_ids,
                padding_mask,
                attn_kwargs,
            )
            attention_mask = None

        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        return compute_lm_head_logits(
            self.lm_head,
            hidden_states,
            logits_to_keep,
            is_thd=is_thd,
            output_hidden_states=output_hidden_states,
        )

    def update_moe_gate_bias(self) -> None:
        self.model.update_moe_gate_bias()

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or (
            torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.init_weights(buffer_device)
        final_out_std = self.config.hidden_size**-0.5
        cutoff_factor = 3
        with buffer_device:
            nn.init.trunc_normal_(
                self.lm_head.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))


ModelClass = KimiLinearForCausalLM

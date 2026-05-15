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

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import _has_dtensor_params, cast_model_to_dtype
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig
from nemo_automodel.components.models.mimo_v2_flash.state_dict_adapter import MiMoV2FlashStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.components.moe.state_dict_utils import is_dtensor
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


def _dtensor_to_full_tensor(value: torch.Tensor) -> torch.Tensor:
    if not is_dtensor(value):
        return value

    result = value.to_local().contiguous()
    for mesh_dim, placement in enumerate(value.placements):
        placement_name = placement.__class__.__name__
        if placement_name == "Replicate":
            continue
        group = value.device_mesh.get_group(mesh_dim)
        if placement_name == "Shard":
            gathered = [torch.empty_like(result) for _ in range(value.device_mesh.size(mesh_dim))]
            dist.all_gather(gathered, result, group=group)
            result = torch.cat(gathered, dim=placement.dim).contiguous()
        elif placement_name == "Partial":
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        else:
            raise NotImplementedError(f"Unsupported DTensor placement for MiMo gate: {placement}")
    return result


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
    probs = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)

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
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq.to(dtype=dtype), persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class MiMoV2FlashAttention(nn.Module):
    """MiMo-V2-Flash attention with full and sliding-window variants."""

    def __init__(self, config: MiMoV2FlashConfig, backend: BackendConfig, is_swa: bool, layer_idx: int):
        super().__init__()
        self.config = config
        self.backend = backend
        self.layer_idx = layer_idx
        self.is_swa = is_swa

        if is_swa:
            self.head_dim = config.swa_head_dim
            self.v_head_dim = config.swa_v_head_dim
            self.num_attention_heads = config.swa_num_attention_heads
            self.num_key_value_heads = config.swa_num_key_value_heads
        else:
            self.head_dim = config.head_dim
            self.v_head_dim = config.v_head_dim
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads

        self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
        self.rope_dim = self.rope_dim - (self.rope_dim % 2)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = float(config.attention_dropout or 0.0)
        self.scaling = self.head_dim**-0.5
        self.v_scale = getattr(config, "attention_value_scale", None)

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
        if has_sink:
            self.register_buffer("attention_sink_bias", torch.empty(config.num_attention_heads, dtype=torch.float32))
        else:
            self.attention_sink_bias = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        query_rope, query_nope = query_states.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        key_rope, key_nope = key_states.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        query_rope, key_rope = _apply_rotary_pos_emb(query_rope, key_rope, cos, sin)
        query_states = torch.cat([query_rope, query_nope], dim=-1)
        key_states = torch.cat([key_rope, key_nope], dim=-1)

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
        if self.attention_sink_bias is not None:
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
            self.mlp = MiMoV2MoE(moe_config, backend)
            self.mlp.gate.weight = nn.Parameter(
                self.mlp.gate.weight.to(torch.float32),
                requires_grad=self.mlp.gate.weight.requires_grad,
            )
        else:
            self.mlp = MLP(
                config.hidden_size,
                config.intermediate_size,
                backend.linear,
                dtype=dtype,
                activation="swiglu",
                bias=False,
            )

        self.input_layernorm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
        self.post_attention_layernorm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
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


class MiMoV2MoE(MoE):
    """MiMo MoE wrapper that preserves fp32 routing weights.

    The HF reference computes sigmoid router probabilities and the weighted
    expert accumulation in fp32, only casting the final MoE result back to the
    hidden-state dtype.  Automodel's generic Gate normally returns weights in
    the activation dtype, so MiMo keeps the routing probabilities in fp32 before
    passing them to the selected expert backend.
    """

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        cp_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        if cp_mesh is None:
            cp_mesh = self.cp_mesh

        shape = x.size()
        x = x.view(-1, self.dim)
        if padding_mask is not None:
            token_mask = (~padding_mask).flatten()
        else:
            token_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        weights, indices = self._gate_forward_hf(x)
        if self.backend.dispatcher == "torch" and hasattr(self.experts, "gate_and_up_projs"):
            y = self._forward_hf_loop(x, token_mask, weights, indices)
            return y.view(shape)

        y = self.experts(x, token_mask, weights, indices)
        return y.view(shape)

    def _gate_forward_hf(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_weight = _dtensor_to_full_tensor(self.gate.weight)
        scores = F.linear(x.float(), gate_weight.float(), bias=None).sigmoid()
        if self.gate.score_func != "sigmoid":
            raise NotImplementedError(f"Unsupported MiMo gate score function: {self.gate.score_func}")

        scores_for_choice = scores
        if self.gate.e_score_correction_bias is not None:
            correction_bias = _dtensor_to_full_tensor(self.gate.e_score_correction_bias)
            scores_for_choice = scores_for_choice + correction_bias.float().unsqueeze(0)

        if self.gate.n_groups > 1:
            scores_for_choice = scores_for_choice.view(x.size(0), self.gate.n_groups, -1)
            group_scores = scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1)
            group_idx = torch.topk(group_scores, k=self.gate.topk_groups, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand_as(scores_for_choice).reshape(x.size(0), -1)
            scores_for_choice = scores_for_choice.reshape(x.size(0), -1).masked_fill(~score_mask.bool(), float("-inf"))

        indices = torch.topk(scores_for_choice, k=self.gate.topk, dim=-1, sorted=False)[1]
        weights = scores.gather(1, indices)
        if self.gate.norm_topk_prob and self.gate.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.gate.route_scale
        return weights, indices

    def _forward_hf_loop(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        gate_and_up_projs = self.experts.gate_and_up_projs
        down_projs = self.experts.down_projs
        if is_dtensor(gate_and_up_projs):
            ep_mesh = gate_and_up_projs.device_mesh
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
            gate_and_up_projs = gate_and_up_projs.to_local()
            down_projs = down_projs.to_local()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        gate_and_up_projs = gate_and_up_projs.to(x.dtype)
        down_projs = down_projs.to(x.dtype)
        local_num_tokens = x.size(0)

        if ep_size > 1:
            ep_group = ep_mesh.get_group()
            local_len = torch.tensor([local_num_tokens], device=x.device, dtype=torch.int64)
            gathered_len_t = [torch.zeros_like(local_len) for _ in range(ep_size)]
            dist.all_gather(gathered_len_t, local_len, group=ep_group)
            gathered_lens = [int(t.item()) for t in gathered_len_t]
            max_len = max(gathered_lens)

            def _all_gather_dim0_var(local_tensor: torch.Tensor) -> torch.Tensor:
                if max_len > local_tensor.size(0):
                    pad_shape = (max_len - local_tensor.size(0),) + tuple(local_tensor.shape[1:])
                    pad = torch.zeros(pad_shape, dtype=local_tensor.dtype, device=local_tensor.device)
                    local_tensor = torch.cat([local_tensor, pad], dim=0)
                gathered = [torch.empty_like(local_tensor) for _ in range(ep_size)]
                dist.all_gather(gathered, local_tensor, group=ep_group)
                gathered = [g[:n] for g, n in zip(gathered, gathered_lens)]
                return torch.cat(gathered, dim=0)

            x = _all_gather_dim0_var(x)
            weights = _all_gather_dim0_var(weights.float())
            indices = _all_gather_dim0_var(indices)
            token_mask = _all_gather_dim0_var(token_mask)

        y = torch.zeros_like(x, dtype=torch.float32)
        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        for expert_idx in range(experts_start_idx, experts_end_idx):
            selected = (indices == expert_idx) & token_mask.unsqueeze(-1)
            token_indices, weight_indices = torch.where(selected)
            if token_indices.numel() == 0:
                continue

            local_expert_idx = expert_idx - experts_start_idx
            expert_input = x[token_indices]
            gate_and_up = gate_and_up_projs[local_expert_idx]
            gate_weight, up_weight = torch.chunk(gate_and_up, 2, dim=-1)
            gate = expert_input @ gate_weight
            up = expert_input @ up_weight
            expert_output = (F.silu(gate) * up) @ down_projs[local_expert_idx]
            expert_weights = weights[token_indices, weight_indices]
            y.index_add_(0, token_indices, expert_output * expert_weights.unsqueeze(-1))

        if ep_size > 1:
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()

        return y.to(x.dtype)


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
            score_func=config.scoring_func,
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0.0,
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
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
        self.norm = MiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype)
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

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
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
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if inputs_embeds is None:
            if self.embed_tokens is not None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds must be provided")
                inputs_embeds = self.embed_tokens(input_ids)
            else:
                inputs_embeds = input_ids
        if inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided")

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if padding_mask is None and isinstance(attention_mask, torch.Tensor):
            padding_mask = _derive_padding_mask(attention_mask)

        causal_mask_mapping = self._build_causal_mask_mapping(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        swa_position_embeddings = self.swa_rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers.values():
            layer_position_embeddings = (
                swa_position_embeddings if decoder_layer.attention_type == "sliding_attention" else position_embeddings
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=layer_position_embeddings,
                padding_mask=padding_mask,
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

    _keep_in_fp32_modules_strict = ["mlp.gate.weight", "mlp.gate.e_score_correction_bias", "attention_sink_bias"]
    _pp_keep_self_forward = True
    _skip_init_weights_on_load = True

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

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **kwargs,
        )
        if self.lm_head is None:
            return hidden
        if isinstance(logits_to_keep, int):
            hidden = hidden[:, -logits_to_keep:, :] if logits_to_keep else hidden
        else:
            hidden = hidden[:, logits_to_keep, :]
        return self.lm_head(hidden)

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device)
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


ModelClass = MiMoV2FlashForCausalLM

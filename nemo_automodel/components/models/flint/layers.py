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

"""Flint's sink-aware attention and shared dense/MoE decoder blocks.

Eager, SDPA and TE attention have distinct numerical validation scopes. Magi
uses model-owned packed token plans and requires an upstream learned-sink
training kernel. FP32 component parity does not establish full-model parity
with the serving kernels' BF16 rounding order.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

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
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.shared.utils import dtype_from_str


def _validate_backend(backend: BackendConfig) -> None:
    """Reject backend choices outside the supported native component contract.

    Args:
        backend: Shared backend settings, preserved without mutation.
    """
    if backend.attn not in ("eager", "sdpa", "te", "magi"):
        raise ValueError("Flint attention requires eager, sdpa, te, or magi with differentiable learned sinks")
    if backend.linear != "torch" or backend.rms_norm not in ("torch", "torch_fp32"):
        raise ValueError("Flint requires torch linear and torch/torch_fp32 RMSNorm")
    if backend.rope != "torch" or backend.rope_fusion:
        raise ValueError("Flint currently requires rope='torch' and rope_fusion=False")
    if backend.experts not in ("torch", "torch_mm") or backend.dispatcher not in ("torch", "deepep", "hybridep"):
        raise ValueError("Flint requires torch/torch_mm experts and torch/deepep/hybridep dispatch")
    if backend.gate_precision is not None or backend.fake_balanced_gate:
        raise ValueError("Flint requires gate_precision=None and fake_balanced_gate=False")
    if backend.cuda_graph.modules or backend.compile_router_weight or backend.shared_expert_overlap:
        raise ValueError(
            "Flint currently requires CUDA graphs, compiled router weighting and shared-expert overlap off"
        )


class _FlintSinkParameters(nn.Module):
    """Own the trainable FP32 denominator logits in a separate dtype unit."""

    def __init__(self, num_heads: int) -> None:
        """Allocate one FP32 scalar per query head.

        Args:
            num_heads: Number of query heads, including within GQA groups.
        """
        super().__init__()
        self.softmax_offset = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Return unscaled sink logits through the module's normal call path.

        Returns:
            FP32 tensor of shape [query_heads], with independent storage and
            gradients to the parameter. The clone remains valid when nested
            FSDP reshards/frees its gathered parameter after this call returns.
        """
        return self.softmax_offset.clone()


class FlintAttention(nn.Module):
    """Separate Q/K/V, per-head RMSNorm, selective Neox RoPE and learned sinks."""

    _keep_in_fp32_modules_strict = ["_fp32_params"]

    def __init__(self, config: FlintConfig, layer_idx: int, backend: BackendConfig) -> None:
        """Create one attention layer without allocating inference caches.

        Args:
            config: Typed architecture settings with explicit head dimension.
            layer_idx: Index into the configuration's full/sliding schedule.
            backend: Supported eager/torch backend settings.
        """
        super().__init__()
        _validate_backend(backend)
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError("layer_idx must index an existing Flint layer")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.eps = config.rms_norm_eps
        self.initializer_range = config.initializer_range
        self.backend = backend
        self.attention_compute_dtype = torch.float16 if config.attention_compute_dtype == "fp16" else None
        if self.attention_compute_dtype is not None and backend.attn != "te":
            raise ValueError("Flint explicit attention_compute_dtype requires the TE backend")
        self._te_cp_size = 1
        self.rope_theta = float(config.rope_parameters["rope_theta"])
        self.use_rope = layer_idx not in config.no_rope_layer_indices
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        dtype = dtype_from_str(config.dtype, torch.float32)
        self.q_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_heads * self.head_dim, dtype=dtype
        )
        self.k_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_kv_heads * self.head_dim, dtype=dtype
        )
        self.v_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.num_kv_heads * self.head_dim, dtype=dtype
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, self.hidden_size, dtype=dtype
        )
        self.q_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=self.eps, dtype=dtype)
        self.k_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=self.eps, dtype=dtype)
        if backend.attn == "te":
            attn_module, _ = initialize_attn_module_and_func(
                "te",
                self.num_heads,
                self.head_dim,
                self.head_dim,
                self.head_dim**-0.5,
                num_gqa_groups=self.num_kv_heads,
                softmax_type="learnable",
                attention_dropout=0.0,
            )
            if attn_module is None:
                raise RuntimeError("Transformer Engine did not construct a learned-sink attention module")
            # The public BF16 loader temporarily sets Torch's default dtype;
            # TE otherwise constructs this protected offset in BF16 as well.
            attn_module.float()
            # Register the shared DPA exactly once under the existing sink FQN.
            # This preserves HF/native sink conversion and gives FSDP a callable
            # unit spanning the complete saved-offset lifetime.
            attn_module._flint_keep_sink_gathered = True
            attn_module._flint_te_cp_size = 1
            self._fp32_params = attn_module
        else:
            self._fp32_params = _FlintSinkParameters(self.num_heads)
        self._sdpa_func = None
        if backend.attn == "sdpa":
            _, self._sdpa_func = initialize_attn_module_and_func(
                "sdpa",
                self.num_heads,
                self.head_dim,
                self.head_dim,
                self.head_dim**-0.5,
                num_gqa_groups=self.num_kv_heads,
            )

    @property
    def attn_module(self) -> nn.Module | None:
        """Expose shared TE attention for CP discovery without a second state path."""
        return self._fp32_params if self.backend.attn == "te" else None

    def _apply_rope(self, values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Rotate corresponding channels in the two halves of each head.

        Args:
            values: Tensor [batch, sequence, heads, head_dim] or THD [tokens, heads, head_dim].
            position_ids: Integer tensor [batch, sequence] or [tokens] on values' device.

        Returns:
            Rotated tensor of shape [batch, sequence, heads, head_dim], in values'
            dtype. Frequencies are freshly computed in FP32, never stored BF16.
        """
        frequencies = self.rope_theta ** (
            -torch.arange(0, self.head_dim, 2, device=values.device, dtype=torch.float32) / self.head_dim
        )
        angles = position_ids.float().unsqueeze(-1) * frequencies
        cosine, sine = angles.cos().unsqueeze(-2), angles.sin().unsqueeze(-2)
        first, second = values.float().chunk(2, dim=-1)
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1).to(values.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Compute full-sequence causal attention with a zero-value sink category.

        Args:
            hidden_states: Floating projection-dtype tensor [batch, sequence,
                hidden_size], or [tokens, hidden_size] for explicit THD packing.
            position_ids: Optional integer tensor of shape [batch, sequence] or
                document-local [tokens] for THD;
                contiguous zero-based positions are used by default.
            attention_mask: Optional binary tensor of shape [batch, sequence],
                with one/True for valid tokens. Both padded queries and keys are
                excluded from real attention; the sink remains valid. TE BSHD
                supports trailing padding; arbitrary padding uses THD packing.
            **attn_kwargs: Shared TE THD boundaries, padding metadata and formats,
                model-owned Magi plan, or integer document_ids [batch, sequence]
                for the eager/SDPA block-diagonal mask.

        Returns:
            Tensor of shape [batch, sequence, hidden_size] in the output
            projection's compute dtype, equal to hidden_states' dtype when
            autocast is disabled. An explicit TE FP16 attention boundary restores
            this projection dtype before the output linear. The input is neither
            mutated nor aliased.
        """
        is_thd = attn_kwargs.get("qkv_format") == "thd"
        expected_ndim = 2 if is_thd else 3
        if hidden_states.ndim != expected_ndim or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("Flint attention expects [batch, sequence, hidden_size] or explicit THD [tokens, hidden]")
        leading = hidden_states.shape[:-1]
        batch, sequence = (1, hidden_states.shape[0]) if is_thd else hidden_states.shape[:2]
        if batch == 0 or sequence == 0:
            raise ValueError("Flint attention requires nonempty batch and sequence axes")
        if hidden_states.dtype != self.q_proj.weight.dtype:
            raise ValueError("Flint hidden_states dtype must match the projection weights")
        if position_ids is None and is_thd:
            raise ValueError("Packed Flint attention requires document-local position_ids")
        if position_ids is None:
            position_ids = torch.arange(sequence, device=hidden_states.device).expand(batch, -1)
        expected_positions = (sequence,) if is_thd else (batch, sequence)
        if position_ids.shape != expected_positions or position_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Flint position_ids must be an integer tensor [batch, sequence]")
        if position_ids.device != hidden_states.device or torch.any(position_ids < 0):
            raise ValueError("Flint position_ids must be nonnegative and on hidden_states' device")
        if attention_mask is not None and not is_thd:
            if attention_mask.shape != (batch, sequence) or attention_mask.device != hidden_states.device:
                raise ValueError("Flint attention_mask must have shape [batch, sequence] on hidden_states' device")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("Flint attention_mask must contain only zero/one values")
        query = self.q_norm(self.q_proj(hidden_states).reshape(*leading, self.num_heads, self.head_dim))
        key = self.k_norm(self.k_proj(hidden_states).reshape(*leading, self.num_kv_heads, self.head_dim))
        value = self.v_proj(hidden_states).reshape(*leading, self.num_kv_heads, self.head_dim)
        projection_dtype = value.dtype
        if self.use_rope:
            query = self._apply_rope(query, position_ids)
            key = self._apply_rope(key, position_ids)
        if self.attention_compute_dtype is not None:
            if projection_dtype != torch.float32:
                raise ValueError("Flint explicit attention_compute_dtype requires actual FP32 projection compute")
            query, key, value = (tensor.to(self.attention_compute_dtype) for tensor in (query, key, value))
        if self.backend.attn == "magi":
            plan = attn_kwargs.get("_flint_magi_plan")
            if not is_thd or plan is None:
                raise ValueError("Flint Magi attention requires model-prepared THD token ownership and mask keys")
            output = plan.attention(
                query,
                key,
                value,
                sink=self._fp32_params(),
                window=self.sliding_window or 0,
                scale=self.head_dim**-0.5,
            )
            return self.o_proj(output.reshape(*leading, self.num_heads * self.head_dim))
        if self.backend.attn == "te":
            if attn_kwargs.get("document_ids") is not None:
                raise ValueError("Flint TE packed documents require THD cu_seqlens metadata")
            if attention_mask is not None and bool(
                (~attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()).any()
            ):
                raise ValueError("Flint TE BSHD padding must be trailing; left/holey padding requires packed THD")
            window = (-1, 0) if self.sliding_window is None else (self.sliding_window - 1, 0)
            module = self._fp32_params
            if getattr(module, "cp_group", None) is not None:
                # TE's non-vanilla softmax CP path supports A2A head exchange,
                # unlike the P2P/AllGather transports used by vanilla attention.
                # Every KV group must divide evenly across the context ranks.
                cp_size = module.cp_group.size()
                if self.num_heads % cp_size or self.num_kv_heads % cp_size:
                    raise ValueError(
                        "Flint learned-sink TE CP uses A2A and requires CP size to divide both Q and KV heads"
                    )
                communication = "a2a"
                if module.cp_comm_type != communication:
                    module.set_context_parallel_group(
                        module.cp_group,
                        module.cp_global_ranks,
                        module.cp_stream,
                        cp_comm_type=communication,
                    )
            query, key, value, te_kwargs = preprocess_args_and_kwargs_for_attn(
                query, key, value, attention_mask, "te", window_size=window, **attn_kwargs
            )
            output = module(query, key, value, **te_kwargs)
            output = postprocess_output_for_attn(output, "te")
            width = self.num_heads * self.head_dim
            if is_thd and output.numel() != sequence * width:
                # TE can return compact valid queries for logical cu_seqlens
                # plus separate physical offsets. Restore the residual's physical
                # rows through a differentiable scatter, preserving actual kernel
                # masking and keeping token-id equality out of padding decisions.
                padding = attn_kwargs.get("padding_mask")
                if padding is None or padding.shape != (sequence,):
                    raise ValueError("Compact Flint TE output requires physical THD padding_mask [tokens]")
                indices = (~padding.bool()).nonzero(as_tuple=False).flatten()
                if output.numel() != indices.numel() * width:
                    raise ValueError("Flint TE compact output rows do not match authoritative padding metadata")
                output = output.new_zeros(sequence, width).index_copy(0, indices, output.reshape(-1, width))
            return self.o_proj(output.reshape(*leading, width).to(projection_dtype))
        if is_thd:
            raise ValueError("Packed THD Flint attention currently requires Transformer Engine")
        positions = torch.arange(sequence, device=hidden_states.device)
        distances = positions[:, None] - positions[None, :]
        allowed = (distances >= 0).expand(batch, -1, -1)
        if self.sliding_window is not None:
            allowed = allowed & (distances < self.sliding_window)
        if attention_mask is not None:
            valid = attention_mask.bool()
            allowed = allowed & valid.unsqueeze(1) & valid.unsqueeze(2)
        document_ids = attn_kwargs.get("document_ids")
        if document_ids is not None:
            if document_ids.shape != (batch, sequence):
                raise ValueError("Flint document_ids must have shape [batch, sequence]")
            allowed = allowed & (document_ids.unsqueeze(1) == document_ids.unsqueeze(2))
            allowed = allowed & (document_ids > 0).unsqueeze(1) & (document_ids > 0).unsqueeze(2)
        repeats = self.num_heads // self.num_kv_heads
        query = query.transpose(1, 2).float()
        key = key.repeat_interleave(repeats, dim=2).transpose(1, 2).float()
        value = value.repeat_interleave(repeats, dim=2).transpose(1, 2).float()
        # Explicit casts alone do not defeat an enclosing autocast matmul policy.
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            sinks = self._fp32_params().reshape(1, self.num_heads, 1, 1).expand(batch, -1, sequence, -1)
            if self.backend.attn == "sdpa":
                # A zero-value sentinel category is algebraically the learned
                # sink, including dQ/dK and sink gradients. No detached LSE is
                # used to approximate the changed normalization denominator.
                sentinel_key = key.new_zeros(batch, self.num_heads, 1, self.head_dim)
                sentinel_value = value.new_zeros(batch, self.num_heads, 1, self.head_dim)
                real_bias = query.new_zeros(batch, self.num_heads, sequence, sequence)
                real_bias = real_bias.masked_fill(~allowed.unsqueeze(1), -torch.inf)
                bias = torch.cat((real_bias, sinks), dim=-1)
                output = self._sdpa_func(
                    query,
                    torch.cat((key, sentinel_key), dim=-2),
                    torch.cat((value, sentinel_value), dim=-2),
                    attn_mask=bias,
                    is_causal=False,
                    enable_gqa=False,
                )
                output = output.transpose(1, 2).reshape(batch, sequence, self.num_heads * self.head_dim)
            else:
                scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dim)
                scores = scores.masked_fill(~allowed.unsqueeze(1), -torch.inf)
                probabilities = torch.softmax(torch.cat((scores, sinks), dim=-1), dim=-1)[..., :-1]
                output = (
                    (probabilities @ value).transpose(1, 2).reshape(batch, sequence, self.num_heads * self.head_dim)
                )
        return self.o_proj(output.to(projection_dtype))

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float | None = None) -> None:
        """Initialize projections, head-normalization scales and trainable sinks.

        Args:
            buffer_device: Device context after any meta materialization.
            init_std: Optional normal-initialization deviation for projections.
        """
        std = self.initializer_range if init_std is None else init_std
        with buffer_device:
            for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                nn.init.normal_(linear.weight, mean=0.0, std=std)
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()
            nn.init.zeros_(self._fp32_params.softmax_offset)


class FlintDecoderLayer(nn.Module):
    """Pre-norm attention and a dense or shared NeMo MoE, each with a residual."""

    def __init__(self, config: FlintConfig, layer_idx: int, backend: BackendConfig) -> None:
        """Construct one block using the explicit dense/MoE schedule.

        Args:
            config: Typed Flint architecture and CE-only training settings.
            layer_idx: Index into the configured layer schedule.
            backend: Supported eager/torch settings; passed to shared components.
        """
        super().__init__()
        self.self_attn = FlintAttention(config, layer_idx, backend)
        self.residual_dtype = config.residual_dtype
        self.layer_idx = layer_idx
        dtype = dtype_from_str(config.dtype, torch.float32)
        self.initializer_range = config.initializer_range
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.moe_config: MoEConfig | None = None
        self.mlp: MLP | MoE
        if layer_idx in config.mlp_only_layers:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)
        else:
            if config.num_experts_per_tok < 2:
                raise ValueError("Flint shared-MoE normalization currently requires num_experts_per_tok >= 2")
            if config.router_aux_loss_coef != 0:
                raise ValueError(
                    "Flint currently supports the explicitly selected CE-only objective (router_aux_loss_coef=0)"
                )
            self.moe_config = MoEConfig(
                dim=config.hidden_size,
                inter_dim=config.intermediate_size,
                moe_inter_dim=config.moe_intermediate_size,
                n_routed_experts=config.num_experts,
                n_shared_experts=1,
                n_activated_experts=config.num_experts_per_tok,
                n_expert_groups=0,
                n_limited_groups=0,
                train_gate=True,
                gate_bias_update_factor=0.0,
                aux_loss_coeff=0.0,
                score_func="sigmoid_with_bias",
                route_scale=1.0,
                norm_topk_prob=True,
                router_weights_fp32=True,
                router_bias=False,
                expert_bias=False,
                force_e_score_correction_bias=False,
                shared_expert_gate=False,
                shared_expert_inter_dim=config.shared_expert_intermediate_size,
                apply_router_weight_after_down=True,
                dtype=dtype,
            )
            self.mlp = MoE(self.moe_config, backend)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Apply the block while preserving the shared MoE's padding convention.

        Args:
            hidden_states: Residual tensor [batch, sequence, hidden_size] or
                THD [tokens, hidden_size]. Explicit residual_dtype preserves its
                chosen dtype while normalized branches use projection dtype.
            position_ids: Integer [batch, sequence] or document-local [tokens].
            attention_mask: Optional binary [batch, sequence], with one for valid
                tokens; shared MoE instead receives True for padding.
            padding_mask: Optional True-for-padding expert mask [batch, sequence]
                or THD [tokens], derived from real/padded document lengths.
            **attn_kwargs: Shared TE metadata or an immutable model-owned Magi
                plan retained across activation-checkpoint recomputation.

        Returns:
            Residual tensor in the input layout and configured stream dtype. Padded-query outputs
            are unspecified and must be excluded from the training objective.
        """
        stream_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(self.residual_dtype)
        if stream_dtype is not None:
            hidden_states = hidden_states.to(stream_dtype)
        branch_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states).to(branch_dtype),
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        if stream_dtype is not None:
            hidden_states = hidden_states.to(stream_dtype)
        normalized = self.post_attention_layernorm(hidden_states).to(branch_dtype)
        if isinstance(self.mlp, MoE):
            if padding_mask is None and attention_mask is not None:
                padding_mask = ~attention_mask.bool()
            mlp_output = self.mlp(normalized, padding_mask=padding_mask)
        else:
            mlp_output = self.mlp(normalized)
        hidden_states = hidden_states + mlp_output
        return hidden_states.to(stream_dtype) if stream_dtype is not None else hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float | None = None) -> None:
        """Initialize every component without depending on a base checkpoint.

        Args:
            buffer_device: Device context after any meta materialization.
            init_std: Optional normal-initialization deviation for projections/experts.
        """
        std = self.initializer_range if init_std is None else init_std
        self.self_attn.init_weights(buffer_device, std)
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.mlp.init_weights(buffer_device, std)

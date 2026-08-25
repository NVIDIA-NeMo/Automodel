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

"""Diffusion-specific layers for ``diffusion_gemma``.

The stateless leaf layers (RMSNorm, the per-layer-type rotary embedding, the
dense SwiGLU MLP, the self-conditioning gated MLP, and the RoPE/GQA helpers) are
**imported directly from the released transformers ``diffusion_gemma``
implementation** so the model tracks Google's release. This module keeps only
the pieces the reference implementation cannot provide:

* :class:`DiffusionGemmaAttention` — a single mask-driven attention used by both
  the causal (encoder) and bidirectional (decoder) passes of AM's shared stack.
  Unlike the reference's two ``Cache``-coupled attention classes, it returns the
  freshly computed ``(K, V)`` as plain tensors and accepts ``encoder_kv`` as
  plain tensors, so the backbone can thread KV between the two passes without a
  HF ``Cache`` object. ``scaling = 1.0`` (per-head scale folded into
  ``q_norm``/``k_norm``); full-attention layers have no ``v_proj`` (values reuse
  the keys), sliding layers do.
* :class:`DiffusionGemmaMoEDecoderLayer` — composes the reference's attention +
  norms + MLP with NeMo's ``Gemma4MoE`` (``GroupedExperts`` + ``Gemma4Gate``)
  instead of the reference's dense-matmul ``DiffusionGemmaTextExperts``, which
  does not shard under FSDP. The dense MLP and the MoE branch run in parallel
  and are summed, routing on the unnormalized post-attention residual — same as
  ``gemma4_moe``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaSelfConditioning as DiffusionGemmaSelfConditioning,
)
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaText4MLP as DiffusionGemmaMLP,
)
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaTextRotaryEmbedding as DiffusionGemmaTextRotaryEmbedding,
)

from nemo_automodel.components.attention.utils import initialize_attn_module_and_func
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import Gemma4MoE
from nemo_automodel.components.moe.layers import MoEConfig

# Leaf layers are reused directly from the released transformers diffusion_gemma
# implementation so the model tracks Google's release. ``DiffusionGemmaMLP`` is
# the reference's ``DiffusionGemmaText4MLP``. ``DiffusionGemmaRMSNorm`` /
# ``DiffusionGemmaTextRotaryEmbedding`` / ``DiffusionGemmaSelfConditioning`` are
# re-exported here because ``model.py`` and the backbone compose them.


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Eager scaled-dot-product attention with an additive 4-D mask.

    The mask is expected to be additive (``0`` keep, ``-inf`` mask) and already
    sliced to the layer's key axis (``[B, 1, Lq, Lkv]``). No softcap is applied
    to attention scores (Gemma4 only softcaps the final logits).
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


class DiffusionGemmaAttention(nn.Module):
    """Diffusion attention shared by the causal (encoder) and bidirectional
    (decoder) passes.

    ``is_causal`` is informational only — the actual causal/bidirectional/
    block-diagonal structure is provided by the additive ``attention_mask`` the
    caller passes. When ``encoder_kv`` is supplied (the bidirectional canvas
    pass), the layer concatenates ``[encoder_K ; canvas_K]`` on the key axis and
    returns the freshly computed canvas K/V so the caller can build the encoder
    KV cache during the causal pass.
    """

    def __init__(self, config: DiffusionGemmaTextConfig, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim
        num_key_value_heads = config.num_key_value_heads if self.is_sliding else config.num_global_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = config.attention_dropout
        self.backend = backend

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # Full-attention layers have no v_proj; values reuse the (pre-norm) keys.
        self.v_proj = (
            nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
            if self.is_sliding
            else None
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = DiffusionGemmaRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps, with_scale=False)

        self.attn_module = None
        if backend.attn == "te":
            self.attn_module, _ = initialize_attn_module_and_func(
                attn_impl="te",
                num_attention_heads=self.num_attention_heads,
                num_qk_channels=self.head_dim,
                num_v_channels=self.head_dim,
                softmax_scale=self.scaling,
                attn_mask_type="causal",
                qkv_format="bshd",
                num_gqa_groups=self.num_key_value_heads,
                attention_dropout=self.attention_dropout,
            )

    def _set_cp_transport(self, comm_type: str) -> None:
        module = self.attn_module
        if module is None or getattr(module, "cp_group", None) is None:
            return
        if getattr(module, "cp_comm_type", None) == comm_type:
            return
        module.set_context_parallel_group(
            module.cp_group,
            getattr(module, "cp_global_ranks", None),
            getattr(module, "cp_stream", None),
            cp_comm_type=comm_type,
        )

    def _set_attention_type(self, attention_type: str) -> None:
        """Switch the shared TE DPA between encoder self- and decoder cross-attention."""
        module = self.attn_module
        if module is None or getattr(module, "attention_type", None) == attention_type:
            return
        if hasattr(module, "fast_setattr"):
            module.fast_setattr("attention_type", attention_type)
        else:
            module.attention_type = attention_type
        for backend_name in ("flash_attention", "fused_attention", "unfused_attention"):
            backend = getattr(module, backend_name, None)
            if backend is not None:
                backend.attention_type = attention_type

    def _gather_canvas(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Differentiably gather BSHD canvas K/V and restore global order."""
        group = getattr(self.attn_module, "cp_group", None)
        if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
            return tensor
        from torch.distributed.nn.functional import all_gather

        parts = list(all_gather(tensor.contiguous(), group=group))
        index_parts = [torch.empty_like(indices) for _ in range(dist.get_world_size(group))]
        dist.all_gather(index_parts, indices.contiguous(), group=group)
        gathered = torch.cat(parts, dim=1)
        order = torch.argsort(torch.cat(index_parts).to(gathered.device))
        return gathered.index_select(1, order)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        encoder_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        cp_encoder_indices: torch.Tensor | None = None,
        cp_canvas_indices: torch.Tensor | None = None,
        cp_encoder_length: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)

        value_states = self.v_norm(value_states)

        use_te = self.attn_module is not None
        if not use_te:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        # Freshly computed canvas/clean K/V before any cross-attention concat.
        # The causal pass returns these to populate the read-only encoder KV
        # cache; the bidirectional pass prepends them as ``encoder_kv``.
        layer_kv = (key_states, value_states)

        if encoder_kv is not None and use_te and cp_encoder_indices is not None:
            if cp_canvas_indices is None or cp_encoder_length is None:
                raise ValueError("DiffusionGemma TE CP decode requires canvas indices and encoder length.")
            # The encoder cache is already partitioned as [encoder ; canvas-pad].
            # Replace local canvas-pad slots with the actual per-layer canvas K/V.
            full_key = self._gather_canvas(key_states, cp_canvas_indices)
            full_value = self._gather_canvas(value_states, cp_canvas_indices)
            enc_key, enc_value = encoder_kv
            replace = (cp_encoder_indices >= cp_encoder_length) & (
                cp_encoder_indices < cp_encoder_length + full_key.shape[1]
            )
            destinations = replace.nonzero(as_tuple=False).flatten()
            sources = cp_encoder_indices[replace] - cp_encoder_length
            key_states = enc_key.index_copy(1, destinations, full_key.index_select(1, sources))
            value_states = enc_value.index_copy(1, destinations, full_value.index_select(1, sources))
        elif encoder_kv is not None:
            enc_key, enc_value = encoder_kv
            seq_dim = 1 if use_te else 2
            key_states = torch.cat([enc_key, key_states], dim=seq_dim)
            value_states = torch.cat([enc_value, value_states], dim=seq_dim)

        if use_te:
            if encoder_kv is None:
                self._set_attention_type("self")
                self._set_cp_transport("all_gather" if self.is_sliding else "p2p")
                attn_output = self.attn_module(
                    query_states,
                    key_states,
                    value_states,
                    qkv_format="bshd",
                    attn_mask_type="causal",
                    window_size=(self.sliding_window - 1, 0) if self.is_sliding else (-1, 0),
                )
            else:
                self._set_attention_type("cross")
                self._set_cp_transport("p2p")
                attn_output = self.attn_module(
                    query_states,
                    key_states,
                    value_states,
                    qkv_format="bshd",
                    attn_mask_type="no_mask",
                    window_size=(-1, -1),
                    core_attention_bias_type="post_scale_bias",
                    core_attention_bias=attention_mask,
                )
        else:
            attn_output = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, layer_kv


def _build_moe_config(config: DiffusionGemmaTextConfig, moe_config: MoEConfig | None) -> MoEConfig:
    """Build a NeMo :class:`MoEConfig` from the DiffusionGemma text config.

    Matches ``gemma4_moe``'s defaults: geglu experts, softmax routing,
    ``train_gate=True`` (the recipe freezes the gate separately), no aux loss.
    """
    if moe_config is not None:
        return moe_config
    # Honor the model's configured master dtype (model.torch_dtype). Without an
    # explicit dtype, MoEConfig falls back to its bf16 default, so the
    # GroupedExperts — ~88% of the params — are materialized bf16 even when the
    # recipe requests an fp32 master (model.torch_dtype: float32). That left AdamW
    # optimizing bf16 expert params (the "trainable bf16 params" warning) while the
    # rest of the model was fp32. Thread the configured dtype through here.
    cfg_dtype = getattr(config, "torch_dtype", None)
    if isinstance(cfg_dtype, str):
        from nemo_automodel.shared.utils import dtype_from_str

        cfg_dtype = dtype_from_str(cfg_dtype, torch.bfloat16)
    expert_dtype = cfg_dtype if isinstance(cfg_dtype, torch.dtype) else torch.bfloat16
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=0,
        n_activated_experts=config.top_k_experts,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_activation="geglu",
        softmax_before_topk=False,
        dtype=expert_dtype,
    )


class DiffusionGemmaMoEDecoderLayer(nn.Module):
    """Single shared decoder layer used by both the causal and bidirectional passes.

    Reuses NeMo's ``Gemma4MoE`` (``GroupedExperts`` + ``Gemma4Gate``) for the MoE
    branch; the dense MLP runs in parallel and the two are summed. ``layer_scalar``
    is a per-layer output scale (identity unless present in the checkpoint).
    """

    def __init__(
        self,
        config: DiffusionGemmaTextConfig,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        self.self_attn = DiffusionGemmaAttention(config=config, layer_idx=layer_idx, backend=backend)
        # ``DiffusionGemmaMLP`` is the reference's ``DiffusionGemmaText4MLP`` (takes layer_idx).
        self.mlp = DiffusionGemmaMLP(config, layer_idx)

        self.input_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_1 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = DiffusionGemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.moe = Gemma4MoE(moe_config, backend, config)

        # Per-layer output scaling. Registered on every layer so DCP can always
        # load it; ones (identity) when the checkpoint has no value.
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        encoder_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        cp_encoder_indices: torch.Tensor | None = None,
        cp_canvas_indices: torch.Tensor | None = None,
        cp_encoder_length: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, layer_kv = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            encoder_kv=encoder_kv,
            cp_encoder_indices=cp_encoder_indices,
            cp_canvas_indices=cp_canvas_indices,
            cp_encoder_length=cp_encoder_length,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Dense MLP + MoE in parallel (routing on the raw post-attention
        # residual; experts on its pre_feedforward_layernorm_2 norm).
        residual = hidden_states
        dense_out = self.pre_feedforward_layernorm(hidden_states)
        dense_out = self.mlp(dense_out)
        dense_out = self.post_feedforward_layernorm_1(dense_out)

        moe_input = self.pre_feedforward_layernorm_2(hidden_states)
        moe_out = self.moe(moe_input, padding_mask=padding_mask, gate_input=hidden_states)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]
        moe_out = self.post_feedforward_layernorm_2(moe_out)

        hidden_states = dense_out + moe_out
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, layer_kv

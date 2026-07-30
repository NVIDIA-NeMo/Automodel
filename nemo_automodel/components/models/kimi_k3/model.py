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

"""Native AutoModel implementation of the Moonshot Kimi K3 architecture."""

from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.packing import get_unpad_data, is_indexed_packed_mask
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.cp import (
    KimiPackedContext,
    all_gather_sequence,
    build_document_causal_mask,
    build_fla_cp_context,
    doc_ids_from_attention_mask,
    doc_ids_from_cu_seqlens,
    document_causal_flex_attention,
    shard_batch_for_kimi_cp,
)
from nemo_automodel.components.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import Gate, MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.import_utils import UnavailableError, safe_import_from
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

_FLA_MSG = "Kimi K3 requires the flash-linear-attention/fla extra. Install with `uv sync --extra fla`."
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
    _FUSED_KDA_GATE_HAS_LOWER_BOUND = _KDA_GATE_OK and "lower_bound" in inspect.signature(fused_kda_gate).parameters
except (TypeError, ValueError):
    _FUSED_KDA_GATE_HAS_G_BIAS = False
    _FUSED_KDA_GATE_HAS_LOWER_BOUND = False


def _require_fla() -> None:
    if not all((_SHORT_CONV_OK, _FUSED_RMSNORM_GATED_OK, _CHUNK_KDA_OK, _RECURRENT_KDA_OK, _KDA_GATE_OK)):
        raise UnavailableError(_FLA_MSG)


def _torch_kda_gate(
    g: torch.Tensor,
    a_log: torch.Tensor,
    head_dim: int,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Compute K3's KDA decay gate with torch FP32 operations.

    Args:
        g: Raw gate tensor of shape [batch, sequence, heads * head_dim] or
            [batch, sequence, heads, head_dim].
        a_log: Log decay tensor of shape [heads].
        head_dim: Per-head KDA dimension.
        dt_bias: Gate bias tensor of shape [heads * head_dim].
        lower_bound: Optional lower bound for K3's bounded decay function.

    Returns:
        FP32 decay tensor of shape [batch, sequence, heads, head_dim].
    """
    gate = g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)
    num_heads = gate.shape[-2]
    gate = gate.float() + dt_bias.float().view(num_heads, head_dim)
    decay = a_log.float().view(num_heads, 1).exp()
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(decay * gate)
    return -decay * F.softplus(gate)


def _fused_kda_gate(
    g: torch.Tensor,
    a_log: torch.Tensor,
    head_dim: int,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Call FLA's fused KDA gate across supported FLA APIs."""
    if _FUSED_KDA_GATE_HAS_G_BIAS:
        if lower_bound is not None:
            return _torch_kda_gate(g, a_log, head_dim, dt_bias, lower_bound)
        return fused_kda_gate(g, a_log.view(1, 1, -1, 1), head_dim, g_bias=dt_bias)

    gate_input = g if g.shape[-1] == head_dim else g.reshape(*g.shape[:-1], -1, head_dim)
    kwargs = {"dt_bias": dt_bias}
    if _FUSED_KDA_GATE_HAS_LOWER_BOUND:
        kwargs["lower_bound"] = lower_bound
    elif lower_bound is not None:
        return _torch_kda_gate(gate_input, a_log, head_dim, dt_bias, lower_bound)
    return fused_kda_gate(gate_input, a_log, **kwargs)


class SituAndMul(nn.Module):
    """K3 SiTU gated activation with fp32 nonlinearities."""

    def __init__(self, beta: float = 1.0, linear_beta: float | None = None) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SiTU to ``[... , 2 * intermediate]`` gate/up projections."""
        gate, up = x.chunk(2, dim=-1)
        gate = gate.float()
        up = up.float()
        activated = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (activated * up).to(x.dtype)


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
    packed_context: "KimiPackedContext | None",
    *,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Create the additive causal attention mask for full-attention layers.

    Args:
        inputs_embeds: Tensor of shape [batch, sequence, hidden].
        packed_context: Optional document layout of the batch. When it marks more
            than one document per row, the mask is block-diagonal so tokens never
            attend across packed documents.
        dtype: Floating-point dtype used for the additive mask values.

    Returns:
        Additive causal mask tensor of shape [batch, 1, sequence, sequence].
    """
    batch_size, seq_len = inputs_embeds.shape[:2]
    if packed_context is not None:
        return build_document_causal_mask(
            packed_context.doc_ids,
            packed_context.doc_ids,
            q_global_start=0,
            dtype=dtype,
        )
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, device=inputs_embeds.device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :].expand(batch_size, 1, -1, -1)


def _packed_context_from_inputs(
    inputs_embeds: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> KimiPackedContext | None:
    """Derive the document layout of a batch that was not sharded for context parallelism.

    Args:
        inputs_embeds: Tensor of shape [batch, sequence, hidden].
        attention_mask: Optional binary or indexed packing mask of shape [batch, sequence].
        cu_seqlens: Optional cumulative document lengths of shape [documents + 1] from
            the THD packed path.

    Returns:
        The document layout, or None when the batch is a single unpadded document per
        row and needs no document bookkeeping.
    """
    if attention_mask is not None and attention_mask.ndim == 2:
        return KimiPackedContext(doc_ids=doc_ids_from_attention_mask(attention_mask))
    if cu_seqlens is not None and isinstance(cu_seqlens, torch.Tensor):
        boundaries = cu_seqlens.squeeze(0) if cu_seqlens.ndim == 2 else cu_seqlens
        return KimiPackedContext(doc_ids=doc_ids_from_cu_seqlens(boundaries, inputs_embeds.shape[1]))
    return None


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


class KimiK3MLP(nn.Module):
    """Dense or shared K3 SiTU MLP."""

    def __init__(
        self,
        config: KimiK3TextConfig,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, dtype=dtype)
        self.act_fn = SituAndMul(
            beta=config.activation_situ_beta or 1.0,
            linear_beta=config.activation_situ_linear_beta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform ``hidden_states`` of shape ``[..., hidden]``."""
        gate_up = torch.cat((self.gate_proj(hidden_states), self.up_proj(hidden_states)), dim=-1)
        return self.down_proj(self.act_fn(gate_up))

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        with buffer_device:
            for module in (self.gate_proj, self.up_proj, self.down_proj):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)


class KimiMLAAttention(nn.Module):
    """Kimi MLA full-attention layer copied from the HF reference math."""

    def __init__(self, config: KimiK3TextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.backend = backend
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if not config.mla_use_nope:
            raise ValueError("Kimi K3 requires mla_use_nope=True.")
        self.scaling = self.q_head_dim**-0.5

        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False, dtype=dtype)
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank, dtype=dtype)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.q_head_dim,
                bias=False,
                dtype=dtype,
            )
        else:
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
        self.use_output_gate = config.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                dtype=dtype,
            )
        self.attn_module = None
        self.attn_func = None
        if backend.attn != "eager":
            if backend.attn not in ("te", "sdpa"):
                raise ValueError(f"Kimi K3 MLA does not support backend.attn={backend.attn!r}.")
            attention_kwargs = {"attention_dropout": self.attention_dropout} if backend.attn == "te" else {}
            self.attn_module, self.attn_func = initialize_attn_module_and_func(
                attn_impl=backend.attn,
                num_attention_heads=self.num_heads,
                num_qk_channels=self.q_head_dim,
                num_v_channels=self.v_head_dim,
                softmax_scale=self.scaling,
                attn_mask_type="causal",
                qkv_format="bshd",
                num_gqa_groups=self.num_key_value_heads,
                **attention_kwargs,
            )
        self._cp_mesh = None

    def setup_cp_attention(self, cp_mesh) -> None:
        """Attach the context-parallel mesh used to gather full-sequence keys and values.

        Called by the MoE parallelizer's ``apply_cp`` for every attention block.

        Args:
            cp_mesh: One-dimensional context-parallel device mesh.
        """
        self._cp_mesh = cp_mesh

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        packed_context: "KimiPackedContext | None" = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run MLA full attention.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden]; the sequence axis
                holds this rank's contiguous shard under context parallelism.
            attention_mask: Optional additive attention mask of shape [batch, 1, sequence, sequence].
            padding_mask: Optional boolean mask of shape [batch, sequence], where true marks padding.
            packed_context: Optional document layout of the batch, required under
                context parallelism.
            **kwargs: Extra attention options accepted for HF compatibility.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        del kwargs
        if packed_context is not None and packed_context.cp_enabled:
            return self._forward_with_cp(hidden_states, packed_context)
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.q_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        key_states, value_states = self._expand_key_value_groups(key_states, value_states, seq_length)

        if self.backend.attn == "eager":
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        else:
            query_states = query_states.transpose(1, 2).contiguous()
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()

            if packed_context is not None and packed_context.has_multiple_documents:
                if self.backend.attn != "te":
                    backend_attention_mask = attention_mask
                    query_states, key_states, value_states, attention_kwargs = preprocess_args_and_kwargs_for_attn(
                        query_states,
                        key_states,
                        value_states,
                        backend_attention_mask,
                        self.backend.attn,
                    )
                else:
                    if attention_mask is None:
                        raise ValueError("Packed K3 MLA attention requires an additive attention mask.")
                    attention_kwargs = {
                        "attention_mask": attention_mask.ne(0),
                        "attn_mask_type": "arbitrary",
                    }
            else:
                backend_attention_mask = padding_mask.logical_not() if padding_mask is not None else None
                query_states, key_states, value_states, attention_kwargs = preprocess_args_and_kwargs_for_attn(
                    query_states,
                    key_states,
                    value_states,
                    backend_attention_mask,
                    self.backend.attn,
                )
            if self.backend.attn == "sdpa":
                attention_kwargs["dropout_p"] = self.attention_dropout if self.training else 0.0
            attn_output = self.attn_func(query_states, key_states, value_states, **attention_kwargs)
            attn_output = postprocess_output_for_attn(attn_output, self.backend.attn)

        attn_output = attn_output.reshape(batch_size, seq_length, -1)
        if self.use_output_gate:
            attn_output = attn_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_output)

    def _expand_key_value_groups(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat key/value heads to match the query heads.

        Args:
            key_states: Tensor of shape [batch, key_value_heads, sequence, qk_head_dim].
            value_states: Tensor of shape [batch, key_value_heads, sequence, v_head_dim].
            seq_length: Sequence length of the key/value tensors.

        Returns:
            Key and value tensors expanded to [batch, heads, sequence, head_dim].
        """
        if self.num_key_value_groups == 1:
            return key_states, value_states
        batch_size = key_states.shape[0]
        key_states = key_states[:, :, None, :, :].expand(
            batch_size, self.num_key_value_heads, self.num_key_value_groups, seq_length, self.q_head_dim
        )
        value_states = value_states[:, :, None, :, :].expand(
            batch_size, self.num_key_value_heads, self.num_key_value_groups, seq_length, self.v_head_dim
        )
        return (
            key_states.reshape(batch_size, self.num_heads, seq_length, self.q_head_dim),
            value_states.reshape(batch_size, self.num_heads, seq_length, self.v_head_dim),
        )

    def _forward_with_cp(self, hidden_states: torch.Tensor, packed_context: KimiPackedContext) -> torch.Tensor:
        """Run MLA attention over a contiguous context-parallel shard.

        Queries stay local while the compressed KV latent -- ``kv_lora_rank +
        qk_rope_head_dim`` values per token, far smaller than the expanded per-head
        keys and values -- is all-gathered across the context-parallel group and
        expanded locally. Attention then runs as FlexAttention with a causal,
        per-document block mask over the full sequence.

        Args:
            hidden_states: Tensor of shape [batch, local_sequence, hidden].
            packed_context: Document layout of the batch.

        Returns:
            Tensor of shape [batch, local_sequence, hidden].
        """
        if self._cp_mesh is None:
            raise RuntimeError(
                "Kimi MLA attention received a context-parallel batch but no CP mesh; "
                "apply_cp must run before the forward pass."
            )
        cp_group = self._cp_mesh.get_group()
        batch_size, local_seq_length = hidden_states.shape[:-1]

        if self.q_lora_rank is not None:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_proj(hidden_states)
        q_states = q_states.view(batch_size, local_seq_length, -1, self.q_head_dim).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = all_gather_sequence(self.kv_a_proj_with_mqa(hidden_states), cp_group, dim=1)
        full_seq_length = compressed_kv.shape[1]
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        key_shape = (batch_size, full_seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, full_seq_length, self.qk_rope_head_dim)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)
        key_states, value_states = self._expand_key_value_groups(key_states, value_states, full_seq_length)

        attn_output = document_causal_flex_attention(
            query_states,
            key_states,
            value_states,
            q_doc_ids=packed_context.local_doc_ids,
            kv_doc_ids=packed_context.doc_ids,
            q_global_start=packed_context.seq_start,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, local_seq_length, -1)
        if self.use_output_gate:
            attn_output = attn_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attn_output)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        with buffer_device:
            modules = [self.kv_a_proj_with_mqa, self.kv_b_proj, self.o_proj]
            if self.q_lora_rank is not None:
                modules.extend((self.q_a_proj, self.q_b_proj))
                self.q_a_layernorm.reset_parameters()
            else:
                modules.append(self.q_proj)
            if self.use_output_gate:
                modules.append(self.g_proj)
            for module in modules:
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


class _KimiFp32Module(nn.Module):
    """Keep a callable FLA operator in its own fp32 FSDP unit."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._fp32_params = module

    @property
    def weight(self) -> nn.Parameter:
        """Expose the wrapped weight under the reference module API."""
        return self._fp32_params.weight

    def reset_parameters(self) -> None:
        """Reset the wrapped operator."""
        self._fp32_params.reset_parameters()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the operator while its fp32 FSDP unit is unsharded."""
        return self._fp32_params(*args, **kwargs)


class KimiKDAFp32Params(nn.Module):
    """Own KDA recurrent-decay parameters and compute the FP32 decay gate."""

    def __init__(self, num_heads: int, projection_size: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

    def forward(
        self,
        g: torch.Tensor,
        head_dim: int,
        lower_bound: float | None,
        use_fused_gate: bool = True,
    ) -> torch.Tensor:
        """Compute the KDA decay while this holder's FSDP unit is unsharded.

        Args:
            g: Raw gate tensor of shape [batch, sequence, heads * head_dim].
            head_dim: Per-head KDA dimension.
            lower_bound: Optional lower bound for K3's bounded decay function.
            use_fused_gate: Whether to use FLA's fused gate kernel.

        Returns:
            FP32 decay tensor of shape [batch, sequence, heads, head_dim].
        """
        a_log = self.A_log.contiguous()
        dt_bias = self.dt_bias.contiguous()
        if use_fused_gate:
            return _fused_kda_gate(g, a_log, head_dim, dt_bias, lower_bound)
        return _torch_kda_gate(g, a_log, head_dim, dt_bias, lower_bound)


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention backed by FLA KDA kernels."""

    A_log = _KimiKDAFp32Param("A_log")
    dt_bias = _KimiKDAFp32Param("dt_bias")

    def __init__(self, config: KimiK3TextConfig, layer_idx: int) -> None:
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
        self.q_conv1d = _KimiFp32Module(
            ShortConvolution(
                hidden_size=projection_k_size,
                kernel_size=self.conv_size,
                activation="silu",
                dtype=torch.float32,
            )
        )
        self.k_conv1d = _KimiFp32Module(
            ShortConvolution(
                hidden_size=projection_k_size,
                kernel_size=self.conv_size,
                activation="silu",
                dtype=torch.float32,
            )
        )
        self.v_conv1d = _KimiFp32Module(
            ShortConvolution(
                hidden_size=projection_size,
                kernel_size=self.conv_size,
                activation="silu",
                dtype=torch.float32,
            )
        )

        self._fp32_params = KimiKDAFp32Params(self.num_heads, projection_size)
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False, dtype=dtype)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False, dtype=dtype)
        self.use_full_rank_gate = config.linear_attn_config.get("use_full_rank_gate", False)
        self.gate_lower_bound = config.linear_attn_config.get("gate_lower_bound")
        if self.use_full_rank_gate:
            self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False, dtype=dtype)
        else:
            self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=dtype)
            self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False, dtype=dtype)
        self.o_norm = _KimiFp32Module(
            FusedRMSNormGated(
                self.head_dim,
                eps=config.rms_norm_eps,
                activation="sigmoid",
                dtype=torch.float32,
            )
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False, dtype=dtype)
        self._cp_mesh = None

    def setup_cp_attention(self, cp_mesh) -> None:
        """Attach the context-parallel mesh used to build FLA's CP context.

        Called by the MoE parallelizer's ``apply_cp`` for every attention block.

        Args:
            cp_mesh: One-dimensional context-parallel device mesh.
        """
        self._cp_mesh = cp_mesh

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_context: "KimiPackedContext | None" = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run KDA linear attention.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden]; the sequence axis
                holds this rank's contiguous shard under context parallelism.
            attention_mask: Optional binary padding mask of shape [batch, sequence] where 1 marks valid tokens.
            packed_context: Optional document layout of the batch, required under
                context parallelism and used to reset the recurrent state at every
                packed-document boundary.
            **kwargs: Optional KDA kwargs, including ``cu_seqlens`` for packed sequences.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        if packed_context is not None and packed_context.cp_enabled:
            return self._forward_with_cp(hidden_states, packed_context)

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")
            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError("Kimi KDA attention_mask must have shape [batch, sequence].")

        batch_size, q_len, _ = hidden_states.shape

        cu_seqlens = kwargs.get("cu_seqlens")
        indices = None
        if is_indexed_packed_mask(attention_mask):
            # Packed rows: unpad to a single flat sequence and reset the recurrent
            # state per document instead of per row.
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = _index_first_axis(hidden_states.reshape(batch_size * q_len, -1), indices).unsqueeze(0)
        elif attention_mask is not None and getattr(self.config, "kda_unpad_inputs", True):
            indices, cu_seqlens, _ = _get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = _index_first_axis(hidden_states.reshape(batch_size * q_len, -1), indices).unsqueeze(0)

        o = self._kda_core(hidden_states, cu_seqlens=cu_seqlens)
        if indices is not None:
            o = _pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o

    def _forward_with_cp(self, hidden_states: torch.Tensor, packed_context: KimiPackedContext) -> torch.Tensor:
        """Run KDA over a contiguous context-parallel shard.

        FLA's context-parallel kernels take the *global* ``cu_seqlens`` and derive
        each rank's local segments, passing the recurrent state (and the short
        convolution's boundary tokens) rank to rank. Batch rows are processed one
        at a time because FLA's variable-length path expects a single flattened
        sequence per call.

        Args:
            hidden_states: Tensor of shape [batch, local_sequence, hidden].
            packed_context: Document layout of the batch.

        Returns:
            Tensor of shape [batch, local_sequence, hidden].
        """
        if self._cp_mesh is None:
            raise RuntimeError(
                "Kimi KDA attention received a context-parallel batch but no CP mesh; "
                "apply_cp must run before the forward pass."
            )
        cp_group = self._cp_mesh.get_group()
        outputs = [
            self._kda_core(
                hidden_states[row : row + 1],
                cp_context=build_fla_cp_context(packed_context, row, cp_group, self.conv_size),
            )
            for row in range(hidden_states.shape[0])
        ]
        return torch.cat(outputs, dim=0)

    def _kda_core(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cp_context: Any = None,
    ) -> torch.Tensor:
        """Run the KDA projections, convolutions and delta-rule kernel.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden]; the batch must be
                one whenever ``cu_seqlens`` or ``cp_context`` is given.
            cu_seqlens: Optional cumulative document lengths of shape [documents + 1].
            cp_context: Optional FLA context-parallel context, which supersedes
                ``cu_seqlens`` with its per-rank local segments.

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        kernel_kwargs: dict[str, Any] = {} if cp_context is None else {"cp_context": cp_context}
        conv_kwargs = dict(cache=None, output_final_state=False, cu_seqlens=cu_seqlens, **kernel_kwargs)

        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), **conv_kwargs)
        k, _ = self.k_conv1d(x=self.k_proj(hidden_states), **conv_kwargs)
        v, _ = self.v_conv1d(x=self.v_proj(hidden_states), **conv_kwargs)
        g = self.f_b_proj(self.f_a_proj(hidden_states)).contiguous()
        beta = self.b_proj(hidden_states).float().sigmoid()

        q = q.reshape(*q.shape[:-1], self.num_k_heads, self.head_k_dim).contiguous()
        k = k.reshape(*k.shape[:-1], self.num_k_heads, self.head_k_dim).contiguous()
        v = v.reshape(*v.shape[:-1], self.num_heads, self.head_dim).contiguous()
        beta = beta.contiguous()
        g = self._fp32_params(
            g,
            self.head_dim,
            self.gate_lower_bound,
            getattr(self.config, "kda_use_fused_gate", True),
        ).contiguous()
        use_qk_l2norm_in_kernel = getattr(self.config, "kda_use_qk_l2norm_in_kernel", True)
        if not use_qk_l2norm_in_kernel:
            q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(q.dtype)
            k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(k.dtype)

        # The recurrent kernel cannot pass state across ranks, so context parallelism
        # always runs the chunked kernel.
        mode = "chunk" if cp_context is not None else self.mode
        kernel = chunk_kda if mode == "chunk" else fused_recurrent_kda
        kernel_options = {
            "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
            "transpose_state_layout": True,
        }
        if mode == "chunk":
            kernel_options["safe_gate"] = self.gate_lower_bound is not None
        o, _ = kernel(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            # Under CP the final state is owned by FLA's rank-to-rank handoff.
            output_final_state=cp_context is None,
            cu_seqlens=cu_seqlens,
            **kernel_options,
            **kernel_kwargs,
        )

        if self.use_full_rank_gate:
            gate = self.g_proj(hidden_states)
        else:
            gate = self.g_b_proj(self.g_a_proj(hidden_states))
        gate = gate.reshape(*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        o = self.o_norm(o, gate).to(hidden_states.dtype)
        o = o.reshape(o.shape[0], o.shape[1], -1).contiguous()
        return self.o_proj(o)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        with buffer_device:
            self.A_log.uniform_(1, 16).log_()
            self.dt_bias.zero_()
            modules = [
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.f_a_proj,
                self.f_b_proj,
                self.b_proj,
                self.o_proj,
            ]
            if self.use_full_rank_gate:
                modules.append(self.g_proj)
            else:
                modules.extend((self.g_a_proj, self.g_b_proj))
            for module in modules:
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
                conv.reset_parameters()
            if hasattr(self.o_norm, "reset_parameters"):
                self.o_norm.reset_parameters()


def _weighted_situ(
    gate_up: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Apply SiTU and routing weights to ``[tokens, 2 * intermediate]`` projections."""
    input_dtype = gate_up.dtype
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (activated * up * routing_weights.float()).to(input_dtype)


class KimiK3Gate(Gate):
    """K3's fp32 sigmoid router with correction-bias-only expert selection."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Route ``[tokens, hidden]`` states and return fp32 top-k weights."""
        del token_mask, cp_mesh
        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        if self.score_func == "sigmoid_with_bias":
            scores = logits.sigmoid()
        elif self.score_func == "softmax_with_bias":
            scores = logits.softmax(dim=-1)
        else:
            raise ValueError(f"Kimi K3 requires a correction-bias router, got {self.score_func!r}.")

        scores_for_choice = scores
        if self.e_score_correction_bias is not None:
            scores_for_choice = scores_for_choice + self.e_score_correction_bias.unsqueeze(0)
        if self.n_groups > 1 and self.n_groups > self.topk_groups:
            grouped = scores_for_choice.view(hidden_states.shape[0], self.n_groups, -1)
            group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
            group_indices = group_scores.topk(self.topk_groups, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores).scatter_(1, group_indices, 1)
            score_mask = group_mask.unsqueeze(-1).expand_as(grouped).reshape_as(scores_for_choice)
            scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))

        indices = scores_for_choice.topk(self.topk, dim=-1, sorted=False)[1]
        weights = scores.gather(1, indices)
        if self.topk > 1 and self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices, None


class KimiK3MoE(MoE):
    """K3 routed experts with latent projections and a SiTU shared expert."""

    def __init__(self, config: KimiK3TextConfig, moe_config: MoEConfig, backend: BackendConfig) -> None:
        nn.Module.__init__(self)
        self.backend = backend
        self.dim = moe_config.dim
        self.n_routed_experts = moe_config.n_routed_experts
        self.n_activated_experts = moe_config.n_activated_experts
        self.gate = KimiK3Gate(moe_config, gate_precision=torch.float32)
        expert_activation = partial(
            _weighted_situ,
            beta=config.activation_situ_beta or 1.0,
            linear_beta=config.activation_situ_linear_beta,
        )
        if backend.dispatcher in ("deepep", "hybridep", "uccl_ep") and get_world_size_safe() > 1:
            self.experts = GroupedExpertsDeepEP(
                moe_config,
                backend=backend,
                dispatcher_backend=backend.dispatcher,
                dispatcher_num_sms=backend.dispatcher_num_sms,
                dispatcher_share_token_dispatcher=backend.dispatcher_share_token_dispatcher,
                dispatcher_async_dispatch=backend.dispatcher_async_dispatch,
            )
            self.experts.expert_activation = expert_activation
        else:
            self.experts = GroupedExperts(moe_config, backend=backend)
            self.experts.expert_activation_grouped = expert_activation
        shared_intermediate = config.moe_intermediate_size * (config.num_shared_experts or 0)
        self.shared_experts = (
            KimiK3MLP(config, intermediate_size=shared_intermediate, dtype=moe_config.dtype)
            if shared_intermediate > 0
            else None
        )
        self.shared_expert_gate = None
        self.routed_expert_down_proj = nn.Linear(
            config.hidden_size,
            config.routed_expert_hidden_size,
            bias=False,
            dtype=moe_config.dtype,
        )
        self.routed_expert_up_proj = nn.Linear(
            config.routed_expert_hidden_size,
            config.hidden_size,
            bias=False,
            dtype=moe_config.dtype,
        )
        self.routed_expert_norm = (
            KimiRMSNorm(config.routed_expert_hidden_size, eps=config.rms_norm_eps, dtype=moe_config.dtype)
            if config.latent_moe_use_norm
            else None
        )
        self.cp_mesh = None
        self._situ = SituAndMul(
            beta=config.activation_situ_beta or 1.0,
            linear_beta=config.activation_situ_linear_beta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        cp_mesh: Any = None,
    ) -> torch.Tensor:
        """Run K3 MoE on ``[batch, sequence, hidden]`` states."""
        shape = hidden_states.shape
        identity = hidden_states.reshape(-1, shape[-1])
        token_mask = (
            ~padding_mask.flatten()
            if padding_mask is not None
            else torch.ones(identity.shape[0], dtype=torch.bool, device=identity.device)
        )
        gate_cp_mesh = cp_mesh if cp_mesh is not None else self.cp_mesh
        weights, indices, _ = self.gate(identity, token_mask, gate_cp_mesh)
        routed_input = self.routed_expert_down_proj(identity)
        if not self.training and not self._has_distributed_experts():
            routed = self._forward_reference_order(routed_input, indices, weights)
        else:
            routed = self.experts(routed_input, token_mask, weights, indices)
        if self.routed_expert_norm is not None:
            routed = self.routed_expert_norm(routed)
        output = self.routed_expert_up_proj(routed)
        if self.shared_experts is not None:
            output = output + self.shared_experts(identity)
        return output.view(shape)

    def _has_distributed_experts(self) -> bool:
        """Whether grouped expert parameters are DTensors."""
        return hasattr(self.experts.gate_and_up_projs, "to_local") or hasattr(self.experts.down_projs, "to_local")

    def _forward_reference_order(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Match the checkpoint implementation's expert-ordered inference loop."""
        counts = topk_ids.new_zeros((topk_ids.shape[0], self.n_routed_experts))
        counts.scatter_(1, topk_ids, 1)
        tokens_per_expert = counts.sum(dim=0).cpu().tolist()
        sorted_ids = topk_ids.reshape(-1).argsort()
        sorted_tokens = hidden_states[sorted_ids // topk_ids.shape[1]]

        outputs = []
        start = 0
        for expert_idx, num_tokens in enumerate(tokens_per_expert):
            end = start + num_tokens
            if num_tokens:
                tokens = sorted_tokens[start:end]
                gate_up = tokens @ self.experts.gate_and_up_projs[expert_idx].to(tokens.dtype)
                activated = self._situ(gate_up)
                outputs.append(activated @ self.experts.down_projs[expert_idx].to(tokens.dtype))
            start = end

        routed = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty((0, hidden_states.shape[-1]))
        unpermuted = torch.empty_like(routed)
        unpermuted[sorted_ids] = routed
        return (
            unpermuted.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(-1))
            .sum(dim=1)
            .type(routed.dtype)
        )

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.gate.init_weights(buffer_device, init_std)
        self.experts.init_weights(buffer_device, init_std)
        with buffer_device:
            for module in (self.routed_expert_down_proj, self.routed_expert_up_proj):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if self.routed_expert_norm is not None:
                self.routed_expert_norm.reset_parameters()
        if self.shared_experts is not None:
            self.shared_experts.init_weights(buffer_device, init_std)


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Linear,
    norm: KimiRMSNorm,
) -> torch.Tensor:
    """Mix ``[tokens, hidden]`` with prior ``[tokens, blocks, hidden]`` residuals."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_fp32 = values.float()
    variance = values_fp32.pow(2).mean(-1, keepdim=True)
    keys = values_fp32 * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
    probabilities = (keys * score_weight).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probabilities, values_fp32).squeeze(1).to(values.dtype)


class KimiDecoderLayer(nn.Module):
    """Kimi decoder block with KDA/MLA attention and dense or MoE MLP."""

    def __init__(self, config: KimiK3TextConfig, layer_idx: int, moe_config: MoEConfig, backend: BackendConfig) -> None:
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
            KimiDeltaAttention(config, layer_idx)
            if self.is_linear_attn
            else KimiMLAAttention(config, layer_idx, backend)
        )

        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        # Both branches are named ``mlp``: the custom-MoE parallelizer discovers MoE
        # layers by that attribute (as MiniMax does), and the checkpoint's
        # ``block_sparse_moe`` naming is handled by the state-dict adapter.
        if self.is_moe_layer:
            self.mlp = KimiK3MoE(config, moe_config, backend)
        else:
            self.mlp = KimiK3MLP(config, dtype=dtype)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.attn_res_block_size = config.attn_res_block_size
            self.self_attention_res_norm = KimiRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=dtype,
            )
            self.mlp_res_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
            self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False, dtype=dtype)
            self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        block_residual: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run one Kimi decoder layer.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            attention_mask: KDA layers receive a binary mask [batch, sequence]; MLA layers receive an additive
                causal mask [batch, 1, sequence, sequence].
            padding_mask: Optional boolean tensor of shape [batch, sequence], where true marks padding tokens.
            block_residual: Prior block starts with shape [batch * sequence, blocks, hidden].
            **attn_kwargs: Extra attention kwargs forwarded to KDA/MLA.

        Returns:
            Tensor of shape [batch, sequence, hidden], plus updated block residuals when enabled.
        """
        if self.use_attn_residuals:
            if block_residual is None:
                raise ValueError("K3 attention residual layers require block_residual.")
            return self._forward_attn_residual(
                hidden_states,
                block_residual,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe_layer:
            if attention_mask is not None and attention_mask.ndim == 2 and padding_mask is None:
                padding_mask = attention_mask.bool().logical_not()
            hidden_states = self.mlp(hidden_states, padding_mask)
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def _forward_attn_residual(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one decoder layer using K3's learned block-residual mixing.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            block_residual: Tensor of shape [batch * sequence, blocks, hidden].
            attention_mask: KDA padding mask or MLA additive causal mask.
            padding_mask: Optional boolean tensor of shape [batch, sequence], where true marks padding tokens.
            **attn_kwargs: Extra attention arguments.

        Returns:
            Updated hidden states and block residuals.
        """
        batch_size, sequence_length, hidden_size = hidden_states.shape
        prefix_sum = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(
                prefix_sum.reshape(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, sequence_length, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (block_residual, prefix_sum.reshape(-1, hidden_size).unsqueeze(1)),
                dim=1,
            )
            prefix_sum = None

        attention_output = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
        mlp_input = _apply_attn_res(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, sequence_length, hidden_size)
        mlp_input = self.post_attention_layernorm(mlp_input)
        if self.is_moe_layer:
            if attention_mask is not None and attention_mask.ndim == 2 and padding_mask is None:
                padding_mask = attention_mask.bool().logical_not()
            mlp_output = self.mlp(mlp_input, padding_mask)
        else:
            mlp_output = self.mlp(mlp_input)
        return prefix_sum + mlp_output, block_residual

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device, init_std)
        self.mlp.init_weights(buffer_device, init_std)
        if self.use_attn_residuals:
            self.self_attention_res_norm.reset_parameters()
            self.mlp_res_norm.reset_parameters()
            with buffer_device:
                nn.init.normal_(self.self_attention_res_proj.weight, mean=0.0, std=init_std)
                nn.init.normal_(self.mlp_res_proj.weight, mean=0.0, std=init_std)


def _build_moe_config(
    config: KimiK3TextConfig,
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
        score_func=f"{config.moe_router_activation_func}_with_bias",
        route_scale=config.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=config.moe_renormalize,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        # The checkpoint reference rounds each BF16 expert output before applying
        # its FP32 router weight and reducing the top-k outputs.
        apply_router_weight_after_down=True,
        dtype=model_dtype,
        shared_expert_gate=False,
        shared_expert_inter_dim=config.moe_intermediate_size,
        force_e_score_correction_bias=True,
        moe_latent_size=config.routed_expert_hidden_size,
    )
    if moe_overrides:
        moe_defaults.update(moe_overrides)
    return MoEConfig(**moe_defaults)


def _partition_attn_residual_blocks(
    num_layers: int,
    block_size: int,
    num_stages: int,
    *,
    allow_output_only_stage: bool = False,
) -> list[range]:
    """Partition decoder layers without splitting an attention-residual block."""
    if num_layers < 1:
        raise ValueError("K3 pipeline parallelism requires at least one decoder layer.")
    if block_size < 1:
        raise ValueError("K3 attention-residual block size must be positive.")
    if num_stages < 1:
        raise ValueError("K3 pipeline parallelism requires at least one stage.")

    num_blocks = math.ceil(num_layers / block_size)
    if num_stages > num_blocks:
        if allow_output_only_stage and num_stages == num_blocks + 1:
            block_ranges = _partition_attn_residual_blocks(num_layers, block_size, num_blocks)
            return [*block_ranges, range(num_layers, num_layers)]
        raise ValueError(
            f"K3 has {num_blocks} attention-residual blocks, so it cannot be split into "
            f"{num_stages} pipeline stages without splitting a block."
        )

    blocks_per_stage, extra_blocks = divmod(num_blocks, num_stages)
    ranges: list[range] = []
    first_block = 0
    for stage_idx in range(num_stages):
        stage_blocks = blocks_per_stage + (stage_idx < extra_blocks)
        first_layer = first_block * block_size
        last_layer = min((first_block + stage_blocks) * block_size, num_layers)
        ranges.append(range(first_layer, last_layer))
        first_block += stage_blocks
    return ranges


def _seed_dtensor_rng_for_pipeline_stage(model: nn.Module) -> None:
    """Initialize DTensor RNG without a world broadcast during PP weight init."""
    try:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._random import manual_seed as dtensor_manual_seed
    except ImportError:
        return

    dtensor_param = next((param for param in model.parameters() if isinstance(param, DTensor)), None)
    if dtensor_param is None:
        return

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    base_seed = torch.initial_seed() - rank
    layers = getattr(getattr(model, "model", None), "layers", {})
    first_layer = min((int(layer_idx) for layer_idx in layers), default=0)
    dtensor_manual_seed((base_seed + first_layer) % (2**64), dtensor_param.device_mesh)


class KimiK3TextModel(nn.Module):
    """Kimi Linear decoder backbone with trainable Automodel MoE layers."""

    def __init__(
        self,
        config: KimiK3TextConfig,
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
        self.use_attn_residuals = config.attn_res_block_size is not None
        if self.use_attn_residuals:
            self.output_attn_res_norm = KimiRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=model_dtype,
            )
            self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False, dtype=model_dtype)

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
        block_residual: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        kimi_packed_context: KimiPackedContext | None = None,
        kimi_packed_doc_ids: torch.Tensor | None = None,
        kimi_packed_seq_start: int = 0,
        kimi_packed_cp_size: int = 1,
        **attn_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the Kimi Linear decoder.

        Args:
            input_ids: Optional token ids of shape [batch, sequence].
            block_residual: Prior attention-residual block starts with shape
                [batch * sequence, blocks, hidden]. Pipeline stages after the
                first receive this as their second positional activation.
            inputs_embeds: Optional embeddings of shape [batch, sequence, hidden].
            attention_mask: Optional binary or indexed packing mask of shape [batch, sequence].
            position_ids: Optional positions of shape [batch, sequence]; accepted for HF compatibility.
            padding_mask: Optional boolean tensor of shape [batch, sequence], where true marks padding tokens.
            cache_position: Optional position vector of shape [sequence].
            kimi_packed_context: Optional document layout attached by
                :func:`~nemo_automodel.components.models.kimi_k3.cp.shard_batch_for_kimi_cp`;
                required under context parallelism and otherwise derived here.
            kimi_packed_doc_ids: Pipeline-safe global document map used to
                reconstruct ``kimi_packed_context`` after microbatch chunking.
            kimi_packed_seq_start: Global offset of this CP rank's sequence shard.
            kimi_packed_cp_size: Number of context-parallel sequence shards.
            **attn_kwargs: Additional attention kwargs used by packed or THD execution.

        Returns:
            Tensor of shape [batch, sequence, hidden], or the hidden states and
            block residuals when this is a non-final pipeline stage.
        """
        del position_ids
        if input_ids is not None and input_ids.is_floating_point():
            if inputs_embeds is not None:
                raise ValueError("Pipeline hidden states and inputs_embeds cannot both be specified.")
            inputs_embeds = input_ids
            input_ids = None
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("Only the first K3 pipeline stage can embed token ids.")
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        if kimi_packed_context is not None and kimi_packed_doc_ids is not None:
            raise ValueError("Pass either kimi_packed_context or pipeline-safe Kimi CP metadata, not both.")
        packed_context = kimi_packed_context
        if packed_context is None and kimi_packed_doc_ids is not None:
            packed_context = KimiPackedContext(
                doc_ids=kimi_packed_doc_ids,
                seq_start=kimi_packed_seq_start,
                cp_size=kimi_packed_cp_size,
            )
        if packed_context is None:
            packed_context = _packed_context_from_inputs(
                inputs_embeds,
                attention_mask=attention_mask,
                cu_seqlens=attn_kwargs.get("cu_seqlens"),
            )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)
        causal_mask = (
            None
            if packed_context is not None and packed_context.cp_enabled
            else _make_causal_mask(inputs_embeds, packed_context, dtype=inputs_embeds.dtype)
        )
        hidden_states = inputs_embeds
        if self.use_attn_residuals and block_residual is None:
            first_layer_idx = min((int(layer_idx) for layer_idx in self.layers), default=0)
            if first_layer_idx != 0:
                raise ValueError(
                    f"K3 pipeline stage beginning at layer {first_layer_idx} requires block_residual from "
                    "the preceding stage."
                )
            block_residual = hidden_states.new_zeros(
                hidden_states.shape[0] * hidden_states.shape[1],
                0,
                hidden_states.shape[2],
            )

        for decoder_layer in self.layers.values():
            layer_mask = linear_attn_mask if decoder_layer.is_linear_attn else causal_mask
            layer_padding_mask = padding_mask
            if decoder_layer.is_linear_attn and layer_mask is not None:
                layer_padding_mask = layer_mask.bool().logical_not()
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                padding_mask=layer_padding_mask,
                block_residual=block_residual,
                packed_context=packed_context,
                **attn_kwargs,
            )
            if self.use_attn_residuals:
                hidden_states, block_residual = layer_output
            else:
                hidden_states = layer_output

        if self.norm is None:
            if self.use_attn_residuals:
                return hidden_states, block_residual
            return hidden_states
        if self.use_attn_residuals:
            hidden_states = self._apply_output_attn_res(hidden_states, block_residual)
        return self.norm(hidden_states)

    def _apply_output_attn_res(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Mix final ``[batch, sequence, hidden]`` states with block starts."""
        batch_size, sequence_length, hidden_size = hidden_states.shape
        return _apply_attn_res(
            hidden_states.reshape(-1, hidden_size),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
        ).view(batch_size, sequence_length, hidden_size)

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.layers.values():
                if block.is_moe_layer and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or (
            torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
        )
        init_std = self.config.initializer_range
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=init_std)
                if self.padding_idx is not None:
                    self.embed_tokens.weight[self.padding_idx].zero_()
            if self.norm is not None:
                self.norm.reset_parameters()
            if self.use_attn_residuals:
                if self.output_attn_res_norm is not None:
                    self.output_attn_res_norm.reset_parameters()
                if self.output_attn_res_proj is not None:
                    nn.init.normal_(self.output_attn_res_proj.weight, mean=0.0, std=init_std)
        for layer in self.layers.values():
            layer.init_weights(buffer_device, init_std)


class KimiK3ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Kimi Linear causal LM with native trainable MoE layers."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules = [
        "_fp32_params",
        "e_score_correction_bias",
    ]
    _keep_in_fp32_modules_strict = ["_fp32_params"]
    # Kimi Linear owns context parallelism end to end: it shards the batch itself
    # (contiguous slices, as FLA's CP kernels require) and each layer type carries
    # its own transport, so CP does not depend on the attention backend.
    _owns_cp_attention = True
    # Packed documents are masked by the model itself: MLA gets a document-blocked
    # causal mask and KDA resets its recurrent state on per-document ``cu_seqlens``.
    _owns_packed_attention = True
    # K3's forward carries the attention-residual accumulator between pipeline
    # stages, which the generic Hugging Face pipeline patch does not understand.
    _pp_keep_self_forward: bool = True
    _pp_return_hidden_states_supported: bool = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for Kimi Linear."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: KimiK3TextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "KimiK3ForCausalLM":
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "KimiK3ForCausalLM":
        config = KimiK3Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: KimiK3Config | KimiK3TextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        num_hidden_layers: int | None = None,
        kda_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        text_config = getattr(config, "text_config", config)
        if num_hidden_layers is not None or kda_mode is not None:
            text_config = copy.deepcopy(text_config)
        if num_hidden_layers is not None:
            if not 1 <= num_hidden_layers <= text_config.num_hidden_layers:
                raise ValueError(
                    f"num_hidden_layers must be in [1, {text_config.num_hidden_layers}], got {num_hidden_layers}."
                )
            text_config.num_hidden_layers = num_hidden_layers
            linear_attn_config = copy.deepcopy(text_config.linear_attn_config)
            linear_attn_config["kda_layers"] = [
                layer_idx for layer_idx in linear_attn_config["kda_layers"] if layer_idx <= num_hidden_layers
            ]
            linear_attn_config["full_attn_layers"] = [
                layer_idx for layer_idx in linear_attn_config["full_attn_layers"] if layer_idx <= num_hidden_layers
            ]
            text_config.linear_attn_config = linear_attn_config
        if kda_mode is not None:
            text_config.kda_mode = kda_mode
        text_config._validate()

        reject_unsupported_tie_word_embeddings(type(self), text_config)
        self.config = text_config
        self.backend = copy.copy(backend) if backend is not None else BackendConfig()
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = KimiK3TextModel(
            text_config,
            self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        model_dtype = get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
            dtype=model_dtype,
        )
        self.vocab_size = text_config.vocab_size
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = KimiK3StateDictAdapter(
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

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep every K3 attention-residual block within one pipeline stage."""
        del text_model
        block_size = self.model.config.attn_res_block_size
        if block_size is None:
            return module_names_per_stage

        layer_ranges = _partition_attn_residual_blocks(
            self.model.config.num_hidden_layers,
            block_size,
            len(module_names_per_stage),
            allow_output_only_stage=True,
        )
        multimodal_names = {
            f"{layers_prefix}vision_tower",
            f"{layers_prefix}mm_projector",
            "vision_tower",
            "mm_projector",
        }
        output_residual_names = {
            f"{layers_prefix}output_attn_res_norm",
            f"{layers_prefix}output_attn_res_proj",
        }
        fixed: list[list[str]] = []
        for stage_idx, (stage_modules, layer_range) in enumerate(zip(module_names_per_stage, layer_ranges)):
            names = [
                name
                for name in stage_modules
                if not name.startswith(f"{layers_prefix}layers.")
                and name not in multimodal_names
                and name not in output_residual_names
            ]
            names.extend(f"{layers_prefix}layers.{layer_idx}" for layer_idx in layer_range)
            if stage_idx == 0:
                if getattr(self, "vision_tower", None) is not None:
                    names.append("vision_tower")
                if getattr(self, "mm_projector", None) is not None:
                    names.append("mm_projector")
            if stage_idx == len(module_names_per_stage) - 1:
                if getattr(self.model, "output_attn_res_norm", None) is not None:
                    names.append(f"{layers_prefix}output_attn_res_norm")
                if getattr(self.model, "output_attn_res_proj", None) is not None:
                    names.append(f"{layers_prefix}output_attn_res_proj")
            fixed.append(names)
        return fixed

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return static PP metadata for hidden states and block residuals."""
        text_config = getattr(self.config, "text_config", self.config)
        hidden_size = text_config.hidden_size
        layer_indices = sorted(int(layer_idx) for layer_idx in self.model.layers)
        first_layer = layer_indices[0] if layer_indices else text_config.num_hidden_layers
        last_layer = layer_indices[-1] if layer_indices else first_layer - 1
        block_size = text_config.attn_res_block_size

        def meta(*shape: int, tensor_dtype: torch.dtype = dtype) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=tensor_dtype)

        if is_first:
            inputs_meta = (meta(microbatch_size, seq_len, tensor_dtype=torch.long),)
        elif block_size is None:
            inputs_meta = (meta(microbatch_size, seq_len, hidden_size),)
        else:
            prior_blocks = math.ceil(first_layer / block_size)
            inputs_meta = (
                meta(microbatch_size, seq_len, hidden_size),
                meta(microbatch_size * seq_len, prior_blocks, hidden_size),
            )

        emits_hidden_states = getattr(self, "_pp_return_hidden_states", False) is True
        if self.lm_head is not None and not emits_hidden_states:
            head_dtype = getattr(getattr(self.lm_head, "weight", None), "dtype", dtype)
            outputs_meta = (meta(microbatch_size, seq_len, text_config.vocab_size, tensor_dtype=head_dtype),)
        elif self.model.norm is not None or block_size is None:
            outputs_meta = (meta(microbatch_size, seq_len, hidden_size),)
        else:
            blocks_after_stage = math.ceil((last_layer + 1) / block_size)
            outputs_meta = (
                meta(microbatch_size, seq_len, hidden_size),
                meta(microbatch_size * seq_len, blocks_after_stage, hidden_size),
            )
        return inputs_meta, outputs_meta

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        block_residual: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run Kimi Linear causal LM.

        Args:
            input_ids: Optional token ids of shape [batch, sequence].
            block_residual: Prior K3 attention-residual block starts. Pipeline
                stages after the first receive this as their second activation.
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
            # THD packing drops the placeholder batch axis, but every Kimi layer
            # works in [batch, sequence, ...]. Restore it and let ``cu_seqlens``
            # carry the document boundaries; ``compute_lm_head_logits`` leaves the
            # already-batched [1, tokens, hidden] result untouched.
            if input_ids is not None and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if inputs_embeds is not None and inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            if padding_mask is not None and padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(0)

        hidden_states = self.model(
            input_ids,
            block_residual,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        if isinstance(hidden_states, tuple):
            return hidden_states
        if getattr(self, "_pp_return_hidden_states", False) is True:
            return hidden_states
        return compute_lm_head_logits(
            self.lm_head,
            hidden_states,
            logits_to_keep,
            is_thd=is_thd,
            output_hidden_states=output_hidden_states,
        )

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Hand the recipe Kimi K3's own context-parallel batch sharding.

        KDA's recurrent state (and FLA's CP kernels) require every rank to own one
        contiguous slice of the sequence, so Kimi K3 replaces the default
        load-balanced context-parallel sharding with
        :func:`~nemo_automodel.components.models.kimi_k3.cp.shard_batch_for_kimi_cp`.

        Args:
            batch: Full-sequence batch; left untouched until the returned sharder runs.
            num_chunks: Accepted for CP hook signature parity; K3 uses one contiguous shard.

        Returns:
            Batch updates carrying the model-owned context-parallel sharder.
        """
        from nemo_automodel.components.distributed.context_parallel.sharder import (  # noqa: PLC0415
            ContextParallelSharder,
            contiguous_local_indices,
        )

        cp_mesh = getattr(self, "cp_mesh", None)
        if cp_mesh is None:
            raise RuntimeError("Kimi K3 context-parallel input preparation requires a CP mesh.")
        for module in self.modules():
            if isinstance(module, (KimiMLAAttention, KimiDeltaAttention)):
                module.setup_cp_attention(cp_mesh)

        del batch, num_chunks
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_for_kimi_cp,
                local_token_global_indices=contiguous_local_indices,
            )
        }

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
        _seed_dtensor_rng_for_pipeline_stage(self)
        self.model.init_weights(buffer_device)
        final_out_std = self.config.hidden_size**-0.5
        cutoff_factor = 3
        with buffer_device:
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))


ModelClass = KimiK3ForCausalLM

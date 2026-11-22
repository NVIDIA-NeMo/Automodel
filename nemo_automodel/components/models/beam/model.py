# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Native Beam decoder: gated attention, RoPE/NoPE with root10 SSMax, sandwich norms,
depth-scaled residuals, dense/MoE SwiGLU, and soft-capped logits. The module tree
preserves HF names; BeamStateDictAdapter converts the grouped expert weights."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.beam.layers import (
    Block,
    hf_rope_tables,
    layer_rope_parameters,
    reject_flex_attention_mask,
)
from nemo_automodel.components.models.beam.parallelization import PARALLELIZER
from nemo_automodel.components.models.beam.state_dict_adapter import BeamStateDictAdapter
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
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

if TYPE_CHECKING:
    from transformers.models.beam import BeamConfig


def beam_moe_config(config: BeamConfig, model_dtype: torch.dtype, overrides: dict[str, Any] | None = None) -> MoEConfig:
    """Map Beam's sigmoid/grouped-top-k router and SwiGLU experts to MoEConfig.

    Correction bias affects selection only and is never updated. Router storage and
    compute remain fp32. Automodel applies routed_scaling_factor to routing weights;
    HF applies it after expert aggregation, differing only in rounding."""
    n_group = getattr(config, "n_group", None) or 0
    topk_group = getattr(config, "topk_group", None) or 0
    defaults = dict(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=getattr(config, "moe_intermediate_size", config.intermediate_size),
        n_routed_experts=getattr(config, "num_experts", 0) or 0,
        n_shared_experts=0,
        n_activated_experts=getattr(config, "num_experts_per_tok", 1),
        n_expert_groups=n_group if n_group > 1 else 0,
        n_limited_groups=topk_group if n_group > 1 else 0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid_with_bias",
        route_scale=float(getattr(config, "routed_scaling_factor", 1.0) or 1.0),
        aux_loss_coeff=0.0,
        norm_topk_prob=bool(getattr(config, "norm_topk_prob", True)),
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=False,
        force_e_score_correction_bias=True,
        gate_dtype=torch.float32,
        router_weights_fp32=True,
        dtype=model_dtype,
    )
    if overrides:
        defaults.update(overrides)
    return MoEConfig(**defaults)


# Child-module names kept out of the model dtype cast (see BeamForCausalLM.initialize_weights): the MoE router.
FP32_SKIP_MODULES = ("gate",)


def resolve_backend_gate_precision(backend: BackendConfig) -> BackendConfig:
    """Default router compute to fp32 and reject lower precision to preserve HF expert selection."""
    if backend.gate_precision is None:
        return replace(backend, gate_precision=torch.float32)
    if backend.gate_precision != torch.float32:
        raise ValueError(
            f"Beam routes in fp32 to match the reference model; backend.gate_precision must be float32 or unset, "
            f"got {backend.gate_precision}."
        )
    return backend


class BeamModel(nn.Module):
    """Embedding, decoder layers and final norm (the ``model.`` subtree of the HF checkpoint)."""

    def __init__(
        self,
        config: BeamConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
        self.backend = backend
        self.config = config
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.moe_config = moe_config or beam_moe_config(config, model_dtype, moe_overrides)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=model_dtype)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        # One rotary embedding per layer type that uses RoPE (``rope_parameters[layer_type]`` not None); layer
        # types with ``None`` run NoPE + SSMax and get no frequencies.
        device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else None
        # Named ``rotary_emb`` so the pipeline splitter keeps it on every stage (it whitelists that attribute).
        self.rotary_emb = nn.ModuleDict()
        for layer_type in dict.fromkeys(config.layer_types):
            params = layer_rope_parameters(config, layer_type)
            if params is None:
                continue
            self.rotary_emb[layer_type] = RotaryEmbedding(
                head_dim=self.head_dim,
                base=params["rope_theta"],
                dtype=torch.float32,
                initial_context_length=params.get("original_max_position_embeddings", 4096),
                scaling_factor=params.get("factor", 1.0),
                ntk_alpha=params.get("beta_slow", 1.0),
                ntk_beta=params.get("beta_fast", 32.0),
                partial_rotary_factor=params.get("partial_rotary_factor", 1.0),
                device=device,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Run this pipeline stage with one shared FP32 RoPE-table pair per layer type.

        Args:
            input_ids: Integer tensor [batch, sequence] or packed [tokens]; later PP stages
                receive floating hidden states with an extra final hidden axis. Positions are local under CP.
            position_ids: Optional integer tensor [batch, sequence] or [tokens], absolute within each document.
            attention_mask: Optional [batch, sequence] bool/0-1 mask; True/1 means valid. None on flex.
            padding_mask: Optional bool tensor matching the position axes; True skips routing.
            **attn_kwargs: Backend metadata: qkv_format (bshd/thd), integer cu_seqlens and
                cu_seqlens_padded [documents + 1], scalar max_seqlen, cp_size and cp_rank.

        Returns:
            Hidden states [batch, sequence, hidden] or [tokens, hidden], local under CP,
            in the final layer/norm dtype."""
        if position_ids is None:
            # ``input_ids`` is [B, S] (or [T] packed); on later pipeline stages it carries the hidden states
            # [B, S, H] instead, so the sequence length is the second dimension, never the last.
            seq_len = input_ids.shape[1] if input_ids.dim() > 1 else input_ids.shape[0]
            position_ids = torch.arange(0, seq_len, device=input_ids.device)
            if input_ids.dim() > 1:
                position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
        if self.backend.attn == "flex":
            reject_flex_attention_mask(attention_mask)  # presence check only, no device sync
        freqs_by_type = {
            layer_type: position_ids_to_freqs_cis(
                rotary_emb,
                position_ids,
                qkv_format=attn_kwargs.get("qkv_format", "bshd"),
                for_fused_rope=self.backend.rope_fusion,
                cp_size=attn_kwargs.get("cp_size", 1),
            )
            for layer_type, rotary_emb in self.rotary_emb.items()
        }
        # Unfused RoPE: the expanded fp32 (cos, sin) pair once per RoPE layer type per forward, shared by every layer
        # of that type (built inside each layer, autograd would retain one pair per layer). Fused RoPE keeps
        # ``freqs_cis``. Nothing is cached across forwards.
        tables_by_type = {} if self.backend.rope_fusion else {lt: hf_rope_tables(f) for lt, f in freqs_by_type.items()}
        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids
        for layer in self.layers.values():
            # ``h`` is passed positionally on purpose: torch's activation checkpoint saves and restores the CUDA RNG
            # state only for devices found in the positional tensor arguments (``get_device_states(*args)``); with a
            # keyword-only call the recompute would draw fresh dropout masks (attention_dropout > 0) and produce
            # gradients for a different realization than the forward.
            h = layer(
                h,
                freqs_cis=freqs_by_type.get(layer.layer_type),
                rope_tables=tables_by_type.get(layer.layer_type),
                position_ids=position_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )
        return self.norm(h) if self.norm else h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize this stage's parameters and select the rotary-buffer device."""
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            for rotary_emb in self.rotary_emb.values():
                rotary_emb.device = buffer_device
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class BeamForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Beam causal LM on Automodel's MoE stack (see module docstring)."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # Our own forward handles PP stage routing, THD and CP like the other native MoE models.
    _pp_keep_self_forward: bool = True
    # fp32 router contract (HF ``_keep_in_fp32_modules_strict``): the router weight and the correction bias keep
    # their original fp32 values through the model cast and the checkpoint load, compute in fp32 (``gate_precision``),
    # and get their own fp32 FSDP unit (see ``beam.parallelization``), so they stay F32 in checkpoints and exports.
    _keep_in_fp32_modules_strict: list[str] = ["mlp.gate.weight", "e_score_correction_bias"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        # CP requires TE: full layers use p2p; sliding layers use all_gather, or a2a for packed input (CP <= KV heads).
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: BeamConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> BeamForCausalLM:
        """Construct the native model from an HF BeamConfig and optional backend/MoE overrides."""
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> BeamForCausalLM:
        """Construct from the checkpoint's config; framework checkpoint loading supplies the weights."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: BeamConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = resolve_backend_gate_precision(backend or BackendConfig())
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = BeamModel(config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides)
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=model_dtype
        )
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = BeamStateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=model_dtype
            )

    def get_input_embeddings(self) -> nn.Module | None:
        """Return this stage's embedding module, or None when another stage owns it."""
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module | None) -> None:
        """Replace this stage's embedding module; None marks a later pipeline stage."""
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module | None:
        """Return this stage's LM head, or None when another stage owns it."""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module | None) -> None:
        """Replace this stage's LM head; None marks a non-final pipeline stage."""
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Return soft-capped logits or pipeline-stage hidden states.

        Args:
            input_ids: Integer tensor [batch, sequence] or packed [tokens]; later PP stages
                receive floating hidden states with an extra final hidden axis. Positions are local under CP.
            position_ids: Optional integer tensor [batch, sequence] or [tokens], absolute within each document.
            attention_mask: Optional [batch, sequence] bool/0-1 mask; True/1 means valid. None on flex.
            padding_mask: Optional bool tensor matching the position axes; True skips routing.
            logits_to_keep: Zero for all positions, a positive final-position count, or integer
                tensor [selected_positions] indexing the sequence axis.
            output_hidden_states: Whether to return the final hidden states.
            **attn_kwargs: Backend metadata: qkv_format (bshd/thd), integer cu_seqlens and
                cu_seqlens_padded [documents + 1], scalar max_seqlen, cp_size and cp_rank.

        Returns:
            CausalLMOutputWithPast: logits [batch, selected_sequence, vocab] or packed
            [1, selected_tokens, vocab], FP32 with softcapping, otherwise in the head's dtype.
            Stages without a head pass hidden states [batch, sequence, hidden] or [1, tokens, hidden]
            in logits. Optional hidden_states has that full hidden layout. Positions are local
            under CP; loss, cache and attention fields are unset."""
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        is_thd = attn_kwargs.get("qkv_format") == "thd"
        if is_thd:
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None
        if getattr(self, "_cp_enabled", False):
            # Set by the CP parallelizer (moe.parallelizer.apply_cp). The sequence is sharded per rank while a
            # padded batch mask would span the full sequence, and TE rejects padding masks under CP; causal
            # masking is implicit. ``padding_mask`` (normally supplied by the CP sharder) still drives routing.
            attention_mask = None
        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        out = compute_lm_head_logits(
            self.lm_head, hidden, logits_to_keep, is_thd=is_thd, output_hidden_states=output_hidden_states
        )
        # Final-logit soft cap, as in HF: cap * tanh(logits / cap), computed in fp32. Only the stage
        # that owns the head has logits (on other PP stages ``lm_head`` is None and the hidden
        # states pass through in the logits slot).
        if self.final_logit_softcapping and self.lm_head is not None and out.logits is not None:
            cap = float(self.final_logit_softcapping)
            out.logits = (out.logits.float() / cap).tanh() * cap
        return out

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        """Initialize weights and cast to dtype while preserving original FP32 router values."""
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
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
        # The router Gate (fp32 weight + fp32 correction bias) is detached during the cast instead of being cast and
        # restored: on FSDP2-sharded (DTensor) parameters the keyword restore only swaps the outer tensor metadata
        # and leaves a bf16-rounded local shard behind, which would violate the fp32 router contract.
        cast_model_to_dtype(self, dtype, skip_modules=FP32_SKIP_MODULES)
        with buffer_device:
            for rotary_emb in self.model.rotary_emb.values():
                rotary_emb.device = buffer_device


BeamForCausalLM.parallelizer = PARALLELIZER
ModelClass = BeamForCausalLM

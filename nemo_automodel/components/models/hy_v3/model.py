# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""HYV3ForCausalLM — Tencent Hy3-preview (295B MoE) SFT support.

Architecture (from tencent/Hy3-preview config.json):
  - 80 transformer layers; layer 0 is dense, layers 1-79 are MoE
  - MoE: 192 routed experts + 1 shared expert, top-8 activated
  - Sigmoid routing with expert-bias correction (e_score_correction_bias)
  - GQA: 64 Q heads, 8 KV heads, head_dim=128
  - Per-head QK RMSNorm before RoPE
  - 256K context, rope_theta=11158840
"""

from dataclasses import dataclass
from typing import Any, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.lm_head import LMHead, slice_logits_to_keep
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.hy_v3.layers import HYV3Attention
from nemo_automodel.components.models.hy_v3.state_dict_adapter import HYV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Block(nn.Module):
    def __init__(self, layer_idx: int, config: Any, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.self_attn = HYV3Attention(config, backend)

        # Layers 0..(first_k_dense_replace-1) are dense; the rest are MoE.
        first_k_dense = getattr(config, "first_k_dense_replace", 1)
        if layer_idx < first_k_dense:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear)
        else:
            self.mlp = MoE(moe_config, backend)

        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        x = x + attn_out

        mlp_out = self._mlp(x=self.post_attention_layernorm(x), padding_mask=padding_mask)
        x = x + mlp_out
        return x

    def _mlp(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if isinstance(self.mlp, MLP):
            return self.mlp(x)
        assert isinstance(self.mlp, MoE)
        return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class HYV3Model(nn.Module):
    def __init__(
        self,
        config: Any,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides.")

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.num_experts,
            n_shared_experts=getattr(config, "num_shared_experts", 0),
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="sigmoid",
            route_scale=getattr(config, "router_scaling_factor", 1.0),
            aux_loss_coeff=0.0,
            norm_topk_prob=getattr(config, "route_norm", False),
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
            # Ensures e_score_correction_bias buffer is created so HF checkpoints load cleanly
            force_e_score_correction_bias=getattr(config, "moe_router_enable_expert_bias", False),
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
        )
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        base, rope_scaling, _ = get_rope_config(config)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
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
        """Run the decoder layers on padded or packed (THD) input.

        Packed input arrives batched and is squeezed to [tokens, ...] for the layers, which TE's packed kernels
        take; the hidden states come back batched.

        Args:
            input_ids: Token IDs of shape [batch, sequence], or packed token IDs of shape [1, tokens]. Later
                pipeline stages pass hidden states of shape [batch, sequence, hidden] or [1, tokens, hidden].
            position_ids: Optional positions of shape [batch, sequence]; packed input requires them, as
                [1, tokens].
            attention_mask: Optional Tensor of shape [batch, sequence], 1 at valid tokens and 0 at padding, or,
                with SDPA and an explicit padding_mask, a dense mask of shape [batch, 1, sequence, sequence];
                packed input ignores it.
            padding_mask: Optional bool mask of shape [batch, sequence] or [1, tokens], true at padding.
            **attn_kwargs: Backend metadata. Packed input sets ``qkv_format="thd"`` and passes int32
                ``cu_seqlens`` and ``cu_seqlens_padded`` of shape [1, documents + 1], padded with -1000, and
                ``max_seqlen``.

        Returns:
            Hidden states of shape [batch, sequence, hidden], or [1, tokens, hidden] for packed input.
        """
        if attn_kwargs.get("qkv_format") == "thd":  # TE's packed kernels take [tokens, ...]
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        if position_ids is None:
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            )

        freqs_cis = position_ids_to_freqs_cis(
            self.rotary_emb,
            position_ids,
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        for layer in self.layers.values():
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )

        h = self.norm(h) if self.norm else h
        return h.unsqueeze(0) if h.dim() == 2 else h  # packed (THD) [tokens, hidden] -> [1, tokens, hidden]

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class HYV3ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules_strict = ["mlp.gate.e_score_correction_bias"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: Any,
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
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=False)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        *,
        logits_dtype: torch.dtype | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        self.model = HYV3Model(config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, dtype=torch.bfloat16, logits_dtype=logits_dtype)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = HYV3StateDictAdapter(
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
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        compute_logits: bool = True,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run the decoder and project the kept positions to vocabulary logits.

        Args:
            input_ids: Token IDs of shape [batch, sequence], or packed (THD) IDs of shape [1, tokens], which the
                decoder squeezes to [tokens]. Later pipeline stages pass hidden states of shape
                [batch, sequence, hidden] or [1, tokens, hidden].
            position_ids: Optional positions of shape [batch, sequence], or [1, tokens] for packed input, which
                requires them.
            attention_mask: Optional Tensor of shape [batch, sequence], 1 at valid tokens and 0 at padding, or,
                with SDPA and an explicit padding_mask, a dense mask of shape [batch, 1, sequence, sequence];
                packed input ignores it.
            padding_mask: Optional bool Tensor of shape [batch, sequence], or [1, tokens] when packed, true at
                padding; when None, the MoE layers derive it from attention_mask.
            logits_to_keep: ``0`` keeps every position, a positive int keeps the last N positions, and an integer
                Tensor of shape [kept] keeps those positions.
            compute_logits: Whether to materialize logits. Fused-linear loss callers disable it off-PP when they consume
                the final hidden states directly; the LM head then does not run and the logits are empty, of shape
                [..., 0].
            output_hidden_states: Whether the output carries the final hidden states; None reads
                ``config.output_hidden_states``.
            **attn_kwargs: Backend metadata for the decoder. Packed input sets ``qkv_format="thd"`` and passes
                int32 ``cu_seqlens`` and ``cu_seqlens_padded`` of shape [1, documents + 1], padded with -1000, and
                ``max_seqlen``.

        Returns:
            CausalLMOutputWithPast whose ``logits`` have shape [batch, kept, vocab], or [1, kept, vocab] for packed
            input, in the head's ``logits_dtype`` when set and otherwise in its weight dtype, which autocast may
            lower; on a pipeline stage without the head, ``logits`` holds the hidden states instead.
            ``hidden_states``, when requested, are the final hidden states of shape [batch, sequence, hidden], or
            [1, tokens, hidden] for packed input.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )

        if compute_logits:
            logits = hidden if self.lm_head is None else self.lm_head(slice_logits_to_keep(hidden, logits_to_keep))
        else:
            logits = hidden.new_empty((*hidden.shape[:-1], 0))
        return CausalLMOutputWithPast(logits=logits, hidden_states=hidden if output_hidden_states else None)

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.model.layers.values():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
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

        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


ModelClass = HYV3ForCausalLM

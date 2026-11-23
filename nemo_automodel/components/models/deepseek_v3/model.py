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

from dataclasses import dataclass, replace
from typing import Any, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

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
from nemo_automodel.components.models.common.utils import yield_fp32_model
from nemo_automodel.components.models.deepseek_v3.layers import MLA
from nemo_automodel.components.models.deepseek_v3.rope_utils import freqs_cis_from_position_ids, precompute_freqs_cis
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Block(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        config: DeepseekV3Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        self.self_attn = MLA(config, backend, latent_norm_eps=1e-6)
        self.is_moe_layer = layer_idx >= config.first_k_dense_replace

        # Thread dtype from config.torch_dtype so the block's own params stay
        # aligned with the rest of the model (fp32 under fp32 master weights).
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        if not self.is_moe_layer:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)
        else:
            self.mlp = MoE(moe_config, backend)
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            padding_mask (torch.Tensor): Boolean tensor indicating padding positions.

        Returns:
            torch.Tensor: Output tensor after block computation.
            torch.Tensor | None: Auxiliary loss for load balancing (if applicable).
        """
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        x = x + attn_out

        mlp_out = self._mlp(
            x=self.post_attention_layernorm(x),
            padding_mask=padding_mask,
        )
        x = x + mlp_out

        return x

    def _mlp(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.is_moe_layer:
            return self.mlp(x)
        return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class DeepseekV3Model(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
        # Resolve model dtype once; thread it explicitly to every sub-module
        # so fp32 master weights work even when construction is not wrapped in
        # local_torch_dtype().
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=1e-3,
            score_func="sigmoid",
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0,
            norm_topk_prob=config.norm_topk_prob,
            # HF gathers topk_weights from the fp32 scores and returns them with
            # no cast back, so expert compute sees fp32. Without this, Gate.forward
            # applies weights.type_as(x) and hands over bf16. Set in moe_defaults,
            # above the moe_overrides update, so a caller override still wins.
            router_weights_fp32=True,
            dtype=model_dtype,
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=model_dtype)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )

        self.max_seq_len = config.max_position_embeddings
        rope_theta, rope_scaling, _ = get_rope_config(config)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.qk_rope_head_dim,
                self.max_seq_len,
                rope_theta,
                rope_scaling,
            ),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Run the decoder layers on padded or packed (THD) input.

        Packed input arrives batched and is squeezed to [tokens, ...] for the layers, which TE's packed kernels
        take; the hidden states come back batched.
        Callers that already squeezed packed input for their own preprocessing pass [tokens] layouts,
        which the squeeze leaves unchanged.

        Args:
            input_ids: Token IDs of shape [batch, sequence], or packed token IDs of shape [1, tokens]. Later
                pipeline stages pass hidden states of shape [batch, sequence, hidden] or [1, tokens, hidden].
            inputs_embeds: Optional embeddings of shape [batch, sequence, hidden] used instead of input_ids,
                or packed [tokens, hidden] from a caller that already squeezed packed input.
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

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
            )

        with torch.no_grad():
            freqs_cis = freqs_cis_from_position_ids(
                position_ids,
                self.freqs_cis,
                qkv_format=attn_kwargs.get("qkv_format", "bshd"),
                for_fused_rope=self.backend.rope_fusion,
                cp_size=attn_kwargs.get("cp_size", 1),
            )

        h = inputs_embeds

        # Apply the transformer layers.
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

    def update_moe_gate_bias(self) -> None:
        """Update correction biases only for gates with adaptation enabled."""
        with torch.no_grad():
            for _, block in self.layers.named_children():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            rope_theta, rope_scaling, _ = get_rope_config(self.config)
            self.freqs_cis = precompute_freqs_cis(
                self.config.qk_rope_head_dim,
                self.max_seq_len,
                rope_theta,
                rope_scaling,
            )
            self.freqs_cis = self.freqs_cis.to(buffer_device)
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class DeepseekV3ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    _uses_native_fa4 = True
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        # CP demonstrated on the Moonlight-16B-A3B convergence run
        # (examples/convergence/tulu3/models/moonlight-16b/moonlight_16b_ep8_cp2_flashoptim.yaml).
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: DeepseekV3Config,
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
        config = DeepseekV3Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: DeepseekV3Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        *,
        logits_dtype: torch.dtype | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        # HF's DeepseekV3TopkRouter runs the router projection in fp32, so default
        # gate_precision to fp32. Scoring is already fp32 via Gate's score_dtype
        # default - this covers the projection only.
        # replace() rather than in-place: the caller's BackendConfig may be shared
        # with other models, which must not inherit a model-owned default.
        resolved_backend = backend or BackendConfig()
        if resolved_backend.gate_precision is None:
            resolved_backend = replace(resolved_backend, gate_precision=torch.float32)
        self.backend = resolved_backend
        self.model = DeepseekV3Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, dtype=model_dtype, logits_dtype=logits_dtype)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = DeepSeekV3StateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=model_dtype
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
                ``max_seqlen``; context parallelism adds ``cp_size`` and ``cp_rank``.

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

        hidden_states = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )

        if compute_logits:
            logits = (
                hidden_states
                if self.lm_head is None
                else self.lm_head(slice_logits_to_keep(hidden_states, logits_to_keep))
            )
        else:
            logits = hidden_states.new_empty((*hidden_states.shape[:-1], 0))
        return CausalLMOutputWithPast(logits=logits, hidden_states=hidden_states if output_hidden_states else None)

    def update_moe_gate_bias(self) -> None:
        """Update correction biases only for gates with adaptation enabled."""
        with torch.no_grad():
            for _, block in self.model.layers.named_children():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        # Sample the random init in fp32, then cast to the resident dtype. Directly sampling in
        # bf16 distorts the init variance/mean and produces exploding early gradients for
        # from-scratch pretraining; see yield_fp32_model for the full rationale.
        with yield_fp32_model(self, dtype):
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

        with buffer_device:
            rope_theta, rope_scaling, _ = get_rope_config(self.config)
            self.model.freqs_cis = precompute_freqs_cis(
                self.config.qk_rope_head_dim,
                self.model.max_seq_len,
                rope_theta,
                rope_scaling,
            )


ModelClass = DeepseekV3ForCausalLM

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
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig, get_rope_config, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.step3p5.layers import (
    Step3p5Attention,
    Step3p5MLP,
    Step3p5RMSNorm,
)
from nemo_automodel.components.models.step3p5.state_dict_adapter import Step3p5StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def parse_moe_layers_enum(moe_layers_enum: str | int | tuple | list | None, num_hidden_layers: int) -> set[int]:
    """Parse moe_layers_enum to get set of MoE layer indices.

    Args:
        moe_layers_enum: Tuple/list of layer indices, integer, comma-separated string, or None.
            HF Step-3.5-Flash uses tuple format like (3, 4, 5, ..., 44).
        num_hidden_layers: Total number of hidden layers.

    Returns:
        Set of layer indices that should be MoE layers.
    """
    if moe_layers_enum is not None:
        if isinstance(moe_layers_enum, (tuple, list)):
            return set(int(i) for i in moe_layers_enum)
        elif isinstance(moe_layers_enum, int):
            return {moe_layers_enum}
        elif isinstance(moe_layers_enum, str):
            return set(int(i) for i in moe_layers_enum.strip().split(","))
        else:
            raise ValueError(f"Unsupported moe_layers_enum type: {type(moe_layers_enum)}")
    else:
        # Default: all layers except layer 0 are MoE
        return set(range(1, num_hidden_layers))


def _keep_step_router_bias_fp32(module: nn.Module) -> None:
    """Keep Step router correction bias in fp32 after module-wide dtype casts."""
    for submodule in module.modules():
        bias = getattr(submodule, "e_score_correction_bias", None)
        if bias is not None:
            submodule.e_score_correction_bias.data = bias.data.float()

        master = getattr(submodule, "e_score_correction_bias_master", None)
        if master is not None:
            submodule.e_score_correction_bias_master.data = master.data.float()


class Block(nn.Module):
    """Step3p5 transformer block with attention, MLP/MoE, and shared experts."""

    def __init__(
        self,
        layer_idx: int,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Handle need_fp32_gate config for MoE gate precision
        if getattr(config, "need_fp32_gate", False) and backend.gate_precision is None:
            backend.gate_precision = torch.float32

        # Attention
        self.self_attn = Step3p5Attention(config, layer_idx, backend)

        # Determine attention type for this layer
        layer_types = getattr(config, "layer_types", [])
        self.attention_type = layer_types[layer_idx] if layer_types else "full_attention"

        # Determine if this is an MoE layer
        moe_layers_enum = getattr(config, "moe_layers_enum", None)
        moe_layers = parse_moe_layers_enum(moe_layers_enum, config.num_hidden_layers)
        self.is_moe_layer = layer_idx in moe_layers

        # Get swiglu limits for this layer
        swiglu_limits_shared = getattr(config, "swiglu_limits_shared", None)
        swiglu_limits = getattr(config, "swiglu_limits", None)

        swiglu_limit_shared = None
        if swiglu_limits_shared and swiglu_limits_shared[layer_idx]:
            if swiglu_limits_shared[layer_idx] != 0:
                swiglu_limit_shared = swiglu_limits_shared[layer_idx]

        swiglu_limit = None
        if swiglu_limits and swiglu_limits[layer_idx]:
            if swiglu_limits[layer_idx] != 0:
                swiglu_limit = swiglu_limits[layer_idx]

        # MLP or MoE with shared expert
        if self.is_moe_layer:
            # Preserve every generic MoE capability while applying Step's per-layer
            # clamp to the field consumed by all routed-expert implementations.
            layer_moe_config = replace(
                moe_config,
                n_shared_experts=0,  # Shared expert handled separately in Step3p5
                expert_activation="swiglu_step",
                swiglu_limit=swiglu_limit if swiglu_limit is not None else moe_config.swiglu_limit,
            )
            self.moe = MoE(layer_moe_config, backend)

            # Shared expert with its own intermediate size and swiglu limit
            # HF uses share_expert_dims (plural), but we also support share_expert_dim for compatibility
            share_expert_dim = getattr(config, "share_expert_dims", None) or getattr(
                config, "share_expert_dim", config.intermediate_size
            )
            self.share_expert = Step3p5MLP(
                config,
                backend,
                intermediate_size=share_expert_dim,
                swiglu_limit=swiglu_limit_shared,
            )
            self.mlp = None
        else:
            # Regular MLP for non-MoE layers
            self.mlp = Step3p5MLP(
                config,
                backend,
                intermediate_size=config.intermediate_size,
                swiglu_limit=swiglu_limit_shared,
            )
            self.moe = None
            self.share_expert = None

        # Layer norms using Step3p5RMSNorm
        self.input_layernorm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        # Attention
        residual = x
        x = self.input_layernorm(x)
        attn_out = self.self_attn(
            x,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **attn_kwargs,
        )
        x = residual + attn_out

        # FFN (MLP or MoE + shared expert)
        residual = x
        x = self.post_attention_layernorm(x)

        if self.is_moe_layer:
            share_out = self.share_expert(x)
            moe_out = self.moe(x, padding_mask)
            x = residual + share_out + moe_out
        else:
            mlp_out = self.mlp(x)
            x = residual + mlp_out

        return x

    def init_weights(self, buffer_device: torch.device) -> None:
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)

        if self.mlp is not None:
            self.mlp.init_weights(buffer_device)
        if self.moe is not None:
            self.moe.init_weights(buffer_device)
        if self.share_expert is not None:
            self.share_expert.init_weights(buffer_device)


class Step3p5Model(nn.Module):
    """Step3p5 transformer model."""

    def __init__(
        self,
        config: Any,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
        self.config.num_experts = config.moe_num_experts

        # Build MoE config from Step3p5 config
        use_router_bias = getattr(config, "use_moe_router_bias", False)
        router_activation = getattr(config, "moe_router_activation", "softmax")
        if use_router_bias and router_activation == "sigmoid":
            score_func = "sigmoid_with_bias"
        else:
            score_func = "sigmoid" if router_activation == "sigmoid" else "softmax"

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=getattr(config, "moe_intermediate_size", config.intermediate_size),
            n_routed_experts=self.config.num_experts,
            n_shared_experts=0,  # Step3p5 handles shared experts separately
            n_activated_experts=getattr(config, "moe_top_k", 2),
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func=score_func,
            route_scale=getattr(config, "moe_router_scaling_factor", 1.0),
            aux_loss_coeff=0.0,
            norm_topk_prob=True,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu_step",
            dtype=get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16),
            force_e_score_correction_bias=use_router_bias,
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        # Embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16),
        )

        # Transformer blocks
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)

        # Final norm
        self.norm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embeddings
        # For Step3p5, we use the base rope_theta (first layer's theta if it's a list)
        rope_theta = config.rope_theta
        if isinstance(rope_theta, list):
            rope_theta = rope_theta[0]

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # Get partial_rotary_factor for the first layer (used for RotaryEmbedding initialization)
        partial_rotary_factors = getattr(config, "partial_rotary_factors", None)
        partial_rotary_factor = partial_rotary_factors[0] if partial_rotary_factors else 1.0

        _, rope_scaling, _ = get_rope_config(config)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=rope_theta,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            partial_rotary_factor=partial_rotary_factor,
            device=torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
        )

        # Check if model has sliding window attention
        layer_types = getattr(config, "layer_types", [])
        self.has_sliding_layers = "sliding_attention" in layer_types

    def _apply(self, fn):
        result = super()._apply(fn)
        _keep_step_router_bias_fp32(self)
        return result

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
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Step3p5Model requires input_ids or inputs_embeds.")

        if inputs_embeds is None and self.embed_tokens is None:
            if input_ids is not None and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32):
                inputs_embeds = input_ids
                input_ids = None
            else:
                raise ValueError("inputs_embeds must be provided when embed_tokens is not present.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = (
                torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
                .unsqueeze(0)
                .expand(inputs_embeds.shape[0], -1)
            )

        # Compute freqs_cis from RotaryEmbedding
        freqs_cis = position_ids_to_freqs_cis(
            self.rotary_emb,
            position_ids,
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

        h = inputs_embeds

        for layer in self.layers.values():
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                **attn_kwargs,
            )

        h = self.norm(h) if self.norm else h
        return h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            # Ensure rotary embedding uses correct device
            self.rotary_emb.device = buffer_device

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class Step3p5ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Step3p5 model for causal language modeling."""

    _keep_in_fp32_modules = ["rotary_emb"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = False

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
        # Import Step3p5Config dynamically to avoid hard dependency
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except Exception:
            raise ValueError(
                f"Could not load config from {pretrained_model_name_or_path}. "
                "Make sure the model path contains a valid config.json."
            )
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = Step3p5Model(config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides)
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=model_dtype
        )

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Step3p5StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=model_dtype,
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
        output_hidden_states: Optional[bool] = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Forward pass returning :class:`~transformers.modeling_outputs.CausalLMOutputWithPast`.

        Supports both BSHD format (``input_ids`` shape ``[B, S]``) and THD format
        (``input_ids`` shape ``[1, T]``); when ``attn_kwargs["qkv_format"] == "thd"``,
        inputs are squeezed to THD before the base-model forward and ``logits`` (and the
        final ``hidden_states``) are unsqueezed back to a leading-batch dimension on exit.

        Args:
            input_ids: Input token IDs.
            position_ids: Optional position indices.
            attention_mask: Optional 2D padding mask.
            padding_mask: Optional padding mask used by the THD squeeze helper.
            logits_to_keep: If ``0`` (default), compute logits for all positions; if ``> 0``
                (or a tensor), only compute logits for the last ``logits_to_keep`` positions
                (avoids materialising the full logit matrix during generation / fused CE).
            output_hidden_states: Whether to carry the final hidden states on the output.
            **attn_kwargs: Additional arguments forwarded to the base model.

        Returns:
            :class:`~transformers.modeling_outputs.CausalLMOutputWithPast` with ``logits`` and,
            when ``output_hidden_states`` is set, the final ``hidden_states``.
        """
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

        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )

        return compute_lm_head_logits(
            self.lm_head, hidden, logits_to_keep, is_thd=is_thd, output_hidden_states=output_hidden_states
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )

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
            # Ensure rotary embedding uses correct device after dtype move
            self.model.rotary_emb.device = buffer_device


ModelClass = Step3p5ForCausalLM

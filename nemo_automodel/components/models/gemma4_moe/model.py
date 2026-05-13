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

"""Gemma4 MoE NeMo Automodel support.

Replaces the HF-native Gemma4 MoE (dense matmul over all experts) with NeMo's
GroupedExperts backend, enabling Expert Parallelism (EP) via the standard
MoE parallelizer.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.shared.import_utils import UnavailableError, UnavailableMeta


def _make_missing(name: str):
    return UnavailableMeta(name, (), {"_msg": "transformers.models.gemma4 is not available."})


try:
    from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
    from transformers.models.gemma4 import modeling_gemma4 as _g4
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config,
        Gemma4TextConfig,
    )

    Gemma4RMSNorm = _g4.Gemma4RMSNorm
    Gemma4TextModel = _g4.Gemma4TextModel
    Gemma4TextScaledWordEmbedding = _g4.Gemma4TextScaledWordEmbedding

    # These classes were renamed in transformers 5.5 (Gemma4X → Gemma4TextX)
    # TODO have only transformers 5.5 version of these classes ?
    Gemma4Attention = getattr(_g4, "Gemma4TextAttention", None) or _g4.Gemma4Attention
    Gemma4DecoderLayer = getattr(_g4, "Gemma4TextDecoderLayer", None) or _g4.Gemma4DecoderLayer
    Gemma4MLP = getattr(_g4, "Gemma4TextMLP", None) or _g4.Gemma4MLP
    Gemma4RotaryEmbedding = getattr(_g4, "Gemma4TextRotaryEmbedding", None) or _g4.Gemma4RotaryEmbedding
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration as HFGemma4ForConditionalGeneration,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4Model as HFGemma4Model,
    )

    _GEMMA4_HF_AVAILABLE = True
except (ModuleNotFoundError, ImportError, AttributeError):
    _GEMMA4_HF_AVAILABLE = False
    Gemma4Config = _make_missing("Gemma4Config")
    Gemma4TextConfig = _make_missing("Gemma4TextConfig")
    Gemma4Attention = _make_missing("Gemma4Attention")
    Gemma4DecoderLayer = _make_missing("Gemma4DecoderLayer")
    Gemma4MLP = _make_missing("Gemma4MLP")
    Gemma4RMSNorm = _make_missing("Gemma4RMSNorm")
    Gemma4RotaryEmbedding = _make_missing("Gemma4RotaryEmbedding")
    Gemma4TextModel = _make_missing("Gemma4TextModel")
    Gemma4TextScaledWordEmbedding = _make_missing("Gemma4TextScaledWordEmbedding")
    HFGemma4ForConditionalGeneration = _make_missing("Gemma4ForConditionalGeneration")
    HFGemma4Model = _make_missing("Gemma4Model")
    BaseModelOutputWithPast = _make_missing("BaseModelOutputWithPast")
    ModelOutput = _make_missing("ModelOutput")

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.gemma4_moe.mtp import build_gemma4_mtp, build_mtp_config_from_kwargs
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE, MoEConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import Gemma4MoEStateDictAdapter

if _GEMMA4_HF_AVAILABLE:

    @dataclass
    class Gemma4WithMTPCausalLMOutput(ModelOutput):
        """Gemma4 forward output augmented with MTP fields.

        ``mtp_per_depth_h`` and ``mtp_loss_scaling_factor`` must be declared
        dataclass fields (rather than attributes set after construction) so
        they survive output-restructuring layers like FSDP2's mixed-precision
        output cast, which rebuild ``ModelOutput`` instances from declared
        fields only (see PR #2161 commit 0b2889ab).
        """

        logits: Optional[torch.Tensor] = None
        hidden_states: Optional[tuple[torch.Tensor, ...]] = None
        mtp_per_depth_h: Optional[list[torch.Tensor]] = None
        mtp_loss_scaling_factor: Optional[float] = None
else:
    Gemma4WithMTPCausalLMOutput = _make_missing("Gemma4WithMTPCausalLMOutput")


# ---------------------------------------------------------------------------
# Gemma4-specific router that outputs NeMo-compatible (weights, indices)
# ---------------------------------------------------------------------------
class Gemma4Gate(nn.Module):
    """Gemma4 Router reimplemented to output NeMo Gate format.

    HF Gemma4Router applies: RMSNorm(no_scale) → root_size scaling → learnable
    scale → Linear → softmax → top-k → renormalize which is different from the standard Gate class in layer.py.
    This class reproduces that logic but returns (weights, indices, aux_loss) as expected by GroupedExperts.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_experts = config.num_experts
        self.topk = config.top_k_experts
        self.num_experts = num_experts

        self.norm = Gemma4RMSNorm(hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(hidden_size, num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        scalar_root_size = hidden_size**-0.5
        self.register_buffer("root_size", torch.tensor(scalar_root_size), persistent=False)

    def forward(self, x, token_mask=None, cp_mesh=None):
        x_norm = self.norm(x)
        x_norm = x_norm * self.root_size.to(x_norm.dtype)
        x_norm = x_norm * self.scale.to(x_norm.dtype)

        expert_scores = self.proj(x_norm)
        router_probs = F.softmax(expert_scores, dim=-1)

        # Top-k on raw scores (matching HF Gemma4Router behaviour)
        _, indices = torch.topk(expert_scores, k=self.topk, dim=-1)
        weights = router_probs.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        return weights, indices, None

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        pass


# ---------------------------------------------------------------------------
# Gemma4MoE – MoE subclass with Gemma4Gate instead of the default Gate
# ---------------------------------------------------------------------------
class Gemma4MoE(MoE):
    """NeMo MoE that uses Gemma4Gate (with pre-norm routing) instead of
    the standard Gate. Subclasses MoE so that ``isinstance(m, MoE)`` is True,
    which the EP parallelizer relies on."""

    def __init__(self, moe_config: MoEConfig, backend: BackendConfig, text_config: Gemma4TextConfig):
        super().__init__(moe_config, backend)
        # Replace the gate created by MoE.__init__ with Gemma4-specific gate
        self.gate = Gemma4Gate(text_config)

    def forward(self, x, padding_mask=None, cp_mesh=None, *, gate_input=None):
        """Forward with optional separate gate input.

        HF Gemma4 passes unnormalized residual to the router and normalized
        input to the experts.  The decoder layer calls this with
        ``gate_input=x`` (raw residual) so the gate receives unnormalized
        input while experts receive ``pre_feedforward_layernorm_2(x)``.
        """
        if cp_mesh is None:
            cp_mesh = self.cp_mesh

        shape = x.size()
        x = x.view(-1, self.dim)
        if padding_mask is not None:
            token_mask = (~padding_mask).flatten()
        else:
            token_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        # Use separate gate_input for routing when provided (fixes double-norm)
        g = gate_input.view(-1, self.dim) if gate_input is not None else x
        weights, indices, aux_loss = self.gate(g, token_mask, cp_mesh)

        x_latent = self.fc1_latent_proj(x) if self.fc1_latent_proj is not None else x
        y = self.experts(x_latent, token_mask, weights, indices)
        if self.fc2_latent_proj is not None:
            y = self.fc2_latent_proj(y)
        return y.view(shape)


# ---------------------------------------------------------------------------
# Custom decoder layer
# ---------------------------------------------------------------------------
class Gemma4MoEDecoderLayer(nn.Module):
    """Gemma4 decoder layer with NeMo MoE backend.

    Reuses HF attention and dense MLP, replaces HF Router+MoEBlock with
    NeMo Gemma4MoE (Gemma4Gate + GroupedExperts).
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        # Reuse HF modules
        self.self_attn = Gemma4Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4MLP(config, layer_idx)

        # Norms
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # NeMo MoE
        self.moe = Gemma4MoE(moe_config, backend, config)

        # layer_scalar: per-layer output scaling. We register a buffer on every layer so DCP
        # can always load the weight when present.
        # It is present only for sliding window layers. Regular attentionlayers without a
        # checkpoint value for the layer_scalar keep ones (identity scaling).
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        padding_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # --- Attention ---
        residual = x
        x = self.input_layernorm(x)
        x, _ = self.self_attn(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        x = self.post_attention_layernorm(x)
        x = residual + x

        # --- Dense MLP + MoE in parallel ---
        residual = x

        dense_out = self.pre_feedforward_layernorm(x)
        dense_out = self.mlp(dense_out)
        dense_out = self.post_feedforward_layernorm_1(dense_out)

        moe_input = self.pre_feedforward_layernorm_2(x)
        moe_out = self.moe(moe_input, padding_mask, gate_input=x)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]
        moe_out = self.post_feedforward_layernorm_2(moe_out)

        x = dense_out + moe_out
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        # Apply per-layer output scaling, multiplied by 1 if it is not present in the checkpoint,
        # otherwise uses the scalar value from the checkpoint.
        x = x * self.layer_scalar

        return x


# ---------------------------------------------------------------------------
# Text model backend
# ---------------------------------------------------------------------------
class Gemma4MoETextModelBackend(nn.Module):
    """Gemma4 text decoder rebuilt with NeMo MoE blocks."""

    def __init__(
        self,
        config: Gemma4TextConfig,
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

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=getattr(config, "moe_intermediate_size", None)
            or getattr(config, "expert_intermediate_size", None),
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
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )

        self.layers = nn.ModuleDict(
            {
                str(layer_id): Gemma4MoEDecoderLayer(config, layer_id, self.moe_config, backend)
                for layer_id in range(config.num_hidden_layers)
            }
        )

        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if past_key_values is not None or use_cache:
            raise NotImplementedError("KV cache not supported for the Gemma4 MoE backend.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if padding_mask is None and attention_mask is not None:
            padding_mask = attention_mask.bool().logical_not()

        hidden_states = inputs_embeds

        # Build causal masks. When use_bidirectional_attention == "vision" (e.g.
        # gemma-4-26B-A4B, gemma-4-31B), HF uses create_causal_mask_mapping to
        # build a vision-aware mask where tokens inside the same vision group
        # attend to each other bidirectionally (not just causally). Missing this
        # logic causes gen_kl_error to be ~10x higher on multimodal inputs.
        if getattr(self.config, "use_bidirectional_attention", None) == "vision":
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping

            causal_mask_mapping = create_causal_mask_mapping(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=pixel_values,
                is_training=self.training,
            )
        else:
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for decoder_layer in self.layers.values():
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                padding_mask=padding_mask,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value


# ---------------------------------------------------------------------------
# Wrapper that exposes language_model properties for parallelizer
# ---------------------------------------------------------------------------
class Gemma4MoEModel(HFGemma4Model):
    """Thin wrapper that exposes ``language_model`` internals as properties
    expected by the NeMo training loop."""

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    @property
    def norm(self):
        return self.language_model.norm


# ---------------------------------------------------------------------------
# Top-level conditional-generation model
# ---------------------------------------------------------------------------
class Gemma4ForConditionalGeneration(HFCheckpointingMixin, HFGemma4ForConditionalGeneration, MoEFSDPSyncMixin):
    supports_gradient_checkpointing = True
    """Gemma4 VL conditional generation model with NeMo MoE backend.

    When the checkpoint has ``enable_moe_block=True`` in its text config,
    replaces the HF-native language model with ``Gemma4MoETextModelBackend``
    (NeMo GroupedExperts + Gemma4Gate).  Otherwise falls through to vanilla HF.
    """

    @classmethod
    def from_config(
        cls,
        config: Gemma4Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        if not _GEMMA4_HF_AVAILABLE:
            raise UnavailableError("transformers.models.gemma4 is not available.")
        config = Gemma4Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Gemma4Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        text_config: dict | None = None,
        *,
        mtp_num_layers: int = 0,
        mtp_layer_pattern: str = "*",
        mtp_loss_scaling_factor: float = 0.1,
        **kwargs,
    ):
        if not _GEMMA4_HF_AVAILABLE:
            raise UnavailableError("transformers.models.gemma4 is not available.")
        backend = backend or BackendConfig()

        # Merge text_config overrides (e.g. from YAML) into the proper config
        # object before HF __init__ which needs a real PretrainedConfig.
        if text_config is not None and isinstance(text_config, dict):
            cfg_text = config.text_config if hasattr(config, "text_config") else config
            for k, v in text_config.items():
                setattr(cfg_text, k, v)

        # Compat: older checkpoints used expert_intermediate_size, v5.5+ uses moe_intermediate_size.
        cfg_text = config.text_config if hasattr(config, "text_config") else config
        if not getattr(cfg_text, "moe_intermediate_size", None) and getattr(cfg_text, "expert_intermediate_size", None):
            cfg_text.moe_intermediate_size = cfg_text.expert_intermediate_size
        if not getattr(cfg_text, "expert_intermediate_size", None) and getattr(cfg_text, "moe_intermediate_size", None):
            cfg_text.expert_intermediate_size = cfg_text.moe_intermediate_size

        # Initialize the HF parent (creates self.model, self.lm_head, vision tower, etc.)
        super().__init__(config)

        self.backend = backend

        text_config = config.text_config if hasattr(config, "text_config") else config
        enable_moe = getattr(text_config, "enable_moe_block", False)

        # --- Optional MTP head (works for both dense and MoE Gemma4 variants) ---
        # MTP is opt-in via constructor kwargs (HF Gemma4 configs have no MTP fields).
        # The block is created here so checkpoint reload sees consistent state.
        self.mtp_config = build_mtp_config_from_kwargs(
            mtp_num_layers=mtp_num_layers,
            mtp_layer_pattern=mtp_layer_pattern,
            mtp_loss_scaling_factor=mtp_loss_scaling_factor,
        )
        if self.mtp_config.enabled:
            mtp_dtype = get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16)
            self.mtp = build_gemma4_mtp(text_config, self.mtp_config, dtype=mtp_dtype)
        else:
            self.mtp = None

        if not enable_moe:
            # Dense Gemma4 — keep vanilla HF model.
            return

        # --- MoE path: replace the text model ---
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model.__class__ = Gemma4MoEModel
        self.model.language_model = Gemma4MoETextModelBackend(
            text_config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )

        # Expose moe_config for the MoE parallelizer assertion
        self.model.moe_config = self.model.language_model.moe_config

        self.vocab_size = text_config.vocab_size
        pad_token_id = getattr(text_config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else -1

        # State dict adapter for HF ↔ NeMo weight conversion
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Gemma4MoEStateDictAdapter(
                text_config,
                self.model.language_model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16),
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if cache_position is None and input_ids is not None:
            seq_len = input_ids.shape[-1]
            cache_position = torch.arange(seq_len, device=input_ids.device)

        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        mtp_active = self.mtp is not None and self.training
        if not getattr(text_config, "enable_moe_block", False):
            # Dense path — delegate to HF forward.  When MTP is active we
            # additionally request hidden states so we can read the post-final-norm
            # last hidden state to feed into the MTP head.
            if mtp_active:
                kwargs.pop("output_hidden_states", None)
                hf_out = super().forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    pixel_values=pixel_values,
                    image_position_ids=image_position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    output_hidden_states=True,
                    **kwargs,
                )
                if hf_out.hidden_states is None:
                    raise RuntimeError(
                        "Gemma4 dense forward did not return hidden states; MTP requires output_hidden_states=True."
                    )
                hidden_states = hf_out.hidden_states[-1]
                mtp_per_depth_h = self.mtp(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    embed_fn=self.model.language_model.embed_tokens,
                    position_ids=position_ids,
                    attention_mask=None,
                )
                return Gemma4WithMTPCausalLMOutput(
                    logits=hf_out.logits,
                    hidden_states=hf_out.hidden_states,
                    mtp_per_depth_h=mtp_per_depth_h,
                    mtp_loss_scaling_factor=self.mtp_config.loss_scaling_factor,
                )

            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )

        # --- MoE forward path ---
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Handle vision tokens
        if pixel_values is not None:
            image_features = self.model.get_image_features(
                pixel_values, image_position_ids=image_position_ids, return_dict=True
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            if mm_token_type_ids is not None:
                special_image_mask = mm_token_type_ids == 1
            elif input_ids is not None:
                special_image_mask = input_ids == self.config.image_token_id
            else:
                special_image_mask = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

            image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        outputs = self.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            padding_mask=padding_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            logits = hidden_states

        if (final_logit_softcapping := getattr(text_config, "final_logit_softcapping", None)) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        if mtp_active:
            mtp_per_depth_h = self.mtp(
                input_ids=input_ids,
                hidden_states=hidden_states,
                embed_fn=self.model.language_model.embed_tokens,
                position_ids=position_ids,
                attention_mask=None,
            )
            return Gemma4WithMTPCausalLMOutput(
                logits=logits,
                hidden_states=None,
                mtp_per_depth_h=mtp_per_depth_h,
                mtp_loss_scaling_factor=self.mtp_config.loss_scaling_factor,
            )

        return logits

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        # MTP head: random init the auxiliary layers so they can be trained
        # on top of a pretrained backbone (Gemma4 ships without MTP weights).
        if getattr(self, "mtp", None) is not None:
            mtp_buffer_device = buffer_device or (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            for sublayer in self.mtp.layers:
                sublayer.init_weights(buffer_device=mtp_buffer_device)

        if not getattr(text_config, "enable_moe_block", False):
            self.to(dtype)
            return

        # Guard: HF's super().__init__() calls post_init() -> init_weights() ->
        # initialize_weights() *before* __init__ replaces the language model
        # with Gemma4MoETextModelBackend (which uses ModuleDict).  At that
        # point layers is still HF's ModuleList which leads to an AttributeError, just cast and return.
        # Needed only when constructing the model directly, doesn't affect when loading a ckpt via from_pretrained().
        language_model = self.model.language_model
        if not isinstance(language_model, Gemma4MoETextModelBackend):
            self.to(dtype)
            return

        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            for layer in language_model.layers.values():
                layer.moe.init_weights(buffer_device)

        self.to(dtype)


if _GEMMA4_HF_AVAILABLE:
    ModelClass = Gemma4ForConditionalGeneration

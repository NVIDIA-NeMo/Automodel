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

"""HyMT2ForCausalLM — Tencent Hy-MT2-30B-A3B (translation MoE) SFT support.

Architecture (from tencent/Hy-MT2-30B-A3B config.json):
  - 48 transformer layers; layer 0 is dense, layers 1-47 are MoE
  - MoE: 128 routed experts + 1 shared expert, top-8 activated
  - Sigmoid routing with expert-bias correction, router_scaling_factor=2.826
  - route_norm = True (normalize top-k weights to sum to 1)
  - GQA: 32 Q heads, 4 KV heads, head_dim=128, hidden_size=2048
  - Per-head Q/K RMSNorm (qk_norm=True) before RoPE
  - 256K context, rope_theta=11158840
  - dense intermediate_size=6912, moe_intermediate_size=expert_hidden_dim=768
  - vocab_size=120832
  - enable_lm_head_fp32 = True (HF reference upcasts lm_head to fp32)

Notes vs. ``components/models/hy_v3`` (Hy3-preview 295B):
  - Smaller everywhere (48L / 128 experts / 32+4 heads / hidden=2048).
  - Adds an in-model ``enable_lm_head_fp32`` fallback (applies when the
    YAML's ``lm_head_precision`` is not set). The preferred path is to set
    ``distributed.moe.lm_head_precision: float32`` in the YAML, which the
    MoE parallelizer handles via ``MixedPrecisionPolicy``.
  - ``score_func`` is driven by ``config.moe_router_use_sigmoid`` instead
    of being hard-coded.
"""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.hy_mt2.layers import HyMT2Attention
from nemo_automodel.components.models.hy_mt2.state_dict_adapter import HyMT2StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _resolve_score_func(config: Any) -> str:
    """Map ``config.moe_router_use_sigmoid`` to a gate ``score_func`` name.

    Returns "sigmoid" when the flag is True (Hy-MT2 default) and "softmax"
    otherwise. The bias-aware variants ("sigmoid_with_bias" /
    "softmax_with_bias") are selected at the gate level by the presence of
    ``e_score_correction_bias`` plus expert-group routing, which Hy-MT2 does
    not use (n_expert_groups=0).
    """
    use_sigmoid = bool(getattr(config, "moe_router_use_sigmoid", True))
    return "sigmoid" if use_sigmoid else "softmax"


class Block(nn.Module):
    """Single Hy-MT2 transformer block: attention + (dense MLP | MoE) + residual norms."""

    def __init__(self, layer_idx: int, config: Any, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.self_attn = HyMT2Attention(config, backend)

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


class HyMT2Model(nn.Module):
    """Hy-MT2 backbone: token embeddings + transformer blocks + final RMSNorm.

    The MoE / dense split is governed by ``config.first_k_dense_replace``
    (layer 0 dense, the rest MoE for the published Hy-MT2-30B-A3B). The
    MoE configuration is assembled from the HF config fields and forwarded
    to every MoE-bearing ``Block``.
    """

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

        # ``expert_hidden_dim`` and ``moe_intermediate_size`` are synonyms in
        # the on-disk config. Prefer ``expert_hidden_dim`` when present
        # (matches the field name used by the HF reference for the expert MLP
        # hidden dim); fall back to ``moe_intermediate_size`` otherwise.
        moe_inter = getattr(config, "expert_hidden_dim", None)
        if moe_inter is None:
            moe_inter = config.moe_intermediate_size

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=moe_inter,
            n_routed_experts=config.num_experts,
            n_shared_experts=getattr(config, "num_shared_experts", 0),
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func=_resolve_score_func(config),
            route_scale=getattr(config, "router_scaling_factor", 1.0),
            aux_loss_coeff=0.0,
            norm_topk_prob=getattr(config, "route_norm", True),
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
            # Ensures e_score_correction_bias buffer is created so HF
            # checkpoints with ``expert_bias`` load cleanly.
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
            device=torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu"),
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
        return h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class HyMT2ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Hy-MT2-30B-A3B causal-LM wrapper.

    Mixes in ``MoEFSDPSyncMixin`` so EP / FSDP2 expert-gradient sync works
    out of the box (set ``distributed.ep_size`` in the YAML; must divide
    ``num_experts``=128). The ``HFCheckpointingMixin`` provides
    ``from_pretrained`` / ``save_pretrained`` over the HF safetensors layout.
    """

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

        # The on-disk Hy-MT2 checkpoint declares ``model_type: hy_v3`` so
        # ``AutoConfig`` returns a ``HYV3Config`` instance. Our model code
        # is duck-typed against the field names (which match) so this works
        # transparently.
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=False)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = HyMT2Model(config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        # In-model fp32 fallback for the lm_head matmul. The preferred wiring
        # is the YAML ``distributed.moe.lm_head_precision: float32``, which
        # the MoE parallelizer enables via ``MixedPrecisionPolicy``. When that
        # path is not used, ``enable_lm_head_fp32`` in the model config still
        # triggers the in-forward upcast.
        self._enable_lm_head_fp32 = bool(getattr(config, "enable_lm_head_fp32", False))
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = HyMT2StateDictAdapter(
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
        output_hidden_states: Optional[bool] = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Forward pass returning :class:`CausalLMOutputWithPast`.

        Args:
            input_ids: Token IDs. BSHD: ``[B, S]``; THD: ``[1, T]`` (squeezed internally).
            position_ids: Optional position indices.
            attention_mask: Optional 2D padding mask.
            padding_mask: Optional padding mask used by the THD squeeze helper.
            logits_to_keep: If 0 (default) compute logits for all positions; if > 0
                only compute logits for the last ``logits_to_keep`` positions
                (avoids materialising the full logit matrix during generation /
                enables fused-linear cross-entropy in training).
            output_hidden_states: Whether to return the final hidden states (the
                input to ``lm_head``) on the output. Defaults to the config flag.
            **attn_kwargs: Additional attention kwargs forwarded to the base model
                (e.g. qkv_format, cu_seqlens, CP kwargs).

        Returns:
            :class:`~transformers.modeling_outputs.CausalLMOutputWithPast` with
            ``logits`` and, when ``output_hidden_states`` is set, the final
            ``hidden_states``.
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        is_thd = "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd"
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

        # Optionally restrict logit computation to the last few positions.
        # When logits_to_keep == 0 we compute all positions (training default);
        # avoid slicing in that case because DTensor cannot slice a full range.
        # Handle both 2D [T, H] (THD/packed) and 3D [B, S, H] hidden states.
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            lm_head_input = hidden
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            if hidden.dim() == 2:
                lm_head_input = hidden[slice_indices, :]
            else:
                lm_head_input = hidden[:, slice_indices, :]

        if self.lm_head is None:
            logits = lm_head_input
        elif (
            self._enable_lm_head_fp32
            and self.lm_head.weight.dtype == torch.float32
            and lm_head_input.dtype != torch.float32
        ):
            # The MoE parallelizer (``distributed.moe.lm_head_precision:
            # float32`` in the YAML) has already promoted ``lm_head.weight`` to
            # fp32. Feed it fp32 input via ``nn.Linear`` -- which is
            # DTensor-aware under FSDP2 -- and cast logits back to the input
            # dtype. We must NOT use ``F.linear`` directly with a manually
            # ``.float()``-ed weight here, because that bypasses nn.Linear's
            # DTensor redistribution and crashes with
            # "got mixed torch.Tensor and DTensor".
            original_dtype = lm_head_input.dtype
            logits = self.lm_head(lm_head_input.float()).to(original_dtype)
        else:
            logits = self.lm_head(lm_head_input)

        if is_thd:
            logits = logits.unsqueeze(0)

        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden if output_hidden_states else None,
        )

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.model.layers.values():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        if buffer_device is None:
            buffer_device = (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if torch.cuda.is_available()
                else torch.device("cpu")
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
            self.model.rotary_emb.device = buffer_device


ModelClass = HyMT2ForCausalLM

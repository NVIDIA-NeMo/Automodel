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

"""BailingMoeV2 model (Ling 2.0 family).

Architecture summary (from the public ``inclusionAI/Ling-{mini,flash,1T}-2.0``
checkpoints):

- GQA attention with per-head QK-RMSNorm and partial RoPE
  (rotates the first ``head_dim * partial_rotary_factor`` channels only).
- ``first_k_dense_replace`` dense MLP layers at the start of the stack;
  the remaining layers are sigmoid-routed grouped MoE with shared experts
  and an aux-loss-free per-expert bias (DeepSeek-V3-style routing).
- Single shared expert with intermediate size ``moe_intermediate_size``.
- MTP heads (``num_nextn_predict_layers``) are disabled in all published
  checkpoints and intentionally not modeled here.

Example (YAML):

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: inclusionAI/Ling-mini-2.0
```
"""

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
from nemo_automodel.components.models.ling_v2.layers import BailingMoeV2Attention
from nemo_automodel.components.models.ling_v2.state_dict_adapter import BailingMoeV2StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Block(nn.Module):
    """Single transformer block: attention + (dense MLP or MoE) + residuals."""

    def __init__(
        self,
        layer_idx: int,
        config: BailingMoeV2Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = BailingMoeV2Attention(config, backend)

        if layer_idx < config.first_k_dense_replace:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear)
        else:
            self.mlp = MoE(moe_config, backend)

        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps
        )

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

    def init_weights(self, buffer_device: torch.device) -> None:
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class BailingMoeV2Model(nn.Module):
    """Embedding + decoder stack + final norm.  No LM head."""

    def __init__(
        self,
        config: BailingMoeV2Config,
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

        # MoE wiring: DeepSeek-V3-style sigmoid + grouped topk + per-expert bias
        # + shared expert.  The framework's ``Gate`` (score_func='sigmoid', n_groups>1,
        # force_e_score_correction_bias=True) is bit-equivalent to BailingMoeV2Gate.
        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.num_experts,
            n_shared_experts=config.num_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            # Aux-loss-free routing: bias buffer is loaded from the HF checkpoint
            # but not updated by SFT (set to >0 to enable DeepSeek-V3-style bias
            # auto-tuning during pretraining).
            gate_bias_update_factor=0.0,
            force_e_score_correction_bias=bool(config.moe_router_enable_expert_bias),
            score_func=config.score_function,
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0.0,
            norm_topk_prob=config.norm_topk_prob,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            shared_expert_inter_dim=config.moe_intermediate_size,
            shared_expert_activation="swiglu",
            softmax_before_topk=False,
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
        self.head_dim = config.head_dim
        rope_scaling = getattr(config, "rope_scaling", None) or {}

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=int(config.rope_theta),
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", config.max_position_embeddings),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            partial_rotary_factor=float(getattr(config, "partial_rotary_factor", 1.0)),
            device=torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else None,
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
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            position_ids = (
                torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.shape[0], -1)
            )

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
                **attn_kwargs,
            )

        h = self.norm(h) if self.norm else h
        return h

    def update_moe_gate_bias(self) -> None:
        """No-op for SFT; published Ling checkpoints freeze the expert_bias buffer."""
        with torch.no_grad():
            for _, block in self.layers.named_children():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()

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


class BailingMoeV2ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Causal-LM head wrapping ``BailingMoeV2Model``."""

    # ``e_score_correction_bias`` must stay in fp32 even when the rest of the
    # model is bf16; tiny quantization errors in the bias change routing.
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    # PP compatibility: our forward computes ``freqs_cis`` inline and threads it
    # through the decoder blocks (gpt_oss-style rotary convention).  The generic
    # ``patch_hf_model_for_pp`` would replace our forward with an HF-style one
    # that calls ``self.model.rotary_emb(hidden_states, position_ids)`` expecting
    # a ``(cos, sin)`` return — that signature mismatches our
    # ``RotaryEmbedding.forward(query, key)`` and crashes inside
    # ``apply_rotary_emb`` with a tensor-shape mismatch at ``torch.cat``.
    # Setting this flag instructs the PP split to leave our forwards intact.
    _pp_keep_self_forward: bool = True

    @classmethod
    def from_config(
        cls,
        config: BailingMoeV2Config,
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
        config = BailingMoeV2Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: BailingMoeV2Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = BailingMoeV2Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = BailingMoeV2StateDictAdapter(
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
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
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
        logits = self.lm_head(hidden) if self.lm_head else hidden
        if "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd":
            logits = logits.unsqueeze(0)
        return logits

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for _, block in self.model.layers.named_children():
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


ModelClass = BailingMoeV2ForCausalLM

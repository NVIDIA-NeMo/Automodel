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

"""DFlash draft model with a dense Kimi K3 MLA backbone, plus its build helpers.

The Qwen3 draft (``draft_qwen3.py``) documents the DFlash contract: the draft
predicts a whole ``block_size`` block in one non-causal forward whose keys and
values are ``[target-hidden context | noise block]``, with the block structure
supplied entirely by the attention mask built in
``nemo_automodel.components.speculative.dflash.core``.

This module keeps that contract and swaps the backbone for Kimi K3's:

* **MLA with Q-LoRA and a compressed KV latent.** The attention subclasses the
  target's own :class:`KimiMLAAttention` so the draft's projection layout stays
  in lockstep with the layers whose hidden states it consumes; only the forward
  is replaced, because the draft's keys and values span the context as well as
  the queried noise block.
* **NoPE.** K3 requires ``mla_use_nope=True``: no rotary is applied anywhere in
  its full-attention layers (the ``qk_rope_head_dim`` slice is a plain
  head-shared extension of the key). The draft therefore has no rotary
  embedding and ignores ``position_ids``.
* **SiTU feed-forward and fp32-variance RMSNorm**, reusing the target's
  ``KimiK3MLP`` / ``KimiRMSNorm`` modules.

The draft is always dense: K3's KDA linear-attention layers, routed and shared
experts, MTP heads, and the learned attention-residual mixer live in the target
only. Because the draft has no FlexAttention path, it consumes the dense
additive DFlash mask (``attention_backend='sdpa'``).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.model import KimiK3MLP, KimiMLAAttention, KimiRMSNorm
from nemo_automodel.components.speculative.dflash.draft_qwen3 import build_target_layer_ids


class KimiK3DFlashAttention(KimiMLAAttention):
    """Non-causal K3 MLA whose keys/values are ``[context | noise-block]``.

    Inherits the target's MLA projections unchanged and replaces only the
    forward. Queries come from the draft (noise) tokens only. K3's MLA emits one
    K/V per query head (``kv_b_proj`` produces ``num_attention_heads`` slices of
    ``qk_nope_head_dim + v_head_dim``), so there is no GQA group repeat and the
    target's ``_expand_key_value_groups`` is not needed here.
    """

    def __init__(self, config: KimiK3TextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__(config, layer_idx, backend)
        if self.num_key_value_heads != self.num_heads:
            raise ValueError(
                "Kimi K3 DFlash MLA expects one K/V head per query head, got "
                f"num_key_value_heads={self.num_key_value_heads} vs num_attention_heads={self.num_heads}."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Attend the draft block over ``[context | noise]``.

        Unlike the target's eager forward, this runs
        ``F.scaled_dot_product_attention``: a DFlash block batch makes both the
        query and the key axis sequence-scale (``num_anchors * block_size`` by
        ``sequence + num_anchors * block_size``), so materializing the score
        matrix is not affordable. The dense additive mask is passed through
        unchanged, which keeps the visibility semantics identical.

        Args:
            hidden_states: Tensor of shape ``[batch, blocks * block_size, hidden]``;
                the draft (noise) tokens, which are the queries.
            target_hidden: Tensor of shape ``[batch, sequence, hidden]``; the
                projected target-hidden context prepended to the keys and values.
            attention_mask: Additive mask of shape
                ``[batch, 1, blocks * block_size, sequence + blocks * block_size]``,
                or None.
            **kwargs: Ignored; accepted so the decoder layer can forward extras.

        Returns:
            Tensor of shape ``[batch, blocks * block_size, hidden]``.
        """
        del kwargs
        batch_size, query_length = hidden_states.shape[:-1]
        key_value_length = target_hidden.shape[1] + query_length

        if self.q_lora_rank is not None:
            query_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            query_states = self.q_proj(hidden_states)
        query_states = query_states.view(batch_size, query_length, self.num_heads, self.q_head_dim).transpose(1, 2)

        # ``kv_a_proj_with_mqa`` is a bias-free Linear, so projecting the context
        # and the noise block separately and concatenating the (much narrower)
        # latents is exact, and avoids copying the two at hidden width.
        compressed_kv = torch.cat(
            [self.kv_a_proj_with_mqa(target_hidden), self.kv_a_proj_with_mqa(hidden_states)],
            dim=1,
        )
        key_states, key_nope_states = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        key_states = self.kv_b_proj(self.kv_a_layernorm(key_states))
        key_states = key_states.view(
            batch_size,
            key_value_length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        key_states, value_states = torch.split(key_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # NoPE: this slice carries no rotary, it is one head-shared key extension
        # broadcast over the query heads.
        key_nope_states = key_nope_states.view(batch_size, 1, key_value_length, self.qk_rope_head_dim)
        key_states = torch.cat([key_states, key_nope_states.expand(*key_states.shape[:-1], -1)], dim=-1)

        attention_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, query_length, -1)
        if self.use_output_gate:
            attention_output = attention_output * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(attention_output)


class KimiK3DFlashDecoderLayer(nn.Module):
    """Pre-norm K3 MLA block over ``[context | noise]`` followed by a dense SiTU MLP."""

    def __init__(self, config: KimiK3TextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.self_attn = KimiK3DFlashAttention(config, layer_idx, backend)
        self.mlp = KimiK3MLP(config)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one draft layer.

        Args:
            hidden_states: Tensor of shape ``[batch, blocks * block_size, hidden]``.
            target_hidden: Tensor of shape ``[batch, sequence, hidden]``.
            attention_mask: Additive mask of shape
                ``[batch, 1, blocks * block_size, sequence + blocks * block_size]``,
                or None.

        Returns:
            Tensor of shape ``[batch, blocks * block_size, hidden]``.
        """
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden,
            attention_mask,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class KimiK3DFlashDraftModel(nn.Module):
    """DFlash draft model: a small dense non-causal K3 MLA stack over ``[context | noise]``.

    The draft owns no embedding table and no LM head: the DFlash trainer embeds
    the ``[anchor, MASK, ...]`` blocks with the frozen target's ``embed_tokens``
    and decodes this model's output with the frozen target's ``lm_head``.
    """

    _no_split_modules = ["KimiK3DFlashDecoderLayer"]

    def __init__(self, config: KimiK3TextConfig) -> None:
        super().__init__()
        self.config = config
        dflash_config = getattr(config, "dflash_config", {}) or {}
        if dflash_config.get("projector_type", None) is not None:
            # The Domino correction head is implemented on the Qwen3 draft only;
            # accepting the flag here would silently train a plain DFlash draft.
            raise ValueError(
                "The Kimi K3 DFlash draft does not implement a draft projector "
                f"(got projector_type={dflash_config['projector_type']!r})."
            )
        self.target_layer_ids = list(
            dflash_config.get(
                "target_layer_ids",
                build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
            )
        )
        # The draft replaces the MLA forward with its own SDPA path over the
        # dense DFlash mask, so the only thing the parent's backend selects here
        # is which extra attention modules it builds at init: "eager" builds
        # none, which is what this subclass needs.
        backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch", rope_fusion=False)
        self.layers = nn.ModuleList(
            [KimiK3DFlashDecoderLayer(config, layer_idx, backend) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = KimiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Output transform some targets apply around the frozen lm_head; identity
        # unless the target's ``dflash_config`` sets these (see compute_logits).
        multiplier = float(dflash_config.get("output_multiplier", 1.0))
        if multiplier <= 0:
            raise ValueError(f"output_multiplier must be > 0 when set, got {multiplier}.")
        self.output_multiplier = multiplier
        softcap = dflash_config.get("final_logit_softcapping", None)
        if softcap is not None and float(softcap) <= 0:
            raise ValueError(f"final_logit_softcapping must be > 0 when set, got {softcap}.")
        self.final_logit_softcapping = None if softcap is None else float(softcap)

    def forward(
        self,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        noise_embedding: torch.Tensor | None = None,
        target_hidden: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the draft blocks' hidden states.

        Args:
            position_ids: Unused. K3's MLA is NoPE, so the draft has no rotary
                embedding; the argument is accepted to keep the trainer's call
                signature identical across DFlash drafts.
            attention_mask: Additive DFlash mask of shape
                ``[batch, 1, blocks * block_size, sequence + blocks * block_size]``.
            noise_embedding: Tensor of shape ``[batch, blocks * block_size, hidden]``.
            target_hidden: Tensor of shape
                ``[batch, sequence, len(target_layer_ids) * hidden]``.
            **kwargs: Ignored.

        Returns:
            Tensor of shape ``[batch, blocks * block_size, hidden]``.
        """
        del position_ids, kwargs
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden: torch.Tensor, output_head: nn.Module) -> torch.Tensor:
        """Project draft hidden states to logits, applying the target's output transform.

        See ``Qwen3DFlashDraftModel.compute_logits`` -- same contract, identity
        unless the target's ``dflash_config`` sets ``output_multiplier`` /
        ``final_logit_softcapping``.

        Args:
            hidden: Tensor of shape [..., hidden]; draft hidden states with
                arbitrary leading dimensions.
            output_head: The frozen target's output projection.

        Returns:
            Tensor of shape [..., vocab].
        """
        logits = output_head(hidden)
        if self.output_multiplier != 1.0:
            logits = logits * self.output_multiplier
        if self.final_logit_softcapping is not None:
            logits = torch.tanh(logits / self.final_logit_softcapping) * self.final_logit_softcapping
        return logits


def build_kimi_k3_dflash_draft_config(
    target_config,
    *,
    num_draft_layers: int,
    num_target_layers: int,
    block_size: int,
    dflash_config: dict,
    attention_backend: str,
) -> KimiK3TextConfig:
    """Build a dense MLA DFlash draft config from a Kimi K3 target's text config.

    The draft consumes the target's frozen ``embed_tokens`` / ``lm_head`` and
    fuses its hidden states, so it keeps the target's MLA dims, hidden size, and
    vocabulary and only shrinks the depth. Everything the draft does not build is
    switched off explicitly rather than left at the target's value, so the
    serialized draft config describes the draft and not the target: KDA linear
    attention, routed and shared experts, MTP layers, and the learned
    attention-residual mixer all stay in the target.

    Args:
        target_config: The Kimi K3 target's text config (``kimi_linear``).
        num_draft_layers: Number of draft decoder layers.
        num_target_layers: Depth of the target's text backbone; recorded so a
            reloaded draft config still describes which target it was trained on.
        block_size: DFlash block size.
        dflash_config: The recipe's DFlash block, carrying ``mask_token_id`` and
            ``target_layer_ids`` (the target layers whose hidden states the draft
            consumes, which set the ``fc`` input width).
        attention_backend: The draft's attention implementation. Recorded on the
            config for the serving runtime; the draft itself always attends over
            the dense additive mask.

    Returns:
        A ``KimiK3TextConfig`` describing the draft.

    Raises:
        ValueError: If ``target_config`` is not a Kimi K3 text config.
    """
    if target_config.model_type != KimiK3TextConfig.model_type:
        raise ValueError(
            f"Kimi K3 DFlash expects a {KimiK3TextConfig.model_type!r} text config, got {target_config.model_type!r}."
        )
    draft_config = copy.deepcopy(target_config)

    draft_config.architectures = ["KimiK3DFlashDraftModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.num_experts = None
    draft_config.num_shared_experts = 0
    draft_config.linear_attn_config = {"kda_layers": [], "full_attn_layers": list(range(1, num_draft_layers + 1))}
    draft_config.num_nextn_predict_layers = 0
    draft_config.attn_res_block_size = None
    draft_config.tie_word_embeddings = False
    draft_config.block_size = block_size
    draft_config.dflash_config = dict(dflash_config)
    draft_config._attn_implementation = attention_backend
    return draft_config


def build_kimi_k3_dflash_target_kwargs(recipe_cfg) -> dict:
    """Extra ``from_pretrained`` kwargs for a frozen Kimi K3 DFlash target.

    Two things a Qwen3-shaped target does not need:

    * ``config`` pins the architecture to the text-only ``KimiK3ForCausalLM``. A K3
      checkpoint declares the multimodal ``KimiK3ForConditionalGeneration``, which
      would additionally build the vision tower that DFlash never reads.
    * ``backend`` selects the expert-parallel token dispatcher and the HF
      state-dict adapter (which also dequantizes an FP8 base checkpoint on load),
      mirroring the frozen large-MoE target backends the DSpark recipe builds.
      ``experts`` defaults to ``torch_mm`` rather than ``gmm`` because the latter
      needs the optional ``grouped_gemm`` package. ``attn`` is left at ``eager``
      and is inert -- ``KimiK3ForCausalLM`` never reads ``backend.attn``, since
      its MLA and KDA layers each have a fixed attention path -- and
      ``gate_precision`` is left unset because K3 already defaults it to fp32.

    Args:
        recipe_cfg: The recipe's ``recipe_args`` mapping.

    Returns:
        Keyword arguments to merge into the target's ``from_pretrained`` call.
    """
    return {
        "config": {"architectures": ["KimiK3ForCausalLM"]},
        "backend": BackendConfig(
            attn="eager",
            linear="torch",
            rms_norm="torch_fp32",
            rope_fusion=False,
            dispatcher=str(recipe_cfg.get("target_dispatcher", "hybridep")),
            experts=str(recipe_cfg.get("target_experts", "torch_mm")),
            enable_hf_state_dict_adapter=True,
            enable_fsdp_optimizations=bool(recipe_cfg.get("target_enable_fsdp_optimizations", True)),
        ),
    }


__all__ = [
    "KimiK3DFlashDraftModel",
    "build_kimi_k3_dflash_draft_config",
    "build_kimi_k3_dflash_target_kwargs",
]

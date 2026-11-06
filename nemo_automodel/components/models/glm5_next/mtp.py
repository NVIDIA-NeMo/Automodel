# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash multi-token prediction (MTP) head.

The released checkpoint stores one MTP layer after the decoder stack, as
``model.language_model.layers.{num_hidden_layers}``. It is a plain pre-norm
residual block (no mHC) with the same NoPE MLA + KPool indexer attention and
routed MoE as the sparse decoder layers:

    fused = eh_proj(cat(enorm(next_token_embedding), hnorm(h_prev)))
    hidden = fused + self_attn(input_layernorm(fused))
    output = final_layernorm(hidden + mlp(post_attention_layernorm(hidden)))

``h_prev`` is the backbone output after its final norm, and the shared
``lm_head`` projects ``output`` for the auxiliary loss. ``final_layernorm``
is stored as ``shared_head.norm`` in the checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.mtp import MTPConfig, MTPModule
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.cp import Glm5NextPackedContext
from nemo_automodel.components.models.glm5_next.layers import Glm5NextRMSNorm, Glm5NextSparseAttention
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE


class Glm5NextMTPSublayer(nn.Module):
    """One GLM-5.3-Flash MTP depth: embedding fusion, sparse MLA, MoE and final norm.

    Args:
        config: GLM-5.3 text config.
        layer_idx: Checkpoint layer index of this depth (``num_hidden_layers + depth``).
        moe_config: MoE config shared with the decoder's sparse layers.
        backend: Kernel backend selection.
        dtype: Parameter dtype of the fusion projection and norms.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size
        eps = config.rms_norm_eps
        self.enorm = Glm5NextRMSNorm(hidden_size, eps, dtype)
        self.hnorm = Glm5NextRMSNorm(hidden_size, eps, dtype)
        self.eh_proj = initialize_linear_module(backend.linear, 2 * hidden_size, hidden_size, bias=False, dtype=dtype)
        self.input_layernorm = Glm5NextRMSNorm(hidden_size, eps, dtype)
        self.self_attn = Glm5NextSparseAttention(config, layer_idx, backend)
        self.post_attention_layernorm = Glm5NextRMSNorm(hidden_size, eps, dtype)
        self.mlp = MoE(moe_config, backend)
        self.final_layernorm = Glm5NextRMSNorm(hidden_size, eps, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor,
        packed_context: Glm5NextPackedContext,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict one token further ahead.

        Args:
            hidden_states: Previous depth's output (the backbone's final-norm
                output for depth 1) with shape ``[batch, local_sequence, hidden]``.
            embed_input: Embedding of the token ``depth`` positions ahead, in the
                same local layout, zero where that token leaves the document.
            packed_context: Document map shared with the backbone attention.
            padding_mask: Optional ``[batch, local_sequence]`` mask, ``True`` on padding.

        Returns:
            Normalized hidden states ``[batch, local_sequence, hidden]`` for the
            shared ``lm_head`` and the next depth.
        """
        fused = self.eh_proj(torch.cat((self.enorm(embed_input), self.hnorm(hidden_states)), dim=-1))
        hidden = fused + self.self_attn(
            self.input_layernorm(fused),
            packed_context=packed_context,
            padding_mask=padding_mask,
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden), padding_mask)
        return self.final_layernorm(hidden)

    def update_moe_gate_bias(self) -> None:
        """Update this depth's no-aux-loss routing correction bias."""
        moe = unwrap_checkpoint_wrapper(self.mlp)
        if isinstance(moe, MoE) and moe.gate.bias_update_factor > 0:
            moe.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize a checkpoint-free MTP depth on ``buffer_device``."""
        for norm in (
            self.enorm,
            self.hnorm,
            self.input_layernorm,
            self.post_attention_layernorm,
            self.final_layernorm,
        ):
            norm.reset_parameters()
        with buffer_device:
            nn.init.normal_(self.eh_proj.weight, mean=0.0, std=init_std)
        self.self_attn.init_weights(buffer_device, init_std)
        self.mlp.init_weights(buffer_device, init_std)


class Glm5NextMTP(MTPModule):
    """GLM-5.3-Flash MTP stack: one :class:`Glm5NextMTPSublayer` per depth.

    Args:
        config: GLM-5.3 text config.
        mtp_config: Enabled MTP config with the single-sublayer pattern ``"*"``.
        moe_config: MoE config shared with the decoder's sparse layers.
        backend: Kernel backend selection.
        dtype: Parameter dtype of the fusion projection and norms.
    """

    def __init__(
        self,
        config: Glm5NextTextConfig,
        mtp_config: MTPConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype,
    ) -> None:
        if mtp_config.layer_pattern != "*":
            raise ValueError(f"GLM-5.3 MTP uses one sublayer per depth (pattern '*'), got {mtp_config.layer_pattern!r}")
        base_layer_idx = config.num_hidden_layers

        def build_sublayer(*, global_idx: int, **_: object) -> Glm5NextMTPSublayer:
            return Glm5NextMTPSublayer(config, base_layer_idx + global_idx, moe_config, backend, dtype)

        super().__init__(mtp_config, ["*"], build_sublayer)

    def update_moe_gate_bias(self) -> None:
        """Update every depth's routing correction bias."""
        for layer in self.layers:
            unwrap_checkpoint_wrapper(layer).update_moe_gate_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize every depth on ``buffer_device``."""
        for layer in self.layers:
            unwrap_checkpoint_wrapper(layer).init_weights(buffer_device, init_std)


__all__ = ["Glm5NextMTP", "Glm5NextMTPSublayer"]

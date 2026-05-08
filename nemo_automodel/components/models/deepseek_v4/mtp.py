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

"""DeepSeek V4 Multi-Token Prediction (MTP) sublayer and builder.

MTP blocks are standard pre-norm attention + MoE blocks — NO HC machinery.
The main backbone produces [B, S, hidden] hidden states (after hc_head + norm),
which are what the MTP blocks receive as input.

Key design points:
  - DeepseekV4MTPSublayer does not use HC (Hyper-Connection) machinery.
  - compress_ratios is forced to None for MTP attention layers (indices beyond
    the backbone layer count would IndexError otherwise).
  - Rotary embeddings are shared references from the main model (not
    registered as submodules to avoid polluting the state dict).
  - Each MTP depth uses one attention sublayer + one MoE sublayer (pattern "*E").
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.models.common.mtp import MTPConfig, MTPModule
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4Attention
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE


class DeepseekV4MTPSublayer(nn.Module):
    """Single MTP sublayer for DeepSeek V4.

    One sublayer corresponds to one position in the per-depth pattern. For the
    ``"*E"`` pattern, sublayer 0 is the attention block (with fusion) and
    sublayer 1 is the MoE block (with final norm).

    Args:
        config: DeepseekV4Config for the main model.
        layer_idx: Global layer index (>= num_hidden_layers for MTP layers).
        moe_config: MoEConfig shared with backbone MoE blocks.
        backend: BackendConfig for kernel selection.
        dtype: Model dtype.
        rotary_emb: Shared reference to the main model's rotary embedding.
        rotary_emb_compress: Shared reference to the compress rotary embedding.
        has_fusion: Whether this sublayer is the first in its depth (owns
            enorm/hnorm/eh_proj for fusing embed + hidden).
        has_final_norm: Whether this sublayer is the last in its depth (owns
            final_layernorm).
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype,
        rotary_emb,
        rotary_emb_compress,
        has_fusion: bool,
        has_final_norm: bool,
    ):
        super().__init__()
        H = config.hidden_size
        eps = config.rms_norm_eps
        model_dtype = dtype

        self.has_fusion = has_fusion
        self.has_final_norm = has_final_norm

        if has_fusion:
            self.enorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=model_dtype)
            self.hnorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=model_dtype)
            self.eh_proj = initialize_linear_module(backend.linear, 2 * H, H, bias=False, dtype=model_dtype)

        # Patch config: force compress_ratio=0 for MTP attention (indices beyond
        # compress_ratios list would IndexError in DeepseekV4Attention.__init__).
        mtp_cfg = copy.copy(config)
        mtp_cfg.compress_ratios = None
        self.self_attn = DeepseekV4Attention(mtp_cfg, layer_idx=layer_idx, backend=backend)

        self.mlp = MoE(moe_config, backend)
        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=model_dtype)
        self.post_attention_layernorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=model_dtype)

        if has_final_norm:
            self.final_layernorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=model_dtype)

        # Store rotary refs WITHOUT registering as submodules so they do not
        # appear in state_dict() or named_parameters().
        object.__setattr__(self, "_rotary_emb", rotary_emb)
        object.__setattr__(self, "_rotary_emb_compress", rotary_emb_compress)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **_ignored,
    ) -> torch.Tensor:
        """Forward pass for one MTP sublayer.

        Args:
            hidden_states: ``[B, S, hidden]`` tensor from the previous sublayer
                (or from the backbone's final norm for depth 0, sublayer 0).
            embed_input: ``[B, S, hidden]`` future-token embedding (only for
                the first sublayer of each depth, i.e. ``has_fusion=True``).
            position_ids: ``[B, S]`` position indices for RoPE.
            attention_mask: 4D additive causal mask ``[B, 1, S, S]``.
            padding_mask: Boolean padding mask for MoE routing.
            **_ignored: Extra kwargs forwarded from MTPModule but unused here.

        Returns:
            Updated ``[B, S, hidden]`` hidden state.
        """
        if self.has_fusion:
            assert embed_input is not None, "embed_input required for fusion sublayer"
            e = self.enorm(embed_input)
            h = self.hnorm(hidden_states)
            hidden_states = self.eh_proj(torch.cat([e, h], dim=-1))

        # Compute position embeddings from position_ids.
        if position_ids is not None:
            position_embeddings = self._rotary_emb(hidden_states, position_ids)
            position_embeddings_compress = self._rotary_emb_compress(hidden_states, position_ids)
        else:
            seq_len = hidden_states.shape[1]
            pid = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(hidden_states.shape[0], -1)
            position_embeddings = self._rotary_emb(hidden_states, pid)
            position_embeddings_compress = self._rotary_emb_compress(hidden_states, pid)

        # Attention sub-block (pre-norm residual).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_embeddings_compress=position_embeddings_compress,
            rotary_compress=self._rotary_emb_compress,
        )
        hidden_states = residual + hidden_states

        # MoE sub-block (pre-norm residual).
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)
        hidden_states = residual + hidden_states

        if self.has_final_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize sublayer weights.

        Args:
            buffer_device: Device to use for buffer initialization. Defaults
                to CPU if not provided.
        """
        init_std = float(getattr(self.self_attn.config, "initializer_range", 0.02))
        if self.has_fusion:
            self.enorm.reset_parameters()
            self.hnorm.reset_parameters()
            if buffer_device is not None:
                with buffer_device:
                    nn.init.trunc_normal_(self.eh_proj.weight, mean=0.0, std=init_std)
            else:
                nn.init.trunc_normal_(self.eh_proj.weight, mean=0.0, std=init_std)
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device or torch.device("cpu"))
        self.mlp.init_weights(buffer_device or torch.device("cpu"))
        if self.has_final_norm:
            self.final_layernorm.reset_parameters()


def build_mtp_config_from_hf(config, *, loss_scaling_factor: float = 0.1) -> MTPConfig:
    """Build an MTPConfig from a DeepseekV4Config.

    Args:
        config: DeepseekV4Config instance.
        loss_scaling_factor: Coefficient applied to the summed MTP CE loss.

    Returns:
        MTPConfig with ``num_layers`` from ``config.num_nextn_predict_layers``
        and ``layer_pattern="*E"`` (one attention + one MoE sublayer per depth).
    """
    num_layers = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    # Each MTP depth = 1 attention sublayer + 1 MoE sublayer -> pattern "*E".
    pattern = "*E" if num_layers > 0 else ""
    return MTPConfig(num_layers=num_layers, layer_pattern=pattern, loss_scaling_factor=loss_scaling_factor)


def build_deepseek_v4_mtp(
    config,
    mtp_config: MTPConfig,
    backend: BackendConfig,
    moe_config: MoEConfig,
    dtype: torch.dtype,
    rotary_emb,
    rotary_emb_compress,
) -> MTPModule:
    """Construct an MTPModule for DeepSeek V4.

    Args:
        config: DeepseekV4Config for the main model.
        mtp_config: MTPConfig describing depth count and pattern.
        backend: BackendConfig for kernel selection.
        moe_config: MoEConfig shared with the backbone.
        dtype: Model dtype.
        rotary_emb: Shared reference to the main model's rotary embedding.
        rotary_emb_compress: Shared reference to the compress rotary embedding.

    Returns:
        Constructed MTPModule with all sublayers initialized.
    """
    base_layer_idx = config.num_hidden_layers

    def factory(*, global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm):
        del depth, block_type  # V4 MTP always uses attention+MoE regardless of pattern symbol
        return DeepseekV4MTPSublayer(
            config=config,
            layer_idx=base_layer_idx + global_idx,
            moe_config=moe_config,
            backend=backend,
            dtype=dtype,
            rotary_emb=rotary_emb,
            rotary_emb_compress=rotary_emb_compress,
            has_fusion=(sublayer_idx == 0),
            has_final_norm=(sublayer_idx == mtp_config.pattern_length - 1),
        )

    return MTPModule(mtp_config=mtp_config, sublayer_factory=factory)

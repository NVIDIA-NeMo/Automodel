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

"""NemotronV3-specific Multi-Token Prediction wiring.

Glue between the model-agnostic
:mod:`nemo_automodel.components.models.common.mtp` scaffolding and the
NemotronV3 decoder block. Each MTP sublayer is a :class:`NemotronV3Block`
configured for the requested per-depth block type (``"attention"`` or
``"moe"``) plus, when relevant, the depth-level fusion modules (``enorm``,
``hnorm``, ``eh_proj``) and ``final_layernorm``.

The internal parameter naming mirrors HuggingFace's flat
``mtp.layers.{global_idx}.*`` convention used by the released Super V3
checkpoint, so the state-dict adapter performs an effectively 1-to-1 mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.mtp import MTPConfig, MTPModule
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block


class NemotronV3MTPSublayer(NemotronV3Block):
    """One MTP sublayer for NemotronV3.

    Inherits :class:`NemotronV3Block` so it has the same ``norm`` + ``mixer``
    + residual structure as a main-backbone layer; optionally adds the fusion
    modules (``enorm``/``hnorm``/``eh_proj``) on the first sublayer of each
    depth and ``final_layernorm`` on the last sublayer of each depth.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        block_type: str,
        moe_config=None,
        backend: BackendConfig | None = None,
        has_fusion: bool = False,
        has_final_norm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            config,
            layer_idx=layer_idx,
            moe_config=moe_config,
            backend=backend,
            block_type=block_type,
        )
        self.has_fusion = has_fusion
        self.has_final_norm = has_final_norm

        if has_fusion:
            H = config.hidden_size
            eps = config.layer_norm_epsilon
            self.enorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps)
            self.hnorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps)
            self.eh_proj = initialize_linear_module(
                backend.linear,
                2 * H,
                H,
                bias=False,
                dtype=dtype,
            )
        if has_final_norm:
            self.final_layernorm = initialize_rms_norm_module(
                backend.rms_norm,
                config.hidden_size,
                eps=config.layer_norm_epsilon,
            )

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize sublayer weights, including fusion modules when present."""
        super().init_weights(buffer_device=buffer_device)
        init_std = getattr(self.config, "initializer_range", 0.02)
        if self.has_fusion:
            self.enorm.reset_parameters()
            self.hnorm.reset_parameters()
            with buffer_device:
                nn.init.normal_(self.eh_proj.weight, mean=0.0, std=init_std)
                if self.eh_proj.bias is not None:
                    nn.init.zeros_(self.eh_proj.bias)
        if self.has_final_norm:
            self.final_layernorm.reset_parameters()


def build_nemotron_v3_mtp(
    config,
    mtp_config: MTPConfig,
    backend: BackendConfig,
    moe_config,
    dtype: torch.dtype,
) -> MTPModule:
    """Construct the NemotronV3 MTP block.

    Args:
        config: HF NemotronH config.
        mtp_config: Parsed MTP runtime config.
        backend: Backend configuration shared with the main backbone.
        moe_config: MoE configuration shared with the main backbone (required
            when the MTP pattern contains MoE sublayers).
        dtype: Target dtype for newly created linear modules.

    Returns:
        A configured :class:`MTPModule`. Caller should not invoke this when
        ``mtp_config.enabled`` is ``False``.
    """
    base_layer_idx = config.num_hidden_layers

    def factory(*, global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm):
        return NemotronV3MTPSublayer(
            config,
            layer_idx=base_layer_idx + global_idx,
            block_type=block_type,
            moe_config=moe_config,
            backend=backend,
            has_fusion=has_fusion,
            has_final_norm=has_final_norm,
            dtype=dtype,
        )

    return MTPModule(mtp_config=mtp_config, sublayer_factory=factory)


def build_mtp_config_from_hf(
    config,
    *,
    loss_scaling_factor: float = 0.1,
) -> MTPConfig:
    """Construct an :class:`MTPConfig` from an HF NemotronH config.

    Reads ``num_nextn_predict_layers`` and ``mtp_hybrid_override_pattern``
    directly off the HF config object (both present on the released Super V3
    ``config.json``). Returns a disabled config (``num_layers=0``) when MTP
    is not configured.

    Args:
        config: HF NemotronH config.
        loss_scaling_factor: Training-only scaling factor applied to the
            summed per-depth loss. Not stored on the HF config; supplied by
            the recipe YAML.

    Returns:
        :class:`MTPConfig`.
    """
    num_layers = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    pattern = getattr(config, "mtp_hybrid_override_pattern", "") or ""
    return MTPConfig(
        num_layers=num_layers,
        layer_pattern=pattern,
        loss_scaling_factor=loss_scaling_factor,
    )

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

# Hybrid layer symbols matching HF Nemotron-H's ``hybrid_override_pattern``
# and ``mtp_hybrid_override_pattern``. Specific to this model family; lives
# here rather than in common scaffolding because no other model family uses
# this exact symbol convention.
_PATTERN_SYMBOL_TO_BLOCK_TYPE = {
    "M": "mamba",
    "*": "attention",
    "-": "mlp",
    "E": "moe",
}


def parse_mtp_layer_pattern(pattern: str) -> list[str]:
    """Parse a NemotronH MTP layer pattern (e.g. ``"*E"``) into block types.

    Args:
        pattern: Pattern string using symbols ``M`` (mamba), ``*`` (attention),
            ``-`` (mlp), ``E`` (moe).

    Returns:
        List of block-type names (``"mamba"``, ``"attention"``, ``"mlp"``, ``"moe"``).

    Raises:
        ValueError: If the pattern is empty or contains unknown symbols.
    """
    if not pattern:
        raise ValueError("MTP layer pattern is empty")
    blocks: list[str] = []
    for ch in pattern:
        if ch not in _PATTERN_SYMBOL_TO_BLOCK_TYPE:
            raise ValueError(
                f"Unknown MTP layer symbol {ch!r} in pattern {pattern!r}; "
                f"valid symbols are {sorted(_PATTERN_SYMBOL_TO_BLOCK_TYPE.keys())}"
            )
        blocks.append(_PATTERN_SYMBOL_TO_BLOCK_TYPE[ch])
    return blocks


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run optional fusion (first sublayer of a depth), the base block, and
        optional final_layernorm (last sublayer of a depth).

        Keeping the fusion + final-norm calls inside the sublayer's own forward
        ensures FSDP2's pre-forward unshard hook fires for every parameter we
        touch, so children like ``enorm``/``hnorm``/``eh_proj``/``final_layernorm``
        are never accessed while their weights are still sharded DTensors.
        """
        if self.has_fusion:
            assert embed_input is not None, "first MTP sublayer requires embed_input"
            e = self.enorm(embed_input)
            h = self.hnorm(hidden_states)
            hidden_states = self.eh_proj(torch.cat([e, h], dim=-1))
        hidden_states = super().forward(hidden_states, **kwargs)
        if self.has_final_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

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
    block_types_per_sublayer = parse_mtp_layer_pattern(mtp_config.layer_pattern)

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

    return MTPModule(
        mtp_config=mtp_config,
        block_types_per_sublayer=block_types_per_sublayer,
        sublayer_factory=factory,
    )


def build_mtp_config_from_hf(
    config,
    *,
    loss_scaling_factor: float = 0.1,
    num_nextn_predict_layers: int | None = None,
    use_repeated_layer: bool = False,
) -> MTPConfig:
    """Construct an :class:`MTPConfig` from an HF NemotronH config.

    Reads ``num_nextn_predict_layers`` and ``mtp_hybrid_override_pattern``
    directly off the HF config object (both present on the released Super V3
    ``config.json``). Returns a disabled config (``num_layers=0``) when MTP
    is not configured.

    Args:
        config: HF NemotronH config.
        loss_scaling_factor: Auxiliary-loss weight applied to the summed
            per-depth CE (default ``0.1``). Not stored on the HF config;
            override programmatically when constructing the model.
        num_nextn_predict_layers: Optional override for the HF config's
            ``num_nextn_predict_layers`` field. When ``None``, uses the value
            from ``config``. Set explicitly when the trained model used
            weight-tied MTP iterations (``use_repeated_layer=True``) and the
            HF export only retains the physical depth count.
        use_repeated_layer: When ``True``, build only one physical MTP depth
            and reuse it across all iterations. Mirrors Megatron's
            ``--mtp-use-repeated-layer``. Defaults to ``False``.

    Returns:
        :class:`MTPConfig`.
    """
    if num_nextn_predict_layers is None:
        num_layers = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    else:
        num_layers = int(num_nextn_predict_layers)
    pattern = getattr(config, "mtp_hybrid_override_pattern", "") or ""
    return MTPConfig(
        num_layers=num_layers,
        layer_pattern=pattern,
        loss_scaling_factor=loss_scaling_factor,
        use_repeated_layer=use_repeated_layer,
    )

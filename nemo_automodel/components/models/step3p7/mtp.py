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

"""Step3 Multi-Token Prediction blocks.

Step checkpoints store MTP depths after the main decoder layers as
``model.layers.{num_hidden_layers + depth}.*``.  Each depth has the same
decoder block structure plus fusion modules (``enorm``, ``hnorm``, ``eh_proj``)
and an MTP-local shared head under ``transformer.shared_head``.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.mtp import MTPConfig, roll_tensor
from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step3p5.model import Block
from nemo_automodel.components.moe.config import MoEConfig


def _get_indexed_value(values: Any, index: int, default: Any) -> Any:
    if values is None:
        return default
    if isinstance(values, (list, tuple)):
        if not values:
            return default
        if index < len(values):
            return values[index]
        return values[-1]
    return values


def _ensure_indexed(values: Any, index: int, value: Any) -> list[Any]:
    if values is None:
        values = []
    elif not isinstance(values, list):
        values = [values]
    else:
        values = list(values)
    fill = values[-1] if values else value
    if len(values) <= index:
        values.extend([fill] * (index + 1 - len(values)))
    values[index] = value
    return values


def _make_mtp_block_config(config: Any, layer_idx: int, depth: int) -> Any:
    """Return a shallow config copy patched for a dense sliding-attention MTP layer."""
    mtp_cfg = copy.copy(config)
    mtp_cfg.num_hidden_layers = max(int(getattr(config, "num_hidden_layers", 0) or 0), layer_idx + 1)
    mtp_cfg.moe_layers_enum = ()

    mtp_layer_types = getattr(config, "mtp_layer_types", None)
    layer_type = _get_indexed_value(mtp_layer_types, depth, "sliding_attention")
    mtp_cfg.layer_types = _ensure_indexed(getattr(config, "layer_types", None), layer_idx, layer_type)

    mtp_swiglu_limits = getattr(config, "mtp_swiglu_limits", None)
    mtp_cfg.swiglu_limits = _ensure_indexed(
        getattr(config, "swiglu_limits", None),
        layer_idx,
        _get_indexed_value(mtp_swiglu_limits, depth, 0.0),
    )

    mtp_swiglu_limits_shared = getattr(config, "mtp_swiglu_limits_shared", None)
    mtp_cfg.swiglu_limits_shared = _ensure_indexed(
        getattr(config, "swiglu_limits_shared", None),
        layer_idx,
        _get_indexed_value(mtp_swiglu_limits_shared, depth, 0.0),
    )

    mtp_partial_rotary_factors = getattr(config, "mtp_partial_rotary_factors", None)
    mtp_cfg.partial_rotary_factors = _ensure_indexed(
        getattr(config, "partial_rotary_factors", None),
        layer_idx,
        _get_indexed_value(mtp_partial_rotary_factors, depth, 1.0),
    )

    mtp_rope_theta = getattr(config, "mtp_rope_theta", None)
    rope_theta = getattr(config, "rope_theta", 10000.0)
    if isinstance(rope_theta, (list, tuple)) or mtp_rope_theta is not None:
        mtp_cfg.rope_theta = _ensure_indexed(
            rope_theta if isinstance(rope_theta, (list, tuple)) else None,
            layer_idx,
            _get_indexed_value(mtp_rope_theta, depth, 10000.0),
        )

    mtp_use_rope_layers = getattr(config, "mtp_use_rope_layers", None)
    if getattr(config, "use_rope_layers", None) is not None or mtp_use_rope_layers is not None:
        mtp_cfg.use_rope_layers = _ensure_indexed(
            getattr(config, "use_rope_layers", None),
            layer_idx,
            _get_indexed_value(mtp_use_rope_layers, depth, True),
        )

    return mtp_cfg


class Step3p5MTPSharedHead(nn.Module):
    """Per-depth Step MTP prediction head."""

    def __init__(self, config: Any, backend: BackendConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.norm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(hidden_states))

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        self.norm.reset_parameters()
        final_out_std = self.output.weight.shape[1] ** -0.5
        with buffer_device:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-3 * final_out_std,
                b=3 * final_out_std,
            )


class Step3p5MTPBlock(Block):
    """One Step MTP prediction depth."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        depth: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype,
    ) -> None:
        mtp_cfg = _make_mtp_block_config(config, layer_idx, depth)
        super().__init__(layer_idx, mtp_cfg, moe_config, backend)
        self.enorm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = Step3p5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = initialize_linear_module(
            backend.linear,
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
        )
        self.transformer = nn.Module()
        self.transformer.shared_head = Step3p5MTPSharedHead(config, backend, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e = self.enorm(embed_input)
        h = self.hnorm(hidden_states)
        hidden_states = self.eh_proj(torch.cat([e, h], dim=-1))
        hidden_states = super().forward(
            hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            position_ids=position_ids,
            **attn_kwargs,
        )
        logits = self.transformer.shared_head(hidden_states)
        return hidden_states, logits

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        super().init_weights(buffer_device)
        self.enorm.reset_parameters()
        self.hnorm.reset_parameters()
        with buffer_device:
            nn.init.trunc_normal_(self.eh_proj.weight, mean=0.0, std=0.02)
        self.transformer.shared_head.init_weights(buffer_device)


class Step3p5MTPModule(nn.Module):
    """Stack of Step MTP depths."""

    def __init__(
        self,
        config: Any,
        mtp_config: MTPConfig,
        backend: BackendConfig,
        moe_config: MoEConfig,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if not mtp_config.enabled:
            raise ValueError("Step3p5MTPModule constructed with disabled MTPConfig")
        self.mtp_config = mtp_config
        base_layer_idx = int(getattr(config, "mtp_base_layer_idx", config.num_hidden_layers))
        self.layers = nn.ModuleList(
            [
                Step3p5MTPBlock(
                    config=config,
                    layer_idx=base_layer_idx + depth,
                    depth=depth,
                    moe_config=moe_config,
                    backend=backend,
                    dtype=dtype,
                )
                for depth in range(mtp_config.num_layers)
            ]
        )

    @property
    def num_depths(self) -> int:
        return self.mtp_config.num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        input_ids: torch.LongTensor | None = None,
        embed_fn=None,
        embed_inputs: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        **block_kwargs: Any,
    ) -> list[torch.Tensor]:
        per_depth_logits: list[torch.Tensor] = []
        cur_input_ids = input_ids
        if embed_inputs is not None and len(embed_inputs) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} MTP embedding tensors, got {len(embed_inputs)}")
        if embed_inputs is None and (cur_input_ids is None or embed_fn is None):
            raise ValueError("MTP requires either embed_inputs or both input_ids and embed_fn")

        for depth, block in enumerate(self.layers):
            if embed_inputs is None:
                cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
                decoder_input = embed_fn(cur_input_ids)
            else:
                decoder_input = embed_inputs[depth]
            hidden_states, logits = block(
                hidden_states,
                embed_input=decoder_input,
                freqs_cis=freqs_cis,
                position_ids=position_ids,
                **block_kwargs,
            )
            per_depth_logits.append(logits)
        return per_depth_logits


def build_mtp_config_from_hf(config: Any, *, loss_scaling_factor: float = 0.1) -> MTPConfig:
    """Build Step MTP runtime config from HF-style config fields."""
    num_layers = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    return MTPConfig(
        num_layers=num_layers,
        layer_pattern="*" if num_layers > 0 else "",
        loss_scaling_factor=loss_scaling_factor,
    )


def build_step3p5_mtp(
    config: Any,
    mtp_config: MTPConfig,
    backend: BackendConfig,
    moe_config: MoEConfig,
    dtype: torch.dtype,
) -> Step3p5MTPModule:
    """Construct Step MTP depths."""
    return Step3p5MTPModule(
        config=config,
        mtp_config=mtp_config,
        backend=backend,
        moe_config=moe_config,
        dtype=dtype,
    )

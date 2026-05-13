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

"""Gemma4 Multi-Token Prediction (MTP) glue.

Wires the model-agnostic :mod:`nemo_automodel.components.models.common.mtp`
scaffold to a Gemma4 backbone. Each MTP sublayer wraps HF's
``Gemma4TextDecoderLayer`` configured for a clean, dense, full-attention block
(no per-layer-input, no MoE branch, no KV-sharing) so the MTP head is
independent of the variant-specific text-config knobs (E2B/E4B per-layer-input,
26B-A4B MoE, KV-sharing in E4B,E2B, etc.). The MTP module attaches as a sibling of
``model.language_model`` and works for every Gemma4 variant because the only
inputs it consumes from the backbone are ``hidden_size``, ``rms_norm_eps``,
``rope_parameters['full_attention']``, and the post-final-norm hidden state.

Optional fusion modules (``enorm``, ``hnorm``, ``eh_proj``) live on the first
sublayer of each depth; ``final_layernorm`` lives on the last sublayer of each
depth. Both are run from inside the sublayer's own ``forward`` so FSDP2's
pre-forward unshard hook covers every parameter we touch as done in nemotron_v3 MTP
PR #2161 commit 0b2889ab.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common.mtp import MTPConfig, MTPModule
from nemo_automodel.shared.import_utils import safe_import

_G4_HAS_HF, _g4_modeling = safe_import("transformers.models.gemma4.modeling_gemma4")
if _G4_HAS_HF:
    Gemma4RMSNorm = _g4_modeling.Gemma4RMSNorm
    Gemma4TextDecoderLayer = _g4_modeling.Gemma4TextDecoderLayer
    Gemma4TextRotaryEmbedding = _g4_modeling.Gemma4TextRotaryEmbedding
else:
    Gemma4RMSNorm = None  # type: ignore[assignment]
    Gemma4TextDecoderLayer = None  # type: ignore[assignment]
    Gemma4TextRotaryEmbedding = None  # type: ignore[assignment]


def _make_mtp_text_config(text_config: Any) -> Any:
    """Return a shallow copy of ``text_config`` configured for the MTP sublayer.

    The MTP sublayer is a single full-attention dense decoder layer that does
    NOT participate in the backbone's per-layer-input, MoE, KV-sharing, or
    sliding-window machinery. Constructing the sublayer with a pruned config
    avoids having to thread per-layer-input tensors through MTP, lets us pass
    ``layer_idx=0`` regardless of the backbone depth, and produces a clean
    parameter set under ``mtp.layers.{k}.*``.

    Args:
        text_config: HF ``Gemma4TextConfig`` from the backbone.

    Returns:
        A copy of ``text_config`` with: ``hidden_size_per_layer_input=None``,
        ``enable_moe_block=False``, ``num_kv_shared_layers=0``,
        ``layer_types=["full_attention"]``, ``attention_k_eq_v=False``,
        and ``rope_parameters`` pruned to ``{"full_attention": ...}``.
    """
    cfg = copy.copy(text_config)
    cfg.hidden_size_per_layer_input = None
    cfg.enable_moe_block = False
    cfg.num_kv_shared_layers = 0
    cfg.layer_types = ["full_attention"]
    cfg.attention_k_eq_v = False
    rope_params = getattr(text_config, "rope_parameters", None)
    if isinstance(rope_params, dict) and "full_attention" in rope_params:
        cfg.rope_parameters = {"full_attention": rope_params["full_attention"]}
    return cfg


class Gemma4MTPSublayer(nn.Module):
    """One MTP sublayer for Gemma4.

    Wraps a single ``Gemma4TextDecoderLayer`` (built against an MTP-pruned
    text config) plus optional fusion modules (``enorm``/``hnorm``/``eh_proj``)
    on the first sublayer of a depth (aka MTP module) and ``final_layernorm`` on the last
    sublayer of a depth. Naming matches the HF flat ``mtp.layers.{i}.*``
    convention used by other MTP-bearing models (NemotronV3, DeepSeek-V4).

    Each sublayer owns its own :class:`Gemma4TextRotaryEmbedding`. This is
    necessary because ``MTPModule`` rolls ``position_ids`` cumulatively per
    depth (aka MTP module) — so the cos/sin precomputed by the backbone for the un-rolled
    positions would be stale. The rotary module has no trainable parameters
    (only buffers), so the per-sublayer copy is essentially free.

    Args:
        mtp_text_config: MTP-pruned text config (from :func:`_make_mtp_text_config`).
        layer_idx: Layer index inside ``mtp_text_config.layer_types`` (always 0
            for Gemma4 since the pruned config has a single full-attention layer).
        has_fusion: When True, instantiates ``enorm``/``hnorm``/``eh_proj`` for
            depth-level token+hidden fusion.
        has_final_norm: When True, instantiates ``final_layernorm`` applied at
            the end of this sublayer's (i.e. MTP module) forward.
        dtype: Dtype for the linear/norm parameters created here.
    """

    def __init__(
        self,
        mtp_text_config: Any,
        *,
        layer_idx: int = 0,
        has_fusion: bool = False,
        has_final_norm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if not _G4_HAS_HF:
            raise ImportError(
                "transformers.models.gemma4 is not available; install transformers>=5.5 to use Gemma4MTPSublayer."
            )
        self.config = mtp_text_config
        self.layer_idx = layer_idx
        self.has_fusion = has_fusion
        self.has_final_norm = has_final_norm

        # The actual transformer block (attention + MLP + Gemma's pre/post norms).
        self.block = Gemma4TextDecoderLayer(mtp_text_config, layer_idx=layer_idx)
        # Owned per-sublayer so we can compute (cos, sin) for *rolled* position_ids.
        self.rotary_emb = Gemma4TextRotaryEmbedding(mtp_text_config)

        H = mtp_text_config.hidden_size
        eps = mtp_text_config.rms_norm_eps
        if has_fusion:
            self.enorm = Gemma4RMSNorm(H, eps=eps)
            self.hnorm = Gemma4RMSNorm(H, eps=eps)
            self.eh_proj = nn.Linear(2 * H, H, bias=False, dtype=dtype)
        if has_final_norm:
            self.final_layernorm = Gemma4RMSNorm(H, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run optional fusion, the Gemma4 decoder block, and optional final_layernorm.

        ``position_ids`` is the rolled tensor produced by ``MTPModule``; we
        recompute ``position_embeddings`` from it via the owned rotary module
        rather than reusing the backbone's stale cos/sin.

        Keeping fusion + final_layernorm calls inside this sublayer's forward
        ensures FSDP2's pre-forward unshard hook fires for every parameter we
        touch on the way through (matches the NemotronV3 MTP pattern).
        """
        kwargs.pop("position_embeddings", None)
        if self.has_fusion:
            assert embed_input is not None, "first MTP sublayer requires embed_input"
            e = self.enorm(embed_input)
            h = self.hnorm(hidden_states)
            hidden_states = self.eh_proj(torch.cat([e, h], dim=-1))
        position_embeddings = self.rotary_emb(hidden_states, position_ids, layer_type="full_attention")
        # Gemma4TextDecoderLayer ignores per_layer_input when
        # hidden_size_per_layer_input is None on the config.
        hidden_states = self.block(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        if self.has_final_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize sublayer weights, including fusion / final_layernorm."""
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        device_ctx = buffer_device if buffer_device is not None else torch.device("cpu")
        # Initialize the wrapped Gemma4 decoder block via HF's _init_weights.
        for m in self.block.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif Gemma4RMSNorm is not None and isinstance(m, Gemma4RMSNorm):
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
        if self.has_fusion:
            self.enorm.reset_parameters()
            self.hnorm.reset_parameters()
            with device_ctx:
                nn.init.normal_(self.eh_proj.weight, mean=0.0, std=init_std)
                if self.eh_proj.bias is not None:
                    nn.init.zeros_(self.eh_proj.bias)
        if self.has_final_norm:
            self.final_layernorm.reset_parameters()


def build_gemma4_mtp(
    text_config: Any,
    mtp_config: MTPConfig,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> MTPModule:
    """Construct the Gemma4 MTP block.

    Args:
        text_config: HF ``Gemma4TextConfig`` from the backbone.
        mtp_config: Parsed MTP runtime config (``num_layers``,
            ``layer_pattern``, ``loss_scaling_factor``).
        dtype: Dtype for newly-created MTP parameters.

    Returns:
        A configured :class:`MTPModule`. Caller should not invoke this when
        ``mtp_config.enabled`` is False.
    """
    if not _G4_HAS_HF:
        raise ImportError("transformers.models.gemma4 is not available; install transformers>=5.5 to use Gemma4 MTP.")
    if not mtp_config.enabled:
        raise ValueError("build_gemma4_mtp called with a disabled MTPConfig")

    mtp_text_config = _make_mtp_text_config(text_config)

    def factory(*, global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm):
        # block_type comes from the MTP pattern string ("*", "-", "E", ...).
        # Gemma4 supports only the dense full-attention sublayer ("*"); MoE
        # MTP sublayers are not implemented currently (would require the existing
        # Gemma4MoE block which depends on text_config.enable_moe_block).
        if block_type != "attention":
            raise NotImplementedError(
                f"Gemma4 MTP sublayer of type {block_type!r} is not supported; "
                f"use layer_pattern='*' (single dense full-attention sublayer per depth)."
            )
        return Gemma4MTPSublayer(
            mtp_text_config,
            layer_idx=0,
            has_fusion=has_fusion,
            has_final_norm=has_final_norm,
            dtype=dtype,
        )

    return MTPModule(mtp_config=mtp_config, sublayer_factory=factory)


def build_mtp_config_from_kwargs(
    *,
    mtp_num_layers: int = 0,
    mtp_layer_pattern: str = "*",
    mtp_loss_scaling_factor: float = 0.1,
) -> MTPConfig:
    """Build an MTPConfig for Gemma4 from constructor kwargs.

    HF Gemma4 configs do NOT carry MTP fields (Gemma4 has no native MTP),
    so the recipe activates MTP by passing kwargs through
    ``Gemma4ForConditionalGeneration.__init__``. Returns a disabled config
    (``num_layers=0``) when MTP is not requested.

    Args:
        mtp_num_layers: Number of MTP depths (``D``). ``0`` disables MTP.
        mtp_layer_pattern: Per-depth sublayer pattern. Only ``"*"`` (single
            full-attention sublayer) is currently supported for Gemma4.
        mtp_loss_scaling_factor: Coefficient for the summed per-depth CE loss.

    Returns:
        :class:`MTPConfig`.
    """
    return MTPConfig(
        num_layers=int(mtp_num_layers or 0),
        layer_pattern=str(mtp_layer_pattern or ""),
        loss_scaling_factor=float(mtp_loss_scaling_factor),
    )

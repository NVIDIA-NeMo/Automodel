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

"""Configuration for the BAGEL mixed-modal LLM.

The config wraps a text config and a vision config. The flags
``visual_und`` / ``visual_gen`` gate which paths are built at init time:

* Stage 1 (understanding-only): ``visual_und=True``, ``visual_gen=False``.
  Only the ViT + connector + LM path is active.
* Stage 2 (joint): ``visual_gen=True``. Activates the MoT ``*_moe_gen``
  parameter siblings, the VAE encode path, and the flow-matching head.

The checkpoint config names the nested configs ``llm_config`` / ``vit_config``.
AM prefers ``text_config`` / ``vision_config`` to match the rest of the VLM
tree. We accept both sets of keys on input and expose both attributes on the
instance so that:

* ``BagelConfig.from_pretrained("ByteDance-Seed/BAGEL-7B-MoT")`` works
  against the checkpoint ``config.json``.
* AM-native YAML (``_target_: nemo_automodel...BagelConfig``) can use the
  AM-flavored key names without surprises.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig

from nemo_automodel.components.models.bagel.modeling_siglip_navit import SiglipVisionConfig


@dataclass(frozen=True, kw_only=True)
class BagelBackendConfig:
    """Backends that BAGEL allows users to select."""

    linear: Literal["torch", "te"] = "torch"
    rms_norm: Literal["torch", "torch_fp32", "te"] = "torch_fp32"
    # Fuse silu(gate)*up into one compiled pointwise kernel in Qwen2MLP.
    fused_swiglu: bool = False
    # Fuse the RoPE rotate/mul/add into one compiled pointwise kernel in attention.
    fused_rope: bool = False
    # MLM-style grouped MoT routing: permute und/gen tokens into contiguous blocks once.
    mot_grouped: bool = False

    def __post_init__(self) -> None:
        if self.linear not in {"torch", "te"}:
            raise ValueError(f"Unsupported BAGEL linear backend: {self.linear!r}")
        if self.rms_norm not in {"torch", "torch_fp32", "te"}:
            raise ValueError(f"Unsupported BAGEL RMSNorm backend: {self.rms_norm!r}")


def resolve_bagel_backend(backend: Mapping[str, Any] | BagelBackendConfig | None = None) -> BagelBackendConfig:
    """Resolve a mapping against BAGEL's stable torch defaults."""
    if backend is None:
        return BagelBackendConfig()
    if isinstance(backend, BagelBackendConfig):
        return backend

    if not isinstance(backend, Mapping):
        raise TypeError(f"BAGEL backend must be a mapping or BagelBackendConfig, got {type(backend)!r}")

    overrides = dict(backend)
    unknown_fields = sorted(overrides.keys() - {"linear", "rms_norm", "fused_swiglu", "fused_rope", "mot_grouped"})
    if unknown_fields:
        raise TypeError(f"Unknown BAGEL backend field(s): {unknown_fields}")
    return BagelBackendConfig(**overrides)


BagelLayerModule = Literal["Qwen2DecoderLayer", "Qwen2MoTDecoderLayer"]


class BagelTextConfig(Qwen2Config):
    """Qwen2 text config with BAGEL's packed/MoT extension fields."""

    qk_norm: bool = True
    layer_module: BagelLayerModule | None = "Qwen2DecoderLayer"
    freeze_und: bool = False
    is_causal: bool = True
    partial_rotary_factor: float = 1.0
    pad_token_id: int | None = 151643

    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)
        if self.pad_token_id is None:
            self.pad_token_id = 151643
        if self.layer_module not in (None, "Qwen2DecoderLayer", "Qwen2MoTDecoderLayer"):
            raise ValueError(
                "BAGEL text_config.layer_module must be "
                "'Qwen2DecoderLayer', 'Qwen2MoTDecoderLayer', or None "
                f"(got {self.layer_module!r})."
            )


class BagelVAEConfig(PretrainedConfig):
    """VAE metadata consumed by BAGEL's visual-generation path."""

    model_type = "bagel_vae"

    z_channels: int | None = None
    downsample: int | None = None


def _coerce_text_config(cfg: Mapping[str, Any] | BagelTextConfig | None) -> BagelTextConfig:
    """Coerce ``cfg`` into BAGEL's typed Qwen2 text config."""
    if cfg is None:
        return BagelTextConfig()
    if isinstance(cfg, Mapping):
        return BagelTextConfig(**dict(cfg))
    if not isinstance(cfg, BagelTextConfig):
        raise TypeError(f"text_config must be a mapping / BagelTextConfig / None (got {type(cfg).__name__})")
    return cfg


def _coerce_vision_config(cfg: Mapping[str, Any] | SiglipVisionConfig | None) -> SiglipVisionConfig | None:
    """Coerce ``cfg`` into a ``SiglipVisionConfig`` (our ``rope``-flag variant)."""
    if cfg is None:
        return None
    if isinstance(cfg, Mapping):
        return SiglipVisionConfig(**dict(cfg))
    if not isinstance(cfg, SiglipVisionConfig):
        raise TypeError(f"vision_config must be a mapping / SiglipVisionConfig / None (got {type(cfg).__name__})")
    return cfg


def _coerce_vae_config(cfg: Mapping[str, Any] | BagelVAEConfig | None) -> BagelVAEConfig | None:
    """Coerce ``cfg`` into BAGEL's typed VAE metadata config."""
    if cfg is None:
        return None
    if isinstance(cfg, Mapping):
        return BagelVAEConfig(**dict(cfg))
    if not isinstance(cfg, BagelVAEConfig):
        raise TypeError(f"vae_config must be a mapping / BagelVAEConfig / None (got {type(cfg).__name__})")
    return cfg


class BagelConfig(PretrainedConfig):
    """Top-level BAGEL config.

    The text and vision sub-configs are nested :class:`PretrainedConfig`
    instances (not bare dicts) so callers can introspect them the same way
    they would with any other HF config.

    Attribute aliases:
    * ``text_config`` <-> ``llm_config`` (both point at the same object)
    * ``vision_config`` <-> ``vit_config`` (ditto)
    """

    model_type = "bagel"
    sub_configs = {
        "text_config": BagelTextConfig,
        "vision_config": SiglipVisionConfig,
        "vae_config": BagelVAEConfig,
    }

    def __init__(
        self,
        vision_config: Mapping[str, Any] | SiglipVisionConfig | None = None,
        text_config: Mapping[str, Any] | BagelTextConfig | None = None,
        *,
        visual_und: bool = True,
        visual_gen: bool = False,
        stage: int | str | None = None,
        llm_path: str = "",
        vit_path: str = "",
        vae_path: str = "",
        max_latent_size: int = 64,
        latent_patch_size: int = 2,
        vit_patch_size: int = 14,
        vit_max_num_patch_per_side: int = 70,
        connector_act: str = "gelu_pytorch_tanh",
        interpolate_pos: bool = False,
        vit_select_layer: int = -2,
        vit_rope: bool = False,
        text_cond_dropout_prob: float = 0.1,
        vae_cond_dropout_prob: float = 0.3,
        vit_cond_dropout_prob: float = 0.3,
        timestep_shift: float = 1.0,
        pad_token_id: int = 151643,
        # Checkpoint aliases.
        llm_config: Mapping[str, Any] | BagelTextConfig | None = None,
        vit_config: Mapping[str, Any] | SiglipVisionConfig | None = None,
        vae_config: Mapping[str, Any] | BagelVAEConfig | None = None,
        **kwargs: Any,
    ) -> None:
        # Resolve aliases: if both are passed and conflict, prefer the AM name.
        if text_config is None and llm_config is not None:
            text_config = llm_config
        if vision_config is None and vit_config is not None:
            vision_config = vit_config

        self.text_config = _coerce_text_config(text_config)
        self.vision_config = _coerce_vision_config(vision_config)
        self.vae_config = _coerce_vae_config(vae_config)

        self.visual_und = visual_und
        self.visual_gen = visual_gen
        self.stage = stage

        # Paths (informational; bookkeeping for how the checkpoint was built).
        self.llm_path = llm_path
        self.vit_path = vit_path
        self.vae_path = vae_path

        # Flow-matching parameters - kept regardless of stage so state-dict
        # shapes are reproducible after init, even when the gen path is dormant.
        self.max_latent_size = max_latent_size
        self.latent_patch_size = latent_patch_size

        # ViT configuration (mostly carried on self.vision_config; these
        # top-level attributes mirror the BAGEL checkpoint config for
        # backwards compatibility with that code path).
        self.vit_patch_size = vit_patch_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.vit_select_layer = vit_select_layer
        self.vit_rope = vit_rope

        # Dropout knobs consumed by the BAGEL data pipeline.
        self.text_cond_dropout_prob = text_cond_dropout_prob
        self.vae_cond_dropout_prob = vae_cond_dropout_prob
        self.vit_cond_dropout_prob = vit_cond_dropout_prob
        self.timestep_shift = timestep_shift

        super().__init__(pad_token_id=pad_token_id, **kwargs)
        if not self.architectures:
            self.architectures = ["BagelForUnifiedMultimodal"]

    # Checkpoint-named read aliases (so downstream code can do either
    # ``config.llm_config`` or ``config.text_config`` interchangeably).
    @property
    def llm_config(self) -> BagelTextConfig:
        return self.text_config

    @llm_config.setter
    def llm_config(self, value: Mapping[str, Any] | BagelTextConfig | None) -> None:
        self.text_config = _coerce_text_config(value)

    @property
    def vit_config(self) -> SiglipVisionConfig | None:
        return self.vision_config

    @vit_config.setter
    def vit_config(self, value: Mapping[str, Any] | SiglipVisionConfig | None) -> None:
        self.vision_config = _coerce_vision_config(value)

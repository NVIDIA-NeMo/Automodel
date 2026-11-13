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

"""Typed view of the Praxis checkpoint ``config.json``.

The checkpoint config is a nested JSON document. The sections the model consumes are
parsed into frozen dataclasses with explicit fields; unknown keys raise so that a
checkpoint of another model variant is refused instead of silently misread.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, TypeVar

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_T = TypeVar("_T")

MODEL_TYPE = "praxis"
FORMAT_VERSION = 2


class PraxisConfigError(ValueError):
    """Raised when ``config.json`` is not a supported Praxis single-arm config."""


def _section(parent: Mapping[str, JsonValue], name: str) -> Mapping[str, JsonValue]:
    value = parent.get(name, {})
    if not isinstance(value, Mapping):
        raise PraxisConfigError(f"config section {name!r} must be an object")
    return value


def _dataclass_from(cls: type[_T], section: Mapping[str, JsonValue], where: str) -> _T:
    """Build ``cls`` from ``section`` with its field names; unknown keys are an error."""
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(section) - names)
    if unknown:
        raise PraxisConfigError(f"config: unknown keys in {where}: {unknown}")
    kwargs: dict[str, JsonValue] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name in section:
            value = section[f.name]
            kwargs[f.name] = tuple(value) if isinstance(value, list) else value
    try:
        return cls(**kwargs)  # type: ignore[arg-type]
    except TypeError as err:
        raise PraxisConfigError(f"config: {where}: {err}") from err


@dataclass(frozen=True)
class AttentionConfig:
    """``backbone.attention``."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    gated: bool = True
    qk_norm_eps: float = 1e-6
    qk_norm_compute_dtype: str = "float32"
    rope_after_norm: bool = False
    flex_block_size: int = 128
    backend: Literal["triton", "flash"] = "triton"


@dataclass(frozen=True)
class RopeConfig:
    """``backbone.rope``: integer-position rotary embedding of the trunk."""

    base: int = 26_000_000
    dim: int = 128
    effective_dim: int = 64
    max_seq_len: int = 90_000_000
    efficient_cache_max: int = 1_000_000
    max_intra: int = 360
    max_inter: int = 129_600
    chunked_video_layout: bool = True
    n_chunks: int | None = None
    time_scale: float = 1.0
    n_tiles: int | None = None
    ref_placement: Literal["group"] = "group"


@dataclass(frozen=True)
class MoeSectionConfig:
    """``backbone.moe``: a DeepSeek-style sigmoid router with a shared expert."""

    num_routed_experts: int
    top_k_routed: int
    expert_hidden: int
    num_experts: int | None = None
    shared_expert: bool = True
    top_k: int | None = None
    router_activation: Literal["sigmoid"] = "sigmoid"
    router_dtype: str = "float32"
    routed_scaling_factor: float = 2.5
    no_expert_slots: bool = False
    routed_weight_layout: Literal["expert_first"] = "expert_first"
    dispatch_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if not self.shared_expert:
            raise PraxisConfigError("config: backbone.moe.shared_expert must be true")
        if self.num_experts is None:
            object.__setattr__(self, "num_experts", self.num_routed_experts + 1)
        if self.top_k is None:
            object.__setattr__(self, "top_k", self.top_k_routed + 1)
        if self.num_experts != self.num_routed_experts + 1 or self.top_k != self.top_k_routed + 1:
            raise PraxisConfigError("config: backbone.moe counts one shared expert on top of the routed ones")


@dataclass(frozen=True)
class ProjectionConfig:
    """One entry of ``input_projections`` / ``output_projections``."""

    c_in: int | None = None
    c_out: int | None = None
    move_dim: bool = False
    timestep_modulated: bool = False


@dataclass(frozen=True)
class TowerConfig:
    """``backbone.encoder`` or ``backbone.decoder``."""

    depth: int
    dense_blocks: tuple[int, ...]
    text_in: ProjectionConfig | None
    video_in: ProjectionConfig
    video_out: ProjectionConfig
    tap_blocks: tuple[int, ...] = ()
    modulation_dim: int | None = None
    unpatch_factor: tuple[int, int, int] = (1, 1, 1)
    positive_zero_residual_add: bool = True

    @classmethod
    def from_json(cls, section: Mapping[str, JsonValue], where: str) -> TowerConfig:
        known = {
            "depth",
            "dense_blocks",
            "tap_blocks",
            "input_projections",
            "output_projections",
            "modulation_dim",
            "unpatch_factor",
            "positive_zero_residual_add",
        }
        unknown = sorted(set(section) - known)
        if unknown:
            raise PraxisConfigError(f"config: unknown keys in {where}: {unknown}")
        inputs = _section(section, "input_projections")
        outputs = _section(section, "output_projections")
        text = inputs.get("TextEncodingTensor")
        text_in = (
            None if text is None else _dataclass_from(ProjectionConfig, _section(inputs, "TextEncodingTensor"), where)
        )
        video_in = _dataclass_from(ProjectionConfig, _section(inputs, "VideoTensor"), where)
        video_out = (
            _dataclass_from(ProjectionConfig, _section(outputs, "VideoTensor"), where)
            if "VideoTensor" in outputs
            else ProjectionConfig()
        )
        if video_in.c_in is None:
            raise PraxisConfigError(f"config: {where} needs VideoTensor c_in")
        unpatch = section.get("unpatch_factor", [1, 1, 1])
        if not isinstance(unpatch, list) or len(unpatch) != 3:
            raise PraxisConfigError(f"config: {where}.unpatch_factor must have 3 entries")
        modulation_dim = section.get("modulation_dim")
        depth = section["depth"]
        dense_blocks = section["dense_blocks"]
        tap_blocks = section.get("tap_blocks", [])
        if not isinstance(depth, int) or not isinstance(dense_blocks, list) or not isinstance(tap_blocks, list):
            raise PraxisConfigError(f"config: {where} depth/dense_blocks/tap_blocks have a wrong type")
        if modulation_dim is not None and not isinstance(modulation_dim, int):
            raise PraxisConfigError(f"config: {where}.modulation_dim must be an int")
        return cls(
            depth=depth,
            dense_blocks=tuple(int(i) for i in dense_blocks),
            tap_blocks=tuple(int(i) for i in tap_blocks),
            text_in=text_in,
            video_in=video_in,
            video_out=video_out,
            modulation_dim=modulation_dim,
            unpatch_factor=(int(unpatch[0]), int(unpatch[1]), int(unpatch[2])),
            positive_zero_residual_add=bool(section.get("positive_zero_residual_add", True)),
        )


@dataclass(frozen=True)
class BackboneConfig:
    """``backbone``: the two-tower video transformer."""

    dim: int
    timestep_dim: int
    num_registers: int
    attention: AttentionConfig
    rope: RopeConfig
    moe: MoeSectionConfig
    dense_ffn_hidden: int
    encoder: TowerConfig
    decoder: TowerConfig
    modulation_dtype: Literal["bfloat16", "float32", "mixed"] = "mixed"
    norm_eps: float = 1e-6
    norm_compute_dtype: str = "float32"
    layout: Literal["chunk_causal"] = "chunk_causal"

    @classmethod
    def from_json(cls, section: Mapping[str, JsonValue]) -> BackboneConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(section) - known)
        if unknown:
            raise PraxisConfigError(f"config: unknown keys in backbone: {unknown}")
        scalars = {k: v for k, v in section.items() if k not in ("attention", "rope", "moe", "encoder", "decoder")}
        for name in ("dim", "timestep_dim", "num_registers", "dense_ffn_hidden"):
            if not isinstance(scalars.get(name), int):
                raise PraxisConfigError(f"config: backbone.{name} must be an int")
        if scalars.get("layout", "chunk_causal") != "chunk_causal":
            raise PraxisConfigError(f"config: backbone.layout must be 'chunk_causal', got {scalars['layout']!r}")
        if scalars.get("modulation_dtype", "mixed") not in ("bfloat16", "float32", "mixed"):
            raise PraxisConfigError(f"config: unknown backbone.modulation_dtype {scalars['modulation_dtype']!r}")
        return cls(
            attention=_dataclass_from(AttentionConfig, _section(section, "attention"), "backbone.attention"),
            rope=_dataclass_from(RopeConfig, _section(section, "rope"), "backbone.rope"),
            moe=_dataclass_from(MoeSectionConfig, _section(section, "moe"), "backbone.moe"),
            encoder=TowerConfig.from_json(_section(section, "encoder"), "backbone.encoder"),
            decoder=TowerConfig.from_json(_section(section, "decoder"), "backbone.decoder"),
            **scalars,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ExpertRopeConfig:
    """``action_expert.rope``: the expert reuses the trunk rotary positions."""

    kind: Literal["trunk"]
    dim: int
    effective_dim: int
    base: int
    table_size: int
    order: Literal["before_norm"]


@dataclass(frozen=True)
class FlexKernelOptions:
    """``action_expert.flex_kernel_options``: pinned flex-attention template parameters."""

    fwd_BLOCK_M: int
    fwd_BLOCK_N: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class ActionExpertConfig:
    """``action_expert``: the small transformer that denoises the actions."""

    dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    trunk_dim: int
    action_patch_dim: int
    proprio_patch_dim: int
    num_future_per_chunk: int
    max_chunks: int
    max_action_tokens: int
    rope: ExpertRopeConfig
    global_order: tuple[str, ...] = ("register", "anchor", "text")
    tap_norm: bool = True
    clean_only_conditioning: bool = False
    rope_cross: bool = False
    rope_self: bool = False
    rope_base: float = 100.0
    absolute_positions: bool = True
    action_pos_table: bool = False
    future_chunk_offset: Literal["zero"] = "zero"
    attention: Literal["sdpa", "flex_dynamic"] = "sdpa"
    flex_kernel_options: FlexKernelOptions | None = None
    head_plus_zero: bool = False
    layout_stamp: int | None = None

    @classmethod
    def from_json(cls, section: Mapping[str, JsonValue]) -> ActionExpertConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(section) - known)
        if unknown:
            raise PraxisConfigError(f"config: unknown keys in action_expert: {unknown}")
        if "rope" not in section:
            raise PraxisConfigError("config: action_expert.rope is required (trunk rope layout)")
        scalars = {k: v for k, v in section.items() if k not in ("rope", "flex_kernel_options", "global_order")}
        options = section.get("flex_kernel_options")
        global_order = section.get("global_order", ["register", "anchor", "text"])
        if not isinstance(global_order, list):
            raise PraxisConfigError("config: action_expert.global_order must be a list")
        cfg = cls(
            rope=_dataclass_from(ExpertRopeConfig, _section(section, "rope"), "action_expert.rope"),
            flex_kernel_options=(
                None
                if options is None
                else _dataclass_from(FlexKernelOptions, _section(section, "flex_kernel_options"), "flex_kernel_options")
            ),
            global_order=tuple(str(k) for k in global_order),
            **scalars,  # type: ignore[arg-type]
        )
        if not cfg.absolute_positions or cfg.action_pos_table or cfg.rope_cross or cfg.rope_self:
            raise PraxisConfigError("config: only the single-arm trunk-rope action expert is supported")
        if sorted(cfg.global_order) != ["anchor", "register", "text"]:
            raise PraxisConfigError("config: action_expert.global_order must permute register, anchor, text")
        if cfg.num_layers % 2:
            raise PraxisConfigError("config: action_expert.num_layers must be even")
        if cfg.rope.dim != cfg.head_dim:
            raise PraxisConfigError("config: action_expert.rope.dim must equal head_dim")
        return cfg


@dataclass(frozen=True)
class ChunkingConfig:
    """``chunking``: the window geometry in raw frames and latents."""

    n_chunks: int
    action_chunk: int
    action_frames_per_token: int
    latents_per_chunk: int
    past_frames: int
    future_frames: int
    window_frames: int
    chunk_geometry: tuple[tuple[int, int], ...]
    grid_hz: float
    global_frame_anchor: bool = True
    contiguous_chunks: bool = True
    context_format: str = "vxt"
    proprio_frames: int = 1

    @classmethod
    def from_json(cls, section: Mapping[str, JsonValue]) -> ChunkingConfig:
        geometry = section.get("chunk_geometry")
        if not isinstance(geometry, list):
            raise PraxisConfigError("config: chunking.chunk_geometry must be a list")
        rest = {k: v for k, v in section.items() if k != "chunk_geometry"}
        geom = tuple((int(g[0]), int(g[1])) for g in geometry)  # type: ignore[index]
        return _dataclass_from(cls, {**rest, "chunk_geometry": list(geom)}, "chunking")


@dataclass(frozen=True)
class VideoConfig:
    """``video``: the canvas of the camera views and the VAE geometry."""

    height: int
    width: int
    views: tuple[str, ...]
    layout: str
    tile_bounds_rows_cols: dict[str, tuple[int, int, int, int]]
    vae: str
    vae_encode_chunk_frames: int
    latent_channels: int
    vae_stride: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    tile_height: int = 256
    tile_width: int = 416
    canvas: str = "tiles"
    keep_aspect: bool = True
    antialias: bool = False
    aspect_tol: float = 0.03
    vae_encode_impl: str = "megakernel"
    vae_megakernel_chunk_frames: int | None = None

    @classmethod
    def from_json(cls, section: Mapping[str, JsonValue]) -> VideoConfig:
        bounds = section.get("tile_bounds_rows_cols", {})
        if not isinstance(bounds, Mapping):
            raise PraxisConfigError("config: video.tile_bounds_rows_cols must be an object")
        rest = {k: v for k, v in section.items() if k != "tile_bounds_rows_cols"}
        cfg = _dataclass_from(cls, {**rest, "tile_bounds_rows_cols": {}}, "video")
        tiles = {str(k): (int(v[0]), int(v[1]), int(v[2]), int(v[3])) for k, v in bounds.items()}  # type: ignore[index]
        return cls(**{**cfg.__dict__, "tile_bounds_rows_cols": tiles})


@dataclass(frozen=True)
class TextEncoderConfig:
    """``text_encoder``: the frozen Qwen text encoder and the unconditional caption."""

    model: str
    hidden_layer: int
    pad_to_multiple_of: int
    trim_padding: bool
    hidden_dim: int
    uncond: Literal["blank_caption"] = "blank_caption"
    blank_caption: str = " "


@dataclass(frozen=True)
class ActionSpaceConfig:
    """``action_space``: the per-frame action row layout."""

    dim: int
    layout: str
    pose_layout: str


@dataclass(frozen=True)
class NormalizationConfig:
    """``normalization``: the q01/q99 action statistics and the proprio mean/std."""

    domain: str
    action_q01: tuple[float, ...]
    action_q99: tuple[float, ...]
    q99_normed_dims: tuple[int, ...]
    proprio_mean: tuple[float, ...]
    proprio_std: tuple[float, ...]
    action_formula: str
    proprio_formula: str


@dataclass(frozen=True)
class PraxisConfig:
    """The parsed ``config.json`` of a Praxis single-arm checkpoint."""

    backbone: BackboneConfig
    action_expert: ActionExpertConfig
    chunking: ChunkingConfig
    video: VideoConfig
    text_encoder: TextEncoderConfig
    action_space: ActionSpaceConfig
    normalization: NormalizationConfig
    format_version: int = FORMAT_VERSION
    model_type: str = MODEL_TYPE
    embodiment: Literal["single_arm"] = "single_arm"
    torch_dtype: str = "bfloat16"
    fp32_tensors: tuple[str, ...] = ("*.ffn.expert_bias",)
    pose_frame: Literal["flange"] = "flange"

    @classmethod
    def from_json(cls, config: Mapping[str, JsonValue]) -> PraxisConfig:
        """Parse and validate a checkpoint config document."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(config) - known)
        if unknown:
            raise PraxisConfigError(f"config: unknown top-level keys: {unknown}")
        if config.get("model_type", MODEL_TYPE) != MODEL_TYPE:
            raise PraxisConfigError(f"config: model_type must be {MODEL_TYPE!r}")
        if config.get("format_version") != FORMAT_VERSION:
            raise PraxisConfigError(f"config: format_version must be {FORMAT_VERSION}")
        if config.get("embodiment") != "single_arm":
            raise PraxisConfigError('config: only "embodiment": "single_arm" is supported')
        fp32 = config.get("fp32_tensors", ["*.ffn.expert_bias"])
        if not isinstance(fp32, list):
            raise PraxisConfigError("config: fp32_tensors must be a list")
        return cls(
            backbone=BackboneConfig.from_json(_section(config, "backbone")),
            action_expert=ActionExpertConfig.from_json(_section(config, "action_expert")),
            chunking=ChunkingConfig.from_json(_section(config, "chunking")),
            video=VideoConfig.from_json(_section(config, "video")),
            text_encoder=_dataclass_from(TextEncoderConfig, _section(config, "text_encoder"), "text_encoder"),
            action_space=_dataclass_from(ActionSpaceConfig, _section(config, "action_space"), "action_space"),
            normalization=_dataclass_from(NormalizationConfig, _section(config, "normalization"), "normalization"),
            format_version=FORMAT_VERSION,
            model_type=MODEL_TYPE,
            embodiment="single_arm",
            torch_dtype=str(config.get("torch_dtype", "bfloat16")),
            fp32_tensors=tuple(str(p) for p in fp32),
            pose_frame="flange",
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> PraxisConfig:
        """Read and parse ``<path>/config.json``."""
        document = json.loads((Path(path) / "config.json").read_text())
        if not isinstance(document, dict):
            raise PraxisConfigError("config.json must hold an object")
        return cls.from_json(document)

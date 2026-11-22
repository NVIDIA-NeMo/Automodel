# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint conversion for the released Muse Spark Open dump."""

from __future__ import annotations

import re

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig

_RAW_EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.feed_forward\.experts\.(w13_weight|w2_weight)$")
_NATIVE_EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.feed_forward\.experts\.(gate_and_up_projs|down_projs)$")
_RAW_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_REDUNDANT_VISION_PREFIX = "modalities.vision_perception.encoder.vision_model."
_LEGACY_VISION_PREFIX = "modalities.vision_perception.encoder.vision_module."
_AUDIO_PREFIX = "modalities.soft_audio_perception.interaction_encoder."
_VISION_PROJECTION = "modalities.vision_perception.encoder.vision_projection.weight"


def _raw_to_native_key(key: str) -> str | None:
    """Map one serialized checkpoint key to the native module hierarchy."""
    if key.startswith((_REDUNDANT_VISION_PREFIX, _LEGACY_VISION_PREFIX)):
        return None
    if key == _VISION_PROJECTION:
        return "vision_projection.weight"
    if key.startswith(_AUDIO_PREFIX):
        return "audio_encoder." + key.removeprefix(_AUDIO_PREFIX)
    expert = _RAW_EXPERT_RE.fullmatch(key)
    if expert:
        layer, projection = expert.groups()
        native_projection = "gate_and_up_projs" if projection == "w13_weight" else "down_projs"
        return f"model.layers.{layer}.feed_forward.experts.{native_projection}"
    key = key.replace(".feed_forward.router.weight", ".feed_forward.gate.weight")
    key = key.replace(".feed_forward.expert_bias", ".feed_forward.gate.e_score_correction_bias")
    key = key.replace(".feed_forward.pooler.weight", ".feed_forward.fc1_latent_proj.weight")
    key = key.replace(".feed_forward.unpooler.weight", ".feed_forward.fc2_latent_proj.weight")
    return key


def _native_to_raw_key(key: str) -> str | None:
    """Map one native tensor key to the released checkpoint hierarchy."""
    if key == "model.rotary_emb.inv_freq":
        return None
    if key == "vision_projection.weight":
        return _VISION_PROJECTION
    if key.startswith("audio_encoder."):
        return _AUDIO_PREFIX + key.removeprefix("audio_encoder.")
    expert = _NATIVE_EXPERT_RE.fullmatch(key)
    if expert:
        layer, projection = expert.groups()
        raw_projection = "w13_weight" if projection == "gate_and_up_projs" else "w2_weight"
        return f"model.layers.{layer}.feed_forward.experts.{raw_projection}"
    key = key.replace(".feed_forward.gate.weight", ".feed_forward.router.weight")
    key = key.replace(".feed_forward.gate.e_score_correction_bias", ".feed_forward.expert_bias")
    key = key.replace(".feed_forward.fc1_latent_proj.weight", ".feed_forward.pooler.weight")
    key = key.replace(".feed_forward.fc2_latent_proj.weight", ".feed_forward.unpooler.weight")
    return key


class MuseSparkStateDictAdapter(StateDictAdapter):
    """Convert raw grouped Muse experts and multimodal namespaces to AutoModel.

    The checkpoint stores expert matrices as ``[E,out,in]`` while the reusable
    grouped-GEMM implementation consumes ``[E,in,out]``. Under expert
    parallelism, conversion first selects this rank's contiguous expert range,
    transposes only that slice, and wraps the local result as an EP DTensor.
    """

    def __init__(
        self,
        config: MuseSpark1xOmniConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype

    def to_hf(
        self,
        state_dict: dict[str, torch.Tensor],
        exclude_key_regex: str | None = None,
        **kwargs: str | bool | None,
    ) -> dict[str, torch.Tensor]:
        """Convert native tensors to released Muse Spark Open checkpoint keys."""
        converted: dict[str, torch.Tensor] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn,
                tensor,
                exclude_key_regex=exclude_key_regex,
                **kwargs,
            ):
                converted[key] = value
        return converted

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: torch.Tensor, **kwargs: str | bool | None
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert a native tensor, preserving grouped expert tensors as views."""
        raw_key = _native_to_raw_key(fqn)
        exclude = kwargs.get("exclude_key_regex")
        if raw_key is None or (exclude is not None and re.search(exclude, raw_key)):
            return []
        if _NATIVE_EXPERT_RE.fullmatch(fqn) or fqn.endswith(".feed_forward.gate.weight"):
            tensor = tensor.transpose(-1, -2)
        return [(raw_key, tensor)]

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        **kwargs: str | bool | None,
    ) -> dict[str, torch.Tensor]:
        """Convert released keys and slice grouped experts for the EP owner rank."""
        del kwargs
        num_experts = self.moe_config.n_routed_experts
        start_expert, end_expert, rank = 0, num_experts, None
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(
                device_mesh,
                num_experts,
            )
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )

        converted: dict[str, torch.Tensor] = {}
        for raw_key, value in hf_state_dict.items():
            layer = _RAW_LAYER_RE.match(raw_key)
            if layer is not None and int(layer.group(1)) >= self.config.num_hidden_layers:
                continue
            native_key = _raw_to_native_key(raw_key)
            if native_key is None:
                continue
            if _RAW_EXPERT_RE.fullmatch(raw_key):
                if state_dict_utils.is_dtensor(value):
                    value = value.transpose(-1, -2)
                else:
                    value = value[start_expert:end_expert].transpose(-1, -2).contiguous().to(self.dtype)
                    if device_mesh is not None:
                        value = state_dict_utils.create_dtensor_from_local(value, device_mesh, rank)
            elif raw_key.endswith(".feed_forward.router.weight"):
                value = value.transpose(-1, -2).contiguous().to(torch.float32)
            elif raw_key.endswith(".feed_forward.expert_bias"):
                value = value.to(torch.float32)
            converted[native_key] = value

        if "model.rotary_emb.inv_freq" not in converted:
            positions = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32)
            converted["model.rotary_emb.inv_freq"] = 1.0 / (
                self.config.rope_theta ** (positions / self.config.head_dim)
            )
        return converted


__all__ = ["MuseSparkStateDictAdapter"]

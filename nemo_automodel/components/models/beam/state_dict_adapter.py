# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""State-dict adapter between Hugging Face Beam checkpoints and the native grouped-experts layout."""

import re
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_submesh,
    is_dtensor,
    validate_dtensor_expert_sharding,
)
from nemo_automodel.components.quantization.nvfp4 import NVFP4_BLOCK_SIZE

# Native packed-parameter suffix (see ``NVFP4ExpertStorageMixin``) -> ModelOpt per-expert checkpoint tensor.
_NVFP4_CHECKPOINT_SUFFIX = {"_nvfp4": "weight", "_scales": "weight_scale", "_global": "weight_scale_2"}
_NVFP4_PARAM_SUFFIX = {ckpt: param for param, ckpt in _NVFP4_CHECKPOINT_SUFFIX.items()}
_NVFP4_PROJECTIONS = {"gate_and_up_projs": ("gate_proj", "up_proj"), "down_projs": ("down_proj",)}

if TYPE_CHECKING:
    from transformers.models.beam import BeamConfig


class BeamStateDictAdapter(Qwen3MoeStateDictAdapter):
    """Converts between Hugging Face Beam checkpoints and the grouped-experts native format.

    Beam stores its experts exactly like Qwen3-MoE, so the Qwen3-MoE conversion (including fused PEFT v5
    ParamWrapper LoRA export/import and low-memory DCP loading) applies unchanged. The native module tree
    mirrors the HF one, so every tensor outside the experts keeps its HF name and passes through unchanged
    (attention projections and ``gate_bias``, q/k norms, the four per-layer norms, the dense MLP,
    ``mlp.gate.weight`` and the ``mlp.gate.e_score_correction_bias`` buffer, embeddings, final norm,
    ``lm_head``). Experts differ:

    HF checkpoint (per expert, the ``qwen2_moe`` conversion mapping)::

        model.layers.{L}.mlp.experts.{E}.gate_proj.weight   [moe_inter_dim, hidden]
        model.layers.{L}.mlp.experts.{E}.up_proj.weight     [moe_inter_dim, hidden]
        model.layers.{L}.mlp.experts.{E}.down_proj.weight   [hidden, moe_inter_dim]

    Native (grouped)::

        model.layers.{L}.mlp.experts.gate_and_up_projs      [n_experts, hidden, 2 * moe_inter_dim]
        model.layers.{L}.mlp.experts.down_projs             [n_experts, moe_inter_dim, hidden]

    ``to_hf`` therefore writes the per-expert layout that ``transformers`` itself produces with
    ``save_pretrained``, not the stacked in-memory layout.

    With ``set_expert_storage_format("nvfp4")`` (selected by ``apply_nvfp4_to_moe_experts`` for experts built
    on meta) the routed experts are instead mapped onto the packed parameters of ``NVFP4ExpertStorageMixin``
    without ever materializing them in floating point. A ModelOpt NVFP4 export stores each expert projection
    as ``weight`` (``uint8`` fp4-e2m1 codes, two per byte, [out, in / 2]), ``weight_scale`` (``float8_e4m3fn``,
    one per 16 columns, [out, in / 16]) and ``weight_scale_2`` (``float32`` per-tensor scale, 0-dim); the
    inference-only ``input_scale`` is dropped. Natively, gate and up rows are concatenated (gate first) and the
    experts stacked::

        model.layers.{L}.mlp.experts.gate_and_up_projs_nvfp4    uint8          [n_experts, 2 * moe_inter_dim, hidden / 2]
        model.layers.{L}.mlp.experts.gate_and_up_projs_scales   float8_e4m3fn  [n_experts, 2 * moe_inter_dim, hidden / 16]
        model.layers.{L}.mlp.experts.gate_and_up_projs_global   float32        [n_experts, 2 * moe_inter_dim]
        model.layers.{L}.mlp.experts.down_projs_nvfp4           uint8          [n_experts, hidden, moe_inter_dim / 2]
        model.layers.{L}.mlp.experts.down_projs_scales          float8_e4m3fn  [n_experts, hidden, moe_inter_dim / 16]
        model.layers.{L}.mlp.experts.down_projs_global          float32        [n_experts, hidden]

    The per-tensor scale is kept once per output row (``*_global``) so the fused gate/up parameter carries the
    two projections' scales exactly. During a checkpoint load the codes and block scales are exposed as
    per-expert views into the model's own storage (DCP writes the checkpoint bytes straight through them,
    including the EP-sharded path), and the 0-dim ``weight_scale_2`` entries are read into scalars that
    ``from_hf`` expands per row.
    """

    def __init__(
        self,
        config: "BeamConfig",
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
        expert_storage_format: str = "bf16",
    ) -> None:
        super().__init__(config, moe_config, backend, dtype)
        self.expert_storage_format = "bf16"
        self.set_expert_storage_format(expert_storage_format)

    # ------------------------------------------------------------------
    # Expert storage format
    # ------------------------------------------------------------------

    def set_expert_storage_format(self, storage_format: str) -> None:
        """Select the routed-expert storage emitted during checkpoint loading (``"bf16"`` or ``"nvfp4"``)."""
        if storage_format not in ("bf16", "nvfp4"):
            raise ValueError(f"Unsupported expert storage format: {storage_format}")
        self.expert_storage_format = storage_format

    @property
    def _nvfp4_expert_dims(self) -> dict[str, tuple[int, int]]:
        """Per-projection checkpoint shape ``[out, in]`` of the routed experts (before packing)."""
        hidden, inter = self.moe_config.expert_dim, self.moe_config.moe_inter_dim
        return {"gate_proj": (inter, hidden), "up_proj": (inter, hidden), "down_proj": (hidden, inter)}

    # ------------------------------------------------------------------
    # to_hf: grouped packed parameters -> per-expert ModelOpt tensors
    # ------------------------------------------------------------------

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert one native tensor to the HF layout, splitting packed NVFP4 experts per expert.

        Packed expert parameters (``*_nvfp4`` / ``*_scales`` / ``*_global``) only exist on NVFP4-resident
        modules, so they are mapped regardless of the selected storage format; everything else follows the
        Qwen3-MoE conversion.
        """
        packed = self._convert_nvfp4_expert_to_hf(fqn, tensor, **kwargs)
        if packed is None:
            return super().convert_single_tensor_to_hf(fqn, tensor, **kwargs)
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if exclude_key_regex:
            packed = [(k, v) for k, v in packed if not re.match(exclude_key_regex, k)]
        return packed

    def _convert_nvfp4_expert_to_hf(
        self, fqn: str, tensor: Any, *, for_checkpoint_load: bool = False, **kwargs
    ) -> list[tuple[str, Any]] | None:
        """Split one grouped packed expert parameter into per-expert ModelOpt checkpoint tensors.

        Args:
            fqn: Native parameter name; returns ``None`` unless it is a packed expert parameter.
            tensor: Grouped tensor ``[experts, ...]`` in checkpoint orientation (gate rows before up rows for the
                fused projection). An EP DTensor contributes only the experts local to this rank.
            for_checkpoint_load: Emit DCP destinations: the codes and block scales are views into ``tensor``
                (registered as loaded in place when it is model storage, i.e. a DTensor or a CUDA tensor) and
                each ``weight_scale_2`` is a fresh 0-dim tensor that ``from_hf`` expands per row.
            **kwargs: Other export options, ignored here.

        Returns:
            ``(hf_key, tensor)`` pairs: ``weight`` ``uint8`` [out, in / 2], ``weight_scale`` ``float8_e4m3fn``
            [out, in / 16] and ``weight_scale_2`` ``float32`` 0-dim per expert and projection.
        """
        expert_segment = self._expert_path_segment
        match = re.search(
            rf"layers\.(\d+)\.{re.escape(expert_segment)}\.(gate_and_up_projs|down_projs)(_nvfp4|_scales|_global)$",
            fqn,
        )
        if match is None:
            return None
        layer_num, base, param_suffix = match.groups()
        n_experts = self.moe_config.n_routed_experts
        inter = self.moe_config.moe_inter_dim
        if is_dtensor(tensor):
            validate_dtensor_expert_sharding(tensor, n_experts, fqn)
        splits = self._split_experts_weights(tensor, n_experts)
        model_storage = is_dtensor(tensor) or (
            isinstance(tensor, torch.Tensor) and tensor.is_cuda and not tensor.is_meta
        )
        if for_checkpoint_load and param_suffix != "_global" and model_storage:
            self._register_inplace_loaded_key(fqn, None)

        checkpoint_suffix = _NVFP4_CHECKPOINT_SUFFIX[param_suffix]
        projections = _NVFP4_PROJECTIONS[base]
        result = []
        for i, expert_tensor in enumerate(splits):
            expert_id = self._last_expert_ids[i]
            pieces = expert_tensor.split(inter, dim=0) if len(projections) == 2 else (expert_tensor,)
            for projection, piece in zip(projections, pieces):
                key = (
                    f"{self._hf_prefix}layers.{layer_num}.{expert_segment}.{expert_id}.{projection}.{checkpoint_suffix}"
                )
                if param_suffix == "_global":
                    if for_checkpoint_load:
                        value = torch.empty((), dtype=torch.float32, device=piece.device)
                    else:
                        value = self._nvfp4_tensor_scale(piece, key)
                else:
                    value = piece if for_checkpoint_load else piece.contiguous()
                result.append((key, value))
        return result

    @staticmethod
    def _nvfp4_tensor_scale(row_scales: torch.Tensor, key: str) -> torch.Tensor:
        """Collapse one projection's per-row scales ``[out]`` to the 0-dim per-tensor scale ModelOpt stores.

        Shape-only (meta) tensors, as passed by ``get_hf_state_dict_keys`` before sharding, skip the value check.
        """
        if not row_scales.is_meta and not torch.equal(row_scales, row_scales[:1].expand_as(row_scales)):
            raise ValueError(
                f"{key}: the per-row tensor scales differ, so this projection has no single ModelOpt "
                "weight_scale_2 (only experts loaded from a ModelOpt NVFP4 checkpoint can be exported)"
            )
        return row_scales[0].clone()

    # ------------------------------------------------------------------
    # from_hf: per-expert ModelOpt tensors -> grouped packed parameters
    # ------------------------------------------------------------------

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert an HF checkpoint to the native format.

        With NVFP4 expert storage the per-expert packed tensors are aggregated first (see
        ``_aggregate_nvfp4_experts``); every other key, including LoRA adapters, follows the Qwen3-MoE path.
        """
        if self.expert_storage_format != "nvfp4":
            return super().from_hf(hf_state_dict, device_mesh=device_mesh, **kwargs)
        for key in hf_state_dict:
            if f".{self._expert_path_segment}." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break
        packed, view_loaded = self._aggregate_nvfp4_experts(hf_state_dict, device_mesh)
        state_dict = super().from_hf(hf_state_dict, device_mesh=device_mesh, **kwargs)
        state_dict.update(packed)
        if view_loaded:
            self._inplace_loaded_native_keys -= view_loaded
            self._view_loaded_native_keys = self.view_loaded_native_keys | view_loaded
        return state_dict

    def _aggregate_nvfp4_experts(
        self, hf_state_dict: dict[str, Any], device_mesh: DeviceMesh | None
    ) -> tuple[dict[str, Any], set[str]]:
        """Pop the per-expert NVFP4 tensors of this rank's experts and stack them into the packed parameters.

        Args:
            hf_state_dict: Checkpoint mapping; every routed-expert ``weight`` / ``weight_scale`` /
                ``weight_scale_2`` / ``input_scale`` entry is removed from it.
            device_mesh: Optional expert-parallel mesh selecting the local experts.

        Returns:
            The packed native entries (``*_nvfp4`` ``uint8``, ``*_scales`` ``float8_e4m3fn``, ``*_global``
            ``float32`` with the per-tensor scale repeated per output row; gate rows before up rows), wrapped as
            EP DTensors when a mesh is given, and the native keys whose destinations DCP already filled in place.

        Raises:
            RuntimeError: If a required expert tensor is missing.
            ValueError: If a tensor does not have the ModelOpt NVFP4 dtype or shape.
        """
        n_experts = self.moe_config.n_routed_experts
        expert_segment = self._expert_path_segment
        pattern = re.compile(
            rf"^(?P<prefix>(?:model\.)?(?:language_model\.)?)layers\.(?P<layer>\d+)\.{re.escape(expert_segment)}\."
            rf"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<suffix>weight|weight_scale|weight_scale_2|input_scale)$"
        )
        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert, rank = 0, n_experts, None
        local_experts = range(start_expert, end_expert)
        inplace_loaded: set[str] = getattr(self, "_inplace_loaded_native_keys", None) or set()

        # native grouped key -> expert id -> (projection, checkpoint suffix) -> tensor
        groups: dict[str, dict[int, dict[tuple[str, str], torch.Tensor]]] = {}
        view_loaded: set[str] = set()
        for key in list(hf_state_dict.keys()):
            match = pattern.match(key)
            if match is None:
                continue
            value = hf_state_dict.pop(key)
            suffix = match.group("suffix")
            if suffix == "input_scale":
                continue
            base = "down_projs" if match.group("proj") == "down_proj" else "gate_and_up_projs"
            native_key = f"{match.group('prefix')}layers.{match.group('layer')}.{expert_segment}.{base}"
            if native_key + _NVFP4_PARAM_SUFFIX[suffix] in inplace_loaded:
                # DCP wrote the checkpoint bytes through the view; the model already holds them.
                view_loaded.add(native_key + _NVFP4_PARAM_SUFFIX[suffix])
                del value
                continue
            expert_id = int(match.group("expert"))
            if start_expert <= expert_id < end_expert:
                groups.setdefault(native_key, {}).setdefault(expert_id, {})[(match.group("proj"), suffix)] = value

        out: dict[str, Any] = {}
        for native_key, experts in groups.items():
            base = native_key.rsplit(".", 1)[1]
            projections = _NVFP4_PROJECTIONS[base]
            # Entries whose destinations DCP filled in place are already in the model: only the rest are rebuilt.
            suffixes = [s for s in _NVFP4_PARAM_SUFFIX if native_key + _NVFP4_PARAM_SUFFIX[s] not in view_loaded]
            needed = [
                (expert_id, projection, suffix)
                for expert_id in local_experts
                for projection in projections
                for suffix in suffixes
                if (projection, suffix) not in experts.get(expert_id, {})
            ]
            if needed:
                missing = [
                    f"{native_key[: -len(base)]}{expert_id}.{projection}.{suffix}"
                    for expert_id, projection, suffix in needed
                ]
                raise RuntimeError(
                    f"NVFP4 expert tensors missing from checkpoint{f' (rank {rank})' if rank is not None else ''}: "
                    f"{len(missing)} of {len(local_experts) * len(projections) * len(suffixes)} required "
                    f"tensors not found for {native_key}. First few: {missing[:5]}"
                )
            stacks: dict[str, list[torch.Tensor]] = {suffix: [] for suffix in suffixes}
            for expert_id in local_experts:
                for suffix in suffixes:
                    pieces = [
                        self._nvfp4_checkpoint_tensor(
                            f"{native_key[: -len(base)]}{expert_id}.{projection}.{suffix}",
                            projection,
                            suffix,
                            experts[expert_id][(projection, suffix)],
                        )
                        for projection in projections
                    ]
                    stacks[suffix].append(torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0])
                del experts[expert_id]
            for suffix, pieces in stacks.items():
                out[native_key + _NVFP4_PARAM_SUFFIX[suffix]] = create_dtensor_from_local(
                    torch.stack(pieces, dim=0), device_mesh, rank
                )
            del stacks
        return out, view_loaded

    def _nvfp4_checkpoint_tensor(self, key: str, projection: str, suffix: str, value: torch.Tensor) -> torch.Tensor:
        """Validate one ModelOpt NVFP4 checkpoint tensor and return it in the native per-expert layout.

        Returns:
            ``weight`` as ``uint8`` [out, in / 2], ``weight_scale`` as ``float8_e4m3fn`` [out, in / 16] and
            ``weight_scale_2`` as ``float32`` [out] (the per-tensor scale repeated per output row).
        """
        out_dim, in_dim = self._nvfp4_expert_dims[projection]
        tensor = value.to_local() if is_dtensor(value) else value
        if suffix == "weight":
            if tensor.dtype not in (torch.uint8, torch.int8):
                raise ValueError(f"{key}: expected uint8 packed fp4 codes, got {tensor.dtype}")
            expected = (out_dim, in_dim // 2)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{key}: packed shape {tuple(tensor.shape)} does not match the configured {expected} "
                    f"(moe_inter_dim={self.moe_config.moe_inter_dim}, expert_dim={self.moe_config.expert_dim})"
                )
            return tensor.view(torch.uint8)
        if suffix == "weight_scale":
            if tensor.dtype != torch.float8_e4m3fn:
                raise ValueError(f"{key}: expected float8_e4m3fn block scales, got {tensor.dtype}")
            expected = (out_dim, in_dim // NVFP4_BLOCK_SIZE)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{key}: block-scale shape {tuple(tensor.shape)} does not match {expected} "
                    f"(one scale per {NVFP4_BLOCK_SIZE} columns)"
                )
            return tensor
        if tensor.numel() != 1:
            raise ValueError(f"{key}: expected a scalar per-tensor scale, got shape {tuple(tensor.shape)}")
        scale = tensor.reshape(()).to(torch.float32)
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError(f"{key}: the per-tensor scale must be finite and positive, got {scale.item()}")
        return scale.expand(out_dim).contiguous()

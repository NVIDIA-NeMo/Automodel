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

"""Native names and bounded E4M3 reconstruction using existing DCP load parts.

Matrices retain separate gate/up projections and individual expert identities.
Grouped expert storage exports the same released identities through transposed views.
"""

import re
from collections.abc import Callable, Iterator
from functools import partial
from itertools import islice

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart, StateDictAdapter
from nemo_automodel.components.moe import state_dict_utils

_NAMES = (
    ("tok_embeddings.", "model.embed_tokens."),
    ("output.", "lm_head."),
    ("lerope.", "model.rotary_emb."),
    ("norm.", "model.norm."),
)
_LAYER_NAMES = (
    ("attention_norm.", "input_layernorm."),
    ("ffn_norm.", "post_attention_layernorm."),
    ("attention.indexer.ssmax_scaling.", "self_attn.indexer.ssmax."),
    ("attention.ssmax_scaling.", "self_attn.ssmax."),
    ("attention.wq.", "self_attn.q_proj."),
    ("attention.wkv_a_with_mqa.", "self_attn.kv_a_proj_with_mqa."),
    ("attention.kv_a_norm.", "self_attn.kv_a_layernorm."),
    ("attention.wkv_b.", "self_attn.kv_b_proj."),
    ("attention.wo.", "self_attn.o_proj."),
    ("attention.indexer.", "self_attn.indexer."),
    ("feed_forward.", "mlp."),
    ("shared_experts.", "mlp.shared_experts."),
    ("experts.", "mlp.experts."),
    ("gate.", "mlp.gate."),
)
_QUANTIZED = re.compile(
    r"layers\.\d+\.(?:attention\.(?:wq|wo|indexer\.wq_b)|(?:feed_forward|shared_experts|experts\.\d+)\.w[123])\.weight$"
)
_AUXILIARY_PREFIXES = (
    "vision_encoder.",
    "vision_language_adapter.",
    "patch_merger.",
    "pre_mm_projector_norm.",
    "mtp_dspark.",
)


def _native_key(key: str) -> str:
    """Map a released text tensor name to its model-owned native parameter."""
    if key.startswith("base_model.model."):
        return "base_model.model." + _native_key(key.removeprefix("base_model.model."))
    if key.startswith(_AUXILIARY_PREFIXES):
        return key
    for old, new in _NAMES:
        if key.startswith(old):
            return new + key[len(old) :]
    match = re.fullmatch(r"layers\.(\d+)\.(.+)", key)
    if match is None:
        raise ValueError(f"Unknown Mistral Large 4 checkpoint key: {key}")
    tail = match[2]
    for old, new in _LAYER_NAMES:
        if tail.startswith(old):
            tail = new + tail[len(old) :]
            break
    else:
        raise ValueError(f"Unknown Mistral Large 4 layer key: {key}")
    tail = re.sub(
        r"(^mlp(?:\.shared_experts|\.experts\.\d+)?)\.w([123])\.",
        lambda m: m[1] + "." + {"1": "gate_proj", "2": "down_proj", "3": "up_proj"}[m[2]] + ".",
        tail,
    )
    return f"model.layers.{match[1]}.{tail}"


def _released_key(key: str) -> str:
    """Reverse the text mapping without allocating or reading tensor values."""
    if key.startswith("base_model.model."):
        return "base_model.model." + _released_key(key.removeprefix("base_model.model."))
    if key.startswith(_AUXILIARY_PREFIXES):
        return key
    for old, new in _NAMES:
        if key.startswith(new):
            return old + key[len(new) :]
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", key)
    if match is None:
        raise ValueError(f"Unknown Mistral Large 4 native key: {key}")
    tail = re.sub(
        r"(^mlp(?:\.shared_experts|\.experts\.\d+)?)\.(gate_proj|down_proj|up_proj)\.",
        lambda m: m[1] + ".w" + {"gate_proj": "1", "down_proj": "2", "up_proj": "3"}[m[2]] + ".",
        match[2],
    )
    # The generic dense mlp prefix must follow shared/expert/gate mappings.
    for old, new in sorted(_LAYER_NAMES, key=lambda item: -len(item[1])):
        if tail.startswith(new):
            return f"layers.{match[1]}." + old + tail[len(new) :]
    raise ValueError(f"Unknown Mistral Large 4 native layer key: {key}")


def _local_offsets(tensor: DTensor) -> tuple[int, int]:
    """Find global coordinates of a contiguous two-dimensional DTensor shard.

    Args:
        tensor: Matrix [global_output, global_input], with regular Shard or
            Replicate placements on any mesh axes. Partial and strided shards
            are unsupported. Repeated sharding of either axis is allowed.

    Returns:
        Global output and input offsets of the rank-local matrix.
    """
    offsets, shape = [0, 0], list(tensor.shape)
    for axis, placement in enumerate(tensor.placements):
        if type(placement) not in (Shard, Replicate):
            raise ValueError(f"Unsupported Mistral FP8 placement: {placement}")
        if isinstance(placement, Shard):
            dim = placement.dim % 2
            size, offset = Shard.local_shard_size_and_offset(
                shape[dim], tensor.device_mesh.size(axis), tensor.device_mesh.get_local_rank(axis)
            )
            offsets[dim] += offset
            shape[dim] = size
    return offsets[0], offsets[1]


@torch.no_grad()
def _decode_into(weight: torch.Tensor, scale: torch.Tensor, destination: torch.Tensor) -> None:
    """Reconstruct into existing floating storage, using global scale coordinates.

    Args:
        weight: E4M3 matrix [global_output, global_input], local or a regular
            Shard/Replicate DTensor. No projection transpose has occurred.
        scale: Replicated FP32 grid [ceil(global_output/128), ceil(global_input/128)]
            on the weight's device, holding positive finite numeric multipliers.
        destination: BF16/FP32 matrix with the same global shape and placements
            as weight. Written in place without changing parameter identity.
            The local tensor may be a noncontiguous view.

    FP32 conversion and elementwise scale temporaries cover at most 1024x1024
    values each. FP8 source values are cast to FP32, scaled, then cast once
    into destination storage. Neither source input is modified.
    """
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected a two-dimensional E4M3 native weight")
    if destination.dtype not in (torch.bfloat16, torch.float32) or destination.shape != weight.shape:
        raise ValueError("FP8 destination must have the source shape and BF16 or FP32 dtype")
    expected = tuple((n + 127) // 128 for n in weight.shape)
    if isinstance(scale, DTensor) or scale.dtype != torch.float32 or tuple(scale.shape) != expected:
        raise ValueError(f"Expected replicated numeric FP32 scale grid {expected}")
    if scale.device != weight.device or destination.device != weight.device:
        raise ValueError("Weight, scale, and destination must share a device")
    if isinstance(weight, DTensor) != isinstance(destination, DTensor):
        raise ValueError("Source and destination must share DTensor placement semantics")
    if isinstance(weight, DTensor):
        if weight.placements != destination.placements or weight.device_mesh != destination.device_mesh:
            raise ValueError("Source and destination DTensor layouts differ")
        row_offset, col_offset = _local_offsets(weight)
        local, target = weight.to_local(), destination.to_local()
    else:
        row_offset, col_offset = 0, 0
        local, target = weight, destination
    if local.is_meta:
        raise ValueError("Materialize checkpoint destinations before decoding")
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError("FP8 scales must be positive and finite")
    for row in range(0, local.shape[0], 1024):
        rows = (torch.arange(row, min(row + 1024, local.shape[0]), device=local.device) + row_offset) // 128
        for col in range(0, local.shape[1], 1024):
            cols = (torch.arange(col, min(col + 1024, local.shape[1]), device=local.device) + col_offset) // 128
            chunk = local[row : row + 1024, col : col + 1024].float()
            chunk.mul_(scale[rows[:, None], cols[None, :]])
            target[row : row + 1024, col : col + 1024].copy_(chunk)


def _finish_load_group(parts: list[CheckpointLoadPart]) -> None:
    """Decode each FP8 matrix [out, in] sequentially into final model storage."""
    for part in parts:
        part.finish()


class MistralLarge4StateDictAdapter(StateDictAdapter):
    """Lossless name mapping and bounded source-FP8 initialization.

    Individual and grouped experts export the same released gate/up/down matrices. Vision tensors
    retain their native names/layouts and load directly into the conditional
    model. DSpark draft tensors keep their released names and are not loaded. Native floating save/resume never
    attaches original FP8 scales. FSDP and balanced expert-subset loading are
    supported. Pipeline peers group local destinations into equal numbers of
    nonempty load parts, retaining bounded FP8 staging.
    """

    _supports_low_memory_dcp_load = True

    def __init__(
        self, dtype: torch.dtype = torch.bfloat16, *, grouped_experts: bool = False, n_routed_experts: int = 0
    ) -> None:
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Mistral Large 4 import requires BF16 or FP32 storage")
        if grouped_experts and n_routed_experts < 1:
            raise ValueError("Grouped Mistral import requires a positive expert count")
        self.grouped_experts = grouped_experts
        self.n_routed_experts = n_routed_experts
        self.dtype = dtype
        self._grouped_load_views: dict[str, tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}

    def iter_checkpoint_load_parts(
        self, model_state_dict: dict[str, torch.Tensor], device_mesh: DeviceMesh | None = None
    ) -> Iterator[CheckpointLoadPart]:
        """Yield bounded destination groups through the existing checkpoint loader hook.

        Args:
            model_state_dict: Model names mapped to final tensors; linear weights
                use [output, input]; grouped gate/up uses [E, hidden, 2*intermediate]
                and down [E, intermediate, hidden]. Other tensors keep checkpoint axes. DTensors may shard either matrix
                axis; every rank must iterate the same number of parts. Nonquantized values
                alias final model storage. Quantized values use local FP8 staging.
            device_mesh: Existing loader argument; each DTensor carries its mesh.

        Returns:
            Parts covering every model key exactly once. PP groups at most
            ceil(local_key_count / minimum_rank_key_count) destinations per
            part, otherwise one. Each quantized destination stages its local FP8 matrices and replicated
            FP32 scale grids (one grouped parameter bank for grouped experts). Decoding is
            sequential with at most two 1024x1024 FP32 temporaries.
        """
        del device_mesh
        parts = self._iter_individual_load_parts(model_state_dict)
        if not dist.is_initialized() or dist.get_world_size() == 1:
            yield from parts
            return
        count = len(model_state_dict)
        minimum = torch.tensor(count, device=next(iter(model_state_dict.values())).device, dtype=torch.long)
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        groups = int(minimum.item())
        if groups == 0:
            raise ValueError("Mistral Large 4 cannot load a rank without parameters")
        # All ranks enter exactly `groups` DCP collectives. Each rank uses
        # one or ceil(local_tensor_count / groups) destinations per collective.
        # Without PP every rank has the same count, so each part stays one tensor.
        for index in range(groups):
            size = count // groups + int(index < count % groups)
            group = list(islice(parts, size))
            yield CheckpointLoadPart(
                {key: tensor for part in group for key, tensor in part.checkpoint_tensors.items()},
                frozenset(key for part in group for key in part.model_keys),
                frozenset(key for part in group for key in part.temporary_checkpoint_keys),
                partial(_finish_load_group, group),
            )
            del group

    def _iter_individual_load_parts(self, model_state_dict: dict[str, torch.Tensor]) -> Iterator[CheckpointLoadPart]:
        """Yield one parameter per part, staging at most one local expert bank.

        Args:
            model_state_dict: Native tensors; linear matrices [out, in], grouped
                gate/up [experts, hidden, 2*intermediate] and down
                [experts, intermediate, hidden]. EP shards axis 0 and expert
                FSDP shards axis 1. Other tensors retain checkpoint axes.

        Returns:
            Parts whose FP8 staging covers only the local parameter shard.
            Floating checkpoint destinations alias final storage; decoding writes
            into transposed views without replacing parameters. Each grouped
            parameter completes all its local experts in one loader transaction.
        """
        for model_key in sorted(model_state_dict):
            sources = self.convert_single_tensor_to_hf(model_key, model_state_dict[model_key])
            tensors, temporary, decodes = {}, set(), []
            for source_key, destination in sources:
                if _QUANTIZED.fullmatch(source_key):
                    if destination.ndim != 2 or destination.dtype not in (torch.bfloat16, torch.float32):
                        raise ValueError(f"Invalid floating matrix destination: {model_key}")
                    if isinstance(destination, DTensor):
                        _local_offsets(destination)
                    weight = torch.empty_like(destination, dtype=torch.float8_e4m3fn)
                    scale = torch.empty(
                        tuple((n + 127) // 128 for n in destination.shape),
                        device=destination.device,
                        dtype=torch.float32,
                    )
                    scale_key = source_key.removesuffix(".weight") + ".qscale_weight"
                    tensors.update({source_key: weight, scale_key: scale})
                    temporary.update((source_key, scale_key))
                    decodes.append(partial(_decode_into, weight, scale, destination))
                else:
                    tensors[source_key] = destination
            yield CheckpointLoadPart(
                tensors, frozenset({model_key}), frozenset(temporary), partial(_finish_decodes, decodes)
            )
            del tensors, decodes, sources
            if temporary:
                del weight, scale

    def from_hf(
        self, hf_state_dict: dict[str, torch.Tensor], device_mesh: DeviceMesh | None = None
    ) -> dict[str, torch.Tensor]:
        """Convert a dependency-complete source mapping without consuming it.

        Args:
            hf_state_dict: Native checkpoint tensors, with weights [output, input]
                and FP32 scale grids [ceil(output/128), ceil(input/128)]. BF16
                tensors retain checkpoint axes and may alias the returned values.
                FP8 matrices may be regular DTensor shards with global shapes.
            device_mesh: Existing loader argument; DTensors own their placements.

        Returns:
            Model names with matching axes, or fused grouped gate/up
            [E, hidden, 2*intermediate] and down [E, intermediate, hidden].
            FP8 weights and grouped banks get independent BF16/FP32 storage.
            Scale tensors are consumed only as dependencies. Grouped expert
            FP8 DTensor destinations use iter_checkpoint_load_parts. Floating
            DCP destinations created by to_hf(for_checkpoint_load=True) retain
            the original grouped storage and EP/FSDP placements after loading.
        """
        # Floating DCP destinations are views of the original grouped tensor.
        # Return that tensor with its exact EP/FSDP placement after DCP fills it;
        # rebuilding from owner-local matrices would lose topology and duplicate storage.
        pending, self._grouped_load_views = self._grouped_load_views, {}
        result = {}
        consumed = set()
        for name, (tensor, views, local_views) in pending.items():
            missing = set(views) - hf_state_dict.keys()
            if missing:
                raise ValueError(f"Incomplete Mistral grouped checkpoint destination: {sorted(missing)}")
            for key, destination in views.items():
                loaded = hf_state_dict[key]
                if loaded.shape != destination.shape or loaded.dtype != destination.dtype:
                    raise ValueError(f"Mistral grouped checkpoint destination changed shape or dtype: {key}")
                # DTensor.from_local moves CPU-offloaded expert views to the
                # CUDA mesh's device. Such destinations no longer alias the
                # source grouped bank. Install their local values explicitly.
                source = loaded.to_local() if isinstance(loaded, DTensor) else loaded
                target = local_views[key]
                if source.shape != target.shape:
                    raise ValueError(f"Mistral grouped checkpoint local shard changed shape: {key}")
                if source.device != target.device or source.data_ptr() != target.data_ptr():
                    with torch.no_grad():
                        target.copy_(source)
                consumed.add(key)
            result[name] = tensor
        used_scales = set()
        for key, tensor in hf_state_dict.items():
            if key in consumed:
                continue
            if key.endswith(".qscale_weight"):
                continue
            if tensor.dtype == torch.float8_e4m3fn:
                if not _QUANTIZED.fullmatch(key):
                    raise ValueError(f"Unexpected quantized tensor: {key}")
                scale_key = key.removesuffix(".weight") + ".qscale_weight"
                if scale_key not in hf_state_dict:
                    raise ValueError(f"Missing FP8 scale: {scale_key}")
                decoded = torch.empty_like(tensor, dtype=self.dtype)
                _decode_into(tensor, hf_state_dict[scale_key], decoded)
                used_scales.add(scale_key)
                tensor = decoded
            result[_native_key(key)] = tensor
        orphaned = {k for k in hf_state_dict if k.endswith(".qscale_weight")} - used_scales
        if orphaned:
            raise ValueError(f"Orphaned scales without FP8 weights: {sorted(orphaned)}")
        return self._group_experts(result, device_mesh) if self.grouped_experts else result

    def _group_experts(self, state: dict[str, torch.Tensor], device_mesh: DeviceMesh | None) -> dict[str, torch.Tensor]:
        """Fuse complete per-layer expert banks for the in-memory import path.

        Args:
            state: Floating native matrices [out, in]. Expert inputs must be
                complete unsharded matrices; the streaming loader supports
                sharded destinations without making this in-memory copy.
            device_mesh: Optional MoE mesh; only the EP axis participates here.

        Returns:
            New mapping with grouped [E, hidden, 2*intermediate] gate/up and
            [E, intermediate, hidden] down. Gate precedes up. Grouped storage
            is independent; other values alias state. EP uses Shard(0).
        """
        mesh = state_dict_utils.get_submesh(device_mesh, ("ep",)) if device_mesh is not None else None
        first, stop = state_dict_utils.get_expert_range_for_rank_from_mesh(mesh, self.n_routed_experts)
        pattern = re.compile(r"(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
        prefixes = {match.group(1) for key in state if (match := pattern.fullmatch(key))}
        result = {key: value for key, value in state.items() if not pattern.fullmatch(key)}
        for prefix in sorted(prefixes):
            banks = {"gate_and_up_projs": [], "down_projs": []}
            for expert in range(first, stop):
                keys = [f"{prefix}.{expert}.{projection}_proj.weight" for projection in ("gate", "up", "down")]
                if any(key not in state for key in keys):
                    raise ValueError(f"Incomplete Mistral expert bank: {prefix}, expert {expert}")
                gate, up, down = (state[key] for key in keys)
                if any(isinstance(value, DTensor) for value in (gate, up, down)):
                    raise ValueError("Use iter_checkpoint_load_parts for sharded grouped Mistral imports")
                banks["gate_and_up_projs"].append(torch.cat((gate.T, up.T), dim=-1))
                banks["down_projs"].append(down.T)
            for projection, values in banks.items():
                tensor = torch.stack(values)
                if mesh is not None:
                    tensor = DTensor.from_local(tensor, mesh, [Shard(0)], run_check=False)
                result[f"{prefix}.{projection}"] = tensor
        return result

    def to_hf(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        device_mesh: DeviceMesh | None = None,
        v4_compatible: bool = False,
        legacy_paramwrapper_layout: bool = False,
        for_checkpoint_load: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Export floating training weights with released checkpoint names.

        Args:
            state_dict: Model tensors with checkpoint axes; matrix [output, input]
                or grouped experts [E, hidden, 2*intermediate] / [E, intermediate, hidden],
                optionally EP Shard(0) and expert-FSDP Shard(1) DTensors.
            exclude_key_regex: Existing checkpoint API filter on model names.
            quantization: Unsupported; source FP8 is decoded on import, training saves floating weights.
            device_mesh: Existing export hook; DTensors carry their EP ownership.
            v4_compatible: Existing hook; released names have one layout.
            legacy_paramwrapper_layout: Existing hook; this model has no grouped ParamWrapper.
            for_checkpoint_load: Existing hook; floating destinations always alias model storage.

        Returns:
            Floating tensors alias inputs and retain their training dtype.
        """
        del device_mesh, v4_compatible, legacy_paramwrapper_layout
        if quantization:
            raise NotImplementedError("Mistral Large 4 exports floating training checkpoints only")
        selected = (
            (k, v) for k, v in state_dict.items() if exclude_key_regex is None or re.match(exclude_key_regex, k) is None
        )
        self._grouped_load_views = {}
        converted = {}
        for name, tensor in selected:
            views = dict(self.convert_single_tensor_to_hf(name, tensor, quantization=quantization))
            converted.update(views)
            if for_checkpoint_load and name.endswith((".experts.gate_and_up_projs", ".experts.down_projs")):
                self._grouped_load_views[name] = (tensor, views, self._local_grouped_checkpoint_views(name, tensor))
        return converted

    def _local_grouped_checkpoint_views(self, name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keep original local storage views even when distributed export changes device.

        Args:
            name: Native grouped gate/up or down parameter name.
            tensor: Grouped [E, hidden, 2*intermediate] or [E, intermediate, hidden]
                tensor, optionally EP/FSDP sharded and CPU offloaded.

        Returns:
            Released per-expert names to plain local [out, in] views. Their
            matrix dimensions follow the original inner-axis FSDP shard and
            their device stays unchanged; no distributed wrapper is constructed.
        """
        local, first, stop = state_dict_utils.get_expert_slice_for_rank(tensor, tensor.shape[0])
        prefix, projection = name.rsplit(".", 1)
        views = {}
        for offset, expert in enumerate(range(first, stop)):
            if projection == "gate_and_up_projs":
                gate, up = local[offset].chunk(2, dim=-1)
                matrices = (("gate_proj", gate.T), ("up_proj", up.T))
            else:
                matrices = (("down_proj", local[offset].T),)
            for matrix_name, matrix in matrices:
                views[_released_key(f"{prefix}.{expert}.{matrix_name}.weight")] = matrix
        return views

    def forced_hf_dtype_mapping(self, state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
        """Prevent export from recasting decoded weights to the source FP8 dtype.

        Args:
            state_dict: Already-exported tensors retaining checkpoint axes, matrices
                [output, input], optionally DTensors. Only dtype is inspected.

        Returns:
            Released-key dtype overrides for every floating tensor. FP32 master
            parameters remain FP32 and BF16 weights remain BF16; no re-quantization.
        """
        return {key: str(value.dtype).removeprefix("torch.") for key, value in state_dict.items()}

    def map_peft_target_module_to_hf(self, name: str, *, v4_compatible: bool = False) -> str:
        """Use the same projection mapping in adapter metadata and tensor export."""
        del v4_compatible
        return _released_key(name + ".weight").removesuffix(".weight")

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: torch.Tensor, *, quantization: bool = False
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert one floating tensor for checkpoint import or export.

        Args:
            fqn: Model parameter name.
            tensor: Floating tensor with native axes, [output, input] for matrices;
                grouped gate/up [E, hidden, 2*intermediate], down [E, intermediate, hidden].
                EP shards axis 0; expert FSDP shards axis 1. Splits/transposes alias storage.
                Floating exports preserve DTensor layouts.
            quantization: Unsupported; exports preserve the floating training dtype.

        Returns:
            Native (name, tensor) pairs whose values alias the input.
        """
        if quantization:
            raise NotImplementedError("Mistral Large 4 exports floating training checkpoints only")
        grouped = re.fullmatch(r"(.*\.mlp\.experts)\.(gate_and_up_projs|down_projs)", fqn)
        if grouped:
            prefix, projection = grouped.groups()
            experts, identities = state_dict_utils.split_experts_weights_dtensor_aware(tensor, tensor.shape[0])
            result = []
            for expert, identity in zip(experts, identities, strict=True):
                if projection == "gate_and_up_projs":
                    gate, up = expert.chunk(2, dim=-1)
                    matrices = (("gate_proj", gate.T), ("up_proj", up.T))
                else:
                    matrices = (("down_proj", expert.T),)
                for name, matrix in matrices:
                    result.extend(
                        self.convert_single_tensor_to_hf(
                            f"{prefix}.{identity}.{name}.weight", matrix, quantization=quantization
                        )
                    )
            return result
        name = _released_key(fqn)
        return [(name, tensor)]


def _finish_decodes(decodes: list[Callable[[], None]]) -> None:
    """Decode local FP8 expert matrices sequentially into existing model views."""
    for decode in decodes:
        decode()

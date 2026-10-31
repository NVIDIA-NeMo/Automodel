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

"""State-dict adapter for the MiniMax M3 (text) backbone.

Converts between the released HF checkpoint layout and the native AutoModel
layout:

* ``block_sparse_moe.{gate,e_score_correction_bias}`` -> ``mlp.gate.*``
* ``block_sparse_moe.experts.{e}.{w1,w3,w2}`` -> grouped ``mlp.experts.*``
  (gate/up/down) via ``MoESplitExpertsStateDictMixin``
* ``block_sparse_moe.shared_experts.*`` -> ``shared_experts.*`` (a sibling of
  ``mlp`` on the decoder block)
* dense (non-MoE) layers keep ``mlp.{gate,up,down}_proj.*`` unchanged

MXFP8 weights (FP8 e4m3 with e8m0/uint8 block scales along the input dim)
are dequantized to the training dtype on load. M3.1 adds NVFP4 packed routed
experts and MXFP8 indexer projections. Its native load-part iterator writes
directly into final sharded storage while bounding temporary memory to one
grouped expert parameter or ordinary projection.
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart, StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import (
    MiniMaxM3QuantizationConfig,
    MiniMaxM3VLConfig,
    MiniMaxM3VLTextConfig,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import (
    get_expert_slice_for_rank,
    is_dtensor,
    split_experts_weights_dtensor_aware,
)

# MXFP8 block layout from config.json: weight_block_size = [1, 32] (1 row, 32 cols).
MXFP8_BLOCK_SIZE = 32
# e8m0 exponent for scale 2**0 == 1.0 (bias 127); load-time placeholder value.
_MXFP8_SCALE_INV_IDENTITY = 127

# HF-format weight keys stored as MXFP8 in the released checkpoint: attention +
# dense-MLP + (shared) expert projections. NOT quantized (no *_scale_inv): the
# router gate, norms, embed_tokens, lm_head, vision/projector. The legacy DSA
# indexer is BF16; M3.1 includes its q/k projections in MXFP8 separately.
_MXFP8_QUANT_KEY_RE = re.compile(
    r"\.layers\.\d+\.(?:"
    r"self_attn\.[qkvo]_proj"
    r"|mlp\.(?:gate|up|down)_proj"
    r"|block_sparse_moe\.experts\.\d+\.w[123]"
    r"|block_sparse_moe\.shared_experts\.(?:gate|up|down)_proj"
    r")\.weight$"
)


def _should_quantize_mxfp8_key(key: str) -> bool:
    """True for HF-format weight keys stored as MXFP8 in the checkpoint."""
    return bool(_MXFP8_QUANT_KEY_RE.search(key))


def create_mxfp8_scale_inv(weight: torch.Tensor, block_size: int = MXFP8_BLOCK_SIZE) -> torch.Tensor:
    """Load-time placeholder scale_inv (e8m0/uint8, GLOBAL shape ``[out, ceil(in/block)]``).

    Emitted by ``to_hf(quantization=True)`` so the DCP planner requests the
    checkpoint's ``*_scale_inv`` tensors; the values here are overwritten by the
    load. Kept a regular (non-DTensor) tensor with global shape -- the per-shard
    slice happens in ``dequantize_mxfp8`` (mirrors deepseek_v3).
    """
    out_dim, in_dim = weight.shape[-2], weight.shape[-1]
    n_blocks = (in_dim + block_size - 1) // block_size
    dev = weight.to_local().device if is_dtensor(weight) else weight.device
    return torch.full((out_dim, n_blocks), _MXFP8_SCALE_INV_IDENTITY, dtype=torch.uint8, device=dev)


def _slice_mxfp8_scale_for_dtensor(
    scale_inv: torch.Tensor, weight_dtensor: torch.Tensor, weight_local: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Slice a global scale_inv to a DTensor weight's local shard.

    MXFP8 block is ``[1, block_size]``: dim 0 (out) is full-resolution (block 1, so a
    row range maps 1:1) and dim 1 (in) is grouped by ``block_size``. Custom MoE is
    always tp=1, so sharding is on dim 0 (FSDP / ep_shard); dim 1 handled for safety.
    """
    from torch.distributed.tensor import Shard

    block_per_dim = (1, block_size)
    slices = [slice(None), slice(None)]
    for mesh_dim, placement in enumerate(weight_dtensor.placements):
        if isinstance(placement, Shard) and placement.dim < 2:
            sdim = placement.dim
            bs = block_per_dim[sdim]
            mesh_size = weight_dtensor.device_mesh.size(mesh_dim)
            coord = weight_dtensor.device_mesh.get_local_rank(mesh_dim=mesh_dim)
            local_size = weight_local.shape[sdim]
            global_blocks = scale_inv.shape[sdim]
            chunk = ((global_blocks * bs) + mesh_size - 1) // mesh_size
            g_start = coord * chunk
            g_end = g_start + local_size
            slices[sdim] = slice(g_start // bs, min((g_end + bs - 1) // bs, global_blocks))
    return scale_inv[slices[0], slices[1]].contiguous()


def _dequantize_mxfp8_local(w_local: torch.Tensor, scale_local: torch.Tensor, block_size: int, dtype) -> torch.Tensor:
    w = w_local.to(torch.float32)
    scale = torch.exp2(scale_local.to(torch.float32) - 127.0).repeat_interleave(block_size, dim=1)
    if scale.shape[1] != w.shape[1]:
        scale = scale[:, : w.shape[1]]
    return (w * scale).to(dtype)


def dequantize_mxfp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    block_size: int = MXFP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an MXFP8 weight (FP8 e4m3 + e8m0/uint8 block scales) to ``dtype``.

    ``weight`` is FP8 ``e4m3`` ``[out, in]``; ``scale_inv`` holds e8m0 (uint8)
    exponents ``[out, ceil(in/block_size)]`` with dequant scale for input-block
    ``b`` = ``2 ** (scale_inv[:, b] - 127)`` (MX e8m0; confirmed vs sglang). Handles
    DTensor weights: the local shard is dequantized against the matching slice of a
    global ``scale_inv`` and rewrapped with the weight's placements.
    """
    weight_is_dtensor = is_dtensor(weight)
    scale_is_dtensor = is_dtensor(scale_inv)
    w_local = weight.to_local() if weight_is_dtensor else weight
    s_local = scale_inv.to_local() if scale_is_dtensor else scale_inv
    if weight_is_dtensor and not scale_is_dtensor and s_local.shape[0] != w_local.shape[0]:
        s_local = _slice_mxfp8_scale_for_dtensor(scale_inv, weight, w_local, block_size)
    out_local = _dequantize_mxfp8_local(w_local, s_local.to(w_local.device), block_size, dtype)
    if weight_is_dtensor:
        from torch.distributed.tensor import DTensor

        return DTensor.from_local(out_local, weight.device_mesh, weight.placements)
    return out_local


_NVFP4_BLOCK_SIZE = 16
_NVFP4_EXPERT_RE = re.compile(r"\.block_sparse_moe\.experts\.\d+\.w[123]\.weight$")
_MXFP8_INDEXER_RE = re.compile(r"\.self_attn\.index_[qk]_proj\.weight$")
_NVFP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


_NVFP4_QUANTIZATION_CONFIG = MiniMaxM3QuantizationConfig(
    quant_method="mxfp8",
    weight_block_size=[1, MXFP8_BLOCK_SIZE],
    moe_quant_format="nvfp4-pack-quantized",
    group_size=_NVFP4_BLOCK_SIZE,
)


def _uses_nvfp4_experts(config: MiniMaxM3VLConfig | MiniMaxM3VLTextConfig) -> bool:
    """Recognize the released M3.1 mixed MXFP8/NVFP4 checkpoint layout."""
    quantization = config.quantization_config
    if quantization is None or "moe_quant_algo" not in quantization:
        return False
    if quantization["moe_quant_algo"] != "NVFP4":
        return False
    mismatched = [
        field
        for field, value in _NVFP4_QUANTIZATION_CONFIG.items()
        if field not in quantization or quantization[field] != value
    ]
    if mismatched:
        raise ValueError(
            "MiniMax M3 NVFP4 experts require mxfp8/nvfp4-pack-quantized "
            f"with weight_block_size=[1, 32] and group_size=16; mismatched fields: {mismatched}"
        )
    return True


def _quantized_destination(
    weight: torch.Tensor, divisor: int, dtype: torch.dtype, device: torch.device | None = None
) -> torch.Tensor:
    """Allocate a compressed checkpoint destination matching a weight shard.

    Args:
        weight: Final floating-point projection of global shape [output, input].
            A DTensor may shard either axis; every input-axis shard must align
            to divisor. Its local storage may be CPU or CUDA and noncontiguous.
        divisor: Compression factor for the input axis, such as 2, 16, or 32.
        dtype: Checkpoint storage dtype.
        device: Materialization device for a meta template. A CUDA mesh uses
            CUDA staging even when final model storage is CPU offloaded.

    Returns:
        Independent destination of global shape [output, input / divisor],
        with the same DTensor mesh and placements when applicable.
    """
    local = weight.to_local() if isinstance(weight, DTensor) else weight
    if weight.ndim != 2 or weight.shape[1] % divisor or local.shape[1] % divisor:
        raise ValueError(f"Projection and local input sizes must align to {divisor}: {weight.shape}, {local.shape}")
    if isinstance(weight, DTensor):
        if any(not isinstance(placement, (Shard, Replicate)) for placement in weight.placements):
            raise ValueError("MiniMax M3 checkpoint destinations require Shard or Replicate placements")
        for placement, mesh_size in zip(weight.placements, weight.device_mesh.shape):
            if isinstance(placement, Shard) and placement.dim == 1 and weight.shape[1] % (mesh_size * divisor):
                raise ValueError("MiniMax M3 input-axis shards must end on quantization block boundaries")
    allocation_device = device if device is not None else local.device
    if isinstance(weight, DTensor) and allocation_device.type != "meta":
        allocation_device = torch.device(weight.device_mesh.device_type)
    destination = torch.empty((local.shape[0], local.shape[1] // divisor), dtype=dtype, device=allocation_device)
    if isinstance(weight, DTensor):
        shape = torch.Size((weight.shape[0], weight.shape[1] // divisor))
        return DTensor.from_local(
            destination, weight.device_mesh, weight.placements, run_check=False, shape=shape, stride=(shape[1], 1)
        )
    return destination


def _dequantize_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode the compressed-tensors NVFP4 projection layout used by M3.1.

    Args:
        packed: Uint8 projection of shape [output, input / 2]. Even input
            columns use the low nibble, odd columns the high nibble.
        scales: E4M3 block scales of shape [output, input / 16].
        global_scale: Positive FP32 scalar in the reciprocal convention.
        dtype: Floating-point output dtype. Computation takes place in FP32.

    Returns:
        Independent tensor of shape [output, input], on packed's device.
        Values are E2M1 * (block_scale / global_scale); activation scales
        do not participate in weight dequantization. Inputs are not mutated.

    Reference:
        https://github.com/vllm-project/compressed-tensors/blob/e69c8dc58aa152e5f5e36e85800ec1d2e7de5271/src/compressed_tensors/quantization/lifecycle/forward_helpers.py
    """
    packed_local = packed.to_local() if isinstance(packed, DTensor) else packed
    scales_local = scales.to_local() if isinstance(scales, DTensor) else scales
    global_local = global_scale.to_local() if isinstance(global_scale, DTensor) else global_scale
    if (
        packed_local.dtype != torch.uint8
        or scales_local.dtype != torch.float8_e4m3fn
        or packed_local.ndim != 2
        or scales_local.ndim != 2
        or packed_local.shape[0] != scales_local.shape[0]
        or packed_local.shape[1] != scales_local.shape[1] * (_NVFP4_BLOCK_SIZE // 2)
        or global_local.numel() != 1
    ):
        raise ValueError("Invalid MiniMax M3 NVFP4 packed projection, block-scale shape/dtype, or global scalar")
    if not bool(torch.isfinite(global_local).all()) or float(global_local) <= 0:
        raise ValueError("MiniMax M3 NVFP4 weight_global_scale must be finite and positive")
    lookup = torch.tensor(_NVFP4_VALUES, dtype=torch.float32, device=packed_local.device)
    rows, packed_columns = packed_local.shape
    unpacked = torch.empty((rows, packed_columns * 2), dtype=torch.float32, device=packed_local.device)
    unpacked[:, 0::2] = lookup[(packed_local & 15).long()]
    unpacked[:, 1::2] = lookup[(packed_local >> 4).long()]
    effective_scale = scales_local.float() / global_local.to(device=packed_local.device, dtype=torch.float32)
    unpacked.view(rows, -1, _NVFP4_BLOCK_SIZE).mul_(effective_scale.unsqueeze(-1))
    result = unpacked.to(dtype)
    if isinstance(packed, DTensor):
        shape = torch.Size((packed.shape[0], packed.shape[1] * 2))
        return DTensor.from_local(
            result, packed.device_mesh, packed.placements, run_check=False, shape=shape, stride=(shape[1], 1)
        )
    return result


@dataclass
class _QuantizedProjection:
    """Temporary checkpoint tensors plus the final projection view they fill.

    Attributes:
        destination: Final floating-point [output, input] projection view.
            For experts this aliases the original rank-local CPU/CUDA storage;
            ordinary projections may retain their global DTensor placements.
        weight: FP8 [output, input] or packed uint8 [output, input / 2]
            checkpoint staging tensor, with the source DTensor placements.
        scale: Uint8 [output, input / 32] MXFP8 exponents or E4M3
            [output, input / 16] NVFP4 scales, matching weight's placements.
        global_scale: Optional replicated FP32 scalar for NVFP4. It divides
            the block scales; None denotes MXFP8.
    """

    destination: torch.Tensor
    weight: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor | None = None

    def finish(self) -> None:
        """Decode one [output, input] projection into existing local storage."""
        destination = self.destination.to_local() if isinstance(self.destination, DTensor) else self.destination
        if self.global_scale is None:
            weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
            scale = self.scale.to_local() if isinstance(self.scale, DTensor) else self.scale
            decoded = _dequantize_mxfp8_local(weight, scale, MXFP8_BLOCK_SIZE, destination.dtype)
        else:
            decoded = _dequantize_nvfp4(self.weight, self.scale, self.global_scale, destination.dtype)
            if isinstance(decoded, DTensor):
                decoded = decoded.to_local()
        with torch.no_grad():
            destination.copy_(decoded)


def _finish_quantized_projections(projections: tuple[_QuantizedProjection, ...]) -> None:
    """Decode each projection separately, bounding unpacked temporary storage."""
    for projection in projections:
        projection.finish()


class MiniMaxM3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert MiniMax M3 HF checkpoints to/from the native grouped-expert format."""

    def __init__(
        self,
        config: MiniMaxM3VLTextConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        self._nvfp4_experts = _uses_nvfp4_experts(config)

    def _checkpoint_projection_views(
        self, fqn: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """Separate checkpoint metadata from final local projection views.

        Args:
            fqn: Native model state-dict name.
            tensor: Final model tensor. Grouped experts use global
                [experts, input, 2 * intermediate] gate/up or
                [experts, intermediate, output] down layouts. EP shards the
                expert axis; FSDP may additionally shard the input axis.

        Returns:
            Tuples of HF name, metadata template, and final local destination.
            Each projection uses [output, input] layout. Expert templates are
            meta tensors preserving non-expert DTensor placements; destinations
            alias the original local CPU/CUDA storage. Keeping them separate
            avoids DTensor.from_local moving offloaded CPU parameters to CUDA.
        """
        if fqn.endswith((".experts.gate_and_up_projs", ".experts.down_projs")):
            if not self._expert_checkpoint_tensors_use_model_storage:
                raise ValueError("MiniMax M3 NVFP4 streaming requires experts backed by final grouped weight storage")
            weights, expert_ids = split_experts_weights_dtensor_aware(
                tensor.to(device="meta"), self.moe_config.n_routed_experts
            )
            local, start_expert, end_expert = get_expert_slice_for_rank(tensor, self.moe_config.n_routed_experts)
            if expert_ids != list(range(start_expert, end_expert)):
                raise ValueError("MiniMax M3 checkpoint expert metadata disagrees with local expert ownership")
            prefix, projection = fqn.rsplit(".", 1)
            projections = []
            for local_index, (weight, expert_id) in enumerate(zip(weights, expert_ids)):
                local_weight = local[local_index]
                if projection == "gate_and_up_projs":
                    intermediate = self.moe_config.moe_inter_dim
                    projections.extend(
                        [
                            (
                                self._native_key_to_hf(f"{prefix}.{expert_id}.gate_proj.weight"),
                                weight[:, :intermediate].transpose(0, 1),
                                local_weight[:, :intermediate].transpose(0, 1),
                            ),
                            (
                                self._native_key_to_hf(f"{prefix}.{expert_id}.up_proj.weight"),
                                weight[:, intermediate:].transpose(0, 1),
                                local_weight[:, intermediate:].transpose(0, 1),
                            ),
                        ]
                    )
                else:
                    projections.append(
                        (
                            self._native_key_to_hf(f"{prefix}.{expert_id}.down_proj.weight"),
                            weight.transpose(0, 1),
                            local_weight.transpose(0, 1),
                        )
                    )
            return projections
        return [
            (key, value, value)
            for key, value in self.convert_single_tensor_to_hf(fqn, tensor, for_checkpoint_load=True)
        ]

    def iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        checkpoint_metadata: Mapping[str, "TensorStorageMetadata"] | None = None,
    ) -> Iterator[CheckpointLoadPart] | None:
        """Load M3.1 mixed quantized weights into bounded model-owned parts.

        Args:
            model_state_dict: Final native storage, including global grouped
                expert tensors [experts, input, output]. DTensor expert-axis
                and input-axis shards, and CPU-offloaded local storage, are
                preserved. The iterator never creates a full model copy.
            device_mesh: Existing expert mesh; placements are read from each
                destination DTensor.
            checkpoint_metadata: Unused; part destinations derive from model shapes.

        Returns:
            One direct-load part followed by one part per quantized native
            parameter, or None for earlier BF16/MXFP8 checkpoints.
        """
        if not self._nvfp4_experts:
            return None
        return self._iter_mixed_checkpoint_parts(model_state_dict)

    def _iter_mixed_checkpoint_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        hf_prefix: str = "",
    ) -> Iterator[CheckpointLoadPart]:
        """Yield bounded M3.1 load parts over final native model storage.

        Args:
            model_state_dict: Native names and final CPU/CUDA/DTensor storage
                with layouts documented by iter_checkpoint_load_parts.
            hf_prefix: HF wrapper namespace prepended to text checkpoint keys.

        Yields:
            A direct part and per-parameter temporary parts. Each native key
            is completed once. Expert parts retain packed local projections,
            then decode and copy one projection at a time in finish().
        """
        direct = {}
        direct_model_keys = set()
        quantized_keys = []
        for fqn, tensor in model_state_dict.items():
            if fqn.endswith("_extra_state"):
                direct_model_keys.add(fqn)
                continue
            projections = self._checkpoint_projection_views(fqn, tensor)
            if any(_should_quantize_mxfp8_key(key) or _MXFP8_INDEXER_RE.search(key) for key, _, _ in projections):
                quantized_keys.append(fqn)
            else:
                direct.update((hf_prefix + key, value) for key, _, value in projections)
                direct_model_keys.add(fqn)
        if direct:
            yield CheckpointLoadPart(
                checkpoint_tensors=direct,
                model_keys=frozenset(direct_model_keys),
                temporary_checkpoint_keys=frozenset(),
                finish=lambda: None,
            )
        elif direct_model_keys:
            raise ValueError("MiniMax M3 bookkeeping tensors require at least one direct checkpoint destination")

        for fqn in quantized_keys:
            checkpoint_tensors = {}
            conversions = []
            for key, template, destination in self._checkpoint_projection_views(fqn, model_state_dict[fqn]):
                checkpoint_key = hf_prefix + key
                if _NVFP4_EXPERT_RE.search(key):
                    prefix = checkpoint_key.removesuffix(".weight")
                    packed = _quantized_destination(template, 2, torch.uint8, destination.device)
                    scales = _quantized_destination(
                        template, _NVFP4_BLOCK_SIZE, torch.float8_e4m3fn, destination.device
                    )
                    local = packed.to_local() if isinstance(packed, DTensor) else packed
                    global_scale = torch.empty((), dtype=torch.float32, device=local.device)
                    checkpoint_tensors.update(
                        {
                            prefix + ".weight_packed": packed,
                            prefix + ".weight_scale": scales,
                            prefix + ".weight_global_scale": global_scale,
                            prefix + ".input_global_scale": torch.empty((), dtype=torch.float32, device=local.device),
                        }
                    )
                    conversions.append(_QuantizedProjection(destination, packed, scales, global_scale))
                else:
                    weight = _quantized_destination(template, 1, torch.float8_e4m3fn, destination.device)
                    scales = _quantized_destination(template, MXFP8_BLOCK_SIZE, torch.uint8, destination.device)
                    checkpoint_tensors[checkpoint_key] = weight
                    checkpoint_tensors[checkpoint_key + "_scale_inv"] = scales
                    conversions.append(_QuantizedProjection(destination, weight, scales))
            yield CheckpointLoadPart(
                checkpoint_tensors=checkpoint_tensors,
                model_keys=frozenset((fqn,)),
                temporary_checkpoint_keys=frozenset(checkpoint_tensors),
                finish=partial(_finish_quantized_projections, tuple(conversions)),
            )
            del checkpoint_tensors, conversions

    @property
    def _expert_path_segment(self) -> str:
        return "mlp.experts"

    def _dequantize(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decode quantized projections in place before grouped-expert merging.

        Args:
            state_dict: HF projection weights and quantization companions.
                MXFP8 uses [output, input] FP8 weights and [output, input / 32]
                exponent scales. NVFP4 uses [output, input / 2] packed bytes,
                [output, input / 16] E4M3 scales, and an FP32 global scalar.
                Distributed tensors retain their matching mesh and placements.

        Returns:
            The same mapping, with floating-point [output, input] weights.
            Consumed packed weights, scales, and activation-scale keys are
            removed; unrelated state is unchanged.
        """
        for packed_key in [key for key in state_dict if key.endswith(".weight_packed")]:
            prefix = packed_key.removesuffix(".weight_packed")
            state_dict[prefix + ".weight"] = _dequantize_nvfp4(
                state_dict.pop(packed_key),
                state_dict.pop(prefix + ".weight_scale"),
                state_dict.pop(prefix + ".weight_global_scale"),
                self.dtype,
            )
            state_dict.pop(prefix + ".input_global_scale", None)
        scale_inv_keys = []
        for key, weight in state_dict.items():
            scale_key = key + "_scale_inv"
            if key.endswith(".weight") and scale_key in state_dict:
                state_dict[key] = dequantize_mxfp8(weight, state_dict[scale_key], dtype=self.dtype)
                scale_inv_keys.append(scale_key)
        for key in scale_inv_keys:
            state_dict.pop(key, None)
        return state_dict

    def _prepare_hf_for_merge(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Prepare Hugging Face projection tensors for grouped-expert merging.

        Args:
            state_dict: Mapping mutated in place. Expert projection tensors have shape
                [expert_hidden, hidden] for gate/up and [hidden, expert_hidden] for down;
                MXFP8 projection tensors are dequantized before their keys are renamed.
        """
        self._dequantize(state_dict)
        for key in list(state_dict):
            new_key = self._hf_key_to_native(key)
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)

    @property
    def _mtp_enabled(self) -> bool:
        return int(getattr(self.config, "num_mtp_modules", 0) or 0) > 0

    def _hf_key_to_native(self, key: str) -> str:
        key = key.replace(".block_sparse_moe.gate.weight", ".mlp.gate.weight")
        key = key.replace(".block_sparse_moe.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
        key = key.replace(".block_sparse_moe.shared_experts.", ".shared_experts.")
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w1\.weight$", r".mlp.experts.\1.gate_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w3\.weight$", r".mlp.experts.\1.up_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w2\.weight$", r".mlp.experts.\1.down_proj.weight", key)
        # Sparse-attention index branch lives under an ``indexer`` submodule.
        key = key.replace(".self_attn.index_", ".self_attn.indexer.index_")
        return key

    def _native_key_to_hf(self, key: str) -> str:
        key = re.sub(r"\.mlp\.experts\.(\d+)\.gate_proj\.weight$", r".block_sparse_moe.experts.\1.w1.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.up_proj\.weight$", r".block_sparse_moe.experts.\1.w3.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.down_proj\.weight$", r".block_sparse_moe.experts.\1.w2.weight", key)
        key = key.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
        key = key.replace(".mlp.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias")
        key = key.replace(".shared_experts.", ".block_sparse_moe.shared_experts.")
        key = key.replace(".self_attn.indexer.index_", ".self_attn.index_")
        return key

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Convert an HF checkpoint to native format (operates in-place to limit peak memory)."""
        moe_detected = False
        for key in hf_state_dict.keys():
            if ".block_sparse_moe.experts." in key and key.endswith((".weight", ".weight_packed")):
                self._uses_model_prefix = key.startswith("model.")
                moe_detected = True
                break
        if not moe_detected:
            # Fallback for text-only / pruned checkpoints with no MoE expert keys.
            self._uses_model_prefix = any(k.startswith("model.") for k in hf_state_dict)

        # MTP tensors are converted separately (the transformer_layer is a full
        # decoder block); dropped entirely when the model has no MTP module.
        mtp_keys = {k: hf_state_dict.pop(k) for k in list(hf_state_dict) if ".mtp." in k}

        self._prepare_hf_for_merge(hf_state_dict)
        native = self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

        if mtp_keys and self._mtp_enabled:
            native.update(self._mtp_from_hf(mtp_keys, device_mesh))
        return native

    def _mtp_from_hf(self, mtp_keys: dict[str, Any], device_mesh: Optional["DeviceMesh"] = None) -> dict[str, Any]:
        """Convert MTP tensors through a temporary one-layer text namespace."""
        pattern = re.compile(r"(?P<pfx>.*?)mtp\.layers\.(?P<d>\d+)\.(?P<rest>.+)")
        tl_hf: dict[str, Any] = {}
        passthrough: dict[str, Any] = {}
        for key, value in mtp_keys.items():
            m = pattern.match(key)
            if m is None:
                passthrough[key] = value
                continue
            depth, rest = m.group("d"), m.group("rest")
            if rest.startswith("transformer_layer."):
                tl_hf[f"layers.{depth}.{rest[len('transformer_layer.') :]}"] = value
            else:
                passthrough[key] = value

        self._dequantize(passthrough)  # eh_proj is MXFP8
        native = dict(passthrough)
        self._prepare_hf_for_merge(tl_hf)

        # The backbone merge already reset the per-load record. Preserve it while
        # processing MTP, then translate the temporary layers.* names back to
        # the native model.mtp.layers.* namespace used by the checkpoint key diff.
        prior_view_keys = set(self.view_loaded_native_keys)
        tl_native = self._from_hf_w_merged_experts(tl_hf, device_mesh, reset_view_loaded_keys=False)
        for key, value in tl_native.items():
            m = re.match(r"layers\.(\d+)\.(.+)", key)
            native[f"model.mtp.layers.{m.group(1)}.transformer_layer.{m.group(2)}"] = value
        new_view_keys = self.view_loaded_native_keys - prior_view_keys
        self._view_loaded_native_keys = prior_view_keys | {
            re.sub(
                r"^layers\.(\d+)\.",
                r"model.mtp.layers.\1.transformer_layer.",
                key,
            )
            for key in new_view_keys
        }
        return native

    def to_hf(
        self,
        state_dict: dict[str, torch.Tensor],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            ):
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: torch.Tensor, **kwargs) -> list[tuple[str, torch.Tensor]]:
        if ".mtp." in fqn:
            return self._mtp_tensor_to_hf(fqn, tensor, **kwargs)

        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        quantization = kwargs.get("quantization", False)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = [(self._native_key_to_hf(k), v) for k, v in expert_result]
        else:
            result = [(self._native_key_to_hf(fqn), tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            # Re-emit the source quantization key-set for the DCP load planner.
            # M3.1 uses NVFP4 expert destinations and also quantizes the indexer;
            # earlier checkpoints retain their MXFP8/BF16 layout. Placeholder
            # values are overwritten by checkpoint loading before from_hf.
            quantized: list[tuple[str, torch.Tensor]] = []
            for key, value in result:
                if self._nvfp4_experts and _NVFP4_EXPERT_RE.search(key):
                    prefix = key.removesuffix(".weight")
                    packed = _quantized_destination(value, 2, torch.uint8)
                    scales = _quantized_destination(value, _NVFP4_BLOCK_SIZE, torch.float8_e4m3fn)
                    local = value.to_local() if isinstance(value, DTensor) else value
                    quantized.extend(
                        [
                            (prefix + ".weight_packed", packed),
                            (prefix + ".weight_scale", scales),
                            (
                                prefix + ".weight_global_scale",
                                torch.empty((), dtype=torch.float32, device=local.device),
                            ),
                            (prefix + ".input_global_scale", torch.empty((), dtype=torch.float32, device=local.device)),
                        ]
                    )
                elif _should_quantize_mxfp8_key(key) or (self._nvfp4_experts and _MXFP8_INDEXER_RE.search(key)):
                    value = value.to(dtype=torch.float8_e4m3fn)
                    quantized.append((key, value))
                    quantized.append((key + "_scale_inv", create_mxfp8_scale_inv(value)))
                else:
                    quantized.append((key, value))
            return quantized

        return result

    def _mtp_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        m = re.match(r"(?P<head>.*?)mtp\.layers\.(?P<d>\d+)\.(?P<rest>.+)", fqn)
        if m is None:
            exclude_key_regex = kwargs.get("exclude_key_regex", None)
            if exclude_key_regex and re.match(exclude_key_regex, fqn):
                return []
            return [(fqn, tensor)]
        head, depth, rest = m.group("head"), m.group("d"), m.group("rest")
        if rest.startswith("transformer_layer."):
            suffix = rest[len("transformer_layer.") :]
            converted = self.convert_single_tensor_to_hf(
                f"mtp.layers.{depth}.{suffix}", tensor, prefix_override="mtp.", **kwargs
            )
            tl_prefix = f"{head}mtp.layers.{depth}.transformer_layer."
            return [(k.replace(f"mtp.layers.{depth}.", tl_prefix, 1), v) for k, v in converted]
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        if exclude_key_regex and re.match(exclude_key_regex, fqn):
            return []
        return [(fqn, tensor)]


class MiniMaxM3VLStateDictAdapter(StateDictAdapter):
    """VLM adapter: splits the M3 VL checkpoint into text / vision / projector parts.

    The released checkpoint stores the language backbone under
    ``language_model.model.*`` / ``language_model.lm_head`` and the vision side
    under ``vision_tower.vision_model.*`` with the projector / patch-merger at
    top level (``multi_modal_projector.*`` / ``patch_merge_mlp.*``).  The native
    VLM keeps the text model at ``model.*`` / ``lm_head`` and nests the projector
    / merger under ``vision_tower.*``.  Text tensors are delegated to
    :class:`MiniMaxM3StateDictAdapter` (block_sparse_moe -> mlp, index branch,
    MXFP8 dequant, grouped experts); vision tensors are BF16 and pass through.
    """

    def __init__(
        self,
        config: MiniMaxM3VLConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.text_adapter = MiniMaxM3StateDictAdapter(config.text_config, moe_config, backend, dtype=dtype)
        self.text_adapter._nvfp4_experts = _uses_nvfp4_experts(config)

    def iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        checkpoint_metadata: Mapping[str, "TensorStorageMetadata"] | None = None,
    ) -> Iterator[CheckpointLoadPart] | None:
        """Stream text quantization while loading native vision storage directly.

        Args:
            model_state_dict: Final model tensors. Text layouts follow the
                text adapter's iter_checkpoint_load_parts contract; vision
                tensors retain their arbitrary native shapes and DTensor
                placements. DCP writes every destination in place.
            device_mesh: Existing expert mesh; tensor placements are retained.
            checkpoint_metadata: Unused; part destinations derive from model shapes.

        Returns:
            Complete, bounded M3.1 load parts, or None for earlier checkpoints.
        """
        if not self.text_adapter._nvfp4_experts:
            return None
        return self._iter_mixed_checkpoint_parts(model_state_dict)

    def _iter_mixed_checkpoint_parts(self, model_state_dict: dict[str, torch.Tensor]) -> Iterator[CheckpointLoadPart]:
        """Yield direct vision storage and bounded text parameter load parts.

        Args:
            model_state_dict: Final native tensors with the layout and alias
                contract documented by iter_checkpoint_load_parts.

        Yields:
            Parts whose model_keys cover the input exactly once.
        """
        vision_keys = frozenset(key for key in model_state_dict if key.startswith("vision_tower."))
        if vision_keys:
            yield CheckpointLoadPart(
                checkpoint_tensors={
                    self._map_non_text_to_hf(key): model_state_dict[key]
                    for key in vision_keys
                    if not key.endswith("_extra_state")
                },
                model_keys=vision_keys,
                temporary_checkpoint_keys=frozenset(),
                finish=lambda: None,
            )
        text_state = {key: tensor for key, tensor in model_state_dict.items() if key not in vision_keys}
        yield from self.text_adapter._iter_mixed_checkpoint_parts(text_state, hf_prefix="language_model.")

    @property
    def view_loaded_native_keys(self) -> set[str]:
        """Native text keys loaded through the inner adapter's checkpoint views."""
        return set(self.text_adapter.view_loaded_native_keys)

    @staticmethod
    def _map_non_text_from_hf(key: str) -> str | None:
        if ".mtp." in key:
            return None  # MTP (Stage 4)
        if key.startswith("multi_modal_projector.") or key.startswith("patch_merge_mlp."):
            return "vision_tower." + key
        return key  # vision_tower.vision_model.* passes through

    @staticmethod
    def _map_non_text_to_hf(key: str) -> str:
        if key.startswith("vision_tower.multi_modal_projector.") or key.startswith("vision_tower.patch_merge_mlp."):
            return key[len("vision_tower.") :]
        return key

    def from_hf(self, hf_state_dict: dict[str, torch.Tensor], device_mesh=None, **kwargs) -> dict[str, Any]:
        # Native text keys are model.* / lm_head.* (self.model + self.lm_head);
        # the checkpoint's language_model. prefix is stripped here and re-added in to_hf.
        text_hf: dict[str, Any] = {}
        native: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if key.startswith("language_model."):
                text_hf[key[len("language_model.") :]] = value
                continue
            mapped = self._map_non_text_from_hf(key)
            if mapped is not None:
                native[mapped] = value

        native.update(self.text_adapter.from_hf(text_hf, device_mesh=device_mesh, **kwargs))
        return native

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex=None, quantization: bool = False, **kwargs):
        hf_state_dict: dict[str, Any] = {}
        for key, tensor in state_dict.items():
            for hf_key, hf_tensor in self.convert_single_tensor_to_hf(
                key, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            ):
                hf_state_dict[hf_key] = hf_tensor
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: torch.Tensor, **kwargs) -> list[tuple[str, torch.Tensor]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        if fqn.startswith("vision_tower."):
            hf_key = self._map_non_text_to_hf(fqn)
            if exclude_key_regex and re.match(exclude_key_regex, hf_key):
                return []
            return [(hf_key, tensor)]
        # Text backbone (model.* / lm_head.*): delegate, then re-add language_model. prefix.
        converted = self.text_adapter.convert_single_tensor_to_hf(fqn, tensor, **kwargs)
        return [("language_model." + k, v) for k, v in converted]

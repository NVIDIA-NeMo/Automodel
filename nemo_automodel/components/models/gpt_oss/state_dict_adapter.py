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

import gc
import math
import re
from collections import defaultdict
from collections.abc import Iterator
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import GptOssConfig

from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart, StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]

_DECODER_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")
_MXFP4_EXPERT_SUFFIXES = (
    "mlp.experts.gate_and_up_projs",
    "mlp.experts.down_projs",
)


@torch.no_grad()
def _finish_mxfp4_loads(
    adapter: "GPTOSSStateDictAdapter",
    conversions: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
) -> None:
    """Install one decoder layer's packed MXFP4 expert tensors into model storage.

    Args:
        adapter: GPT-OSS adapter that decodes the checkpoint's MXFP4 representation.
        conversions: Tuples of ``(target, blocks, scales)``. Each ``target`` is a final BF16 or FP32 model tensor with
            layout ``[experts, input_features, output_features]``. ``blocks`` is a temporary uint8 checkpoint tensor
            with layout ``[experts, output_features, input_features / 32, 16]``; each byte holds two FP4 values.
            ``scales`` is its temporary uint8 exponent tensor with layout
            ``[experts, output_features, input_features / 32]``. This function mutates each target in place and does
            not retain the temporary tensors.
    """
    for target, blocks, scales in conversions:
        converted = adapter._convert_moe_packed_tensors(blocks, scales, dtype=target.dtype)
        target.copy_(converted)
        del converted


class GPTOSSStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        config: GptOssConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

        # Key mapping from HF GPT OSS format to internal format
        self.hf_to_internal_map = {
            # Router mapping
            "mlp.router.weight": "mlp.gate.weight",
            "mlp.router.bias": "mlp.gate.bias",
            "mlp.experts.gate_up_proj": "mlp.experts.gate_and_up_projs",
            "mlp.experts.down_proj": "mlp.experts.down_projs",
        }
        if self.backend.attn == "te":
            self.hf_to_internal_map["self_attn.sinks"] = "self_attn.attn_module.softmax_offset"

        # Reverse mapping for to_hf conversion
        self.internal_to_hf_map = {v: k for k, v in self.hf_to_internal_map.items() if v is not None}

    # replace _apply_key_mapping with leaf-aware in-place replacement
    def _apply_key_mapping(self, state_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
        for key in list(state_dict.keys()):
            new_key = key
            for pattern, replacement in mapping.items():
                if replacement is not None and key.endswith(pattern):
                    new_key = key[: -len(pattern)] + replacement
                    break
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)
        return state_dict

    def _model_to_hf_key(self, model_key: str) -> str:
        """Map one native GPT-OSS tensor name to its Hugging Face name."""
        for pattern, replacement in self.internal_to_hf_map.items():
            if model_key.endswith(pattern):
                return model_key[: -len(pattern)] + replacement
        return model_key

    def iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> Iterator[CheckpointLoadPart] | None:
        """Load a single-GPU GPT-OSS MXFP4 checkpoint one decoder layer at a time.

        Ordinary checkpoint tensors load directly into their final model tensors. Each decoder-layer part allocates
        only that layer's packed uint8 expert blocks and scales. After DCP fills them, the part decodes one projection
        at a time into the existing BF16 or FP32 model tensor and releases the packed values before advancing. Backend
        bookkeeping entries ending in ``_extra_state`` are not checkpoint tensors and keep their initialized values.

        This path intentionally requires a complete, non-distributed decoder. Distributed GPT-OSS loading already
        uses rank-local DCP tensors, while some distributed expert backends expose state-dict tensors that do not own
        the final parameter storage.

        Args:
            model_state_dict: Native names mapped to final model tensors. Expert projection tensors must have layout
                ``[experts, input_features, output_features]``, use BF16 or FP32, and own ordinary single-device
                storage. Other tensors retain arbitrary model-defined shapes, dtypes, devices, strides, and storage.
            device_mesh: Optional distributed mesh. A non-``None`` mesh disables this single-device path.

        Returns:
            One direct-load part for ordinary tensors followed by one bounded temporary-load part per decoder layer,
            or ``None`` when the model is distributed, the decoder is partial, or an expert tensor has an unsupported
            layout or dtype.
        """
        if device_mesh is not None:
            return None
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            return None

        num_hidden_layers = getattr(self.config, "num_hidden_layers", None)
        if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
            return None
        present_layer_indices = {
            int(layer_match.group(1))
            for model_key in model_state_dict
            if (layer_match := _DECODER_LAYER_KEY.match(model_key)) is not None
        }
        if present_layer_indices != set(range(num_hidden_layers)):
            return None

        expert_model_keys_by_layer: dict[int, list[str]] = defaultdict(list)
        for model_key, target in model_state_dict.items():
            if not model_key.endswith(_MXFP4_EXPERT_SUFFIXES):
                continue
            layer_match = _DECODER_LAYER_KEY.match(model_key)
            if layer_match is None:
                return None
            if (
                not isinstance(target, torch.Tensor)
                or isinstance(target, torch.distributed.tensor.DTensor)
                or target.dtype not in (torch.bfloat16, torch.float32)
                or target.ndim != 3
                or target.shape[1] % 32 != 0
            ):
                return None
            expert_model_keys_by_layer[int(layer_match.group(1))].append(model_key)

        if set(expert_model_keys_by_layer) != set(range(num_hidden_layers)):
            return None
        expected_expert_suffixes = set(_MXFP4_EXPERT_SUFFIXES)
        for model_keys in expert_model_keys_by_layer.values():
            found_expert_suffixes = {
                suffix for model_key in model_keys for suffix in _MXFP4_EXPERT_SUFFIXES if model_key.endswith(suffix)
            }
            if found_expert_suffixes != expected_expert_suffixes:
                return None
        return self._iter_checkpoint_load_parts(model_state_dict, expert_model_keys_by_layer)

    def _iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        expert_model_keys_by_layer: dict[int, list[str]],
    ) -> Iterator[CheckpointLoadPart]:
        """Build lazy direct and per-layer MXFP4 load parts.

        Args:
            model_state_dict: Native names mapped to final single-device tensors. Expert tensors have layout
                ``[experts, input_features, output_features]`` and BF16 or FP32 dtype. All direct destinations are
                mutated in place by DCP.
            expert_model_keys_by_layer: Complete decoder-layer indices mapped to the two native expert projection
                names owned by each layer.

        Yields:
            A direct-load part followed by one part per decoder layer. Each layer part owns uint8 ``blocks`` tensors
            with layout ``[experts, output_features, input_features / 32, 16]`` and matching ``scales`` tensors with
            layout ``[experts, output_features, input_features / 32]`` until its finish callback returns.
        """
        expert_model_keys = {
            model_key for layer_model_keys in expert_model_keys_by_layer.values() for model_key in layer_model_keys
        }
        direct_model_keys = [model_key for model_key in model_state_dict if model_key not in expert_model_keys]
        yield CheckpointLoadPart(
            checkpoint_tensors={
                self._model_to_hf_key(model_key): model_state_dict[model_key]
                for model_key in direct_model_keys
                if not model_key.endswith("_extra_state")
            },
            model_keys=frozenset(direct_model_keys),
            temporary_checkpoint_keys=frozenset(),
            finish=lambda: None,
        )

        for layer_idx in range(len(expert_model_keys_by_layer)):
            model_keys = expert_model_keys_by_layer[layer_idx]
            checkpoint_tensors: dict[str, torch.Tensor] = {}
            temporary_checkpoint_keys: set[str] = set()
            conversions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for model_key in model_keys:
                target = model_state_dict[model_key]
                hf_key = self._model_to_hf_key(model_key)
                n_experts, input_features, output_features = target.shape
                blocks = torch.empty(
                    (n_experts, output_features, input_features // 32, 16),
                    dtype=torch.uint8,
                    device=target.device,
                )
                scales = torch.empty(
                    (n_experts, output_features, input_features // 32),
                    dtype=torch.uint8,
                    device=target.device,
                )
                blocks_key = f"{hf_key}_blocks"
                scales_key = f"{hf_key}_scales"
                checkpoint_tensors[blocks_key] = blocks
                checkpoint_tensors[scales_key] = scales
                temporary_checkpoint_keys.update((blocks_key, scales_key))
                conversions.append((target, blocks, scales))

            yield CheckpointLoadPart(
                checkpoint_tensors=checkpoint_tensors,
                model_keys=frozenset(model_keys),
                temporary_checkpoint_keys=frozenset(temporary_checkpoint_keys),
                finish=partial(_finish_mxfp4_loads, self, tuple(conversions)),
            )
            del checkpoint_tensors, temporary_checkpoint_keys, conversions, blocks, scales

    def _dequantize_block_scale_tensors(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        layer_name_to_quantized_weights = defaultdict(dict)

        # create the mapping from layer name to quantized weights {layer_name: {"blocks"/"scales": value}}
        for key, value in list(state_dict.items()):
            if key.endswith("_blocks") or key.endswith("_scales"):
                layer_name, quantized_name = key.rsplit("_", 1)
                layer_name_to_quantized_weights[layer_name][quantized_name] = value
                del state_dict[key]

        # Dequantize experts one layer at a time, popping each entry from
        # ``layer_name_to_quantized_weights`` so the quantized blocks/scales
        # are released as soon as their dequantized output is produced.
        # Holding the full quantized dict alive through the loop pinned every
        # layer's blocks/scales until the very end, inflating peak memory.
        for layer_name in list(layer_name_to_quantized_weights.keys()):
            quantized_dict = layer_name_to_quantized_weights.pop(layer_name)
            dequantized_weights = self._convert_moe_packed_tensors(quantized_dict["blocks"], quantized_dict["scales"])
            del quantized_dict
            state_dict[layer_name] = dequantized_weights

        # clean up the memory
        torch.cuda.empty_cache()
        gc.collect()
        return state_dict

    def _convert_moe_packed_tensors(
        self,
        blocks,
        scales,
        dtype: torch.dtype = torch.bfloat16,
        rows_per_chunk: int = 32768 * 1024,
    ) -> torch.Tensor:
        """
        Convert the mxfp4 weights to bfloat16.

        Source: https://github.com/huggingface/transformers/blob/869735d37d0f929311ac6611728c482a4414ba8c/src/transformers/integrations/mxfp4.py#L77
        """
        # Check if blocks and scales are on CPU, and move to GPU if so
        if not blocks.is_cuda and torch.cuda.is_available() and torch.distributed.get_world_size() > 1:
            blocks = blocks.cuda()
            scales = scales.cuda()

        scales = scales.to(torch.int32) - 127  # that's because 128=2**7

        assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

        *prefix_shape, G, B = blocks.shape
        rows_total = math.prod(prefix_shape) * G

        blocks = blocks.reshape(rows_total, B)
        scales = scales.reshape(rows_total, 1)

        if isinstance(blocks, torch.distributed.tensor.DTensor):
            out = torch.distributed.tensor.empty(
                (rows_total, B * 2), placements=blocks.placements, device_mesh=blocks.device_mesh, dtype=dtype
            )
        else:
            out = torch.empty((rows_total, B * 2), dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)

            blk = blocks[r0:r1]
            exp = scales[r0:r1]
            sub = out[r0:r1]

            # Work on local shards to avoid DTensor advanced indexing
            blk_local = blk.to_local() if hasattr(blk, "to_local") else blk
            sub_local = sub.to_local() if hasattr(sub, "to_local") else sub
            exp_local = exp.to_local() if hasattr(exp, "to_local") else exp

            # Ensure uint8 for nibble extraction
            blk_local = blk_local.to(torch.uint8)

            # nibble indices -> int64 (local)
            idx_lo_local = (blk_local & 0x0F).to(torch.long)
            idx_hi_local = (blk_local >> 4).to(torch.long)

            sub_local[:, 0::2] = lut[idx_lo_local]
            sub_local[:, 1::2] = lut[idx_hi_local]

            torch.ldexp(sub_local, exp_local, out=sub_local)
            del idx_lo_local, idx_hi_local, blk_local, exp_local, sub_local, blk, exp, sub

        out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
        del blocks, scales, lut

        # Final logical layout is (n_experts, 2880, hidden_dim) after transpose.
        out = out.transpose(1, 2).contiguous()
        # Restore desired DTensor sharding: shard experts (dim 0) by 'ep' and hidden dim (dim 2) by 'ep_shard'.
        if isinstance(out, torch.distributed.tensor.DTensor):
            mesh_dim_names = out.device_mesh.mesh_dim_names
            if "ep" in mesh_dim_names or "ep_shard" in mesh_dim_names:
                placements = []
                for dim_name in mesh_dim_names:
                    if dim_name == "ep":
                        placements.append(torch.distributed.tensor.Shard(0))
                    elif dim_name == "ep_shard":
                        placements.append(torch.distributed.tensor.Shard(2))
                    else:
                        placements.append(torch.distributed.tensor.Replicate())
                if placements != out.placements:
                    out = out.redistribute(placements=tuple(placements))
        return out

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: str | None = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format."""
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format in-place.
        - Apply key mappings from HF to internal format
        - Dequantize block/scale tensors (freeing originals)

        Operates in-place on the input dict to avoid allocating a full copy,
        reducing peak memory from 2x to ~1x model size.
        """
        # Detect model prefix usage
        for key in hf_state_dict.keys():
            if key.startswith("model."):
                self._uses_model_prefix = True
                break

        self._dequantize_block_scale_tensors(hf_state_dict)
        self._apply_key_mapping(hf_state_dict, self.hf_to_internal_map)

        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        quantization = kwargs.get("quantization", False)
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        hf_fqn = self._model_to_hf_key(fqn)

        if exclude_key_regex:
            if re.match(exclude_key_regex, hf_fqn):
                return []

        if quantization:
            if hf_fqn.endswith("gate_up_proj") or hf_fqn.endswith("down_proj"):
                layer_name, projection_type = hf_fqn.rsplit(".", 1)
                n_experts, _, dim = tensor.shape

                if isinstance(tensor, torch.distributed.tensor.DTensor):
                    device_mesh = tensor.device_mesh
                    # Ensure quantized tensors shard only along dim 0 for safe flattening in conversion
                    orig_placements = tensor.placements
                    safe_placements = []
                    found_shard_dim0 = False
                    for p in orig_placements:
                        if isinstance(p, torch.distributed.tensor.Shard):
                            if p.dim == 0 and not found_shard_dim0:
                                safe_placements.append(p)
                                found_shard_dim0 = True
                            else:
                                safe_placements.append(torch.distributed.tensor.Replicate())
                        else:
                            safe_placements.append(p)
                    blocks_tensors = torch.distributed.tensor.ones(
                        (n_experts, dim, 90, 16),
                        placements=tuple(safe_placements),
                        device_mesh=device_mesh,
                        dtype=torch.uint8,
                    )
                    scales_tensors = torch.distributed.tensor.ones(
                        (n_experts, dim, 90),
                        placements=tuple(safe_placements),
                        device_mesh=device_mesh,
                        dtype=torch.uint8,
                    )
                else:
                    blocks_tensors = torch.ones((n_experts, dim, 90, 16), dtype=torch.uint8)
                    scales_tensors = torch.ones((n_experts, dim, 90), dtype=torch.uint8)

                return [
                    (f"{layer_name}.{projection_type}_blocks", blocks_tensors),
                    (f"{layer_name}.{projection_type}_scales", scales_tensors),
                ]

        return [(hf_fqn, tensor)]

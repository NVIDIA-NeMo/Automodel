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
from typing import TYPE_CHECKING, Any, Optional

import torch
from transformers import GptOssConfig

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig

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

    # replace _apply_key_mapping with leaf-aware replacement
    def _apply_key_mapping(self, state_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
        keys_to_remove = set()
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for pattern, replacement in mapping.items():
                if replacement is not None and key.endswith(pattern):
                    new_key = key[: -len(pattern)] + replacement
                    break
            new_state_dict[new_key] = value
            keys_to_remove.add(key)

        for key in keys_to_remove:
            del state_dict[key]
        return new_state_dict

    def _dequantize_block_scale_tensors(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        layer_name_to_quantized_weights = defaultdict(dict)

        # create the mapping from layer name to quantized weights {layer_name: {"blocks"/"scales": value}}
        for key, value in list(state_dict.items()):
            if key.endswith("_blocks") or key.endswith("_scales"):
                layer_name, quantized_name = key.rsplit("_", 1)
                layer_name_to_quantized_weights[layer_name][quantized_name] = value
                del state_dict[key]

        # dequantize the experts
        for layer_name, quantized_dict in layer_name_to_quantized_weights.items():
            dequantized_weights = self._convert_moe_packed_tensors(quantized_dict["blocks"], quantized_dict["scales"])
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
            placements = []
            mesh_dim_names = out.device_mesh.mesh_dim_names
            for dim_name in mesh_dim_names:
                if dim_name == "ep":
                    placements.append(torch.distributed.tensor.Shard(0))
                elif dim_name == "ep_shard":
                    placements.append(torch.distributed.tensor.Shard(2))
                else:
                    raise ValueError(f"Unexpected dimension name: {dim_name}")
            if placements != out.placements:
                out = out.redistribute(placements=tuple(placements))
        return out

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, quantization: bool = False, **kwargs
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
        """Convert HF checkpoint to native format.
        - Apply key mappings from HF to internal format
        - Add quantization block and scale tensors
        """
        # Detect model prefix usage
        for key in hf_state_dict.keys():
            if key.startswith("model."):
                self._uses_model_prefix = True
                break

        native_state_dict = dict(hf_state_dict)
        native_state_dict = self._dequantize_block_scale_tensors(native_state_dict)
        native_state_dict = self._apply_key_mapping(native_state_dict, self.hf_to_internal_map)

        return native_state_dict

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

        hf_fqn = fqn
        for pattern, replacement in self.internal_to_hf_map.items():
            if fqn.endswith(pattern):
                hf_fqn = fqn[: -len(pattern)] + replacement
                break

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

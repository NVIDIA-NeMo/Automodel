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

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import dequantize_from_fp8
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import create_dtensor_from_local, is_dtensor

logger = logging.getLogger(__name__)

# HF checkpoint prefix for the text backbone inside the multimodal wrapper
_HF_PREFIX = "language_model."

# Keys that should NOT be quantized (layernorms, embeddings, gates)
_NON_QUANTIZED_PATTERNS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "mlp.gate.weight",
]


def _should_quantize_key(key: str) -> bool:
    """Check if a key should be quantized based on its name.

    Handles both standard keys (*.weight) and Mistral4 aggregated expert keys
    (*.gate_up_proj, *.down_proj) which don't have a .weight suffix.
    Only text model weights are FP8; vision tower, projector, and lm_head are not.
    """
    # Vision tower and projector are never quantized
    if "vision_tower" in key or "multi_modal_projector" in key:
        return False
    # Expert aggregated keys (no .weight suffix but should be quantized)
    if key.endswith((".gate_up_proj", ".down_proj")) and ".mlp.experts." in key:
        return True
    # Standard weight keys
    if not key.endswith(".weight"):
        return False
    return not any(pattern in key for pattern in _NON_QUANTIZED_PATTERNS)


def _dequantize_state_dict(state_dict: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    """Dequantize FP8 weights in-place. Handles both per-tensor and block-wise formats.

    Mistral 4 HF checkpoint has two FP8 patterns:
    - Standard weights: ``*.weight`` + ``*.weight_scale_inv`` (attention, shared experts)
    - Expert weights: ``mlp.experts.gate_up_proj`` + ``mlp.experts.gate_up_proj_scale_inv`` (no .weight suffix)
    """
    keys_to_remove = set()
    keys_to_update = {}

    for key in list(state_dict.keys()):
        # Skip scale/activation keys themselves
        if key.endswith("_scale_inv") or key.endswith("_activation_scale") or key.endswith(".activation_scale"):
            continue

        scale_key = key + "_scale_inv"
        if scale_key not in state_dict:
            continue

        weight = state_dict[key]
        scale_inv = state_dict[scale_key]

        # Handle DTensor: extract local tensor, dequant, re-wrap
        weight_is_dtensor = is_dtensor(weight)
        weight_local = weight.to_local() if weight_is_dtensor else weight
        scale_local = scale_inv.to_local() if is_dtensor(scale_inv) else scale_inv
        scale_local = scale_local.to(device=weight_local.device)
        # Squeeze broadcast dims: [n_experts, 1, 1] -> [n_experts], [] stays []
        scale_local = scale_local.squeeze()

        if scale_local.numel() == 1:
            # Per-tensor dequantization (scalar scale)
            dequantized = (weight_local.float() * scale_local.float()).to(dtype)
        elif scale_local.dim() == 1 and weight_local.dim() == 3:
            # Per-expert per-tensor dequantization: scale_inv [n_experts], weight [n_local_experts, d1, d2]
            # After EP sharding, weight may have fewer experts than scale_inv.
            # Slice scale to match the local expert count.
            n_local = weight_local.shape[0]
            if scale_local.shape[0] != n_local:
                # Determine which slice of experts this rank has
                if weight_is_dtensor:
                    from torch.distributed._tensor import Shard
                    for placement in weight.placements:
                        if isinstance(placement, Shard) and placement.dim == 0:
                            rank_in_ep = weight.device_mesh.get_local_rank()
                            n_total = scale_local.shape[0]
                            chunk_size = n_total // weight.device_mesh.size()
                            start = rank_in_ep * chunk_size
                            scale_local = scale_local[start:start + n_local]
                            break
                    else:
                        scale_local = scale_local[:n_local]
                else:
                    scale_local = scale_local[:n_local]
            dequantized = (weight_local.float() * scale_local.float().view(-1, 1, 1)).to(dtype)
        elif scale_local.dim() == 2 and weight_local.dim() == 2:
            # Block-wise dequantization
            dequantized = dequantize_from_fp8(weight_local, scale_local, dtype=dtype, name=key)
        else:
            # Fallback
            dequantized = dequantize_from_fp8(weight_local, scale_local, dtype=dtype, name=key)

        if weight_is_dtensor:
            from torch.distributed._tensor import DTensor
            keys_to_update[key] = DTensor.from_local(dequantized, weight.device_mesh, weight.placements)
        else:
            keys_to_update[key] = dequantized

        keys_to_remove.add(scale_key)

    # Apply updates
    for key, value in keys_to_update.items():
        state_dict[key] = value

    # Remove all scale_inv and activation_scale keys
    for key in list(state_dict.keys()):
        if key.endswith("_scale_inv") or key.endswith("_activation_scale") or key.endswith(".activation_scale"):
            keys_to_remove.add(key)

    for key in keys_to_remove:
        state_dict.pop(key, None)

    logger.debug(f"[FP8 Dequant] Processed {len(keys_to_update)} weights, removed {len(keys_to_remove)} scale keys")
    return state_dict


def _convert_aggregated_experts(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert aggregated expert weights from HF format to native format.

    HF format (aggregated 3D tensors):
        mlp.experts.gate_up_proj  [128, 2*moe_inter_dim, hidden_size]
        mlp.experts.down_proj     [128, hidden_size, moe_inter_dim]

    Native format:
        mlp.experts.gate_and_up_projs  [128, hidden_size, 2*moe_inter_dim]
        mlp.experts.down_projs         [128, moe_inter_dim, hidden_size]
    """
    keys_to_remove = []
    keys_to_add = {}

    for key in list(state_dict.keys()):
        if ".mlp.experts.gate_up_proj" in key and not key.endswith(("_scale_inv", "_activation_scale")):
            native_key = key.replace(".mlp.experts.gate_up_proj", ".mlp.experts.gate_and_up_projs")
            keys_to_add[native_key] = state_dict[key].transpose(1, 2)
            keys_to_remove.append(key)

        elif ".mlp.experts.down_proj" in key and not key.endswith(("_scale_inv", "_activation_scale")):
            native_key = key.replace(".mlp.experts.down_proj", ".mlp.experts.down_projs")
            keys_to_add[native_key] = state_dict[key].transpose(1, 2)
            keys_to_remove.append(key)

    for key in keys_to_remove:
        state_dict.pop(key)
    state_dict.update(keys_to_add)

    return state_dict


def _inject_missing_gate_bias(state_dict: dict[str, Any], n_routed_experts: int) -> dict[str, Any]:
    """Inject zero ``e_score_correction_bias`` for MoE layers that lack it.

    Some checkpoints (e.g. vv4) don't include the gate bias — it starts at zero
    and is learned during training.  The model always expects the key, so we
    inject ``torch.zeros(n_routed_experts)`` for any layer that has a gate weight
    but no bias.
    """
    gate_weight_keys = [k for k in state_dict if k.endswith(".mlp.gate.weight")]
    for gw_key in gate_weight_keys:
        bias_key = gw_key.replace(".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")
        if bias_key not in state_dict:
            state_dict[bias_key] = torch.zeros(n_routed_experts, dtype=torch.float32)
    return state_dict


class Mistral4StateDictAdapter(StateDictAdapter):
    """State dict adapter for Mistral 4 **text-only** (CausalLM).

    Handles:
    1. Stripping ``language_model.`` prefix from HF keys
    2. FP8 dequantization (per-tensor and block-wise)
    3. Aggregated expert weight conversion (3D tensors → native format)
    4. Removing activation scale keys
    """

    def __init__(
        self,
        config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype

    def _strip_prefix(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Strip ``language_model.`` prefix from all keys."""
        new_sd = {}
        for key, value in state_dict.items():
            if key.startswith(_HF_PREFIX):
                new_sd[key[len(_HF_PREFIX):]] = value
            else:
                new_sd[key] = value
        return new_sd

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        state_dict = self._strip_prefix(hf_state_dict)
        state_dict = _dequantize_state_dict(state_dict, self.dtype)
        state_dict = _convert_aggregated_experts(state_dict)
        state_dict = _inject_missing_gate_bias(state_dict, self.moe_config.n_routed_experts)
        return state_dict

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted:
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex")
        quantization = kwargs.get("quantization", False)

        # Drop e_score_correction_bias when building quantization skeleton —
        # some checkpoints don't include it (zero-init, injected during from_hf).
        if quantization and "e_score_correction_bias" in fqn:
            return []

        if ".mlp.experts.gate_and_up_projs" in fqn:
            hf_key = _HF_PREFIX + fqn.replace(".mlp.experts.gate_and_up_projs", ".mlp.experts.gate_up_proj")
            result = [(hf_key, tensor.transpose(1, 2).contiguous())]
        elif ".mlp.experts.down_projs" in fqn:
            hf_key = _HF_PREFIX + fqn.replace(".mlp.experts.down_projs", ".mlp.experts.down_proj")
            result = [(hf_key, tensor.transpose(1, 2).contiguous())]
        else:
            result = [(_HF_PREFIX + fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            for key, value in result:
                if _should_quantize_key(key):
                    fp8_value = value.to(dtype=torch.float8_e4m3fn)
                    # Create scale_inv placeholder: scalar for standard, [n_experts,1,1] / [n_experts] for experts
                    if fp8_value.dim() == 3:
                        scale_inv = torch.ones(fp8_value.shape[0], 1, 1, dtype=torch.float32, device=value.device)
                        act_scale = torch.ones(fp8_value.shape[0], dtype=torch.float32, device=value.device)
                    else:
                        scale_inv = torch.ones([], dtype=torch.float32, device=value.device)
                        act_scale = torch.ones([], dtype=torch.float32, device=value.device)
                    quantized_result.append((key, fp8_value))
                    quantized_result.append((key + "_scale_inv", scale_inv))
                    # Checkpoint also has activation_scale keys
                    if key.endswith(".weight"):
                        quantized_result.append((key.replace(".weight", ".activation_scale"), act_scale))
                    else:
                        quantized_result.append((key + "_activation_scale", act_scale))
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result


class Mistral4MultimodalStateDictAdapter(StateDictAdapter):
    """State dict adapter for the full **multimodal** Mistral 4 (ForConditionalGeneration).

    Checkpoint key prefixes → native model key prefixes:
        ``language_model.model.X``    → ``model.language_model.X``  (text backbone)
        ``language_model.lm_head.X``  → ``lm_head.X``              (LM head)
        ``vision_tower.X``            → ``model.vision_tower.X``    (Pixtral)
        ``multi_modal_projector.X``   → ``model.multi_modal_projector.X``

    FP8 dequantization is applied only to text-model weights (vision/projector are not quantized).
    Expert weights are converted from aggregated 3D format to native format.
    """

    def __init__(
        self,
        config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype

    def _remap_keys_from_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Remap checkpoint keys to native model keys."""
        new_sd = {}
        for key, value in state_dict.items():
            if key.startswith("language_model.lm_head."):
                # language_model.lm_head.weight -> model.language_model.lm_head.weight
                suffix = key[len("language_model."):]
                new_key = "model.language_model." + suffix
                new_sd[new_key] = value
            elif key.startswith("language_model.model."):
                # language_model.model.layers.0.X -> model.language_model.model.layers.0.X
                suffix = key[len("language_model.model."):]
                new_key = "model.language_model.model." + suffix
                new_sd[new_key] = value
            elif key.startswith("vision_tower."):
                # vision_tower.X -> model.vision_tower.X
                new_key = "model." + key
                new_sd[new_key] = value
            elif key.startswith("multi_modal_projector."):
                # multi_modal_projector.X -> model.multi_modal_projector.X
                new_key = "model." + key
                new_sd[new_key] = value
            else:
                new_sd[key] = value
        return new_sd

    def _remap_keys_to_hf(self, key: str) -> str:
        """Remap a single native key back to checkpoint format."""
        if key.startswith("model.language_model.lm_head."):
            suffix = key[len("model.language_model."):]
            return "language_model." + suffix
        elif key.startswith("model.language_model.model."):
            suffix = key[len("model.language_model.model."):]
            return "language_model.model." + suffix
        elif key.startswith("model.language_model."):
            suffix = key[len("model.language_model."):]
            return "language_model." + suffix
        elif key.startswith("model.vision_tower."):
            return key[len("model."):]
        elif key.startswith("model.multi_modal_projector."):
            return key[len("model."):]
        return key

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format.

        Pipeline:
        1. Remap checkpoint keys to native model keys
        2. Dequantize FP8 weights (text model only; vision/projector are not quantized)
        3. Convert aggregated expert weights to native format
        """
        state_dict = self._remap_keys_from_hf(hf_state_dict)
        state_dict = _dequantize_state_dict(state_dict, self.dtype)
        state_dict = _convert_aggregated_experts(state_dict)
        state_dict = _inject_missing_gate_bias(state_dict, self.moe_config.n_routed_experts)
        return state_dict

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted:
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex")
        quantization = kwargs.get("quantization", False)

        if quantization and "e_score_correction_bias" in fqn:
            return []

        # Convert native expert tensors back to HF aggregated format
        if ".mlp.experts.gate_and_up_projs" in fqn:
            hf_fqn = fqn.replace(".mlp.experts.gate_and_up_projs", ".mlp.experts.gate_up_proj")
            hf_key = self._remap_keys_to_hf(hf_fqn)
            result = [(hf_key, tensor.transpose(1, 2).contiguous())]
        elif ".mlp.experts.down_projs" in fqn:
            hf_fqn = fqn.replace(".mlp.experts.down_projs", ".mlp.experts.down_proj")
            hf_key = self._remap_keys_to_hf(hf_fqn)
            result = [(hf_key, tensor.transpose(1, 2).contiguous())]
        else:
            # All other keys: remap prefix
            result = [(self._remap_keys_to_hf(fqn), tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            for key, value in result:
                if _should_quantize_key(key):
                    fp8_value = value.to(dtype=torch.float8_e4m3fn)
                    # [n_experts,1,1] scale_inv / [n_experts] act_scale for experts, scalar for standard
                    if fp8_value.dim() == 3:
                        scale_inv = torch.ones(fp8_value.shape[0], 1, 1, dtype=torch.float32, device=value.device)
                        act_scale = torch.ones(fp8_value.shape[0], dtype=torch.float32, device=value.device)
                    else:
                        scale_inv = torch.ones([], dtype=torch.float32, device=value.device)
                        act_scale = torch.ones([], dtype=torch.float32, device=value.device)
                    quantized_result.append((key, fp8_value))
                    quantized_result.append((key + "_scale_inv", scale_inv))
                    if key.endswith(".weight"):
                        quantized_result.append((key.replace(".weight", ".activation_scale"), act_scale))
                    else:
                        quantized_result.append((key + "_activation_scale", act_scale))
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result

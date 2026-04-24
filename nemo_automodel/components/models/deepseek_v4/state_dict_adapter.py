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

"""State dict adapter for DeepSeek V4.

HF V4 uses different key names compared to V3/V3.2.  This adapter performs
the necessary renaming on top of the standard FP8 dequantization and
per-expert weight aggregation.

Key mapping (HF -> internal):
  embed.weight                          -> model.embed_tokens.weight
  norm.weight                           -> model.norm.weight
  head.weight                           -> lm_head.weight
  layers.{i}.attn_norm.weight           -> model.layers.{i}.input_layernorm.weight
  layers.{i}.ffn_norm.weight            -> model.layers.{i}.post_attention_layernorm.weight
  layers.{i}.attn.*                     -> model.layers.{i}.self_attn.*
  layers.{i}.ffn.gate.weight            -> model.layers.{i}.mlp.gate.weight
  layers.{i}.ffn.gate.bias             -> model.layers.{i}.mlp.gate.e_score_correction_bias
  layers.{i}.ffn.gate.tid2eid          -> model.layers.{i}.mlp.gate.tid2eid  (hash layers only)
  layers.{i}.ffn.shared_experts.w1.*   -> model.layers.{i}.mlp.shared_experts.gate_proj.*
  layers.{i}.ffn.shared_experts.w3.*   -> model.layers.{i}.mlp.shared_experts.up_proj.*
  layers.{i}.ffn.shared_experts.w2.*   -> model.layers.{i}.mlp.shared_experts.down_proj.*
  layers.{i}.ffn.experts.{j}.w1.weight -> aggregated into model.layers.{i}.mlp.experts.gate_and_up_projs
  layers.{i}.ffn.experts.{j}.w3.weight -> aggregated into model.layers.{i}.mlp.experts.gate_and_up_projs
  layers.{i}.ffn.experts.{j}.w2.weight -> aggregated into model.layers.{i}.mlp.experts.down_projs
  layers.{i}.hc_attn_base/fn/scale     -> model.layers.{i}.hc_attn_base/fn/scale
  layers.{i}.hc_ffn_base/fn/scale      -> model.layers.{i}.hc_ffn_base/fn/scale

FP8 note: HF V4 stores scale as `<key>.scale` (not `<key>.weight_scale_inv` like V3).
Both suffixes are handled by the dequantization step.
"""

from __future__ import annotations

import enum
import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    BLOCK_SIZE,
    dequantize_from_fp8,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_expert_slice_for_rank,
    get_submesh,
    is_dtensor,
    should_load_expert_for_rank,
)

# V4 Flash routed-expert weights are stored as FP4 (e2m1fn) packed two values per
# int8 byte, with FP8 (e8m0fnu) per-row scales covering 32-column groups:
#   weight: int8 with shape [out, in // 2]         (low nibble + high nibble = 2 fp4 values)
#   scale:  float8_e8m0fnu with shape [out, in // 32]
# Non-expert weights (attention, norms, embed, lm_head, shared experts) use the
# standard FP8 e4m3fn with BLOCK_SIZE×BLOCK_SIZE (128×128) scaling.
FP4_COL_BLOCK = 32

# FP4 e2m1 value table: low 3 bits -> mantissa/exponent, MSB -> sign.
# Layout: [positive values for 0-7, negative values for 8-15].
_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# HF V4 key -> internal key  (simple renames; expert & FP8 handled separately)
_HF_TO_INTERNAL_RENAMES: list[tuple[re.Pattern, str]] = [
    # Top-level
    (re.compile(r"^embed\.(.+)$"), r"model.embed_tokens.\1"),
    (re.compile(r"^norm\.(.+)$"), r"model.norm.\1"),
    (re.compile(r"^head\.(.+)$"), r"lm_head.\1"),
    # Per-layer norms
    (re.compile(r"^layers\.(\d+)\.attn_norm\.(.+)$"), r"model.layers.\1.input_layernorm.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn_norm\.(.+)$"), r"model.layers.\1.post_attention_layernorm.\2"),
    # Attention sub-keys
    (re.compile(r"^layers\.(\d+)\.attn\.(.+)$"), r"model.layers.\1.self_attn.\2"),
    # MoE gate (score weight + optional bias correction + hash table)
    (re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias$"), r"model.layers.\1.mlp.gate.e_score_correction_bias"),
    (re.compile(r"^layers\.(\d+)\.ffn\.gate\.(.+)$"), r"model.layers.\1.mlp.gate.\2"),
    # Shared expert (w1=gate, w3=up, w2=down)
    (
        re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w1\.(.+)$"),
        r"model.layers.\1.mlp.shared_experts.gate_proj.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w3\.(.+)$"),
        r"model.layers.\1.mlp.shared_experts.up_proj.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w2\.(.+)$"),
        r"model.layers.\1.mlp.shared_experts.down_proj.\2",
    ),
    # Latent projections (fc1: hidden→latent, fc2: latent→hidden)
    (
        re.compile(r"^layers\.(\d+)\.ffn\.fc1_latent_proj\.(.+)$"),
        r"model.layers.\1.mlp.fc1_latent_proj.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.ffn\.fc2_latent_proj\.(.+)$"),
        r"model.layers.\1.mlp.fc2_latent_proj.\2",
    ),
    # HC (hash-clustering) parameters
    (re.compile(r"^layers\.(\d+)\.(hc_(?:attn|ffn)_(?:base|fn|scale))$"), r"model.layers.\1.\2"),
]

# Routed-expert pattern in HF V4 format
_EXPERT_PATTERN = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


class _HashBiasScope(enum.Enum):
    """Key-format scope for :meth:`DeepSeekV4StateDictAdapter._drop_hash_layer_gate_bias`."""

    INTERNAL = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$")
    HF = re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias$")


def _rename_hf_key(key: str) -> str:
    """Apply simple rename rules; returns the key unchanged if no rule matches."""
    for pattern, replacement in _HF_TO_INTERNAL_RENAMES:
        new_key, n = pattern.subn(replacement, key)
        if n:
            return new_key
    return key


class DeepSeekV4StateDictAdapter(StateDictAdapter):
    """State dict adapter for DeepSeek V4."""

    def __init__(
        self,
        config: DeepseekV4Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype

    # ------------------------------------------------------------------
    # from_hf
    # ------------------------------------------------------------------

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to internal format.

        Steps:
          1. Dequantize FP8 weights (scale suffix is either `.scale` or `_scale_inv`).
          2. Aggregate per-expert routed weights into stacked tensors.
          3. Rename remaining keys using the HF -> internal mapping table.
        """
        hf_state_dict = self._dequantize(hf_state_dict)
        hf_state_dict = self._aggregate_experts(hf_state_dict, device_mesh)
        return self._rename_all(hf_state_dict)

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Dequantize FP8 weights.  Handles both `.scale` and `_scale_inv` suffixes."""
        scale_keys_to_remove: list[str] = []
        for key in list(state_dict.keys()):
            weight = state_dict[key]
            # tid2eid is int32 — skip dequantization entirely
            if key.endswith(".tid2eid"):
                continue
            # HF V4 uses `<base>.scale`; V3 used `<base>.weight_scale_inv`
            scale_key = None
            if key.endswith(".weight"):
                base = key[: -len(".weight")]
                if base + ".scale" in state_dict:
                    scale_key = base + ".scale"
                elif key + "_scale_inv" in state_dict:
                    scale_key = key + "_scale_inv"

            if scale_key is not None:
                scale = state_dict[scale_key]
                if self._is_expert_weight_key(key):
                    state_dict[key] = self._dequantize_expert_fp4(weight, scale, self.dtype)
                else:
                    state_dict[key] = dequantize_from_fp8(weight, scale, dtype=self.dtype, name=key)
                scale_keys_to_remove.append(scale_key)

        for k in scale_keys_to_remove:
            state_dict.pop(k, None)
        return state_dict

    def _aggregate_experts(
        self,
        state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None,
    ) -> dict[str, Any]:
        """Aggregate per-expert weights (w1/w2/w3) into stacked gate_and_up/down tensors."""
        n_experts = self.moe_config.n_routed_experts

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_per_rank = end_expert - start_expert
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert = 0, n_experts
            expected_per_rank = n_experts
            rank = None

        # layer -> {"gate_and_up": {expert_id: {"w1": ..., "w3": ...}}, "down": {expert_id: tensor}}
        by_layer: dict[str, dict] = {}
        out: dict[str, Any] = {}

        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            m = _EXPERT_PATTERN.match(key)
            if m is None:
                out[key] = value
                continue

            layer_num, expert_num, which = m.group(1), int(m.group(2)), m.group(3)

            if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                continue

            if layer_num not in by_layer:
                by_layer[layer_num] = {"gate_and_up": {}, "down": {}}

            if which in ("w1", "w3"):
                if expert_num not in by_layer[layer_num]["gate_and_up"]:
                    by_layer[layer_num]["gate_and_up"][expert_num] = {}
                by_layer[layer_num]["gate_and_up"][expert_num][which] = value
            else:  # w2 = down_proj
                by_layer[layer_num]["down"][expert_num] = value

            # Once all experts for this layer's gate_and_up are ready, stack them.
            # The sub-dict is popped below, so later iterations that touch the
            # same layer (e.g. the paired w2 key) must tolerate its absence.
            gu_layer = by_layer[layer_num].get("gate_and_up")
            if gu_layer is not None:
                all_ready = len(gu_layer) == expected_per_rank and all(
                    isinstance(d, dict) and "w1" in d and "w3" in d for d in gu_layer.values()
                )
                if all_ready:
                    expert_ids = sorted(gu_layer.keys())
                    tensors = []
                    for eid in expert_ids:
                        gate_w = gu_layer[eid]["w1"]
                        up_w = gu_layer[eid]["w3"]
                        if is_dtensor(gate_w):
                            gate_w = gate_w.to_local()
                        if is_dtensor(up_w):
                            up_w = up_w.to_local()
                        tensors.append(torch.cat([gate_w.T, up_w.T], dim=-1))
                    stacked = torch.stack(tensors, dim=0).to(self.dtype)
                    native_key = f"model.layers.{layer_num}.mlp.experts.gate_and_up_projs"
                    out[native_key] = create_dtensor_from_local(stacked, device_mesh, rank)
                    del by_layer[layer_num]["gate_and_up"]

            # Once all experts for this layer's down are ready, stack them.
            down_layer = by_layer[layer_num].get("down")
            if down_layer is not None and len(down_layer) == expected_per_rank:
                expert_ids = sorted(down_layer.keys())
                tensors = []
                for eid in expert_ids:
                    w = down_layer[eid]
                    if is_dtensor(w):
                        w = w.to_local()
                    tensors.append(w.T)
                stacked = torch.stack(tensors, dim=0).to(self.dtype)
                native_key = f"model.layers.{layer_num}.mlp.experts.down_projs"
                out[native_key] = create_dtensor_from_local(stacked, device_mesh, rank)
                del by_layer[layer_num]["down"]

        return out

    def _rename_all(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply the HF->internal rename table to every key."""
        return {_rename_hf_key(k): v for k, v in state_dict.items()}

    # ------------------------------------------------------------------
    # to_hf
    # ------------------------------------------------------------------

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert internal state dict to HF V4 format.

        Splits stacked expert weights back to per-expert w1/w2/w3 tensors,
        applies key renaming in reverse, and optionally quantizes to FP8.
        """
        state_dict = self._drop_hash_layer_gate_bias(state_dict, _HashBiasScope.INTERNAL)

        hf_state_dict: dict[str, Any] = {}

        for fqn, tensor in state_dict.items():
            converted = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for hf_key, hf_val in converted:
                hf_state_dict[hf_key] = hf_val

        # Belt-and-suspenders: re-run the hash-layer bias filter on the HF-side
        # keys in case any intermediate step emitted them in HF format directly
        # (observed in practice during DCP load even after the internal-side drop).
        hf_state_dict = self._drop_hash_layer_gate_bias(hf_state_dict, _HashBiasScope.HF)
        return hf_state_dict

    def _checkpoint_num_hash_layers(self) -> int:
        """Read ``num_hash_layers`` directly from the checkpoint's config.json.

        We cannot rely on ``self.config.num_hash_layers`` alone: a YAML can
        legitimately override the model's hash-layer count to 0 (e.g. to
        disable hash routing in the forward path), but the on-disk checkpoint
        still has its original value and therefore still omits gate.bias for
        the first ``num_hash_layers`` layers.  To decide what to drop at load
        time we must know the checkpoint's own value.
        """
        import json as _json
        import os as _os

        ckpt_path = getattr(self.config, "_name_or_path", None) or getattr(self.config, "name_or_path", None)
        if not ckpt_path:
            return 0
        cfg_json = _os.path.join(ckpt_path, "config.json")
        if not _os.path.isfile(cfg_json):
            return 0
        try:
            with open(cfg_json) as f:
                data = _json.load(f)
        except Exception:
            return 0
        return int(data.get("num_hash_layers", 0) or 0)

    def _drop_hash_layer_gate_bias(self, state_dict: dict[str, Any], scope: "_HashBiasScope") -> dict[str, Any]:
        """The first ``num_hash_layers`` layers use hash-clustering routing and
        their HF checkpoint has no ``ffn.gate.bias`` / ``e_score_correction_bias``
        tensor.  The model side, however, creates the bias parameter uniformly
        for every layer (Automodel's generic Gate always materializes it when
        ``gate_bias_update_factor > 0``).  Drop those bias keys before load so
        DCP does not raise ``Missing key in checkpoint state_dict`` for them.

        ``scope`` selects which key format to match — the pre-rename internal
        form (``model.layers.{i}.mlp.gate.e_score_correction_bias``) or the
        post-rename HF form (``layers.{i}.ffn.gate.bias``).
        """
        # Prefer the checkpoint's own num_hash_layers over the (possibly YAML
        # overridden) model config — we need to match the on-disk layout.
        num_hash_layers = self._checkpoint_num_hash_layers()
        if num_hash_layers <= 0:
            num_hash_layers = int(getattr(self.config, "num_hash_layers", 0) or 0)
        if num_hash_layers <= 0:
            return state_dict
        hash_layer_ids = {str(i) for i in range(num_hash_layers)}
        pat = scope.value
        filtered: dict[str, Any] = {}
        for key, value in state_dict.items():
            m = pat.match(key)
            if m is not None and m.group(1) in hash_layer_ids:
                continue
            filtered[key] = value
        return filtered

    # Internal -> HF name table (inverse of _HF_TO_INTERNAL_RENAMES)
    _INTERNAL_TO_HF_RENAMES: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^model\.embed_tokens\.(.+)$"), r"embed.\1"),
        (re.compile(r"^model\.norm\.(.+)$"), r"norm.\1"),
        (re.compile(r"^lm_head\.(.+)$"), r"head.\1"),
        (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.(.+)$"), r"layers.\1.attn_norm.\2"),
        (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.(.+)$"), r"layers.\1.ffn_norm.\2"),
        (re.compile(r"^model\.layers\.(\d+)\.self_attn\.(.+)$"), r"layers.\1.attn.\2"),
        # Gate (bias correction key mapped back to `bias`)
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$"),
            r"layers.\1.ffn.gate.bias",
        ),
        (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.(.+)$"), r"layers.\1.ffn.gate.\2"),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.(.+)$"),
            r"layers.\1.ffn.shared_experts.w1.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.(.+)$"),
            r"layers.\1.ffn.shared_experts.w3.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.(.+)$"),
            r"layers.\1.ffn.shared_experts.w2.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.fc1_latent_proj\.(.+)$"),
            r"layers.\1.ffn.fc1_latent_proj.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.fc2_latent_proj\.(.+)$"),
            r"layers.\1.ffn.fc2_latent_proj.\2",
        ),
        (re.compile(r"^model\.layers\.(\d+)\.(hc_(?:attn|ffn)_(?:base|fn|scale))$"), r"layers.\1.\2"),
    ]

    def _internal_key_to_hf(self, key: str) -> str:
        for pattern, replacement in self._INTERNAL_TO_HF_RENAMES:
            new_key, n = pattern.subn(replacement, key)
            if n:
                return new_key
        return key

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        quantization = kwargs.get("quantization", False)
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        # Split stacked gate_and_up_projs into per-expert w1 + w3
        result = self._split_merged_expert(fqn, tensor)

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        # Rename internal keys to HF keys
        result = [(self._internal_key_to_hf(k), v) for k, v in result]

        if quantization:
            quantized = []
            for key, value in result:
                if key.endswith(".weight") and not self._is_non_quantized(key):
                    base = key[: -len(".weight")]
                    if self._is_expert_weight_key(key):
                        # V4 Flash routed experts are stored as FP4 e2m1 packed two
                        # values per int8 byte, with per-row / 32-col e8m0 scales.
                        # DCP validates shape + dtype against the checkpoint BEFORE
                        # dequantization happens, so the placeholders must match the
                        # on-disk layout exactly.  We emit empty tensors (content is
                        # overwritten by dcp.load) with the packed shape/dtype.
                        int8_val, e8m0_scale = self._build_fp4_expert_placeholders(value)
                        quantized.append((key, int8_val))
                        quantized.append((base + ".scale", e8m0_scale))
                        continue
                    if is_dtensor(value):
                        # Preserve DTensor structure so DCP knows the global shape
                        # and can shard the checkpoint load correctly.  Converting
                        # only the local shard to a plain tensor strips the mesh /
                        # placement metadata and causes a shape mismatch (e.g.
                        # local [128, 4096] vs checkpoint global [512, 4096]).
                        local_fp8 = value.to_local().to(torch.float8_e4m3fn)
                        fp8_val = DTensor.from_local(local_fp8, value.device_mesh, value.placements)
                    else:
                        fp8_val = value.cpu().to(torch.float8_e4m3fn)
                    scale = torch.ones(self._scale_shape(value), dtype=torch.float32)
                    quantized.append((key, fp8_val))
                    quantized.append((base + ".scale", scale))
                else:
                    quantized.append((key, value))
            return quantized

        return result

    @staticmethod
    def _build_fp4_expert_placeholders(value: Any) -> tuple[Any, Any]:
        """Return (int8 packed weight, float8_e8m0fnu scale) placeholders whose
        shapes / dtypes match the on-disk V4 Flash routed-expert layout.

        The current `value` is the dequantized bf16 tensor with shape [out, in];
        the checkpoint tensor is int8 [out, in // 2] with an e8m0 scale
        [out, in // 32].  DCP only uses these placeholders for shape/dtype
        validation and as the destination buffer — contents are overwritten on
        load, so we build empty tensors instead of re-packing real data.
        """
        if is_dtensor(value):
            local = value.to_local()
            in_dim = local.shape[-1]
            assert in_dim % FP4_COL_BLOCK == 0, f"V4 expert in-dim {in_dim} must be divisible by {FP4_COL_BLOCK}"
            packed = torch.empty(*local.shape[:-1], in_dim // 2, dtype=torch.int8, device=local.device)
            scale = torch.empty(
                *local.shape[:-1],
                in_dim // FP4_COL_BLOCK,
                dtype=torch.float8_e8m0fnu,
                device=local.device,
            )
            packed_d = DTensor.from_local(packed, value.device_mesh, value.placements)
            scale_d = DTensor.from_local(scale, value.device_mesh, value.placements)
            return packed_d, scale_d

        in_dim = value.shape[-1]
        assert in_dim % FP4_COL_BLOCK == 0, f"V4 expert in-dim {in_dim} must be divisible by {FP4_COL_BLOCK}"
        packed = torch.empty(*value.shape[:-1], in_dim // 2, dtype=torch.int8)
        scale = torch.empty(*value.shape[:-1], in_dim // FP4_COL_BLOCK, dtype=torch.float8_e8m0fnu)
        return packed, scale

    _NON_QUANTIZED_PATTERNS = [
        "attn_norm.weight",
        "ffn_norm.weight",
        "norm.weight",
        "head.weight",
        "embed.weight",
        "ffn.gate.weight",
        "ffn.gate.bias",
        "ffn.gate.tid2eid",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "attn.attn_sink",
        # Latent projections are stored as BF16 in the V4 checkpoint (not FP8).
        "ffn.fc1_latent_proj.weight",
        "ffn.fc2_latent_proj.weight",
    ]

    def _is_non_quantized(self, hf_key: str) -> bool:
        return any(pat in hf_key for pat in self._NON_QUANTIZED_PATTERNS)

    @staticmethod
    def _is_expert_weight_key(key: str) -> bool:
        return "ffn.experts." in key

    def _scale_shape(self, weight: torch.Tensor) -> tuple[int, int]:
        r, c = weight.shape
        return ((r + BLOCK_SIZE - 1) // BLOCK_SIZE, (c + BLOCK_SIZE - 1) // BLOCK_SIZE)

    def _expert_scale_shape(self, weight: torch.Tensor) -> tuple[int, int]:
        """Scale shape for an FP4 routed-expert weight tensor.

        The weight argument should be the *unpacked* tensor (in the model-side
        state dict, experts are already materialized at full dtype), so its
        last dim is the true `in` dim and the scale has `in // 32` columns.
        """
        r, c = weight.shape
        return (r, (c + FP4_COL_BLOCK - 1) // FP4_COL_BLOCK)

    @staticmethod
    def _dequantize_expert_fp4(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Unpack FP4 e2m1 packed-int8 weight and apply the per-row / 32-col e8m0 scale.

        Packed layout: `weight.int8` holds two FP4 values per byte — the low nibble
        at even column index, the high nibble at the following odd column — so the
        logical shape is `[out, weight.size(-1) * 2]`.
        """
        weight_local = weight.to_local() if is_dtensor(weight) else weight
        scale_local = scale.to_local() if is_dtensor(scale) else scale

        # Step 1: unpack two FP4 values from each byte.
        weight_u8 = weight_local.contiguous().view(torch.uint8)
        low = (weight_u8 & 0x0F).long()
        high = ((weight_u8 >> 4) & 0x0F).long()
        # Interleave (low, high) per byte so column indices match the original layout.
        table = _FP4_E2M1_TABLE.to(weight_u8.device)
        fp4_vals = torch.stack([table[low], table[high]], dim=-1).flatten(-2)  # [out, in]

        # Step 2: decode e8m0 scale to fp32. e8m0 stores 2^(e-127), or 0 when e==0.
        # A simple .to(torch.float32) works when PyTorch supports the e8m0 dtype;
        # fall back to the explicit formula otherwise.
        scale_u8 = scale_local.contiguous().view(torch.uint8).int()
        scale_f32 = torch.where(
            scale_u8 == 0,
            torch.zeros_like(scale_u8, dtype=torch.float32),
            torch.pow(2.0, (scale_u8 - 127).float()),
        )

        # Step 3: broadcast scale across the 32 columns it covers.
        scale_expanded = scale_f32.repeat_interleave(FP4_COL_BLOCK, dim=-1)
        scale_expanded = scale_expanded[..., : fp4_vals.shape[-1]]
        return (fp4_vals * scale_expanded).to(dtype)

    def _split_merged_expert(self, fqn: str, tensor: Any) -> list[tuple[str, Any]]:
        """Inverse of expert aggregation: split gate_and_up/down stacks into per-expert keys.

        Handles DTensor inputs (EP-sharded) by working on the local shard only,
        emitting keys only for the experts owned by the current rank.
        """
        gate_up_pat = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.gate_and_up_projs$")
        down_pat = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.down_projs$")

        m = gate_up_pat.match(fqn)
        if m:
            layer_num = m.group(2)
            n_total = self.moe_config.n_routed_experts
            local_tensor, start_eid, end_eid = get_expert_slice_for_rank(tensor, n_total)
            inter_dim = local_tensor.shape[-1] // 2
            result = []
            for local_i in range(local_tensor.shape[0]):
                t = local_tensor[local_i]  # [hidden_dim, 2*inter_dim]
                gate_t, up_t = t.split(inter_dim, dim=-1)
                eid = start_eid + local_i
                result.append((f"layers.{layer_num}.ffn.experts.{eid}.w1.weight", gate_t.T))
                result.append((f"layers.{layer_num}.ffn.experts.{eid}.w3.weight", up_t.T))
            return result

        m = down_pat.match(fqn)
        if m:
            layer_num = m.group(2)
            n_total = self.moe_config.n_routed_experts
            local_tensor, start_eid, end_eid = get_expert_slice_for_rank(tensor, n_total)
            result = []
            for local_i in range(local_tensor.shape[0]):
                eid = start_eid + local_i
                result.append((f"layers.{layer_num}.ffn.experts.{eid}.w2.weight", local_tensor[local_i].T))
            return result

        return [(fqn, tensor)]

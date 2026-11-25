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

"""State-dict conversion between the released ``tencent/Hy4-flash`` checkpoint and AutoModel.

The released safetensors use the *original* module names (the transformers reference renames them on load via
``conversion_mapping.py``). Per tensor:

==========================================================  ==========================================================
Released key (``model.layers.{L}.`` prefix unless noted)    AutoModel key
==========================================================  ==========================================================
``hc_attn_layer.hc_pre.hc_{fn,base,scale}``                 ``attn_hc.{fn,base,scale}``
``hc_mlp_layer.hc_pre.hc_{fn,base,scale}``                  ``ffn_hc.{fn,base,scale}``
``model.hc_head.hc_head_{fn,base,scale}``                   ``model.hc_head.hc_{fn,base,scale}``
``self_attn.learnable_sink_param``                          ``self_attn.sinks_param.weight``
``compressed_indexer.intra_group_ape``                      ``compressed_indexer.intra_group_ape.weight``
``self_attn.linear_gate.weight``                            ``self_attn.gate_proj.weight``
``self_attn.post_proj_norm.weight``                         ``post_self_attn_layernorm.weight``
``mlp.combine_output_norm.weight`` (MoE layers)             ``post_mlp_layernorm.weight``
``mlp.post_fc2_norm.weight`` (dense layer)                  ``post_mlp_layernorm.weight``
``mlp.routed_expert_down_proj.weight`` ([latent, hidden])   ``mlp.fc1_latent_proj.weight``
``mlp.routed_expert_up_proj.weight`` ([hidden, latent])     ``mlp.fc2_latent_proj.weight``
``mlp.experts.gate_up_proj`` [E, 2I, latent]                ``mlp.experts.gate_and_up_projs`` [E, latent, 2I]
``mlp.experts.down_proj`` [E, latent, I]                    ``mlp.experts.down_projs`` [E, I, latent]
==========================================================  ==========================================================

Every other language-model key (attention projections, indexer, router, shared expert, Engram, norms, embeddings,
``lm_head``) is identical. ``model.dfly_layers.*`` (the inference-only speculative drafter) is not loaded.
The Engram tables hold exactly ``sum(head vocab sizes)`` rows; the owner-sharded table pads them to a multiple of
the owner group size, so loading adds zero rows and saving exposes only the logical rows.
"""

from __future__ import annotations

import math
import re

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig

_LAYER = r"model\.layers\.(\d+)\."
# (released regex, native replacement); the native direction is the inverse of each pair.
_RENAMES: tuple[tuple[str, str], ...] = (
    (r"\.hc_attn_layer\.hc_pre\.hc_(fn|base|scale)$", r".attn_hc.\1"),
    (r"\.hc_mlp_layer\.hc_pre\.hc_(fn|base|scale)$", r".ffn_hc.\1"),
    (r"^model\.hc_head\.hc_head_(fn|base|scale)$", r"model.hc_head.hc_\1"),
    (r"\.self_attn\.learnable_sink_param$", ".self_attn.sinks_param.weight"),
    (r"\.compressed_indexer\.intra_group_ape$", ".compressed_indexer.intra_group_ape.weight"),
    (r"\.self_attn\.linear_gate\.weight$", ".self_attn.gate_proj.weight"),
    (r"\.self_attn\.post_proj_norm\.weight$", ".post_self_attn_layernorm.weight"),
    (r"\.mlp\.routed_expert_down_proj\.weight$", ".mlp.fc1_latent_proj.weight"),
    (r"\.mlp\.routed_expert_up_proj\.weight$", ".mlp.fc2_latent_proj.weight"),
)
# Native -> released, written out because the group references of the forward rules do not invert mechanically.
_INVERSE_RENAMES: tuple[tuple[str, str], ...] = (
    (r"\.attn_hc\.(fn|base|scale)$", r".hc_attn_layer.hc_pre.hc_\1"),
    (r"\.ffn_hc\.(fn|base|scale)$", r".hc_mlp_layer.hc_pre.hc_\1"),
    (r"^model\.hc_head\.hc_(fn|base|scale)$", r"model.hc_head.hc_head_\1"),
    (r"\.self_attn\.sinks_param\.weight$", ".self_attn.learnable_sink_param"),
    (r"\.compressed_indexer\.intra_group_ape\.weight$", ".compressed_indexer.intra_group_ape"),
    (r"\.self_attn\.gate_proj\.weight$", ".self_attn.linear_gate.weight"),
    (r"\.post_self_attn_layernorm\.weight$", ".self_attn.post_proj_norm.weight"),
    (r"\.mlp\.fc1_latent_proj\.weight$", ".mlp.routed_expert_down_proj.weight"),
    (r"\.mlp\.fc2_latent_proj\.weight$", ".mlp.routed_expert_up_proj.weight"),
)
_POST_MLP_NORM_RELEASED = re.compile(r"(model\.layers\.\d+)\.mlp\.(?:combine_output_norm|post_fc2_norm)\.weight$")
_POST_MLP_NORM_NATIVE = re.compile(r"(model\.layers\.(\d+))\.post_mlp_layernorm\.weight$")
_EXPERT = re.compile(_LAYER + r"mlp\.experts\.(gate_up_proj|down_proj)$")
_NATIVE_EXPERT = re.compile(_LAYER + r"mlp\.experts\.(gate_and_up_projs|down_projs)$")
_ENGRAM_TABLE = re.compile(_LAYER + r"engram_layer\.multi_head_embedding\.embedding\.weight$")
_SKIPPED = re.compile(r"(model\.dfly_layers\.|.*_extra_state)")


class HY4FlashStateDictAdapter(StateDictAdapter):
    """Bridge the released Hy4-flash language-model tensors and the AutoModel module layout."""

    def __init__(
        self, config: HY4FlashConfig, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        text = config.text_config
        self._engram_rows = {
            layer_id - 1: sum(size for order in text.engram_layer_head_vocab_sizes[str(layer_id)] for size in order)
            for layer_id in (text.engram_layer_ids or [])
        }

    # ------------------------------------------------------------------ key mapping
    def _to_native_key(self, key: str) -> str:
        norm = _POST_MLP_NORM_RELEASED.fullmatch(key)
        if norm:
            return f"{norm.group(1)}.post_mlp_layernorm.weight"
        for pattern, replacement in _RENAMES:
            key, count = re.subn(pattern, replacement, key)
            if count:
                break
        return key

    def _to_released_key(self, key: str) -> str:
        norm = _POST_MLP_NORM_NATIVE.fullmatch(key)
        if norm:
            released = (
                "post_fc2_norm"
                if self.config.text_config.mlp_layer_types[int(norm.group(2))] == "dense"
                else "combine_output_norm"
            )
            return f"{norm.group(1)}.mlp.{released}.weight"
        for pattern, replacement in _INVERSE_RENAMES:
            key, count = re.subn(pattern, replacement, key)
            if count:
                break
        return key

    # ------------------------------------------------------------------ Engram owner padding
    def _logical_engram_rows(self, tensor: torch.Tensor, layer: int) -> torch.Tensor:
        """View of the checkpoint rows of an owner-padded table.

        Args:
            tensor: Tensor of shape [padded_rows, head_dim]; a DTensor must be ``Shard(0)`` on a 1-D owner mesh.
            layer: Zero-based decoder layer of the table.

        Returns:
            Aliasing view of shape [rows, head_dim].
        """
        rows = self._engram_rows[layer]
        if not state_dict_utils.is_dtensor(tensor):
            return tensor[:rows]
        if tensor.device_mesh.ndim != 1 or tensor.placements != (Shard(0),):
            raise ValueError("Engram tables require Shard(0) on a one-dimensional owner mesh")
        local = tensor.to_local()
        start = tensor.device_mesh.get_local_rank() * math.ceil(tensor.shape[0] / tensor.device_mesh.size())
        valid = max(0, min(local.shape[0], rows - start))
        return DTensor.from_local(
            local[:valid],
            tensor.device_mesh,
            tensor.placements,
            shape=torch.Size((rows, tensor.shape[1])),
            stride=(tensor.shape[1], 1),
        )

    def _pad_engram_rows(self, tensor: torch.Tensor, layer: int) -> torch.Tensor:
        """Inverse of :meth:`_logical_engram_rows`: restore equal owner storage with zero rows.

        Args:
            tensor: Tensor of shape [rows, head_dim], optionally a ``Shard(0)`` DTensor with uneven local rows.

        Returns:
            Tensor of shape [padded_rows, head_dim].
        """
        if not state_dict_utils.is_dtensor(tensor):
            return tensor
        owners = tensor.device_mesh.size()
        local_rows = math.ceil(self._engram_rows[layer] / owners)
        local = tensor.to_local()
        if local.shape[0] < local_rows:
            local = torch.nn.functional.pad(local, (0, 0, 0, local_rows - local.shape[0]))
        return DTensor.from_local(
            local,
            tensor.device_mesh,
            tensor.placements,
            shape=torch.Size((local_rows * owners, tensor.shape[1])),
            stride=(tensor.shape[1], 1),
        )

    # ------------------------------------------------------------------ released -> native
    def from_hf(
        self, hf_state_dict: dict[str, torch.Tensor], device_mesh: DeviceMesh | None = None, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Convert released tensors to AutoModel names and layouts.

        Args:
            hf_state_dict: Released-name tensors. Stacked experts have shape [experts, out, in]; they are either
                complete CPU tensors (sliced here to this rank's expert range) or already expert-sharded DTensors.
            device_mesh: Expert mesh selecting this rank's experts.

        Returns:
            Native tensors; grouped experts have shape [experts, in, out].
        """
        n_experts = self.moe_config.n_routed_experts
        converted: dict[str, torch.Tensor] = {}
        for key in list(hf_state_dict):
            value = hf_state_dict.pop(key)
            if _SKIPPED.fullmatch(key):
                continue
            expert = _EXPERT.fullmatch(key)
            if expert:
                native = f"model.layers.{expert[1]}.mlp.experts." + (
                    "gate_and_up_projs" if expert[2] == "gate_up_proj" else "down_projs"
                )
                if state_dict_utils.is_dtensor(value):
                    converted[native] = value.transpose(1, 2)
                else:
                    start, end = (
                        state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
                        if device_mesh is not None
                        else (0, n_experts)
                    )
                    local = value[start:end].transpose(1, 2).to(self.dtype)
                    rank = (
                        None
                        if device_mesh is None
                        else state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                        if "ep" in device_mesh.mesh_dim_names
                        else device_mesh.get_rank()
                    )
                    converted[native] = (
                        state_dict_utils.create_dtensor_from_local(local, device_mesh, rank)
                        if device_mesh is not None
                        else local
                    )
                continue
            table = _ENGRAM_TABLE.fullmatch(key)
            if table:
                converted[key] = self._pad_engram_rows(value, int(table[1]))
                continue
            native = self._to_native_key(key)
            if native in converted:
                raise ValueError(f"Multiple checkpoint tensors map to {native}")
            converted[native] = value
        return converted

    # ------------------------------------------------------------------ native -> released
    def to_hf(
        self, state_dict: dict[str, torch.Tensor], exclude_key_regex: str | None = None, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Export native tensors under the released names and layouts."""
        out: dict[str, torch.Tensor] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
            ):
                out[key] = value
        return out

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: torch.Tensor, **kwargs: object
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert one native tensor.

        Args:
            fqn: Native parameter name.
            tensor: Tensor in native layout (grouped experts [experts, in, out]; Engram tables [padded_rows, dim]).

        Returns:
            One ``(released key, tensor)`` pair, or none for keys the release has no counterpart for.
        """
        if _SKIPPED.fullmatch(fqn):
            return []
        expert = _NATIVE_EXPERT.fullmatch(fqn)
        if expert:
            key = f"model.layers.{expert[1]}.mlp.experts." + (
                "gate_up_proj" if expert[2] == "gate_and_up_projs" else "down_proj"
            )
            result = [(key, tensor.transpose(1, 2))]
        else:
            key = self._to_released_key(fqn)
            table = _ENGRAM_TABLE.fullmatch(key)
            result = [(key, self._logical_engram_rows(tensor, int(table[1])) if table else tensor)]
        exclude = kwargs.get("exclude_key_regex")
        return [(k, v) for k, v in result if not (exclude and re.match(str(exclude), k))]

    def forced_hf_dtype_mapping(self, state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
        """Keep fp32 parameters (hyper-connections, sinks, router, indexer pooling, head) in fp32 on export."""
        return {
            self._to_released_key(key): "float32"
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32 and not _SKIPPED.fullmatch(key)
        }

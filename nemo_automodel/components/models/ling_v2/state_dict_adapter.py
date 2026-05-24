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

"""HF <-> NeMo state-dict adapter for BailingMoeV2 (Ling 2.0).

Handles the rename map between the HuggingFace checkpoint layout

    model.word_embeddings.weight
    model.layers.{N}.attention.query_key_value.weight      # fused [Q | K | V]
    model.layers.{N}.attention.dense.weight
    model.layers.{N}.attention.query_layernorm.weight
    model.layers.{N}.attention.key_layernorm.weight
    model.layers.{N}.mlp.gate.weight
    model.layers.{N}.mlp.gate.expert_bias
    model.layers.{N}.mlp.experts.{E}.{gate_proj,up_proj,down_proj}.weight
    model.layers.{N}.mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight

and the native NeMo layout used by this package

    model.embed_tokens.weight
    model.layers.{N}.self_attn.{q_proj,k_proj,v_proj,o_proj}.weight
    model.layers.{N}.self_attn.{q_norm,k_norm}.weight
    model.layers.{N}.mlp.gate.weight
    model.layers.{N}.mlp.gate.e_score_correction_bias
    model.layers.{N}.mlp.experts.{gate_and_up_projs,down_projs}
    model.layers.{N}.mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight

The per-expert grouping is delegated to ``MoESplitExpertsStateDictMixin``; this
adapter only normalises the surrounding key names and splits the fused QKV.
"""

import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

# Map of single-key renames applied in both directions.  Each tuple is
# (HF substring, native substring); replacement is whole-substring and
# stops after the first match.
_RENAME_PAIRS_HF_TO_NATIVE: tuple[tuple[str, str], ...] = (
    ("model.word_embeddings.", "model.embed_tokens."),
    (".attention.dense.", ".self_attn.o_proj."),
    (".attention.query_layernorm.", ".self_attn.q_norm."),
    (".attention.key_layernorm.", ".self_attn.k_norm."),
    (".mlp.gate.expert_bias", ".mlp.gate.e_score_correction_bias"),
)

_LAYER_QKV_RE = re.compile(r"^(?P<prefix>(?:.*\.)?layers\.\d+)\.attention\.query_key_value\.weight$")


def _rename_hf_to_native(key: str) -> str:
    for hf, native in _RENAME_PAIRS_HF_TO_NATIVE:
        if hf in key:
            return key.replace(hf, native)
    return key


def _rename_native_to_hf(key: str) -> str:
    # Reverse renames; order matters only for the expert_bias rule which is
    # the longest match and applied first to avoid the substring overlap with
    # ".mlp.gate.weight".
    for hf, native in _RENAME_PAIRS_HF_TO_NATIVE:
        if native in key:
            return key.replace(native, hf)
    return key


class BailingMoeV2StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """State-dict adapter for BailingMoeV2 / Ling 2.0 checkpoints."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    # ---- HF -> native ----------------------------------------------------

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        renamed = self._split_fused_qkv_and_rename(hf_state_dict)
        return self._from_hf_w_merged_experts(renamed, device_mesh)

    def _split_fused_qkv_and_rename(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Split each fused ``query_key_value`` weight into q/k/v and apply renames."""
        out: dict[str, Any] = {}
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim

        for key, tensor in hf_state_dict.items():
            m = _LAYER_QKV_RE.match(key)
            if m:
                expected = q_size + 2 * kv_size
                if tensor.shape[0] != expected:
                    raise ValueError(
                        f"Fused qkv weight {key} has shape[0]={tensor.shape[0]} but expected "
                        f"{expected} = num_heads({num_heads}) * head_dim({head_dim}) + 2 * "
                        f"num_kv_heads({num_kv_heads}) * head_dim({head_dim})."
                    )
                q, k, v = torch.split(tensor, [q_size, kv_size, kv_size], dim=0)
                prefix = m.group("prefix")
                out[f"{prefix}.self_attn.q_proj.weight"] = q.contiguous()
                out[f"{prefix}.self_attn.k_proj.weight"] = k.contiguous()
                out[f"{prefix}.self_attn.v_proj.weight"] = v.contiguous()
                continue
            out[_rename_hf_to_native(key)] = tensor

        return out

    # ---- native -> HF ----------------------------------------------------

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        del quantization  # Bailing MoE V2 ships BF16 only; no FP8 path.
        hf_state_dict: dict[str, Any] = {}

        # Collect q/k/v per layer so we can re-fuse them.
        pending_qkv: dict[str, dict[str, torch.Tensor]] = {}

        for fqn, tensor in state_dict.items():
            # Try expert merging first (these tensors live under .mlp.experts.*)
            converted = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
            if converted is not None:
                for k, v in converted:
                    hf_state_dict[k] = v
                continue

            m = re.match(r"^(?P<prefix>(?:.*\.)?layers\.\d+)\.self_attn\.(?P<proj>[qkv])_proj\.weight$", fqn)
            if m:
                pending_qkv.setdefault(m.group("prefix"), {})[m.group("proj")] = tensor
                continue

            hf_state_dict[_rename_native_to_hf(fqn)] = tensor

        for prefix, parts in pending_qkv.items():
            if {"q", "k", "v"} - parts.keys():
                # Partial set (e.g. only one rank shard available) — drop back to per-proj keys
                for proj, t in parts.items():
                    hf_state_dict[f"{prefix}.attention.{proj}_proj.weight"] = t
                continue
            fused = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            hf_state_dict[f"{prefix}.attention.query_key_value.weight"] = fused.contiguous()

        if exclude_key_regex:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.search(exclude_key_regex, k)}

        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single native tensor to HuggingFace format.

        ``q_proj`` / ``k_proj`` / ``v_proj`` tensors cannot be re-fused without
        their two siblings; the caller should batch them through :meth:`to_hf`
        instead.  This single-tensor path emits the per-projection HF key (which
        is **not** the standard fused name) so that the value is not silently
        dropped during DCP save adapters that walk tensors one-by-one.
        """
        converted = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if converted is not None:
            return converted

        m = re.match(r"^(?P<prefix>(?:.*\.)?layers\.\d+)\.self_attn\.(?P<proj>[qkv])_proj\.weight$", fqn)
        if m:
            return [(f"{m.group('prefix')}.attention.{m.group('proj')}_proj.weight", tensor)]

        return [(_rename_native_to_hf(fqn), tensor)]

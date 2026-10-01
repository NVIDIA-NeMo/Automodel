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

"""State dict conversion between the tencent/HunyuanImage-3.0 checkpoint and the native model.

On disk (release)                                         Native
  model.wte.weight                                          model.embed_tokens.weight
  model.layers.{L}.mlp.gate.wg.weight          [E, D]       model.layers.{L}.mlp.gate.weight
  model.layers.{L}.mlp.experts.{e}.gate_and_up_proj.weight  model.layers.{L}.mlp.experts.gate_and_up_projs  [E, D, 2I]
      [2I, D] = [up; gate] (activation on the second half)
  model.layers.{L}.mlp.experts.{e}.down_proj.weight         model.layers.{L}.mlp.experts.down_projs         [E, I, D]
  model.layers.{L}.mlp.shared_mlp.*                         model.layers.{L}.shared_mlp.*  (same fused layout)

Attention, norms, ``lm_head`` and every image-side module (``vae``, ``vision_model``, ``patch_embed``, ...) keep their
names. Decoder layers at index >= ``num_hidden_layers`` are dropped on load, so a model built with fewer layers can be
loaded from the full checkpoint.
"""

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

_LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
_HF_EXPERT_FUSED_RE = re.compile(r"^(?P<prefix>.*\.mlp\.experts\.\d+)\.gate_and_up_proj\.weight$")
_NATIVE_EXPERT_SPLIT_RE = re.compile(r"^(?P<prefix>.*\.mlp\.experts\.\d+)\.(?P<proj>gate_proj|up_proj)\.weight$")

_HF_TO_NATIVE = (
    (re.compile(r"^model\.wte\."), "model.embed_tokens."),
    (re.compile(r"\.mlp\.gate\.wg\.weight$"), ".mlp.gate.weight"),
    (re.compile(r"\.mlp\.shared_mlp\."), ".shared_mlp."),
)
_NATIVE_TO_HF = (
    (re.compile(r"^model\.embed_tokens\."), "model.wte."),
    (re.compile(r"\.mlp\.gate\.weight$"), ".mlp.gate.wg.weight"),
    (re.compile(r"\.shared_mlp\."), ".mlp.shared_mlp."),
)


def _rename(key: str, rules) -> str:
    for pattern, repl in rules:
        key, n = pattern.subn(repl, key)
        if n:
            break
    return key


class HunyuanImage3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Bridges the native grouped-experts layout and the HunyuanImage-3.0 release layout."""

    # The mixin's low-memory DCP path reads per-expert gate/up keys directly from disk, which this checkpoint
    # does not have (it stores them fused).
    _supports_low_memory_dcp_load = False

    def __init__(self, config: Any, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    def _is_dropped_layer(self, key: str) -> bool:
        m = _LAYER_RE.match(key)
        return bool(m and int(m.group(1)) >= self.config.num_hidden_layers)

    def from_hf(self, hf_state_dict: dict[str, Any], device_mesh: DeviceMesh | None = None, **kwargs) -> dict[str, Any]:
        native: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if self._is_dropped_layer(key):
                continue
            m = _HF_EXPERT_FUSED_RE.match(key)
            if m:
                up, gate = value.chunk(2, dim=0)
                native[f"{m['prefix']}.gate_proj.weight"] = gate
                native[f"{m['prefix']}.up_proj.weight"] = up
                continue
            native[_rename(key, _HF_TO_NATIVE)] = value
        return self._from_hf_w_merged_experts(native, device_mesh)

    def _fuse_and_rename(self, pairs: list[tuple[str, Any]], exclude_key_regex: str | None) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        pending: dict[str, dict[str, Any]] = {}
        for key, value in pairs:
            m = _NATIVE_EXPERT_SPLIT_RE.match(key)
            if m:
                parts = pending.setdefault(m["prefix"], {})
                parts[m["proj"]] = value
                if len(parts) == 2:
                    fused = torch.cat([parts["up_proj"], parts["gate_proj"]], dim=0)
                    out.append((f"{m['prefix']}.gate_and_up_proj.weight", fused))
                    del pending[m["prefix"]]
                continue
            out.append((_rename(key, _NATIVE_TO_HF), value))
        if pending:
            raise ValueError(f"Unpaired expert gate/up projections: {sorted(pending)}")
        if exclude_key_regex:
            out = [(k, v) for k, v in out if not re.match(exclude_key_regex, k)]
        return out

    @staticmethod
    def _no_inplace_views(kwargs: dict[str, Any]) -> dict[str, Any]:
        # The release stores each expert's projections fused and half-swapped ([up; gate]), which cannot be expressed
        # as a view of the grouped [gate | up] storage. Checkpoint loads therefore read into temporaries that
        # ``from_hf`` splits and merges, instead of DCP writing through views that this adapter would re-fuse into a
        # copy (leaving the model's gate/up storage unloaded).
        return {**kwargs, "for_checkpoint_load": False}

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: str | None = None, **kwargs) -> dict[str, Any]:
        split = self._to_hf_w_split_experts(state_dict, **self._no_inplace_views(kwargs))
        return dict(self._fuse_and_rename(list(split.items()), exclude_key_regex))

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        pairs = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **self._no_inplace_views(kwargs))
        if pairs is None:
            pairs = [(fqn, tensor)]
        return self._fuse_and_rename(pairs, kwargs.get("exclude_key_regex"))

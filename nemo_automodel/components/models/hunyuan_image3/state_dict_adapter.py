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

"""State dict conversion between the tencent/HunyuanImage-3.0 checkpoint and the Automodel implementation.

On disk (release)                                         Automodel
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

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hunyuan_image3.config import HunyuanImage3Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

_LAYER_RE = re.compile(r"^(?:model\.)?layers\.(\d+)\.")
_HF_EXPERT_FUSED_RE = re.compile(r"^(?P<prefix>.*\.mlp\.experts\.\d+)\.gate_and_up_proj\.weight$")
_SPLIT_EXPERT_RE = re.compile(r"^(?P<prefix>.*\.mlp\.experts\.\d+)\.(?P<proj>gate_proj|up_proj)\.weight$")

_HF_TO_AUTOMODEL = (
    (re.compile(r"^model\.wte\."), "model.embed_tokens."),
    (re.compile(r"\.mlp\.gate\.wg\.weight$"), ".mlp.gate.weight"),
    (re.compile(r"\.mlp\.shared_mlp\."), ".shared_mlp."),
)
_AUTOMODEL_TO_HF = (
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
    """Bridges Automodel's grouped-experts layout and the HunyuanImage-3.0 release layout."""

    # The mixin's low-memory DCP path reads per-expert gate/up keys directly from disk, which this checkpoint
    # does not have (it stores them fused).
    _supports_low_memory_dcp_load = False

    def __init__(
        self,
        config: HunyuanImage3Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    def _is_dropped_layer(self, key: str) -> bool:
        m = _LAYER_RE.match(key)
        return bool(m and int(m.group(1)) >= self.config.num_hidden_layers)

    def from_hf(
        self, hf_state_dict: dict[str, torch.Tensor], device_mesh: DeviceMesh | None = None, **kwargs: object
    ) -> dict[str, torch.Tensor]:
        converted: dict[str, torch.Tensor] = {}
        for key, value in hf_state_dict.items():
            if self._is_dropped_layer(key):
                continue
            m = _HF_EXPERT_FUSED_RE.match(key)
            if m:
                up, gate = value.chunk(2, dim=0)
                converted[f"{m['prefix']}.gate_proj.weight"] = gate
                converted[f"{m['prefix']}.up_proj.weight"] = up
                continue
            converted[_rename(key, _HF_TO_AUTOMODEL)] = value
        return self._from_hf_w_merged_experts(converted, device_mesh)

    def _fuse_and_rename(
        self, pairs: list[tuple[str, torch.Tensor]], exclude_key_regex: str | None
    ) -> list[tuple[str, torch.Tensor]]:
        out: list[tuple[str, torch.Tensor]] = []
        pending: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in pairs:
            m = _SPLIT_EXPERT_RE.match(key)
            if m:
                parts = pending.setdefault(m["prefix"], {})
                parts[m["proj"]] = value
                if len(parts) == 2:
                    fused = torch.cat([parts["up_proj"], parts["gate_proj"]], dim=0)
                    out.append((f"{m['prefix']}.gate_and_up_proj.weight", fused))
                    del pending[m["prefix"]]
                continue
            out.append((_rename(key, _AUTOMODEL_TO_HF), value))
        if pending:
            raise ValueError(f"Unpaired expert gate/up projections: {sorted(pending)}")
        if exclude_key_regex:
            out = [(k, v) for k, v in out if not re.match(exclude_key_regex, k)]
        return out

    def _fused_gate_up_load_destinations(self, fqn: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Checkpoint-load destinations for one layer's grouped ``gate_and_up_projs``.

        The release stores each expert's projections fused and half-swapped (``[up; gate]``), which cannot be a view
        of the grouped ``[gate | up]`` storage. DCP therefore reads them into empty host tensors (one per local
        expert, so no extra GPU memory), and ``from_hf`` splits and merges them into the model. The key is not marked
        as loaded in place, so the merge runs.
        """
        self._split_experts_weights(tensor, self.moe_config.n_routed_experts)
        dim = self.moe_config.dim
        inter = self.moe_config.moe_inter_dim
        base = fqn[: -len("gate_and_up_projs")]
        dtype = tensor.dtype
        return [
            (f"{base}{expert_id}.gate_and_up_proj.weight", torch.empty(2 * inter, dim, dtype=dtype, device="cpu"))
            for expert_id in self._last_expert_ids
        ]

    def to_hf(
        self, state_dict: dict[str, torch.Tensor], exclude_key_regex: str | None = None, **kwargs: object
    ) -> dict[str, torch.Tensor]:
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
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if kwargs.get("for_checkpoint_load") and fqn.endswith(".mlp.experts.gate_and_up_projs"):
            pairs = self._fused_gate_up_load_destinations(fqn, tensor)
            if exclude_key_regex:
                pairs = [(k, v) for k, v in pairs if not re.match(exclude_key_regex, k)]
            return pairs
        # down_projs keep the release layout, so the mixin can still load them through in-place views.
        pairs = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if pairs is None:
            pairs = [(fqn, tensor)]
        return self._fuse_and_rename(pairs, exclude_key_regex)

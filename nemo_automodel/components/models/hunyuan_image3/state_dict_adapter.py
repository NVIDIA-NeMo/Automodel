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

"""State dict conversion between the tencent/HunyuanImage-3.0 checkpoint and Automodel's native layout.

Released checkpoint (HF)                                   Native
  model.wte.weight                                           model.embed_tokens.weight
  model.ln_f.weight                                          model.norm.weight
  model.layers.{L}.mlp.gate.wg.weight          [E, H]        model.layers.{L}.mlp.gate.weight
  model.layers.{L}.mlp.shared_mlp.*                          model.layers.{L}.mlp.shared_experts.*
  model.layers.{L}.mlp.experts.{e}.gate_and_up_proj.weight   model.layers.{L}.mlp.experts.gate_and_up_projs
        [2I, H], rows = [up; gate]                                 [E, H, 2I], columns = [gate | up]
  model.layers.{L}.mlp.experts.{e}.down_proj.weight [H, I]   model.layers.{L}.mlp.experts.down_projs [E, I, H]

The routed experts go through the shared per-expert split / merge of ``MoESplitExpertsStateDictMixin``, which works
with separate ``gate_proj`` / ``up_proj`` keys; this adapter fuses them into the released ``[up; gate]`` tensor on
the way out and splits them on the way in. On a checkpoint load the mixin hands out views into the model weight,
but DCP cannot write one fused checkpoint tensor through two views, so each fused tensor gets a host buffer whose
halves ``from_hf`` copies into the views; the grouped tensor then counts as loaded in place.

The VAE (``vae.*``) and the vision encoder (``vision_model.*``, ``vision_aligner.*``) of the release are not part of
the training model; their keys are dropped on load and absent on save.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

# (native regex, released regex); each side is both a pattern and the other's replacement.
_RENAMES: tuple[tuple[str, str], ...] = (
    (r"^model\.embed_tokens\.weight$", r"^model\.wte\.weight$"),
    (r"^model\.norm\.weight$", r"^model\.ln_f\.weight$"),
    (r"\.mlp\.gate\.weight$", r"\.mlp\.gate\.wg\.weight$"),
    (r"\.mlp\.shared_experts\.", r"\.mlp\.shared_mlp\."),
)


def _rename_table(src: int, dst: int) -> tuple[tuple[re.Pattern[str], str], ...]:
    return tuple((re.compile(pair[src]), re.sub(r"[\\^$]", "", pair[dst])) for pair in _RENAMES)


_NATIVE_TO_HF_RENAMES = _rename_table(0, 1)
_HF_TO_NATIVE_RENAMES = _rename_table(1, 0)
# Release modules that the training model does not contain.
_UNUSED_HF_PREFIXES = ("vae.", "vision_model.", "vision_aligner.")

_SPLIT_EXPERT_KEY = re.compile(r"^(?P<stem>.*\.mlp\.experts\.\d+)\.(?P<proj>gate_proj|up_proj)\.weight$")
_FUSED_EXPERT_KEY = re.compile(r"^(?P<stem>.*\.mlp\.experts\.\d+)\.gate_and_up_proj\.weight$")


def _rename(key: str, renames: tuple[tuple[re.Pattern[str], str], ...]) -> str:
    for pattern, replacement in renames:
        new_key, count = pattern.subn(replacement, key)
        if count:
            return new_key
    return key


def _all_alias(pairs: list[tuple[str, Any]], tensor: Any) -> bool:
    """Check whether plain checkpoint destinations alias the grouped model weight.

    Args:
        pairs: Per-expert entries with tensors of shape [expert_hidden, hidden]. DTensors retain a remaining
            mesh dimension and must use the distributed conversion path, even when their local storage aliases.
        tensor: Grouped tensor of shape [experts, hidden, 2 * expert_hidden], possibly a DTensor.

    Returns:
        Whether every destination is a plain tensor aliasing the source's local storage.
    """
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    if local.is_meta:
        return False
    storage = local.untyped_storage().data_ptr()
    return all(
        isinstance(value, torch.Tensor)
        and not isinstance(value, DTensor)
        and not value.is_meta
        and value.untyped_storage().data_ptr() == storage
        for _, value in pairs
    )


def _group_split_experts(pairs: list[tuple[str, Any]]) -> tuple[dict[str, tuple[Any, Any]], list[tuple[str, Any]]]:
    """Pair per-expert ``up_proj`` / ``gate_proj`` entries by expert.

    Args:
        pairs: ``(key, tensor)`` entries; the gate/up tensors have shape [expert_hidden, hidden].

    Returns:
        ``({expert stem: (up, gate)}, other entries)``.
    """
    experts: dict[str, dict[str, Any]] = {}
    others: list[tuple[str, Any]] = []
    for key, value in pairs:
        match = _SPLIT_EXPERT_KEY.match(key)
        if match is None:
            others.append((key, value))
        else:
            experts.setdefault(match.group("stem"), {})[match.group("proj")] = value
    incomplete = [stem for stem, parts in experts.items() if len(parts) != 2]
    if incomplete:
        raise ValueError(f"Expert gate/up halves without a partner: {sorted(incomplete)}")
    return {stem: (parts["up_proj"], parts["gate_proj"]) for stem, parts in experts.items()}, others


class HunyuanImage3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between the released HunyuanImage-3.0 checkpoint and the native grouped-expert model."""

    def __init__(self, config: Any, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        # Checkpoint-load views awaiting their DCP host buffer, keyed by released fused key: (up view, gate view).
        self._fused_load_destinations: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def map_peft_target_module_to_hf(self, name: str, *, v4_compatible: bool = False) -> str:
        """Match PEFT target names to the released fused projections.

        Args:
            name: Target-module path after the shared exporter expands combined projections.
            v4_compatible: Legacy export selection; both formats use the same released module names.

        Returns:
            Released module path, with shared experts renamed and split QKV targets reunited.
        """
        name = _rename(name, _NATIVE_TO_HF_RENAMES)
        return re.sub(r"\.self_attn\.(?:q_proj|k_proj|v_proj)$", ".self_attn.qkv_proj", name)

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Convert a native state dict to released checkpoint keys.

        With ``for_checkpoint_load=True`` the fused expert entries become host buffers for DCP (one
        [2 * expert_hidden, hidden] tensor per local expert, 50 MB in bf16 for the release, about 13 GB per rank with
        8 local experts) that :meth:`from_hf` copies into the model weight; a new load conversion forgets the views
        of an earlier one.
        """
        if kwargs.get("for_checkpoint_load"):
            self._fused_load_destinations = {}
        out: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
            ):
                out[key] = value
        return out

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one native tensor to one or more released checkpoint entries.

        Args:
            fqn: Native parameter name.
            tensor: Native tensor; grouped ``gate_and_up_projs`` have shape [experts, hidden, 2 * expert_hidden]
                with columns ``[gate | up]`` and ``down_projs`` [experts, expert_hidden, hidden] (DTensors sharded
                on the expert axis under expert parallelism).
            **kwargs: Forwarded to the shared expert split (``exclude_key_regex`` filters the output keys).

        Returns:
            ``(key, tensor)`` entries in the released layout: per local expert, ``gate_and_up_proj`` of shape
            [2 * expert_hidden, hidden] (rows ``[up; gate]``) and ``down_proj`` of shape [hidden, expert_hidden];
            other tensors are renamed only. On a checkpoint load whose split entries are views into the model
            weight, the fused entries are host buffers that :meth:`from_hf` copies into those views; otherwise they
            are new contiguous tensors.
        """
        exclude_key_regex = kwargs.pop("exclude_key_regex", None)
        pairs = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if pairs is None:
            pairs = [(fqn, tensor)]
        else:
            in_place = kwargs.get("for_checkpoint_load", False) and _all_alias(pairs, tensor)
            experts, pairs = _group_split_experts(pairs)
            for stem, (up, gate) in experts.items():
                fused_key = _rename(f"{stem}.gate_and_up_proj.weight", _NATIVE_TO_HF_RENAMES)
                if in_place:
                    fused = torch.empty((up.shape[0] + gate.shape[0], up.shape[1]), dtype=up.dtype, device="cpu")
                    self._fused_load_destinations[fused_key] = (up, gate)
                else:
                    fused = torch.cat([up, gate], dim=0)
                pairs.append((fused_key, fused))
        out = []
        for key, value in pairs:
            key = _rename(key, _NATIVE_TO_HF_RENAMES)
            if exclude_key_regex and re.match(exclude_key_regex, key):
                continue
            out.append((key, value))
        return out

    def from_hf(
        self, hf_state_dict: dict[str, Any], device_mesh: DeviceMesh | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Convert released checkpoint entries to the native layout (local experts only under EP).

        Args:
            hf_state_dict: Released entries; consumed (popped) by this call. Per-expert ``gate_and_up_proj`` has
                shape [2 * expert_hidden, hidden] with rows ``[up; gate]``.
            device_mesh: Mesh whose ``ep`` axis selects the local experts, or ``None`` for all experts.
            **kwargs: Unused; accepted for the base-class signature.

        Returns:
            Native state dict; grouped ``gate_and_up_projs`` of shape [local_experts, hidden, 2 * expert_hidden]
            (columns ``[gate | up]``) and ``down_projs`` of shape [local_experts, expert_hidden, hidden]. Grouped
            tensors loaded in place through DCP destinations are absent (see ``view_loaded_native_keys``).
        """
        native: dict[str, Any] = {}
        intermediate = self.moe_config.moe_inter_dim
        destinations, self._fused_load_destinations = self._fused_load_destinations, {}
        for key in list(hf_state_dict):
            value = hf_state_dict.pop(key)
            if key.startswith(_UNUSED_HF_PREFIXES):
                continue
            fused = _FUSED_EXPERT_KEY.match(key)
            if fused is None:
                native[_rename(key, _HF_TO_NATIVE_RENAMES)] = value
                continue
            if value.shape[0] != 2 * intermediate:
                raise ValueError(f"{key}: expected {2 * intermediate} rows, got {tuple(value.shape)}")
            up, gate = value[:intermediate], value[intermediate:]
            if key in destinations:
                # DCP filled the host buffer: move the halves into the model weight and hand the mixin the split
                # keys it expects for an in-place loaded group.
                views = destinations[key]
                with torch.no_grad():
                    views[0].copy_(up)
                    views[1].copy_(gate)
                up, gate = views
            stem = fused.group("stem")
            native[f"{stem}.up_proj.weight"] = up
            native[f"{stem}.gate_proj.weight"] = gate
        return self._from_hf_w_merged_experts(native, device_mesh)

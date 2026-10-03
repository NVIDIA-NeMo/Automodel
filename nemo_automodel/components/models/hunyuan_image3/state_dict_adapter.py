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
the way out and splits them on the way in. The fused tensors are new storage, so the gate/up conversion never asks
DCP to write through views of the model weights; ``from_hf`` rebuilds those grouped tensors after the read.

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

_NATIVE_TO_HF_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^model\.embed_tokens\.weight$"), "model.wte.weight"),
    (re.compile(r"^model\.norm\.weight$"), "model.ln_f.weight"),
    (re.compile(r"\.mlp\.gate\.weight$"), ".mlp.gate.wg.weight"),
    (re.compile(r"\.mlp\.shared_experts\."), ".mlp.shared_mlp."),
)
_HF_TO_NATIVE_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^model\.wte\.weight$"), "model.embed_tokens.weight"),
    (re.compile(r"^model\.ln_f\.weight$"), "model.norm.weight"),
    (re.compile(r"\.mlp\.gate\.wg\.weight$"), ".mlp.gate.weight"),
    (re.compile(r"\.mlp\.shared_mlp\."), ".mlp.shared_experts."),
)
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
    """Whether every entry is a view into ``tensor``'s (local) storage, i.e. a checkpoint-load view."""
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    if local.is_meta:
        return False
    storage = local.untyped_storage().data_ptr()
    return all(
        isinstance(value, torch.Tensor) and not value.is_meta and value.untyped_storage().data_ptr() == storage
        for _, value in pairs
    )


def _fuse_gate_up(pairs: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
    """Merge per-expert ``gate_proj`` / ``up_proj`` entries into the released ``gate_and_up_proj``.

    Args:
        pairs: ``(key, tensor)`` entries. Per-expert ``gate_proj`` / ``up_proj`` tensors have shape
            [expert_hidden, hidden]; other entries pass through unchanged.

    Returns:
        Entries where each expert's gate/up pair became one new tensor of shape [2 * expert_hidden, hidden] whose
        rows are ``[up; gate]``.
    """
    out: list[tuple[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    for key, value in pairs:
        match = _SPLIT_EXPERT_KEY.match(key)
        if match is None:
            out.append((key, value))
            continue
        parts = pending.setdefault(match.group("stem"), {})
        parts[match.group("proj")] = value
        if len(parts) == 2:
            stem = match.group("stem")
            out.append((f"{stem}.gate_and_up_proj.weight", torch.cat([parts["up_proj"], parts["gate_proj"]], dim=0)))
            del pending[stem]
    if pending:
        raise ValueError(f"Expert gate/up halves without a partner: {sorted(pending)}")
    return out


class HunyuanImage3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between the released HunyuanImage-3.0 checkpoint and the native grouped-expert model."""

    def __init__(self, config: Any, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        # Fused checkpoint-load destinations handed to DCP, keyed by released key: (buffer, up view, gate view).
        self._fused_load_destinations: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _fused_load_destination_pairs(self, fqn: str, pairs: list[tuple[str, Any]]) -> list[tuple[str, Any]]:
        """Replace in-place gate/up views with one fused DCP destination per expert.

        The mixin registered ``fqn`` as loaded in place and returned, per local expert, ``gate_proj`` / ``up_proj``
        views into the model's grouped weight. The checkpoint stores both halves in one ``[2 * expert_hidden,
        hidden]`` tensor, which DCP cannot write through two separate views, so DCP gets a host buffer of that shape
        and :meth:`from_hf` copies its halves into the views afterwards.

        Args:
            fqn: Native name of the grouped ``gate_and_up_projs`` tensor.
            pairs: Per-expert ``(key, view)`` entries from the mixin; ``gate_proj`` / ``up_proj`` views have shape
                [expert_hidden, hidden] and alias the model weight.

        Returns:
            Per expert, one ``(fused key, buffer)`` entry whose buffer has shape [2 * expert_hidden, hidden].
        """
        views: dict[str, dict[str, torch.Tensor]] = {}
        for key, view in pairs:
            match = _SPLIT_EXPERT_KEY.match(key)
            if match is None:
                raise ValueError(f"Unexpected entry {key!r} while splitting {fqn}")
            views.setdefault(match.group("stem"), {})[match.group("proj")] = view
        out = []
        for stem, parts in views.items():
            up, gate = parts["up_proj"], parts["gate_proj"]
            buffer = torch.empty((up.shape[0] + gate.shape[0], up.shape[1]), dtype=up.dtype, device="cpu")
            fused_key = _rename(f"{stem}.gate_and_up_proj.weight", _NATIVE_TO_HF_RENAMES)
            self._fused_load_destinations[fused_key] = (buffer, up, gate)
            out.append((fused_key, buffer))
        return out

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Convert a native state dict to released checkpoint keys."""
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
            other tensors are renamed only. On a checkpoint load the fused entries are host buffers that
            :meth:`from_hf` copies into the model weight (see :meth:`_fused_load_destination_pairs`); otherwise
            they are new contiguous tensors.
        """
        exclude_key_regex = kwargs.pop("exclude_key_regex", None)
        pairs = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if pairs is None:
            pairs = [(fqn, tensor)]
        elif fqn.endswith(".gate_and_up_projs") and _all_alias(pairs, tensor):
            pairs = self._fused_load_destination_pairs(fqn, pairs)
        else:
            pairs = _fuse_gate_up(pairs)
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
            if fused is not None and key in destinations:
                # DCP filled the fused host buffer; move its halves into the model weight and hand the mixin the
                # split keys it expects for an in-place loaded group.
                _, up, gate = destinations[key]
                if value.shape != (up.shape[0] + gate.shape[0], up.shape[1]):
                    raise ValueError(
                        f"{key}: expected shape {(up.shape[0] + gate.shape[0], up.shape[1])}, got {tuple(value.shape)}"
                    )
                with torch.no_grad():
                    up.copy_(value[: up.shape[0]])
                    gate.copy_(value[up.shape[0] :])
                stem = fused.group("stem")
                native[f"{stem}.up_proj.weight"] = up
                native[f"{stem}.gate_proj.weight"] = gate
                continue
            if fused is not None:
                if value.shape[0] != 2 * intermediate:
                    raise ValueError(f"{key}: expected {2 * intermediate} rows, got {tuple(value.shape)}")
                stem = fused.group("stem")
                native[f"{stem}.up_proj.weight"] = value[:intermediate]
                native[f"{stem}.gate_proj.weight"] = value[intermediate:]
                continue
            native[_rename(key, _HF_TO_NATIVE_RENAMES)] = value
        return self._from_hf_w_merged_experts(native, device_mesh)

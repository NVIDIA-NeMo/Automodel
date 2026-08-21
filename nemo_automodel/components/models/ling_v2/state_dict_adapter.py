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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadGroup, StateDictAdapter
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
_NATIVE_LAYER_QKV_RE = re.compile(r"^(?P<prefix>(?:.*\.)?layers\.\d+)\.self_attn\.(?P<projection>[qkv])_proj\.weight$")
_STREAMABLE_EXPERT_WEIGHT = re.compile(
    r"^(?P<root>(?:.*\.)?layers\.\d+\.mlp\.experts)\."
    r"(?P<projection>gate_and_up_projs|down_projs)$"
)


@dataclass
class _LingQKVGroupBuilder:
    """Mutable builder for one fused-QKV checkpoint dependency group.

    Attributes:
        projections: Mapping from ``q``, ``k``, and ``v`` to a native parameter name and its final tensor. Query
            tensors have shape [q_hidden, hidden]; key and value tensors have shape [kv_hidden, hidden].
    """

    projections: dict[str, tuple[str, torch.Tensor]]


@dataclass
class _LingExpertGroupBuilder:
    """Mutable builder for one split-expert checkpoint dependency group.

    Attributes:
        destinations: Checkpoint gate/up views of shape [expert_hidden, hidden] and down views of shape
            [hidden, expert_hidden]. Every view uses the grouped model weights as its memory.
        native_keys: Names of the final grouped expert parameters completed by ``destinations``.
        projections: Native grouped projection names already added to the builder.
    """

    destinations: dict[str, torch.Tensor]
    native_keys: set[str]
    projections: set[str]


@dataclass
class _LingFusedQKVLoadGroup(CheckpointLoadGroup):
    """One fused QKV checkpoint tensor and the three model weights it fills.

    Attributes:
        destinations: One checkpoint tensor of shape [q_heads * head_dim + 2 * kv_heads * head_dim, hidden]. The
            tensor is temporary storage released before the next checkpoint group is requested.
        native_keys: Names of the three final Q, K, and V parameters completed by this group.
        q_proj: Final query parameter of shape [q_heads * head_dim, hidden], mutated by :meth:`install`.
        k_proj: Final key parameter of shape [kv_heads * head_dim, hidden], mutated by :meth:`install`.
        v_proj: Final value parameter of shape [kv_heads * head_dim, hidden], mutated by :meth:`install`.
    """

    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor

    @torch.no_grad()
    def install(self) -> None:
        """Split the loaded fused tensor and copy its Q, K, and V slices into final parameter storage."""
        if len(self.destinations) != 1:
            raise ValueError(f"Ling fused-QKV group expected one checkpoint destination, got {len(self.destinations)}")
        fused_qkv = next(iter(self.destinations.values()))
        expected_shape = (self.q_proj.shape[0] + self.k_proj.shape[0] + self.v_proj.shape[0], self.q_proj.shape[1])
        if tuple(fused_qkv.shape) != expected_shape:
            raise ValueError(
                f"Ling fused-QKV destination has shape {tuple(fused_qkv.shape)}, expected {expected_shape}"
            )

        q_value, k_value, v_value = torch.split(
            fused_qkv,
            [self.q_proj.shape[0], self.k_proj.shape[0], self.v_proj.shape[0]],
            dim=0,
        )
        self.q_proj.copy_(q_value)
        self.k_proj.copy_(k_value)
        self.v_proj.copy_(v_value)


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

    @property
    def supports_checkpoint_load_groups(self) -> bool:
        """Whether Ling can load single-device QKV and expert transformations one layer at a time."""
        return (
            self.backend.experts in {"torch", "torch_mm"}
            and self.backend.dispatcher == "torch"
            and not self.moe_config.expert_bias
        )

    def _add_expert_checkpoint_views(
        self,
        builder: _LingExpertGroupBuilder,
        *,
        expert_root: str,
        native_name: str,
        projection: str,
        grouped_weight: torch.Tensor,
    ) -> None:
        """Add split HF views of one final grouped expert parameter.

        Args:
            builder: Mutable dependency group that retains the produced final-storage views.
            expert_root: Checkpoint prefix ending in ``layers.{layer}.mlp.experts``.
            native_name: Native name completed by ``grouped_weight``.
            projection: Either ``gate_and_up_projs`` or ``down_projs``.
            grouped_weight: Final gate/up tensor of shape [experts, hidden, 2 * expert_hidden] or final down tensor of
                shape [experts, expert_hidden, hidden]. The emitted checkpoint destinations are non-contiguous views
                that use this model weight memory.

        Raises:
            ValueError: If the projection is duplicated or its tensor shape is incompatible with the Ling layout.
        """
        if projection in builder.projections:
            raise ValueError(f"Duplicate Ling expert projection {expert_root}.{projection}")

        n_experts = self.moe_config.n_routed_experts
        expert_hidden = self.moe_config.moe_inter_dim
        if projection == "gate_and_up_projs":
            expected_shape = (n_experts, self.moe_config.dim, 2 * expert_hidden)
        else:
            expected_shape = (n_experts, expert_hidden, self.moe_config.dim)
        if tuple(grouped_weight.shape) != expected_shape:
            raise ValueError(
                f"Ling grouped expert parameter {native_name} has shape {tuple(grouped_weight.shape)}, "
                f"expected {expected_shape}"
            )

        checkpoint_views = self._checkpoint_load_views(native_name, grouped_weight)
        duplicate_keys = sorted(checkpoint_views.keys() & builder.destinations.keys())
        if duplicate_keys:
            raise ValueError(
                f"Duplicate Ling expert checkpoint destinations for {expert_root} (examples={duplicate_keys[:5]})"
            )
        builder.destinations.update(checkpoint_views)

        builder.native_keys.add(native_name)
        builder.projections.add(projection)

    @staticmethod
    def _build_qkv_load_group(prefix: str, builder: _LingQKVGroupBuilder) -> _LingFusedQKVLoadGroup:
        """Allocate one fused checkpoint tensor backed by three final projection parameters.

        Args:
            prefix: Native layer prefix ending in ``layers.{layer}``.
            builder: Q/K/V parameters with shapes [q_hidden, hidden], [kv_hidden, hidden], and [kv_hidden, hidden].

        Returns:
            Load group with one temporary tensor of shape [q_hidden + 2 * kv_hidden, hidden]. Its installation copies
            the three row ranges into the final Q/K/V parameters.

        Raises:
            ValueError: If a projection is missing or the final parameters disagree in rank, shape, device, or dtype.
        """
        missing_projections = {"q", "k", "v"} - builder.projections.keys()
        if missing_projections:
            raise ValueError(f"Incomplete Ling fused-QKV group {prefix}: missing {sorted(missing_projections)}")
        q_name, q_proj = builder.projections["q"]
        k_name, k_proj = builder.projections["k"]
        v_name, v_proj = builder.projections["v"]
        if q_proj.ndim != 2 or k_proj.ndim != 2 or v_proj.ndim != 2:
            raise ValueError(
                f"Ling fused-QKV group {prefix} requires rank-2 parameters, got "
                f"Q={tuple(q_proj.shape)}, K={tuple(k_proj.shape)}, V={tuple(v_proj.shape)}"
            )
        if k_proj.shape != v_proj.shape or q_proj.shape[1] != k_proj.shape[1]:
            raise ValueError(
                f"Ling fused-QKV group {prefix} has incompatible shapes "
                f"Q={tuple(q_proj.shape)}, K={tuple(k_proj.shape)}, V={tuple(v_proj.shape)}"
            )
        if q_proj.device != k_proj.device or q_proj.device != v_proj.device:
            raise ValueError(f"Ling fused-QKV group {prefix} spans multiple devices")
        if q_proj.dtype != k_proj.dtype or q_proj.dtype != v_proj.dtype:
            raise ValueError(f"Ling fused-QKV group {prefix} spans multiple dtypes")

        checkpoint_name = f"{prefix}.attention.query_key_value.weight"
        fused_destination = torch.empty(
            (q_proj.shape[0] + k_proj.shape[0] + v_proj.shape[0], q_proj.shape[1]),
            dtype=q_proj.dtype,
            device=q_proj.device,
        )
        return _LingFusedQKVLoadGroup(
            destinations={checkpoint_name: fused_destination},
            native_keys=frozenset({q_name, k_name, v_name}),
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
        )

    @staticmethod
    def _build_expert_load_group(
        expert_root: str,
        builder: _LingExpertGroupBuilder,
        n_experts: int,
    ) -> CheckpointLoadGroup:
        """Validate and freeze one complete layer of final-storage expert destinations.

        Args:
            expert_root: Checkpoint prefix ending in ``layers.{layer}.mlp.experts``.
            builder: Per-expert gate, up, and down checkpoint views. Gate/up views have shape [expert_hidden, hidden],
                down views have shape [hidden, expert_hidden], and all views use the grouped model weights as memory.
            n_experts: Number of checkpoint experts required for the layer.

        Returns:
            No-op installation group containing exactly three destinations per expert.

        Raises:
            ValueError: If either grouped projection or any split expert destination is missing.
        """
        missing_projections = {"gate_and_up_projs", "down_projs"} - builder.projections
        if missing_projections:
            raise ValueError(f"Incomplete Ling expert group {expert_root}: missing {sorted(missing_projections)}")
        expected_destinations = 3 * n_experts
        if len(builder.destinations) != expected_destinations:
            raise ValueError(
                f"Ling expert group {expert_root} has {len(builder.destinations)} checkpoint destinations, "
                f"expected {expected_destinations}"
            )
        return CheckpointLoadGroup(
            destinations=builder.destinations,
            native_keys=frozenset(builder.native_keys),
        )

    def iter_checkpoint_load_groups(
        self,
        checkpoint_state: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
    ) -> Iterator[CheckpointLoadGroup]:
        """Yield Ling checkpoint tensors in groups that are installed before the next group is read.

        Args:
            checkpoint_state: Canonical native names and final Ling model tensors. Attention Q/K/V parameters have
                shapes [q_hidden, hidden], [kv_hidden, hidden], and [kv_hidden, hidden]. Grouped gate/up experts have
                shape [experts, hidden, 2 * expert_hidden], and grouped down experts have shape
                [experts, expert_hidden, hidden].
            device_mesh: Must be ``None``. Distributed Ling loads continue to use the rank-sharded DCP path.

        Returns:
            Iterator whose ordinary and expert destinations use model weight memory directly. Each attention group
            uses one temporary fused-QKV tensor and splits it into Q, K, and V before the iterator advances.

        Raises:
            RuntimeError: If the configured expert backend does not support checkpoint load groups.
            ValueError: If a distributed mesh, unsupported tensor layout, incomplete dependency group, or allocating
                ordinary destination is encountered.
        """
        if not self.supports_checkpoint_load_groups:
            raise RuntimeError(
                f"{type(self).__name__} does not support checkpoint load groups for experts={self.backend.experts}, "
                f"dispatcher={self.backend.dispatcher}"
            )
        if device_mesh is not None:
            raise ValueError("Ling checkpoint load groups currently require a single-device model")

        ordinary_destinations: dict[str, torch.Tensor] = {}
        ordinary_native_keys: set[str] = set()
        qkv_groups: dict[str, _LingQKVGroupBuilder] = {}
        expert_groups: dict[str, _LingExpertGroupBuilder] = {}
        n_experts = self.moe_config.n_routed_experts

        for native_name, state_tensor in checkpoint_state.items():
            qkv_match = _NATIVE_LAYER_QKV_RE.match(native_name)
            if qkv_match is not None:
                prefix = qkv_match.group("prefix")
                projection = qkv_match.group("projection")
                builder = qkv_groups.setdefault(prefix, _LingQKVGroupBuilder(projections={}))
                if projection in builder.projections:
                    raise ValueError(f"Duplicate Ling {projection.upper()} projection for {prefix}")
                builder.projections[projection] = (native_name, state_tensor)
                continue

            expert_match = _STREAMABLE_EXPERT_WEIGHT.match(native_name)
            if expert_match is not None:
                expert_root = expert_match.group("root")
                projection = expert_match.group("projection")
                builder = expert_groups.setdefault(
                    expert_root,
                    _LingExpertGroupBuilder(destinations={}, native_keys=set(), projections=set()),
                )
                self._add_expert_checkpoint_views(
                    builder,
                    expert_root=expert_root,
                    native_name=native_name,
                    projection=projection,
                    grouped_weight=state_tensor,
                )
                continue

            checkpoint_name, checkpoint_destination = self._single_checkpoint_load_view(native_name, state_tensor)
            if checkpoint_name in ordinary_destinations:
                raise ValueError(f"Duplicate ordinary Ling checkpoint destination {checkpoint_name}")
            ordinary_destinations[checkpoint_name] = checkpoint_destination
            ordinary_native_keys.add(native_name)

        if not qkv_groups:
            raise ValueError("Ling checkpoint load plan found no fused-QKV layers")
        if not expert_groups:
            raise ValueError("Ling checkpoint load plan found no grouped expert layers")

        if ordinary_destinations:
            yield CheckpointLoadGroup(
                destinations=ordinary_destinations,
                native_keys=frozenset(ordinary_native_keys),
            )

        for prefix, builder in qkv_groups.items():
            yield self._build_qkv_load_group(prefix, builder)

        for expert_root, builder in expert_groups.items():
            yield self._build_expert_load_group(expert_root, builder, n_experts)

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
        exclude_key_regex: str | None = None,
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

            m = _NATIVE_LAYER_QKV_RE.match(fqn)
            if m:
                pending_qkv.setdefault(m.group("prefix"), {})[m.group("projection")] = tensor
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

        m = _NATIVE_LAYER_QKV_RE.match(fqn)
        if m:
            return [(f"{m.group('prefix')}.attention.{m.group('projection')}_proj.weight", tensor)]

        return [(_rename_native_to_hf(fqn), tensor)]

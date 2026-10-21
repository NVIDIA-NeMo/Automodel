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

"""Strict Flint checkpoint conversion using the shared split-expert layout."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import get_expert_range_for_rank_from_mesh

if TYPE_CHECKING:
    from nemo_automodel.components.models.flint.config import FlintConfig

_EXPERT_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_DTYPES = (torch.bfloat16, torch.float32)
_LORA_EXPERT_KEY = re.compile(
    r"^(?:base_model\.model\.)?model\.layers\.(\d+)\.mlp\.experts\."
    r"(lora_gate_and_up_A|lora_gate_and_up_B|lora_down_A|lora_down_B)$"
)
_LORA_LINEAR_KEY = re.compile(r"^(.+)\.(lora_A|lora_B)\.weight$")


@dataclass(frozen=True)
class _LoadAlias:
    """Hold one destination's storage alive and identify its exact tensor view."""

    storage: torch.UntypedStorage
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> _LoadAlias | None:
        """Identify local storage without copying any parameter payload.

        Args:
            tensor: Ordinary HF-shaped destination or DTensor with that global
                shape; the local tensor retains its existing strided storage.

        Returns:
            Storage/view identity, or None for a shape-only meta destination.
        """
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        if local.is_meta:
            return None
        return cls(
            local.untyped_storage(),
            local.storage_offset(),
            tuple(local.shape),
            tuple(local.stride()),
            local.dtype,
            local.device,
        )

    def matches(self, tensor: torch.Tensor | None) -> bool:
        """Check the exact destination storage, offset, layout, dtype and device.

        Args:
            tensor: Current HF expert tensor, possibly a different Python view
                of the original destination. None denotes an absent key.

        Returns:
            Whether this view still aliases the recorded destination exactly.
        """
        if tensor is None:
            return False
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        return (
            not local.is_meta
            and local.device == self.device
            and local.dtype == self.dtype
            and tuple(local.shape) == self.shape
            and tuple(local.stride()) == self.stride
            and local.storage_offset() == self.offset
            and local.untyped_storage().data_ptr() == self.storage.data_ptr()
        )


class FlintStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert split BF16 base or FP32 resume weights without losing source names.

    HF experts are ``[intermediate, hidden]`` gate/up and ``[hidden,
    intermediate]`` down tensors. Native grouped weights are ``[experts,
    hidden, 2 * intermediate]`` (gate before up) and ``[experts,
    intermediate, hidden]``. Shared expert paths and the FP32 sink holder are
    renamed; all other ordinary layouts are unchanged.

    ``from_hf`` consumes its input mapping, as the shared merging implementation
    does. Runtime storage remains FP32 when requested. ``to_hf`` never casts
    training weights to BF16: consolidated-export dtype belongs to the shared
    writer's original-checkpoint metadata. Export may allocate contiguous
    per-expert tensors; its memory use is distinct from direct loading.
    """

    _supports_low_memory_dcp_load = True

    def __init__(
        self,
        config: FlintConfig,
        moe_config: MoEConfig | None,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Bind the model-owned shape contract and resident parameter dtype.

        Args:
            config: Flint architecture with explicit query-head dimension.
            moe_config: Shared expert configuration matching the Flint widths,
                or None for a configuration containing only dense layers.
            backend: Runtime backend; only the tested plain Torch load path
                opts into the single-process low-memory capability.
            dtype: Grouped expert resident dtype, BF16 or FP32. Sinks and the
                output head always use FP32; this does not set export dtype.
        """
        if dtype not in _DTYPES:
            raise ValueError(f"Flint checkpoint conversion requires BF16 or FP32 resident dtype, got {dtype}")
        has_moe = len(config.mlp_only_layers) < config.num_hidden_layers
        if has_moe and moe_config is None:
            raise ValueError("Flint MoE layers require a shared MoE configuration")
        if moe_config is not None and (
            moe_config.n_routed_experts != config.num_experts
            or moe_config.dim != config.hidden_size
            or moe_config.moe_inter_dim != config.moe_intermediate_size
            or moe_config.n_shared_experts != 1
            or moe_config.shared_expert_inter_dim != config.shared_expert_intermediate_size
        ):
            raise ValueError("Flint MoE configuration must match the routed/shared checkpoint dimensions")
        if moe_config is not None and (
            moe_config.router_bias or moe_config.expert_bias or moe_config.shared_expert_gate
        ):
            raise ValueError("Flint checkpoints have no router/expert bias or shared-expert gate")
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        self._hf_shapes = self._checkpoint_shapes()
        self._source_hf_shapes = self._checkpoint_shapes(config.schedule_num_hidden_layers)
        self._native_shapes = self._native_tensor_shapes()
        self._pipeline_native_keys: frozenset[str] | None = None
        self._pending_load_shapes: dict[str, tuple[int, ...]] | None = None
        self._pending_load_aliases: dict[str, _LoadAlias] = {}
        self._pending_expert_aliases: dict[str, dict[str, _LoadAlias]] = {}

    @property
    def supports_low_memory_dcp_load(self) -> bool:
        """Opt in only for the tested single-process model-backed Torch weights."""
        multiple_ranks = torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1
        return (
            self.moe_config is not None
            and not multiple_ranks
            and self.backend.dispatcher == "torch"
            and self.backend.experts == "torch"
            and super().supports_low_memory_dcp_load
        )

    def _checkpoint_shapes(self, num_layers: int | None = None) -> dict[str, tuple[int, ...]]:
        """Enumerate source HF shapes without allocating any checkpoint tensors."""
        config = self.config
        hidden, intermediate = config.hidden_size, config.moe_intermediate_size
        shapes = {
            "model.embed_tokens.weight": (config.vocab_size, hidden),
            "model.norm.weight": (hidden,),
            "lm_head.weight": (config.vocab_size, hidden),
        }
        query, kv = config.num_attention_heads * config.head_dim, config.num_key_value_heads * config.head_dim
        for layer in range(config.num_hidden_layers if num_layers is None else num_layers):
            prefix = f"model.layers.{layer}."
            shapes.update(
                {
                    prefix + "input_layernorm.weight": (hidden,),
                    prefix + "post_attention_layernorm.weight": (hidden,),
                    prefix + "self_attn.q_proj.weight": (query, hidden),
                    prefix + "self_attn.k_proj.weight": (kv, hidden),
                    prefix + "self_attn.v_proj.weight": (kv, hidden),
                    prefix + "self_attn.o_proj.weight": (hidden, query),
                    prefix + "self_attn.q_norm.weight": (config.head_dim,),
                    prefix + "self_attn.k_norm.weight": (config.head_dim,),
                    prefix + "self_attn.softmax_offset": (config.num_attention_heads,),
                }
            )
            mlps = [("mlp", config.intermediate_size)]
            if layer not in config.mlp_only_layers:
                shapes[prefix + "mlp.gate.weight"] = (config.num_experts, hidden)
                mlps = [("mlp.shared_expert", config.shared_expert_intermediate_size)]
                mlps.extend((f"mlp.experts.{expert}", intermediate) for expert in range(config.num_experts))
            for module, width in mlps:
                shapes[prefix + module + ".gate_proj.weight"] = (width, hidden)
                shapes[prefix + module + ".up_proj.weight"] = (width, hidden)
                shapes[prefix + module + ".down_proj.weight"] = (hidden, width)
        return shapes

    @staticmethod
    def _native_name(hf_name: str) -> str:
        """Translate ordinary HF paths into the native shared-expert/sink paths."""
        return hf_name.replace(".mlp.shared_expert.", ".mlp.shared_experts.").replace(
            ".self_attn.softmax_offset", ".self_attn._fp32_params.softmax_offset"
        )

    @staticmethod
    def _hf_name(native_name: str) -> str:
        """Reverse only the two Flint-owned ordinary module path changes."""
        return native_name.replace(".mlp.shared_experts.", ".mlp.shared_expert.").replace(
            ".self_attn._fp32_params.softmax_offset", ".self_attn.softmax_offset"
        )

    def _is_backend_extra_state(self, name: str) -> bool:
        """Identify only instantiated TE bookkeeping, never source HF weights."""
        if not name.endswith("._extra_state"):
            return False
        module = name.removesuffix("._extra_state")
        if self.backend.attn == "te" and module + ".softmax_offset" in self._native_shapes:
            return True
        if self.backend.linear == "te" and module.endswith("_proj") and module + ".weight" in self._native_shapes:
            return True
        if self.backend.rms_norm == "te" and module.endswith(("layernorm", "_norm", ".norm")):
            return module + ".weight" in self._native_shapes
        return False

    def _native_lora_shapes(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
        """Validate the shapes of existing native linear and grouped LoRA tensors.

        Args:
            tensors: Native BF16/FP32 tensors, optionally carrying the shared
                PEFT ``base_model.model.`` prefix. Linear A/B weights are
                [rank,input]/[output,rank]; grouped expert A/B weights are
                [experts,input,rank]/[experts,rank,output]. DTensor shapes
                are global and retain their expert-axis placements.

        Returns:
            Exact allowed LoRA shapes derived from the corresponding known
            base weight dimensions and each positive adapter rank.
        """
        shapes = {}
        ranks: dict[str, int] = {}
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            expert = _LORA_EXPERT_KEY.fullmatch(name)
            if expert:
                layer, suffix = int(expert[1]), expert[2]
                base = f"model.layers.{layer}.mlp.experts."
                if base + "gate_and_up_projs" not in self._native_shapes or tensor.ndim != 3:
                    raise ValueError(f"Invalid Flint expert LoRA tensor: {name}")
                hidden, width, experts = (
                    self.config.hidden_size,
                    self.config.moe_intermediate_size,
                    self.config.num_experts,
                )
                rank = tensor.shape[2] if suffix.endswith("_A") else tensor.shape[1]
                if rank <= 0:
                    raise ValueError(f"Flint LoRA rank must be positive: {name}")
                group = name.rsplit(".", 1)[0]
                shapes[name] = {
                    "lora_gate_and_up_A": (experts, hidden, rank),
                    "lora_gate_and_up_B": (experts, rank, 2 * width),
                    "lora_down_A": (experts, width, rank),
                    "lora_down_B": (experts, rank, hidden),
                }[suffix]
            elif linear := _LORA_LINEAR_KEY.fullmatch(name):
                base = linear[1].removeprefix("base_model.model.") + ".weight"
                if base not in self._native_shapes or len(self._native_shapes[base]) != 2 or tensor.ndim != 2:
                    raise ValueError(f"Invalid Flint linear LoRA tensor: {name}")
                output_width, input_width = self._native_shapes[base]
                rank = tensor.shape[0] if linear[2] == "lora_A" else tensor.shape[1]
                if rank <= 0:
                    raise ValueError(f"Flint LoRA rank must be positive: {name}")
                group = linear[1]
                shapes[name] = (rank, input_width) if linear[2] == "lora_A" else (output_width, rank)
            else:
                continue
            if group in ranks and ranks[group] != rank:
                raise ValueError(f"Flint LoRA tensors in {group} must share one rank")
            ranks[group] = rank
        return shapes

    def _native_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Describe global native layouts, including the two grouped projections."""
        shapes = {
            self._native_name(key): shape for key, shape in self._hf_shapes.items() if not _EXPERT_KEY.fullmatch(key)
        }
        for layer in range(self.config.num_hidden_layers):
            if layer in self.config.mlp_only_layers:
                continue
            prefix = f"model.layers.{layer}.mlp.experts."
            shapes[prefix + "gate_and_up_projs"] = (
                self.config.num_experts,
                self.config.hidden_size,
                2 * self.config.moe_intermediate_size,
            )
            shapes[prefix + "down_projs"] = (
                self.config.num_experts,
                self.config.moe_intermediate_size,
                self.config.hidden_size,
            )
        return shapes

    def _convert_lora_from_hf(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Validate and join complete expert PEFT families using the shared mixin.

        Args:
            tensors: Adapter-only HF tensors. Linear A/B weights have shape
                [rank,input]/[output,rank]; per-expert gate/up/down use those
                layouts independently. Meta tensors validate metadata without
                payload reads; materialized duplicated gate/up A values must
                be identical because they share one native adapter.

        Returns:
            Native linear and grouped expert LoRA weights in layouts documented
            by _native_lora_shapes, preserving the existing PEFT outer prefix.
        """
        renamed = {self._native_name(name): tensor for name, tensor in tensors.items()}
        converted = self._recombine_lora_expert_keys(renamed) if self.moe_config is not None else renamed
        self._validate_tensors(converted, self._native_lora_shapes(converted))
        for name in converted:
            if _LORA_EXPERT_KEY.fullmatch(name):
                parent = name.rsplit(".", 1)[0]
                required = {
                    parent + "." + suffix
                    for suffix in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")
                }
            else:
                match = _LORA_LINEAR_KEY.fullmatch(name)
                required = {match[1] + "." + suffix + ".weight" for suffix in ("lora_A", "lora_B")}
            if required - converted.keys():
                raise ValueError(f"Flint LoRA checkpoint has an incomplete adapter family: {name}")
        if set(self.get_hf_state_dict_keys(converted)) != tensors.keys():
            raise ValueError("Flint LoRA checkpoint must contain complete expert adapter families")
        for name, tensor in tensors.items():
            if ".experts." in name and name.endswith(".gate_proj.lora_A.weight"):
                up = tensors.get(name.replace(".gate_proj.", ".up_proj."))
                if up is None or (not tensor.is_meta and not torch.equal(tensor, up)):
                    raise ValueError(f"Flint LoRA gate/up A tensors must agree: {name}")
        return converted

    @staticmethod
    def _validate_tensors(tensors: dict[str, torch.Tensor], shapes: dict[str, tuple[int, ...]]) -> None:
        """Reject unknown names, incompatible global shapes and unsupported dtypes.

        Args:
            tensors: Mapping of parameter names to BF16/FP32 tensors. Ordinary
                HF tensors use ``[out, in]`` matrices or one-dimensional norm/
                sink scales. Native grouped tensors use ``[experts, hidden,
                2 * intermediate]`` and ``[experts, intermediate, hidden]``.
                DTensor shapes are global, retaining their original placements.
            shapes: Required global shape for every allowed mapping key.
        """
        for key, tensor in tensors.items():
            if key not in shapes:
                raise ValueError(f"Unexpected Flint checkpoint tensor: {key}")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Flint checkpoint entry {key} must be a tensor")
            if tuple(tensor.shape) != shapes[key]:
                raise ValueError(f"Flint tensor {key} has shape {tuple(tensor.shape)}; expected {shapes[key]}")
            if tensor.dtype not in _DTYPES:
                raise TypeError(f"Flint tensor {key} must use BF16 or FP32, got {tensor.dtype}")

    def to_hf(
        self,
        state_dict: dict[str, torch.Tensor],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        *,
        for_checkpoint_load: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Expose HF tensors, preserving resident precision and load aliases.

        Args:
            state_dict: Native weights with the global layouts documented on
                this adapter. Tensor dtype/device/placements are preserved;
                load destinations alias native storage on the plain Torch path.
            exclude_key_regex: HF/native names to omit, typically backend state.
            quantization: Must be false: the supplied checkpoint is unquantized.
            for_checkpoint_load: Build a complete set of destination views for
                DCP. Save/export callers leave this false; grouped exports may
                allocate contiguous tensors without mutating source values.
            **kwargs: Shared checkpoint protocol arguments forwarded to the
                expert mixin, including device mesh and serialization options.

        Returns:
            HF-keyed tensors in split-expert ``[out, in]`` layout. Plain Torch
            load destinations are potentially noncontiguous aliases; save-mode
            expert outputs are contiguous. No export dtype cast is performed.
        """
        if quantization:
            raise ValueError("Flint checkpoint conversion does not support quantized weights")
        selected = {
            key: value
            for key, value in state_dict.items()
            if not self._is_backend_extra_state(key) and (not exclude_key_regex or not re.match(exclude_key_regex, key))
        }
        self._validate_tensors(selected, self._native_shapes | self._native_lora_shapes(selected))
        if for_checkpoint_load:
            adapter_only = bool(selected) and all(".lora_" in name for name in selected)
            pipeline_partition = self._pipeline_native_keys is not None
            required_shapes = self._native_shapes
            if pipeline_partition and not adapter_only:
                # A real PP mesh owns complete decoder layers, with the
                # embedding on its first stage and norm/head on its last.
                layers = {key.split(".")[2] for key in self._pipeline_native_keys if key.startswith("model.layers.")}
                required_shapes = {
                    key: shape
                    for key, shape in self._native_shapes.items()
                    if (key.startswith("model.layers.") and key.split(".")[2] in layers)
                    or key in self._pipeline_native_keys
                    or (
                        key in ("model.norm.weight", "lm_head.weight")
                        and (
                            "model.norm.weight" in self._pipeline_native_keys
                            or "lm_head.weight" in self._pipeline_native_keys
                        )
                    )
                }
            missing = set() if adapter_only else required_shapes.keys() - selected.keys()
            if missing:
                raise ValueError(f"Missing native Flint load destinations: {sorted(missing)[:5]}")
            self._inplace_loaded_native_keys = set()
            self._pending_load_shapes = None
            self._pending_load_aliases = {}
            self._pending_expert_aliases = {}
        converted = {}
        for key, value in selected.items():
            for hf_key, tensor in self.convert_single_tensor_to_hf(
                key, value, for_checkpoint_load=for_checkpoint_load, **kwargs
            ):
                if not exclude_key_regex or not re.match(exclude_key_regex, hf_key):
                    converted[hf_key] = tensor
        if for_checkpoint_load:
            self._pending_load_shapes = {key: tuple(value.shape) for key, value in converted.items()}
            self._pending_load_aliases = {
                key: alias for key, tensor in converted.items() if (alias := _LoadAlias.capture(tensor)) is not None
            }
        return converted

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: torch.Tensor, **kwargs: Any
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert one global native weight using the shared expert machinery.

        Args:
            fqn: Exact native model parameter name.
            tensor: BF16/FP32 native tensor. Ordinary shapes are unchanged;
                grouped shapes are ``[experts, hidden, 2 * intermediate]`` or
                ``[experts, intermediate, hidden]``. DTensors retain global
                shapes and the shared mixin's adjusted rank-local placements.
            **kwargs: Shared conversion flags, including for_checkpoint_load.

        Returns:
            HF name/tensor pairs; plain Torch load-mode tensors alias the
            input storage. Save-mode expert matrices may be copies.
        """
        if self._is_backend_extra_state(fqn):
            return []
        self._validate_tensors({fqn: tensor}, self._native_shapes | self._native_lora_shapes({fqn: tensor}))
        if kwargs.get("quantization", False):
            raise ValueError("Flint checkpoint conversion does not support quantized weights")
        result = (
            self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
            if self.moe_config is not None
            else None
        )
        if result is None:
            return [(self._hf_name(fqn), tensor)]
        if (
            kwargs.get("for_checkpoint_load", False)
            and self.supports_low_memory_dcp_load
            and not isinstance(tensor, DTensor)
            and not tensor.is_meta
            and all(value.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr() for _, value in result)
        ):
            # The shared mixin marks CUDA/DTensor aliases. The same tested
            # write-through contract also holds for ordinary CPU load views.
            self._register_inplace_loaded_key(fqn, prefix_override=None)
        if kwargs.get("for_checkpoint_load", False) and fqn in getattr(self, "_inplace_loaded_native_keys", set()):
            aliases = {key: _LoadAlias.capture(value) for key, value in result}
            if aliases and all(alias is not None for alias in aliases.values()):
                self._pending_expert_aliases[fqn] = {key: alias for key, alias in aliases.items() if alias is not None}
        return result

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Validate and consume HF weights, joining only the owned expert rows.

        Args:
            hf_state_dict: Complete BF16 base or FP32 resume mapping in HF
                ``[out, in]`` layouts, consumed by this call. After to_hf in
                load mode, keys must exactly match that call's DCP destinations.
                DTensors retain global shapes and shard placements.
            device_mesh: Optional shared expert mesh selecting global expert
                IDs owned by this rank; no distributed capability is advertised.
            **kwargs: Compatibility arguments of the shared adapter interface.

        Returns:
            Native mapping with FP32 sinks/head, unchanged ordinary layouts and
            resident-dtype grouped expert weights. View-loaded grouped keys are
            omitted and recorded in view_loaded_native_keys; their existing
            native storage has already been populated. The input is consumed.
        """
        try:
            if hf_state_dict and all(".lora_" in name for name in hf_state_dict):
                # The shared PEFT loader supplies adapter-only state, without
                # the frozen base checkpoint. Keep that lifecycle distinct
                # from complete base-weight validation below.
                converted = self._convert_lora_from_hf(hf_state_dict)
                hf_state_dict.clear()
                self._view_loaded_native_keys = set()
                return converted
            if hf_state_dict.keys() == self._source_hf_shapes.keys():
                self._validate_tensors(hf_state_dict, self._source_hf_shapes)
                for key in self._source_hf_shapes.keys() - self._hf_shapes.keys():
                    del hf_state_dict[key]
            else:
                self._validate_tensors(hf_state_dict, self._hf_shapes)
            planned_load = self._pending_load_shapes is not None and any(
                alias.matches(hf_state_dict.get(key)) for key, alias in self._pending_load_aliases.items()
            )
            required = self._pending_load_shapes if planned_load else None
            if required is None:
                start, end = get_expert_range_for_rank_from_mesh(device_mesh, self.config.num_experts)
                required = {
                    key: shape
                    for key, shape in self._hf_shapes.items()
                    if not (match := _EXPERT_KEY.fullmatch(key)) or start <= int(match.group(2)) < end
                }
            missing = required.keys() - hf_state_dict.keys()
            if missing:
                raise ValueError(f"Missing Flint checkpoint tensors: {sorted(missing)[:5]}")
            if planned_load and hf_state_dict.keys() - required.keys():
                raise ValueError("Flint checkpoint load returned tensors outside the planned rank-local destinations")
            # DCP can fail before from_hf. A later, unrelated source mapping
            # must not inherit skip marks from that abandoned destination plan.
            self._inplace_loaded_native_keys = {
                native_key
                for native_key, aliases in self._pending_expert_aliases.items()
                if aliases and all(alias.matches(hf_state_dict.get(key)) for key, alias in aliases.items())
            }
            for key in list(hf_state_dict):
                native_name = self._native_name(key)
                if native_name != key or key == "lm_head.weight":
                    value = hf_state_dict.pop(key)
                    if native_name.endswith("._fp32_params.softmax_offset") or key == "lm_head.weight":
                        value = value.to(torch.float32)
                    hf_state_dict[native_name] = value
            if self.moe_config is None:
                self._view_loaded_native_keys = set()
                converted = dict(hf_state_dict)
                hf_state_dict.clear()
                return converted
            return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)
        except (ValueError, TypeError, RuntimeError):
            self._view_loaded_native_keys = set()
            raise
        finally:
            self._pending_load_shapes = None
            self._pending_load_aliases = {}
            self._pending_expert_aliases = {}
            self._inplace_loaded_native_keys = set()

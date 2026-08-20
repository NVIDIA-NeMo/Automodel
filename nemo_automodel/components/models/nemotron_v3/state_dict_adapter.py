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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadGroup, StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

logger = logging.getLogger(__name__)

_MAMBA_FP32_PARAMS_TO_BARE = re.compile(r"(\.mixer)\._fp32_params\.")
_MAMBA_FP32_PARAM_NAMES = ("A_log", "dt_bias", "D")
_STREAMABLE_EXPERT_WEIGHT = re.compile(
    r"^(?P<root>(?:.+\.)?layers\.\d+\.mixer\.experts)\."
    r"(?P<projection>gate_and_up_projs|down_projs)$"
)


@dataclass
class _NemotronExpertGroupBuilder:
    """Mutable builder for one dependency-complete expert layer group."""

    destinations: dict[str, torch.Tensor]
    expert_ids_by_projection: dict[str, set[int]]
    native_keys: set[str]


def _strip_mamba_fp32_holder_key(key: str) -> str:
    return _MAMBA_FP32_PARAMS_TO_BARE.sub(r"\1.", key)


def _route_mamba_fp32_holder_key(key: str) -> str:
    if "._fp32_params." in key or ".mixer." not in key:
        return key
    head, tail = key.rsplit(".mixer.", 1)
    if tail not in _MAMBA_FP32_PARAM_NAMES:
        return key
    return f"{head}.mixer._fp32_params.{tail}"


def _is_mamba_fp32_state_key(key: str) -> bool:
    if ".mixer." not in key:
        return False
    _, tail = key.rsplit(".mixer.", 1)
    if tail in _MAMBA_FP32_PARAM_NAMES:
        return True
    if tail.startswith("_fp32_params."):
        return tail[len("_fp32_params.") :] in _MAMBA_FP32_PARAM_NAMES
    return False


def _upcast_mamba_fp32_state_tensor(key: str, value: Any) -> Any:
    if _is_mamba_fp32_state_key(key) and isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
        return value.to(torch.float32)
    return value


class NemotronV3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """State dict adapter for NemotronV3 models.

    Converts between HuggingFace checkpoint format and internal NeMo format.

    HF format uses 'backbone' prefix:
        - backbone.embed_tokens.weight
        - backbone.layers.{}.norm.weight
        - backbone.layers.{}.mixer.* (mamba/attention/moe components)
        - backbone.norm_f.weight
        - lm_head.weight

    Internal format uses 'model' prefix:
        - model.embed_tokens.weight
        - model.layers.{}.norm.weight
        - model.layers.{}.mixer.* (mamba/attention/moe components)
        - model.norm.weight
        - lm_head.weight

    For MoE layers:
        - HF: Split per-expert weights (experts.{}.up_proj.weight, experts.{}.down_proj.weight)
        - Internal: Merged expert weights (experts.gate_and_up_projs, experts.down_projs)

    NemotronV3 uses ReLU² activation (non-gated), so gate_and_up_projs has
    shape [n_experts, dim, inter_dim] instead of [n_experts, dim, 2*inter_dim].

    Note: NemotronV3 uses 'mixer' instead of 'mlp' in layer paths.
    """

    _supports_write_through_checkpoint_load = True

    def __init__(
        self,
        config,
        moe_config: MoEConfig | None,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        # moe_config is None for dense Nemotron-H variants (no MoE layers); the
        # expert merge/split paths below are only reached for ``.mixer.experts.`` keys,
        # which dense checkpoints never contain.
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

        # Mapping for expert weights (HF split → internal merged)
        self.from_hf_map = {
            "model.layers.{}.mixer.experts.{}.up_proj.weight": "model.layers.{}.mixer.experts.gate_and_up_projs",
            "model.layers.{}.mixer.experts.{}.down_proj.weight": "model.layers.{}.mixer.experts.down_projs",
        }

    @property
    def supports_streaming_checkpoint_load(self) -> bool:
        """Whether this adapter can stream split experts into final single-device storage.

        TE/DeepEP is configured for distributed runs, but the MoE layer deliberately falls back to ``GroupedExperts``
        when world size is one. The adapter configuration still says ``experts="te"``, so the legacy converter assumes
        the grouped tensors are TE stack copies and rebuilds them. For non-gated ReLU-squared experts, the runtime
        grouped tensors instead provide per-expert transposed views that write through to final storage. Expert bias
        and other activations retain the allocating fallback until their layouts have concrete coverage.
        """
        return (
            self.moe_config is not None
            and self.backend.experts == "te"
            and self.moe_config.expert_activation == "relu2"
            and not self.moe_config.expert_bias
        )

    def iter_checkpoint_load_groups(
        self,
        model_part: torch.nn.Module,
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> Iterator[CheckpointLoadGroup]:
        """Yield final-storage destinations for the single-device TE fallback.

        Args:
            model_part: Nemotron V3 model whose ordinary parameters and grouped expert parameters are final load
                destinations. Native input projections have shape [experts, hidden, expert_hidden], and native down
                projections have shape [experts, expert_hidden, hidden].
            device_mesh: Must be ``None``. Distributed expert parameters require a rank-symmetric group plan.

        Returns:
            Iterator whose first group contains ordinary parameter aliases and whose remaining groups contain one
            complete MoE layer each. Expert destinations use HF shapes [expert_hidden, hidden] for ``up_proj`` and
            [hidden, expert_hidden] for ``down_proj`` and are transposed views of final grouped parameter storage.

        Raises:
            RuntimeError: If this adapter configuration did not opt into streaming checkpoint loading.
            ValueError: If a distributed mesh, an allocating ordinary conversion, an expert bias, or an incomplete
                expert group is encountered.
        """
        if not self.supports_streaming_checkpoint_load:
            raise RuntimeError(f"{type(self).__name__} does not support streaming for backend {self.backend.experts}")
        if device_mesh is not None:
            raise ValueError("Nemotron V3 streaming checkpoint loading currently requires a single-device model")
        moe_config = self.moe_config
        if moe_config is None:
            raise RuntimeError("Nemotron V3 streaming checkpoint loading requires an MoE configuration")

        ordinary_destinations: dict[str, torch.Tensor] = {}
        ordinary_native_keys: set[str] = set()
        expert_groups: dict[str, _NemotronExpertGroupBuilder] = {}
        expected_expert_ids = set(range(moe_config.n_routed_experts))

        for parameter_name, parameter in model_part.named_parameters():
            native_name = canonical_parameter_fqn(parameter_name)
            # PEFT is applied before the base checkpoint is loaded. Its adapter parameters are intentionally absent
            # from the pretrained checkpoint and retain the initialization performed by initialize_model_weights().
            if "lora" in native_name:
                continue

            expert_match = _STREAMABLE_EXPERT_WEIGHT.match(native_name)
            if expert_match is not None:
                expert_root = expert_match.group("root")
                native_projection_name = expert_match.group("projection")
                if native_projection_name == "gate_and_up_projs":
                    projection_name = "up_proj"
                    expected_shape = (
                        moe_config.n_routed_experts,
                        moe_config.expert_dim,
                        moe_config.moe_inter_dim,
                    )
                else:
                    projection_name = "down_proj"
                    expected_shape = (
                        moe_config.n_routed_experts,
                        moe_config.moe_inter_dim,
                        moe_config.expert_dim,
                    )

                if tuple(parameter.shape) != expected_shape:
                    raise ValueError(
                        f"Grouped expert parameter {native_name} has shape {tuple(parameter.shape)}, "
                        f"expected {expected_shape}"
                    )

                builder = expert_groups.setdefault(
                    expert_root,
                    _NemotronExpertGroupBuilder(destinations={}, expert_ids_by_projection={}, native_keys=set()),
                )
                split_weights = self._split_experts_weights(parameter.detach(), moe_config.n_routed_experts)
                expert_ids = list(self._last_expert_ids)
                for expert_id, expert_weight in zip(expert_ids, split_weights, strict=True):
                    checkpoint_name = self._native_key_to_hf(f"{expert_root}.{expert_id}.{projection_name}.weight")
                    if checkpoint_name in builder.destinations:
                        raise ValueError(f"Duplicate expert checkpoint destination: {checkpoint_name}")
                    checkpoint_destination = expert_weight.transpose(0, 1)
                    expected_checkpoint_shape = (
                        (moe_config.moe_inter_dim, moe_config.expert_dim)
                        if projection_name == "up_proj"
                        else (moe_config.expert_dim, moe_config.moe_inter_dim)
                    )
                    if tuple(checkpoint_destination.shape) != expected_checkpoint_shape:
                        raise ValueError(
                            f"Expert checkpoint destination {checkpoint_name} has shape "
                            f"{tuple(checkpoint_destination.shape)}, expected {expected_checkpoint_shape}"
                        )
                    builder.destinations[checkpoint_name] = checkpoint_destination
                builder.expert_ids_by_projection[native_projection_name] = set(expert_ids)
                builder.native_keys.add(native_name)
                del split_weights
                continue

            destination = parameter.detach()
            converted = self.convert_single_tensor_to_hf(native_name, destination, quantization=False)
            if len(converted) != 1:
                raise ValueError(
                    f"Ordinary Nemotron parameter {native_name} produced {len(converted)} checkpoint destinations"
                )
            checkpoint_name, checkpoint_destination = converted[0]
            if not isinstance(checkpoint_destination, torch.Tensor):
                raise ValueError(
                    f"Ordinary Nemotron destination {checkpoint_name} is {type(checkpoint_destination).__name__}, "
                    "expected Tensor"
                )
            aliases_parameter = (
                checkpoint_destination.device == destination.device
                and checkpoint_destination.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr()
            )
            if not aliases_parameter:
                raise ValueError(
                    f"Ordinary Nemotron destination {checkpoint_name} does not alias final parameter {native_name}"
                )
            if checkpoint_name in ordinary_destinations:
                raise ValueError(f"Duplicate ordinary Nemotron checkpoint destination: {checkpoint_name}")
            ordinary_destinations[checkpoint_name] = checkpoint_destination
            ordinary_native_keys.add(native_name)

        for group_key, builder in expert_groups.items():
            for projection_name in ("gate_and_up_projs", "down_projs"):
                expert_ids = builder.expert_ids_by_projection.get(projection_name, set())
                if expert_ids != expected_expert_ids:
                    missing_experts = sorted(expected_expert_ids - expert_ids)
                    raise ValueError(
                        f"Incomplete expert group {group_key}.{projection_name}: "
                        f"missing expert ids {missing_experts[:10]}"
                    )

        if not expert_groups:
            raise ValueError("Nemotron V3 streaming load plan found no grouped expert layers")

        if ordinary_destinations:
            yield CheckpointLoadGroup(
                destinations=ordinary_destinations,
                native_keys=frozenset(ordinary_native_keys),
            )

        for group_key in sorted(expert_groups):
            builder = expert_groups[group_key]
            yield CheckpointLoadGroup(
                destinations=builder.destinations,
                native_keys=frozenset(builder.native_keys),
            )

    @property
    def _hf_prefix(self) -> str:
        """NemotronV3 HF format uses 'backbone.' prefix."""
        return "backbone."

    @property
    def _expert_path_segment(self) -> str:
        """NemotronV3 uses 'mixer.experts' instead of 'mlp.experts'."""
        return "mixer.experts"

    @property
    def _v5_peft_target_parameters(self) -> tuple[str, ...]:
        """Nemotron V3 exposes fused non-gated expert parameters in Transformers v5."""
        return ("mixer.experts.up_proj", "mixer.experts.down_proj")

    @staticmethod
    def _native_key_to_hf(key: str) -> str:
        """Normalize a native Nemotron V3 key to its public HF namespace."""
        key = _strip_mamba_fp32_holder_key(key)
        key = re.sub(r"^model\.", "backbone.", key)
        key = re.sub(r"^backbone\.norm\.weight$", "backbone.norm_f.weight", key)
        key = re.sub(r"^backbone\.embed_tokens\.weight$", "backbone.embeddings.weight", key)
        return key

    @staticmethod
    def _hf_key_to_native(key: str) -> str:
        """Normalize a public HF Nemotron V3 key to its native namespace."""
        key = re.sub(r"^((?:base_model\.model\.)?backbone)\.norm_f\.weight$", r"\1.norm.weight", key)
        key = re.sub(r"^((?:base_model\.model\.)?backbone)\.embeddings\.weight$", r"\1.embed_tokens.weight", key)
        return re.sub(
            r"^(?P<outer>base_model\.model\.)?backbone\.",
            lambda match: f"{match.group('outer') or ''}model.",
            key,
        )

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: str | None = None, **kwargs) -> dict[str, Any]:
        """Convert from internal model state dict to HuggingFace format.

        Args:
            state_dict: Internal format state dict
            exclude_key_regex: Optional regex pattern to exclude keys
            **kwargs: Additional arguments

        Returns:
            HuggingFace format state dict
        """
        hf_state_dict = {}
        for fqn in list(state_dict.keys()):
            tensor = state_dict.pop(fqn)
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to internal format.

        - Rename backbone → model
        - Rename norm_f → norm
        - Aggregate per-expert weights into grouped tensors
        - If device_mesh is provided, only load experts needed for the current rank
        - Process MTP keys (``mtp.layers.{i}.*``) separately, reusing the
          same MoE expert-merge logic for the MoE sublayer of each MTP depth.

        Args:
            hf_state_dict: HuggingFace format state dict
            device_mesh: Optional device mesh for distributed expert loading
            **kwargs: Additional arguments

        Returns:
            Internal format state dict
        """
        # Drop checkpoint keys for backbone layers past ``num_hidden_layers``
        # (e.g. when loading the first N layers of a larger checkpoint for a
        # downsized smoke run). The matcher tolerates both ``backbone.layers.{i}``
        # and ``model.layers.{i}`` since the prefix is normalized after this.
        num_layers = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        if num_layers > 0:
            layer_idx_pattern = re.compile(r"^(?:backbone|model)\.layers\.(\d+)\.")
            for key in list(hf_state_dict.keys()):
                m = layer_idx_pattern.match(key)
                if m is not None and int(m.group(1)) >= num_layers:
                    hf_state_dict.pop(key)

        # Separate MTP keys; they live in their own top-level namespace and
        # are not subject to the backbone/model rename.
        mtp_state_dict: dict[str, Any] = {}
        backbone_state_dict: dict[str, Any] = {}
        for key in list(hf_state_dict.keys()):
            value = hf_state_dict.pop(key)
            if key.startswith("mtp."):
                mtp_state_dict[key] = value
            else:
                backbone_state_dict[key] = value

        # Detect if HF checkpoint uses 'backbone' or 'model' prefix. Only
        # look at backbone keys; MTP keys never carry a backbone/model prefix.
        for key in backbone_state_dict.keys():
            if ".mixer.experts." in key:
                self._uses_model_prefix = not key.startswith("backbone.")
                break

        # First, rename backbone → model and norm_f → norm
        renamed_state_dict = {}
        for key in list(backbone_state_dict.keys()):
            value = backbone_state_dict.pop(key)
            new_key = self._hf_key_to_native(key)

            new_key = _route_mamba_fp32_holder_key(new_key)
            renamed_state_dict[new_key] = _upcast_mamba_fp32_state_tensor(new_key, value)

        # Then merge experts using the mixin method. Dense Nemotron-H variants have no
        # experts (moe_config is None) and no '.mixer.experts.' keys, so the merge is a
        # pure pass-through — skip it; the mixin would otherwise dereference
        # moe_config.n_routed_experts.
        if self.moe_config is None:
            merged = renamed_state_dict
        else:
            merged = self._from_hf_w_merged_experts(renamed_state_dict, device_mesh)

        # Re-route MTP keys through the standard merge with prefix stripped.
        if mtp_state_dict:
            stripped: dict[str, Any] = {}
            for key, value in mtp_state_dict.items():
                stripped_key = key[len("mtp.") :] if key.startswith("mtp.") else key
                stripped_key = _route_mamba_fp32_holder_key(stripped_key)
                stripped[stripped_key] = _upcast_mamba_fp32_state_tensor(stripped_key, value)
            # reset_view_loaded_keys=False: this is the second merge of a single from_hf (after the
            # backbone merge above), so accumulate MTP view-loaded keys onto the backbone's record.
            prior_view_keys = set(self.view_loaded_native_keys)
            merged_mtp = self._from_hf_w_merged_experts(stripped, device_mesh, reset_view_loaded_keys=False)
            for key, value in merged_mtp.items():
                merged[f"mtp.{key}"] = value
            # The merge loop records view-loaded keys in mtp.-stripped form (it only ever sees
            # stripped keys); re-prefix them so the checkpoint loader's key-diff matches them
            # against the model's real mtp.* parameter names instead of flagging them as
            # missing/unexpected.
            new_view_keys = self._view_loaded_native_keys - prior_view_keys
            self._view_loaded_native_keys = prior_view_keys | {f"mtp.{key}" for key in new_view_keys}

        return merged

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from internal format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in internal format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        # MTP keys live in their own ``mtp.*`` namespace; route them through
        # the standard expert-split path with the prefix overridden so
        # emitted HF keys stay under ``mtp.`` instead of ``backbone.``.
        if fqn.startswith("mtp."):
            fqn = _strip_mamba_fp32_holder_key(fqn)
            expert_split = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, prefix_override="mtp.")
            result = expert_split if expert_split is not None else [(fqn, tensor)]
            result = [(key, _upcast_mamba_fp32_state_tensor(key, value)) for key, value in result]
            if exclude_key_regex:
                result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
            return result

        # Try to convert merged expert weights to split experts. Dense variants have no
        # experts (moe_config is None), so skip straight to the standard rename path.
        expert_result = (
            None
            if self.moe_config is None
            else self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        )
        if expert_result is not None:
            # The shared expert converter preserves the native input prefix for
            # LoRA keys. Route every result through Nemotron's adapter-specific
            # model -> backbone normalization just like ordinary tensors.
            result = [(self._native_key_to_hf(key), value) for key, value in expert_result]
        else:
            new_fqn = self._native_key_to_hf(fqn)
            result = [(new_fqn, _upcast_mamba_fp32_state_tensor(new_fqn, tensor))]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    def forced_hf_dtype_mapping(self, state_dict: dict[str, Any]) -> dict[str, str]:
        """Return HF export dtype overrides for tensors that are intrinsically fp32."""
        forced: dict[str, str] = {}
        for fqn, value in state_dict.items():
            if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
                continue
            if _is_mamba_fp32_state_key(fqn) or "e_score_correction_bias" in fqn:
                forced[fqn] = "F32"
        return forced

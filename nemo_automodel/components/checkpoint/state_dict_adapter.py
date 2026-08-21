# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class CheckpointLoadGroup:
    """One set of checkpoint tensors that must be loaded and installed together.

    Attributes:
        destinations: Checkpoint keys and the tensors DCP should fill. A tensor can use model weight memory directly
            or be a temporary tensor needed for conversion.
        native_keys: Model state-dict keys completed by this group. Every key must be ready in the model before
            :meth:`install` returns.
    """

    destinations: dict[str, torch.Tensor]
    native_keys: frozenset[str]

    def install(self) -> None:
        """Finish converting this group's loaded tensors into model weights.

        The default does nothing because the destinations already use model weight memory. A group that uses temporary
        tensors overrides this method, performs its conversion, and releases those tensors before the next group.
        """
        return None


class StateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    """

    _supports_write_through_checkpoint_load: bool = False
    _supports_checkpoint_load_without_full_copy: bool = False
    _supports_checkpoint_load_groups: bool = False

    @property
    def supports_write_through_checkpoint_load(self) -> bool:
        """Whether every checkpoint tensor is loaded directly into the model's existing weight memory.

        Enable this only when writing every tensor returned by ``to_hf`` for base-checkpoint loading updates the
        model itself. This lets the loader skip a complete CPU copy of the checkpoint.
        """
        return self._supports_write_through_checkpoint_load

    @property
    def supports_checkpoint_load_without_full_copy(self) -> bool:
        """Whether DCP can load this adapter without another full set of model weights.

        Large checkpoint tensors must be loaded into the model's existing weight memory. Small temporary tensors are
        allowed when they can be applied and discarded without making a model-sized copy. For example, Gemma4 loads
        a scale tensor and applies it to already-loaded expert weights.
        """
        return self._supports_checkpoint_load_without_full_copy

    @property
    def supports_checkpoint_load_groups(self) -> bool:
        """Whether the adapter can load a base checkpoint in one or more memory-safe groups.

        Adapters should opt in only when :meth:`iter_checkpoint_load_groups` covers every tensor returned by
        :meth:`get_checkpoint_load_state`. Each group must finish updating the model and release any temporary tensors
        before the next group.
        """
        return self._supports_checkpoint_load_groups

    def get_checkpoint_load_state(self, model_part: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Collect the model tensors that a Hugging Face base checkpoint must populate.

        Args:
            model_part: Model whose parameters and persistent buffers use their native model layouts. Tensor ranks and
                axis orders are model-specific. LoRA tensors and runtime-only ``_extra_state`` entries are excluded
                because they are not stored in the Hugging Face base checkpoint.

        Returns:
            Mapping from canonical native state-dict names to detached tensors with the same shapes, strides, dtypes,
            devices, and storage as the model parameters and persistent buffers.

        Raises:
            ValueError: If a checkpoint-owned state entry is not a tensor or canonical names collide.
        """
        checkpoint_state: dict[str, torch.Tensor] = {}
        for state_name, state_value in model_part.state_dict(keep_vars=True).items():
            native_name = canonical_parameter_fqn(state_name)
            if "lora" in native_name or native_name.rsplit(".", 1)[-1] == "_extra_state":
                continue
            if not isinstance(state_value, torch.Tensor):
                raise ValueError(
                    f"Checkpoint-owned state entry {native_name} is {type(state_value).__name__}, expected Tensor"
                )
            if native_name in checkpoint_state:
                raise ValueError(f"Multiple model state entries resolve to checkpoint key {native_name}")
            checkpoint_state[native_name] = state_value.detach()
        return checkpoint_state

    def _checkpoint_load_views(
        self,
        native_name: str,
        native_tensor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Convert one model tensor into checkpoint-layout views of the same storage.

        Args:
            native_name: Canonical native state-dict name for ``native_tensor``.
            native_tensor: Model tensor with arbitrary rank and model-specific axis order. Every returned checkpoint
                tensor must be a view of this tensor's storage; non-contiguous views are allowed.

        Returns:
            Mapping from Hugging Face checkpoint names to tensors with checkpoint-specific shapes and axis orders.
            Every tensor uses ``native_tensor`` storage, so DCP writes update the model directly.

        Raises:
            ValueError: If conversion returns no tensors, returns duplicate names or non-tensors, or allocates storage.
        """
        if native_tensor.is_meta:
            raise ValueError(f"Checkpoint load destination {native_name} is still on the meta device")

        converted = self.convert_single_tensor_to_hf(
            native_name,
            native_tensor,
            quantization=False,
            for_checkpoint_load=True,
            track_inplace_load=False,
        )
        if not converted:
            raise ValueError(f"Checkpoint conversion for {native_name} produced no destinations")

        destinations: dict[str, torch.Tensor] = {}
        source_storage = native_tensor.untyped_storage().data_ptr()
        for checkpoint_name, checkpoint_tensor in converted:
            if not isinstance(checkpoint_tensor, torch.Tensor):
                raise ValueError(
                    f"Checkpoint destination {checkpoint_name} is {type(checkpoint_tensor).__name__}, expected Tensor"
                )
            if checkpoint_tensor.device != native_tensor.device:
                raise ValueError(
                    f"Checkpoint destination {checkpoint_name} is on {checkpoint_tensor.device}, "
                    f"but model tensor {native_name} is on {native_tensor.device}"
                )
            if checkpoint_tensor.untyped_storage().data_ptr() != source_storage:
                raise ValueError(
                    f"Checkpoint destination {checkpoint_name} does not use model tensor {native_name} storage"
                )
            if checkpoint_name in destinations:
                raise ValueError(f"Checkpoint conversion for {native_name} produced duplicate key {checkpoint_name}")
            destinations[checkpoint_name] = checkpoint_tensor
        return destinations

    def _single_checkpoint_load_view(
        self,
        native_name: str,
        native_tensor: torch.Tensor,
    ) -> tuple[str, torch.Tensor]:
        """Return the only checkpoint-layout view produced for one model tensor.

        Args:
            native_name: Canonical native state-dict name for ``native_tensor``.
            native_tensor: Model tensor with arbitrary rank and model-specific axis order. The returned tensor has the
                checkpoint layout and uses the same storage.

        Returns:
            Hugging Face checkpoint name and its tensor view.

        Raises:
            ValueError: If conversion produces anything other than one checkpoint tensor.
        """
        destinations = self._checkpoint_load_views(native_name, native_tensor)
        if len(destinations) != 1:
            raise ValueError(
                f"Checkpoint conversion for {native_name} produced {len(destinations)} destinations, expected one"
            )
        return next(iter(destinations.items()))

    def iter_checkpoint_load_groups(
        self,
        checkpoint_state: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
    ) -> Iterator[CheckpointLoadGroup]:
        """Build one direct-write group for adapters whose conversions return model-storage views.

        Args:
            checkpoint_state: Canonical native names and model tensors with model-specific shapes and axis orders.
                Each converted checkpoint tensor must be a view of its corresponding model tensor.
            device_mesh: Optional device mesh describing the final distributed tensor placements.

        Returns:
            One group containing every checkpoint-layout view. Model adapters override this method only when a
            conversion needs temporary tensors that must be installed and released before later groups are loaded.

        Raises:
            RuntimeError: If the adapter has not opted into checkpoint load groups.
            ValueError: If a distributed mesh is provided or two model tensors map to the same checkpoint key.
        """
        if not self.supports_checkpoint_load_groups:
            raise RuntimeError(f"{type(self).__name__} does not support checkpoint load groups")
        if device_mesh is not None:
            raise ValueError("The default checkpoint load group requires a single-device model")

        destinations: dict[str, torch.Tensor] = {}
        for native_name, native_tensor in checkpoint_state.items():
            converted = self._checkpoint_load_views(native_name, native_tensor)
            duplicate_keys = sorted(converted.keys() & destinations.keys())
            if duplicate_keys:
                raise ValueError(
                    f"Multiple model tensors map to {len(duplicate_keys)} checkpoint keys "
                    f"(examples={duplicate_keys[:5]})"
                )
            destinations.update(converted)

        yield CheckpointLoadGroup(
            destinations=destinations,
            native_keys=frozenset(checkpoint_state),
        )

    @abstractmethod
    def to_hf(self, state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        Args:
            state_dict: The native model state dict

        Returns:
            The converted HuggingFace format state dict
        """
        pass

    @abstractmethod
    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Obtain native model state dict from HuggingFace format.

        Args:
            hf_state_dict: The HuggingFace format state dict
            device_mesh: Optional device mesh for DTensor expert parallelism.
                        If provided, only loads experts needed for the current rank.

        Returns:
            The converted native model state dict
        """
        pass

    @abstractmethod
    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format.
            Returns a list because some native tensors may split into multiple HF tensors.
        """
        pass

    def get_hf_state_dict_keys(self, state_dict: dict[str, Any]) -> list[str]:
        """Return the Hugging Face keys produced by ``to_hf`` without converting real weights.

        Args:
            state_dict: Native model state mapping. Tensor values may have
                arbitrary rank and axis order and retain their exact parameter
                or buffer layouts.

        Returns:
            Hugging Face state-dict keys in adapter iteration order.
        """
        shape_only_state = {
            key: torch.empty_like(value, device="meta") if isinstance(value, torch.Tensor) else value
            for key, value in state_dict.items()
        }
        return list(self.to_hf(shape_only_state, exclude_key_regex=r".*_extra_state.*", quantization=False))

    def map_peft_target_module_to_hf(self, name: str) -> str:
        """Translate a PEFT target-module name to the HuggingFace layout.

        adapter_config.json's target_modules are collected from native module
        names. Adapters whose ``to_hf`` renames modules (e.g. Kimi K3's
        ``mlp.experts.{E}.gate_proj`` -> ``block_sparse_moe.experts.{E}.w1``)
        should override this with the same renames so PEFT can resolve the
        entries against the converted checkpoint.

        Args:
            name: A target-module name in native layout.

        Returns:
            The name in HuggingFace layout. Defaults to the name unchanged.
        """
        return name

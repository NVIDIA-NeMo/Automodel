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

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class CheckpointLoadGroup:
    """One dependency-complete unit of a bounded checkpoint load.

    Attributes:
        destinations: Mapping from checkpoint FQNs to destination tensors. Each tensor has the exact shape and axis
            order recorded for that FQN in checkpoint metadata. A destination may alias final model storage or may be
            bounded temporary storage consumed by :meth:`install`.
        native_keys: Native model state-dict keys completed by this group. Their tensors may have arbitrary ranks and
            model-owned layouts; every key must be fully installed before :meth:`install` returns.
    """

    destinations: dict[str, torch.Tensor]
    native_keys: frozenset[str]

    def install(self) -> None:
        """Install loaded destinations into final model storage.

        The default implementation is a no-op for destinations that already alias final model storage. Allocating
        conversion groups override this method, complete their model-owned transformation, and release references to
        temporary storage before returning.
        """
        return None


class StateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    """

    _supports_write_through_checkpoint_load: bool = False
    _supports_checkpoint_load_without_full_copy: bool = False
    _supports_streaming_checkpoint_load: bool = False

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
    def supports_streaming_checkpoint_load(self) -> bool:
        """Whether the adapter can load a base checkpoint as bounded dependency groups.

        Adapters should opt in only when :meth:`iter_checkpoint_load_groups` covers every required pretrained model
        tensor and each yielded group writes through to final storage or installs bounded temporary storage before the
        next group is requested.
        """
        return self._supports_streaming_checkpoint_load

    def iter_checkpoint_load_groups(
        self,
        model_part: torch.nn.Module,
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> Iterator[CheckpointLoadGroup]:
        """Yield dependency-complete groups for bounded base-checkpoint loading.

        Args:
            model_part: Model part that owns the final parameter and buffer storage populated by yielded groups.
            device_mesh: Optional device mesh describing the final distributed tensor placements.

        Returns:
            Iterator of groups. Each group's destination tensors have checkpoint-native shapes and axis order; the
            iterator must not retain an installed group's temporary tensors after advancing.

        Raises:
            NotImplementedError: If the adapter does not implement streaming checkpoint loading.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement streaming checkpoint loading")

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
        """Return the Hugging Face keys produced by ``to_hf``.

        Args:
            state_dict: Native model state mapping. Tensor values may have
                arbitrary rank and axis order and retain their exact parameter
                or buffer layouts.

        Returns:
            Hugging Face state-dict keys in adapter iteration order.
        """
        return list(self.to_hf(state_dict, exclude_key_regex=r".*_extra_state.*", quantization=False))

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

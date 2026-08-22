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

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class CheckpointLoadPart:
    """One part of a checkpoint that is loaded and finished before the next part.

    DCP fills ``checkpoint_tensors`` using their Hugging Face names and layouts. ``finish`` then converts any
    temporary checkpoint tensors into the final model tensors named by ``model_keys`` before the loader advances to
    the next part. Tensors that already have the model's exact dtype and layout may point directly at model storage.

    Attributes:
        checkpoint_tensors: Hugging Face checkpoint names mapped to DCP destinations. Destinations may use final model
            storage or temporary storage owned by this part.
        model_keys: Native model state-dict names completed after ``finish`` returns.
        temporary_checkpoint_keys: Names in ``checkpoint_tensors`` whose destinations use temporary storage rather
            than final model storage. The loader uses these keys to report the largest per-rank temporary allocation.
        finish: Callback that installs temporary checkpoint tensors into final model storage. It must not save those
            temporary tensors anywhere outside this load part.
    """

    checkpoint_tensors: dict[str, torch.Tensor]
    model_keys: frozenset[str]
    temporary_checkpoint_keys: frozenset[str]
    finish: Callable[[], None]


class StateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    """

    _supports_low_memory_dcp_load: bool = False

    @property
    def supports_low_memory_dcp_load(self) -> bool:
        """Whether DCP can load the checkpoint with zero or small temporary tensors.

        Most checkpoint tensors must load directly into the model's existing weight memory. Small temporary tensors
        are allowed when they are converted and released after the read. Enable this only when the extra memory is
        safely below a full model copy, including when loading on one GPU.
        """
        return self._supports_low_memory_dcp_load

    def iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> Iterator[CheckpointLoadPart] | None:
        """Optionally load and finish a quantized checkpoint in small parts.

        Ordinary adapters do not need to implement this method. It is only for adapters that cannot let DCP write
        checkpoint tensors directly into model storage because a dtype or layout conversion is required, but can
        complete that conversion for one small part of the model at a time.

        Args:
            model_state_dict: Native model names mapped to the final parameter and persistent-buffer tensors that the
                checkpoint must populate. Tensors retain their model-specific shapes, axis orders, dtypes, devices,
                strides, distributed placements, and storage.
            device_mesh: Optional device mesh describing distributed model storage.

        Returns:
            An iterator of dependency-complete load parts, or ``None`` when the adapter does not support this path.
            Across all parts, ``model_keys`` must cover ``model_state_dict`` exactly once.
        """
        return None

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
            The general adapter interface returns a list because transforming adapters may split one native tensor
            into multiple Hugging Face tensors. Passthrough adapters return exactly one entry unless the key is
            excluded.
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


class PassthroughStateDictAdapter(StateDictAdapter):
    """Adapter for models whose native tensors already have the exact Hugging Face representation.

    Every tensor keeps the same key, value, shape, axis order, dtype, device, strides, and storage. Conversion only
    creates a new Python mapping and may exclude selected keys; it never transforms or copies tensor data.
    """

    _supports_low_memory_dcp_load = True

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Expose model tensors without changing their keys or tensor representation.

        Args:
            state_dict: Model state containing tensors of arbitrary shape and axis order. Tensor values retain their
                exact value, shape, axis order, dtype, device, strides, and storage.
            exclude_key_regex: Optional regular expression selecting keys to omit.
            **kwargs: Compatibility options that do not alter passthrough tensors.

        Returns:
            A new mapping whose tensor values are the original tensor objects and therefore alias the input storage.
        """
        if exclude_key_regex is None:
            return dict(state_dict)
        return {key: value for key, value in state_dict.items() if not re.search(exclude_key_regex, key)}

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Expose exactly one unchanged tensor unless its key is excluded.

        Args:
            fqn: Fully qualified model tensor name.
            tensor: Model tensor of arbitrary shape and axis order. The returned tensor is the same object, preserving
                its value, shape, axis order, dtype, device, strides, and storage.
            **kwargs: Adapter options, including an optional exclusion regex.

        Returns:
            One unchanged key/tensor pair, or an empty list when excluded. This passthrough implementation never
            splits one tensor into multiple outputs.
        """
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if isinstance(exclude_key_regex, str) and re.search(exclude_key_regex, fqn):
            return []
        return [(fqn, tensor)]

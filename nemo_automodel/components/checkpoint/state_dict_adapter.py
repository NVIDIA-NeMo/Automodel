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
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


def _get_hf_keys_lazy(adapter: Any, state_dict: dict[str, Any]) -> list[str]:
    """Compute HF key list from native state_dict without materializing tensors."""
    out: list[str] = []
    for fqn in state_dict:
        hf_keys = getattr(adapter, "get_hf_keys_for_native_key", None)
        if hf_keys is not None:
            keys = hf_keys(fqn)
            if keys is not None:
                out.extend(keys)
                continue
        out.append(fqn)
    return out


def _get_native_keys_lazy(adapter: Any, hf_state_dict: dict[str, Any]) -> list[str]:
    """Compute native key list from HF state_dict without consuming it."""
    seen: set[str] = set()
    out: list[str] = []
    get_native = getattr(adapter, "get_native_key_for_hf_key", None)
    for hf_key in hf_state_dict.keys():
        if not isinstance(hf_key, str):
            continue
        if get_native is not None:
            native_key = get_native(hf_key)
            if native_key is not None:
                if native_key not in seen:
                    seen.add(native_key)
                    out.append(native_key)
                continue
        if hf_key not in seen:
            seen.add(hf_key)
            out.append(hf_key)
    return out


class LazyHFStateDict:
    """Dict-like wrapper that converts native -> HF on key access (JIT). Reduces peak GPU memory."""

    def __init__(self, state_dict: dict[str, Any], adapter: Any) -> None:
        self._state_dict = state_dict
        self._adapter = adapter
        self._keys: Optional[list[str]] = None

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def keys(self) -> Iterator[str]:
        if self._keys is None:
            self._keys = _get_hf_keys_lazy(self._adapter, self._state_dict)
        return iter(self._keys)

    def __getitem__(self, hf_key: str) -> Any:
        get_native = getattr(self._adapter, "get_native_key_for_hf_key", None)
        if get_native is not None:
            native_fqn = get_native(hf_key)
            if native_fqn is not None:
                tensor = self._state_dict.get(native_fqn)
                if tensor is not None:
                    return self._adapter.get_tensor_for_hf_key(native_fqn, tensor, hf_key)
        return self._state_dict[hf_key]

    def __contains__(self, key: object) -> bool:
        if self._keys is None:
            self._keys = _get_hf_keys_lazy(self._adapter, self._state_dict)
        return key in self._keys

    def __len__(self) -> int:
        if self._keys is None:
            self._keys = _get_hf_keys_lazy(self._adapter, self._state_dict)
        return len(self._keys)

    def items(self) -> Iterator[tuple[str, Any]]:
        for k in self.keys():
            yield k, self[k]


class LazyNativeStateDict:
    """Dict-like wrapper that converts HF -> native on key access (JIT). Merges incrementally to limit peak memory.
    When native_backing is set (round-trip from LazyHFStateDict), returns tensors from it directly (zero copy).
    """

    def __init__(
        self,
        hf_state_dict: dict[str, Any],
        adapter: Any,
        device_mesh: Optional["DeviceMesh"] = None,
        native_backing: Optional[dict[str, Any]] = None,
    ) -> None:
        self._hf_state_dict = hf_state_dict
        self._adapter = adapter
        self._device_mesh = device_mesh
        self._native_backing = native_backing
        self._keys = None
        if getattr(adapter, "_validate_expert_availability", None) is not None and getattr(adapter, "moe_config", None) is not None:
            adapter._validate_expert_availability(
                hf_state_dict, adapter.moe_config.n_routed_experts, device_mesh
            )

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def keys(self) -> Iterator[str]:
        if self._keys is None:
            self._keys = _get_native_keys_lazy(self._adapter, self._hf_state_dict)
        return iter(self._keys)

    def __getitem__(self, native_key: str) -> Any:
        if self._native_backing is not None and native_key in self._native_backing:
            return self._native_backing[native_key]
        get_merged = getattr(self._adapter, "get_merged_tensor_for_native_key", None)
        if get_merged is not None:
            merged = get_merged(native_key, self._hf_state_dict, self._device_mesh)
            if merged is not None:
                return merged
        return self._hf_state_dict[native_key]

    def __contains__(self, key: object) -> bool:
        if self._keys is None:
            self._keys = _get_native_keys_lazy(self._adapter, self._hf_state_dict)
        return key in self._keys

    def __len__(self) -> int:
        if self._keys is None:
            self._keys = _get_native_keys_lazy(self._adapter, self._hf_state_dict)
        return len(self._keys)

    def items(self) -> Iterator[tuple[str, Any]]:
        for k in self.keys():
            yield k, self[k]


class StateDictAdapter(ABC):
    """Abstract base class for state dict transformations.

    This class defines the interface for converting between native model
    state dict format and other model state dict formats.
    """

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

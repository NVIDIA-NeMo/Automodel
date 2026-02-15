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

"""Lazy state dict wrappers for JIT conversion (native <-> HF) to limit peak GPU memory.

Lives in the models component so adapters (combined_projection, qwen3_moe, etc.)
can use it without pulling in the checkpoint component.
"""

import re as _re
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


def get_hf_keys_lazy(adapter: Any, state_dict: dict[str, Any]) -> list[str]:
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


def get_native_keys_lazy(adapter: Any, hf_state_dict: dict[str, Any]) -> list[str]:
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


class LazyHFStateDict(Mapping):
    """Dict-like wrapper that converts native -> HF on key access (JIT). Reduces peak GPU memory.

    Args:
        state_dict: Native model state dict.
        adapter: State dict adapter with key-mapping helpers.
        exclude_key_regex: Optional regex pattern. Keys matching this pattern are
            excluded from iteration/containment (e.g. ``r".*_extra_state.*"``).
    """

    def __init__(self, state_dict: dict[str, Any], adapter: Any, exclude_key_regex: Optional[str] = None) -> None:
        self._state_dict = state_dict
        self._adapter = adapter
        self._exclude_key_regex = exclude_key_regex
        self._keys: Optional[list[str]] = None

    def _compute_keys(self) -> list[str]:
        keys = get_hf_keys_lazy(self._adapter, self._state_dict)
        if self._exclude_key_regex:
            keys = [k for k in keys if not _re.match(self._exclude_key_regex, k)]
        return keys

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def keys(self) -> Iterator[str]:
        if self._keys is None:
            self._keys = self._compute_keys()
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
            self._keys = self._compute_keys()
        return key in self._keys

    def __len__(self) -> int:
        if self._keys is None:
            self._keys = self._compute_keys()
        return len(self._keys)

    def values(self) -> Iterator[Any]:
        for k in self.keys():
            yield self[k]

    def items(self) -> Iterator[tuple[str, Any]]:
        for k in self.keys():
            yield k, self[k]


class LazyNativeStateDict(MutableMapping):
    """Dict-like wrapper that converts HF -> native on key access (JIT). Merges incrementally to limit peak memory.
    When native_backing is set (round-trip from LazyHFStateDict), returns tensors from it directly (zero copy).

    Supports in-place mutation (``__setitem__``, ``__delitem__``, ``pop``,
    ``update``) via an overlay so that downstream code (e.g. key-renaming
    helpers, ``_extra_state`` injection) can treat this object like a plain
    ``dict`` without materialising the full lazy mapping.
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
        self._base_keys: Optional[list[str]] = None
        # Overlay for mutation – avoids materialising the full lazy mapping.
        self._overrides: dict[str, Any] = {}
        self._deleted: set[str] = set()
        if (
            getattr(adapter, "_validate_expert_availability", None) is not None
            and getattr(adapter, "moe_config", None) is not None
        ):
            adapter._validate_expert_availability(hf_state_dict, adapter.moe_config.n_routed_experts, device_mesh)

    # Internal helpers
    def _get_base_keys(self) -> list[str]:
        if self._base_keys is None:
            self._base_keys = get_native_keys_lazy(self._adapter, self._hf_state_dict)
        return self._base_keys

    # Read API
    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def keys(self) -> Iterator[str]:
        seen: set[str] = set()
        for k in self._get_base_keys():
            if k not in self._deleted or k in self._overrides:
                seen.add(k)
                yield k
        for k in self._overrides:
            if k not in seen:
                yield k

    def __getitem__(self, native_key: str) -> Any:
        if native_key in self._overrides:
            return self._overrides[native_key]
        if native_key in self._deleted:
            raise KeyError(native_key)
        if self._native_backing is not None and native_key in self._native_backing:
            return self._native_backing[native_key]
        get_merged = getattr(self._adapter, "get_merged_tensor_for_native_key", None)
        if get_merged is not None:
            merged = get_merged(native_key, self._hf_state_dict, self._device_mesh)
            if merged is not None:
                return merged
        return self._hf_state_dict[native_key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if key in self._overrides:
            return True
        if key in self._deleted:
            return False
        return key in self._get_base_keys()

    def __len__(self) -> int:
        base = set(self._get_base_keys())
        return len((base - self._deleted) | set(self._overrides))

    # Mutation API
    def __setitem__(self, key: str, value: Any) -> None:
        self._overrides[key] = value
        self._deleted.discard(key)

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError(key)
        self._overrides.pop(key, None)
        self._deleted.add(key)

    def pop(self, key: str, *default: Any) -> Any:
        try:
            value = self[key]
        except KeyError:
            if default:
                return default[0]
            raise
        self._overrides.pop(key, None)
        self._deleted.add(key)
        return value

    def update(self, other: Any = None, **kwargs: Any) -> None:
        if other is not None:
            if hasattr(other, "items"):
                for k, v in other.items():
                    self[k] = v
            elif hasattr(other, "keys"):
                for k in other.keys():
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    # Iteration helpers
    def values(self) -> Iterator[Any]:
        for k in self.keys():
            yield self[k]

    def items(self) -> Iterator[tuple[str, Any]]:
        for k in self.keys():
            yield k, self[k]

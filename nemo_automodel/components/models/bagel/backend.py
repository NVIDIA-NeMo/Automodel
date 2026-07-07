# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Linear and RMSNorm backend configuration for BAGEL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, kw_only=True)
class BagelBackendConfig:
    """Backends that BAGEL allows users to select."""

    linear: Literal["torch", "te"] = "torch"
    rms_norm: Literal["torch", "torch_fp32", "te"] = "torch_fp32"

    def __post_init__(self) -> None:
        if self.linear not in {"torch", "te"}:
            raise ValueError(f"Unsupported BAGEL linear backend: {self.linear!r}")
        if self.rms_norm not in {"torch", "torch_fp32", "te"}:
            raise ValueError(f"Unsupported BAGEL RMSNorm backend: {self.rms_norm!r}")


def resolve_bagel_backend(backend: Any = None) -> BagelBackendConfig:
    """Resolve a mapping against BAGEL's stable torch defaults."""
    if backend is None:
        return BagelBackendConfig()
    if isinstance(backend, BagelBackendConfig):
        return backend

    if hasattr(backend, "to_dict"):
        backend = backend.to_dict()
    if not isinstance(backend, Mapping):
        raise TypeError(f"BAGEL backend must be a mapping or BagelBackendConfig, got {type(backend)!r}")

    overrides = dict(backend)
    unknown_fields = sorted(overrides.keys() - {"linear", "rms_norm"})
    if unknown_fields:
        raise TypeError(f"Unknown BAGEL backend field(s): {unknown_fields}")
    return BagelBackendConfig(**overrides)

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Backend configuration helpers for BAGEL.

BAGEL has custom packed FlexAttention and unfused RoPE implementations. Only
the linear and RMSNorm backends are currently selectable; keeping the full
``BackendConfig`` explicit prevents its CUDA/TE-dependent defaults from
changing BAGEL behavior across containers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

from nemo_automodel.components.models.common import BackendConfig


def default_bagel_backend() -> BackendConfig:
    """Return BAGEL's deterministic torch/FlexAttention baseline."""
    return BackendConfig(
        attn="flex",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        dispatcher_num_sms=20,
        dispatcher_share_token_dispatcher=True,
        dispatcher_async_dispatch=False,
        enable_deepep=None,
        fake_balanced_gate=False,
        fake_gate_noise=0.0,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=False,
        te_fp8=None,
        gate_precision=None,
        compile_attn=False,
    )


def _validate_bagel_backend(backend: BackendConfig) -> None:
    unsupported: list[str] = []
    if backend.attn != "flex":
        unsupported.append(
            f"attn={backend.attn!r}; BAGEL training uses its packed FlexAttention path, so attn must be 'flex'"
        )
    if backend.rope_fusion:
        unsupported.append("rope_fusion=True; BAGEL currently uses its packed unfused RoPE implementation")
    if backend.te_fp8 is not None:
        unsupported.append("te_fp8 is set; BAGEL does not yet preserve Transformer Engine FP8 state across checkpoints")
    if backend.compile_attn:
        unsupported.append("compile_attn=True; BAGEL compiles its FlexAttention kernel directly")
    if unsupported:
        raise ValueError("Unsupported BAGEL backend configuration: " + "; ".join(unsupported))


def resolve_bagel_backend(backend: Any = None) -> BackendConfig:
    """Resolve a mapping or ``BackendConfig`` against BAGEL's stable defaults."""
    if backend is None:
        resolved = default_bagel_backend()
    elif isinstance(backend, BackendConfig):
        resolved = backend
    else:
        if hasattr(backend, "to_dict"):
            backend = backend.to_dict()
        if not isinstance(backend, Mapping):
            raise TypeError(f"BAGEL backend must be a mapping or BackendConfig, got {type(backend)!r}")
        overrides = dict(backend)
        valid_fields = {field.name for field in fields(BackendConfig)}
        unknown_fields = sorted(overrides.keys() - valid_fields)
        if unknown_fields:
            raise TypeError(f"Unknown BAGEL backend field(s): {unknown_fields}")
        resolved = replace(default_bagel_backend(), **overrides)

    _validate_bagel_backend(resolved)
    return resolved


def bagel_backend_summary(backend: BackendConfig) -> str:
    """Return the user-facing subset of backend choices BAGEL consumes."""
    return (
        f"attn={backend.attn}, linear={backend.linear}, rms_norm={backend.rms_norm}, "
        f"rope_fusion={backend.rope_fusion}"
    )

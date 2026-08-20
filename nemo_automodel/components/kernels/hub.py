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

"""Thin wrapper around the Hugging Face ``kernels`` library.

AutoModel call sites that today import ``flash_attn`` directly can use these
helpers to load the same symbols from ``kernels-community`` Hub repos instead.
The implementation mirrors the upstream loader API documented at
https://github.com/huggingface/kernels and used by ``transformers.integrations.hub_kernels``.

For reproducible/offline deployments, pin versions with ``kernels lock`` and load
via ``load_kernel`` / ``get_locked_kernel`` from the upstream package.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from contextlib import nullcontext
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from nemo_automodel.shared.import_utils import safe_import, safe_import_from

if TYPE_CHECKING:
    from nemo_automodel.components.kernels.config import HubKernelConfig

logger = logging.getLogger(__name__)

# Well-known kernels-community repos (see https://huggingface.co/kernels-community/kernels).
HUB_FLASH_ATTN2 = "kernels-community/flash-attn2"
HUB_FLASH_ATTN3 = "kernels-community/flash-attn3"
HUB_FLASH_ATTN4 = "kernels-community/flash-attn4"
HUB_LIGER_KERNELS = "kernels-community/liger-kernels"
HUB_MAMBA_SSM = "kernels-community/mamba-ssm"
HUB_CAUSAL_CONV1D = "kernels-community/causal-conv1d"
HUB_MEGABLOCKS = "kernels-community/megablocks"
HUB_FLA = "kernels-community/fla"
HUB_ACTIVATION = "kernels-community/activation"
HUB_ROTARY = "kernels-community/rotary"

# P0/P1 repos to lock in container images (see docs/investigations/kernels-hub-integration.md).
RECOMMENDED_LOCK_REPOS: tuple[str, ...] = (
    HUB_FLASH_ATTN2,
    HUB_LIGER_KERNELS,
    HUB_ACTIVATION,
    HUB_ROTARY,
    HUB_MAMBA_SSM,
)

_HAS_KERNELS_LIB, _kernels_mod = safe_import("kernels")
HAS_KERNELS_LIB = _HAS_KERNELS_LIB
HAS_COMPILED_FA, _ = safe_import("flash_attn")

# Map transformers attn_implementation strings to Hub repos (mirrors hub_kernels fallback).
_FLASH_ATTN_HUB_BY_IMPL: dict[str, str] = {
    "flash_attention_2": HUB_FLASH_ATTN2,
    "flash_attention_3": HUB_FLASH_ATTN3,
    "flash_attention_4": HUB_FLASH_ATTN4,
}


def is_hub_attn_implementation(attn_implementation: str) -> bool:
    """Return True when ``attn_implementation`` is a Hub kernel repo id."""
    if not attn_implementation:
        return False
    repo_id = attn_implementation.split(":", 1)[0].split("@", 1)[0]
    return "/" in repo_id and not repo_id.startswith("http")


def has_hub_kernel(repo_id: str, *, version: int = 1) -> bool:
    """Check whether a Hub kernel build exists for the current environment.

    Args:
        repo_id: Hub repository id (e.g. ``kernels-community/flash-attn2``).
        version: Kernel major version branch (``version=1`` → latest on ``v1``).

    Returns:
        True when the ``kernels`` package is installed and a compatible build
        exists for the active PyTorch/CUDA configuration.
    """
    if not HAS_KERNELS_LIB:
        return False
    try:
        return bool(_kernels_mod.has_kernel(repo_id, version=version))
    except Exception:
        logger.debug("has_kernel(%r) raised; treating as unavailable", repo_id, exc_info=True)
        return False


@functools.lru_cache(maxsize=16)
def get_hub_kernel(repo_id: str, *, version: int = 1) -> ModuleType | None:
    """Load and cache a kernel module from the Hub.

    Uses the upstream ``kernels.get_kernel`` loader. Results are cached per
    process so repeated lookups (e.g. varlen in every CP layer) do not
    re-download or re-import.

    Args:
        repo_id: Hub repository id.
        version: Kernel major version.

    Returns:
        The imported kernel module, or ``None`` when unavailable.
    """
    if not has_hub_kernel(repo_id, version=version):
        return None
    try:
        return _kernels_mod.get_kernel(repo_id, version=version)
    except Exception:
        logger.warning("Failed to load Hub kernel %r (version=%s)", repo_id, version, exc_info=True)
        return None


def _hub_flash_attn_module(attn_implementation: str | None = None) -> ModuleType | None:
    """Resolve a flash-attention Hub module for an implementation string."""
    if attn_implementation and is_hub_attn_implementation(attn_implementation):
        repo_id = attn_implementation.split(":", 1)[0].split("@", 1)[0]
        # Hub attention repos other than FA2/3/4 may omit version= in the repo card;
        # default to v1 for kernels-community builds.
        return get_hub_kernel(repo_id, version=1)

    for impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
        repo_id = _FLASH_ATTN_HUB_BY_IMPL[impl]
        mod = get_hub_kernel(repo_id, version=1)
        if mod is not None:
            return mod
    return None


def has_flash_attn_available(*, attn_implementation: str | None = None) -> bool:
    """Return True when flash attention is available via pip or the Hub.

    Checks compiled ``flash-attn`` first (preferred when installed for TE/ABI
    compatibility), then falls back to kernels-community Hub builds.
    """
    if HAS_COMPILED_FA:
        return True
    return _hub_flash_attn_module(attn_implementation) is not None


def resolve_attn_implementation(
    attn_implementation: str,
    *,
    hub_kernels: HubKernelConfig | None = None,
) -> str:
    """Resolve ``attn_implementation`` aliases and ``BackendConfig.hub_kernels`` overrides."""
    if hub_kernels is not None and hub_kernels.attn_repo:
        return hub_kernels.attn_repo
    if attn_implementation == "hub":
        return HUB_FLASH_ATTN2
    return attn_implementation


def extract_hub_kernels_config(kwargs: dict[str, Any]) -> HubKernelConfig | None:
    """Read ``hub_kernels`` from a ``backend`` entry in model init kwargs."""
    backend = kwargs.get("backend")
    if backend is None:
        return None
    hub_kernels = getattr(backend, "hub_kernels", None)
    if hub_kernels is None and isinstance(backend, dict):
        hub_kernels = backend.get("hub_kernels")
    if hub_kernels is None:
        return None
    if isinstance(hub_kernels, dict):
        from nemo_automodel.components.kernels.config import HubKernelConfig

        return HubKernelConfig(**hub_kernels)
    return hub_kernels


def get_flash_attn_func(*, attn_implementation: str | None = None) -> Callable[..., Any] | None:
    """Return ``flash_attn_func`` from pip or the Hub."""
    _, compiled_func = safe_import_from("flash_attn", "flash_attn_func")
    if compiled_func is not None:
        return compiled_func

    hub_mod = _hub_flash_attn_module(attn_implementation)
    if hub_mod is None:
        return None
    flash_func = getattr(hub_mod, "flash_attn_func", None)
    if flash_func is None:
        logger.debug("Hub flash-attn module %r has no flash_attn_func", hub_mod)
    return flash_func


def apply_hub_kernels_to_model(
    model: nn.Module,
    *,
    kernel_config: Any | None = None,
    allow_all_kernels: bool = False,
) -> nn.Module:
    """Apply Transformers ``kernelize`` to an already-built custom model."""
    try:
        from transformers.integrations.hub_kernels import (
            allow_all_hub_kernels,
            is_kernels_available,
            kernelize,
        )
    except ImportError:
        logger.warning("use_kernels=True but transformers hub_kernels integration is unavailable")
        return model

    if not is_kernels_available():
        logger.warning("use_kernels=True but the kernels package is not installed")
        return model

    if not isinstance(model, nn.Module):
        logger.warning("Skipping Hub kernelize for non-nn.Module model: %s", type(model))
        return model

    if kernel_config is not None and hasattr(model, "set_use_kernels"):
        model.set_use_kernels(True, kernel_config)
    elif kernel_config is not None and hasattr(model, "config"):
        model.config.kernel_config = kernel_config

    ctx = allow_all_hub_kernels() if allow_all_kernels else nullcontext()
    with ctx:
        kernelize(model)
    logger.info("Applied Hub kernels via transformers.kernelize")
    return model


def get_flash_attn_varlen_func(*, attn_implementation: str | None = None) -> Callable[..., Any] | None:
    """Return ``flash_attn_varlen_func`` from pip or the Hub.

    Call sites such as blockdiag CP and packed VLM inference can use this
    instead of ``safe_import_from("flash_attn", "flash_attn_varlen_func")``.

    Args:
        attn_implementation: Optional transformers/NeMo attention string. When
            it is a Hub repo id, that repo is tried first.

    Returns:
        The varlen function, or ``None`` when neither pip nor Hub provides it.
    """
    _, compiled_varlen = safe_import_from("flash_attn", "flash_attn_varlen_func")
    if compiled_varlen is not None:
        return compiled_varlen

    hub_mod = _hub_flash_attn_module(attn_implementation)
    if hub_mod is None:
        return None
    varlen = getattr(hub_mod, "flash_attn_varlen_func", None)
    if varlen is None:
        logger.debug("Hub flash-attn module %r has no flash_attn_varlen_func", hub_mod)
    return varlen

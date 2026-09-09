# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer

_WARNED_CONTEXTS: set[str] = set()
_DTYPE_RESOLVED_CONTEXTS: set[str] = set()
_TORCH_ADAM_TARGETS = {
    "torch.optim.Adam",
    "torch.optim.AdamW",
    "torch.optim.adam.Adam",
    "torch.optim.adamw.AdamW",
}


def resolve_storage_dtype(
    cfg_model: Any | None,
    cfg_opt: Any | None,
    *,
    is_peft: bool = False,
    context: str = "recipe",
    logger: logging.Logger | None = None,
) -> None:
    """Default model storage dtype to float32 for full-parameter ``torch.optim`` training.

    Built-in ``torch.optim`` optimizers update the resident parameter in place and keep
    no internal fp32 master copy, so the model parameters *are* the master copy. Storing
    them in bf16 therefore makes optimizer updates and state bf16, which degrades training
    precision relative to frameworks that keep an fp32 master. To avoid that, when the user
    has not explicitly chosen a storage dtype we default ``cfg_model.torch_dtype`` to
    ``float32`` so the parameters act as the fp32 master copy. fp32 storage is never
    numerically worse than bf16 for these optimizers; the only cost is memory, which an
    explicit ``model.torch_dtype=bfloat16`` opts out of.

    No-ops (leaving the dtype unchanged) when:
      * ``is_peft`` is True (base weights are frozen; only small adapters train), or
      * the optimizer is not a ``torch.optim`` optimizer (e.g. TE ``FusedAdam``, DeepSpeed,
        or bitsandbytes optimizers, which manage their own master / state precision and so
        live outside the ``torch.optim`` namespace), or
      * ``model.torch_dtype`` is already set to a concrete (non-``auto``) value.

    The decision is mutated on every rank (so all ranks agree) but logged only on rank
    zero. It is idempotent: once set, a second call sees the explicit value and returns.

    Args:
        cfg_model: The model config node/dict (must expose/accept ``torch_dtype``).
        cfg_opt: The optimizer config node/dict (read for ``_target_``).
        is_peft: Whether this is a PEFT/LoRA run.
        context: Short label used for log de-duplication.
        logger: Optional logger; defaults to this module's logger.
    """
    if cfg_model is None or is_peft:
        return
    if not _is_torch_optim_config(cfg_opt):
        return

    current = _get_cfg_attr(cfg_model, "torch_dtype")
    if current is not None and not (isinstance(current, str) and current == "auto"):
        # User explicitly chose a storage dtype; always honor it.
        return

    _set_cfg_attr(cfg_model, "torch_dtype", "float32")

    if _is_rank_zero() and context not in _DTYPE_RESOLVED_CONTEXTS:
        log = logger if logger is not None else logging.getLogger(__name__)
        log.info(
            "No explicit model.torch_dtype set for full-parameter training with a "
            "torch.optim optimizer (which keeps no fp32 master copy). Defaulting "
            "model.torch_dtype=float32 so model parameters act as the fp32 master copy. "
            "Set model.torch_dtype=bfloat16 to keep bf16 storage, or use Transformer "
            "Engine FusedAdam with master_weights=True. See "
            "docs/guides/mixed-precision-training.md."
        )
        _DTYPE_RESOLVED_CONTEXTS.add(context)


def warn_if_torch_adam_with_bf16_params(
    *,
    optimizer: Optimizer | Iterable[Optimizer] | None = None,
    optimizer_cfg: Any | None = None,
    parameters: Iterable[torch.nn.Parameter] | None = None,
    is_peft: bool = False,
    context: str = "recipe",
    logger: logging.Logger | None = None,
) -> None:
    """Warn about full-parameter bf16 training with vanilla torch Adam optimizers."""
    if is_peft or not _is_rank_zero() or context in _WARNED_CONTEXTS:
        return

    if not (_is_torch_adam_optimizer(optimizer) or _is_torch_adam_config(optimizer_cfg)):
        return

    params = parameters if parameters is not None else _iter_optimizer_params(optimizer)
    if not _has_trainable_bf16_param(params):
        return

    log = logger if logger is not None else logging.getLogger(__name__)
    log.warning(
        "Detected torch.optim.Adam/AdamW with trainable bf16 model parameters. Updates and Adam state "
        "will use bf16, which saves memory but may affect training stability, convergence, or final loss. "
        "For maximum stability, set model.torch_dtype=float32 with FSDP mixed precision, or use Transformer "
        "Engine FusedAdam with master_weights=True. See docs/guides/mixed-precision-training.md."
    )

    _WARNED_CONTEXTS.add(context)


def _get_cfg_attr(cfg: Any, key: str) -> Any:
    """Read ``key`` from a config node or dict, returning None when absent."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(key, None)
    return getattr(cfg, key, None)


def _set_cfg_attr(cfg: Any, key: str, value: Any) -> None:
    """Set ``key`` on a config node or dict."""
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _is_rank_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _is_torch_adam_optimizer(optimizer: Optimizer | Iterable[Optimizer] | None) -> bool:
    if optimizer is None:
        return False
    optimizers = optimizer if isinstance(optimizer, Iterable) else [optimizer]
    return any(isinstance(opt, (torch.optim.Adam, torch.optim.AdamW)) for opt in optimizers)


def _is_torch_adam_config(optimizer_cfg: Any | None) -> bool:
    target = getattr(optimizer_cfg, "_target_", None)
    if target is None and isinstance(optimizer_cfg, dict):
        target = optimizer_cfg.get("_target_", None)
    if target is None:
        return False
    if isinstance(target, str):
        return target in _TORCH_ADAM_TARGETS
    if target in (torch.optim.Adam, torch.optim.AdamW):
        return True
    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", "")
    return f"{module}.{qualname}" in _TORCH_ADAM_TARGETS


def _is_torch_optim_config(optimizer_cfg: Any | None) -> bool:
    """True when the optimizer ``_target_`` is a built-in ``torch.optim`` optimizer.

    These optimizers update the resident parameter in place and keep no internal fp32
    master copy, so the storage dtype *is* the master-weight dtype and fp32 storage is
    always safe. Optimizers outside the ``torch.optim`` namespace (TE ``FusedAdam``,
    DeepSpeed, bitsandbytes 8-bit, Muon, ...) manage their own master / state precision
    and are deliberately excluded.
    """
    target = getattr(optimizer_cfg, "_target_", None)
    if target is None and isinstance(optimizer_cfg, dict):
        target = optimizer_cfg.get("_target_", None)
    if target is None:
        return False
    if isinstance(target, str):
        return target.startswith("torch.optim.")
    module = getattr(target, "__module__", "") or ""
    return module == "torch.optim" or module.startswith("torch.optim.")


def _iter_optimizer_params(optimizer: Optimizer | Iterable[Optimizer] | None) -> Iterable[torch.nn.Parameter]:
    if optimizer is None:
        return ()
    optimizers = optimizer if isinstance(optimizer, Iterable) else [optimizer]
    params = []
    for opt in optimizers:
        for group in getattr(opt, "param_groups", []):
            group_params = group.get("params", ())
            if isinstance(group_params, torch.nn.Parameter):
                params.append(group_params)
            else:
                params.extend(group_params)
    return params


def _has_trainable_bf16_param(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return any(
        getattr(param, "requires_grad", False) and getattr(param, "dtype", None) is torch.bfloat16
        for param in parameters
    )

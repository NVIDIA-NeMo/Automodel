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
    requested_dtype: str | torch.dtype | None,
    *,
    uses_model_params_as_master_weights: bool,
    is_peft: bool = False,
    context: str = "recipe",
    logger: logging.Logger | None = None,
) -> str | torch.dtype | None:
    """Resolve model storage dtype for full-parameter training without mutating config.

    Built-in ``torch.optim`` optimizers update the resident parameter in place and keep
    no internal fp32 master copy, so the model parameters *are* the master copy. Storing
    them in bf16 therefore makes optimizer updates and state bf16, which degrades training
    precision relative to frameworks that keep an fp32 master. When the user has not
    explicitly chosen a storage dtype, return ``float32`` so the parameters act as the
    fp32 master copy. The declarative config remains unchanged; recipes pass the returned
    value to model construction as a runtime override.

    Returns ``requested_dtype`` unchanged when:
      * ``is_peft`` is True (base weights are frozen; only small adapters train), or
      * the optimizer manages master weights separately, or
      * ``model.torch_dtype`` is already set to a concrete (non-``auto``) value.

    Args:
        requested_dtype: Declarative ``model.torch_dtype`` value, or ``None`` when omitted.
        uses_model_params_as_master_weights: Whether the optimizer updates resident model
            parameters without maintaining a separate master-weight copy.
        is_peft: Whether this is a PEFT/LoRA run.
        context: Short label used for log de-duplication.
        logger: Optional logger; defaults to this module's logger.

    Returns:
        The effective model storage dtype to pass to model construction.
    """
    if is_peft or not uses_model_params_as_master_weights:
        return requested_dtype
    if requested_dtype is not None and not (isinstance(requested_dtype, str) and requested_dtype == "auto"):
        # User explicitly chose a storage dtype; always honor it.
        return requested_dtype

    if _is_rank_zero() and context not in _DTYPE_RESOLVED_CONTEXTS:
        log = logger if logger is not None else logging.getLogger(__name__)
        log.info(
            "No explicit model.torch_dtype set for full-parameter training with a "
            "torch.optim optimizer (which keeps no fp32 master copy). Defaulting the "
            "effective model.torch_dtype to float32 so model parameters act as the fp32 "
            "master copy. Set model.torch_dtype=bfloat16 to keep bf16 storage, or use "
            "Transformer Engine FusedAdam with master_weights=True. See "
            "docs/guides/mixed-precision-training.mdx."
        )
        _DTYPE_RESOLVED_CONTEXTS.add(context)
    return "float32"


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
        "Engine FusedAdam with master_weights=True. See docs/guides/mixed-precision-training.mdx."
    )

    _WARNED_CONTEXTS.add(context)


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

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

import torch
import torch.distributed as dist
from torch.optim import Optimizer

_WARNED_CONTEXTS: set[str] = set()


def warn_if_torch_adam_with_bf16_params(
    *,
    optimizer: Optimizer | Iterable[Optimizer] | None = None,
    parameters: Iterable[torch.nn.Parameter] | None = None,
    is_peft: bool = False,
    context: str = "recipe",
    logger: logging.Logger | None = None,
) -> None:
    """Warn about full-parameter bf16 training with vanilla torch Adam optimizers."""
    if is_peft or not _is_rank_zero() or context in _WARNED_CONTEXTS:
        return

    if not _is_torch_adam_optimizer(optimizer):
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


def _is_rank_zero() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _is_torch_adam_optimizer(optimizer: Optimizer | Iterable[Optimizer] | None) -> bool:
    if optimizer is None:
        return False
    optimizers = optimizer if isinstance(optimizer, Iterable) else [optimizer]
    return any(isinstance(opt, (torch.optim.Adam, torch.optim.AdamW)) for opt in optimizers)


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

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

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from nemo_automodel.components.loss.config import (
    FusedLinearCEConfig,
    KDLossConfig,
    LossConfig,
    MaskedCrossEntropyConfig,
    TEParallelCEConfig,
)

if TYPE_CHECKING:
    from torch import nn


def _get_loss_class(config: LossConfig) -> type:
    """Map a typed config to its loss class (lazy imports to avoid heavy deps at module level)."""
    if isinstance(config, MaskedCrossEntropyConfig):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        return MaskedCrossEntropy
    if isinstance(config, FusedLinearCEConfig):
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        return FusedLinearCrossEntropy
    if isinstance(config, TEParallelCEConfig):
        from nemo_automodel.components.loss.te_parallel_ce import TEParallelCrossEntropy

        return TEParallelCrossEntropy
    if isinstance(config, KDLossConfig):
        from nemo_automodel.components.loss.kd_loss import KDLoss

        return KDLoss
    raise ValueError(f"Unknown loss config type: {type(config).__name__}. Use loss_factory for custom losses.")


def build_loss_fn(
    config: LossConfig | None = None,
    *,
    loss_factory: Callable[..., nn.Module] | None = None,
    loss_kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build a loss function.

    Accepts either a ``LossConfig`` (preferred for external integrations)
    or an explicit ``(loss_factory, loss_kwargs)`` pair (used by
    ``_component_builders`` when resolving from YAML).

    Args:
        config: Typed loss config.  When provided, ``loss_factory`` and
            ``loss_kwargs`` are derived from it.
        loss_factory: Callable or class that creates the loss function.
            Ignored when ``config`` is provided.
        loss_kwargs: Optional keyword arguments passed to the loss factory.
            Ignored when ``config`` is provided.

    Returns:
        Instantiated loss function.
    """
    if config is not None:
        loss_factory = _get_loss_class(config)
        loss_kwargs = config.to_kwargs()
    elif loss_factory is None:
        raise ValueError("Either config or loss_factory must be provided")

    return loss_factory(**dict(loss_kwargs or {}))

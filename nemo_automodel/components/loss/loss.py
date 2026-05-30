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

"""Typed loss configs + builder (TorchTitan-style).

Each loss config is a plain dataclass exposing its full parameter surface as
named fields (no opaque ``**kwargs``) and a ``build()`` method that constructs
the loss module directly — lazy imports keep optional kernel deps out of module
load.  Reading the dataclass tells you exactly what you can configure.

:func:`build_loss_fn` is the single entry point.  It dispatches on its argument:

- a typed :class:`LossConfig` instance — the Automodel-native path; per-loss
  construction delegates to ``config.build()``.
- a loss *dotted path* or *class* plus arbitrary ``**loss_kwargs`` — the escape
  hatch for external integrations (e.g. veRL) and the YAML recipe path.  Adding
  a new typed config never forces the caller to change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


# ---------------------------------------------------------------------------
# Typed loss configs
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """Base loss config.  Subclasses expose their full field surface and
    implement :meth:`build`."""

    def build(self) -> nn.Module:
        """Construct the loss module."""
        raise NotImplementedError(f"{type(self).__name__} must implement build()")


@dataclass
class MaskedCrossEntropyConfig(LossConfig):
    """``MaskedCrossEntropy``.

    Attributes:
        fp32_upcast: Cast logits to float32 before computing CE.
        ignore_index: Label value marking padding tokens.
        reduction: Reduction mode — ``"sum"``, ``"mean"``, or ``"none"``.
    """

    fp32_upcast: bool = True
    ignore_index: int = -100
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        return MaskedCrossEntropy(
            fp32_upcast=self.fp32_upcast,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


@dataclass
class FusedLinearCEConfig(LossConfig):
    """``FusedLinearCrossEntropy``.

    Attributes:
        ignore_index: Label value marking padding tokens.
        logit_softcapping: Softcap logits before CE (0 = disabled).
        reduction: Reduction mode.
    """

    ignore_index: int = -100
    logit_softcapping: float = 0.0
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        return FusedLinearCrossEntropy(
            ignore_index=self.ignore_index,
            logit_softcapping=self.logit_softcapping,
            reduction=self.reduction,
        )


@dataclass
class TEParallelCEConfig(LossConfig):
    """``TEParallelCrossEntropy``.

    Attributes:
        ignore_index: Label value marking padding tokens.
        reduction: Reduction mode.
    """

    ignore_index: int = -100
    reduction: str = "sum"

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.te_parallel_ce import TEParallelCrossEntropy

        return TEParallelCrossEntropy(
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


@dataclass
class KDLossConfig(LossConfig):
    """``KDLoss`` (knowledge distillation).

    Attributes:
        ignore_index: Label value marking padding tokens.
        temperature: Softmax temperature T.  Loss is scaled by T².
        fp32_upcast: Cast logits to float32 for numerical stability.
    """

    ignore_index: int = -100
    temperature: float = 1.0
    fp32_upcast: bool = True

    def build(self) -> nn.Module:
        from nemo_automodel.components.loss.kd_loss import KDLoss

        return KDLoss(
            ignore_index=self.ignore_index,
            temperature=self.temperature,
            fp32_upcast=self.fp32_upcast,
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_loss_fn(loss: LossConfig | Callable[..., nn.Module], **loss_kwargs: Any) -> nn.Module:
    """Build a loss function.

    Single entry point.  Dispatches on ``loss``:

    - **Typed config** (:class:`LossConfig` instance) — the Automodel-native
      path.  Hyperparameters come from the config; ``**loss_kwargs`` must be
      empty.  Construction delegates to ``config.build()``.
    - **Loss class / callable** (e.g. ``MaskedCrossEntropy``) plus
      ``**loss_kwargs`` — the integration / YAML escape hatch.  The caller
      resolves any dotted path to a callable; the component never does string
      resolution.

    Args:
        loss: Typed :class:`LossConfig` instance, or a loss class/callable to
            construct with ``**loss_kwargs``.
        **loss_kwargs: Constructor kwargs for the class/callable form.  Must be
            empty when ``loss`` is a typed config.

    Returns:
        Instantiated loss function.
    """
    if isinstance(loss, LossConfig):
        if loss_kwargs:
            raise ValueError(
                "Loss hyperparameters must be set on the config, not passed as keyword "
                f"arguments to build_loss_fn (got {sorted(loss_kwargs)})."
            )
        return loss.build()
    if isinstance(loss, type) and issubclass(loss, LossConfig):
        raise TypeError(f"Pass a LossConfig instance, not the class {loss.__name__} (e.g. {loss.__name__}()).")
    if not callable(loss):
        raise TypeError(
            f"build_loss_fn expects a LossConfig or a loss class/callable, got {type(loss).__name__}.  "
            "Resolve dotted paths in the caller."
        )
    return loss(**loss_kwargs)


__all__ = [
    "FusedLinearCEConfig",
    "KDLossConfig",
    "LossConfig",
    "MaskedCrossEntropyConfig",
    "TEParallelCEConfig",
    "build_loss_fn",
]

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

"""Public config surface for the loss component.

Look here for the typed parameters that drive loss function selection.
Look at ``api.py`` for the builder that consumes these configs.

Loss hierarchy::

    LossConfig                   (generic fallback — any loss via name + extra_kwargs)
    ├── MaskedCrossEntropyConfig (MaskedCrossEntropy)
    ├── FusedLinearCEConfig      (FusedLinearCrossEntropy)
    ├── TEParallelCEConfig       (TEParallelCrossEntropy)
    └── KDLossConfig             (KDLoss)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Loss configs
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """Generic loss configuration (fallback for third-party loss functions).

    For known losses, prefer the typed subclasses below.
    For unknown / new losses, use this class directly with ``extra_kwargs``.

    Attributes:
        name: Dotted import path to the loss class.
        extra_kwargs: Pass-through keyword arguments forwarded to the
            loss constructor.
    """

    name: str = "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_kwargs(self) -> dict[str, Any]:
        """Return the full kwargs dict for the loss constructor."""
        return {**self.extra_kwargs}

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> LossConfig:
        """Create the appropriate typed config from a loss name string.

        Known losses return a typed subclass with full field validation.
        Unknown losses return a base ``LossConfig`` with ``extra_kwargs``.

        Args:
            name: Dotted import path (e.g. ``"nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"``).
            **kwargs: Loss hyper-parameters.

        Returns:
            Typed ``LossConfig`` subclass, or the generic base.
        """
        config_cls = _LOSS_REGISTRY.get(name, None)
        if config_cls is None:
            return LossConfig(name=name, extra_kwargs=kwargs)
        return config_cls(name=name, **kwargs)


@dataclass
class MaskedCrossEntropyConfig(LossConfig):
    """Configuration for ``MaskedCrossEntropy``.

    Attributes:
        fp32_upcast: Cast logits to float32 before computing CE.
        ignore_index: Label value marking padding tokens.
        reduction: Reduction mode — ``"sum"``, ``"mean"``, or ``"none"``.
    """

    name: str = "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"
    fp32_upcast: bool = True
    ignore_index: int = -100
    reduction: str = "sum"

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "fp32_upcast": self.fp32_upcast,
            "ignore_index": self.ignore_index,
            "reduction": self.reduction,
            **self.extra_kwargs,
        }


@dataclass
class FusedLinearCEConfig(LossConfig):
    """Configuration for ``FusedLinearCrossEntropy``.

    Attributes:
        ignore_index: Label value marking padding tokens.
        logit_softcapping: Softcap logits before CE (0 = disabled).
        reduction: Reduction mode.
    """

    name: str = "nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy"
    ignore_index: int = -100
    logit_softcapping: float = 0.0
    reduction: str = "sum"

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "ignore_index": self.ignore_index,
            "logit_softcapping": self.logit_softcapping,
            "reduction": self.reduction,
            **self.extra_kwargs,
        }


@dataclass
class TEParallelCEConfig(LossConfig):
    """Configuration for ``TEParallelCrossEntropy``.

    Attributes:
        ignore_index: Label value marking padding tokens.
        reduction: Reduction mode.
    """

    name: str = "nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy"
    ignore_index: int = -100
    reduction: str = "sum"

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "ignore_index": self.ignore_index,
            "reduction": self.reduction,
            **self.extra_kwargs,
        }


@dataclass
class KDLossConfig(LossConfig):
    """Configuration for ``KDLoss`` (knowledge distillation).

    Attributes:
        ignore_index: Label value marking padding tokens.
        temperature: Softmax temperature T.  Loss is scaled by T².
        fp32_upcast: Cast logits to float32 for numerical stability.
    """

    name: str = "nemo_automodel.components.loss.kd_loss.KDLoss"
    ignore_index: int = -100
    temperature: float = 1.0
    fp32_upcast: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "ignore_index": self.ignore_index,
            "temperature": self.temperature,
            "fp32_upcast": self.fp32_upcast,
            **self.extra_kwargs,
        }


# ---------------------------------------------------------------------------
# Registry — maps dotted name → typed config subclass
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: dict[str, type[LossConfig]] = {
    "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy": MaskedCrossEntropyConfig,
    "nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy": FusedLinearCEConfig,
    "nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy": TEParallelCEConfig,
    "nemo_automodel.components.loss.kd_loss.KDLoss": KDLossConfig,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _resolve_loss(name: str) -> Any:
    """Resolve a dotted path to a loss class."""
    import importlib

    parts = name.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Expected a dotted path like 'nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy', got '{name}'"
        )
    module_path, cls_name = parts
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise ImportError(f"Cannot find '{cls_name}' in module '{module_path}'")
    return cls


__all__ = [
    "LossConfig",
    "MaskedCrossEntropyConfig",
    "FusedLinearCEConfig",
    "TEParallelCEConfig",
    "KDLossConfig",
]

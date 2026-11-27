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

"""Precision of every Transformer Engine GEMM module of a model that trains under a quantized autocast.

A :class:`TEPrecisionMap` is owned by a model family: it knows the family's module names and turns the names
of a model's modules into a :class:`TEPrecisionPlan` for its Transformer Engine GEMM modules and
gives every module the forward of its precision (:meth:`TEPrecisionMap.prepare`). ``TENvfp4Config`` hands it the
model and never interprets module names of a model family itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from torch import nn

TEPrecision: TypeAlias = Literal["nvfp4", "mxfp8", "high_precision"]
"""How the GEMMs of a module run inside the NVFP4 autocast: under the NVFP4 recipe, under Transformer Engine's
MXFP8 recipe, or outside the autocast in the model dtype."""

TE_PRECISIONS: tuple[TEPrecision, ...] = ("nvfp4", "mxfp8", "high_precision")


@dataclass(frozen=True)
class TEModulePrecision:
    """Precision of one Transformer Engine GEMM module (``Linear`` or ``GroupedLinear``)."""

    name: str
    """Qualified name of the GEMM module in ``model.named_modules()``."""
    module_class: str
    """Label of the kind of module, as the model family names it, e.g. ``"mamba_in_proj"``."""
    layer: int | None
    """Index of the decoder layer that holds the module; ``None`` outside the decoder stack."""
    precision: TEPrecision
    """How the GEMMs of the module run."""
    scope: str
    """Qualified name of the module whose forward selects the precision: ``name`` or one of its ancestors.
    An ancestor is used when it reads the active recipe itself, as grouped experts do to pad their inputs."""

    def __post_init__(self) -> None:
        """Reject a scope that does not contain the module."""
        if self.scope != self.name and not self.name.startswith(f"{self.scope}."):
            raise ValueError(f"Scope {self.scope!r} is not {self.name!r} or one of its ancestors")


@dataclass(frozen=True)
class TEPrecisionPlan:
    """Precision of every Transformer Engine GEMM module of one model."""

    modules: tuple[TEModulePrecision, ...]
    """One entry per GEMM module."""
    num_layers: int
    """Number of decoder layers; 0 when the plan has no notion of layers."""
    high_precision_tail_layers: int
    """Number of decoder layers at the end of the stack whose GEMMs all run in high precision."""

    def __post_init__(self) -> None:
        """Reject scopes with two precisions and scopes inside other scopes."""
        scopes = self.scopes()
        for module in self.modules:
            if scopes[module.scope] != module.precision:
                raise ValueError(
                    f"Scope {module.scope!r} holds modules with different precisions "
                    f"({scopes[module.scope]} and {module.precision})"
                )
        for scope in scopes:
            parent = next((other for other in scopes if scope.startswith(f"{other}.")), None)
            if parent is not None:
                raise ValueError(f"Scope {scope!r} lies inside scope {parent!r}")

    def scopes(self) -> dict[str, TEPrecision]:
        """Precision by scope name, in the order of the first module of each scope."""
        scopes: dict[str, TEPrecision] = {}
        for module in self.modules:
            scopes.setdefault(module.scope, module.precision)
        return scopes

    def counts(self) -> dict[str, dict[TEPrecision, int]]:
        """Number of GEMM modules by module class and precision, classes in order of first appearance."""
        counts: dict[str, dict[TEPrecision, int]] = {}
        for module in self.modules:
            by_precision = counts.setdefault(module.module_class, dict.fromkeys(TE_PRECISIONS, 0))
            by_precision[module.precision] += 1
        return counts

    def summary(self) -> str:
        """Table of :meth:`counts` with a line for the layer range, for the startup log."""
        lines = []
        if self.num_layers > 0:
            first_tail_layer = self.num_layers - self.high_precision_tail_layers
            layer_range = f"decoder layers: {self.num_layers}"
            if first_tail_layer > 0:
                layer_range += f"; layers 0-{first_tail_layer - 1} follow the precision of their module class"
            if self.high_precision_tail_layers > 0:
                layer_range += f"; layers {first_tail_layer}-{self.num_layers - 1} run in high precision"
            lines.append(layer_range)
        counts = self.counts()
        width = max([len("module class"), *(len(module_class) for module_class in counts)])
        header = f"{'module class':<{width}}" + "".join(f"  {precision:>14}" for precision in TE_PRECISIONS)
        lines.append(header)
        totals = dict.fromkeys(TE_PRECISIONS, 0)
        for module_class, by_precision in counts.items():
            lines.append(
                f"{module_class:<{width}}" + "".join(f"  {by_precision[precision]:>14}" for precision in TE_PRECISIONS)
            )
            for precision in TE_PRECISIONS:
                totals[precision] += by_precision[precision]
        lines.append(f"{'total':<{width}}" + "".join(f"  {totals[precision]:>14}" for precision in TE_PRECISIONS))
        return "\n".join(lines)


class TEPrecisionMap(ABC):
    """Base class of the rules that give every Transformer Engine GEMM module of a model its precision.

    A model family subclasses it with the module classes it knows, e.g. ``NemotronV3PrecisionMap``; an
    instance is set as ``te_nvfp4.precision_map``.
    """

    @abstractmethod
    def prepare(self, model: nn.Module) -> None:
        """Give every Transformer Engine GEMM module of ``model`` the forward of its precision; idempotent.

        Args:
            model: Model (or pipeline stage) whose modules are inspected by qualified name.

        Raises:
            ValueError: If a GEMM module is not one the map knows, or the map quantizes a module that is not
                a Transformer Engine GEMM module; nothing is left unassigned or silently in high precision.
        """
        ...

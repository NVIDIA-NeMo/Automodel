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

"""Unified Quantization-Aware Training (QAT) configuration for NeMo-AutoModel.

This module provides:
- QATConfig and QATRule: Legacy TorchAO settings or independent weight-rule plans
- Thin wrappers to instantiate and apply torchao QAT quantizers to models (prepare)
- Toggle fake-quant on/off during training (for delayed fake-quant)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Literal

import torch

from nemo_automodel.components.quantization.weight_qat import WeightQuantizationConfig
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

_HAS_TORCHAO_QAT, _ = safe_import("torchao.quantization.qat.linear")
if _HAS_TORCHAO_QAT:
    from torchao.quantization.qat import (
        Int4WeightOnlyQATQuantizer,
        Int8DynActInt4WeightQATQuantizer,
    )
    from torchao.quantization.qat.linear import (
        disable_4w_fake_quant,
        disable_8da4w_fake_quant,
        enable_4w_fake_quant,
        enable_8da4w_fake_quant,
    )


@dataclass(frozen=True)
class QATRule:
    """Select weights independently of PEFT with case-sensitive full-name globs.

    Args:
        target_modules: Nonempty full module-name patterns. An empty string
            selects the root module. YAML lists normalize to immutable tuples.
        weight: Typed numerical settings; plain YAML mappings are converted by
            ConfigNode, not by this component.
    """

    target_modules: tuple[str, ...]
    weight: WeightQuantizationConfig = field(metadata={"instantiate": True})

    def __post_init__(self) -> None:
        if not isinstance(self.target_modules, (tuple, list)) or not self.target_modules:
            raise TypeError("target_modules must be a nonempty tuple/list of full-name glob strings")
        if any(not isinstance(pattern, str) for pattern in self.target_modules):
            raise TypeError("target_modules must contain only strings")
        if not isinstance(self.weight, WeightQuantizationConfig):
            raise TypeError("weight must be a WeightQuantizationConfig")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))


@dataclass
class QATConfig:
    """
    Configuration for Quantization-Aware Training (QAT).

    This config controls how QAT quantizers are instantiated and applied to models.
    QAT is enabled when this config is provided to from_pretrained/from_config.

    Attributes:
        quantizer_type (Literal["int8_dynact_int4weight", "int4_weight_only"]):
            Type of QAT quantizer to use.
            - "int8_dynact_int4weight": Int8 dynamic activation with Int4 weight
              quantization. Uses Int8DynActInt4WeightQATQuantizer from torchao.
            - "int4_weight_only": Int4 weight-only quantization. Uses
                            Int4WeightOnlyQATQuantizer from torchao.
        **quantizer_kwargs: Additional keyword arguments forwarded directly to
            the torchao quantizer constructor (e.g. groupsize, padding_allowed,
            inner_k_tiles).
        target_modules: Independent, nonempty QAT selectors, used with weight.
        weight: Numerical configuration for the simple single-rule form.
        rules: Nonempty typed rules, mutually exclusive with target_modules/weight.
            Both new forms are mutually exclusive with explicit TorchAO settings.
            Plain nested YAML mappings normalize at ConfigNode.instantiate().
    """

    quantizer_type: Literal["int8_dynact_int4weight", "int4_weight_only"] = "int8_dynact_int4weight"
    target_modules: tuple[str, ...] | None = None
    weight: WeightQuantizationConfig | None = field(default=None, metadata={"instantiate": True})
    rules: tuple[QATRule, ...] | None = field(default=None, metadata={"instantiate": True})

    def __init__(
        self,
        quantizer_type: Literal["int8_dynact_int4weight", "int4_weight_only"] | None = None,
        *,
        target_modules: tuple[str, ...] | None = None,
        weight: WeightQuantizationConfig | None = None,
        rules: tuple[QATRule, ...] | None = None,
        **quantizer_kwargs,
    ) -> None:
        if rules is not None and (target_modules is not None or weight is not None):
            raise ValueError("rules and target_modules/weight are mutually exclusive")
        if target_modules is not None or weight is not None or rules is not None:
            if quantizer_type is not None or quantizer_kwargs:
                raise ValueError("weight/rules and TorchAO quantizer_type/kwargs are mutually exclusive")
            if rules is None:
                rule = QATRule(target_modules, weight)
                target_modules = rule.target_modules
            else:
                if not isinstance(rules, (tuple, list)) or not rules:
                    raise TypeError("rules must be a nonempty tuple/list of QATRule objects")
                if any(not isinstance(rule, QATRule) for rule in rules):
                    raise TypeError("rules must contain QATRule objects")
                rules = tuple(rules)
        self.quantizer_type = quantizer_type if quantizer_type is not None else "int8_dynact_int4weight"
        self.quantizer_kwargs = quantizer_kwargs
        self.target_modules = target_modules
        self.weight = weight
        self.rules = rules

    @property
    def is_weight_qat(self) -> bool:
        """Classify weight/rules settings, without asserting module compatibility."""
        return self.weight is not None or self.rules is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize legacy TorchAO settings; weight/rules use the shared boundary.

        Raises:
            ValueError: If weight/rules would be lost by the legacy serializer.
        """
        if self.is_weight_qat:
            raise ValueError("QATConfig.to_dict is legacy only; serialize via ConfigNode/shared boundary")
        return {
            "quantizer_type": self.quantizer_type,
            **self.quantizer_kwargs,
        }

    def create_quantizer(self) -> Int4WeightOnlyQATQuantizer | Int8DynActInt4WeightQATQuantizer:
        """Create a legacy TorchAO quantizer, preserving the existing entry point."""
        if self.is_weight_qat:
            raise ValueError("create_quantizer supports TorchAO settings only; use build for weight/rules")
        return self.build()

    def build(self) -> tuple[QATRule, ...] | Int4WeightOnlyQATQuantizer | Int8DynActInt4WeightQATQuantizer:
        """Construct a resolved numerical rule plan or the legacy TorchAO quantizer.

        Returns:
            Immutable rules with resolved block sizes for the bridge to compose,
            or a TorchAO quantizer. No runtime objects are cached on this config.

        Raises:
            ValueError: If quantizer_type is not recognized.
        """
        if self.is_weight_qat:
            rules = self.rules if self.rules is not None else (QATRule(self.target_modules, self.weight),)
            return tuple(replace(rule, weight=rule.weight.build().config) for rule in rules)
        if not _HAS_TORCHAO_QAT:
            raise ImportError("TorchAO QAT is required for legacy INT4 quantizer settings")
        # Default precision to bfloat16 so fake-quant params match the model dtype,
        # preventing FSDP mixed-dtype errors. User kwargs can override.
        kwargs = {
            "precision": torch.bfloat16,
            "scales_precision": torch.bfloat16,
            **self.quantizer_kwargs,
        }
        if self.quantizer_type == "int8_dynact_int4weight":
            return Int8DynActInt4WeightQATQuantizer(**kwargs)
        elif self.quantizer_type == "int4_weight_only":
            return Int4WeightOnlyQATQuantizer(**kwargs)
        else:
            raise ValueError(f"Unknown quantizer_type: {self.quantizer_type}")


_QUANTIZER_TO_MODE = (
    {Int8DynActInt4WeightQATQuantizer: "8da4w-qat", Int4WeightOnlyQATQuantizer: "4w-qat"} if _HAS_TORCHAO_QAT else {}
)

_DISABLE_FN_BY_MODE = (
    {"8da4w-qat": disable_8da4w_fake_quant, "4w-qat": disable_4w_fake_quant} if _HAS_TORCHAO_QAT else {}
)

_ENABLE_FN_BY_MODE = {"8da4w-qat": enable_8da4w_fake_quant, "4w-qat": enable_4w_fake_quant} if _HAS_TORCHAO_QAT else {}


def get_quantizer_mode(quantizer: object) -> str | None:
    """Return a short mode string for a known torchao QAT quantizer.

    Returns None when the quantizer is unrecognized.
    """

    return _QUANTIZER_TO_MODE.get(type(quantizer), None)


def get_disable_fake_quant_fn(mode: str) -> Callable | None:
    """Return the disable fake-quant function for a given quantizer mode."""

    return _DISABLE_FN_BY_MODE.get(mode, None)


def get_enable_fake_quant_fn(mode: str) -> Callable | None:
    """Return the enable fake-quant function for a given quantizer mode."""

    return _ENABLE_FN_BY_MODE.get(mode, None)


def prepare_qat_model(model, quantizer) -> tuple[object, str | None]:
    """Apply a torchao QAT quantizer to the given model.

    Returns the (possibly wrapped) model and a mode string if recognized.
    """

    if not hasattr(quantizer, "prepare"):
        raise ValueError("Provided quantizer does not implement a prepare(model) method")

    logger.info("Preparing model for QAT using %s", type(quantizer).__name__)
    model = quantizer.prepare(model)
    mode = get_quantizer_mode(quantizer)
    if mode is None:
        logger.warning("Unknown QAT quantizer %s; fake-quant toggling will be unavailable.", type(quantizer).__name__)
    return model, mode


__all__ = [
    "QATConfig",
    "QATRule",
    "get_quantizer_mode",
    "get_disable_fake_quant_fn",
    "get_enable_fake_quant_fn",
    "prepare_qat_model",
]

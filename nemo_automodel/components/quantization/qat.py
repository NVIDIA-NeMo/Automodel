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

"""TorchAO Quantization-Aware Training (QAT) helpers for NeMo-AutoModel.

This module provides:
- QATConfig: Configuration class for QAT settings
- Thin wrappers to instantiate and apply torchao QAT quantizers to models (prepare)
- Toggle fake-quant on/off during training (for delayed fake-quant)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional

import torch

logger = logging.getLogger(__name__)

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
              Good balance of accuracy and inference speed.
            - "int4_weight_only": Int4 weight-only quantization. Uses
              Int4WeightOnlyQATQuantizer from torchao. More aggressive compression,
              may have slightly lower accuracy.
        precision (torch.dtype): Data type for model weights during QAT training.
            Typically torch.bfloat16 for mixed-precision training.
        scales_precision (torch.dtype): Data type for quantization scale factors.
            Using higher precision (e.g., torch.bfloat16) for scales can improve
            accuracy.
    """

    quantizer_type: Literal["int8_dynact_int4weight", "int4_weight_only"] = "int8_dynact_int4weight"
    precision: torch.dtype = torch.bfloat16
    scales_precision: torch.dtype = torch.bfloat16

    def __init__(
        self,
        quantizer_type: Literal["int8_dynact_int4weight", "int4_weight_only"] = "int8_dynact_int4weight",
        precision: torch.dtype = torch.bfloat16,
        scales_precision: torch.dtype = torch.bfloat16,
    ):
        self.quantizer_type = quantizer_type
        self.precision = precision
        self.scales_precision = scales_precision

    @classmethod
    def from_config_node(cls, config_node) -> "QATConfig":
        """Create QATConfig from a configuration node (e.g., Hydra config)."""
        if config_node is None:
            return cls()

        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if hasattr(config_node, field_name):
                kwargs[field_name] = getattr(config_node, field_name)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "quantizer_type": self.quantizer_type,
            "precision": self.precision,
            "scales_precision": self.scales_precision,
        }

    def create_quantizer(self):
        """Create and return the appropriate QAT quantizer based on config.

        Returns:
            A torchao QAT quantizer instance (Int8DynActInt4WeightQATQuantizer
            or Int4WeightOnlyQATQuantizer).

        Raises:
            ValueError: If quantizer_type is not recognized.
        """
        if self.quantizer_type == "int8_dynact_int4weight":
            return Int8DynActInt4WeightQATQuantizer(
                precision=self.precision,
                scales_precision=self.scales_precision,
            )
        elif self.quantizer_type == "int4_weight_only":
            return Int4WeightOnlyQATQuantizer(
                precision=self.precision,
                scales_precision=self.scales_precision,
            )
        else:
            raise ValueError(f"Unknown quantizer_type: {self.quantizer_type}")


_QUANTIZER_TO_MODE = {
    Int8DynActInt4WeightQATQuantizer: "8da4w-qat",
    Int4WeightOnlyQATQuantizer: "4w-qat",
}

_DISABLE_FN_BY_MODE = {
    "8da4w-qat": disable_8da4w_fake_quant,
    "4w-qat": disable_4w_fake_quant,
}

_ENABLE_FN_BY_MODE = {
    "8da4w-qat": enable_8da4w_fake_quant,
    "4w-qat": enable_4w_fake_quant,
}


def get_quantizer_mode(quantizer: object) -> Optional[str]:
    """Return a short mode string for a known torchao QAT quantizer.

    Returns None when the quantizer is unrecognized.
    """

    return _QUANTIZER_TO_MODE.get(type(quantizer), None)


def get_disable_fake_quant_fn(mode: str) -> Optional[Callable]:
    """Return the disable fake-quant function for a given quantizer mode."""

    return _DISABLE_FN_BY_MODE.get(mode, None)


def get_enable_fake_quant_fn(mode: str) -> Optional[Callable]:
    """Return the enable fake-quant function for a given quantizer mode."""

    return _ENABLE_FN_BY_MODE.get(mode, None)


def prepare_qat_model(model, quantizer) -> tuple[object, Optional[str]]:
    """Apply a torchao QAT quantizer to the given model.

    Returns the (possibly wrapped) model and a mode string if recognized.
    """

    if not hasattr(quantizer, "prepare"):
        raise ValueError("Provided quantizer does not implement a prepare(model) method")

    logger.info("Preparing model for QAT using %s".format(type(quantizer).__name__))
    model = quantizer.prepare(model)
    mode = get_quantizer_mode(quantizer)
    if mode is None:
        logger.warning(
            "Unknown QAT quantizer %s; fake-quant toggling will be unavailable.".format(type(quantizer).__name__)
        )
    return model, mode


__all__ = [
    "QATConfig",
    "get_quantizer_mode",
    "get_disable_fake_quant_fn",
    "get_enable_fake_quant_fn",
    "prepare_qat_model",
]

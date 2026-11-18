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
from typing import Callable, Literal, Protocol, TypedDict, Unpack

import torch
import torch.nn as nn

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

QATQuantizerType = Literal["int8_dynact_int4weight", "int4_weight_only"]


class _QATQuantizerKwargs(TypedDict, total=False):
    groupsize: int
    padding_allowed: bool
    inner_k_tiles: int | None
    precision: torch.dtype
    scales_precision: torch.dtype


class _QATConfigRequired(TypedDict):
    quantizer_type: QATQuantizerType


class _QATConfigDict(_QATConfigRequired, _QATQuantizerKwargs, total=False):
    pass


class _QATQuantizer(Protocol):
    def prepare(self, model: nn.Module) -> nn.Module: ...


_QAT_QUANTIZER_KWARG_KEYS = frozenset(
    {"groupsize", "padding_allowed", "inner_k_tiles", "precision", "scales_precision"}
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
        **quantizer_kwargs: Additional keyword arguments forwarded directly to
            the torchao quantizer constructor (e.g. groupsize, padding_allowed,
            inner_k_tiles).
    """

    quantizer_type: QATQuantizerType = "int8_dynact_int4weight"

    def __init__(
        self,
        quantizer_type: QATQuantizerType = "int8_dynact_int4weight",
        **quantizer_kwargs: Unpack[_QATQuantizerKwargs],
    ) -> None:
        self.quantizer_type = quantizer_type
        self.quantizer_kwargs: _QATQuantizerKwargs = quantizer_kwargs

    def to_dict(self) -> _QATConfigDict:
        """Convert config to dictionary."""
        return {
            "quantizer_type": self.quantizer_type,
            **self.quantizer_kwargs,
        }

    def create_quantizer(self) -> Int8DynActInt4WeightQATQuantizer | Int4WeightOnlyQATQuantizer:
        """Create and return the appropriate QAT quantizer based on config.

        Returns:
            A torchao QAT quantizer instance (Int8DynActInt4WeightQATQuantizer
            or Int4WeightOnlyQATQuantizer).

        Raises:
            ValueError: If quantizer_type or quantizer_kwargs are not recognized.
        """
        unknown = set(self.quantizer_kwargs) - _QAT_QUANTIZER_KWARG_KEYS
        if unknown:
            raise ValueError(f"Unknown QAT quantizer kwargs: {sorted(unknown)}")

        # Default precision to bfloat16 so fake-quant params match the model dtype,
        # preventing FSDP mixed-dtype errors. User kwargs can override.
        if self.quantizer_type == "int8_dynact_int4weight":
            if "inner_k_tiles" in self.quantizer_kwargs:
                raise ValueError("inner_k_tiles is not supported by int8_dynact_int4weight")
            return Int8DynActInt4WeightQATQuantizer(
                groupsize=self.quantizer_kwargs.get("groupsize", 256),
                padding_allowed=self.quantizer_kwargs.get("padding_allowed", False),
                precision=self.quantizer_kwargs.get("precision", torch.bfloat16),
                scales_precision=self.quantizer_kwargs.get("scales_precision", torch.bfloat16),
            )

        if self.quantizer_type == "int4_weight_only":
            if "padding_allowed" in self.quantizer_kwargs:
                raise ValueError("padding_allowed is not supported by int4_weight_only")
            return Int4WeightOnlyQATQuantizer(
                groupsize=self.quantizer_kwargs.get("groupsize", 256),
                inner_k_tiles=self.quantizer_kwargs.get("inner_k_tiles", 8),
                precision=self.quantizer_kwargs.get("precision", torch.bfloat16),
                scales_precision=self.quantizer_kwargs.get("scales_precision", torch.bfloat16),
            )

        raise ValueError(f"Unknown quantizer_type: {self.quantizer_type}")


_QUANTIZER_TO_MODE: dict[type, str] = {
    Int8DynActInt4WeightQATQuantizer: "8da4w-qat",
    Int4WeightOnlyQATQuantizer: "4w-qat",
}

_DISABLE_FN_BY_MODE: dict[str, Callable[[nn.Module], None]] = {
    "8da4w-qat": disable_8da4w_fake_quant,
    "4w-qat": disable_4w_fake_quant,
}

_ENABLE_FN_BY_MODE: dict[str, Callable[[nn.Module], None]] = {
    "8da4w-qat": enable_8da4w_fake_quant,
    "4w-qat": enable_4w_fake_quant,
}


def get_quantizer_mode(quantizer: object) -> str | None:
    """Return a short mode string for a known torchao QAT quantizer.

    Returns None when the quantizer is unrecognized.
    """

    return _QUANTIZER_TO_MODE.get(type(quantizer), None)


def get_disable_fake_quant_fn(mode: str) -> Callable[[nn.Module], None] | None:
    """Return the disable fake-quant function for a given quantizer mode."""

    return _DISABLE_FN_BY_MODE.get(mode, None)


def get_enable_fake_quant_fn(mode: str) -> Callable[[nn.Module], None] | None:
    """Return the enable fake-quant function for a given quantizer mode."""

    return _ENABLE_FN_BY_MODE.get(mode, None)


def prepare_qat_model(model: nn.Module, quantizer: _QATQuantizer) -> tuple[nn.Module, str | None]:
    """Apply a torchao QAT quantizer to the given model.

    Returns the (possibly wrapped) model and a mode string if recognized.
    """

    # Runtime guard for callers that pass untyped third-party quantizers.
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
    "get_quantizer_mode",
    "get_disable_fake_quant_fn",
    "get_enable_fake_quant_fn",
    "prepare_qat_model",
]

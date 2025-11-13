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

This module provides thin wrappers to:
- Instantiate and apply torchao QAT quantizers to models (prepare)
- Toggle fake-quant on/off during training (for delayed fake-quant)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torchao.quantization.qat.api import FakeQuantizeConfig

logger = logging.getLogger(__name__)


try:  # torchao qat API (0.7.0+)
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

    HAVE_TORCHAO_QAT = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_TORCHAO_QAT = False
    # Placeholders to keep type-checkers happy; will never be used without torchao
    Int4WeightOnlyQATQuantizer = object  # type: ignore
    Int8DynActInt4WeightQATQuantizer = object  # type: ignore
    disable_4w_fake_quant = None  # type: ignore
    disable_8da4w_fake_quant = None  # type: ignore
    enable_4w_fake_quant = None  # type: ignore
    enable_8da4w_fake_quant = None  # type: ignore


_QUANTIZER_TO_MODE: dict[type, str] = {}
_DISABLE_FN_BY_MODE: dict[str, Callable] = {}
_ENABLE_FN_BY_MODE: dict[str, Callable] = {}

if HAVE_TORCHAO_QAT:
    _QUANTIZER_TO_MODE[Int8DynActInt4WeightQATQuantizer] = "8da4w-qat"
    _QUANTIZER_TO_MODE[Int4WeightOnlyQATQuantizer] = "4w-qat"

    _DISABLE_FN_BY_MODE["8da4w-qat"] = disable_8da4w_fake_quant
    _ENABLE_FN_BY_MODE["8da4w-qat"] = enable_8da4w_fake_quant

    _DISABLE_FN_BY_MODE["4w-qat"] = disable_4w_fake_quant
    _ENABLE_FN_BY_MODE["4w-qat"] = enable_4w_fake_quant


def get_quantizer_mode(quantizer: object) -> Optional[str]:
    """Return a short mode string for a known torchao QAT quantizer.

    Returns None when the quantizer is unrecognized.
    """

    return _QUANTIZER_TO_MODE.get(type(quantizer)) if HAVE_TORCHAO_QAT else None


def get_disable_fake_quant_fn(mode: str) -> Optional[Callable]:
    """Return the disable fake-quant function for a given quantizer mode."""

    return _DISABLE_FN_BY_MODE.get(mode) if HAVE_TORCHAO_QAT else None


def get_enable_fake_quant_fn(mode: str) -> Optional[Callable]:
    """Return the enable fake-quant function for a given quantizer mode."""

    return _ENABLE_FN_BY_MODE.get(mode) if HAVE_TORCHAO_QAT else None


def prepare_qat_model(model, quantizer) -> tuple[object, Optional[str]]:
    """Apply a torchao QAT quantizer to the given model.

    Returns the (possibly wrapped) model and a mode string if recognized.
    """

    if not HAVE_TORCHAO_QAT:
        raise ImportError("torchao QAT is not available. Please install torchao>=0.7.0")

    if not hasattr(quantizer, "prepare"):
        raise ValueError("Provided quantizer does not implement a prepare(model) method")

    logger.info("Preparing model for QAT using %s", type(quantizer).__name__)
    model = quantizer.prepare(model)
    mode = get_quantizer_mode(quantizer)
    if mode is None:
        logger.warning("Unknown QAT quantizer %s; fake-quant toggling will be unavailable.", type(quantizer).__name__)
    return model, mode


try:
    # Import Nemo's LoRA base to identify patched Linear modules
    from nemo_automodel.components._peft.lora import LinearLoRA
except Exception:  # pragma: no cover - optional dependency
    LinearLoRA = None  # type: ignore


def _attach_qat_to_lora_linear(
    linear: nn.Module,
    activation_qat_config: Optional["FakeQuantizeConfig"],
    weight_qat_config: Optional["FakeQuantizeConfig"],
) -> None:
    """In-place augment a LoRA-patched Linear with QAT fake-quantizers and QAT forward.

    This preserves parameter names and FQNs by swapping the class of the existing module.
    """
    if not HAVE_TORCHAO_QAT:
        raise ImportError("torchao QAT is not available. Please install torchao>=0.7.0")

    try:
        from torchao.quantization.qat.api import FakeQuantizeConfig  # type: ignore
        from torchao.quantization.qat.fake_quantizer import FakeQuantizer  # type: ignore
    except Exception as err:  # pragma: no cover - optional dependency
        raise ImportError("QAT helpers require torchao>=0.7.0") from err

    # Basic validity checks
    if getattr(linear, "quant_state", None) is not None:
        raise ValueError("QLoRA base quantization is not compatible with QAT + LoRA")

    # Validate config types if provided and instantiate fake quantizers
    if activation_qat_config is not None and not isinstance(activation_qat_config, FakeQuantizeConfig):
        raise TypeError("activation_qat_config must be a torchao FakeQuantizeConfig or None")
    if weight_qat_config is not None and not isinstance(weight_qat_config, FakeQuantizeConfig):
        raise TypeError("weight_qat_config must be a torchao FakeQuantizeConfig or None")

    activation_fake_quantizer = FakeQuantizer(activation_qat_config) if activation_qat_config else nn.Identity()
    weight_fake_quantizer = FakeQuantizer(weight_qat_config) if weight_qat_config else nn.Identity()

    # Attach as attributes so they are part of the state dict
    setattr(linear, "activation_fake_quantizer", activation_fake_quantizer)
    setattr(linear, "weight_fake_quantizer", weight_fake_quantizer)

    # Define the QAT forward; keep LoRA path identical and fake-quantize only base path
    def qat_lora_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        _x = self.activation_fake_quantizer(x)
        w = self.weight_fake_quantizer(self.weight)
        out = F.linear(_x, w, None)

        if getattr(self, "dropout_position", "post") == "pre":
            x = F.dropout(x, p=getattr(self, "dropout_p", 0.0), training=self.training)
        lora_res = self.lora_B(self.lora_A(x))
        lora_res = lora_res * getattr(self, "scale", 1.0)
        if getattr(self, "dropout_position", "post") == "post":
            lora_res = F.dropout(lora_res, p=getattr(self, "dropout_p", 0.0), training=self.training)
        return out + lora_res

    # Swap class to inject the new forward while preserving MRO and parameters
    NewCls = type("QAT" + linear.__class__.__name__, (linear.__class__,), {"forward": qat_lora_forward})
    linear.__class__ = NewCls


def swap_lora_linear_with_qat(
    module: nn.Module,
    # TODO: make the types Optional[FakeQuantizeConfig] once torchao 0.7+ is default
    activation_qat_config: Optional["FakeQuantizeConfig"] = None,
    weight_qat_config: Optional["FakeQuantizeConfig"] = None,
) -> None:
    """Swap all Nemo LoRA-patched Linear layers with QAT-enabled versions.

    The resulting computation becomes:
        x -> fake_quantize(W_frozen) @ fake_quantize(x) + BAx

    Args:
        module: Root module to traverse.
        activation_qat_config: torchao FakeQuantizeConfig for input activations.
        weight_qat_config: torchao FakeQuantizeConfig for base weights.
    """
    for name, child in module.named_children():
        # Identify Nemo LoRA-patched linears by isinstance to LinearLoRA or the presence of lora attributes
        is_nemo_lora = False
        if LinearLoRA is not None and isinstance(child, LinearLoRA):
            is_nemo_lora = True
        elif (
            hasattr(child, "lora_A")
            and hasattr(child, "lora_B")
            and isinstance(getattr(child, "weight", None), torch.Tensor)
        ):
            is_nemo_lora = True

        if is_nemo_lora:
            _attach_qat_to_lora_linear(child, activation_qat_config, weight_qat_config)
            continue

        swap_lora_linear_with_qat(child, activation_qat_config, weight_qat_config)


__all__ = [
    "HAVE_TORCHAO_QAT",
    "get_quantizer_mode",
    "get_disable_fake_quant_fn",
    "get_enable_fake_quant_fn",
    "prepare_qat_model",
]

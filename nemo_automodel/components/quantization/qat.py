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
from typing import Callable, Optional

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
    "get_quantizer_mode",
    "get_disable_fake_quant_fn",
    "get_enable_fake_quant_fn",
    "prepare_qat_model",
]

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import importlib as _importlib

from nemo_automodel.components.loss.loss import (
    LOSS_CONFIG_REGISTRY,
    FusedLinearCEConfig,
    KDLossConfig,
    LossConfig,
    LossFromFactoryConfig,
    MaskedCrossEntropyConfig,
    TEParallelCEConfig,
    build_loss_config,
    build_loss_module,
)

__all__ = [
    "LOSS_CONFIG_REGISTRY",
    "FusedLinearCEConfig",
    "KDLossConfig",
    "LossConfig",
    "LossFromFactoryConfig",
    "MaskedCrossEntropyConfig",
    "TEParallelCEConfig",
    "build_loss_config",
    "build_loss_module",
]

_LAZY_ATTRS = {
    "BlockDiffusionCrossEntropyLoss": (".dllm_loss", "BlockDiffusionCrossEntropyLoss"),
    "DFlashDecayLoss": (".dllm_loss", "DFlashDecayLoss"),
    "EmbeddingDistillLoss": (".embedding_distill", "EmbeddingDistillLoss"),
    "EmbeddingMSELoss": (".embedding_distill", "EmbeddingMSELoss"),
    "FusedLinearCrossEntropy": (".linear_ce", "FusedLinearCrossEntropy"),
    "HybridDiffusionLLMLoss": (".dllm_loss", "HybridDiffusionLLMLoss"),
    "IDLMLoss": (".dllm_loss", "IDLMLoss"),
    "InfoNCEDistillLoss": (".infonce", "InfoNCEDistillLoss"),
    "InfoNCELoss": (".infonce", "InfoNCELoss"),
    "IntermediateDistillLoss": (".intermediate_distill", "IntermediateDistillLoss"),
    "KDLoss": (".kd_loss", "KDLoss"),
    "LayerCapture": (".intermediate_distill", "LayerCapture"),
    "MDLMCrossEntropyLoss": (".dllm_loss", "MDLMCrossEntropyLoss"),
    "MTPLossConfig": (".mtp", "MTPLossConfig"),
    "MaskedCrossEntropy": (".masked_ce", "MaskedCrossEntropy"),
    "SCDDLoss": (".dllm_loss", "SCDDLoss"),
    "ScoreDistillLoss": (".embedding_distill", "ScoreDistillLoss"),
    "calculate_loss": (".utils", "calculate_loss"),
    "calculate_mtp_loss": (".mtp", "calculate_mtp_loss"),
    "encoder_ar_loss": (".dllm_loss", "encoder_ar_loss"),
    "get_lm_head_weight": (".utils", "_get_lm_head_weight"),
    "listmle_loss": (".listmle", "listmle_loss"),
    "masked_soft_cross_entropy": (".soft_ce", "masked_soft_cross_entropy"),
    "scdd_schedule": (".dllm_loss", "scdd_schedule"),
}

__all__ += sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> object:
    """Load an exported component symbol on first access."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the component's exported symbols."""
    return sorted(__all__)

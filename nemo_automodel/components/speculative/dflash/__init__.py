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

"""DFlash speculative-decoding training components.

DFlash drafts a whole block of tokens in parallel via MASK-token "denoising"
conditioned on the target model's hidden states, in contrast to EAGLE's
autoregressive single-step drafting. See
``nemo_automodel.components.speculative.dflash.core`` for the training wrapper.
"""

from nemo_automodel.components.speculative.dflash.core import (
    DFlashStepMetrics,
    DFlashTrainerModule,
    NoValidAnchorsError,
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3 import (
    Qwen3DFlashDraftModel,
    build_target_layer_ids,
    extract_context_feature,
)
from nemo_automodel.components.speculative.dflash.registry import (
    DFLASH_DRAFT_REGISTRY,
    DFlashDraftSpec,
    resolve_dflash_draft_spec,
)
from nemo_automodel.components.speculative.dflash.target import DFlashTargetBatch, HFDFlashTargetModel

__all__ = [
    "DFlashTrainerModule",
    "DFlashStepMetrics",
    "NoValidAnchorsError",
    "create_dflash_block_mask",
    "create_dflash_sdpa_mask",
    "Qwen3DFlashDraftModel",
    "build_target_layer_ids",
    "extract_context_feature",
    "HFDFlashTargetModel",
    "DFlashTargetBatch",
    "DFlashDraftSpec",
    "DFLASH_DRAFT_REGISTRY",
    "resolve_dflash_draft_spec",
]

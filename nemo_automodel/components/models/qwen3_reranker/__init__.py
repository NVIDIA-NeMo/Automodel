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

"""Qwen3 reranker model for cross-encoder reranking tasks."""

from nemo_automodel.shared.import_utils import safe_import_from

# Guard model exports so the collator remains available for data-only use when the
# optional Qwen3 modelling stack cannot be imported.
_MSG = (
    "Qwen3RerankerForCausalReranking requires the Qwen3 modelling stack; install a "
    "transformers version that provides it."
)
_, Qwen3RerankerConfig = safe_import_from(
    "nemo_automodel.components.models.qwen3_reranker.model", "Qwen3RerankerConfig", msg=_MSG
)
_, Qwen3RerankerForCausalReranking = safe_import_from(
    "nemo_automodel.components.models.qwen3_reranker.model",
    "Qwen3RerankerForCausalReranking",
    msg=_MSG,
)
# The collator depends only on tokenizers and DataCollatorWithPadding, not on the Qwen3
# modelling stack, so it stays importable for data-only use even when the model does not.
_, Qwen3ContextAwareRerankerCollator = safe_import_from(
    "nemo_automodel.components.models.qwen3_reranker.collator",
    "Qwen3ContextAwareRerankerCollator",
    msg="Qwen3ContextAwareRerankerCollator requires transformers' DataCollatorWithPadding.",
)

__all__ = [
    "Qwen3ContextAwareRerankerCollator",
    "Qwen3RerankerForCausalReranking",
    "Qwen3RerankerConfig",
]

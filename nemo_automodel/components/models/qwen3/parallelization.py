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

"""Parallelization contract for the transformers Qwen3 heads NeMo AutoModel does not re-implement."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle
from torch.distributed.tensor.placement_types import Replicate

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.models.qwen2.parallelization import QWEN_SEQUENCE_PARALLEL_PLAN, QWEN_TP_PLAN

# The Qwen text-backbone plan with the ``lm_head`` rule swapped for a replicated ``score`` head:
# ``Qwen3ForSequenceClassification`` pools over the *sequence* dimension in Python, so the classifier logits must
# be replicated (full ``num_labels``) for correct pooling/loss.
QWEN3_SEQUENCE_CLASSIFICATION_TP_PLAN: dict[str, ParallelStyle] = {
    **{fqn: style for fqn, style in QWEN_TP_PLAN.items() if fqn != "lm_head"},
    "score": ColwiseParallel(output_layouts=Replicate()),
}
QWEN3_SEQUENCE_CLASSIFICATION_SEQUENCE_PARALLEL_PLAN: dict[str, ParallelStyle] = {
    fqn: style for fqn, style in QWEN_SEQUENCE_PARALLEL_PLAN.items() if fqn != "lm_head"
}


class Qwen3ForSequenceClassification:
    """Contract for the transformers ``Qwen3ForSequenceClassification``; bound by the loader onto its wrapper."""

    parallel_spec: ParallelSpec = ParallelSpec(
        tp_plan=QWEN3_SEQUENCE_CLASSIFICATION_TP_PLAN,
        sequence_parallel_plan=QWEN3_SEQUENCE_CLASSIFICATION_SEQUENCE_PARALLEL_PLAN,
    )

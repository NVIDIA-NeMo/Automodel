# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Model-owned parallelization for qwen3."""

from __future__ import annotations

from torch import nn
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle
from torch.distributed.tensor.placement_types import Replicate

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.qwen2.parallelization import _parallelize_qwen


def _parallelize_qwen_classification(
    model: nn.Module | None,
    sequence_parallel: bool = False,
) -> dict[str, ParallelStyle]:
    plan = _parallelize_qwen(model, sequence_parallel)
    assert not hasattr(model, "lm_head"), "Expected model not to have lm_head"
    del plan["lm_head"]
    assert hasattr(model, "score"), "Expected model to have score"
    # `Qwen3ForSequenceClassification` pools over the *sequence* dimension in Python.
    # Ensure the classifier logits are replicated (full num_labels) for correct pooling/loss.
    plan["score"] = ColwiseParallel(output_layouts=Replicate())
    return plan


PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_qwen_classification)

CAUSAL_PARALLELIZER = ModelParallelizer(tp_plan=_parallelize_qwen)

__all__ = ["PARALLELIZER"]

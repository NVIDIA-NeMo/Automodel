# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU coverage for Gemma4's model-owned tensor-parallel plan."""

from types import SimpleNamespace

import torch.nn as nn

from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce
from nemo_automodel.components.models.gemma4_moe.parallelization import _gemma4_tp_plan


def test_gemma4_tp_plan_sums_head_local_qk_norm_gradients() -> None:
    """Gemma4 Q/K norm replicas sum contributions from head-sharded Q/K."""
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size_per_layer_input=0)))

    plan = _gemma4_tp_plan(model)

    prefix = "model.language_model.layers.*.self_attn"
    for name in ("q_norm", "k_norm"):
        style = plan[f"{prefix}.{name}"]
        assert isinstance(style, ReplicatedWithGradAllReduce)
        norm = nn.LayerNorm(4)
        style._apply(norm, None)
        assert norm._nemo_tp_replica_grad_reduction == "sum"

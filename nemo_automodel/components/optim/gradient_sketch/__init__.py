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

"""Gradient sketching: rank-r projected gradients with a compact row-wise optimizer.

Implements the optimizer side of LoGRA (Zhang et al., "Scaling LLM reinforcement
learning with low-rank gradient sketches", arXiv:2610.06647). Selected linear
weights never form a dense gradient: backward accumulates ``S = Hᵀ(XAᵀ)`` into a
shared buffer, RowAdam keeps one second moment per output row, and the update
``ΔW = -lr · D · A`` is merged into the full weight each step.
"""

from nemo_automodel.components.optim.gradient_sketch.row_adam import row_adam_direction
from nemo_automodel.components.optim.gradient_sketch.sketch import (
    SketchSettings,
    SketchState,
    fill_projection_,
    install_sketches,
    local_weight_rows,
    make_projection,
    projection_seed,
    select_sketch_targets,
    sketch_buffer,
)

__all__ = [
    "SketchSettings",
    "SketchState",
    "fill_projection_",
    "install_sketches",
    "local_weight_rows",
    "make_projection",
    "projection_seed",
    "row_adam_direction",
    "select_sketch_targets",
    "sketch_buffer",
]

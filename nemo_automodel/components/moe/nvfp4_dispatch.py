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

"""Routed-expert dispatches too large for one Transformer Engine NVFP4 quantization.

Transformer Engine quantizes the rows of all experts of a ``GroupedLinear`` call together, and its NVFP4
kernels address that tensor with 32-bit element offsets (Transformer Engine 2.19 to 2.21). Beyond 2**32
elements with one GEMM per expert, and beyond 2**31 on the grouped-tensor path, they return wrong values
without an error. A dispatch above the first limit is quantized and multiplied in consecutive chunks of rows;
the grouped-tensor path is not chunked and stops the run. FP8 and MXFP8 quantization and unquantized GEMMs
have no such limit.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from transformer_engine.common.recipe import Recipe
    from transformer_engine.pytorch import GroupedLinear

# Elements (rows x columns) of one NVFP4 quantization of routed-expert rows; 1/16 below the failing offsets.
NVFP4_QUANTIZE_MAX_ELEMENTS = 2**32 - 2**28
NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS = 2**31 - 2**27


def plan_dispatch_chunks(rows_per_expert: list[int], *, max_rows: int, align: int) -> list[list[int]]:
    """Cut the rows of a rank's experts into consecutive chunks of at most ``max_rows`` rows.

    The dispatched rows are grouped by expert, so a chunk is a row range of that tensor. A chunk ends before
    the first expert whose rows no longer fit into it, as long as that expert fits a chunk of its own; only an
    expert with more rows than one chunk holds is cut, and then at a multiple of ``align`` rows from its start.

    Args:
        rows_per_expert: Rows of each local expert, in expert order; multiples of ``align``.
        max_rows: Most rows one chunk may hold.
        align: Row multiple that the quantization of one expert's rows needs.

    Returns:
        For every chunk the rows it holds of each expert, zero for the experts outside it. The chunks cover
        the rows of all experts in order; a dispatch without rows is one empty chunk.

    Raises:
        ValueError: If ``max_rows`` is smaller than ``align``.
    """
    capacity = max_rows // align * align
    if capacity <= 0:
        raise ValueError(f"A chunk of at most {max_rows} rows cannot hold a block of {align} rows.")
    chunks: list[list[int]] = [[0] * len(rows_per_expert)]
    used = 0
    for expert, rows in enumerate(rows_per_expert):
        if 0 < used and capacity - used < rows <= capacity:
            chunks.append([0] * len(rows_per_expert))
            used = 0
        while rows > 0:
            if used == capacity:
                chunks.append([0] * len(rows_per_expert))
                used = 0
            taken = min(rows, capacity - used)
            chunks[-1][expert] = taken
            used += taken
            rows -= taken
    return chunks


def nvfp4_dispatch_chunks(
    recipe: Recipe | None, rows_per_expert: list[int], *, width: int, align: int, grouped: bool
) -> list[list[int]] | None:
    """Rows per expert of the calls an NVFP4 dispatch needs; ``None`` for a dispatch that fits one call.

    Args:
        recipe: Active Transformer Engine recipe, ``None`` outside a quantized autocast. A custom recipe can
            hold NVFP4 quantizers and is limited like the NVFP4 recipe.
        rows_per_expert: Rows of each local expert as passed to the GEMMs (padded under a recipe).
        width: Columns of the widest tensor Transformer Engine quantizes for these rows.
        align: Row multiple the recipe pads every expert to.
        grouped: Whether the experts run grouped-tensor GEMMs (``te_grouped_tensor``).

    Returns:
        ``None`` when one call per projection is within the limit, else the rows per expert of every chunk.

    Raises:
        ValueError: If ``grouped`` and the dispatch is beyond the limit of the grouped quantization.
    """
    if recipe is None or not (recipe.nvfp4() or recipe.custom()):
        return None
    elements = sum(rows_per_expert) * width
    if grouped and elements > NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS:
        raise ValueError(
            f"An expert dispatch of {sum(rows_per_expert)} rows holds {elements} elements, more than the "
            f"{NVFP4_GROUPED_QUANTIZE_MAX_ELEMENTS} that Transformer Engine's grouped NVFP4 quantization handles "
            "correctly. Set backend.te_grouped_tensor: false; such dispatches then run one GEMM per expert, in "
            "chunks of rows where needed."
        )
    if elements <= NVFP4_QUANTIZE_MAX_ELEMENTS:
        return None
    return plan_dispatch_chunks(rows_per_expert, max_rows=NVFP4_QUANTIZE_MAX_ELEMENTS // width, align=align)


def expert_mlp_in_chunks(
    gate_up_linear: GroupedLinear,
    down_linear: GroupedLinear,
    activation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    chunks: list[list[int]],
    cast_weights: AbstractContextManager[None],
) -> torch.Tensor:
    """Run the expert MLP once per chunk of rows, so that every quantization covers one chunk.

    Every call goes to the same two ``GroupedLinear`` modules with the rows of its chunk and the rows per
    expert inside that chunk (zero for the other experts), so an expert without rows in a call gets a zero
    gradient from it. The calls after the first reuse the weights the first one quantized.

    Args:
        gate_up_linear: Up projection of the experts, built with one GEMM per expert.
        down_linear: Down projection of the experts, built with one GEMM per expert.
        activation: Expert activation, called with the up-projected rows and their routing weights.
        hidden_states: Dispatched rows of shape [rows, expert_dim], grouped by expert.
        probs: Routing weight of every row, shape [rows, 1].
        chunks: Rows per expert of every chunk, from :func:`nvfp4_dispatch_chunks`.
        cast_weights: Context under which the GEMMs run (casts FP32 parameters to the compute dtype).

    Returns:
        Expert outputs of shape [rows, expert_dim].
    """
    from nemo_automodel.components.models.common.utils import get_is_first_microbatch

    row_counts = [sum(chunk) for chunk in chunks]
    # Transformer Engine stores the quantized weights under a micro-batch flag of True or False, and a later
    # call then passes False to reuse them; under None it stores nothing and the later call passes None too.
    later_call = None if get_is_first_microbatch() is None else False
    outputs: list[torch.Tensor] = []
    for index, (chunk, hidden, chunk_probs) in enumerate(
        zip(chunks, hidden_states.split(row_counts), probs.split(row_counts), strict=True)
    ):
        weight_cache = None if index == 0 else later_call
        with cast_weights:
            hidden = gate_up_linear(hidden, chunk, is_first_microbatch=weight_cache)
        hidden = activation(hidden, chunk_probs)
        with cast_weights:
            outputs.append(down_linear(hidden, chunk, is_first_microbatch=weight_cache))
    return torch.cat(outputs)

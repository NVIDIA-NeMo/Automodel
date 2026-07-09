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

"""Linear graph-shaping helpers shared by dense and PEFT tensor parallelism.

Inductor's async tensor-parallel pass (``torch._inductor.config._micro_pipeline_tp``)
fuses collectives with matmuls by pattern-matching the reshape-mm-reshape graph
that ``F.linear`` produces for 3-D input.  The compile-safe ``torch.bmm`` path
used by ``TPLinear``/``LinearLoRA`` never matches, so async-TP fusion silently
fails to fire.  These helpers detect async-TP tracing and emit the native
linear graph in that mode only.
"""

import torch
import torch.nn.functional as F


def _is_async_tp_linear_enabled() -> bool:
    """Return whether Dynamo is tracing with Inductor's async-TP pass enabled.

    True only when both conditions hold: the caller is being traced by
    torch.compile and ``torch._inductor.config._micro_pipeline_tp`` is set
    (see ``enable_async_tensor_parallel`` in parallelizer.py).  Always False
    in eager mode.
    """
    if not torch.compiler.is_compiling():
        return False
    inductor = getattr(torch, "_inductor", None)
    config = getattr(inductor, "config", None)
    return bool(getattr(config, "_micro_pipeline_tp", False))


def _async_tp_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Emit the native linear graph recognized by async-TP fusion.

    Args:
        x: Input activations of shape ``[..., in_features]``; any number of
            leading dimensions is accepted (typically ``[B, S, in_features]``
            with ``B`` = batch, ``S`` = sequence).  May be a DTensor with
            tensor-parallel placements.
        weight: Weight of shape ``[out_features, in_features]``; may be a
            DTensor sharded for colwise (``Shard(0)``) or rowwise
            (``Shard(1)``) tensor parallelism.
        bias: Optional bias of shape ``[out_features]``.

    Returns:
        Output of shape ``[..., out_features]`` with the same leading
        dimensions as ``x``.
    """
    # Keep bias outside F.linear so the collective sees the matmul result as
    # its direct producer/consumer.  PyTorch lowers 3-D F.linear to the
    # reshape-mm-reshape pattern consumed by micro_pipeline_tp.
    output = F.linear(x, weight)
    return output + bias if bias is not None else output

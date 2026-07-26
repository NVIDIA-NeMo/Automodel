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

"""Untied Ulysses ("UPipe") memory-efficient context-parallel attention.

The kernels live here; the model-side wiring (weight hand-off, head
un-permutation, batch sharding) lives with the model that opts in via
``backend.attn = "upipe"``.
"""

from nemo_automodel.components.distributed.context_parallel.upipe.layout import (
    invert_permutation,
    upipe_head_permutation,
)
from nemo_automodel.components.distributed.context_parallel.upipe.validation import (
    UPIPE_MAX_HEAD_DIM,
    UPIPE_MIN_ROTARY_HEAD_DIM,
    validate_upipe_attention,
    validate_upipe_runtime,
)

__all__ = [
    "UPIPE_MAX_HEAD_DIM",
    "UPIPE_MIN_ROTARY_HEAD_DIM",
    "invert_permutation",
    "rope_tables_from_position_embeddings",
    "upipe_attn_gqa",
    "upipe_head_permutation",
    "validate_upipe_attention",
    "validate_upipe_runtime",
]


def __getattr__(name: str):
    """Defer Triton/FlashAttention imports so CPU-only hosts can import this package.

    ``fused_attn`` and ``rotary`` pull in Triton and FlashAttention at import
    time; validation and the head layout must stay importable without them.
    """
    if name == "upipe_attn_gqa":
        from nemo_automodel.components.distributed.context_parallel.upipe.fused_attn import upipe_attn_gqa

        return upipe_attn_gqa
    if name == "rope_tables_from_position_embeddings":
        from nemo_automodel.components.distributed.context_parallel.upipe.rotary import (
            rope_tables_from_position_embeddings,
        )

        return rope_tables_from_position_embeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

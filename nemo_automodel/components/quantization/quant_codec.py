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

"""Quantized expert-weight storage codec interface.

This is a coordination seam, not an implementation. ``MXFP4ExpertStorageMixin``
(in ``quantized_experts.py``) currently inlines the mxfp4 logic; the codec is
extracted and made the single source of truth when the int4 (GLM) variant lands.

A codec describes ONE storage format (mxfp4 e2m1+e8m0, compressed-tensors int4,
...). Two load paths use it:

- ``pack`` quantizes a floating-point weight to the packed tensors. This is the
  OFFLINE path — e.g. a grid-analysis tool that derives an int4 artifact from a
  bf16 checkpoint when no native quantized checkpoint exists (the GLM-5.1 case:
  ``zai-org/GLM-5.1`` ships plain bf16, no scales to read, so the grid must be
  recovered from values once, offline, then emitted as a packed artifact).
- ``from_checkpoint`` ingests already-quantized checkpoint tensors WITHOUT a
  bf16 round-trip. This is the TRAINING load path: combined with per-chunk
  registration it never materializes bf16 experts, which is what holds at
  GLM-744B scale. For mxfp4/DeepSeek-V4 this is the identity on the checkpoint's
  int8 + e8m0 tensors.
"""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class QuantExpertCodec(Protocol):
    """Storage format for frozen quantized expert weights.

    ``param_names`` is the source of truth for the per-weight storage tensors:
    a base projection registered under FQN ``W`` is stored as the parameters
    ``W + suffix`` for each suffix in ``param_names`` (e.g. ``("_packed",
    "_scales")``). ``pack`` / ``from_checkpoint`` return tensors positionally in
    that order, and ``unpack`` consumes them in that order. The set is variadic
    so a format needing extra tensors (zero-points, second-level scales) adds a
    suffix rather than a privileged positional argument — there is no special
    ``zeros`` slot.

    ``gemm_min_rank_align`` lives here because the GEMM stride constraint differs
    by entry point: ``torch._grouped_mm`` requires 16-byte-aligned strides (LoRA
    rank >= 8 in bf16), while ``grouped_gemm.ops.gmm`` has its own rule. The
    quantized-expert wrapper reads this off the codec/backend seam rather than
    assuming the two paths agree.
    """

    # Suffixes of the storage parameters, in pack/unpack order. Source of truth.
    param_names: tuple[str, ...]
    # Quantization block size along the contraction dim (mxfp4: 32).
    block_size: int
    # Minimum rank/stride alignment required by the target grouped-GEMM entry point.
    gemm_min_rank_align: int

    def pack(self, weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Quantize ``weight`` (checkpoint orientation ``[E, out, in]``) to packed tensors.

        Offline path. ``unpack(*pack(w))`` must round-trip ``w`` value-exactly for
        weights already on the format's grid.
        """
        ...

    def unpack(self, *packed: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize packed tensors back to a dense weight ``[E, out, in]`` in ``out_dtype``."""
        ...

    def from_checkpoint(self, *ckpt_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Map already-quantized checkpoint tensors to storage params, no bf16 round-trip.

        Training load path. Returns tensors in ``param_names`` order.
        """
        ...

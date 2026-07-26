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

"""Head layout bookkeeping for UPipe attention.

The fused op walks heads in an order dictated by how KV heads shard across the
Ulysses group, which under GQA is not the checkpoint's order. This module owns
that mapping. It deliberately imports nothing beyond the standard library so it
stays usable, and testable, on hosts without Triton or FlashAttention.

Why a permutation exists at all
-------------------------------
Stage ``s`` of the fused op projects query heads from ``wq`` row-block ``s``,
and KV heads from ``wk`` row-block ``s // gqa_ratio``. After the all-to-all,
rank ``r`` holds query slot ``r`` of the first and KV slot ``r`` of the second,
so it computes

    query row-block ``s * U + r``   against   KV head ``(s // g) * U + r``

writing the result to output slot ``s * U + r`` (``U`` = Ulysses degree,
``g`` = GQA ratio).

The model requires query head ``h`` to attend KV head ``h // g``. Solving
``h // g == (s // g) * U + r`` for the ``g`` stages that share a KV block
(``s = kv_idx * g + j``) gives

    h(s, r) = ((s // g) * U + r) * g + (s % g)

which is a permutation of ``0 .. n_heads-1`` but is *not* the identity
``s * U + r`` whenever ``g > 1``. So two corrections are needed, and both use
this same permutation:

1. Feed the op ``wq`` with its head blocks reordered, so row-block ``s * U + r``
   really does hold head ``h(s, r)``.
2. Un-permute the output before ``o_proj``, so slot ``s * U + r`` lands back at
   head ``h(s, r)``.

Doing only (2) -- as the reference implementation does -- yields a model whose
query projections are permuted with respect to its KV projections. That is
harmless when pretraining from a random init, since any head assignment is
equally good, but it silently corrupts a pretrained checkpoint. AutoModel
finetunes checkpoints, so it must do both.
"""

from __future__ import annotations


def upipe_head_permutation(n_heads: int, n_kv_heads: int, ulysses_degree: int) -> list[int]:
    """Map each of the fused op's head slots to the logical head it must carry.

    Args:
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        ulysses_degree: Size of the Ulysses process group.

    Returns:
        A list ``perm`` of length ``n_heads`` where ``perm[i]`` is the logical
        head index belonging in slot ``i``. Use it to gather ``wq`` head blocks
        on the way in, and its inverse to scatter the output on the way out.

    Raises:
        ValueError: If the head counts are not divisible as UPipe requires.
    """
    if ulysses_degree <= 0 or n_heads % ulysses_degree != 0:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by ulysses_degree ({ulysses_degree})")
    if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
    if n_kv_heads % ulysses_degree != 0:
        raise ValueError(f"n_kv_heads ({n_kv_heads}) must be divisible by ulysses_degree ({ulysses_degree})")

    gqa_ratio = n_heads // n_kv_heads
    pipe_degree = n_heads // ulysses_degree

    perm = [0] * n_heads
    for stage in range(pipe_degree):
        for rank in range(ulysses_degree):
            slot = stage * ulysses_degree + rank
            perm[slot] = ((stage // gqa_ratio) * ulysses_degree + rank) * gqa_ratio + (stage % gqa_ratio)
    return perm


def invert_permutation(perm: list[int]) -> list[int]:
    """Return the inverse of ``perm``, so that ``inverse[perm[i]] == i``."""
    inverse = [0] * len(perm)
    for slot, logical in enumerate(perm):
        inverse[logical] = slot
    return inverse

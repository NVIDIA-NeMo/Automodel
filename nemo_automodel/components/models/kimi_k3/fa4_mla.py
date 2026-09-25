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

"""FlashAttention 4 document-causal attention for Kimi K3 MLA under context parallelism.

Drop-in replacement for :func:`~nemo_automodel.components.models.kimi_k3.cp.document_causal_flex_attention`,
selected with ``KimiK3TextConfig.mla_cp_attn_backend = "fa4"``. The document / causal / padding predicate is
compiled into FA4 as a CuTe ``mask_mod`` and FA4 skips fully masked tiles through block-sparse metadata, so
MLA's native 192/128 QK/V head dims run without FlexAttention's power-of-two padding.

Optional FA4/CuTe dependencies are loaded on the first call.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from nemo_automodel.shared.import_utils import safe_import

_PAD_DOC_ID = 0

# FA4 SM90 tile shapes: forward (query, key) tiles, and the backward kernel's (query, key) tiles.
_FWD_BLOCK = (128, 128)
_BWD_BLOCK = (64, 96)


@functools.cache
def _load_fa4() -> tuple[Callable, Callable, Any, Any, Any]:
    """Import FA4 entry points once; raise a clear error if they are unavailable."""
    dependencies = [
        safe_import(name)
        for name in (
            "flash_attn.cute.interface",
            "flash_attn.cute.compute_block_sparsity",
            "cutlass",
            "cutlass.cute",
            "flash_attn.cute.utils",
        )
    ]
    if not all(available for available, _ in dependencies):
        raise ImportError(
            "mla_cp_attn_backend='fa4' requires FlashAttention 4 (flash_attn.cute) and nvidia-cutlass-dsl."
        )
    interface, sparsity, cutlass, cute, utils = [module for _, module in dependencies]
    return interface.flash_attn_func, sparsity.compute_block_sparsity, cutlass, cute, utils


@functools.lru_cache(None)
def _document_causal_mask_mod(q_length: int, k_length: int, q_global_start: int, transposed: bool = False):
    """Build the CuTe ``mask_mod`` matching ``cp._document_causal_block_mask``.

    Args:
        q_length: Local query length.
        k_length: Global key length.
        q_global_start: Global sequence offset of the first local query.
        transposed: Evaluate with swapped (key, query) coordinates, used to build the
            backward kernel's key-major block-sparse metadata.

    Returns:
        A ``cute.jit`` callback reading query / key document ids from ``aux_tensors``.
    """
    _, _, cutlass, cute, utils = _load_fa4()

    @cute.jit
    def mask_mod(batch, head, query_idx, key_idx, seqlen_info, aux_tensors):
        if cutlass.const_expr(transposed):
            qi_raw, ki_raw = key_idx[0], query_idx[0]
        else:
            qi_raw, ki_raw = query_idx[0], key_idx[0]
        qi = cutlass.min(qi_raw, q_length - 1)
        ki = cutlass.min(ki_raw, k_length - 1)
        q_doc = aux_tensors[0][batch[0], qi]
        kv_doc = aux_tensors[1][batch[0], ki]
        allowed = (ki_raw <= qi_raw + q_global_start) & (q_doc == kv_doc) & (kv_doc > _PAD_DOC_ID)
        # A padding query reads position 0 so its softmax stays finite; its output is zeroed afterwards.
        padding_query = (q_doc <= _PAD_DOC_ID) & (ki_raw == 0)
        allowed = ((q_doc > _PAD_DOC_ID) & allowed) | padding_query
        in_bounds = (qi_raw < q_length) & (ki_raw < k_length)
        return utils.scalar_to_ssa(allowed & in_bounds, cutlass.Boolean)

    return mask_mod


_METADATA_CACHE: dict[tuple, Any] = {}
_METADATA_GENERATION: list[Any] = [None, None]


def _document_causal_metadata(q_doc_ids: torch.Tensor, kv_doc_ids: torch.Tensor, *, q_global_start: int):
    """Build (and cache for the step) the FA4 mask callback and block-sparse metadata."""
    _, compute_block_sparsity, _, _, _ = _load_fa4()

    pointer = kv_doc_ids.data_ptr()
    if pointer != _METADATA_GENERATION[0]:
        _METADATA_CACHE.clear()
        _METADATA_GENERATION[0] = pointer
        # Hold the tensor so the allocator cannot recycle the address mid-step.
        _METADATA_GENERATION[1] = kv_doc_ids
    batch_size, q_length = q_doc_ids.shape
    k_length = kv_doc_ids.shape[1]
    key = (int(q_global_start), batch_size, q_length, k_length, q_doc_ids.device.type, q_doc_ids.device.index)
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    mask_mod = _document_causal_mask_mod(q_length, k_length, int(q_global_start))
    transposed_mask_mod = _document_causal_mask_mod(q_length, k_length, int(q_global_start), True)
    aux_tensors = [q_doc_ids.contiguous(), kv_doc_ids.contiguous()]
    forward = compute_block_sparsity(
        *_FWD_BLOCK,
        batch_size,
        1,
        q_length,
        k_length,
        mask_mod,
        aux_tensors,
        q_doc_ids.device,
        compute_full_blocks=True,
        use_fast_sampling=False,
    )
    # The backward metadata is indexed by key-block rows and query-block columns, so it is built
    # over the transposed problem; block_size still names the logical (query, key) tile.
    backward = compute_block_sparsity(
        _BWD_BLOCK[1],
        _BWD_BLOCK[0],
        batch_size,
        1,
        k_length,
        q_length,
        transposed_mask_mod,
        aux_tensors,
        q_doc_ids.device,
        compute_full_blocks=True,
        use_fast_sampling=False,
    )
    backward = backward._replace(block_size=_BWD_BLOCK)
    cached = (mask_mod, aux_tensors, forward, backward)
    if len(_METADATA_CACHE) >= 64:
        _METADATA_CACHE.pop(next(iter(_METADATA_CACHE)))
    _METADATA_CACHE[key] = cached
    return cached


def document_causal_fa4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    q_doc_ids: torch.Tensor,
    kv_doc_ids: torch.Tensor,
    q_global_start: int,
    scale: float,
) -> torch.Tensor:
    """Run causal, per-document attention of local queries against global keys with FA4.

    Args:
        query: Tensor of shape [batch, heads, query_sequence, qk_head_dim].
        key: Tensor of shape [batch, heads, key_sequence, qk_head_dim].
        value: Tensor of shape [batch, heads, key_sequence, v_head_dim].
        q_doc_ids: Tensor of shape [batch, query_sequence] with 1-based document ids.
        kv_doc_ids: Tensor of shape [batch, key_sequence] with 1-based document ids.
        q_global_start: Global sequence offset of the first query token.
        scale: Softmax scale applied to the query-key product.

    Returns:
        Tensor of shape [batch, heads, query_sequence, v_head_dim].
    """
    flash_attn_func = _load_fa4()[0]
    mask_mod, aux_tensors, forward, backward = _document_causal_metadata(
        q_doc_ids, kv_doc_ids, q_global_start=q_global_start
    )
    output, _ = flash_attn_func(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        softmax_scale=scale,
        mask_mod=mask_mod,
        aux_tensors=aux_tensors,
        block_sparse_tensors=forward,
        block_sparse_tensors_bwd=backward,
        pack_gqa=False,
    )
    return output.transpose(1, 2).masked_fill((q_doc_ids <= _PAD_DOC_ID)[:, None, :, None], 0)

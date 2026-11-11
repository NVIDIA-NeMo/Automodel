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

"""Attention masks of the chunk-causal window.

Every token of a sequence carries three integer tags: its ``kind`` (global, noised video,
clean video), its ``chunk`` index and its latent ``frame`` index. A mask rule is a pure
function of the tags of the query and the key, so one rule serves both the flex
``BlockMask`` (CUDA) and a dense boolean mask (CPU and SDPA).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

KIND_GLOBAL = 0
KIND_NOISED = 1
KIND_CLEAN = 2

MaskRule = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class TokenTags:
    """Per-token tags of one sequence, all int32 tensors of shape [tokens] on one device."""

    kind: Tensor
    chunk: Tensor
    frame: Tensor

    @property
    def length(self) -> int:
        return int(self.kind.shape[0])


@dataclass(frozen=True)
class Slot:
    """A run of tokens of one kind: ``frames`` latent frames of ``tokens_per_frame`` tokens each.

    A global slot has ``frames == 1`` and ``tokens_per_frame`` tokens.
    """

    kind: int
    frames: int
    tokens_per_frame: int


# A cache key: the slot layout, sizes, device string and flags that determine one mask.
MaskKey = tuple[int | str | bool | tuple[Slot, ...], ...]


def window_tags(slots: tuple[Slot, ...], chunk_size: int, device: torch.device) -> TokenTags:
    """Tags of a window sequence laid out as ``slots`` in order.

    Frame ``f`` of a video slot belongs to chunk ``f // chunk_size``. Global tokens get
    chunk and frame ``-1``.

    Args:
        slots: The runs of the sequence in order.
        chunk_size: Latent frames per chunk.
        device: Device of the tag tensors.

    Returns:
        :class:`TokenTags` of the whole sequence.
    """
    kinds: list[Tensor] = []
    chunks: list[Tensor] = []
    frames: list[Tensor] = []
    for slot in slots:
        n = slot.frames * slot.tokens_per_frame
        kinds.append(torch.full((n,), slot.kind, dtype=torch.int32))
        if slot.kind == KIND_GLOBAL:
            chunks.append(torch.full((n,), -1, dtype=torch.int32))
            frames.append(torch.full((n,), -1, dtype=torch.int32))
        else:
            frame = torch.arange(slot.frames, dtype=torch.int32).repeat_interleave(slot.tokens_per_frame)
            frames.append(frame)
            chunks.append(torch.div(frame, chunk_size, rounding_mode="floor"))
    return TokenTags(
        kind=torch.cat(kinds).to(device),
        chunk=torch.cat(chunks).to(device),
        frame=torch.cat(frames).to(device),
    )


def chunk_causal_rule(tags: TokenTags) -> MaskRule:
    """The teacher-forcing rule of the trunk towers over one sequence.

    * every token attends every global token; a global token attends only global tokens;
    * a clean token attends the clean tokens of its own and earlier chunks;
    * a noised token attends the noised tokens of its own chunk and the clean tokens of
      earlier chunks;
    * a clean token never attends a noised token.
    """

    def rule(q: Tensor, k: Tensor) -> Tensor:
        q_kind, k_kind = tags.kind[q], tags.kind[k]
        q_chunk, k_chunk = tags.chunk[q], tags.chunk[k]
        k_global = k_kind == KIND_GLOBAL
        q_video = q_kind != KIND_GLOBAL
        q_clean, k_clean = q_kind == KIND_CLEAN, k_kind == KIND_CLEAN
        q_noised, k_noised = q_kind == KIND_NOISED, k_kind == KIND_NOISED
        clean_clean = q_clean & k_clean & (k_chunk <= q_chunk)
        noised_clean = q_noised & k_clean & (k_chunk < q_chunk)
        noised_noised = q_noised & k_noised & (k_chunk == q_chunk)
        return k_global | (q_video & (clean_clean | noised_clean | noised_noised))

    return rule


def expert_cross_rule(q_tags: TokenTags, kv_tags: TokenTags, window: int) -> MaskRule:
    """Rule of the action expert's cross attention.

    A query of chunk ``c`` (first frame ``c * chunk_size``, stored in ``q_tags.frame``)
    attends every global key, the clean frames strictly before its chunk and at most
    ``window`` frames before it, and the noised frames of its own chunk.
    """

    def rule(q: Tensor, k: Tensor) -> Tensor:
        k_kind = kv_tags.kind[k]
        k_frame, k_chunk = kv_tags.frame[k], kv_tags.chunk[k]
        q_start, q_chunk = q_tags.frame[q], q_tags.chunk[q]
        clean = (k_kind == KIND_CLEAN) & (k_frame < q_start) & (k_frame >= q_start - window)
        noised = (k_kind == KIND_NOISED) & (k_chunk == q_chunk)
        return (k_kind == KIND_GLOBAL) | clean | noised

    return rule


def expert_self_rule(q_tags: TokenTags) -> MaskRule:
    """Rule of the action expert's self attention: one block per chunk."""

    def rule(q: Tensor, k: Tensor) -> Tensor:
        return q_tags.chunk[q] == q_tags.chunk[k]

    return rule


def dense_mask(rule: MaskRule, q_len: int, kv_len: int, device: torch.device) -> Tensor:
    """Evaluate ``rule`` on the full grid.

    Returns:
        Boolean tensor of shape [q_len, kv_len], true where attention is allowed.
    """
    q = torch.arange(q_len, device=device)[:, None]
    k = torch.arange(kv_len, device=device)[None, :]
    return rule(q, k)


def block_mask(rule: MaskRule, q_len: int, kv_len: int, device: torch.device, block_size: int) -> BlockMask:
    """Build a flex ``BlockMask`` over [q_len, kv_len] tiles of ``block_size`` from ``rule``."""

    def mask_mod(b: Tensor, h: Tensor, q: Tensor, k: Tensor) -> Tensor:
        return rule(q, k)

    return create_block_mask(
        mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device, BLOCK_SIZE=block_size, _compile=True
    )


class MaskCache:
    """Small LRU cache of built masks keyed by the caller's layout key."""

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[MaskKey, BlockMask | Tensor] = OrderedDict()

    def get(self, key: MaskKey, build: Callable[[], BlockMask | Tensor]) -> BlockMask | Tensor:
        mask = self._entries.pop(key, None)
        if mask is None:
            mask = build()
        self._entries[key] = mask
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return mask

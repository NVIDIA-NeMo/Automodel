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

"""Resolved tokenizer IDs for raw-text training."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenIds:
    """Resolved existing vocabulary IDs; no tokens are added to the model."""

    bos: int | None
    eos: int | None
    pad: int


def resolve_token_ids(tokenizer: PreTrainedTokenizerBase, *, pad_token_id: int | None) -> TokenIds:
    """Resolve PAD/BOS/EOS using the effective AutoModel tokenizer.

    Explicit padding may not contradict a declared tokenizer padding ID.
    The fallback order follows TorchTitan: named pad, reserved special, UNK,
    then EOS. Padding is always masked by position, even when PAD equals EOS.
    """
    vocab = tokenizer.get_vocab()
    valid_ids = set(vocab.values())
    bos, eos, declared = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
    for name, value in (("BOS", bos), ("EOS", eos), ("PAD", declared), ("pad_token_id", pad_token_id)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value not in valid_ids):
            raise ValueError(f"{name}={value} is not an existing tokenizer vocabulary ID")
    bos = int(bos) if bos is not None else None
    eos = int(eos) if eos is not None else None
    declared = int(declared) if declared is not None else None
    if pad_token_id is not None:
        if declared is not None and pad_token_id != declared:
            raise ValueError("pad_token_id contradicts the tokenizer's declared padding ID")
        return TokenIds(bos, eos, int(pad_token_id))
    if declared is not None:
        return TokenIds(bos, eos, declared)
    for name in ("<|finetune_right_pad_id|>", "<|pad|>", "<pad>", "[PAD]"):
        candidate = vocab.get(name)
        if candidate is not None and candidate != eos:
            return TokenIds(bos, eos, candidate)
    reserved = sorted(
        index
        for index, token in tokenizer.added_tokens_decoder.items()
        if token.special and "reserved" in token.content.lower() and index != eos and index in valid_ids
    )
    if reserved:
        return TokenIds(bos, eos, reserved[0])
    unknown = tokenizer.unk_token_id
    if unknown is not None and unknown in valid_ids and unknown != eos:
        return TokenIds(bos, eos, unknown)
    if eos is not None:
        return TokenIds(bos, eos, eos)
    raise ValueError(
        "No padding ID is available; set dataloader.zephon.packing.pad_token_id to an existing vocabulary ID"
    )

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

"""Text tokenizer IDs and pretokenized-document validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from nemo_automodel.components.datasets.zephon.config import (
        ZephonPretokenizedTextSourceConfig,
    )


@dataclass(frozen=True)
class TokenIds:
    """Resolved existing vocabulary IDs; no tokens are added to the model."""

    bos: int | None
    eos: int | None
    pad: int


def resolve_token_ids(tokenizer: PreTrainedTokenizerBase, *, pad_token_id: int | None) -> TokenIds:
    """Resolve PAD/BOS/EOS using the effective AutoModel tokenizer.

    Honor the declared tokenizer padding ID or a compatible explicit override.
    Otherwise prefer a named padding token, a reserved special token, UNK, then
    EOS. Reusing an existing ID avoids resizing model embeddings; preferring a
    non-EOS ID keeps padding distinct from document ends when possible. Padding
    is masked by position, including when PAD equals EOS.
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


def _validate_pretokenized_document(
    payload: Mapping[str, Any],
    *,
    config: ZephonPretokenizedTextSourceConfig,
    valid_token_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate tokens and supervision without allocating the delivered payload.

    ``valid_token_ids`` is a sorted int64 vocabulary array constructed once per
    pipeline. Exact vectorized membership preserves holes in tokenizer IDs.
    """
    incompatible = {
        "labels",
        "position_ids",
        "document_ids",
        "padding_mask",
        "_packed_seq_ids",
        "cu_seqlens",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "seq_lens",
        "seq_lens_padded",
    }.intersection(payload)
    if incompatible:
        raise ValueError(
            f"Pretokenized text expects unshifted tokens; unsupported fields: {sorted(incompatible)}. "
            "Supply optional boundaries as Zephon positions with pack_algorithm=null"
        )
    raw_tokens = payload[config.tokens_field]
    if not isinstance(raw_tokens, (Sequence, np.ndarray)) or isinstance(raw_tokens, (str, bytes)):
        raise ValueError("Pretokenized tokens must be a one-dimensional integer sequence")
    tokens = np.asarray(raw_tokens)
    invalid_ids = "Pretokenized tokens must be existing integer tokenizer vocabulary IDs"
    if tokens.ndim != 1 or (tokens.size and tokens.dtype.kind not in "iu"):
        raise ValueError(invalid_ids)
    if tokens.size:
        # Check unsigned bounds before casting: uint64 values must never wrap
        # into a valid signed embedding ID or compare through float64 rounding.
        if tokens.min() < 0 or int(tokens.max()) > np.iinfo(np.int64).max:
            raise ValueError(invalid_ids)
        tokens = tokens.astype(np.int64, copy=False)
        indices = np.searchsorted(valid_token_ids, tokens)
        if (indices >= valid_token_ids.size).any() or not np.array_equal(valid_token_ids[indices], tokens):
            raise ValueError(invalid_ids)
    if "attention_mask" in payload:
        attention = np.asarray(payload["attention_mask"])
        if attention.shape != tokens.shape or not (attention == 1).all():
            raise ValueError(
                "Pretokenized attention_mask must match token length and contain only ones; supply unpadded tokens"
            )
    mask = None
    if config.loss_mask_field is not None and config.loss_mask_field in payload:
        mask = np.asarray(payload[config.loss_mask_field])
        if mask.shape != tokens.shape or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Pretokenized loss_mask must be binary and match unshifted token length")
    return tokens, mask


def normalize_pretokenized_document(
    payload: Mapping[str, Any],
    *,
    config: ZephonPretokenizedTextSourceConfig,
    valid_token_ids: np.ndarray,
    preserve_positions: bool = False,
) -> dict[str, np.ndarray]:
    """Adapt pretokenized fields while preserving IDs and supplied supervision.

    Uniform loss-mask fields keep supervision aligned when masked and unmasked
    documents are packed together. Short and fully masked documents are retained;
    document-boundary masking belongs to to_training's mask_segment_starts option.

    Args:
        payload: Document with token IDs of shape [tokens] and an
            optional binary loss mask of shape [tokens], both before shifting.
            Offline packs may include zero-based reset ``positions`` [tokens].
        config: Source field names; absent loss masks select all tokens.
        valid_token_ids: Sorted vocabulary IDs of shape [vocab].
        preserve_positions: Accept offline boundaries when online packing is off.

    Returns:
        int64 input_ids and uint8 loss_mask arrays, each of shape [tokens].
        Values are preserved; a missing mask becomes all ones. Optional positions
        remain int64 [tokens]. Inputs are not mutated.
    """
    tokens, mask = _validate_pretokenized_document(payload, config=config, valid_token_ids=valid_token_ids)
    result = {
        "input_ids": tokens.astype(np.int64),
        "loss_mask": np.ones(tokens.shape, dtype=np.uint8) if mask is None else mask.astype(np.uint8),
    }
    if "positions" in payload:
        if not preserve_positions:
            raise ValueError(
                "Pretokenized positions require pack_algorithm=null; online packing creates its own boundaries"
            )
        positions = np.asarray(payload["positions"])
        if (
            positions.shape != tokens.shape
            or (positions.size and positions.dtype.kind not in "iu")
            or (positions.size and positions[0] != 0)
            or (positions.size and ((positions < 0) | (positions > tokens.size)).any())
            or ((positions[1:] != 0) & (positions[1:] != positions[:-1] + 1)).any()
        ):
            raise ValueError(
                "Pretokenized positions must match tokens and count from zero, resetting at subdocument starts"
            )
        result["positions"] = positions.astype(np.int64)
    return result


def prepare_text_document_without_packing(payload: dict[str, Any], *, max_length: int) -> dict[str, Any]:
    """Supply boundaries when packing is off, without guessing from token values.

    to_training needs scalar reset positions to derive document IDs and boundary
    loss masks. Offline exports can supply those positions; absent metadata means
    exactly one document, even when the tokens contain multiple EOS markers.

    Args:
        payload: Tokenized document with input_ids [tokens], optional aligned
            loss_mask [tokens], and optional validated positions [tokens].
        max_length: Unshifted token budget, including the final prediction target.

    Returns:
        The input mapping when it already has positions, otherwise a new mapping
        with int64 positions [tokens]. Existing arrays are never changed.

    Raises:
        ValueError: When the document exceeds the configured input/target budget.
    """
    length = len(payload["input_ids"])
    if length > max_length:
        raise ValueError(
            f"Document has {length} unshifted tokens, exceeding seq_len+1={max_length} with online packing disabled"
        )
    if "positions" in payload:
        return payload
    return {**payload, "positions": np.arange(length, dtype=np.int64)}


def measure_pretokenized_document(payload: Mapping[str, Any], *, tokens_field: str) -> int:
    """Count supplied token IDs for mixture calibration, including masked tokens.

    Calibration reads the raw document before field normalization, so it needs
    the configured token field. There is no tokenizer or filtering to replay;
    delivery validates the payload and preserves the token count.

    Args:
        payload: Raw document containing token IDs of shape [tokens].
        tokens_field: Field holding the unshifted token IDs.

    Returns:
        Number of supplied tokens, independent of their loss-mask values.
    """
    return len(payload[tokens_field])

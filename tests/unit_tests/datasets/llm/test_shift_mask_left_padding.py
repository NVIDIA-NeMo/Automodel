# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for _maybe_shift_mask_for_left_padding in formatting_utils."""

from types import SimpleNamespace

from nemo_automodel.components.datasets.llm.formatting_utils import _maybe_shift_mask_for_left_padding


def _make_tokenizer(padding_side: str) -> SimpleNamespace:
    return SimpleNamespace(padding_side=padding_side)


# ── Right-padding tokenizer: no-op ──────────────────────────────────────


def test_right_padding_is_noop():
    tok = _make_tokenizer("right")
    mask = [0, 0, 1, 1, 1, 0, 0, 0]
    attn = [1, 1, 1, 1, 1, 0, 0, 0]
    result = _maybe_shift_mask_for_left_padding(mask, tok, attn)
    assert result is mask, "Should return the exact same list object (no copy)"


# ── Left-padding tokenizer: shift right by pad_len ──────────────────────


def test_left_padding_shifts_mask_right():
    tok = _make_tokenizer("left")
    # 3 pad tokens on the left, content at positions 3-7
    attn = [0, 0, 0, 1, 1, 1, 1, 1]
    # mask built from unpadded positions: 1s at indices 2-4
    mask = [0, 0, 1, 1, 1, 0, 0, 0]
    result = _maybe_shift_mask_for_left_padding(mask, tok, attn)
    # pad_len=3 → shift right by 3: 1s should be at indices 5-7
    assert result == [0, 0, 0, 0, 0, 1, 1, 1]


def test_left_padding_preserves_length():
    tok = _make_tokenizer("left")
    mask = [0, 1, 1, 0, 0]
    attn = [0, 0, 1, 1, 1]
    result = _maybe_shift_mask_for_left_padding(mask, tok, attn)
    assert len(result) == len(mask)


def test_left_padding_zero_pad_is_noop():
    tok = _make_tokenizer("left")
    # No actual padding — all tokens are content
    mask = [0, 1, 1, 0]
    attn = [1, 1, 1, 1]
    result = _maybe_shift_mask_for_left_padding(mask, tok, attn)
    assert result is mask, "pad_len=0 → should return original list"


# ── Edge cases ───────────────────────────────────────────────────────────


def test_attention_mask_none_returns_original():
    tok = _make_tokenizer("left")
    mask = [0, 1, 1, 0]
    result = _maybe_shift_mask_for_left_padding(mask, tok, None)
    assert result is mask


def test_no_padding_side_attr_defaults_to_right():
    tok = SimpleNamespace()  # no padding_side attribute
    mask = [0, 1, 1, 0, 0]
    attn = [0, 0, 1, 1, 1]
    result = _maybe_shift_mask_for_left_padding(mask, tok, attn)
    assert result is mask, "Missing padding_side should default to right (no-op)"


def test_all_padding_shifts_entire_mask():
    tok = _make_tokenizer("left")
    # Entire sequence is padding
    mask = [1, 1, 0, 0]
    attn = [0, 0, 0, 0]
    result = _maybe_shift_mask_for_left_padding(mask, tok, attn)
    assert result == [0, 0, 0, 0]


def test_single_token_content():
    tok = _make_tokenizer("left")
    mask = [1, 0, 0]
    attn = [0, 0, 1]
    result = _maybe_shift_mask_for_left_padding(mask, tok, attn)
    # pad_len=2, shift right by 2: mask[0]=1 → position 2
    assert result == [0, 0, 1]

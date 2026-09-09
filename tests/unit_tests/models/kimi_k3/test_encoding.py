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

import pytest

from nemo_automodel.components.models.kimi_k3 import tokenization as kimi_k3_tokenization
from nemo_automodel.components.models.kimi_k3.encoding import build_chat_segments
from nemo_automodel.components.models.kimi_k3.tokenization import TikTokenTokenizer, _build_kimi_k3_pat_str


def test_kimi_k3_tokenizer_pattern_matches_reference_regex():
    reference_pattern = "|".join(
        [
            r"""[\p{Han}]+""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )

    assert _build_kimi_k3_pat_str() == reference_pattern
    assert "pat_str" not in TikTokenTokenizer.__dict__
    assert {
        name: value for name, value in vars(kimi_k3_tokenization).items() if isinstance(value, str) and "&&" in value
    } == {}


def test_medium_thinking_effort_is_rendered():
    segments = build_chat_segments(
        [{"role": "user", "content": "Hello"}],
        thinking_effort="medium",
    )

    rendered = "".join(segment.text for segment in segments)
    assert "thinking_effort=medium" in rendered


def test_invalid_thinking_effort_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported thinking_effort='extreme'"):
        build_chat_segments(
            [{"role": "user", "content": "Hello"}],
            thinking_effort="extreme",
        )


def test_assistant_segments_produce_direct_token_loss_mask():
    segments = build_chat_segments(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        add_generation_prompt=False,
        thinking=False,
    )
    tokenizer = object.__new__(TikTokenTokenizer)
    tokenizer._encode_text_piece = lambda text, allow_special_tokens: list(text.encode())

    token_ids, assistant_mask = tokenizer._encode_chat_segments(
        segments,
        return_assistant_tokens_mask=True,
    )
    rendered = "".join(segment.text for segment in segments)
    assistant_start = rendered.index("<|open|>response")

    assert len(token_ids) == len(assistant_mask)
    assert not any(assistant_mask[:assistant_start])
    assert all(assistant_mask[assistant_start:])


def test_single_chat_return_dict_is_not_batched(monkeypatch):
    tokenizer = object.__new__(TikTokenTokenizer)
    monkeypatch.setattr(
        tokenizer,
        "pad",
        lambda *args, **kwargs: {
            "input_ids": [[11, 12, 99]],
            "attention_mask": [[1, 1, 0]],
        },
    )

    output = tokenizer._format_chat_token_output(
        [[11, 12]],
        is_batched=False,
        padding="max_length",
        max_length=3,
        return_dict=True,
        assistant_masks=[[0, 1]],
    )

    assert output["input_ids"] == [11, 12, 99]
    assert output["attention_mask"] == [1, 1, 0]
    assert output["assistant_masks"] == [0, 1, 0]

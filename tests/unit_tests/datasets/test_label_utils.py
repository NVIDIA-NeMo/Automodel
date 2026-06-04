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

import importlib

import pytest
import torch


@pytest.fixture()
def label_mod():
    import nemo_automodel.components.datasets.vlm.collate_fns as _m

    return importlib.reload(_m)


def test_build_labels_retries_with_stripped_whitespace(label_mod, monkeypatch):
    """When a tokenizer produces different tokens for leading-whitespace text,
    build_labels should retry with lstripped text and still find the answer."""

    class WhitespaceTokenizer:
        """Tokenizer that produces different tokens for ' Hello' vs 'Hello'."""

        def __call__(self, text, add_special_tokens, return_tensors):
            assert add_special_tokens is False
            assert return_tensors == "pt"
            if text == " Hello":
                return {"input_ids": torch.tensor([[90, 91]])}
            if text == "Hello":
                return {"input_ids": torch.tensor([[10, 11]])}
            return {"input_ids": torch.tensor([[99]])}

        def decode(self, token):
            return ""

    class StubProcessor:
        def __init__(self):
            self.tokenizer = WhitespaceTokenizer()

    monkeypatch.setattr(label_mod, "default_stop_tokens", lambda processor: (), raising=True)

    # Encoded sequence contains stripped tokens [10, 11] but NOT whitespace tokens [90, 91]
    input_ids_batch = torch.tensor([[1, 2, 10, 11, 3]])
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": " Hello"}]},
    ]

    labels = label_mod.build_labels(input_ids_batch, [conversation], StubProcessor())
    assert labels.shape == input_ids_batch.shape
    # Tokens at positions 2,3 (the answer) should be unmasked; rest stays -100
    assert labels.tolist()[0] == [-100, -100, 10, 11, -100]


def test_build_labels_no_retry_when_no_leading_whitespace(label_mod, monkeypatch):
    """When assistant text has no leading whitespace and tokens are not found,
    build_labels should NOT retry and should warn (answer_start stays -1)."""

    call_count = [0]

    class NoRetryTokenizer:
        def __call__(self, text, add_special_tokens, return_tensors):
            call_count[0] += 1
            return {"input_ids": torch.tensor([[90, 91]])}

        def decode(self, token):
            return ""

    class StubProcessor:
        def __init__(self):
            self.tokenizer = NoRetryTokenizer()

    monkeypatch.setattr(label_mod, "default_stop_tokens", lambda processor: (), raising=True)

    input_ids_batch = torch.tensor([[1, 2, 3, 4, 5]])
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    ]

    labels = label_mod.build_labels(input_ids_batch, [conversation], StubProcessor())
    # No match found, all labels stay -100
    assert labels.tolist()[0] == [-100, -100, -100, -100, -100]
    # Tokenizer called only once (no retry since text has no leading whitespace)
    assert call_count[0] == 1


def test_build_labels_includes_stop_token(label_mod, monkeypatch):
    """
    Ensure `build_labels` copies the trailing stop token when it matches the configured set.
    """

    class StubTokenizer:
        def __call__(self, text, add_special_tokens, return_tensors):
            assert text == "assistant text"
            assert add_special_tokens is False
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[5, 6]])}

        def decode(self, token):
            if isinstance(token, list):
                token = token[0]
            if isinstance(token, torch.Tensor):
                token = token.item()
            return "STOP" if token == 7 else str(token)

    class StubProcessor:
        def __init__(self):
            self.tokenizer = StubTokenizer()

    monkeypatch.setattr(label_mod, "default_stop_tokens", lambda processor: ("STOP",), raising=True)

    input_ids_batch = torch.tensor([[1, 5, 6, 7]])
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "assistant text"}]},
    ]

    labels = label_mod.build_labels(input_ids_batch, [conversation], StubProcessor())
    assert labels.shape == input_ids_batch.shape
    assert labels.tolist()[0] == [-100, 5, 6, 7]

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

"""``PreTokenizedDatasetWrapper`` renders the chat template and then tokenizes the text; BOS must appear once."""

import pytest
import torch
from transformers import BatchFeature, ProcessorMixin

from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

BOS, TURN, END, NL, USER, MODEL, Q, A = 2, 105, 106, 107, 1000, 1001, 1002, 1003
_VOCAB = {"<bos>": BOS, "<|turn>": TURN, "<turn|>": END, "\n": NL, "user": USER, "model": MODEL, "q": Q, "a": A}
_VOCAB.update({"u": 1004, "XSENTINELMARKERX": 1005})  # the dummy turn used to derive assistant markers
_PIECES = ("<bos>", "<|turn>", "<turn|>", "\n")
_BODY = [TURN, USER, NL, Q, END, NL, TURN, MODEL, NL, A, END, NL]  # <|turn>user\nq<turn|>\n<|turn>model\na<turn|>\n


def _ids(text: str) -> list[int]:
    ids = []
    while text:
        piece = next((p for p in _PIECES if text.startswith(p)), None) or text.split("<", 1)[0].split("\n", 1)[0]
        ids.append(_VOCAB[piece])
        text = text[len(piece) :]
    return ids


def _render(conversation: list[dict[str, str]], template_bos: bool) -> str:
    turns = "".join(
        f"<|turn>{'model' if message['role'] == 'assistant' else message['role']}\n{message['content']}<turn|>\n"
        for message in conversation
    )
    return ("<bos>" if template_bos else "") + turns


class _Tokenizer:
    pad_token_id = 0
    unk_token_id = 3

    def __init__(self, template_bos: bool, tokenizer_bos: bool, has_bos_token: bool) -> None:
        self.template_bos = template_bos
        self.tokenizer_bos = tokenizer_bos
        self.bos_token = "<bos>" if has_bos_token else None
        self.bos_token_id = BOS if has_bos_token else None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = _ids(text)
        return [BOS] + ids if add_special_tokens and self.tokenizer_bos else ids

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, **kwargs):
        text = _render(conversation, self.template_bos)
        return {"input_ids": _ids(text)} if tokenize else text

    def convert_tokens_to_ids(self, token: str) -> int:
        return _VOCAB.get(token, self.unk_token_id)


class _Processor(ProcessorMixin):
    """Processor whose template has no generation keyword, so the wrapper renders first and tokenizes second."""

    attributes = []
    chat_template = "{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}"

    def __init__(self, template_bos: bool, tokenizer_bos: bool, has_bos_token: bool = True) -> None:
        self.tokenizer = _Tokenizer(template_bos, tokenizer_bos, has_bos_token)

    def apply_chat_template(self, conversations, tokenize=False, **kwargs):
        assert not tokenize
        return [_render(conversation, self.tokenizer.template_bos) for conversation in conversations]

    def __call__(self, text, images=None, videos=None, return_tensors="pt", add_special_tokens=True, **kwargs):
        ids = torch.tensor([self.tokenizer.encode(t, add_special_tokens=add_special_tokens) for t in text])
        return BatchFeature({"input_ids": ids, "attention_mask": torch.ones_like(ids)})


@pytest.mark.parametrize(
    ("template_bos", "tokenizer_bos", "has_bos_token", "expected"),
    [
        (True, True, True, [BOS] + _BODY),  # both would add it: once
        (True, False, True, [BOS] + _BODY),  # only the template
        (False, True, True, [BOS] + _BODY),  # only the tokenizer: its BOS is kept
        (False, False, True, _BODY),  # neither
        (False, False, False, _BODY),  # tokenizer without a BOS token
    ],
)
def test_bos_appears_once(template_bos, tokenizer_bos, has_bos_token, expected):
    example = {"conversation": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    wrapper = PreTokenizedDatasetWrapper(
        [example], _Processor(template_bos, tokenizer_bos, has_bos_token), inject_fake_images=False, max_retries=1
    )
    sample = wrapper[0]
    assert sample["input_ids"].tolist() == expected
    assert sample["labels"].tolist() == [-100] * (len(expected) - 3) + [A, END, -100]

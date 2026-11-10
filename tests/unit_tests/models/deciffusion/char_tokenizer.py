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

"""Deterministic stand-in tokenizer for Deciffusion prompt tests (no downloads)."""

from __future__ import annotations

import re
from types import SimpleNamespace

ADDED = ("<|im_start|>", "<|im_end|>", "<SPECIAL_100>")


class CharTokenizer:
    """One token per character; added tokens are single tokens wherever they occur, as in the real tokenizer."""

    def __init__(self) -> None:
        self.added_tokens_decoder = {1000 + i: SimpleNamespace(content=t) for i, t in enumerate(ADDED)}
        self._ids = {t: 1000 + i for i, t in enumerate(ADDED)}
        self._pattern = re.compile("|".join(re.escape(t) for t in ADDED))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, pos = [], 0
        for match in self._pattern.finditer(text):
            ids += [2000 + ord(c) for c in text[pos : match.start()]]
            ids.append(self._ids[match.group(0)])
            pos = match.end()
        return ids + [2000 + ord(c) for c in text[pos:]]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def decode(self, ids: list[int]) -> str:
        reverse = {v: k for k, v in self._ids.items()}
        return "".join(reverse[i] if i in reverse else chr(i - 2000) for i in ids)

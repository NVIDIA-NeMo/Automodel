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

"""Training labels for the Inkling 2 chat layout used by the MedPix recipes."""

import pytest
import torch

from nemo_automodel.components.datasets.vlm.collate_fns import build_labels_from_template

pytestmark = pytest.mark.cpu_only

_TOKENS = {
    "<|message_system|>": 10,
    "<|message_user|>": 11,
    "<|message_model|>": 12,
    "<|content_text|>": 13,
    "<|content_image|>": 14,
    "<|end_message|>": 15,
    "<|content_model_end_sampling|>": 16,
    "<|unused_200054|>": 17,
}


class _Inkling2Tokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return _TOKENS.get(token, self.unk_token_id)


class Inkling2Processor:
    """Stand-in carrying the processor class name the label builder dispatches on."""

    tokenizer = _Inkling2Tokenizer()


def test_inkling2_labels_cover_answer_through_sampling_terminator() -> None:
    t = _TOKENS
    # Layout the Inkling 2 chat template emits for one MedPix sample: the thinking-effort system
    # message, an image message, a question, then the answer closed by <|end_message|> and the
    # sampling terminator. Padding reuses the terminator id.
    input_ids = torch.tensor(
        [
            [
                t["<|message_system|>"], t["<|content_text|>"], 40, t["<|end_message|>"],
                t["<|message_user|>"], t["<|content_image|>"], t["<|unused_200054|>"], t["<|unused_200054|>"],
                t["<|end_message|>"],
                t["<|message_user|>"], t["<|content_text|>"], 41, t["<|end_message|>"],
                t["<|message_model|>"], t["<|content_text|>"], 50, 51, t["<|end_message|>"],
                t["<|content_model_end_sampling|>"],
                t["<|content_model_end_sampling|>"], t["<|content_model_end_sampling|>"],
            ]
        ]
    )  # fmt: skip

    labels = build_labels_from_template(input_ids, [[]], Inkling2Processor())

    expected = torch.full_like(input_ids, -100)
    expected[0, 15:19] = torch.tensor([50, 51, t["<|end_message|>"], t["<|content_model_end_sampling|>"]])
    torch.testing.assert_close(labels, expected)

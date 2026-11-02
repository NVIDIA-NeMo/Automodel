# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generation-template masks survive post-tokenize hooks (prefix injection, truncation)."""

import torch
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from nemo_automodel.components.datasets.vlm.collate_fns import (
    _GEMMA4_MODEL_TURN,
    _GEMMA4_THINKING_PREFIX,
    default_collate_fn,
    gemma4_inject_thinking_prefix,
    gemma4_prefix_collate_fn,
)
from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

MARKER = [20, 21]
PREFIX = [40, 41]
PAD = 0


class _Tokenizer:
    """Tokenizer stub exposing only what prefix injection and token-id lookups need."""

    pad_token_id = PAD
    unk_token_id = None

    def encode(self, text, add_special_tokens=False):
        return {_GEMMA4_MODEL_TURN: MARKER, _GEMMA4_THINKING_PREFIX: PREFIX}[text]

    def convert_tokens_to_ids(self, token):
        return None


class _GenerationProcessor(ProcessorMixin):
    """Processor stub with a ``{% generation %}`` template returning fixed ids and masks."""

    attributes = []
    chat_template = "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}"

    def __init__(self, input_ids, assistant_masks):
        self.tokenizer = _Tokenizer()
        self._input_ids = torch.tensor(input_ids)
        self._assistant_masks = torch.tensor(assistant_masks)

    def apply_chat_template(self, conversations, **kwargs):
        assert kwargs["tokenize"] and kwargs["return_dict"] and kwargs["return_tensors"] == "pt"
        assert kwargs["return_assistant_tokens_mask"]
        assert len(conversations) == self._input_ids.shape[0]
        return BatchFeature(
            {
                "input_ids": self._input_ids.clone(),
                "attention_mask": (self._input_ids != PAD).long(),
                "assistant_masks": self._assistant_masks.clone(),
            }
        )


def _examples(n):
    return [
        {"conversation": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]} for _ in range(n)
    ]


def test_prefix_injection_extends_assistant_masks():
    batch = _GenerationProcessor([[10, 20, 21, 30, 31]], [[0, 0, 0, 1, 1]]).apply_chat_template(
        _examples(1), tokenize=True, return_dict=True, return_tensors="pt", return_assistant_tokens_mask=True
    )
    out = gemma4_inject_thinking_prefix(batch, _GenerationProcessor([[10]], [[0]]))
    assert out["input_ids"].tolist() == [[10, 20, 21, 40, 41, 30, 31]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1]]
    assert out["assistant_masks"].tolist() == [[0, 0, 0, 0, 0, 1, 1]]


def test_prefix_collate_supervises_only_original_assistant_tokens():
    processor = _GenerationProcessor([[10, 20, 21, 30, 31]], [[0, 0, 0, 1, 1]])
    batch = gemma4_prefix_collate_fn(_examples(1), processor)
    assert "assistant_masks" not in batch
    # Collated inputs drop the final position; labels are shifted exactly once.
    assert batch["input_ids"].tolist() == [[10, 20, 21, 40, 41, 30]]
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, 30, 31]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1]]


def test_prefix_collate_truncation_slices_mask_with_tokens():
    processor = _GenerationProcessor([[10, 20, 21, 30, 31]], [[0, 0, 0, 1, 1]])
    batch = gemma4_prefix_collate_fn(_examples(1), processor, max_length=6)
    # Injection grows the row to 7 tokens; the hook truncates back to 6, keeping
    # target 30 and dropping 31 along with its mask entry.
    assert batch["input_ids"].tolist() == [[10, 20, 21, 40, 41]]
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, 30]]


def test_prefix_collate_zero_fills_mask_for_rows_with_fewer_markers():
    processor = _GenerationProcessor(
        [[10, 20, 21, 30, 31, PAD, PAD], [10, 20, 21, 30, 20, 21, 31]],
        [[0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 1]],
    )
    batch = gemma4_prefix_collate_fn(_examples(2), processor)
    assert batch["input_ids"].tolist() == [
        [10, 20, 21, 40, 41, 30, 31, PAD, PAD, PAD],
        [10, 20, 21, 40, 41, 30, 20, 21, 40, 41],
    ]
    assert batch["labels"].tolist() == [
        [-100, -100, -100, -100, 30, 31, -100, -100, -100, -100],
        [-100, -100, -100, -100, 30, -100, -100, -100, -100, 31],
    ]
    assert batch["attention_mask"].tolist() == [[1] * 7 + [0] * 3, [1] * 10]


def test_default_collate_without_hook_is_unchanged():
    processor = _GenerationProcessor([[10, 20, 21, 30, 31]], [[0, 0, 0, 1, 1]])
    batch = default_collate_fn(_examples(1), processor)
    assert batch["input_ids"].tolist() == [[10, 20, 21, 30]]
    assert batch["labels"].tolist() == [[-100, -100, 30, 31]]


def test_pretokenized_wrapper_hook_keeps_unshifted_labels():
    processor = _GenerationProcessor([[10, 20, 21, 30, 31]], [[0, 0, 0, 1, 1]])
    dataset = PreTokenizedDatasetWrapper(
        _examples(1),
        processor,
        post_tokenize_hook=gemma4_inject_thinking_prefix,
        inject_fake_images=False,
        max_retries=1,
    )
    sample = dataset[0]
    assert sample["input_ids"].tolist() == [10, 20, 21, 40, 41, 30, 31]
    assert sample["labels"].tolist() == [-100, -100, -100, -100, -100, 30, 31]
    assert sample["attention_mask"].tolist() == [1] * 7

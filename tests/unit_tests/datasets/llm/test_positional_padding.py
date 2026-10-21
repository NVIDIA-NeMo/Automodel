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

"""Dataset padding uses tokenizer positions even when token IDs collide."""

import json

import pytest

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from nemo_automodel.components.datasets.llm.formatting_utils import _package_tokenized_example
from nemo_automodel.components.datasets.utils import default_collater


class _ZeroPadTokenizer:
    """Return a real token0 and a real EOS followed by physical token0 padding."""

    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    chat_template = "{% generation %}text{% endgeneration %}"

    def apply_chat_template(self, messages, **kwargs):
        """Return the positional data that a padded tokenizer supplies."""
        return {
            "input_ids": [4, 0, 7, 2, 0, 0, 0, 0],
            "attention_mask": [1, 1, 1, 1, 0, 0, 0, 0],
            "assistant_masks": [0, 0, 1, 1, 0, 0, 0, 0],
        }


def test_chat_dataset_preserves_zero_pad_and_tokenizer_positional_mask(tmp_path):
    """A valid zero ID stays real while the same ID in padded slots is masked."""
    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "assistant", "content": "text"}]}) + "\n")
    tokenizer = _ZeroPadTokenizer()
    dataset = ChatDataset(str(path), tokenizer=tokenizer, seq_length=8, padding="max_length", truncation=True)
    sample = dataset[0]
    assert dataset.pad_token_id == tokenizer.pad_token_id == 0
    assert sample["input_ids"] == [4, 0, 7, 2, 0, 0, 0, 0]
    assert sample["attention_mask"] == [1, 1, 1, 1, 0, 0, 0, 0]
    assert sample["labels"] == [-100, 7, 2, -100, -100, -100, -100, -100]
    batch = default_collater([sample])
    assert batch["padding_mask"].tolist() == [[False, False, False, False, True, True, True, True]]


@pytest.mark.parametrize("unshifted", [False, True])
def test_explicit_mask_preserves_real_zero_eos_and_excludes_identical_padding(unshifted):
    """A positional mask distinguishes repeated real EOS0 from padding0."""
    sample = _package_tokenized_example(
        tokenizer=_ZeroPadTokenizer(),
        input_ids=[4, 0, 7, 0, 0, 0],
        assistant_masks=[1] * 6,
        eos_token_id=0,
        pad_token_id=0,
        seq_length=6,
        padding="max_length",
        unshifted=unshifted,
        tokenizer_attention_mask=[1, 1, 1, 1, 0, 0],
    )
    assert sample["attention_mask"] == [1, 1, 1, 1, 0, 0]
    if unshifted:
        assert sample["loss_mask"] == [1, 1, 1, 1, 0, 0]
    else:
        assert sample["labels"] == [0, 7, 0, -100, -100, -100]


def test_left_padding_never_supervises_prediction_from_a_padding_query():
    """Shifting labels must preserve both query and target positional validity."""
    sample = _package_tokenized_example(
        tokenizer=_ZeroPadTokenizer(),
        input_ids=[0, 0, 5, 0, 7, 2],
        assistant_masks=[1] * 6,
        eos_token_id=2,
        pad_token_id=0,
        seq_length=None,
        tokenizer_attention_mask=[0, 0, 1, 1, 1, 1],
    )
    assert sample["attention_mask"] == [0, 0, 1, 1, 1]
    assert sample["labels"] == [-100, -100, 0, 7, 2]

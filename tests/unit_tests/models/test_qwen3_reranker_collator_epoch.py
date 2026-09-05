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

"""Per-epoch context dropout in Qwen3ContextAwareRerankerCollator."""

from typing import Any, Dict, List

import torch

from nemo_automodel.components.models.qwen3_reranker.collator import Qwen3ContextAwareRerankerCollator


class _FakeTokenizer:
    """Whitespace tokenizer; ids are positional so lengths are all the collator needs."""

    def __call__(
        self,
        texts: List[str],
        max_length: int = None,
        padding: Any = None,
        truncation: bool = False,
    ) -> Dict[str, List[List[int]]]:
        input_ids = []
        for t in texts:
            tokens = t.split()
            if truncation and max_length is not None:
                tokens = tokens[:max_length]
            input_ids.append(list(range(len(tokens))))
        return {"input_ids": input_ids, "attention_mask": [[1] * len(i) for i in input_ids]}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return list(range(len(text.split())))


def _collator(**kwargs) -> Qwen3ContextAwareRerankerCollator:
    return Qwen3ContextAwareRerankerCollator(rerank_max_length=512, tokenizer=_FakeTokenizer(), **kwargs)


def _kept(collator: Qwen3ContextAwareRerankerCollator, queries: List[str]) -> List[bool]:
    """Keep/drop decision for the reasoning field across a fixed set of queries."""
    return [collator._keep_field("reasoning", q, 0.5) for q in queries]


QUERIES = [f"query number {i}" for i in range(64)]


def test_epoch_defaults_to_zero():
    """An untouched collator -- the validation case -- must sit at epoch 0."""
    collator = _collator()

    assert int(collator._epoch[0].item()) == 0


def test_set_epoch_records_the_epoch():
    collator = _collator()

    collator.set_epoch(7)

    assert int(collator._epoch[0].item()) == 7


def test_draw_is_deterministic_within_an_epoch():
    """Same seed and epoch must reproduce the same mask, so ranks and workers agree."""
    a, b = _collator(), _collator()
    a.set_epoch(3)
    b.set_epoch(3)

    assert _kept(a, QUERIES) == _kept(b, QUERIES)


def test_draw_changes_across_epochs():
    """Without this, epoch 2 is an exact prompt-level replay of epoch 1."""
    collator = _collator()

    collator.set_epoch(0)
    epoch0 = _kept(collator, QUERIES)
    collator.set_epoch(1)
    epoch1 = _kept(collator, QUERIES)

    assert epoch0 != epoch1


def test_unset_epoch_matches_epoch_zero():
    """Validation never receives an epoch, so its mix must equal the epoch-0 mix."""
    never_set, explicit = _collator(), _collator()
    explicit.set_epoch(0)

    assert _kept(never_set, QUERIES) == _kept(explicit, QUERIES)


def test_epoch_state_is_shared_memory():
    """collate_fn runs in DataLoader workers.

    Under ``persistent_workers=True`` those workers outlive the epoch boundary, so the
    epoch has to live in shared memory for a parent-side update to reach them; a plain
    int would leave every epoch replaying epoch 0's drops.
    """
    collator = _collator()

    assert isinstance(collator._epoch, torch.Tensor)
    assert collator._epoch.is_shared()
    assert collator._epoch.dtype == torch.int64


def test_set_epoch_mutates_in_place():
    """Rebinding the attribute instead of writing through it would not reach a worker."""
    collator = _collator()
    original = collator._epoch

    collator.set_epoch(5)

    assert collator._epoch is original
    assert int(original[0].item()) == 5


def test_zero_and_one_probabilities_ignore_the_epoch():
    """prob=0 keeps everything and prob=1 drops everything, at any epoch."""
    collator = _collator()
    collator.set_epoch(9)

    assert all(collator._keep_field("reasoning", q, 0.0) for q in QUERIES)
    assert not any(collator._keep_field("reasoning", q, 1.0) for q in QUERIES)

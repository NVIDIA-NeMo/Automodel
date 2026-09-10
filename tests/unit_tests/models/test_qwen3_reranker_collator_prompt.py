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

"""Prompt assembly, per-field truncation and batching in the Qwen3 reranker collator.

The epoch/dropout schedule is covered in ``test_qwen3_reranker_collator_epoch.py``. This
file covers what the schedule feeds into: which of the four prompt modes gets built, that
the instruction always describes the context the prompt actually carries, and that the
chat markers survive truncation.

Drop probabilities are pinned to 0.0 (always keep) or 1.0 (always drop) so a mode is
selected outright rather than sampled -- see ``_keep_field``, which short-circuits both
ends before hashing.
"""

from typing import Any, Dict, List

import pytest
import torch

from nemo_automodel.components.models.qwen3_reranker.collator import Qwen3ContextAwareRerankerCollator

KEEP, DROP = 0.0, 1.0


class _VocabTokenizer:
    """Whitespace tokenizer with a real id<->token mapping.

    The epoch test's fake returns positional ids, which is enough for length arithmetic
    but cannot round-trip: ``_truncate_tokens`` decodes the ids it kept, so asserting on
    truncated text needs ``decode(encode(t)) == t``.
    """

    def __init__(self) -> None:
        self._tok_to_id: Dict[str, int] = {}
        self._id_to_tok: Dict[int, str] = {}

    def _id(self, tok: str) -> int:
        if tok not in self._tok_to_id:
            i = len(self._tok_to_id)
            self._tok_to_id[tok] = i
            self._id_to_tok[i] = tok
        return self._tok_to_id[tok]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return [self._id(t) for t in text.split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self._id_to_tok[i] for i in ids)

    def __call__(
        self,
        texts: List[str],
        max_length: int = None,
        padding: Any = None,
        truncation: bool = False,
        add_special_tokens: bool = True,
    ) -> Dict[str, List[List[int]]]:
        input_ids = []
        for t in texts:
            ids = self.encode(t)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)
        return {"input_ids": input_ids, "attention_mask": [[1] * len(i) for i in input_ids]}

    def pad(
        self,
        features: List[Dict[str, List[int]]],
        padding: Any = True,
        pad_to_multiple_of: int = None,
        return_tensors: str = None,
    ) -> Dict[str, torch.Tensor]:
        width = max(len(f["input_ids"]) for f in features)
        if pad_to_multiple_of:
            width = -(-width // pad_to_multiple_of) * pad_to_multiple_of
        pad_id = self._id("<pad>")
        return {
            "input_ids": torch.tensor([f["input_ids"] + [pad_id] * (width - len(f["input_ids"])) for f in features]),
            "attention_mask": torch.tensor(
                [f["attention_mask"] + [0] * (width - len(f["attention_mask"])) for f in features]
            ),
        }


def _collator(**kwargs) -> Qwen3ContextAwareRerankerCollator:
    kwargs.setdefault("rerank_max_length", 512)
    return Qwen3ContextAwareRerankerCollator(tokenizer=_VocabTokenizer(), **kwargs)


ROW = dict(query="who wrote it", doc="a document", reasoning="a trace", global_query="the big question")


def _format(collator: Qwen3ContextAwareRerankerCollator) -> str:
    return collator._format_one(ROW["query"], ROW["doc"], ROW["reasoning"], ROW["global_query"])


# --------------------------------------------------------------------------------------
# The four prompt modes
# --------------------------------------------------------------------------------------


def test_base_mode_is_byte_identical_to_the_stock_prompt():
    """Both fields dropped must leave NO markers.

    This is the mode that has to match out-of-the-box Qwen3-Reranker exactly: a marker
    leaking in here would make every no-context row a slightly different prompt from the
    one the base model was trained on.
    """
    text = _format(_collator(reasoning_drop_prob=DROP, global_query_drop_prob=DROP))

    assert "#Query" not in text
    assert "#Reasoning Trace" not in text
    assert "#Original Question" not in text
    assert f"<Query>: {ROW['query']}\n" in text
    assert text.startswith("<Instruct>: ")
    assert text.endswith(f"<Document>: {ROW['doc']}")


def test_reasoning_mode_carries_only_the_trace():
    text = _format(_collator(reasoning_drop_prob=KEEP, global_query_drop_prob=DROP))

    assert f"#Reasoning Trace: {ROW['reasoning']}" in text
    assert f"#Query: {ROW['query']}" in text
    assert "#Original Question" not in text


def test_global_query_mode_carries_only_the_original_question():
    text = _format(_collator(reasoning_drop_prob=DROP, global_query_drop_prob=KEEP))

    assert f"#Original Question (global query): {ROW['global_query']}" in text
    assert f"#Query: {ROW['query']}" in text
    assert "#Reasoning Trace" not in text


def test_full_mode_orders_question_then_trace_then_query():
    """Order is part of the contract: the query is last so truncation cannot eat it."""
    text = _format(_collator(reasoning_drop_prob=KEEP, global_query_drop_prob=KEEP))

    assert text.index("#Original Question") < text.index("#Reasoning Trace") < text.index("#Query:")


@pytest.mark.parametrize(
    "r_prob, gq_prob, present",
    [
        (DROP, DROP, frozenset()),
        (KEEP, DROP, frozenset({"reasoning"})),
        (DROP, KEEP, frozenset({"global_query"})),
        (KEEP, KEEP, frozenset({"reasoning", "global_query"})),
    ],
)
def test_instruction_always_matches_the_surviving_fields(r_prob, gq_prob, present):
    """The instruction is chosen AFTER the draws, so it can never over-promise.

    Selecting it beforehand would leave a dropped row advertising context its prompt no
    longer carries -- invisible in the loss, and visible only as a model that ignores its
    context at eval.
    """
    collator = _collator(reasoning_drop_prob=r_prob, global_query_drop_prob=gq_prob)

    assert f"<Instruct>: {collator.DEFAULT_INSTRUCTIONS[present]}\n" in _format(collator)


def test_absent_fields_are_not_a_drop():
    """A row with no context is base mode even when nothing is dropped."""
    collator = _collator(reasoning_drop_prob=KEEP, global_query_drop_prob=KEEP)

    text = collator._format_one(ROW["query"], ROW["doc"], None, None)

    assert text.startswith(f"<Instruct>: {collator.DEFAULT_INSTRUCTIONS[frozenset()]}")
    assert "#Reasoning Trace" not in text and "#Original Question" not in text


def test_whitespace_only_context_counts_as_absent():
    collator = _collator(reasoning_drop_prob=KEEP, global_query_drop_prob=KEEP)

    text = collator._format_one(ROW["query"], ROW["doc"], "   ", "\n\t")

    assert "#Reasoning Trace" not in text and "#Original Question" not in text


# --------------------------------------------------------------------------------------
# Per-field truncation
# --------------------------------------------------------------------------------------


def test_each_field_is_capped_on_its_own_budget():
    """Caps apply per field before assembly, so a long trace cannot starve the query."""
    collator = _collator(
        reasoning_drop_prob=KEEP,
        global_query_drop_prob=KEEP,
        sub_query_max_length=2,
        reasoning_max_length=3,
        global_query_max_length=1,
    )

    text = collator._format_one("q1 q2 q3 q4", "d", "r1 r2 r3 r4 r5", "g1 g2 g3")

    assert "#Original Question (global query): g1\n" in text
    assert "#Reasoning Trace: r1 r2 r3\n" in text
    assert "#Query: q1 q2\n<Document>: d" in text


def test_document_is_uncapped_by_default():
    """passage_max_length defaults to None so the document absorbs the remainder."""
    doc = " ".join(f"d{i}" for i in range(50))

    text = _collator(reasoning_drop_prob=DROP, global_query_drop_prob=DROP)._format_one("q", doc)

    assert text.endswith(f"<Document>: {doc}")


def test_document_cap_applies_when_set():
    text = _collator(reasoning_drop_prob=DROP, global_query_drop_prob=DROP, passage_max_length=2)._format_one(
        "q", "d1 d2 d3 d4"
    )

    assert text.endswith("<Document>: d1 d2")


@pytest.mark.parametrize("limit, text", [(None, "a b c"), (0, "a b c"), (5, ""), (10, "a b c")])
def test_truncate_tokens_is_a_no_op_when_it_cannot_or_need_not_cut(limit, text):
    assert _collator()._truncate_tokens(text, limit) == text


def test_truncate_tokens_keeps_the_head():
    assert _collator()._truncate_tokens("a b c d", 2) == "a b"


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------


def test_call_wraps_every_example_in_the_chat_markers():
    """The prefix/suffix ids are concatenated around the truncated middle, so the markers
    survive regardless of how long the middle was."""
    collator = _collator(reasoning_drop_prob=DROP, global_query_drop_prob=DROP)
    features = [
        {"question": "q1", "doc_text": "d1"},
        {"question": "q2", "doc_text": "d2 d2 d2"},
    ]

    batch = collator(features)

    ids = batch["input_ids"].tolist()
    assert len(ids) == 2
    for row in ids:
        assert row[: len(collator.prefix_ids)] == collator.prefix_ids
        end = row.index(collator.suffix_ids[0], len(collator.prefix_ids))
        assert row[end : end + len(collator.suffix_ids)] == collator.suffix_ids


def test_call_pads_to_a_rectangular_batch():
    collator = _collator(reasoning_drop_prob=DROP, global_query_drop_prob=DROP)

    batch = collator([{"question": "q", "doc_text": "d"}, {"question": "q", "doc_text": "d d d d d"}])

    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()


def test_call_reads_context_columns_when_present():
    """The context columns are optional keys on the feature dict, so a row carrying them
    must produce a longer prompt than the same row without them."""
    collator = _collator(reasoning_drop_prob=KEEP, global_query_drop_prob=KEEP)
    row = {"question": "q", "doc_text": "d"}

    bare = collator([dict(row)])
    with_context = collator([dict(row, reasoning="trace here", global_query="big question")])

    assert with_context["input_ids"].shape[1] > bare["input_ids"].shape[1]


def test_call_emits_labels_only_when_num_labels_is_supplied():
    collator = _collator(reasoning_drop_prob=DROP, global_query_drop_prob=DROP)

    without = collator([{"question": "q", "doc_text": "d"}])
    with_labels = collator([{"question": "q", "doc_text": "d", "num_labels": 3}])

    assert "labels" not in without
    assert torch.equal(with_labels["labels"], torch.zeros(3, dtype=torch.long))


# --------------------------------------------------------------------------------------
# instructions= normalisation and validation
# --------------------------------------------------------------------------------------


def _instruction_map(make_key):
    return {
        make_key(()): "none",
        make_key(("reasoning",)): "r",
        make_key(("global_query",)): "g",
        make_key(("reasoning", "global_query")): "rg",
    }


def test_instructions_accept_tuple_keys():
    collator = _collator(instructions=_instruction_map(tuple))

    assert collator.instructions[frozenset({"reasoning"})] == "r"


def test_instructions_accept_comma_separated_string_keys():
    """YAML cannot express a frozenset, so a config supplies 'reasoning,global_query'."""
    collator = _collator(instructions=_instruction_map(lambda fields: ",".join(fields)))

    assert collator.instructions[frozenset({"reasoning", "global_query"})] == "rg"
    assert collator.instructions[frozenset()] == "none"


def test_instructions_reject_a_key_that_is_not_a_set_tuple_or_string():
    with pytest.raises(TypeError, match="must be str, tuple, or frozenset"):
        _collator(instructions={0: "none"})


def test_instructions_reject_an_unsupported_mode():
    bad = _instruction_map(tuple)
    bad[frozenset({"nonsense"})] = "?"

    with pytest.raises(ValueError, match="unsupported context modes"):
        _collator(instructions=bad)


def test_instructions_reject_a_missing_mode():
    """Every mode reachable under the drop probabilities needs its own instruction."""
    partial = _instruction_map(tuple)
    del partial[("reasoning",)]

    with pytest.raises(ValueError, match="missing an entry"):
        _collator(instructions=partial)

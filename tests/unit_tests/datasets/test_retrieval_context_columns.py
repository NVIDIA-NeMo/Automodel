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

"""Per-query context columns through the inline retrieval dataset.

The group-aware split is covered in ``test_retrieval_group_aware_split.py``. This file
covers the other half of the same change: carrying ``reasoning`` and ``global_query`` from
the raw rows to the collator, and the argument validation that guards the builder.

The loader is monkeypatched rather than reading files, so these stay CPU-only unit tests
with no fixture corpus on disk.
"""

import pytest

from nemo_automodel.components.datasets.llm import retrieval_dataset_inline as mod
from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import (
    _flatten_context_columns,
    make_context_aware_retrieval_dataset,
)

# --------------------------------------------------------------------------------------
# _flatten_context_columns
# --------------------------------------------------------------------------------------


def _bi_encoder_batch() -> dict:
    """Two queries with 2 and 3 documents respectively, plus per-query context."""
    return {
        "question": ["q0", "q1"],
        "doc_text": [["d0a", "d0b"], ["d1a", "d1b", "d1c"]],
        "doc_image": [[None, None], [None, None, None]],
        "reasoning": ["r0", "r1"],
        "global_query": ["g0", "g1"],
    }


def test_context_columns_repeat_once_per_document():
    """Context is per QUERY but the cross-encoder row is per (query, document).

    A listwise group is one softmax over its documents, so every row of a group has to
    carry the same context -- if they disagreed, the comparison would be between prompts
    rather than between documents.
    """
    flattened = _flatten_context_columns(_bi_encoder_batch(), ("reasoning", "global_query"))

    assert flattened["reasoning"] == ["r0", "r0", "r1", "r1", "r1"]
    assert flattened["global_query"] == ["g0", "g0", "g1", "g1", "g1"]
    # and they line up one-for-one with the flattened questions
    assert len(flattened["reasoning"]) == len(flattened["question"]) == 5


def test_context_columns_align_with_their_own_question():
    flattened = _flatten_context_columns(_bi_encoder_batch(), ("reasoning",))

    pairs = set(zip(flattened["question"], flattened["reasoning"]))
    assert pairs == {("q0", "r0"), ("q1", "r1")}


def test_a_requested_column_absent_from_the_batch_is_skipped():
    """Configs may name a column the data does not carry; that must not raise here."""
    batch = _bi_encoder_batch()
    del batch["global_query"]

    flattened = _flatten_context_columns(batch, ("reasoning", "global_query"))

    assert "global_query" not in flattened
    assert flattened["reasoning"] == ["r0", "r0", "r1", "r1", "r1"]


def test_no_context_columns_leaves_the_flattened_batch_untouched():
    flattened = _flatten_context_columns(_bi_encoder_batch(), ())

    assert "reasoning" not in flattened
    assert flattened["question"] == ["q0", "q0", "q1", "q1", "q1"]


# --------------------------------------------------------------------------------------
# Builder argument validation
# --------------------------------------------------------------------------------------


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="model_type must be one of"):
        make_context_aware_retrieval_dataset(data_dir_list=["unused"], model_type="tri_encoder")


def test_unknown_data_type_is_rejected():
    with pytest.raises(ValueError, match="Invalid data type"):
        make_context_aware_retrieval_dataset(data_dir_list=["unused"], data_type="holdout")


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_validation_fraction_outside_zero_to_one_is_rejected(fraction):
    """Rejected where it is supplied, not later.

    A negative fraction would skip the split entirely and return the WHOLE dataset as
    validation -- total leakage with no error. A fraction of 1 or more does raise, but
    from the empty-side check, whose message blames the group count.
    """
    with pytest.raises(ValueError, match="validation_fraction must be in"):
        make_context_aware_retrieval_dataset(data_dir_list=["unused"], validation_fraction=fraction)


# --------------------------------------------------------------------------------------
# Builder wiring, with the loader stubbed
# --------------------------------------------------------------------------------------


class _FakeDataset:
    """Stand-in for ``datasets.Dataset`` exposing only what the builder touches."""

    def __init__(self, rows: dict):
        self._rows = rows
        self.transform = None
        self.shuffled_with = None
        self.selected = None

    @property
    def column_names(self) -> list[str]:
        return sorted(self._rows)

    def __len__(self) -> int:
        return len(self._rows["question"])

    def set_transform(self, fn):
        self.transform = fn

    def shuffle(self, seed=None):
        self.shuffled_with = seed
        return self

    def select(self, rng):
        self.selected = list(rng)
        return self


RAW = {
    "question": ["q0", "q1"],
    "corpus_id": ["c", "c"],
    "pos_doc": [["p0"], ["p1"]],
    "neg_doc": [["n0"], ["n1"]],
    "trace": ["r0", "r1"],
    "origin": ["g0", "g1"],
}


@pytest.fixture
def stub_loader(monkeypatch):
    """Return the dataset the builder will be handed, and capture the columns requested."""
    holder = {}

    def _load_datasets(data_dir_list, concatenate=True, extra_columns=None):
        holder["extra_columns"] = extra_columns
        holder["dataset"] = _FakeDataset(dict(RAW))
        return holder["dataset"], {}

    monkeypatch.setattr(mod, "load_datasets", _load_datasets)
    return holder


def test_builder_requests_the_configured_context_columns_from_the_loader(stub_loader):
    """The loader drops columns it was not asked for, so the names have to be threaded
    through -- otherwise the context silently never arrives."""
    make_context_aware_retrieval_dataset(
        data_dir_list=["unused"], reasoning_column="trace", global_query_column="origin"
    )

    assert set(stub_loader["extra_columns"]) == {"trace", "origin"}


def test_builder_renames_context_columns_to_the_collator_contract(stub_loader):
    """The collator reads ``reasoning``/``global_query``; the dataset may call them
    anything, so the transform has to rename them."""
    make_context_aware_retrieval_dataset(
        data_dir_list=["unused"],
        model_type="bi_encoder",
        reasoning_column="trace",
        global_query_column="origin",
    )

    out = stub_loader["dataset"].transform(dict(RAW))

    assert out["reasoning"] == ["r0", "r1"]
    assert out["global_query"] == ["g0", "g1"]


def test_cross_encoder_transform_flattens_while_bi_encoder_does_not(stub_loader):
    make_context_aware_retrieval_dataset(
        data_dir_list=["unused"], model_type="cross_encoder", reasoning_column="trace", n_passages=2
    )

    out = stub_loader["dataset"].transform(dict(RAW))

    # one row per (query, document) rather than one row per query
    assert len(out["reasoning"]) == len(out["question"]) > len(RAW["question"])


def test_train_split_shuffles_and_windows_only_when_asked(stub_loader):
    make_context_aware_retrieval_dataset(
        data_dir_list=["unused"],
        data_type="train",
        do_shuffle=True,
        seed=7,
        max_train_samples=1,
        train_data_select_offset=1,
    )

    dataset = stub_loader["dataset"]
    assert dataset.shuffled_with == 7
    assert dataset.selected == [1]


def test_eval_split_is_neither_shuffled_nor_windowed(stub_loader):
    """Shuffling an eval split would make the reported metric depend on the seed."""
    make_context_aware_retrieval_dataset(data_dir_list=["unused"], data_type="eval", do_shuffle=True)

    assert stub_loader["dataset"].shuffled_with is None
    assert stub_loader["dataset"].selected is None

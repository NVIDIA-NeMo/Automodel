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

"""Group-aware train/validation split.

The split is computed by two SEPARATE calls (one per config), so the property under test
is that a group's side is a pure function of ``(seed, group, fraction)`` -- independent of
row order and of which other groups happen to be present.
"""

import pytest

from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import (
    _group_aware_split,
    make_context_aware_retrieval_dataset,
)

SEED = 42
GROUP_KEY = "query_id"


class _FakeDataset:
    """Minimal stand-in for a ``datasets.Dataset`` exposing only what the split uses."""

    def __init__(self, groups: list[str]):
        self._groups = list(groups)

    @property
    def column_names(self) -> list[str]:
        return [GROUP_KEY]

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, key: str) -> list[str]:
        assert key == GROUP_KEY
        return self._groups

    def select(self, indices):
        return [self._groups[i] for i in indices]


def _split(groups: list[str], data_type: str, fraction: float = 0.25) -> set[str]:
    """Return the set of group ids selected for ``data_type``."""
    return set(_group_aware_split(_FakeDataset(groups), fraction, GROUP_KEY, data_type, SEED))


def test_train_and_validation_are_complementary():
    groups = [f"q{i}" for i in range(40)]
    train = _split(groups, "train")
    val = _split(groups, "validation")

    assert train | val == set(groups)
    assert train & val == set()
    assert val, "expected a non-empty validation slice at fraction=0.25"


def test_all_rows_of_a_group_land_on_one_side():
    # every group contributes three rows, interleaved so row order cannot be relied on
    groups = [f"q{i}" for i in range(20)] * 3
    train = _split(groups, "train")
    val = _split(groups, "validation")

    assert train & val == set()


def test_side_is_independent_of_row_order():
    groups = [f"q{i}" for i in range(30)]
    assert _split(groups, "validation") == _split(list(reversed(groups)), "validation")


def test_adding_a_group_does_not_move_existing_groups():
    """Growing the dataset must not evict an existing group across the boundary.

    A rank-and-take-lowest-N split fails this: the cutoff is a property of the whole
    group set, so one added group can displace whichever group held the last slot.
    """
    before = [f"q{i}" for i in range(30)]
    after = before + [f"new{i}" for i in range(10)]

    val_before = _split(before, "validation")
    val_after = _split(after, "validation")

    # every original group keeps its side
    assert val_before == val_after & set(before)
    train_before = _split(before, "train")
    assert train_before == _split(after, "train") & set(before)


def test_disjoint_group_universes_do_not_leak():
    """The two builds need not resolve to an identical group set.

    If the train config sees rows the validation config does not, a composition-dependent
    cutoff can put the same group in training on one side and validation on the other.
    """
    train_side_groups = [f"q{i}" for i in range(30)] + ["extra1", "extra2"]
    val_side_groups = [f"q{i}" for i in range(30)]

    train = _split(train_side_groups, "train")
    val = _split(val_side_groups, "validation")

    assert train & val == set(), "a group reached both training and validation"


def test_zero_fraction_sends_everything_to_train():
    groups = [f"q{i}" for i in range(20)]
    assert _split(groups, "train", fraction=0.0) == set(groups)
    assert _split(groups, "validation", fraction=0.0) == set()


def test_eval_is_treated_as_the_validation_side():
    groups = [f"q{i}" for i in range(30)]
    assert _split(groups, "eval") == _split(groups, "validation")


def test_blank_group_value_is_rejected():
    with pytest.raises(ValueError, match="missing or blank"):
        _split(["q0", "", "q2"], "train")


def test_missing_group_column_is_rejected():
    dataset = _FakeDataset(["q0", "q1"])
    with pytest.raises(ValueError, match="not a column of the dataset"):
        _group_aware_split(dataset, 0.25, "no_such_column", "train", SEED)


def test_empty_validation_side_raises():
    """Few groups + a small fraction can hash to zero validation groups."""
    # one group, fraction 0.25 -> the single group almost certainly lands in train
    with pytest.raises(ValueError, match="every group on one side"):
        _split(["only_group"], "train", fraction=0.25)


def test_empty_train_side_raises():
    with pytest.raises(ValueError, match="every group on one side"):
        _split(["only_group"], "train", fraction=0.999)


def test_error_names_the_knobs_that_produced_it():
    try:
        _split(["only_group"], "validation", fraction=0.25)
    except ValueError as e:
        msg = str(e)
    else:
        raise AssertionError("expected ValueError")
    assert "validation_fraction=0.25" in msg
    assert f"seed={SEED}" in msg
    assert GROUP_KEY in msg


def test_healthy_split_does_not_raise():
    groups = [f"q{i}" for i in range(40)]
    assert _split(groups, "train")
    assert _split(groups, "validation")


def test_zero_fraction_never_raises_even_with_one_group():
    """validation_fraction=0 means 'no held-out slice', which is a valid request."""
    assert _split(["only_group"], "train", fraction=0.0) == {"only_group"}


# The range of validation_fraction itself, checked by the builder before any data is read.
# Both ends fail silently otherwise: a negative fraction skips the split so the validation
# build returns the whole dataset, and a fraction of 1 or more reaches the empty-side check,
# which blames the group count instead of the value that was actually wrong.


@pytest.mark.parametrize("fraction", [-0.1, -1.0, 1.0, 1.5])
def test_out_of_range_validation_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match=r"validation_fraction must be in \[0, 1\)"):
        make_context_aware_retrieval_dataset(
            data_dir_list="unread.jsonl",
            validation_fraction=fraction,
            validation_group_key=GROUP_KEY,
        )


def test_out_of_range_fraction_is_caught_before_any_data_is_read():
    """The path does not exist, so reaching the loader would raise something else."""
    with pytest.raises(ValueError, match="validation_fraction"):
        make_context_aware_retrieval_dataset(
            data_dir_list="/nonexistent/path/never/opened.jsonl",
            validation_fraction=-0.5,
        )

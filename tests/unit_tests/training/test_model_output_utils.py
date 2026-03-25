# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_automodel.components.training.model_output_utils import get_final_hidden_states


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AttrOutput:
    """Minimal HF ModelOutput-like object (attribute access)."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


# ---------------------------------------------------------------------------
# None / missing cases
# ---------------------------------------------------------------------------


def test_none_input_returns_none():
    assert get_final_hidden_states(None) is None


def test_dict_missing_key_returns_none():
    assert get_final_hidden_states({}) is None


def test_dict_explicit_none_returns_none():
    assert get_final_hidden_states({"hidden_states": None}) is None


def test_attr_missing_returns_none():
    # Object with no hidden_states attribute
    assert get_final_hidden_states(object()) is None


def test_attr_explicit_none_returns_none():
    assert get_final_hidden_states(_AttrOutput(hidden_states=None)) is None


# ---------------------------------------------------------------------------
# Single tensor (custom model layout)
# ---------------------------------------------------------------------------


def test_dict_single_tensor_returned_as_is():
    t = torch.zeros(2, 4, 8)
    result = get_final_hidden_states({"hidden_states": t})
    assert result is t


def test_attr_single_tensor_returned_as_is():
    t = torch.ones(3, 5, 16)
    result = get_final_hidden_states(_AttrOutput(hidden_states=t))
    assert result is t


# ---------------------------------------------------------------------------
# Tuple / list of tensors (standard HF layout)
# ---------------------------------------------------------------------------


def test_tuple_returns_last_tensor():
    layers = [torch.full((1, 2, 4), float(i)) for i in range(4)]
    result = get_final_hidden_states({"hidden_states": tuple(layers)})
    assert result is layers[-1]


def test_list_returns_last_tensor():
    layers = [torch.zeros(1, 2, 4), torch.ones(1, 2, 4)]
    result = get_final_hidden_states({"hidden_states": layers})
    assert result is layers[-1]


def test_tuple_with_trailing_nones_returns_last_real_tensor():
    t0 = torch.zeros(1, 2, 4)
    t1 = torch.ones(1, 2, 4)
    hidden_states = (t0, t1, None, None)
    result = get_final_hidden_states({"hidden_states": hidden_states})
    assert result is t1


def test_tuple_with_leading_and_trailing_nones():
    real = torch.full((2, 3, 8), 7.0)
    hidden_states = (None, real, None)
    result = get_final_hidden_states({"hidden_states": hidden_states})
    assert result is real


def test_all_none_tuple_returns_none():
    result = get_final_hidden_states({"hidden_states": (None, None)})
    assert result is None


def test_empty_tuple_returns_none():
    result = get_final_hidden_states({"hidden_states": ()})
    assert result is None


def test_single_element_tuple_returns_that_tensor():
    t = torch.zeros(1, 1, 4)
    result = get_final_hidden_states({"hidden_states": (t,)})
    assert result is t


# ---------------------------------------------------------------------------
# TypeError paths
# ---------------------------------------------------------------------------


def test_unexpected_hidden_states_type_raises():
    with pytest.raises(TypeError, match="Unexpected hidden_states type"):
        get_final_hidden_states({"hidden_states": "not_a_tensor"})


def test_tuple_with_non_tensor_entry_raises():
    with pytest.raises(TypeError, match="Expected hidden_states entries to be tensor-like"):
        get_final_hidden_states({"hidden_states": (torch.zeros(2), "bad_entry")})


# ---------------------------------------------------------------------------
# Attr-based access mirrors dict-based access
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hidden_states,expected_val",
    [
        (torch.zeros(2, 4), "self"),          # single tensor → returned as-is
        ((torch.zeros(2, 4), torch.ones(2, 4)), "last"),  # tuple → last
    ],
)
def test_attr_access_consistent_with_dict_access(hidden_states, expected_val):
    if expected_val == "self":
        assert get_final_hidden_states(_AttrOutput(hidden_states)) is hidden_states
    else:
        assert get_final_hidden_states(_AttrOutput(hidden_states)) is hidden_states[-1]

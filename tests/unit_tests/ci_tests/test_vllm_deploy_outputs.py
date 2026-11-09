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

import pytest

from tests.functional_tests.checkpoint_robustness.test_checkpoint_vllm_deploy import _assert_greedy_token_match


@pytest.mark.parametrize("tokens", [[42], [12366, 13, 100265]])
def test_short_identical_greedy_outputs_are_valid(tokens):
    assert _assert_greedy_token_match(tokens, tokens.copy(), "prompt", 0, tokens[-1]) == len(tokens)


@pytest.mark.parametrize(
    ("hf_tokens", "vllm_tokens"),
    [
        ([], []),
        ([12, 34, 56], [12, 34, 57]),
        ([12, 34, 56], [12, 34, 56, 78]),
    ],
)
def test_empty_or_different_short_greedy_outputs_fail(hf_tokens, vllm_tokens):
    with pytest.raises(AssertionError):
        _assert_greedy_token_match(hf_tokens, vllm_tokens, "prompt", 0, 56)


@pytest.mark.parametrize("eos_token_ids", [99, [98, 99], None])
def test_identical_short_outputs_must_end_in_configured_eos(eos_token_ids):
    with pytest.raises(AssertionError, match="did not end in a configured EOS token"):
        _assert_greedy_token_match([12, 34], [12, 34], "prompt", 0, eos_token_ids)


def test_long_outputs_may_diverge_after_required_prefix():
    hf_tokens = [1, 2, 3, 4, 5, 6]
    vllm_tokens = [1, 2, 3, 4, 5, 7]

    assert _assert_greedy_token_match(hf_tokens, vllm_tokens, "prompt", 0, None) == 5


def test_long_outputs_must_match_required_prefix():
    with pytest.raises(AssertionError, match="agree on only 4 leading token"):
        _assert_greedy_token_match([1, 2, 3, 4, 5], [1, 2, 3, 4, 6], "prompt", 0, None)

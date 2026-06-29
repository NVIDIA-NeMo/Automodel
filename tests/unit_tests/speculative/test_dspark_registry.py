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

"""Tests for the DSpark draft registry and the default target-layer spread."""

import pytest

from nemo_automodel.components.speculative.dspark.draft_qwen3 import Qwen3DSparkModel
from nemo_automodel.components.speculative.dspark.registry import (
    build_target_layer_ids,
    resolve_dspark_draft_spec,
)


def test_resolve_qwen3():
    assert resolve_dspark_draft_spec(["Qwen3ForCausalLM"]).draft_cls is Qwen3DSparkModel
    assert resolve_dspark_draft_spec(["Qwen3MoeForCausalLM"]).draft_cls is Qwen3DSparkModel


def test_resolve_unsupported_raises():
    with pytest.raises(ValueError, match="no DSpark draft spec"):
        resolve_dspark_draft_spec(["LlamaForCausalLM"])


def test_build_target_layer_ids_is_sorted_in_range_and_sized():
    ids = build_target_layer_ids(28, 5)
    assert ids == sorted(set(ids))  # strictly increasing, unique
    assert len(ids) == 5
    assert all(1 <= i <= 27 for i in ids)


def test_build_target_layer_ids_single():
    assert build_target_layer_ids(28, 1) == [27]

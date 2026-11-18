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

"""CLEF release export merges the expected low-rank update."""

import pytest
import torch

from nemo_automodel.components.models.clef.export import merge_lora_


def test_lora_merge_updates_only_the_selected_base_parameter():
    state = {"block.weight": torch.eye(2), "other.weight": torch.tensor([7.0])}
    adapter = {
        "block.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "block.lora_B.weight": torch.tensor([[3.0], [4.0]]),
    }
    assert merge_lora_(state, adapter, scale=0.5) == 1
    torch.testing.assert_close(state["block.weight"], torch.tensor([[2.5, 3.0], [2.0, 5.0]]))
    torch.testing.assert_close(state["other.weight"], torch.tensor([7.0]))


def test_head_only_stage_does_not_modify_backbone():
    state = {"block.weight": torch.eye(2)}
    assert merge_lora_(state, {}, scale=0.0) == 0
    torch.testing.assert_close(state["block.weight"], torch.eye(2))


def test_lora_merge_rejects_unmatched_base_parameter():
    with pytest.raises(KeyError, match="no base weight"):
        merge_lora_({}, {"block.lora_A.weight": torch.ones(1, 2)}, scale=1.0)

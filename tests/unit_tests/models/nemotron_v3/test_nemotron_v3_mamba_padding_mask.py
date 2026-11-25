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

"""Which padding mask the backbone hands to its Mamba layers.

A ``[batch, sequence]`` mask of ones multiplies every hidden state by one, so the Mamba layers get no
mask at all. Any other mask (padding zeros, or values other than one) must reach them unchanged.
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block
from nemo_automodel.components.models.nemotron_v3.model import NemotronV3Model
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def model_and_masks(monkeypatch):
    seen: list[tuple[str, torch.Tensor | None]] = []

    def record(self, hidden_states, attention_mask=None, **kwargs):
        seen.append((self.block_type, attention_mask))
        return hidden_states

    monkeypatch.setattr(NemotronV3Block, "forward", record)
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    config = MockNemotronV3Config(layers_block_type=["mamba", "attention"])
    return NemotronV3Model(config, backend=backend).float(), seen


def _mamba_masks(seen: list[tuple[str, torch.Tensor | None]]) -> list[torch.Tensor | None]:
    return [mask for block_type, mask in seen if block_type == "mamba"]


def test_mask_of_ones_is_not_passed_to_mamba_layers(model_and_masks):
    model, seen = model_and_masks
    input_ids = torch.randint(1, 64, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    model(input_ids, attention_mask=mask)
    assert _mamba_masks(seen) == [None]
    # Attention layers are unaffected: without a causal mask they still fall back to the 2D mask.
    assert [m for block_type, m in seen if block_type == "attention"][0] is mask


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=dtype)
        for dtype in (torch.long, torch.bool, torch.float32)
    ]
    # Only a multiply by exactly one is dropped; an indexed mask keeps its existing behaviour.
    + [torch.tensor([[1, 1, 1, 2, 2, 2], [1, 1, 2, 2, 3, 3]])],
    ids=["padding_long", "padding_bool", "padding_float", "values_other_than_one"],
)
def test_other_masks_reach_mamba_layers(model_and_masks, mask):
    model, seen = model_and_masks
    model(torch.randint(1, 64, (2, 6)), attention_mask=mask)
    (mamba_mask,) = _mamba_masks(seen)
    assert mamba_mask is mask

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

"""Vision tower parity against the transformers reference on a tiny random tower (see ``ref_dump_vision.py``)."""

import os

import pytest
import torch

from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashVisionConfig
from nemo_automodel.components.models.hy_v4_flash.vision import HY4FlashVisionTower

DUMP = os.environ.get("HY4_REF_VISION_DUMP")


@pytest.mark.skipif(DUMP is None or not os.path.exists(DUMP), reason="HY4_REF_VISION_DUMP reference dump not provided")
def test_vision_tower_matches_reference():
    dump = torch.load(DUMP, weights_only=False)
    tower = HY4FlashVisionTower(HY4FlashVisionConfig(**dump["kwargs"])).float().eval()
    state = {k.removeprefix("vit."): v for k, v in dump["state"].items()}
    missing, unexpected = tower.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    with torch.no_grad():
        features = tower(dump["pixels"], dump["grid"])
    torch.testing.assert_close(features, dump["features"], rtol=1e-4, atol=1e-4)

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

"""Temperature validation shared by retrieval recipes."""

import pytest

from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe
from nemo_automodel.recipes.retrieval.train_cross_encoder import TrainCrossEncoderRecipe


@pytest.mark.parametrize("recipe_cls", [TrainBiEncoderRecipe, TrainCrossEncoderRecipe])
@pytest.mark.parametrize("temperature", [0.0, -0.3, float("nan"), float("inf"), float("-inf")])
def test_invalid_temperature_fails_at_initialization(recipe_cls, temperature):
    with pytest.raises(ValueError, match="temperature must be finite and greater than 0") as exc_info:
        recipe_cls({"temperature": temperature})
    assert repr(temperature) in str(exc_info.value)


@pytest.mark.parametrize("recipe_cls", [TrainBiEncoderRecipe, TrainCrossEncoderRecipe])
@pytest.mark.parametrize(
    "cfg, expected",
    [({}, 1.0), ({"temperature": 0.3}, 0.3), ({"temperature": 1}, 1), ({"temperature": 2.0}, 2.0)],
)
def test_valid_temperature_preserves_configured_value_and_default(recipe_cls, cfg, expected):
    recipe = recipe_cls(cfg)
    assert recipe.temperature == expected

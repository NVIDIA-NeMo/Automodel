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

from unittest.mock import MagicMock

import pytest

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    _validate_mok_thd_packing,
)


def _recipe_config(*, dispatcher: str, packed_sequence_size: int, packing_strategy: str = "thd") -> RecipeConfig:
    return RecipeConfig(
        ConfigNode(
            {
                "model": {"backend": {"dispatcher": dispatcher}},
                "packed_sequence": {
                    "packing_strategy": packing_strategy,
                    "packed_sequence_size": packed_sequence_size,
                },
            }
        )
    )


@pytest.mark.parametrize("packed_sequence_size", [512, 768, 4096])
def test_train_ft_accepts_aligned_mok_thd_extent(packed_sequence_size: int) -> None:
    _validate_mok_thd_packing(_recipe_config(dispatcher="mok", packed_sequence_size=packed_sequence_size))


@pytest.mark.parametrize("packed_sequence_size", [1, 511, 513, 1025])
def test_train_ft_rejects_unaligned_mok_thd_extent(packed_sequence_size: int) -> None:
    with pytest.raises(ValueError, match="at least 512 and divisible by 256"):
        _validate_mok_thd_packing(_recipe_config(dispatcher="mok", packed_sequence_size=packed_sequence_size))


def test_train_ft_skips_mok_extent_for_unpacked_and_other_dispatchers() -> None:
    _validate_mok_thd_packing(_recipe_config(dispatcher="mok", packed_sequence_size=0))
    _validate_mok_thd_packing(_recipe_config(dispatcher="hybridep", packed_sequence_size=513))
    _validate_mok_thd_packing(_recipe_config(dispatcher="mok", packed_sequence_size=513, packing_strategy="neat"))


def test_train_ft_setup_validates_mok_extent_before_distributed_init(monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_distributed = MagicMock()
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.initialize_distributed", initialize_distributed)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(_recipe_config(dispatcher="mok", packed_sequence_size=513))

    with pytest.raises(ValueError, match="at least 512 and divisible by 256"):
        recipe.setup()

    initialize_distributed.assert_not_called()

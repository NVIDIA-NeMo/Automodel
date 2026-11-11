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

"""CPU checks of the world-action fine-tuning recipe: its discovery and its seeding."""

from nemo_automodel.cli.utils import resolve_recipe_name
from nemo_automodel.recipes.wam.train import draw_seed


def test_recipe_is_discoverable_by_its_bare_name():
    assert resolve_recipe_name("TrainWAMFinetuneRecipe") == "nemo_automodel.recipes.wam.train.TrainWAMFinetuneRecipe"


def test_draw_seed_separates_ranks_micro_steps_and_steps():
    seeds = {draw_seed(3, step, micro, rank, 8) for step in range(3) for micro in range(6) for rank in range(8)}
    assert len(seeds) == 3 * 6 * 8
    assert draw_seed(3, 0, 0, 0, 8) == 3 * 1_000_003

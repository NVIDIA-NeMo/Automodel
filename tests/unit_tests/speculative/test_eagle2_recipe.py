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

from nemo_automodel.cli.utils import resolve_recipe_name
from nemo_automodel.recipes.llm.train_eagle1 import TrainEagle1Recipe
from nemo_automodel.recipes.llm.train_eagle2 import TrainEagle2Recipe


def test_eagle2_recipe_resolves_and_inherits_eagle1():
    assert issubclass(TrainEagle2Recipe, TrainEagle1Recipe)
    assert resolve_recipe_name("TrainEagle2Recipe") == "nemo_automodel.recipes.llm.train_eagle2.TrainEagle2Recipe"

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-node H100 Hy4-flash recipes must carry the NCCL setting the Engram row-owner exchange needs.

With the container's default NCCL network plugin and PXN both active, the Engram ID all-to-all receives corrupted IDs
from ranks on another node (issue #3959; reproduced for Hy4-flash on 2 x 8 H100 with the first Engram layer, and gone
with ``NCCL_PXN_DISABLE=1``). The exchange itself cannot be exercised in a unit test, so this guards the declaration.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
RECIPES = sorted(str(path.relative_to(REPO)) for path in (REPO / "examples").glob("*/hy_v4_flash/*.yaml"))


def test_every_recipe_directory_is_covered() -> None:
    assert {recipe.split("/")[1] for recipe in RECIPES} == {"llm_benchmark", "llm_finetune", "vlm_finetune"}
    assert len(RECIPES) == 5


@pytest.mark.parametrize("recipe", RECIPES)
def test_multi_node_h100_recipe_disables_pxn(recipe: str) -> None:
    with (REPO / recipe).open() as handle:
        ci = yaml.safe_load(handle)["ci"]
    assert ci["cluster_tag"] == "h100" and ci["nodes"] > 1
    assert ci["env_vars"]["NCCL_PXN_DISABLE"] == "1"

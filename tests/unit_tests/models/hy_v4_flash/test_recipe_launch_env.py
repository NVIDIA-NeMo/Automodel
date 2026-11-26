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

"""Multi-node H100 Hy4-flash recipes must carry ``NCCL_PXN_DISABLE=1`` (issue #3959).

Without it the Engram ID all-to-all receives corrupted IDs from ranks on another node. The exchange cannot be
exercised in a unit test, so this guards the declaration.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
RECIPES = sorted(str(path.relative_to(REPO)) for path in (REPO / "examples").glob("*/hy_v4_flash/*.yaml"))


def test_every_recipe_directory_is_covered() -> None:
    assert {recipe.split("/")[1] for recipe in RECIPES} == {"llm_finetune", "llm_pretrain", "vlm_finetune"}
    assert len(RECIPES) == 5


@pytest.mark.parametrize("recipe", RECIPES)
def test_multi_node_h100_recipe_disables_pxn(recipe: str) -> None:
    with (REPO / recipe).open() as handle:
        ci = yaml.safe_load(handle)["ci"]
    assert ci["cluster_tag"] == "h100" and ci["nodes"] > 1
    assert ci["env_vars"]["NCCL_PXN_DISABLE"] == "1"


@pytest.mark.parametrize("recipe", RECIPES)
def test_every_hub_reference_is_pinned_to_one_commit(recipe: str) -> None:
    def references(node: object) -> list[dict[str, object]]:
        if isinstance(node, dict):
            own = [node] if node.get("pretrained_model_name_or_path") == "tencent/Hy4-flash" else []
            return own + [found for value in node.values() for found in references(value)]
        return [found for value in node for found in references(value)] if isinstance(node, list) else []

    with (REPO / recipe).open() as handle:
        config = yaml.safe_load(handle)
    if config["dataset"]["_target_"].endswith("ChatDataset"):  # its training ids come from this tokenizer
        assert config["dataset"]["tokenizer"]["pretrained_model_name_or_path"] == "tencent/Hy4-flash"
    found = references(config)
    revisions = {ref.get("revision") for ref in found}
    # One full commit id for the weights, the config, the processor and the tokenizer: never a moving branch.
    assert found and len(revisions) == 1 and re.fullmatch(r"[0-9a-f]{40}", str(revisions.pop()))

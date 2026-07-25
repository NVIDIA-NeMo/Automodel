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

from pathlib import Path

import pytest
import yaml

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.flow_matching.adapters.simple import SimpleAdapter
from nemo_automodel.recipes._typed_config import RecipeConfig


def test_target_adapter_config_builds_typed_adapter():
    """The canonical adapter target is resolved at the recipe boundary."""
    config = RecipeConfig(
        ConfigNode(
            {
                "flow_matching": {
                    "adapter": {"_target_": "nemo_automodel.components.flow_matching.adapters.simple.SimpleAdapter"},
                    "timestep_sampling": "beta",
                }
            }
        )
    ).flow_matching

    assert isinstance(config.build_adapter(), SimpleAdapter)
    assert config.adapter_type is None
    assert config.timestep_sampling == "beta"


def test_legacy_adapter_type_remains_supported():
    """Existing recipes can continue to select adapters by name."""
    config = RecipeConfig(ConfigNode({"flow_matching": {"adapter_type": "simple"}})).flow_matching

    assert isinstance(config.build_adapter(), SimpleAdapter)


def test_target_and_legacy_adapter_are_mutually_exclusive():
    """Ambiguous adapter construction is rejected early."""
    config = ConfigNode(
        {
            "flow_matching": {
                "adapter": {"_target_": "nemo_automodel.components.flow_matching.adapters.simple.SimpleAdapter"},
                "adapter_type": "simple",
            }
        }
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        RecipeConfig(config).flow_matching


def test_qwen_image_edit_recipe_uses_canonical_typed_targets():
    """The shipped Qwen edit recipe resolves both typed component configs."""
    recipe_path = (
        Path(__file__).resolve().parents[3] / "examples" / "diffusion" / "finetune" / "qwen_image_edit_2511_flow.yaml"
    )
    config = RecipeConfig(ConfigNode(yaml.safe_load(recipe_path.read_text(encoding="utf-8"))))

    assert config.flow_matching.adapter is not None
    assert config.flow_matching.adapter_type is None
    assert type(config.flow_matching.build_adapter()).__name__ == "QwenImageEditAdapter"
    assert type(config.diffusion_dataloader).__name__ == "ImageEditDataloaderConfig"

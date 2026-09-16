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

from pathlib import Path

import pytest
import yaml

from nemo_automodel.components.datasets.llm.formatting_utils import GENERATION_REGEX

REPO_ROOT = Path(__file__).resolve().parents[4]
OLMO_CONFIGS = (
    "olmo_2_0425_1b_instruct_squad.yaml",
    "olmo_2_0425_1b_instruct_squad_peft.yaml",
)


@pytest.mark.parametrize("config_name", OLMO_CONFIGS)
def test_olmo_datasets_use_generation_marked_chat_template(config_name):
    config_path = REPO_ROOT / "examples" / "llm_finetune" / "olmo" / config_name
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    train_template = config["dataset"]["chat_template"]
    assert config["validation_dataset"]["chat_template"] == train_template

    template_path = REPO_ROOT / train_template
    assert template_path.is_file()
    assert GENERATION_REGEX.search(template_path.read_text(encoding="utf-8"))

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

REPO_ROOT = Path(__file__).resolve().parents[4]
BAICHUAN_CONFIGS = (
    "baichuan_2_7b_mock_fp8.yaml",
    "baichuan_2_7b_squad.yaml",
    "baichuan_2_7b_squad_peft.yaml",
)


@pytest.mark.parametrize("config_name", BAICHUAN_CONFIGS)
def test_baichuan_recipes_preserve_special_token_insertion(config_name: str) -> None:
    config_path = REPO_ROOT / "examples" / "llm_finetune" / "baichuan" / config_name
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["tokenizer"]["add_bos_token"] is True
    assert config["tokenizer"]["add_eos_token"] is True

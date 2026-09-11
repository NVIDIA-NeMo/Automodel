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

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples" / "retrieval" / "distillation" / "nemotron3_embed_1b_distill.yaml"


def test_nemotron3_embedding_recipe_preserves_transformers_auto_policy() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    tokenizer_config = config["tokenizer"]

    assert tokenizer_config["_target_"] == "nemo_automodel.NeMoAutoTokenizer.from_pretrained"
    assert tokenizer_config["tokenizer_backend"] == "transformers_auto"
    assert "force_hf" not in tokenizer_config
    assert "add_bos_token" not in tokenizer_config
    assert tokenizer_config["add_eos_token"] is False

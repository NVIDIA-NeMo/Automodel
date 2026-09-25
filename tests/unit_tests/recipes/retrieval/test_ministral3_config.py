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
CONFIG_PATH = REPO_ROOT / "examples" / "retrieval" / "bi_encoder" / "ministral3_3b_instruct.yaml"
EXPECTED_CHECKPOINT = "mistralai/Ministral-3-3B-Instruct-2512-BF16"


def test_ministral3_recipe_uses_bf16_checkpoint() -> None:
    """The shipped training recipe must use the unquantized checkpoint."""
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert raw_config["model"]["pretrained_model_name_or_path"] == EXPECTED_CHECKPOINT
    assert raw_config["tokenizer"]["pretrained_model_name_or_path"] == EXPECTED_CHECKPOINT


def test_ministral3_tokenizer_config_uses_tokenizers_backend_policy() -> None:
    """The shipped recipe must use the explicit serialized-tokenizer policy."""
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    tokenizer_config = raw_config["tokenizer"]

    assert tokenizer_config["_target_"] == "nemo_automodel.NeMoAutoTokenizer.from_pretrained"
    assert tokenizer_config["tokenizer_backend"] == "tokenizers"
    assert tokenizer_config["fix_mistral_regex"] is True
    assert tokenizer_config["split_special_tokens"] is True
    assert tokenizer_config["add_bos_token"] is True
    assert tokenizer_config["add_eos_token"] is False
    assert "force_hf" not in tokenizer_config

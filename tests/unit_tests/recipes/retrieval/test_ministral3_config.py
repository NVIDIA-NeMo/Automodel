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
from unittest.mock import patch

import yaml

from nemo_automodel._transformers.tokenization.registry import TokenizerRegistry
from nemo_automodel.components.config.loader import ConfigNode

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples" / "retrieval" / "bi_encoder" / "ministral3_3b_instruct.yaml"


class _Mistral3Config:
    model_type = "mistral3"


class _FakeMistralTokenizer:
    pad_token_id = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *, padding_side: str):
        assert pretrained_model_name_or_path == "mistralai/Ministral-3-3B-Instruct-2512"
        assert padding_side == "left"
        return cls()


def test_ministral3_tokenizer_config_uses_registered_backend() -> None:
    """The shipped recipe must avoid unsupported HF Mistral tokenizer kwargs."""
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    tokenizer_config = ConfigNode(raw_config["tokenizer"])

    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=_Mistral3Config()),
        patch.object(TokenizerRegistry, "get_custom_tokenizer_cls", return_value=_FakeMistralTokenizer),
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=AssertionError("the recipe must use AutoModel's registered Mistral tokenizer"),
        ),
    ):
        tokenizer = tokenizer_config.instantiate()

    assert isinstance(tokenizer, _FakeMistralTokenizer)

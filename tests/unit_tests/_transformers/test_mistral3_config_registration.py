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

"""Canonical config loading must not depend on incidental model import registration."""

from pathlib import Path

import pytest
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel._transformers.registry import resolve_custom_config_cls
from nemo_automodel.components.models.ministral_bidirectional.model import Mistral3BidirectionalConfig


@pytest.mark.parametrize("registered_with_transformers", [False, True])
def test_mistral3_checkpoint_config_resolves_without_incidental_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered_with_transformers: bool
) -> None:
    config = Mistral3BidirectionalConfig(
        text_config={"model_type": "ministral3", "hidden_size": 16, "pooling": "cls"},
        vision_config={"model_type": "pixtral", "hidden_size": 16},
        pooling="cls",
        temperature=0.5,
    )
    config.save_pretrained(tmp_path)
    if registered_with_transformers:
        assert isinstance(AutoConfig.from_pretrained(tmp_path, trust_remote_code=False), Mistral3BidirectionalConfig)
    else:
        monkeypatch.delitem(CONFIG_MAPPING._extra_content, "mistral3_bidirec", raising=False)
    assert resolve_custom_config_cls("mistral3_bidirec") is Mistral3BidirectionalConfig
    loaded = get_hf_config(str(tmp_path), "eager", trust_remote_code=False)
    assert isinstance(loaded, Mistral3BidirectionalConfig)
    assert loaded.pooling == "cls"
    assert loaded.text_config.pooling == "cls"
    assert loaded.temperature == 0.5

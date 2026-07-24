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

import pytest
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.inkling.configuration import InklingConfig


def test_inkling_config_registered_with_auto_config():
    assert CONFIG_MAPPING["inkling_mm_model"] is InklingConfig

    cfg = AutoConfig.for_model("inkling_mm_model", architectures=["InklingForConditionalGeneration"])

    assert isinstance(cfg, InklingConfig)
    assert cfg.model_type == "inkling_mm_model"
    assert cfg.text_config.model_type == "inkling_text"


def test_inkling_architecture_stays_registered_when_hf_inkling_is_unavailable():
    from nemo_automodel.components.models.inkling.model import _INKLING_HF_AVAILABLE, InklingForConditionalGeneration
    from nemo_automodel.shared.import_utils import UnavailableError

    assert ModelRegistry.has_custom_model("InklingForConditionalGeneration")
    if _INKLING_HF_AVAILABLE:
        pytest.skip("This host's transformers build ships Inkling.")

    with pytest.raises(UnavailableError, match="transformers >= 5.14"):
        InklingForConditionalGeneration.from_config(InklingConfig())

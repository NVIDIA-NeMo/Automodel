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

"""Mistral3 VLM routing must not depend on unrelated import side effects."""

from types import SimpleNamespace

import pytest

from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config


@pytest.mark.parametrize("quantization_config", [None, {"quant_method": "fp8"}])
def test_mistral3_vlm_resolves_to_custom_model(quantization_config):
    config = SimpleNamespace(
        architectures=["Mistral3ForConditionalGeneration"],
        text_config=SimpleNamespace(model_type="ministral3"),
        quantization_config=quantization_config,
    )

    model_cls = _resolve_custom_model_cls_for_config(config)

    assert model_cls is not None
    assert model_cls.__module__ == "nemo_automodel.components.models.mistral3_vlm.model"
    assert model_cls.__name__ == "Mistral3FP8VLMForConditionalGeneration"

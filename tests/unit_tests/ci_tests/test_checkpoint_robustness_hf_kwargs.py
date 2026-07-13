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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import _hf_source_load_kwargs


@pytest.mark.parametrize(
    ("model_type", "expected_attn_implementation"),
    [("nemotron_h", "sdpa"), ("nemotron_flash", "flash_attention_2")],
)
def test_remote_code_attention_implementation(model_type, expected_attn_implementation):
    with patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type=model_type),
    ) as from_pretrained:
        hf_kwargs = _hf_source_load_kwargs(
            {"revision": "model-revision", "token": "model-token"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["attn_implementation"] == expected_attn_implementation
    from_pretrained.assert_called_once_with(
        "model-path",
        trust_remote_code=True,
        revision="model-revision",
        token="model-token",
    )


def test_explicit_attention_implementation_is_preserved():
    with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("must not probe config")):
        hf_kwargs = _hf_source_load_kwargs(
            {"attn_implementation": "eager"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["attn_implementation"] == "eager"

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

"""LLM and VLM construction preserve the same packed-input backend policy."""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel._transformers import NeMoAutoModelForCausalLM, NeMoAutoModelForImageTextToText
from nemo_automodel._transformers.kernel_patches import _apply_preload_overrides
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.llm import train_ft
from nemo_automodel.recipes.vlm import finetune


@pytest.mark.parametrize(
    "recipe,wrapper", [(train_ft, NeMoAutoModelForCausalLM), (finetune, NeMoAutoModelForImageTextToText)]
)
@pytest.mark.parametrize("backend", ["sdpa", "flash_attention_2"])
def test_recipe_preserves_backend_and_reports_packed_inputs(recipe, wrapper, backend, monkeypatch):
    # Replace expensive weight/device construction; exercise recipe config dispatch
    # and the real preload policy that previously replaced SDPA with Flash.
    def load_model(*, attn_implementation, has_packed_sequence, **kwargs):
        assert has_packed_sequence is True
        resolved, _ = _apply_preload_overrides(1, 1, has_packed_sequence, attn_implementation, False)
        model = torch.nn.Module()
        model.config = SimpleNamespace(_attn_implementation=resolved)
        return model

    monkeypatch.setattr(wrapper, "from_pretrained", load_model)
    config = ConfigNode({"_target_": wrapper.from_pretrained, "attn_implementation": backend})
    result = recipe.build_model(
        cfg_model=config,
        cfg_peft=None,
        cfg_freeze=None,
        seed=42,
        has_packed_sequence=True,
    )
    assert result.config._attn_implementation == backend

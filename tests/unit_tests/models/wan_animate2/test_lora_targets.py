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

"""Check the shipped LoRA recipe against the real released transformer."""

from pathlib import Path

from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.config.loader import load_yaml_config


def test_recipe_trains_only_self_attention_input_adapters(tiny_model):
    """The recipe targets every self-attention Q/K/V and freezes base weights."""
    recipe = Path(__file__).resolve().parents[4] / "examples/diffusion/finetune/wan_animate2_flow_lora.yaml"
    config = load_yaml_config(recipe)
    count = apply_lora_to_linear_modules(tiny_model, PeftConfig.from_dict(config.peft.to_dict()))
    expected = {f"blocks.{i}.self_attn.to_{projection}" for i in range(2) for projection in ("q", "k", "v")}
    patched = {name for name, module in tiny_model.named_modules() if isinstance(module, LinearLoRA)}
    assert count == 6
    assert patched == expected
    trainable = {name for name, parameter in tiny_model.named_parameters() if parameter.requires_grad}
    assert trainable == {f"{name}.lora_{side}.weight" for name in expected for side in ("A", "B")}


def test_old_fork_targets_do_not_match_the_released_model(tiny_model):
    """Legacy recipes would freeze all weights without injecting an adapter."""
    count = apply_lora_to_linear_modules(
        tiny_model, PeftConfig(target_modules=["*.self_attn.q", "*.self_attn.k", "*.self_attn.v"])
    )
    assert count == 0
    assert not any(parameter.requires_grad for parameter in tiny_model.parameters())

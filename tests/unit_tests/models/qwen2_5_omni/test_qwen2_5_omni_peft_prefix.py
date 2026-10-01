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

"""PEFT save/resume prefix handling for the Qwen2.5 Omni adapter (CPU)."""

from types import SimpleNamespace

import torch

from nemo_automodel.components.models.qwen2_5_omni.state_dict_adapter import Qwen2_5OmniStateDictAdapter


def _adapter():
    return Qwen2_5OmniStateDictAdapter(config=SimpleNamespace())


def _peft_lora_state_dict(rank=4, dim=32):
    base = "base_model.model.model.layers.0"
    return {
        f"{base}.self_attn.q_proj.lora_A.weight": torch.randn(rank, dim),
        f"{base}.self_attn.q_proj.lora_B.weight": torch.randn(dim, rank),
        f"{base}.mlp.gate_proj.lora_A.weight": torch.randn(rank, dim),
    }


def test_peft_lora_keys_get_thinker_inside_the_peft_prefix():
    """The thinker namespace must land inside the PEFT prefix.

    Before the fix every key got "thinker." prepended on the outside,
    producing ``thinker.base_model.model...`` names that HF PEFT cannot
    attach.
    """
    adapter = _adapter()
    out = adapter.to_hf(_peft_lora_state_dict())

    assert out, "no keys came back from to_hf"
    for key in out:
        assert key.startswith("base_model.model.thinker.model.layers."), key


def test_convert_single_tensor_moves_thinker_inside_the_peft_prefix():
    adapter = _adapter()
    tensor = torch.randn(4, 32)
    result = adapter.convert_single_tensor_to_hf(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", tensor
    )

    assert result == [("base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.weight", tensor)]


def test_full_weights_still_get_the_thinker_prefix():
    adapter = _adapter()
    out = adapter.to_hf({"model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32)})

    assert list(out) == ["thinker.model.layers.0.self_attn.q_proj.weight"]


def test_peft_lora_save_round_trips_for_resume():
    """from_hf must rebuild the exact keys ModelState expects on resume."""
    adapter = _adapter()
    sd = _peft_lora_state_dict()
    back = adapter.from_hf(adapter.to_hf(sd))

    assert set(back) == set(sd)
    for key in sd:
        torch.testing.assert_close(back[key], sd[key])


def test_correctly_named_external_adapter_imports():
    """An adapter written in the proper HF PEFT layout must load.

    Before the fix from_hf only stripped a leading "thinker.", so these
    keys passed through unchanged and never matched a model parameter.
    """
    adapter = _adapter()
    tensor = torch.randn(4, 32)
    back = adapter.from_hf({"base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.weight": tensor})

    assert list(back) == ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]


def test_legacy_malformed_saves_still_resume():
    """Adapters saved before the fix carry the outer thinker prefix."""
    adapter = _adapter()
    tensor = torch.randn(4, 32)
    back = adapter.from_hf({"thinker.base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": tensor})

    assert list(back) == ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]


def test_talker_keys_dropped_in_both_positions():
    """talker/token2wav weights are dropped whether bare (full checkpoints)
    or nested inside the peft prefix (external full-omni adapters)."""
    adapter = _adapter()
    back = adapter.from_hf(
        {
            "thinker.model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
            "talker.model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32),
            "base_model.model.talker.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 32),
        }
    )

    assert list(back) == ["model.layers.0.self_attn.q_proj.weight"]


def test_target_modules_get_the_thinker_namespace():
    """adapter_config.json target_modules must carry thinker. so PEFT's
    suffix matching on the full omni model doesn't also hit the talker's
    structurally identical submodules."""
    adapter = _adapter()

    assert (
        adapter.map_peft_target_module_to_hf("model.layers.0.self_attn.q_proj")
        == "thinker.model.layers.0.self_attn.q_proj"
    )

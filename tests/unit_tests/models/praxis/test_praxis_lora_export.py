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

"""CPU checks of the LoRA merge and the checkpoint writer."""

import json

import torch
from safetensors.torch import save_file

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.models.praxis.config import PraxisConfig
from nemo_automodel.components.models.praxis.lora_export import export_merged, merge_lora, save_checkpoint
from nemo_automodel.components.models.praxis.model import PraxisPolicy
from tests.unit_tests.models.praxis.test_praxis_model import TINY_CONFIG, cpu_backend

LORA_TARGETS = ["*backbone.*.ffn.*", "*action_expert.blocks.*.to_q", "*action_expert.blocks.*.mlp_in"]


def _tiny_policy(seed: int = 0) -> PraxisPolicy:
    torch.manual_seed(seed)
    model = PraxisPolicy(PraxisConfig.from_json(TINY_CONFIG), cpu_backend())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    return model


def _assert_state_equal(policy, state):
    actual = policy.state_dict()
    for key, value in state.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)


def test_merge_lora_folds_linear_and_expert_adapters() -> None:
    w = torch.randn(6, 4)
    a = torch.randn(2, 4)
    b = torch.randn(6, 2)
    experts = torch.randn(3, 5, 4)
    ea = torch.randn(3, 5, 2)
    eb = torch.randn(3, 2, 4)
    state = {
        "m.weight": w.clone(),
        "m.lora_A.weight": a,
        "m.lora_B.weight": b,
        "x.experts.gate_and_up_projs": experts.clone(),
        "x.experts.lora_gate_and_up_A": ea,
        "x.experts.lora_gate_and_up_B": eb,
        "other": torch.ones(1),
    }
    merged = merge_lora(state, scale=2.0)
    assert set(merged) == {"m.weight", "x.experts.gate_and_up_projs", "other"}
    assert torch.allclose(merged["m.weight"], w + 2.0 * (b @ a))
    assert torch.allclose(merged["x.experts.gate_and_up_projs"], experts + 2.0 * torch.bmm(ea, eb))


def test_save_checkpoint_round_trips_through_from_pretrained(tmp_path):
    model = _tiny_policy()
    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_checkpoint(tmp_path / "ckpt", state, model.state_dict_adapter, tmp_path / "config.json")
    assert (tmp_path / "ckpt" / "SHA256SUMS").exists()
    assert json.loads((tmp_path / "ckpt" / "config.json").read_text()) == TINY_CONFIG
    reloaded = PraxisPolicy.from_pretrained(tmp_path / "ckpt", backend=cpu_backend())
    _assert_state_equal(reloaded, state)


def test_export_merged_folds_the_saved_adapters(tmp_path):
    model = _tiny_policy()
    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_checkpoint(tmp_path / "base", base_state, model.state_dict_adapter, tmp_path / "config.json")

    peft = PeftConfig(
        target_modules=LORA_TARGETS, dim=2, alpha=4, lora_dtype=torch.float32, use_memory_efficient_lora=False
    )
    assert apply_lora_to_linear_modules(model, peft) > 0
    assert any("lora_gate_and_up_A" in name for name, _ in model.named_parameters()), (
        "the MoE blocks get expert adapters"
    )
    torch.manual_seed(1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_" in name:
                parameter.normal_()
    with_adapters = {k: v.detach().clone() for k, v in model.state_dict().items()}
    adapters_dir = tmp_path / "step" / "model"
    adapters_dir.mkdir(parents=True)
    save_file(
        {k: v.contiguous() for k, v in with_adapters.items() if ".lora_" in k},
        str(adapters_dir / "adapter_model.safetensors"),
    )
    (adapters_dir / "automodel_peft_config.json").write_text(
        json.dumps({"target_modules": LORA_TARGETS, "lora_dtype": "torch.float32", "use_memory_efficient_lora": False})
    )
    (adapters_dir / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))

    export_merged(tmp_path / "base", adapters_dir, tmp_path / "merged")
    merged = PraxisPolicy.from_pretrained(tmp_path / "merged", backend=cpu_backend())
    _assert_state_equal(merged, merge_lora(with_adapters, peft.alpha / peft.dim))
    assert not any(".lora_" in k for k in merged.state_dict())

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

"""LoRA adapter tensors in the ERNIE 5.5 checkpoint adapter and through the Checkpointer."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.ernie5_5.state_dict_adapter import _is_lora_key
from tests.unit_tests.models.ernie5_5.test_lora import _recipe_targets
from tests.unit_tests.models.ernie5_5.test_state_dict_adapter import _tiny_model

pytestmark = pytest.mark.cpu_only


def _adapter_state():
    return {
        "base_model.model.model.layers.0.attention.wq_a.lora_A.weight": torch.zeros(2, 16),
        "base_model.model.model.layers.0.attention.wq_a.lora_B.weight": torch.zeros(6, 2),
    }


def test_recipe_adapter_names_are_recognized_and_base_names_are_not():
    model = _tiny_model()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=_recipe_targets(), dim=2, alpha=4))
    state = model.state_dict()
    lora = [name for name in state if ".lora_" in name]
    base = [name for name in state if ".lora_" not in name]
    assert lora and base
    assert all(_is_lora_key(name) and _is_lora_key("base_model.model." + name) for name in lora)
    assert not any(_is_lora_key(name) for name in base)


def test_adapter_tensors_pass_through_a_save_and_a_load_without_a_session():
    adapter = _tiny_model().state_dict_adapter
    state = _adapter_state()
    for converted in (adapter.to_hf(state), adapter.to_hf(state, for_checkpoint_load=True), adapter.from_hf(state)):
        assert converted.keys() == state.keys()
        assert all(converted[name] is value for name, value in state.items())
    # A base save still works: no session was opened.
    model = _tiny_model()
    assert model.state_dict_adapter.to_hf(model.state_dict())


def _lora_model(target: str = "model.layers.0.attention.wq_a"):
    model = _tiny_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=[target], dim=2, alpha=4, use_memory_efficient_lora=False)
    )
    model.ep_size = 2  # Select EP install path in this single-process test, not an actual EP mesh.
    return model


def test_mixed_key_planning_preserves_base_and_adapter_keys():
    """Preserve both base HF keys and installed adapter keys in pre-sharding planning."""
    plain = _tiny_model()
    base_keys = plain.state_dict_adapter.get_hf_state_dict_keys(plain.state_dict())
    model = _lora_model()
    state = model.state_dict()
    lora_keys = [name for name in state if ".lora_" in name]
    assert lora_keys
    plan = model.state_dict_adapter.get_hf_state_dict_keys(state)
    assert plan == base_keys + lora_keys
    assert list(model.state_dict_adapter.to_hf(state)) == plan


def test_mixed_key_planning_load_is_rejected_without_an_open_session():
    """A mixed base/adapter state is valid for planning but not for a checkpoint load."""
    model = _lora_model()
    state = model.state_dict()
    with pytest.raises(ValueError):
        model.state_dict_adapter.to_hf(state, for_checkpoint_load=True)
    assert model.state_dict_adapter.get_hf_state_dict_keys(state)


@pytest.mark.parametrize("ep", [False, True], ids=["dcp_install", "ep_install"])
def test_checkpointer_saves_twice_and_restores_adapter(tmp_path, ep):
    """PEFT saves leave no session open and their tensors restore on both install paths."""
    source, target = _lora_model(), _lora_model()
    if not ep:
        del source.ep_size
        del target.ep_size
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if ".lora_" in name:
                parameter.fill_(3)
        for name, parameter in target.named_parameters():
            if ".lora_" in name:
                parameter.zero_()
    expected = {name: value.detach().clone() for name, value in source.named_parameters() if ".lora_" in name}
    base_before = {name: value.detach().clone() for name, value in target.state_dict().items() if ".lora_" not in name}
    assert expected
    config = PeftConfig(target_modules=["model.layers.0.attention.wq_a"], dim=2, alpha=4)
    checkpointer = Checkpointer(
        CheckpointingConfig(is_peft=True, save_consolidated=False, checkpoint_dir=str(tmp_path / "run")), 0, 0, 0
    )
    try:
        for name in ("first", "second"):
            checkpointer.save_model(source, str(tmp_path / name), peft_config=config)
            stored = load_file(str(tmp_path / name / "model" / "adapter_model.safetensors"))
            assert stored.keys() == {f"base_model.model.{key}" for key in expected}
            assert all(torch.equal(stored[f"base_model.model.{key}"], value) for key, value in expected.items())
        checkpointer.load_model(target, str(tmp_path / "second" / "model"), is_init_step=False)
        assert all(torch.equal(target.get_parameter(name), value) for name, value in expected.items())
        assert all(torch.equal(target.state_dict()[name], value) for name, value in base_before.items())
        checkpointer.save_model(target, str(tmp_path / "third"), peft_config=config)
        third = load_file(str(tmp_path / "third" / "model" / "adapter_model.safetensors"))
        assert all(torch.equal(third[f"base_model.model.{key}"], value) for key, value in expected.items())
        with torch.no_grad():
            for name in expected:
                target.get_parameter(name).zero_()
        checkpointer.load_model(target, str(tmp_path / "first" / "model"), is_init_step=False)
        assert all(torch.equal(target.get_parameter(name), value) for name, value in expected.items())
        assert all(torch.equal(target.state_dict()[name], value) for name, value in base_before.items())
    finally:
        checkpointer.close()


@pytest.mark.parametrize("mode", ["valid", "missing", "extra", "bad_shape", "one_expert"])
def test_checkpointer_restores_only_an_adapter_checkpoint_with_the_keys_and_shapes_of_the_model(tmp_path, mode):
    """The expert-parallel install path copies adapter tensors without a check of its own."""
    model = _lora_model("model.layers.1.moe.experts" if mode == "one_expert" else "model.layers.0.attention.wq_a")
    native = {name: value for name, value in model.named_parameters() if ".lora_" in name}
    before = {name: value.detach().clone() for name, value in native.items()}
    values = {f"base_model.model.{name}": torch.full_like(value, 3) for name, value in native.items()}
    last = sorted(values)[-1]
    if mode == "missing":
        values.pop(last)
    elif mode == "extra":
        values["base_model.model.model.layers.0.attention.fake.lora_A.weight"] = torch.ones(1)
    elif mode == "bad_shape":
        values[last] = torch.ones(1)
    elif mode == "one_expert":
        # One expert would broadcast into the adapter slots of all experts.
        values[last] = values[last][:1].clone()
    model_path = tmp_path / "model"
    model_path.mkdir()
    save_file(values, str(model_path / "adapter_model.safetensors"))
    checkpointer = Checkpointer(
        CheckpointingConfig(is_peft=True, save_consolidated=False, checkpoint_dir=str(tmp_path)), 0, 0, 0
    )
    try:
        if mode == "valid":
            checkpointer.load_model(model, str(model_path), is_init_step=False)
        else:
            with pytest.raises(ValueError, match="LoRA checkpoint and model differ"):
                checkpointer.load_model(model, str(model_path), is_init_step=False)
    finally:
        checkpointer.close()
    changed = [not torch.equal(before[name], value) for name, value in native.items()]
    assert all(changed) if mode == "valid" else not any(changed)

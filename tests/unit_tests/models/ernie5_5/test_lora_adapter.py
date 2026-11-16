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

"""ERNIE LoRA key and restore contracts, including Checkpointer integration."""

import copy

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


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.attention.wq_a.lora_A.weight",
        "base_model.model.model.layers.0.attention.wq_a.lora_B.weight",
        "model.layers.1.moe.experts.lora_gate_and_up_A",
        "base_model.model.model.layers.2.moe.experts.lora_down_B",
    ],
)
def test_supported_lora_key_formats(key):
    """Recognize the native and PEFT-prefixed supported LoRA formats."""
    assert _is_lora_key(key)


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.attention.wq_a.lora_A.foo",
        "model.layers.0.attention.wq_a.lora_magnitude_vector.weight",
        "model.layers.1.moe.experts.lora_down_C",
        "model.layers.1.moe.not_experts.lora_down_A",
        "model..lora_A.weight",
        "other.lora_A.weight",
    ],
)
def test_unsupported_lora_key_is_rejected(key):
    """Reject unsupported or malformed LoRA names."""
    adapter = _tiny_model().state_dict_adapter
    assert not _is_lora_key(key)
    with pytest.raises(ValueError, match="Unsupported LoRA state key"):
        adapter.to_hf({key: torch.ones(1)}, for_checkpoint_load=True)


def test_recipe_adapter_names_are_accepted_and_base_names_are_not():
    """The supported recipe targets must agree with the adapter's key grammar."""
    model = _tiny_model()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=_recipe_targets(), dim=2, alpha=4))
    state = model.state_dict()
    lora = [name for name in state if ".lora_" in name]
    base = [name for name in state if ".lora_" not in name]
    assert lora and base
    assert all(_is_lora_key(name) and _is_lora_key("base_model.model." + name) for name in lora)
    assert not any(_is_lora_key(name) for name in base)


def test_dora_key_planning_names_the_unsupported_option():
    """Report DoRA explicitly during the normal pre-sharding key plan."""
    model = _tiny_model()
    apply_lora_to_linear_modules(
        model,
        PeftConfig(target_modules=["model.layers.0.attention.wq_a"], dim=2, alpha=4, use_dora=True),
    )
    with pytest.raises(ValueError, match=r"DoRA.*peft\.use_dora"):
        model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())


def test_dora_magnitude_vector_names_the_unsupported_option():
    """Report DoRA explicitly for the checkpoint-normalized magnitude key."""
    adapter = _tiny_model().state_dict_adapter
    key = "base_model.model.model.layers.0.attention.wq_a.lora_magnitude_vector"
    with pytest.raises(ValueError, match=r"DoRA.*peft\.use_dora"):
        adapter.to_hf({key: torch.ones(6)})


def test_other_magnitude_prefix_is_not_misidentified_as_dora():
    """Keep unknown names distinct from the exact DoRA magnitude names."""
    adapter = _tiny_model().state_dict_adapter
    key = "model.layers.0.attention.wq_a.lora_magnitude_unrecognized"
    with pytest.raises(ValueError, match="Unsupported LoRA state key") as exc:
        adapter.to_hf({key: torch.ones(1)})
    assert "DoRA" not in str(exc.value)


def test_load_accepts_new_mapping_with_expected_contract():
    """Accept a reader-created mapping that matches the receiving model."""
    adapter = _tiny_model().state_dict_adapter
    expected = _adapter_state()
    adapter.to_hf(expected, for_checkpoint_load=True)
    actual = {name: torch.ones_like(value) for name, value in expected.items()}
    result = adapter.from_hf(actual)
    assert result.keys() == expected.keys()
    assert all(torch.equal(result[name], actual[name]) for name in expected)
    assert adapter.to_hf(expected).keys() == expected.keys()


def test_adapter_save_rejects_nontensor_values():
    """Reject a malformed adapter state before it reaches checkpoint storage."""
    adapter = _tiny_model().state_dict_adapter
    state = _adapter_state()
    state[next(iter(state))] = None
    with pytest.raises(ValueError, match="only tensors"):
        adapter.to_hf(state)


def test_adapter_save_does_not_open_a_load_session():
    """Saving an adapter must leave the adapter ready for a later load."""
    adapter = _tiny_model().state_dict_adapter
    state = _adapter_state()
    assert adapter.to_hf(state).keys() == state.keys()
    adapter.to_hf(state, for_checkpoint_load=True)
    assert adapter.from_hf(state).keys() == state.keys()


@pytest.mark.parametrize("change", ["missing", "extra", "shape", "same_numel_shape", "dtype", "nontensor"])
def test_failed_contract_validation_poisons_adapter(change):
    """Refuse future adapter use after a mismatched PEFT load."""
    adapter = _tiny_model().state_dict_adapter
    expected = _adapter_state()
    adapter.to_hf(expected, for_checkpoint_load=True)
    actual = {name: torch.ones_like(value) for name, value in expected.items()}
    keys = list(actual)
    if change == "missing":
        actual.pop(keys[1])
    elif change == "extra":
        actual["model.layers.0.foo.lora_A.weight"] = torch.ones(1)
    elif change == "shape":
        actual[keys[1]] = torch.ones(1)
    elif change == "same_numel_shape":
        actual[keys[1]] = actual[keys[1]].t().contiguous()
    elif change == "dtype":
        actual[keys[1]] = actual[keys[1]].half()
    else:
        actual[keys[1]] = None
    with pytest.raises(ValueError):
        adapter.from_hf(actual)
    repaired = {name: torch.ones_like(value) for name, value in expected.items()}
    with pytest.raises(RuntimeError):
        adapter.from_hf(repaired)
    with pytest.raises(RuntimeError):
        adapter.to_hf(expected)
    with pytest.raises(RuntimeError):
        copy.deepcopy(adapter).to_hf(expected)


def test_load_error_reports_missing_and_unexpected_names():
    """Tell users how the checkpoint's adapter targets differ from the model."""
    adapter = _tiny_model().state_dict_adapter
    expected = _adapter_state()
    adapter.to_hf(expected, for_checkpoint_load=True)
    actual = dict(expected)
    missing = next(iter(actual))
    del actual[missing]
    extra = "base_model.model.model.layers.0.attention.fake.lora_A.weight"
    actual[extra] = torch.ones(1)
    with pytest.raises(ValueError, match="LoRA checkpoint keys") as exc:
        adapter.from_hf(actual)
    message = str(exc.value)
    assert "1 missing" in message and missing in message
    assert "1 unexpected" in message and extra in message


@pytest.mark.parametrize("change", ["shape", "dtype", "nontensor"])
def test_load_error_reports_the_tensor_mismatch(change):
    """Distinguish bad shape, dtype and non-tensor values in restore errors."""
    adapter = _tiny_model().state_dict_adapter
    expected = _adapter_state()
    adapter.to_hf(expected, for_checkpoint_load=True)
    actual = {name: torch.ones_like(value) for name, value in expected.items()}
    name = next(iter(actual))
    if change == "shape":
        actual[name] = actual[name].t().contiguous()
    elif change == "dtype":
        actual[name] = actual[name].half()
    else:
        actual[name] = None
    with pytest.raises(ValueError, match="LoRA checkpoint tensor") as exc:
        adapter.from_hf(actual)
    message = str(exc.value)
    assert name in message
    if change == "nontensor":
        assert "not a tensor" in message and "NoneType" in message
    else:
        assert "checkpoint shape=" in message and "model shape=" in message
        assert str(expected[name].dtype) in message and str(actual[name].dtype) in message


def test_adapter_load_requires_a_session():
    """Reject adapter-only loads that have no model-provided contract."""
    adapter = _tiny_model().state_dict_adapter
    with pytest.raises(RuntimeError, match="matching load session"):
        adapter.from_hf(_adapter_state())


def test_foreign_lora_poisons_open_base_load():
    """Do not let a foreign LoRA mapping close a base load."""
    model = _tiny_model()
    adapter = model.state_dict_adapter
    destinations = adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
    with pytest.raises(RuntimeError):
        adapter.from_hf(_adapter_state())
    with pytest.raises(RuntimeError):
        adapter.from_hf(destinations)


def _lora_model():
    model = _tiny_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    apply_lora_to_linear_modules(
        model,
        PeftConfig(target_modules=["model.layers.0.attention.wq_a"], dim=2, alpha=4, use_memory_efficient_lora=False),
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


def test_grouped_expert_broadcastable_shape_is_rejected_before_install(tmp_path):
    """A checkpoint cannot broadcast one expert into all expert adapter slots."""
    model = _tiny_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["model.layers.1.moe.experts"], dim=2, alpha=4))
    model.ep_size = 2  # Select the EP installation path; this is not a distributed mesh.
    before = {name: value.detach().clone() for name, value in model.named_parameters() if ".lora_" in name}
    assert len(before) == 4
    values = {f"base_model.model.{name}": torch.full_like(value, 3) for name, value in before.items()}
    key = next(name for name in values if name.endswith(".lora_gate_and_up_A"))
    values[key] = values[key][:1].clone()
    model_path = tmp_path / "model"
    model_path.mkdir()
    save_file(values, str(model_path / "adapter_model.safetensors"))
    checkpointer = Checkpointer(
        CheckpointingConfig(is_peft=True, save_consolidated=False, checkpoint_dir=str(tmp_path)), 0, 0, 0
    )
    try:
        with pytest.raises(ValueError, match="LoRA checkpoint"):
            checkpointer.load_model(model, str(model_path), is_init_step=False)
        assert all(torch.equal(model.get_parameter(name), value) for name, value in before.items())
    finally:
        checkpointer.close()


@pytest.mark.parametrize("mode", ["valid", "missing", "extra", "bad_shape", "dtype"])
def test_checkpointer_validates_before_ep_parameter_install(tmp_path, mode):
    """Reject malformed PEFT tensors before installing them on the EP path."""
    model = _lora_model()
    native = {name: value for name, value in model.named_parameters() if value.requires_grad and ".lora_" in name}
    assert len(native) == 2
    before = {name: value.detach().clone() for name, value in native.items()}
    values = {f"base_model.model.{name}": torch.full_like(value, 3) for name, value in native.items()}
    keys = sorted(values)
    if mode == "missing":
        values.pop(keys[-1])
    elif mode == "extra":
        values["base_model.model.model.layers.0.attention.fake.lora_A.weight"] = torch.ones(1)
    elif mode == "bad_shape":
        values[keys[-1]] = torch.ones(1)
    elif mode == "dtype":
        values[keys[-1]] = values[keys[-1]].half()
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
            with pytest.raises(ValueError, match="LoRA checkpoint"):
                checkpointer.load_model(model, str(model_path), is_init_step=False)
    finally:
        checkpointer.close()
    changed = [not torch.equal(before[name], value) for name, value in native.items()]
    assert all(changed) if mode == "valid" else not any(changed)

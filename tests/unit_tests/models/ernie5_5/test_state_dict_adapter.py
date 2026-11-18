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

"""Unit tests for the ERNIE 5.5 state-dict adapter: key plan, unsharded conversions and the load-session lifecycle.

The last tests run the unsharded model in one process: the checkpointer and the public loader load a Hugging Face
checkpoint, and the checkpointer saves the model and restores it. Sharded states (DTensors) are covered in
``test_checkpoint_io.py``.
"""

import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

import nemo_automodel
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.state_dict_adapter import Ernie5_5StateDictAdapter
from tests.unit_tests.models.ernie5_5.conftest import checkpoint_config, checkpoint_style_config

pytestmark = pytest.mark.cpu_only

FUSED_KV = "model.layers.2.self_attn.kv_b_proj.weight"
# Keys of the tiny model: 106 native tensors; 8 model-level Hugging Face tensors, 30 of the HCA + dense layer,
# 49 of the HCA + MoE layer and 50 of the DSA + MoE layer (four routed experts, three matrices each).
NATIVE_KEYS, HF_KEYS = 106, 137


def _backend() -> BackendConfig:
    return BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")


def _tiny_config(dtype: str = "float32", **overrides) -> Ernie5_5Config:
    """Return a three-layer configuration with every layer kind: HCA + dense MLP, HCA + MoE, DSA + MoE."""
    fields = dict(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=3,
        max_position_embeddings=1024,
        compress_ratios=[128, 128, -2],
        num_nextn_predict_layers=0,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=6,
        o_lora_rank=4,
        o_groups=2,
        sliding_window=8,
        hybrid_mla_num_attention_heads=2,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=4,
        hybrid_mla_qk_nope_head_dim=4,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        rope_interleave=True,
        hc_mult=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        moe_latent_size=8,
        torch_dtype=dtype,
    )
    return Ernie5_5Config(**{**fields, **overrides})


def _tiny_model(dtype: str = "float32", seed: int = 5601) -> Ernie5_5_MoeForCausalLM:
    """Build the registered model and fill every parameter and buffer with seeded random values."""
    model = Ernie5_5_MoeForCausalLM(_tiny_config(dtype), _backend())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype))
    return model


def _read_into(destinations: dict[str, torch.Tensor], checkpoint: dict[str, torch.Tensor]) -> None:
    """Copy checkpoint values into the destinations in place, as a checkpoint reader does.

    Args:
        destinations: Hugging Face names mapped to tensors in the released layouts; written in place.
        checkpoint: The same names mapped to tensors of the same shapes.
    """
    assert destinations.keys() == checkpoint.keys()
    with torch.no_grad():
        for name, target in destinations.items():
            target.copy_(checkpoint[name])


def _assert_same_state(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    """Require the same names, dtypes and values.

    Args:
        actual: Names mapped to tensors of any shape.
        expected: Names mapped to the expected tensors.
    """
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype, name
        assert torch.equal(actual[name], value), name


# Construction


def test_registered_model_builds_the_adapter() -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    assert type(adapter) is Ernie5_5StateDictAdapter
    assert adapter.config is model.config
    assert adapter.supports_low_memory_dcp_load
    assert adapter.iter_checkpoint_load_parts(model.state_dict()) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(num_nextn_predict_layers=1, compress_ratios=[128, 128, -2, -2]),
        dict(tie_word_embeddings=True),
    ],
    ids=["mtp_layer", "tied_weights"],
)
def test_constructor_rejects_mtp_layers_and_tied_weights(overrides) -> None:
    with pytest.raises(ValueError, match="untied decoder-only config"):
        Ernie5_5StateDictAdapter(_tiny_config(**overrides))


# Key plan


@pytest.mark.parametrize(
    "overrides,native_keys,hf_keys",
    [
        (dict(num_hidden_layers=3, compress_ratios=[128, 128, -2]), 106, 3185),
        (dict(compress_ratios=checkpoint_style_config()["compress_ratios"][:-1]), 1516, 66110),
    ],
    ids=["three_layers", "all_layers"],
)
def test_key_plan_of_the_released_shapes(overrides, native_keys, hf_keys) -> None:
    config = checkpoint_config(num_nextn_predict_layers=0, **overrides)
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(config, _backend())
    state = model.state_dict()
    keys = model.state_dict_adapter.get_hf_state_dict_keys(state)
    assert len(state) == native_keys
    assert len(keys) == len(set(keys)) == hf_keys
    last_layer = config.num_hidden_layers - 1
    assert keys[0] == "model.layers.0.post_attention_layernorm.weight"
    assert keys[-1] == "model.mhc_contract.hc_head_scale"
    for name in (
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.block_sparse_moe.experts.511.w3.weight",
        f"model.layers.{last_layer}.self_attn.kv_b_proj.weight",
        f"model.layers.{last_layer}.mlp_hyper_connection.alpha_res",
    ):
        assert name in keys, name
    assert not any(key.startswith(f"model.layers.{last_layer + 1}.") for key in keys)
    # Planning keys neither opens a load session nor needs one.
    assert list(model.state_dict_adapter.to_hf(state)) == keys


# Runs in a fresh interpreter so that each run has its own string hash seed. It builds the three-layer released
# shape on the meta device and prints the number of planned Hugging Face keys and a digest of their order.
_KEY_PLAN_SCRIPT = """
import hashlib
import json
import sys

sys.path.insert(0, sys.argv[1])

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM

fields = json.loads(open(sys.argv[2], encoding="utf-8").read())
fields.update(num_hidden_layers=3, compress_ratios=[128, 128, -2], num_nextn_predict_layers=0)
backend = BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")
with torch.device("meta"):
    model = Ernie5_5_MoeForCausalLM(Ernie5_5Config(**fields), backend)
keys = model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
print(json.dumps({"count": len(keys), "digest": hashlib.sha256("\\n".join(keys).encode()).hexdigest()}))
"""


# Seconds that the interpreters of one test may take together; below the hard watchdog of the unit suite.
_INTERPRETER_TIMEOUT = 60.0


def _run_interpreters(arguments: list[str], hash_seeds: tuple[str | None, ...]) -> dict[str | None, str]:
    """Run one Python script in a fresh interpreter per hash seed, all at once, and return what each printed.

    No interpreter outlives the call: when one does not finish in time, it and the others are killed and reaped
    before the exception leaves this function.

    Args:
        arguments: Path of the script, followed by its arguments.
        hash_seeds: ``PYTHONHASHSEED`` of each interpreter; ``None`` leaves the variable unset.

    Returns:
        The standard output of each interpreter, by its hash seed.

    Raises:
        subprocess.TimeoutExpired: If the interpreters take longer than ``_INTERPRETER_TIMEOUT`` together.
        AssertionError: If an interpreter exits with a nonzero status; the message is its standard error.
    """
    base = {name: value for name, value in os.environ.items() if name != "PYTHONHASHSEED"}
    processes: dict[str | None, subprocess.Popen[str]] = {}
    finished: dict[str | None, tuple[str, str]] = {}
    deadline = time.monotonic() + _INTERPRETER_TIMEOUT
    try:
        for seed in hash_seeds:
            environment = base if seed is None else {**base, "PYTHONHASHSEED": seed}
            processes[seed] = subprocess.Popen(
                [sys.executable, *arguments], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        for seed, process in processes.items():
            finished[seed] = process.communicate(timeout=max(1.0, deadline - time.monotonic()))
    finally:
        for seed, process in processes.items():
            if seed not in finished:
                process.kill()
                process.communicate()
    for seed, process in processes.items():
        assert process.returncode == 0, finished[seed][1]
    return {seed: output for seed, (output, _) in finished.items()}


def test_key_plan_does_not_depend_on_the_hash_seed(tmp_path: Path) -> None:
    """Every process must plan the same key order: the ranks of a distributed save are separate processes."""
    script = tmp_path / "key_plan.py"
    script.write_text(_KEY_PLAN_SCRIPT, encoding="utf-8")
    fields = tmp_path / "config.json"
    fields.write_text(json.dumps(checkpoint_style_config()), encoding="utf-8")
    repository_root = str(Path(nemo_automodel.__file__).parents[1])
    outputs = _run_interpreters([str(script), repository_root, str(fields)], ("0", "1", "2"))
    results = {seed: json.loads(output.splitlines()[-1]) for seed, output in outputs.items()}
    config = checkpoint_config(num_hidden_layers=3, compress_ratios=[128, 128, -2], num_nextn_predict_layers=0)
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(config, _backend())
    keys = model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
    expected = {"count": 3185, "digest": hashlib.sha256("\n".join(keys).encode()).hexdigest()}
    assert len(keys) == 3185
    for seed, result in results.items():
        assert result == expected, f"key plan differs with PYTHONHASHSEED={seed}"


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_key_plan_is_read_from_the_materialized_model_before_sharding(dtype) -> None:
    """The framework asks for the keys of the unsharded model with its weights, before it shards the model."""
    model = _tiny_model(dtype)
    adapter = model.state_dict_adapter
    state = model.state_dict()
    before = {name: value.clone() for name, value in state.items()}
    keys = adapter.get_hf_state_dict_keys(state)
    assert len(keys) == len(set(keys)) == HF_KEYS
    assert keys == list(adapter.to_hf(state))
    # Planning reads shapes only: the weights are untouched and no load session is left open.
    _assert_same_state(model.state_dict(), before)
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    assert list(destinations) == keys
    _assert_same_state(adapter.from_hf(destinations), before)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_conversion_of_the_meta_copy_keeps_names_shapes_and_dtypes(dtype) -> None:
    """Key planning converts a shape-only copy of the state; it must describe the tensors that a save writes."""
    model = _tiny_model(dtype)
    adapter = model.state_dict_adapter
    state = model.state_dict()
    on_meta = {name: torch.empty_like(value, device="meta") for name, value in state.items()}
    planned = adapter.to_hf(on_meta, exclude_key_regex=r".*_extra_state.*", quantization=False)
    converted = adapter.to_hf(state)
    assert list(planned) == list(converted)
    for name, value in converted.items():
        assert planned[name].device.type == "meta", name
        assert planned[name].shape == value.shape and planned[name].dtype == value.dtype, name
    # FP32 is kept where the model keeps it, whatever the model dtype.
    assert planned["model.layers.0.self_attention_hyper_connection.alpha_pre"].dtype == torch.float32
    assert planned["model.mhc_contract.hc_head_fn"].dtype == torch.float32


# Conversion of unsharded states without a load session


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_unsharded_state_round_trips(dtype) -> None:
    model = _tiny_model(dtype)
    adapter = model.state_dict_adapter
    state = model.state_dict()
    converted = adapter.to_hf(state)
    assert list(converted) == adapter.get_hf_state_dict_keys(state)
    assert len(converted) == HF_KEYS and len(state) == NATIVE_KEYS
    # The fused K/V projection keeps the dtype of the native projections.
    assert converted[FUSED_KV].dtype == state["model.layers.2.attention.k_b_proj"].dtype
    assert converted["model.embed_tokens.weight"] is state["model.embed_tokens.weight"]
    for _ in range(2):  # No session is opened, so the conversion can be repeated.
        _assert_same_state(adapter.from_hf(converted), state)
    _assert_same_state(adapter.from_hf(adapter.to_hf(state, exclude_key_regex=r".*_extra_state.*")), state)


def test_names_without_the_model_prefix_are_rejected() -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    state = model.state_dict()
    bare = {name.removeprefix("model."): value for name, value in state.items()}
    with pytest.raises(ValueError, match="model. prefix"):
        adapter.to_hf(bare)
    with pytest.raises(ValueError, match="model. prefix"):
        adapter.to_hf(bare, for_checkpoint_load=True)
    # A load that was rejected before it started leaves the adapter usable.
    assert len(adapter.to_hf(state)) == HF_KEYS


def test_empty_state_is_not_saved() -> None:
    with pytest.raises(ValueError, match="Expected model tensors"):
        _tiny_model().state_dict_adapter.to_hf({})


@pytest.mark.parametrize("option", ["quantization", "v4_compatible"])
@pytest.mark.parametrize("for_checkpoint_load", [False, True])
def test_unsupported_modes_are_rejected_before_a_session_opens(option, for_checkpoint_load) -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    state = model.state_dict()
    with pytest.raises(NotImplementedError, match="Quantization and v4 metadata export"):
        adapter.to_hf(state, for_checkpoint_load=for_checkpoint_load, **{option: True})
    assert len(adapter.to_hf(state, for_checkpoint_load=True)) == HF_KEYS


def test_configuration_changed_after_construction_is_rejected() -> None:
    model = _tiny_model()
    model.config.tie_word_embeddings = True
    with pytest.raises(NotImplementedError, match="untied decoder-only"):
        model.state_dict_adapter.to_hf(model.state_dict())


def test_single_tensor_conversion_is_not_supported() -> None:
    model = _tiny_model()
    name = "model.embed_tokens.weight"
    with pytest.raises(NotImplementedError, match="streaming export"):
        model.state_dict_adapter.convert_single_tensor_to_hf(name, model.state_dict()[name])


# Load session on an unsharded state: active, replaced, abandoned, failed


def test_load_session_installs_the_checkpoint_values() -> None:
    model, other = _tiny_model(seed=1), _tiny_model(seed=2)
    adapter = model.state_dict_adapter
    checkpoint = other.state_dict_adapter.to_hf(other.state_dict())
    keys = adapter.get_hf_state_dict_keys(model.state_dict())
    for _ in range(2):  # A finished session leaves the adapter ready for the next one.
        destinations = adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
        assert list(destinations) == keys
        _read_into(destinations, checkpoint)
        loaded = adapter.from_hf(destinations)
        _assert_same_state(loaded, other.state_dict())
    model.load_state_dict(loaded, strict=True)
    _assert_same_state(model.state_dict(), other.state_dict())


def test_unsharded_load_destinations_alias_model_storage() -> None:
    """A load in one process reads into the model: the routed experts are not allocated a second time."""
    # 64 routed experts, so that the experts are most of this small model, as they are of the released one.
    config = _tiny_config(n_routed_experts=64)
    model, other = Ernie5_5_MoeForCausalLM(config, _backend()), Ernie5_5_MoeForCausalLM(config, _backend())
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for tensor in other.state_dict().values():
            tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype))
    adapter = model.state_dict_adapter
    assert adapter.supports_low_memory_dcp_load
    state = model.state_dict()
    expert_elements = sum(tensor.numel() for name, tensor in state.items() if ".moe.experts." in name)
    assert expert_elements > sum(tensor.numel() for tensor in state.values()) / 2
    storages = {tensor.untyped_storage().data_ptr() for tensor in state.values()}
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    temporaries = {
        name: target for name, target in destinations.items() if target.untyped_storage().data_ptr() not in storages
    }
    # Only the fused K/V projection of the DSA layer is read into a temporary, and it is small.
    assert list(temporaries) == [FUSED_KV]
    assert 100 * sum(target.numel() for target in temporaries.values()) < expert_elements
    # Writing through the per-expert destinations alone fills the grouped expert parameters.
    checkpoint = other.state_dict_adapter.to_hf(other.state_dict())
    experts = [name for name in destinations if ".block_sparse_moe.experts." in name]
    assert len(experts) == 2 * 64 * 3
    with torch.no_grad():
        for name in experts:
            destinations[name].copy_(checkpoint[name])
    parameters, expected = dict(model.named_parameters()), dict(other.named_parameters())
    grouped = [name for name in parameters if ".moe.experts." in name]
    assert len(grouped) == 4
    for name in grouped:
        assert torch.equal(parameters[name], expected[name]), name
    # The rest of the load: the other destinations are filled, and from_hf installs the fused projection.
    _read_into(destinations, checkpoint)
    loaded = adapter.from_hf(destinations)
    assert list(loaded) == list(state) and all(loaded[name] is tensor for name, tensor in state.items())
    _assert_same_state(model.state_dict(), other.state_dict())


def test_second_load_or_save_during_a_session_is_rejected() -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    state = model.state_dict()
    expected = {name: value.clone() for name, value in state.items()}
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.to_hf(state)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.get_hf_state_dict_keys(state)
    # The rejected calls do not disturb the open session.
    _assert_same_state(adapter.from_hf(destinations), expected)


def test_replaced_destinations_end_the_adapter() -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    state = model.state_dict()
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(dict(destinations))
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(destinations)
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.to_hf(state)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.get_hf_state_dict_keys(state)


def test_destinations_of_another_adapter_end_the_adapter() -> None:
    """An adapter that is handed another session's destinations cannot know that its own load is intact."""
    model, other = _tiny_model(seed=1), _tiny_model(seed=2)
    adapter = model.state_dict_adapter
    other_state = {name: value.clone() for name, value in other.state_dict().items()}
    own = adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
    foreign = other.state_dict_adapter.to_hf(other.state_dict(), for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(foreign)
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(own)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.to_hf(model.state_dict())
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
    # The session of the other adapter was only looked at: it still completes.
    _assert_same_state(other.state_dict_adapter.from_hf(foreign), other_state)


def test_destinations_of_a_finished_session_end_the_adapter() -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    state = model.state_dict()
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    adapter.from_hf(destinations)
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(destinations)
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.get_hf_state_dict_keys(state)


def test_copy_of_finished_destinations_ends_the_adapter() -> None:
    """A copy of finished destinations holds what the earlier load left in them; it is not a fresh checkpoint."""
    model, other = _tiny_model(seed=1), _tiny_model(seed=2)
    adapter = model.state_dict_adapter
    checkpoint = other.state_dict_adapter.to_hf(other.state_dict())
    destinations = adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
    _read_into(destinations, checkpoint)
    adapter.from_hf(destinations)
    # Tensors that never were destinations are converted without a session, also after a finished load.
    _assert_same_state(adapter.from_hf(dict(checkpoint)), other.state_dict())
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(dict(destinations))
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.to_hf(model.state_dict())


def test_copy_of_the_model_does_not_share_a_load_session() -> None:
    """A deep copy made during a load refuses all work; the open session stays with the original."""
    model, other = _tiny_model(seed=1), _tiny_model(seed=2)
    adapter = model.state_dict_adapter
    # A copy of an idle model has a working adapter of its own, bound to the copied configuration.
    idle = copy.deepcopy(model)
    idle_adapter = idle.state_dict_adapter
    assert idle_adapter is not adapter and idle_adapter.config is idle.config and idle.config is not model.config
    _assert_same_state(idle_adapter.from_hf(idle_adapter.to_hf(idle.state_dict())), model.state_dict())
    checkpoint = other.state_dict_adapter.to_hf(other.state_dict())
    destinations = adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
    during = copy.deepcopy(model)
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        during.state_dict_adapter.from_hf(destinations)
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        during.state_dict_adapter.to_hf(during.state_dict(), for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        during.state_dict_adapter.to_hf(during.state_dict())
    # The copy did not touch the session of the original, which still completes.
    _read_into(destinations, checkpoint)
    _assert_same_state(adapter.from_hf(destinations), other.state_dict())


def test_abandoned_session_ends_the_adapter() -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    state = model.state_dict()
    converted = adapter.to_hf(state)
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    del destinations
    gc.collect()
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        adapter.to_hf(state, for_checkpoint_load=True)
    # The conversion without a session must not become a way around the abandoned load.
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(converted)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.to_hf(state)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.get_hf_state_dict_keys(state)


def _drop_a_key(destinations: dict[str, torch.Tensor]) -> None:
    """Remove the first destination.

    Args:
        destinations: Hugging Face names mapped to destination tensors of any shape; modified in place.
    """
    destinations.pop(next(iter(destinations)))


def _exchange_a_tensor(destinations: dict[str, torch.Tensor]) -> None:
    """Replace the fused K/V destination by a copy of itself: equal values in another tensor object.

    Args:
        destinations: Hugging Face names mapped to destination tensors; the fused K/V projection has shape
            [heads * (key_channels + value_channels), latent]. Modified in place.
    """
    destinations[FUSED_KV] = destinations[FUSED_KV].clone()


def _break_a_layout(destinations: dict[str, torch.Tensor]) -> None:
    """Resize the fused K/V destination in place to [1, 1], a shape that the conversion rejects.

    Args:
        destinations: Hugging Face names mapped to destination tensors; the fused K/V projection has shape
            [heads * (key_channels + value_channels), latent] and is resized in place.
    """
    destinations[FUSED_KV].resize_(1, 1)


@pytest.mark.parametrize(
    "damage,message",
    [
        (_drop_a_key, "Loaded keys changed"),
        (_exchange_a_tensor, "Load destinations were replaced"),
        (_break_a_layout, "Expected combined KV shape"),
    ],
    ids=["keys_changed", "tensor_exchanged", "conversion_failed"],
)
def test_failed_session_ends_the_adapter(damage, message) -> None:
    model = _tiny_model()
    adapter = model.state_dict_adapter
    state = model.state_dict()
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    damage(destinations)
    with pytest.raises(ValueError, match=message):
        adapter.from_hf(destinations)
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(destinations)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.to_hf(state)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.get_hf_state_dict_keys(state)


# One process: the checkpointer loads and saves the unsharded model, and no device mesh is passed


def _write_hf_checkpoint(directory: Path, dtype: str, seed: int) -> Ernie5_5_MoeForCausalLM:
    """Write a Hugging Face format directory that stores every tensor in the model dtype.

    Args:
        directory: Directory to create.
        dtype: Model dtype, also the dtype of every stored tensor.
        seed: Seed of the values.

    Returns:
        A model that holds the stored values: the tensors that it keeps in FP32 are rounded to the stored dtype.
    """
    model = _tiny_model(dtype, seed)
    stored_dtype = getattr(torch, dtype)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.copy_(tensor.to(stored_dtype))
    converted = model.state_dict_adapter.to_hf(model.state_dict())
    directory.mkdir(parents=True)
    save_file(
        {name: value.to(stored_dtype).contiguous() for name, value in converted.items()},
        str(directory / "model.safetensors"),
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(converted, "model.safetensors")})
    )
    model.config.save_pretrained(directory)
    return model


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_checkpointer_loads_an_unsharded_model_in_one_process(tmp_path: Path, dtype: str) -> None:
    """The load-only checkpointer of one process has no expert mesh and converts the whole state eagerly."""
    checkpoint = tmp_path / "hf_checkpoint"
    source = _write_hf_checkpoint(checkpoint, dtype, seed=17)
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_tiny_config(dtype), _backend())
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir="",
        model_save_format="safetensors",
        model_cache_dir=str(tmp_path / "cache"),
        model_repo_id=str(checkpoint),
        save_consolidated=False,
    ).build(0, 0, 0)
    try:
        checkpointer.initialize_model_weights(model, torch.device("cpu"))
        checkpointer.load_base_model(
            model, torch.device("cpu"), str(tmp_path / "cache"), str(checkpoint), load_base_model=True
        )
    finally:
        checkpointer.close()
    _assert_same_state(model.state_dict(), source.state_dict())


def _read_safetensors(directory: Path) -> dict[str, torch.Tensor]:
    """Read the safetensors files that lie directly in a directory.

    Args:
        directory: Directory with at least one ``*.safetensors`` file.

    Returns:
        Tensor names mapped to the stored tensors of any shape; no name is stored twice.
    """
    files = sorted(directory.glob("*.safetensors"))
    assert files, directory
    stored: dict[str, torch.Tensor] = {}
    for file in files:
        tensors = load_file(str(file))
        assert stored.keys().isdisjoint(tensors), file.name
        stored.update(tensors)
    return stored


def _assert_same_values(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    """Require the same names, shapes and values; a file may store a tensor in another floating-point dtype.

    Args:
        actual: Names mapped to tensors of any shape.
        expected: Names mapped to the expected tensors, whose values BF16 represents exactly.
    """
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert actual[name].shape == value.shape, name
        assert torch.equal(actual[name].to(torch.float32), value.to(torch.float32)), name


@pytest.mark.parametrize("save_consolidated", ["false", "final"])
def test_checkpointer_saves_and_restores_an_unsharded_model_in_one_process(
    tmp_path: Path, save_consolidated: str
) -> None:
    """One process saves through the checkpointer, which receives views that are not contiguous, and restores."""
    checkpoint = tmp_path / "hf_checkpoint"
    model = _write_hf_checkpoint(checkpoint, "bfloat16", seed=23)
    adapter = model.state_dict_adapter
    native = {name: value.clone() for name, value in model.state_dict().items()}
    converted = adapter.to_hf(model.state_dict())
    # The per-expert matrices are transposed views of the grouped expert parameters.
    experts = [name for name in converted if ".block_sparse_moe.experts." in name]
    assert len(experts) == 2 * 4 * 3 and not any(converted[name].is_contiguous() for name in experts)
    released = {name: value.clone() for name, value in converted.items()}
    del converted
    # The loader records the key plan on the model before it shards the model; the consolidated export uses it.
    model._pre_shard_hf_state_dict_keys = adapter.get_hf_state_dict_keys(model.state_dict())
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path / "run"),
        model_save_format="safetensors",
        model_cache_dir=str(tmp_path / "cache"),
        model_repo_id=str(checkpoint),
        save_consolidated=save_consolidated,
        is_async=False,
    ).build(0, 0, 0)
    saved = tmp_path / "saved" / "model"
    try:
        checkpointer.save_model(model, str(tmp_path / "saved"), is_final_checkpoint=True)
        _assert_same_values(_read_safetensors(saved), released)
        if save_consolidated == "final":
            _assert_same_values(_read_safetensors(saved / "consolidated"), released)
        with torch.no_grad():
            for tensor in model.state_dict().values():
                tensor.zero_()
        checkpointer.load_model(model, model_path=str(saved), is_init_step=False)
    finally:
        checkpointer.close()
    _assert_same_state(model.state_dict(), native)


def test_public_loader_loads_a_hugging_face_checkpoint_in_one_process(tmp_path: Path) -> None:
    """``NeMoAutoModelForCausalLM`` builds the registered model and loads its weights without any mesh."""
    checkpoint = tmp_path / "hf_checkpoint"
    source = _write_hf_checkpoint(checkpoint, "bfloat16", seed=19)
    # The loader asks CUDA for the current device even for a CPU model, and this test answers with the CPU device.
    # Where CUDA is available, the checkpointer asks again when it chooses the checkpoint reader and uses the
    # answer as a CUDA index. So CUDA is reported as unavailable and the load takes the path of a host without
    # CUDA on every host. Nothing of the load path is replaced.
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")),
    ):
        model = NeMoAutoModelForCausalLM.from_config(
            str(checkpoint),
            load_base_model=True,
            backend=_backend(),
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            force_hf=False,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            cache_dir=str(tmp_path / "cache"),
        )
    # The loader returns an instance of a subclass that it derives to attach the capability properties.
    assert isinstance(model, Ernie5_5_MoeForCausalLM)
    _assert_same_state(model.state_dict(), source.state_dict())

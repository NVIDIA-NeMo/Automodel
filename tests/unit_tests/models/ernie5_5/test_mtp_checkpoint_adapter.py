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

"""Checkpoint ownership, lifecycle and failure agreement for ERNIE MTP."""

import copy
import gc
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.api import CheckpointException

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.ernie5_5.config import mtp_decoder_config
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.mtp_checkpoint_adapter import (
    Ernie5_5MTPCheckpointAdapter,
)
from tests.unit_tests.models.ernie5_5.test_mtp_training_local import _released_owned_tensors, backend, config, model
from tests.unit_tests.models.ernie5_5.test_state_dict_adapter import _read_safetensors

pytestmark = pytest.mark.cpu_only


def candidate(dtype: str = "float32", seed: int = 19):
    """Build a model that already owns the native checkpoint adapter."""
    result = model(dtype, seed=seed)
    assert isinstance(result.state_dict_adapter, Ernie5_5MTPCheckpointAdapter)
    return result


def exported(source) -> dict[str, torch.Tensor]:
    """Clone each released tensor independently so safetensors can store copies."""
    return {
        name: value.detach().contiguous().clone()
        for name, value in source.state_dict_adapter.to_hf(source.state_dict()).items()
    }


def fill(destinations: dict[str, torch.Tensor], checkpoint: dict[str, torch.Tensor]) -> None:
    """Copy released tensors into matching released-layout read destinations.

    Args:
        destinations: Released tensors, including head [streams*hidden, streams].
        checkpoint: Same names and global shapes; values are converted to destination dtype.
    """
    assert destinations.keys() == checkpoint.keys()
    with torch.no_grad():
        for name, destination in destinations.items():
            destination.copy_(checkpoint[name])


def assert_same(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    """Require exact keys, dtypes and values of two tensor mappings in identical layouts.

    Args:
        actual: Native or released tensor mapping in the layout being tested.
        expected: Same tensor names and semantic shapes, used as the reference.
    """
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)


def assert_blocked(adapter, state: dict[str, torch.Tensor]) -> None:
    """Assert that a failed adapter rejects all state operations.

    Args:
        adapter: Isolated MTP checkpoint adapter.
        state: Registered native tensors, including head [streams, streams*hidden].
    """
    with pytest.raises(RuntimeError):
        adapter.to_hf(state)
    with pytest.raises(RuntimeError):
        adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError):
        adapter.get_hf_state_dict_keys(state)


def write_checkpoint(directory: Path, source, *, drop: str | None = None) -> dict[str, torch.Tensor]:
    """Write an independent HF checkpoint, optionally omitting required tensors."""
    values = exported(source)
    if drop == "alias":
        del values["model.layers.2.self_attn.q_a_proj.weight"]
    elif drop == "owned":
        del values["model.layers.2.e_proj.weight"]
    elif drop == "mtp_layer":
        values = {name: value for name, value in values.items() if not name.startswith("model.layers.2.")}
    elif drop == "shared_copies":
        # Keep the nine tensors the MTP layer owns; every other tensor under
        # model.layers.2 is a copy of the last decoder block.
        owned = _released_owned_tensors(source).keys()
        values = {
            name: value for name, value in values.items() if not name.startswith("model.layers.2.") or name in owned
        }
    directory.mkdir(parents=True)
    save_file(values, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(values, "model.safetensors")}) + "\n"
    )
    source.config.save_pretrained(directory)
    return values


def test_destination_ownership_and_exact_native_install() -> None:
    """Install complete HF state into native owners without replacing parameters."""
    source, target = candidate(seed=7), candidate(seed=88)
    checkpoint, state = exported(source), target.state_dict()
    identities = {name: id(value) for name, value in target.named_parameters(remove_duplicate=False)}
    adapter = target.state_dict_adapter
    assert adapter.supports_low_memory_dcp_load
    assert len(adapter.get_hf_state_dict_keys(state)) == 147
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    assert len(destinations) == 147 and len(destinations.alias_pairs) == 49
    assert destinations["model.layers.2.e_proj.weight"].data_ptr() == state["mtp.e_proj.weight"].data_ptr()
    assert destinations["model.layers.2.hc_head_fn"].data_ptr() == state["mtp.hyper_head.hc_fn"].data_ptr()
    for scratch, owner in destinations.alias_pairs.values():
        assert scratch.data_ptr() != owner.data_ptr()
    fill(destinations, checkpoint)
    loaded = adapter.from_hf(destinations)
    assert all(loaded[name] is state[name] for name in state)
    assert_same(target.state_dict(), source.state_dict())
    assert identities == {name: id(value) for name, value in target.named_parameters(remove_duplicate=False)}


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.2.self_attn.q_a_proj.weight",
        "model.layers.2.transformer_layer.self_attention_hyper_connection.mapping_proj.weight",
        "model.layers.2.transformer_layer.self_attention_hyper_connection.alpha_pre",
        "model.layers.2.self_attn.kv_b_proj.weight",
        "model.layers.2.block_sparse_moe.experts.3.w2.weight",
        "model.layers.1.self_attn.q_a_proj.weight",
    ],
)
def test_changed_copy_or_decoder_owner_is_rejected_and_adapter_poisoned(
    name: str,
) -> None:
    """Reject unequal shared copies and permanently block the failed adapter."""
    source, target = candidate(seed=7), candidate(seed=88)
    values = exported(source)
    values[name].view(-1)[0] += 1
    adapter, state = target.state_dict_adapter, target.state_dict()
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    fill(destinations, values)
    with pytest.raises(ValueError, match="MTP shared tensor differs"):
        adapter.from_hf(destinations)
    assert_blocked(adapter, state)


@pytest.mark.parametrize("damage", ["copy", "foreign", "keys", "tensor", "ended", "ended_copy", "abandon"])
def test_session_lifecycle_cannot_be_bypassed(damage: str) -> None:
    """Reject replaced, ended, abandoned or modified load sessions."""
    source, target = candidate(seed=7), candidate(seed=88)
    state, adapter = target.state_dict(), target.state_dict_adapter
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    fill(destinations, exported(source))
    if damage == "abandon":
        del destinations
        gc.collect()
        assert_blocked(adapter, state)
        return
    if damage in ["ended", "ended_copy"]:
        adapter.from_hf(destinations)
    incoming = destinations
    if damage in ["copy", "ended_copy"]:
        incoming = dict(destinations)
    elif damage == "foreign":
        other = candidate(seed=4)
        incoming = other.state_dict_adapter.to_hf(other.state_dict(), for_checkpoint_load=True)
    elif damage == "keys":
        del destinations[next(iter(destinations))]
    elif damage == "tensor":
        name = next(iter(destinations))
        destinations[name] = destinations[name].clone()
    with pytest.raises((ValueError, RuntimeError)):
        adapter.from_hf(incoming)
    assert_blocked(adapter, state)


def test_concurrent_operations_preserve_live_session_and_deepcopy_is_independent() -> None:
    """Keep a live session isolated from concurrent access and copied models."""
    source, target = candidate(seed=7), candidate(seed=88)
    adapter, state = target.state_dict_adapter, target.state_dict()
    idle = copy.deepcopy(target)
    assert idle.state_dict_adapter.config is idle.config
    assert_same(idle.state_dict_adapter.from_hf(exported(source)), source.state_dict())
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    fill(destinations, exported(source))
    during = copy.deepcopy(target)
    assert_blocked(during.state_dict_adapter, during.state_dict())
    with pytest.raises(RuntimeError):
        during.state_dict_adapter.from_hf(destinations)
    assert_blocked(adapter, state)
    assert_same(adapter.from_hf(destinations), source.state_dict())


@pytest.mark.parametrize("drop", ["alias", "owned", "mtp_layer"])
def test_real_checkpointer_refuses_missing_required_tensors(tmp_path: Path, drop: str) -> None:
    """Reject missing HF aliases or owned tensors through the real Checkpointer."""
    checkpoint = tmp_path / "hf"
    source, target = candidate(seed=7), candidate(seed=88)
    write_checkpoint(checkpoint, source, drop=drop)
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir="",
        model_save_format="safetensors",
        model_repo_id=str(checkpoint),
        model_cache_dir=str(tmp_path / "cache"),
        save_consolidated=False,
    ).build(0, 0, 0)
    try:
        with pytest.raises(CheckpointException):
            checkpointer.load_model(target, model_path=str(checkpoint), is_init_step=True)
    finally:
        checkpointer.close()


def test_checkpoint_without_shared_copies_is_refused_by_mtp_and_loaded_by_the_decoder_class(tmp_path: Path) -> None:
    """A checkpoint that stores the shared body once loads as decoder-only, not through the MTP class."""
    checkpoint = tmp_path / "hf"
    source, target = candidate(seed=7), candidate(seed=88)
    stored = write_checkpoint(checkpoint, source, drop="shared_copies")
    assert {name for name in stored if name.startswith("model.layers.2.")} == set(_released_owned_tensors(source))
    assert len(stored) == 147 - 49

    def checkpointer(cache: str):
        return CheckpointingConfig(
            enabled=True,
            checkpoint_dir="",
            model_save_format="safetensors",
            model_repo_id=str(checkpoint),
            model_cache_dir=str(tmp_path / cache),
            save_consolidated=False,
        ).build(0, 0, 0)

    loader = checkpointer("cache_mtp")
    try:
        with pytest.raises(CheckpointException, match=r"Missing key in checkpoint state_dict: model\.layers\.2\."):
            loader.load_model(target, model_path=str(checkpoint), is_init_step=True)
    finally:
        loader.close()

    torch.manual_seed(5)
    decoder = Ernie5_5_MoeForCausalLM(mtp_decoder_config(config()), backend())
    decoder.initialize_weights()
    loader = checkpointer("cache_decoder")
    try:
        loader.load_model(decoder, model_path=str(checkpoint), is_init_step=True)
    finally:
        loader.close()
    assert_same(
        decoder.state_dict(),
        {name: value for name, value in source.state_dict().items() if not name.startswith("mtp.")},
    )


@pytest.mark.parametrize("dtype,consolidate", [("float32", False), ("bfloat16", "final")])
def test_real_checkpointer_base_save_and_fresh_resume(
    tmp_path: Path, caplog, dtype: str, consolidate: bool | str
) -> None:
    """Load, save and freshly resume exact native state with optional consolidation."""
    checkpoint = tmp_path / "hf"
    source, target = candidate(dtype, seed=7), candidate(dtype, seed=88)
    source.mtp.router_state.e_score_correction_bias.copy_(torch.tensor([-0.3, 0.2, 0.1, -0.1]))
    source.model.layers[-1].moe.gate.e_score_correction_bias.copy_(torch.tensor([0.2, -0.1, -0.2, 0.3]))
    released = write_checkpoint(checkpoint, source)
    source._pre_shard_hf_state_dict_keys = source.state_dict_adapter.get_hf_state_dict_keys(source.state_dict())
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path / "run"),
        model_save_format="safetensors",
        model_repo_id=str(checkpoint),
        model_cache_dir=str(tmp_path / "cache"),
        save_consolidated=consolidate,
        is_async=False,
    ).build(0, 0, 0)
    saved = tmp_path / "saved" / "model"
    caplog.set_level("INFO")
    try:
        checkpointer.load_base_model(
            target,
            torch.device("cpu"),
            str(tmp_path / "cache"),
            str(checkpoint),
            load_base_model=True,
        )
        assert_same(target.state_dict(), source.state_dict())
        assert "destinations" in caplog.text
        checkpointer.save_model(source, str(tmp_path / "saved"), is_final_checkpoint=True)
        assert_same(_read_safetensors(saved), released)
        if consolidate:
            assert_same(_read_safetensors(saved / "consolidated"), released)
        fresh = candidate(dtype, seed=99)
        identities = {name: id(value) for name, value in fresh.named_parameters(remove_duplicate=False)}
        checkpointer.load_model(fresh, model_path=str(saved), is_init_step=False)
        assert_same(fresh.state_dict(), source.state_dict())
        assert identities == {name: id(value) for name, value in fresh.named_parameters(remove_duplicate=False)}
    finally:
        checkpointer.close()


@pytest.fixture(scope="module")
def four_rank_records(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Run all sharded layouts once on four ranks and share the records between the cases below.

    Starting four ranks costs far more than the phases themselves, so the
    seven layout and phase-group cases share one process group instead of
    starting one each. Every case still checks its own phase group.
    """
    from tests.unit_tests.models.ernie5_5 import _mtp_checkpoint_sharded

    return _mtp_checkpoint_sharded.run(tmp_path_factory.mktemp("four_rank"))


@pytest.mark.parametrize(
    "layout,phase_group",
    [
        (layout, group)
        for layout in ["ep_shard_ep", "ep"]
        for group in ["roundtrip", "value_mismatch", "failure_agreement"]
    ]
    + [("embed2d", "diagnostic")],
)
def test_four_rank_sharded_checkpoint_load_save_and_failure_agreement(
    four_rank_records: dict[str, dict], layout: str, phase_group: str
) -> None:
    """Exercise a bounded set of real sharded Checkpointer contracts on four ranks."""
    from tests.unit_tests.models.ernie5_5 import _mtp_checkpoint_sharded

    assert phase_group in _mtp_checkpoint_sharded.LAYOUT_GROUPS[layout]
    failures = _mtp_checkpoint_sharded.evaluate(four_rank_records[layout], phase_group)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("error_kind", ["runtime", "checkpoint"])
def test_unexpected_validation_error_agrees_before_decoder_finisher(
    monkeypatch: pytest.MonkeyPatch, error_kind: str
) -> None:
    """An unexpected comparison failure must join exactly one agreement and poison the adapter."""
    source = candidate()
    adapter, state = source.state_dict_adapter, source.state_dict()
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    fill(destinations, exported(candidate()))
    error = (
        RuntimeError("unexpected local validation failure")
        if error_kind == "runtime"
        else CheckpointException("unexpected local validation failure", {})
    )
    agreements, finishers = [], []
    original = adapter._raise_together

    def agreement(session, problem):
        agreements.append(problem)
        return original(session, problem)

    def broken_local(_tensor):
        raise error

    def finish(*_args, **_kwargs):
        finishers.append(True)
        raise AssertionError("Decoder finisher must not run after failed validation")

    monkeypatch.setattr(adapter, "_local", broken_local)
    monkeypatch.setattr(adapter, "_raise_together", agreement)
    monkeypatch.setattr(adapter.decoder, "from_hf", finish)
    with pytest.raises(type(error)) as caught:
        adapter.from_hf(destinations)
    assert caught.value is error
    assert agreements == [error] and not finishers
    assert adapter._failed and adapter._active_session is None
    assert_blocked(adapter, state)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("phase", ["validation", "finisher"])
def test_process_control_interrupts_propagate_without_failure_collective(
    monkeypatch: pytest.MonkeyPatch, interrupt: type[BaseException], phase: str
) -> None:
    """Preserve control interrupts while poisoning the abandoned local load."""
    source = candidate()
    adapter = source.state_dict_adapter
    destinations = adapter.to_hf(source.state_dict(), for_checkpoint_load=True)
    fill(destinations, exported(candidate()))
    agreements = []
    original = adapter._raise_together

    def agreement(session, error):
        agreements.append(error)
        return original(session, error)

    def stop(*_args, **_kwargs):
        raise interrupt("test process-control propagation")

    monkeypatch.setattr(adapter, "_raise_together", agreement)
    if phase == "validation":
        monkeypatch.setattr(adapter, "_local", stop)
    else:
        monkeypatch.setattr(adapter.decoder, "from_hf", stop)
    with pytest.raises(interrupt):
        adapter.from_hf(destinations)
    assert len(agreements) == (0 if phase == "validation" else 1)
    assert adapter._failed

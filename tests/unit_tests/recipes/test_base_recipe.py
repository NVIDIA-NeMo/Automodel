# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import logging
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.checkpoint.checkpointing import save_losses
from nemo_automodel.components.checkpoint.utils import (
    find_latest_checkpoint,
    is_checkpoint_incomplete,
    mark_checkpoint_incomplete,
    read_checkpoint_pointer,
    resolve_restore_from_to_checkpoint_dir,
)
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.recipes.base_recipe import BaseRecipe, is_distributed_stateful

try:
    import expecttest

    HAS_ET = True
except Exception:
    HAS_ET = False


@pytest.fixture(autouse=True)
def _mock_single_rank(monkeypatch):
    """
    Pretend we are running in a single-process, non-distributed setup.
    """
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)
    yield


@pytest.fixture(autouse=True)
def _patch_checkpoint_ops(monkeypatch):
    """
    Replace Checkpointer class with a minimal mock that uses torch.save/torch.load
    so that BaseRecipe can operate without the real checkpoint infrastructure.
    """
    from nemo_automodel.components.checkpoint import checkpointing

    class MockCheckpointer:
        """Mock Checkpointer for testing."""

        def __init__(self, config, dp_rank, tp_rank, pp_rank, moe_mesh=None):
            self.config = config
            self.dp_rank = dp_rank
            self.tp_rank = tp_rank
            self.pp_rank = pp_rank
            self.moe_mesh = moe_mesh
            self.distributed_saves = []
            self.distributed_loads = []
            self.staging_waits = 0

        def save_model(
            self,
            model=None,
            weights_path=None,
            peft_config=None,
            tokenizer=None,
            is_final_checkpoint=False,
        ):
            """Save model state dict."""
            del peft_config, tokenizer, is_final_checkpoint
            if model is None:
                return
            model_dir = os.path.join(weights_path, "model")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

        def load_model(
            self, model, model_path, is_init_step=False, use_checkpoint_id=True, key_mapping=None, quantization=False
        ):
            """Load model state dict."""
            if model is None:
                return
            model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"), weights_only=False))

        def save_optimizer(self, optimizer, model, weights_path, scheduler=None):
            """Save optimizer state dict."""
            if optimizer is None:
                return
            optim_dir = os.path.join(weights_path, "optim")
            os.makedirs(optim_dir, exist_ok=True)
            torch.save(optimizer.state_dict(), os.path.join(optim_dir, "optimizer.pt"))

        def load_optimizer(self, optimizer, model, weights_path, scheduler=None):
            """Load optimizer state dict."""
            if optimizer is None:
                return
            optim_path = os.path.join(weights_path, "optim")
            optimizer.load_state_dict(torch.load(os.path.join(optim_path, "optimizer.pt"), weights_only=False))

        def async_wait(self):
            """No-op for tests to satisfy BaseRecipe interface."""
            return

        def maybe_wait_for_staging(self):
            """Record calls that wait for asynchronous checkpoint staging."""
            self.staging_waits += 1

        def save_on_dp_ranks(self, state, state_name, path):
            """Save stateful object (e.g., dataloader, rng)."""
            state_dir = os.path.join(path, state_name)
            os.makedirs(state_dir, exist_ok=True)
            if self.tp_rank == 0 and self.pp_rank == 0:
                torch.save(state.state_dict(), os.path.join(state_dir, f"{state_name}.pt"))

        def load_on_dp_ranks(self, state, state_name, path):
            """Load stateful object (e.g., dataloader, rng)."""
            state_dir = os.path.join(path, state_name)
            state.load_state_dict(torch.load(os.path.join(state_dir, f"{state_name}.pt"), weights_only=False))

        def save_distributed_state(self, state, state_name, path):
            """Save stateful object through distributed-checkpoint route."""
            self.distributed_saves.append((state_name, path))
            state_dir = os.path.join(path, state_name)
            os.makedirs(state_dir, exist_ok=True)
            torch.save(state.state_dict(), os.path.join(state_dir, f"{state_name}.pt"))

        def load_distributed_state(self, state, state_name, path):
            """Load stateful object through distributed-checkpoint route."""
            self.distributed_loads.append((state_name, path))
            state_dir = os.path.join(path, state_name)
            state.load_state_dict(torch.load(os.path.join(state_dir, f"{state_name}.pt"), weights_only=False))

    monkeypatch.setattr(checkpointing, "Checkpointer", MockCheckpointer)
    yield


class _DummyStateful:
    """
    Lightweight object that mimics the *load_state_dict/state_dict* API.
    """

    def __init__(self):
        """
        ctor
        """
        self.foo = torch.tensor(0.0)

    def state_dict(self):
        """
        retrieve state
        """
        return {"foo": self.foo.clone()}

    def load_state_dict(self, state):
        """
        restore state
        """
        self.foo = state["foo"].clone()


class _DummyDistributedStateful(_DummyStateful):
    """Stateful test object that opts into distributed checkpointing."""

    use_distributed_checkpointing = True


class _ToyModel(HFCheckpointingMixin, nn.Linear):
    """
    Toy model that inherits from HFCheckpointingMixin for testing save_pretrained.
    """

    def __init__(self, in_features, out_features, bias=False):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)


def _checkpoint_dir_names(path):
    """Return AutoModel checkpoint directory names under path sorted by step."""
    names = [p.name for p in path.glob("epoch_*_step_*") if p.is_dir()]
    return sorted(names, key=lambda name: int(name.rsplit("_", maxsplit=1)[1]))


def _simulate_interrupted_save(checkpoint_dir, step, epoch=0):
    """Create the on-disk remains of a save that died before publishing.

    Mirrors what a wall-clock kill leaves behind mid-save: the step directory
    exists and carries the in-progress marker, the model shard is half written,
    and ``optim/`` was never reached.

    Args:
        checkpoint_dir: Checkpoint root the recipe writes into.
        step: Step of the interrupted save.
        epoch: Epoch of the interrupted save.

    Returns:
        Path of the leftover checkpoint directory.
    """
    leftover = checkpoint_dir / f"epoch_{epoch}_step_{step}"
    (leftover / "model").mkdir(parents=True)
    (leftover / "model" / "model.pt").write_bytes(b"truncated")
    mark_checkpoint_incomplete(leftover)
    return leftover


class _ToyRecipe(BaseRecipe):
    """
    Minimal concrete implementation of BaseRecipe for testing.
    """

    def __init__(self, checkpoint_dir, cfg_dict=None, max_recent_checkpoints="default"):
        super().__init__()

        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        checkpoint_config_kwargs = dict(
            enabled=True,
            checkpoint_dir=str(checkpoint_dir),
            model_save_format="safetensors",
            model_cache_dir="",
            model_repo_id="",
            save_consolidated=False,
            is_peft=False,
            model_state_dict_keys=[],
        )
        if max_recent_checkpoints != "default":
            checkpoint_config_kwargs["max_recent_checkpoints"] = max_recent_checkpoints
        checkpoint_config = CheckpointingConfig(**checkpoint_config_kwargs)

        self.checkpointer = Checkpointer(
            config=checkpoint_config,
            dp_rank=0,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        )

        self.model = _ToyModel(2, 2, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.custom_state = _DummyStateful()
        self.peft_config = None

        if cfg_dict is None:
            cfg_dict = {"test": "config"}
        self.cfg = ConfigNode(cfg_dict)


def test_dp_allreduce_uses_world_group_without_device_mesh(tmp_path, monkeypatch):
    """
    DDP does not create a device mesh, so DP reductions should use the default
    process group instead of returning the rank-local tensor unchanged.
    """
    recipe_inst = _ToyRecipe(tmp_path)
    recipe_inst.device_mesh = None
    calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        calls.append((op, group, tensor.device))
        tensor.add_(6.0)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)

    reduced = recipe_inst._dp_allreduce(torch.tensor(2.0))

    assert reduced.item() == 8.0
    assert len(calls) == 1
    assert calls[0][0] == torch.distributed.ReduceOp.SUM
    assert calls[0][1] is None


def test_find_latest_checkpoint(tmp_path):
    """Verify that latest checkpoint discovery supports legacy step-based directory names."""
    (tmp_path / "epoch_0_step_1").mkdir()
    (tmp_path / "step_20").mkdir()
    (tmp_path / "epoch_3_step_5").mkdir()
    (tmp_path / "misc").mkdir()  # should be ignored

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "step_20", "Did not pick the highest step directory"


@pytest.mark.skipif(not HAS_ET, reason="expecttest required")
@pytest.mark.parametrize("symlink_supported", [True, False])
def test_save_and_load_roundtrip(tmp_path, symlink_supported, monkeypatch):
    """
    End-to-end test for BaseRecipe.save_checkpoint/load_checkpoint.

    The test:
      1. Creates a toy recipe.
      2. Performs a single optimizer step and mutates the extra stateful obj.
      3. Saves a checkpoint.
      4. Further mutates the model/extra-state.
      5. Calls load_checkpoint() and asserts that everything was restored to
         the values existing *at save time*.
    """
    print(expecttest)
    recipe_inst = _ToyRecipe(tmp_path)

    # Perform one training step so parameters / optimizer state differ from init.
    x = torch.randn(4, 2)
    recipe_inst.model.train()
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()

    # Mutate the auxiliary object.
    recipe_inst.custom_state.foo += 1

    # Snapshot for later comparison.
    weight_after_step = recipe_inst.model.weight.clone()
    foo_after_step = recipe_inst.custom_state.foo.clone()

    # Patch os.symlink to raise OSError if symlink_supported is False
    if not symlink_supported:

        def raise_os_error(*args, **kwargs):
            raise OSError("Symlink not supported")

        monkeypatch.setattr(os, "symlink", raise_os_error)

    # Save checkpoint.
    recipe_inst.save_checkpoint(epoch=0, step=0, train_loss=float(loss.item()))

    # Check that the correct indicator exists (symlink or text file)
    latest_link = tmp_path / "LATEST"
    latest_txt = tmp_path / "LATEST.txt"

    if symlink_supported:
        assert latest_link.exists(follow_symlinks=False)
        assert not latest_txt.exists()
    else:
        assert not latest_link.exists(follow_symlinks=False)
        assert latest_txt.exists()

    # Further modify everything so that restore must actually change data back.
    recipe_inst.model.weight.data.add_(42.0)
    recipe_inst.custom_state.foo += 5

    # Sanity check that things are indeed different now.
    assert not torch.allclose(recipe_inst.model.weight, weight_after_step)
    assert not torch.allclose(recipe_inst.custom_state.foo, foo_after_step)

    # Restore from latest checkpoint in the directory using 'LATEST' keyword.
    recipe_inst.load_checkpoint(restore_from="LATEST")

    # Expect exact values from the moment of save().
    assert torch.allclose(recipe_inst.model.weight, weight_after_step)
    assert torch.allclose(recipe_inst.custom_state.foo, foo_after_step)


def test_distributed_stateful_routes_through_distributed_checkpointing(tmp_path):
    """Objects that opt in use checkpointer distributed save/load hooks."""
    recipe_inst = _ToyRecipe(tmp_path)
    recipe_inst.distributed_state = _DummyDistributedStateful()
    recipe_inst.distributed_state.foo += 7
    saved_foo = recipe_inst.distributed_state.foo.clone()

    recipe_inst.save_checkpoint(epoch=0, step=10, train_loss=1.0)

    recipe_inst.distributed_state.foo += 13
    recipe_inst.load_checkpoint(restore_from="LATEST")

    ckpt_dir = str(tmp_path / "epoch_0_step_10")
    assert torch.allclose(recipe_inst.distributed_state.foo, saved_foo)
    assert recipe_inst.checkpointer.distributed_saves == [("distributed_state", ckpt_dir)]
    assert recipe_inst.checkpointer.distributed_loads == [("distributed_state", ckpt_dir)]


@pytest.mark.parametrize("wait_for_staging", [False, True])
def test_save_checkpoint_optionally_waits_for_async_staging(tmp_path, wait_for_staging):
    """The staging wait is opt-in so default async checkpoint behavior is unchanged."""
    recipe_inst = _ToyRecipe(tmp_path)
    recipe_inst.checkpointer.config.wait_for_staging = wait_for_staging

    recipe_inst.save_checkpoint(epoch=0, step=0, train_loss=1.0)

    assert recipe_inst.checkpointer.staging_waits == int(wait_for_staging)


def test_untrack_state_removes_state_from_checkpoint_tracking(tmp_path):
    """untrack_state lets recipes opt out of automatic checkpoint tracking."""
    recipe_inst = _ToyRecipe(tmp_path)
    recipe_inst.distributed_state = _DummyDistributedStateful()

    recipe_inst.untrack_state("distributed_state")
    recipe_inst.save_checkpoint(epoch=0, step=11, train_loss=1.0)

    assert recipe_inst.checkpointer.distributed_saves == []
    assert not (tmp_path / "epoch_0_step_11" / "distributed_state").exists()


def test_is_distributed_stateful_requires_opt_in_and_state_api():
    """The distributed checkpoint route is only for explicit stateful objects."""
    assert is_distributed_stateful(_DummyDistributedStateful()) is True
    assert is_distributed_stateful(_DummyStateful()) is False
    assert is_distributed_stateful(SimpleNamespace(use_distributed_checkpointing=True)) is False


def test_load_checkpoint_fresh_start_empty_dir(tmp_path):
    """
    Test that load_checkpoint() with restore_from=None and empty directory works (fresh start).
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Should succeed - no checkpoints exist
    recipe_inst.load_checkpoint(restore_from=None)


def test_setup_and_maybe_collect_garbage(tmp_path, monkeypatch):
    recipe_inst = _ToyRecipe(tmp_path)
    recipe_inst.step_scheduler = SimpleNamespace(gc_every_steps=2, step=1)

    class _GC:
        def __init__(self):
            self.run_called = []

        def run(self, step_count):
            self.run_called.append(step_count)

    gc_obj = _GC()
    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.GarbageCollection", lambda gc_every_steps: gc_obj)

    recipe_inst._setup_garbage_collection()
    assert recipe_inst.garbage_collector is gc_obj

    recipe_inst._maybe_collect_garbage()
    assert gc_obj.run_called == [1]

    recipe_inst.step_scheduler.step = 2
    recipe_inst._maybe_collect_garbage()
    assert gc_obj.run_called == [1, 2]


def test_load_checkpoint_auto_detect_restores_latest(tmp_path):
    """
    Test that load_checkpoint() with restore_from=None auto-detects and restores the
    latest checkpoint when one exists (the old default behavior).
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Perform training and save checkpoint
    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()

    weight_after_step = recipe_inst.model.weight.clone()
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    # Modify model
    recipe_inst.model.weight.data.add_(42.0)
    assert not torch.allclose(recipe_inst.model.weight, weight_after_step)

    # Load with restore_from=None should auto-detect and restore
    recipe_inst.load_checkpoint(restore_from=None)

    # Should restore to saved state
    assert torch.allclose(recipe_inst.model.weight, weight_after_step)


def test_load_checkpoint_with_latest_keyword(tmp_path):
    """
    Test that restore_from='LATEST' loads the latest checkpoint.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Perform training and save checkpoint
    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()

    weight_after_step = recipe_inst.model.weight.clone()
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    # Modify model
    recipe_inst.model.weight.data.add_(42.0)
    assert not torch.allclose(recipe_inst.model.weight, weight_after_step)

    # Load using 'LATEST' keyword
    recipe_inst.load_checkpoint(restore_from="LATEST")

    # Should restore to saved state
    assert torch.allclose(recipe_inst.model.weight, weight_after_step)


def test_load_checkpoint_with_latest_keyword_case_insensitive(tmp_path):
    """
    Test that restore_from='latest' (lowercase) also works.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Save checkpoint
    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()

    weight_after_step = recipe_inst.model.weight.clone()
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    # Modify model
    recipe_inst.model.weight.data.add_(42.0)

    # Load using lowercase 'latest'
    recipe_inst.load_checkpoint(restore_from="latest")

    # Should restore to saved state
    assert torch.allclose(recipe_inst.model.weight, weight_after_step)


@pytest.mark.parametrize("model,compatible", [("toy/model-a", True), ("toy/model-a2", False)])
def test_load_checkpoint_explicit_restore_incompatible_warns_and_continues(tmp_path, model, compatible):
    """
    When an explicit restore_from is given (e.g. "LATEST"), an incompatible
    checkpoint still proceeds with the restore -- the user explicitly asked
    for that checkpoint, so we honour the request and just warn.
    """
    # Create a checkpoint with a specific model signature
    recipe_a = _ToyRecipe(tmp_path, cfg_dict={"model": {"pretrained_model_name_or_path": "toy/model-a"}})

    x = torch.randn(4, 2)
    loss = recipe_a.model(x).sum()
    loss.backward()
    recipe_a.optimizer.step()
    weight_after_step = recipe_a.model.weight.clone()
    recipe_a.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    # Attempt to restore with a (possibly different) model signature using the 'LATEST' keyword
    recipe_b = _ToyRecipe(tmp_path, cfg_dict={"model": {"pretrained_model_name_or_path": model}})

    # Should NOT raise - always restores with explicit restore_from; warns if incompatible
    recipe_b.load_checkpoint(restore_from="LATEST")

    # Both compatible and incompatible cases restore the checkpoint weights
    # because restore_from was explicitly set.
    assert torch.allclose(recipe_b.model.weight, weight_after_step)


def test_load_checkpoint_autodetect_skips_incompatible(tmp_path):
    """
    When restore_from is None (auto-detect), an incompatible checkpoint is
    SKIPPED -- this prevents stale/leftover checkpoints from a different
    training run (e.g. PEFT vs full fine-tune) from breaking training.
    """
    # Create a checkpoint with model-a
    recipe_a = _ToyRecipe(tmp_path, cfg_dict={"model": {"pretrained_model_name_or_path": "toy/model-a"}})

    x = torch.randn(4, 2)
    loss = recipe_a.model(x).sum()
    loss.backward()
    recipe_a.optimizer.step()
    weight_after_step = recipe_a.model.weight.clone()
    recipe_a.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    # Create a new recipe with a DIFFERENT model signature and auto-detect (restore_from=None)
    recipe_b = _ToyRecipe(tmp_path, cfg_dict={"model": {"pretrained_model_name_or_path": "toy/model-b"}})
    weight_before_load = recipe_b.model.weight.clone()

    # Auto-detect should skip the incompatible checkpoint
    recipe_b.load_checkpoint(restore_from=None)

    # Weights should NOT have been restored (incompatible → skipped)
    assert torch.allclose(recipe_b.model.weight, weight_before_load)
    assert not torch.allclose(recipe_b.model.weight, weight_after_step)


def test_load_checkpoint_autodetect_skips_peft_mismatch(tmp_path):
    """
    A checkpoint saved with PEFT config is incompatible with a non-PEFT run
    (and vice-versa) because the checkpoint format differs (adapter-only vs
    full model). Auto-detect should skip such checkpoints.
    """
    # Save a checkpoint WITH a peft section in config
    recipe_peft = _ToyRecipe(
        tmp_path,
        cfg_dict={
            "model": {"pretrained_model_name_or_path": "toy/model-a"},
            "peft": {"dim": 8, "alpha": 32},
        },
    )
    x = torch.randn(4, 2)
    loss = recipe_peft.model(x).sum()
    loss.backward()
    recipe_peft.optimizer.step()
    recipe_peft.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    # Create a new recipe WITHOUT peft (same model architecture)
    recipe_no_peft = _ToyRecipe(
        tmp_path,
        cfg_dict={
            "model": {"pretrained_model_name_or_path": "toy/model-a"},
        },
    )
    weight_before_load = recipe_no_peft.model.weight.clone()

    # Auto-detect should skip because PEFT mismatch
    recipe_no_peft.load_checkpoint(restore_from=None)

    # Weights should NOT have been restored
    assert torch.allclose(recipe_no_peft.model.weight, weight_before_load)


def test_load_checkpoint_with_latest_no_checkpoints_warns(tmp_path):
    """
    Test that restore_from='LATEST' with no checkpoints warns and continues.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Should not raise, just warn and return
    recipe_inst.load_checkpoint(restore_from="LATEST")


def test_load_checkpoint_with_subdirectory_name(tmp_path):
    """
    Test that restore_from='epoch_0_step_100' (subdirectory name) works.
    This is a convenience feature - it looks in checkpoint_dir for the subdirectory.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Perform training and save checkpoint
    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()

    weight_after_step = recipe_inst.model.weight.clone()
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    # Modify model
    recipe_inst.model.weight.data.add_(42.0)
    assert not torch.allclose(recipe_inst.model.weight, weight_after_step)

    # Load using just the subdirectory name (no path separator)
    recipe_inst.load_checkpoint(restore_from="epoch_0_step_100")

    # Should restore to saved state
    assert torch.allclose(recipe_inst.model.weight, weight_after_step)


def test_load_checkpoint_with_full_path(tmp_path):
    """
    Test that restore_from with full path works.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Perform training and save checkpoint
    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()

    weight_after_step = recipe_inst.model.weight.clone()
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    # Modify model
    recipe_inst.model.weight.data.add_(42.0)

    # Load using full path
    ckpt_path = tmp_path / "epoch_0_step_100"
    recipe_inst.load_checkpoint(restore_from=str(ckpt_path))

    # Should restore to saved state
    assert torch.allclose(recipe_inst.model.weight, weight_after_step)


def test_load_checkpoint_nonexistent_subdirectory_fails(tmp_path):
    """
    Test that restore_from with non-existent subdirectory name fails with helpful error.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Create some checkpoints for the error message to list
    (tmp_path / "epoch_0_step_100").mkdir()
    (tmp_path / "epoch_0_step_200").mkdir()

    # Try to load non-existent checkpoint
    with pytest.raises(FileNotFoundError, match="Checkpoint directory does not exist"):
        recipe_inst.load_checkpoint(restore_from="epoch_0_step_999")


def test_load_checkpoint_nonexistent_path_fails(tmp_path):
    """
    Test that restore_from with non-existent full path fails with helpful error.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Try to load non-existent path
    with pytest.raises(FileNotFoundError, match="Checkpoint directory does not exist"):
        recipe_inst.load_checkpoint(restore_from=str(tmp_path / "nonexistent_checkpoint"))


def test_load_checkpoint_multiple_checkpoints_with_latest(tmp_path):
    """
    Test that 'LATEST' correctly picks the highest step number among multiple checkpoints.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Save multiple checkpoints
    for step in [50, 100, 75, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()

        if step == 200:
            # Save the state at step 200 for verification
            weight_at_step_200 = recipe_inst.model.weight.clone()

        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    # Modify model
    recipe_inst.model.weight.data.add_(42.0)

    # Load with LATEST - should pick step 200
    recipe_inst.load_checkpoint(restore_from="LATEST")

    # Should restore to step 200 state
    assert torch.allclose(recipe_inst.model.weight, weight_at_step_200)


def test_checkpoint_retention_default_keeps_all_checkpoints(tmp_path):
    """Without checkpoint.max_recent_checkpoints, checkpoint retention preserves existing keep-all behavior."""
    recipe_inst = _ToyRecipe(tmp_path)

    for step in [50, 100, 75, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert recipe_inst.checkpointer.config.max_recent_checkpoints is None
    assert _checkpoint_dir_names(tmp_path) == [
        "epoch_0_step_50",
        "epoch_0_step_75",
        "epoch_0_step_100",
        "epoch_0_step_200",
    ]


def test_checkpoint_retention_explicit_none_keeps_all_checkpoints(tmp_path):
    """checkpoint.max_recent_checkpoints=None keeps all checkpoints for users who need the full history."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=None)

    for step in [50, 100, 75, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == [
        "epoch_0_step_50",
        "epoch_0_step_75",
        "epoch_0_step_100",
        "epoch_0_step_200",
    ]


def test_step_scheduler_log_includes_checkpoint_retention_policy(tmp_path, caplog):
    """Startup logs should tell users whether checkpoint retention is bounded or disabled."""
    step_scheduler = SimpleNamespace(
        grad_acc_steps=1,
        ckpt_every_steps=5,
        gc_every_steps=None,
        epoch=0,
        num_epochs=1,
        val_every_steps=10,
        max_steps=20,
    )

    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=2)
    with caplog.at_level(logging.INFO):
        recipe_inst._log_step_scheduler_details(step_scheduler)

    assert "Checkpoint retention" in caplog.text
    assert "keeping the most recent 2 checkpoint directories" in caplog.text
    assert "plus pointer-protected checkpoints" in caplog.text
    assert "checkpoint.max_recent_checkpoints=2" in caplog.text

    caplog.clear()
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=None)
    with caplog.at_level(logging.INFO):
        recipe_inst._log_step_scheduler_details(step_scheduler)

    assert "disabled; keeping all checkpoints" in caplog.text
    assert "checkpoint.max_recent_checkpoints=None" in caplog.text

    caplog.clear()
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=2)
    recipe_inst.checkpointer.config.enabled = False
    with caplog.at_level(logging.INFO):
        recipe_inst._log_step_scheduler_details(step_scheduler)

    assert "inactive because checkpointing is disabled" in caplog.text
    assert "keeping the most recent" not in caplog.text


def test_save_checkpoint_replaces_interrupted_checkpoint_dir(tmp_path):
    """Re-saving a step whose previous save was interrupted replaces the leftover directory.

    Checkpoint directory names are derived from the step, so a resumed run
    recomputes the interrupted step and lands on the same path. Treating that as
    fatal aborted rank 0 while the other ranks blocked in the next collective,
    and every later resume window repeated the same collision.
    """
    recipe_inst = _ToyRecipe(tmp_path)
    leftover = _simulate_interrupted_save(tmp_path, step=100)

    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()
    weight_at_step_100 = recipe_inst.model.weight.clone()
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    assert not is_checkpoint_incomplete(leftover)
    assert (leftover / "optim").is_dir()
    assert (leftover / "model" / "model.pt").read_bytes() != b"truncated"

    recipe_inst.model.weight.data.add_(42.0)
    recipe_inst.load_checkpoint(restore_from="LATEST")
    assert torch.allclose(recipe_inst.model.weight, weight_at_step_100)


def test_save_checkpoint_cleanup_failure_is_reported_before_rank_0_raises(tmp_path, monkeypatch):
    """Rank 0 publishes an unremovable leftover through the reduction before it raises.

    Raising ahead of that collective is what left the remaining ranks blocked
    until the cluster reaped the job.
    """
    recipe_inst = _ToyRecipe(tmp_path)
    _simulate_interrupted_save(tmp_path, step=100)
    events = []

    def fail_rmtree(path):
        events.append("rmtree")
        raise OSError("device or resource busy")

    def record_all_reduce(tensor, op=None, group=None):
        events.append(("all_reduce", int(tensor.item())))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.shutil.rmtree", fail_rmtree)

    with pytest.raises(RuntimeError, match="could not be prepared") as excinfo:
        recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0)

    assert events == ["rmtree", ("all_reduce", 1)]
    assert isinstance(excinfo.value.__cause__, OSError)


def test_save_checkpoint_cleanup_failure_aborts_non_zero_ranks(tmp_path, monkeypatch):
    """A rank that did not perform the cleanup still aborts when rank 0's cleanup failed."""
    recipe_inst = _ToyRecipe(tmp_path)

    def rank_0_reported_failure(tensor, op=None, group=None):
        tensor.fill_(1)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", rank_0_reported_failure, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)

    with pytest.raises(RuntimeError, match="could not be prepared"):
        recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0)

    assert not (tmp_path / "epoch_0_step_100").exists()


def test_checkpoint_dir_is_marked_incomplete_while_the_save_is_in_flight(tmp_path, monkeypatch):
    """The in-progress marker covers the whole save window so an interruption stays detectable."""
    recipe_inst = _ToyRecipe(tmp_path)
    observed_during_save = []
    original_save_optimizer = recipe_inst.checkpointer.save_optimizer

    def record_then_save(optimizer, model, weights_path, scheduler=None):
        observed_during_save.append(is_checkpoint_incomplete(weights_path))
        return original_save_optimizer(optimizer, model, weights_path, scheduler)

    monkeypatch.setattr(recipe_inst.checkpointer, "save_optimizer", record_then_save)
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0)

    assert observed_during_save == [True]
    assert not is_checkpoint_incomplete(tmp_path / "epoch_0_step_100")


def test_async_checkpoint_stays_incomplete_until_it_is_published(tmp_path):
    """An async save is only resumable once the deferred publication runs.

    ``LATEST`` for an async save is advanced on the following call, so a job
    killed in between leaves a directory that resume must skip.
    """
    recipe_inst = _ToyRecipe(tmp_path)
    config = recipe_inst.checkpointer.config
    if not hasattr(config, "is_async"):
        raise ValueError("CheckpointingConfig has no field 'is_async'")
    config.is_async = True

    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0)

    checkpoint = tmp_path / "epoch_0_step_100"
    assert is_checkpoint_incomplete(checkpoint)
    assert find_latest_checkpoint(tmp_path) is None

    recipe_inst._complete_pending_checkpoint()

    assert not is_checkpoint_incomplete(checkpoint)
    assert os.path.basename(str(find_latest_checkpoint(tmp_path))) == "epoch_0_step_100"


def test_find_latest_checkpoint_skips_interrupted_checkpoint(tmp_path):
    """Without a LATEST pointer the step scan must not resume from an interrupted save."""
    (tmp_path / "epoch_0_step_100").mkdir()
    _simulate_interrupted_save(tmp_path, step=200)

    latest = find_latest_checkpoint(tmp_path)

    assert latest is not None
    assert latest.name == "epoch_0_step_100"


def test_find_latest_checkpoint_skips_interrupted_pointer_target(tmp_path):
    """A LATEST pointer aimed at an interrupted save falls back to the newest complete checkpoint.

    Re-saving a step that LATEST already targets rewrites that directory in
    place, so an interruption there leaves the pointer aimed at a partial
    checkpoint. Returning it would resume from half-written state.
    """
    complete = tmp_path / "epoch_0_step_50"
    complete.mkdir()
    interrupted = _simulate_interrupted_save(tmp_path, step=100)
    os.symlink(interrupted.name, tmp_path / "LATEST")

    latest = find_latest_checkpoint(tmp_path)

    assert latest is not None
    assert os.path.basename(str(latest)) == "epoch_0_step_50"


def test_find_latest_checkpoint_returns_none_when_only_pointer_target_is_interrupted(tmp_path):
    """With no complete checkpoint to fall back to, auto-resume starts fresh instead of loading a partial one."""
    interrupted = _simulate_interrupted_save(tmp_path, step=100)
    os.symlink(interrupted.name, tmp_path / "LATEST")

    assert find_latest_checkpoint(tmp_path) is None


def test_restore_from_pointer_skips_interrupted_target(tmp_path):
    """An explicit pointer restore does not resolve to a checkpoint an interrupted save left partial."""
    complete = tmp_path / "epoch_0_step_50"
    complete.mkdir()
    interrupted = _simulate_interrupted_save(tmp_path, step=100)
    os.symlink(interrupted.name, tmp_path / "LATEST")

    resolved = resolve_restore_from_to_checkpoint_dir(tmp_path, "LATEST")

    assert os.path.basename(str(resolved)) == "epoch_0_step_50"


def test_save_checkpoint_directory_creation_failure_aborts_every_rank(tmp_path, monkeypatch):
    """A rank-0 failure creating the checkpoint directory aborts collectively, not just on rank 0.

    ``os.makedirs`` and the marker write happen after the leftover cleanup; an
    I/O error there must reach the same reduction, otherwise rank 0 dies while
    its peers block in the next collective.
    """
    recipe_inst = _ToyRecipe(tmp_path)
    events = []

    def fail_makedirs(path, exist_ok=False):
        events.append("makedirs")
        raise OSError("read-only file system")

    def record_all_reduce(tensor, op=None, group=None):
        events.append(("all_reduce", int(tensor.item())))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.os.makedirs", fail_makedirs)

    with pytest.raises(RuntimeError, match="could not be prepared") as excinfo:
        recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0)

    assert events == ["makedirs", ("all_reduce", 1)]
    assert isinstance(excinfo.value.__cause__, OSError)


def test_save_checkpoint_marker_write_failure_aborts_every_rank(tmp_path, monkeypatch):
    """A rank-0 failure writing the in-progress marker aborts collectively."""
    recipe_inst = _ToyRecipe(tmp_path)
    events = []

    def fail_mark(checkpoint_dir):
        events.append("mark")
        raise OSError("disk quota exceeded")

    def record_all_reduce(tensor, op=None, group=None):
        events.append(("all_reduce", int(tensor.item())))

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True, raising=False)
    monkeypatch.setattr(torch.distributed, "all_reduce", record_all_reduce, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.mark_checkpoint_incomplete", fail_mark)

    with pytest.raises(RuntimeError, match="could not be prepared"):
        recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0)

    assert events == ["mark", ("all_reduce", 1)]


def test_publish_checkpoint_marker_clear_failure_does_not_raise(tmp_path, monkeypatch, caplog):
    """A failed marker clear leaves LATEST alone instead of killing rank 0.

    ``_publish_checkpoint`` runs on rank 0 only, so raising there would strand
    the other ranks. The checkpoint stays flagged incomplete, which resume
    already handles by falling back to the previous complete checkpoint.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    def fail_clear(checkpoint_dir):
        raise OSError("stale file handle")

    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.clear_checkpoint_incomplete", fail_clear)

    with caplog.at_level(logging.ERROR):
        recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0)

    assert "Failed to clear the in-progress marker" in caplog.text
    assert read_checkpoint_pointer(tmp_path, "LATEST") is None
    assert is_checkpoint_incomplete(tmp_path / "epoch_0_step_100")
    assert find_latest_checkpoint(tmp_path) is None


def test_save_losses_writes_json_for_local_checkpoint_dir(tmp_path):
    """The loss metadata helper still lands next to a local checkpoint."""
    save_losses({"train_loss": 1.5, "val_loss": 0.25}, str(tmp_path))

    with open(tmp_path / "losses.json") as f:
        assert json.load(f) == {"train_loss": 1.5, "val_loss": 0.25}


def test_save_losses_does_not_raise_when_the_directory_is_missing(tmp_path, caplog):
    """A failed loss-metadata write warns instead of raising on rank 0."""
    with caplog.at_level(logging.WARNING):
        save_losses({"train_loss": 1.0}, str(tmp_path / "absent"))

    assert "Failed to write checkpoint loss metadata" in caplog.text


def test_restore_from_named_pointer_rejects_interrupted_target(tmp_path):
    """A non-LATEST pointer at an interrupted save raises instead of resolving to the pointer path.

    Falling through would return ``<checkpoint_dir>/LOWEST_VAL``, which is the
    symlink itself and resolves straight back to the incomplete checkpoint.
    """
    interrupted = _simulate_interrupted_save(tmp_path, step=100)
    os.symlink(interrupted.name, tmp_path / "LOWEST_VAL")

    with pytest.raises(RuntimeError, match="LOWEST_VAL"):
        resolve_restore_from_to_checkpoint_dir(tmp_path, "LOWEST_VAL")


def test_marker_clear_failure_leaves_best_pointer_and_retention_untouched(tmp_path, monkeypatch):
    """When publication fails, no pointer is advanced and retention does not run.

    LOWEST_VAL must not end up aimed at a directory whose in-progress marker is
    still present.
    """
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    pruned = []
    monkeypatch.setattr(recipe_inst, "_prune_old_checkpoints", lambda: pruned.append(True))

    def fail_clear(checkpoint_dir):
        raise OSError("stale file handle")

    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.clear_checkpoint_incomplete", fail_clear)

    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0, val_loss={"val_loss": 0.1})

    assert is_checkpoint_incomplete(tmp_path / "epoch_0_step_100")
    assert read_checkpoint_pointer(tmp_path, "LATEST") is None
    assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL") is None
    assert pruned == []


def test_async_marker_clear_failure_leaves_best_pointer_untouched(tmp_path, monkeypatch):
    """Deferred async publication gates the best pointer on the same success flag."""
    recipe_inst = _ToyRecipe(tmp_path)
    config = recipe_inst.checkpointer.config
    if not hasattr(config, "is_async"):
        raise ValueError("CheckpointingConfig has no field 'is_async'")
    config.is_async = True

    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=1.0, val_loss={"val_loss": 0.1})

    def fail_clear(checkpoint_dir):
        raise OSError("stale file handle")

    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.clear_checkpoint_incomplete", fail_clear)
    recipe_inst._complete_pending_checkpoint()

    assert read_checkpoint_pointer(tmp_path, "LATEST") is None
    assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL") is None


def test_save_losses_does_not_raise_when_multistorageclient_is_missing(tmp_path, monkeypatch, caplog):
    """A missing MSC install warns instead of raising ImportError on rank 0.

    ``save_losses`` runs on rank 0 before a barrier, so any escaping exception
    strands the other ranks. Remote roots are rejected at config construction,
    but the helper must not be the one to break that contract.
    """
    import nemo_automodel.components.checkpoint.checkpointing as checkpointing_module

    monkeypatch.setattr(checkpointing_module, "MSC_AVAILABLE", False, raising=False)

    with caplog.at_level(logging.WARNING):
        save_losses({"train_loss": 1.0}, "msc://bucket/run/epoch_0_step_1")

    assert "Failed to write checkpoint loss metadata" in caplog.text


def test_checkpoint_retention_deletes_interrupted_checkpoint_a_pointer_targets(tmp_path):
    """A pointer must not keep an interrupted checkpoint alive, and is dropped with it.

    Re-saving a step that LOWEST_VAL already targets and then being interrupted
    leaves the pointer aimed at a checkpoint that can never be restored. Pointer
    protection would otherwise pin it forever and break the documented retention
    window.
    """
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    interrupted = _simulate_interrupted_save(tmp_path, step=100)
    recipe_inst._update_checkpoint_symlink("LOWEST_VAL", str(interrupted))

    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()
    recipe_inst.save_checkpoint(epoch=0, step=200, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_200"]
    assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL") is None


def test_interrupted_lowest_val_target_does_not_poison_best_metric_tracking(tmp_path):
    """A marked LOWEST_VAL target must not seed the best-metric baseline.

    Its ``losses.json`` records a metric the run can never restore. Seeding
    ``_best_val_loss`` from it leaves the cached value below every later
    checkpoint, so LOWEST_VAL is never recreated even after pruning removes the
    marked target and its pointer.
    """
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=2)
    interrupted = _simulate_interrupted_save(tmp_path, step=100)
    (interrupted / "losses.json").write_text('{"val_loss": 0.1}')
    recipe_inst._update_checkpoint_symlink("LOWEST_VAL", str(interrupted))

    for step, val in [(200, 0.2), (300, 0.15)]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()), val_loss={"val_loss": val})

    best = read_checkpoint_pointer(tmp_path, "LOWEST_VAL")
    assert best is not None, "LOWEST_VAL was never recreated after the marked target was pruned"
    assert best.name == "epoch_0_step_300"
    assert recipe_inst._best_val_loss == 0.15


def test_retention_removes_custom_pointers_left_dangling_by_pruning(tmp_path):
    """Any pointer AutoModel invalidates by deleting its target is cleaned up.

    The protection scan covers arbitrary top-level pointers, so dropping
    protection for a marked checkpoint can strand a user's own pointer. Only
    already-broken pointers are removed; a pointer with a live target is kept.
    """
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    interrupted = _simulate_interrupted_save(tmp_path, step=100)
    recipe_inst._update_checkpoint_symlink("PINNED", str(interrupted))
    kept = tmp_path / "epoch_0_step_150"
    kept.mkdir()
    recipe_inst._update_checkpoint_symlink("KEPT", str(kept))

    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()
    recipe_inst.save_checkpoint(epoch=0, step=200, train_loss=float(loss.item()))

    assert not (tmp_path / "epoch_0_step_100").exists()
    assert read_checkpoint_pointer(tmp_path, "PINNED") is None
    # A pointer whose target survived is untouched.
    assert read_checkpoint_pointer(tmp_path, "KEPT") is not None


def test_checkpoint_retention_still_protects_complete_pointer_target(tmp_path):
    """Pointer protection is unchanged for checkpoints that completed normally."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    complete = tmp_path / "epoch_0_step_100"
    complete.mkdir()
    recipe_inst._update_checkpoint_symlink("LOWEST_VAL", str(complete))

    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()
    recipe_inst.save_checkpoint(epoch=0, step=200, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_200"]
    assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL") is not None


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("restore_source", ["auto", "LATEST", "explicit"])
def test_interrupted_save_lifecycle_resumes_from_the_last_complete_checkpoint(tmp_path, is_async, restore_source):
    """Across sync/async saves and restore sources, resume never lands on a partial checkpoint.

    Saves step 100 normally, then simulates an interruption while re-saving it,
    which is the reported failure: the resumed run recomputes step 100 and the
    leftover directory is what it must not load.
    """
    recipe_inst = _ToyRecipe(tmp_path)
    config = recipe_inst.checkpointer.config
    if not hasattr(config, "is_async"):
        raise ValueError("CheckpointingConfig has no field 'is_async'")
    config.is_async = is_async

    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()
    weight_at_step_100 = recipe_inst.model.weight.clone()
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))
    if is_async:
        recipe_inst._complete_pending_checkpoint()

    # The re-save of step 100 is interrupted after the directory is reserved.
    checkpoint = tmp_path / "epoch_0_step_100"
    mark_checkpoint_incomplete(checkpoint)

    recipe_inst.model.weight.data.add_(42.0)
    if restore_source == "auto":
        assert find_latest_checkpoint(tmp_path) is None
        recipe_inst.load_checkpoint()
        # No complete checkpoint remains, so training legitimately starts fresh.
        assert not torch.allclose(recipe_inst.model.weight, weight_at_step_100)
    elif restore_source == "LATEST":
        assert resolve_restore_from_to_checkpoint_dir(tmp_path, "LATEST") is None
    else:
        # An explicit directory is the user's own choice and is still honoured.
        recipe_inst.load_checkpoint(restore_from="epoch_0_step_100")
        assert torch.allclose(recipe_inst.model.weight, weight_at_step_100)


def test_checkpoint_retention_evicts_interrupted_checkpoint_dir(tmp_path):
    """An interrupted checkpoint never occupies a retention slot and is not left behind."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=2)
    _simulate_interrupted_save(tmp_path, step=400)

    for step in [100, 200, 300]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_200", "epoch_0_step_300"]


def test_checkpoint_retention_max_recent_one_preserves_latest_resume(tmp_path):
    """checkpoint.max_recent_checkpoints=1 prunes older checkpoints and keeps LATEST resumable."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    for step in [50, 100, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        if step == 200:
            weight_at_step_200 = recipe_inst.model.weight.clone()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_200"]

    recipe_inst.model.weight.data.add_(42.0)
    recipe_inst.load_checkpoint(restore_from="LATEST")
    assert torch.allclose(recipe_inst.model.weight, weight_at_step_200)


def test_checkpoint_retention_max_recent_two_sliding_window(tmp_path):
    """checkpoint.max_recent_checkpoints=2 keeps the two highest-step checkpoint directories, plus pointer targets."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=2)

    for step in [50, 200, 100, 300]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_200", "epoch_0_step_300"]


def test_checkpoint_retention_ignores_non_automodel_step_directories(tmp_path):
    """Retention only prunes AutoModel-owned epoch_<n>_step_<n> checkpoint directories."""
    backup_dir = tmp_path / "backup_step_50"
    backup_dir.mkdir()
    (backup_dir / "keep.txt").write_text("not an AutoModel checkpoint")
    legacy_checkpoint = tmp_path / "step_300"
    legacy_checkpoint.mkdir()
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    for step in [100, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_200"]
    assert (backup_dir / "keep.txt").read_text() == "not an AutoModel checkpoint"
    assert legacy_checkpoint.is_dir()


def test_checkpoint_retention_ignores_non_pointer_text_files(tmp_path):
    """Plain text files do not pin old checkpoints unless they look like pointer fallback files."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    for step in [100, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))
        if step == 100:
            (tmp_path / "notes.txt").write_text("epoch_0_step_100/losses.json")

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_200"]
    assert (tmp_path / "notes.txt").exists()


def test_checkpoint_retention_preserves_lowest_val_pointer_target(tmp_path):
    """Retention preserves checkpoints targeted by LOWEST_VAL even outside the latest window."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    for step, val_loss in [(100, 0.1), (200, 0.9)]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(
            epoch=0,
            step=step,
            train_loss=float(loss.item()),
            val_loss={"val_loss": val_loss},
        )

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_200"]
    assert (tmp_path / "LOWEST_VAL").exists(follow_symlinks=False) or (tmp_path / "LOWEST_VAL.txt").exists()
    assert recipe_inst.load_checkpoint(restore_from="epoch_0_step_100") is None


def test_checkpoint_restore_from_lowest_val_text_fallback(tmp_path):
    """restore_from=LOWEST_VAL resolves text fallback pointers, not only symlinks."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()
    recipe_inst.save_checkpoint(
        epoch=0,
        step=100,
        train_loss=float(loss.item()),
        val_loss={"val_loss": 0.1},
    )
    (tmp_path / "LOWEST_VAL").unlink(missing_ok=True)
    (tmp_path / "LOWEST_VAL.txt").write_text("epoch_0_step_100")

    assert recipe_inst.load_checkpoint(restore_from="LOWEST_VAL") is None


def test_checkpoint_retention_preserves_arbitrary_checkpoint_pointer_target(tmp_path):
    """Retention preserves checkpoints targeted by any top-level checkpoint pointer."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    for step in [100, 200, 300]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))
        if step == 100:
            recipe_inst._update_checkpoint_symlink("PINNED", str(tmp_path / "epoch_0_step_100"))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_300"]
    assert (tmp_path / "PINNED").exists(follow_symlinks=False) or (tmp_path / "PINNED.txt").exists()


def test_checkpoint_retention_preserves_pointer_to_file_inside_checkpoint(tmp_path):
    """Retention preserves a checkpoint when a top-level pointer targets a file inside it."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    for step in [100, 200, 300]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))
        if step == 100:
            recipe_inst._update_checkpoint_symlink("PINNED_FILE", str(tmp_path / "epoch_0_step_100" / "losses.json"))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_300"]


def test_checkpoint_retention_preserves_text_fallback_pointer_target(tmp_path):
    """Retention preserves checkpoints targeted by symlink fallback text files."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    for step in [100, 200, 300]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))
        if step == 100:
            (tmp_path / "PINNED.txt").write_text("epoch_0_step_100/losses.json")

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_300"]


def test_checkpoint_retention_pointer_scan_failure_skips_pruning(tmp_path, monkeypatch, caplog):
    """Pointer discovery failures preserve every checkpoint rather than risking deletion of a pinned target."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    for step in [100, 200, 300]:
        (tmp_path / f"epoch_0_step_{step}").mkdir()
    recipe_inst._update_checkpoint_symlink("PINNED", str(tmp_path / "epoch_0_step_100"))

    def fail_pointer_scan(_ckpt_root, _checkpoints):
        raise OSError("pointer scan failed")

    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.find_pointer_protected_checkpoints", fail_pointer_scan)
    with caplog.at_level(logging.WARNING):
        recipe_inst._prune_old_checkpoints()

    assert _checkpoint_dir_names(tmp_path) == [
        "epoch_0_step_100",
        "epoch_0_step_200",
        "epoch_0_step_300",
    ]
    assert "skipping pruning" in caplog.text


def test_checkpoint_retention_pointer_classification_failure_skips_pruning(tmp_path, monkeypatch):
    """An I/O error while identifying a top-level pointer must fail closed."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    for step in [100, 200, 300]:
        (tmp_path / f"epoch_0_step_{step}").mkdir()
    recipe_inst._update_checkpoint_symlink("PINNED", str(tmp_path / "epoch_0_step_100"))
    path_type = type(tmp_path)
    real_lstat = path_type.lstat

    def fail_pinned_lstat(path, *args, **kwargs):
        if path == tmp_path / "PINNED":
            raise OSError("cannot classify pointer")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "lstat", fail_pinned_lstat)

    recipe_inst._prune_old_checkpoints()

    assert _checkpoint_dir_names(tmp_path) == [
        "epoch_0_step_100",
        "epoch_0_step_200",
        "epoch_0_step_300",
    ]


@pytest.mark.parametrize("non_finite_value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_restored_best_metric_does_not_block_future_best(tmp_path, non_finite_value):
    """A non-finite metric in LOWEST_VAL metadata is ignored when initializing the restored best."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    old_checkpoint = tmp_path / "epoch_0_step_100"
    new_checkpoint = tmp_path / "epoch_0_step_200"
    old_checkpoint.mkdir()
    new_checkpoint.mkdir()
    (old_checkpoint / "losses.json").write_text(f'{{"val_loss": {non_finite_value}}}')
    recipe_inst._update_checkpoint_symlink("LOWEST_VAL", str(old_checkpoint))

    recipe_inst._update_best_symlink(str(new_checkpoint), 0.5, "val_loss")

    assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL") == new_checkpoint
    assert recipe_inst._best_val_loss == 0.5


@pytest.mark.parametrize("malformed_losses", ["null", "42", "[]", '{"val_loss": 1' + "0" * 400 + "}"])
def test_malformed_restored_best_metric_does_not_block_future_best(tmp_path, malformed_losses):
    """Malformed LOWEST_VAL metadata is ignored when initializing the restored best."""
    recipe_inst = _ToyRecipe(tmp_path)
    old_checkpoint = tmp_path / "epoch_0_step_100"
    new_checkpoint = tmp_path / "epoch_0_step_200"
    old_checkpoint.mkdir()
    new_checkpoint.mkdir()
    (old_checkpoint / "losses.json").write_text(malformed_losses)
    recipe_inst._update_checkpoint_symlink("LOWEST_VAL", str(old_checkpoint))

    recipe_inst._update_best_symlink(str(new_checkpoint), 0.5, "val_loss")

    assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL") == new_checkpoint
    assert recipe_inst._best_val_loss == 0.5


def test_non_utf8_restored_best_metric_does_not_block_future_best(tmp_path):
    """Unreadable LOWEST_VAL metadata is ignored when initializing the restored best."""
    recipe_inst = _ToyRecipe(tmp_path)
    old_checkpoint = tmp_path / "epoch_0_step_100"
    new_checkpoint = tmp_path / "epoch_0_step_200"
    old_checkpoint.mkdir()
    new_checkpoint.mkdir()
    (old_checkpoint / "losses.json").write_bytes(b"\xff")
    recipe_inst._update_checkpoint_symlink("LOWEST_VAL", str(old_checkpoint))

    recipe_inst._update_best_symlink(str(new_checkpoint), 0.5, "val_loss")

    assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL") == new_checkpoint
    assert recipe_inst._best_val_loss == 0.5


@pytest.mark.parametrize("non_finite_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_live_metric_does_not_replace_best_pointer(tmp_path, non_finite_value):
    """A live non-finite validation metric is never eligible for LOWEST_VAL."""
    recipe_inst = _ToyRecipe(tmp_path)
    old_checkpoint = tmp_path / "epoch_0_step_100"
    new_checkpoint = tmp_path / "epoch_0_step_200"
    old_checkpoint.mkdir()
    new_checkpoint.mkdir()
    recipe_inst._update_best_symlink(str(old_checkpoint), 0.5, "val_loss")

    recipe_inst._update_best_symlink(str(new_checkpoint), non_finite_value, "val_loss")

    assert read_checkpoint_pointer(tmp_path, "LOWEST_VAL") == old_checkpoint
    assert recipe_inst._best_val_loss == 0.5


def test_checkpoint_pointer_replace_failure_preserves_existing_target(tmp_path, monkeypatch):
    """A failed atomic pointer swap leaves the previously published pointer intact."""
    recipe_inst = _ToyRecipe(tmp_path)
    old_checkpoint = tmp_path / "epoch_0_step_100"
    new_checkpoint = tmp_path / "epoch_0_step_200"
    old_checkpoint.mkdir()
    new_checkpoint.mkdir()
    recipe_inst._update_checkpoint_symlink("LATEST", str(old_checkpoint))
    real_replace = os.replace

    def fail_latest_replace(source, destination):
        if os.fspath(destination) == os.fspath(tmp_path / "LATEST"):
            raise OSError("filesystem full")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_latest_replace)

    with pytest.raises(OSError, match="filesystem full"):
        recipe_inst._update_checkpoint_symlink("LATEST", str(new_checkpoint))

    assert read_checkpoint_pointer(tmp_path, "LATEST") == old_checkpoint
    assert not list(tmp_path.glob(".LATEST.*.tmp"))


def test_checkpoint_retention_preserves_lowest_val_after_resume(tmp_path):
    """After resuming, a worse validation checkpoint must not replace the previous LOWEST_VAL target."""
    recipe_a = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    for step, val_loss in [(100, 0.1), (200, 0.9)]:
        x = torch.randn(4, 2)
        loss = recipe_a.model(x).sum()
        loss.backward()
        recipe_a.optimizer.step()
        recipe_a.save_checkpoint(
            epoch=0,
            step=step,
            train_loss=float(loss.item()),
            val_loss={"val_loss": val_loss},
        )

    recipe_b = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    recipe_b.load_checkpoint(restore_from="LATEST")
    x = torch.randn(4, 2)
    loss = recipe_b.model(x).sum()
    loss.backward()
    recipe_b.optimizer.step()
    recipe_b.save_checkpoint(
        epoch=0,
        step=300,
        train_loss=float(loss.item()),
        val_loss={"val_loss": 0.8},
    )

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_300"]


def test_checkpoint_retention_prune_failure_is_nonfatal(tmp_path, monkeypatch):
    """A failed retention delete leaves extra checkpoints instead of failing the save."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)

    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()
    recipe_inst.save_checkpoint(epoch=0, step=100, train_loss=float(loss.item()))

    def fail_rmtree(_path):
        raise OSError("checkpoint is busy")

    monkeypatch.setattr("nemo_automodel.recipes.base_recipe.shutil.rmtree", fail_rmtree)

    x = torch.randn(4, 2)
    loss = recipe_inst.model(x).sum()
    loss.backward()
    recipe_inst.optimizer.step()
    recipe_inst.save_checkpoint(epoch=0, step=200, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_200"]


def test_checkpoint_retention_unreadable_pointer_text_skips_pruning(tmp_path):
    """An unreadable pointer-like text file makes pointer discovery fail closed."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    (tmp_path / "PINNED.txt").write_bytes(bytes([0xFF, 0xFE]))

    for step in [100, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_200"]


def test_checkpoint_retention_async_finalization_prunes_pending_checkpoint(tmp_path):
    """Async saves prune only after the pending checkpoint has completed and is published."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    recipe_inst.checkpointer.config.is_async = True

    for step in [100, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_200"]

    recipe_inst._finalize_pending_checkpoint()

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_200"]
    assert (tmp_path / "LATEST").exists(follow_symlinks=False) or (tmp_path / "LATEST.txt").exists()


def test_load_checkpoint_path_with_separator_treated_as_full_path(tmp_path):
    """
    Test that restore_from containing path separator is treated as full path,
    not as subdirectory name.
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Create a nested structure
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    ckpt_dir = nested_dir / "epoch_0_step_100"
    ckpt_dir.mkdir()

    # Manually create checkpoint structure
    model_dir = ckpt_dir / "model"
    model_dir.mkdir()
    torch.save(recipe_inst.model.state_dict(), model_dir / "model.pt")

    optim_dir = ckpt_dir / "optim"
    optim_dir.mkdir()
    torch.save(recipe_inst.optimizer.state_dict(), optim_dir / "optimizer.pt")

    # Also save custom_state since BaseRecipe will try to load it
    torch.save(recipe_inst.custom_state.state_dict(), ckpt_dir / "custom_state.pt")

    # Load using relative path with separator
    recipe_inst.load_checkpoint(restore_from=str(nested_dir / "epoch_0_step_100"))


# ---------------------------------------------------------------------------
# Tests for _make_progress_bar and _update_progress_bar
# ---------------------------------------------------------------------------


class _FakeRecipe:
    """Minimal stand-in that exposes the two progress-bar helpers."""

    _make_progress_bar = BaseRecipe._make_progress_bar
    _update_progress_bar = BaseRecipe._update_progress_bar


def _make_fake_recipe(max_steps=10, step=0):
    r = _FakeRecipe()
    r.step_scheduler = SimpleNamespace(max_steps=max_steps, step=step)
    return r


class TestMakeProgressBar:
    def test_returns_tqdm_on_rank0(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        r = _make_fake_recipe(max_steps=100, step=5)
        pbar = r._make_progress_bar()
        assert pbar is not None
        assert pbar.total == 100
        assert pbar.n == 5
        pbar.close()

    def test_returns_none_on_non_rank0(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
        r = _make_fake_recipe()
        assert r._make_progress_bar() is None

    def test_tolerates_missing_max_steps(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        r = _FakeRecipe()
        r.step_scheduler = SimpleNamespace()  # no max_steps, no step
        pbar = r._make_progress_bar()
        assert pbar is not None
        assert pbar.total is None
        pbar.close()

    def test_explicit_total_and_initial_skip_step_scheduler(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        r = _FakeRecipe()  # deliberately no step_scheduler attribute
        pbar = r._make_progress_bar(total=40, initial=7)
        assert pbar is not None
        assert pbar.total == 40
        assert pbar.n == 7
        pbar.close()


class TestUpdateProgressBar:
    def _make_pbar(self):
        import io

        from tqdm import tqdm

        return tqdm(total=10, file=io.StringIO())

    def test_noop_when_pbar_is_none(self):
        r = _FakeRecipe()
        r._update_progress_bar(None, {"loss": 1.0})  # must not raise

    def test_uses_loss_key(self):
        r = _FakeRecipe()
        pbar = self._make_pbar()
        r._update_progress_bar(pbar, {"loss": 1.2345})
        assert "loss=1.2345" in pbar.postfix
        pbar.close()

    def test_falls_back_to_Loss_Train_Total(self):
        r = _FakeRecipe()
        pbar = self._make_pbar()
        r._update_progress_bar(pbar, {"Loss/Train_Total": 0.5})
        assert "loss=0.5000" in pbar.postfix
        pbar.close()

    def test_uses_lr_key(self):
        r = _FakeRecipe()
        pbar = self._make_pbar()
        r._update_progress_bar(pbar, {"lr": 1e-4})
        assert "lr=" in pbar.postfix
        pbar.close()

    def test_falls_back_to_Train_lr(self):
        r = _FakeRecipe()
        pbar = self._make_pbar()
        r._update_progress_bar(pbar, {"Train/lr": 2e-5})
        assert "lr=" in pbar.postfix
        pbar.close()

    def test_uses_tps_key(self):
        r = _FakeRecipe()
        pbar = self._make_pbar()
        r._update_progress_bar(pbar, {"tps": 512.7})
        assert "tps=513" in pbar.postfix
        pbar.close()

    def test_unknown_keys_produce_no_postfix(self):
        r = _FakeRecipe()
        pbar = self._make_pbar()
        r._update_progress_bar(pbar, {"unknown_metric": 99.0})
        assert not pbar.postfix
        pbar.close()

    # Should succeed without FileNotFoundError

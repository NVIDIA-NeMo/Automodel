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

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.recipes.base_recipe import BaseRecipe, _find_latest_checkpoint, is_distributed_stateful

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
    """Return checkpoint directory names under path sorted by step."""
    names = [p.name for p in path.glob("*step_*") if p.is_dir()]
    return sorted(names, key=lambda name: int(name.rsplit("_", maxsplit=1)[1]))


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
    """
    Verify that the helper returns the directory whose name contains the
    largest step number, irrespective of the exact prefix.
    """
    # Build a few fake checkpoint directories.
    (tmp_path / "epoch_0_step_1").mkdir()
    (tmp_path / "step_20").mkdir()
    (tmp_path / "epoch_3_step_5").mkdir()
    (tmp_path / "misc").mkdir()  # should be ignored

    latest = _find_latest_checkpoint(tmp_path)
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


def test_checkpoint_retention_default_keeps_last_two_checkpoints(tmp_path):
    """Without checkpoint.max_recent_checkpoints, checkpoint retention defaults to a bounded safety window."""
    recipe_inst = _ToyRecipe(tmp_path)

    for step in [50, 100, 75, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert recipe_inst.checkpointer.config.max_recent_checkpoints == 2
    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_100", "epoch_0_step_200"]


def test_checkpoint_retention_can_be_disabled(tmp_path):
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


def test_checkpoint_retention_preserves_lowest_val_after_resume(tmp_path):
    """After resume, a worse validation checkpoint must not replace the previous LOWEST_VAL target."""
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


def test_checkpoint_retention_ignores_unreadable_top_level_text_files(tmp_path):
    """Retention ignores unrelated text files that are not valid checkpoint pointers."""
    recipe_inst = _ToyRecipe(tmp_path, max_recent_checkpoints=1)
    (tmp_path / "NOT_A_POINTER.txt").write_bytes(bytes([0xFF, 0xFE]))

    for step in [100, 200]:
        x = torch.randn(4, 2)
        loss = recipe_inst.model(x).sum()
        loss.backward()
        recipe_inst.optimizer.step()
        recipe_inst.save_checkpoint(epoch=0, step=step, train_loss=float(loss.item()))

    assert _checkpoint_dir_names(tmp_path) == ["epoch_0_step_200"]


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

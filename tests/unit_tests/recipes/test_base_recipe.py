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

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.recipes.base_recipe import BaseRecipe, _find_latest_checkpoint
from nemo_automodel.components.config.loader import ConfigNode

try:
    import expecttest

    HAS_ET = True
except:
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
        
        def save_model(self, model=None, weights_path=None, peft_config=None, tokenizer=None):
            """Save model state dict."""
            if model is None:
                return
            model_dir = os.path.join(weights_path, "model")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

        def load_model(self, model, model_path, is_init_step=False, use_checkpoint_id=True,
                      key_mapping=None, quantization=False):
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


class _ToyModel(HFCheckpointingMixin, nn.Linear):
    """
    Toy model that inherits from HFCheckpointingMixin for testing save_pretrained.
    """

    def __init__(self, in_features, out_features, bias=False):
        nn.Linear.__init__(self, in_features, out_features, bias=bias)


class _ToyRecipe(BaseRecipe):
    """
    Minimal concrete implementation of BaseRecipe for testing.
    """

    def __init__(self, checkpoint_dir, cfg_dict=None):
        super().__init__()

        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        checkpoint_config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(checkpoint_dir),
            model_save_format="safetensors",
            model_cache_dir="",
            model_repo_id="",
            save_consolidated=False,
            is_peft=False,
            model_state_dict_keys=[],
        )

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


def test_load_checkpoint_fresh_start_empty_dir(tmp_path):
    """
    Test that load_checkpoint() with restore_from=None and empty directory works (fresh start).
    """
    recipe_inst = _ToyRecipe(tmp_path)

    # Should succeed - no checkpoints exist
    recipe_inst.load_checkpoint(restore_from=None)


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
def test_load_checkpoint_incompatible_model_warns_and_skips(tmp_path, model, compatible):
    """
    If a checkpoint doesn't match the current model config, warn and skip the restore
    (don't crash). This applies to both explicit 'LATEST' and auto-detect.
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
    weight_before_load = recipe_b.model.weight.clone()

    # Should NOT raise - incompatible checkpoints are warned and skipped
    recipe_b.load_checkpoint(restore_from="LATEST")

    if compatible:
        # Should have restored from checkpoint
        assert torch.allclose(recipe_b.model.weight, weight_after_step)
    else:
        # Should have skipped restore (incompatible), weights unchanged
        assert torch.allclose(recipe_b.model.weight, weight_before_load)

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

    # Should succeed without FileNotFoundError

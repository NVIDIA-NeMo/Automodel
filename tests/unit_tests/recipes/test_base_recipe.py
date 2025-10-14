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

import nemo_automodel.recipes.base_recipe as base_recipe
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
    from nemo_automodel.components.checkpoint import checkpointing_class

    class MockCheckpointer:
        """Mock Checkpointer for testing."""
        
        def __init__(self, config, dp_rank, tp_rank, pp_rank, moe_mesh=None):
            self.config = config
            self.dp_rank = dp_rank
            self.tp_rank = tp_rank
            self.pp_rank = pp_rank
            self.moe_mesh = moe_mesh
        
        def save_model(self, model, path, peft_config=None, tokenizer=None):
            """Save model state dict."""
            if model is None:
                return
            model_dir = os.path.join(path, "model")
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
    
    monkeypatch.setattr(checkpointing_class, "Checkpointer", MockCheckpointer)
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


class _ToyRecipe(BaseRecipe):
    """
    Minimal concrete implementation of BaseRecipe for testing.
    """

    def __init__(self, checkpoint_dir):
        super().__init__()
        
        from nemo_automodel.components.checkpoint.checkpointing_class import CheckpointingConfig, Checkpointer

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

        self.model = nn.Linear(2, 2, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.custom_state = _DummyStateful()
        self.peft_config = None

        self.cfg = ConfigNode({"test": "config"})


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
def test_save_and_load_roundtrip(tmp_path):
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

    # Save checkpoint.
    recipe_inst.save_checkpoint(epoch=0, step=0)

    # Further modify everything so that restore must actually change data back.
    recipe_inst.model.weight.data.add_(42.0)
    recipe_inst.custom_state.foo += 5

    # Sanity check that things are indeed different now.
    assert not torch.allclose(recipe_inst.model.weight, weight_after_step)
    assert not torch.allclose(recipe_inst.custom_state.foo, foo_after_step)

    # Restore from latest checkpoint in the directory.
    recipe_inst.load_checkpoint()

    # Expect exact values from the moment of save().
    assert torch.allclose(recipe_inst.model.weight, weight_after_step)
    assert torch.allclose(recipe_inst.custom_state.foo, foo_after_step)

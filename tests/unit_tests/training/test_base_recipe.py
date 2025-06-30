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

import nemo_automodel.components.training.base_recipe as base_recipe
from nemo_automodel.components.training.base_recipe import BaseRecipe, _find_latest_checkpoint

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
    Replace load_/save_model|optimizer with minimal torch.save/torch.load
    wrappers so that BaseRecipe can operate without the real NeMo helpers.
    """

    def _save_model(model, path, _cfg):
        if model is None:
            return
        torch.save(model.state_dict(), os.path.join(path, "model.pt"))

    def _load_model(model, path, _cfg):
        if model is None:
            return
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))

    def _save_optimizer(opt, _model, path):
        if opt is None:
            return
        torch.save(opt.state_dict(), os.path.join(path, "optimizer.pt"))

    def _load_optimizer(opt, _model, path):
        if opt is None:
            return
        opt.load_state_dict(torch.load(os.path.join(path, "optimizer.pt")))

    monkeypatch.setattr(base_recipe, "save_model", _save_model)
    monkeypatch.setattr(base_recipe, "load_model", _load_model)
    monkeypatch.setattr(
        base_recipe,
        "save_optimizer",
        _save_optimizer := _save_optimizer if "save_optimizer" in locals() else _save_optimizer,
    )
    monkeypatch.setattr(base_recipe, "load_optimizer", _load_optimizer)
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

        # The config object only needs the two attributes used by BaseRecipe.
        self.checkpoint_config = SimpleNamespace(enabled=True, checkpoint_dir=str(checkpoint_dir))

        self.model = nn.Linear(2, 2, bias=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.custom_state = _DummyStateful()


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

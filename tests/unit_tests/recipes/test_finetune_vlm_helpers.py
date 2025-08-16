# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import pytest
import torch
import torch.nn as nn

from nemo_automodel.recipes.vlm.finetune import _freeze_model, _build_optimizer, build_model_and_optimizer


@pytest.fixture(autouse=True)
def _mock_missing_cuda(monkeypatch):
    """Some helper functions unconditionally access torch.cuda APIs. When running on a
    CPU-only build they raise `RuntimeError: Torch not compiled with CUDA`.
    Patch the relevant CUDA APIs with no-op stubs when CUDA is unavailable."""
    if torch.cuda.is_available():
        yield  # nothing to do
        return

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [], raising=False)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None, raising=False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None, raising=False)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0, raising=False)
    yield


class DummyModel(nn.Module):
    """Simple model containing an embedding and a linear layer ("language_model")."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 4)
        # expose as attribute so apply_parameter_freezing can find it
        self.language_model = nn.Linear(4, 4)

    def forward(self, x):  # pragma: no cover – not needed for these unit tests
        return self.language_model(self.embedding(x))


class DummyOptConfig:
    """Mimics an optimizer config object with an *instantiate* method."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.foreach = None  # will be modified by _build_optimizer when tp_size > 1

    def instantiate(self, params):
        # Always return an SGD optimizer for the given params
        return torch.optim.SGD(params, lr=self.lr)

    def get(self, key, default):
        return getattr(self, key, default)

class DummyModelConfig:
    """Mimics the Hydra/OmegaConf model config with an *instantiate* method."""

    def instantiate(self):
        return DummyModel()


# -----------------------------------------------------------------------------
# _freeze_model
# -----------------------------------------------------------------------------

def test_freeze_model_embedding_only():
    """When cfg_freeze is *None*, embeddings should be frozen by default."""
    model = DummyModel()
    assert model.embedding.weight.requires_grad  # sanity check – initially True

    _freeze_model(model, cfg_freeze=None, freeze_embeddings=True)

    # embedding weights must be frozen; linear layer should still train
    assert not model.embedding.weight.requires_grad
    assert model.language_model.weight.requires_grad


def test_freeze_model_with_config(monkeypatch):
    """Custom freeze config should be respected (freeze_language_model only)."""
    model = DummyModel()

    freeze_cfg = {
        "freeze_embeddings": False,  # keep embeddings trainable
        "freeze_language_model": True,
    }

    _freeze_model(model, cfg_freeze=freeze_cfg, freeze_embeddings=False)

    # embedding remains trainable, language_model is frozen
    assert model.embedding.weight.requires_grad
    assert not model.language_model.weight.requires_grad


# -----------------------------------------------------------------------------
# _build_optimizer
# -----------------------------------------------------------------------------

def _count_trainable(p):
    return sum(x.numel() for x in p if x.requires_grad)


def test_build_optimizer_single_tp():
    model = DummyModel()
    cfg_opt = DummyOptConfig(lr=0.05)

    optim = _build_optimizer(model, cfg_opt, tp_size=1)

    # Optimizer should be torch.optim.Optimizer subclass with correct LR
    assert isinstance(optim, torch.optim.Optimizer)
    assert pytest.approx(optim.param_groups[0]["lr"], rel=1e-6) == 0.05

    # cfg_opt.foreach should remain untouched (None)
    assert cfg_opt.foreach is None


def test_build_optimizer_multi_tp():
    model = DummyModel()
    cfg_opt = DummyOptConfig(lr=0.02)

    optim = _build_optimizer(model, cfg_opt, tp_size=2)

    # Optimizer still constructed
    assert isinstance(optim, torch.optim.Optimizer)

    # tp_size > 1 must disable foreach behaviour
    assert cfg_opt.foreach in (False, None)


# -----------------------------------------------------------------------------
# build_model_and_optimizer
# -----------------------------------------------------------------------------

def test_build_model_and_optimizer_basic():
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    device = torch.device("cpu")
    model, optim = build_model_and_optimizer(
        device=device,
        cfg_model=cfg_model,
        cfg_opt=cfg_opt,
        cfg_freeze=None,
        cfg_peft=None,
        model_wrapper=None,
        seed=123,
        tp_size=1,
        freeze_embeddings=True,
    )

    # Check returned objects and their properties
    assert isinstance(model, DummyModel)
    assert isinstance(optim, torch.optim.Optimizer)

    # Model parameters should reside on the requested device
    assert next(model.parameters()).device == device

    # Embedding weights should be frozen by default
    assert not model.embedding.weight.requires_grad

    # Optimizer should hold only trainable parameters
    trainable_param_count = _count_trainable(model.parameters())
    optim_param_count = sum(p.numel() for group in optim.param_groups for p in group["params"])
    assert trainable_param_count == optim_param_count

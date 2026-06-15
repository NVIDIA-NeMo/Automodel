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
"""Engine.save_state / load_state (training resume) — single-process CPU via gloo."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from nemo_automodel.components.training.engine import CheckpointHandle, Engine


class ToyLM(nn.Module):
    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, position_ids=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


@pytest.fixture
def dist_gloo():
    """Single-process gloo process group so DCP has a backend."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29555",
            rank=0,
            world_size=1,
        )
    yield
    # leave the PG initialized for other tests in the session


def _engine(lr=0.5):
    torch.manual_seed(0)
    model = ToyLM()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    return Engine(model_parts=[model], optimizers=[opt]), model


# ── rank helpers (no mesh) ──────────────────────────────────────────────────


def test_rank_helpers_without_mesh():
    eng, _ = _engine()
    assert eng._tp_rank() == 0
    assert eng._pp_rank() == 0


# ── CheckpointHandle ────────────────────────────────────────────────────────


def test_handle_wait_noop_when_sync():
    fake = SimpleNamespace(config=SimpleNamespace(is_async=False))
    CheckpointHandle("p", fake).wait()  # must not raise / not call async_wait


# ── user_state round-trip (no DCP needed) ───────────────────────────────────


def test_user_state_roundtrip(tmp_path, dist_gloo):
    eng, _ = _engine()
    handle = eng.save_state(str(tmp_path / "ck"), user_state={"step": 7})
    assert isinstance(handle, CheckpointHandle)
    eng2, _ = _engine()
    us = eng2.load_state(str(tmp_path / "ck"))
    assert us == {"step": 7}


def test_load_state_returns_none_without_user_state(tmp_path, dist_gloo):
    eng, _ = _engine()
    eng.save_state(str(tmp_path / "ck2"))  # no user_state
    eng2, _ = _engine()
    assert eng2.load_state(str(tmp_path / "ck2")) is None


# ── full model+optimizer round-trip ─────────────────────────────────────────


def test_model_optimizer_resume_roundtrip(tmp_path, dist_gloo):
    # Train one step on engine A, snapshot, then mutate and restore.
    eng, model = _engine(lr=0.5)
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}

    def loss_fn(logits=None, labels=None, num_label_tokens=None):
        import torch.nn.functional as F

        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

    eng.forward_backward(batch, loss_fn=loss_fn)
    eng.optimizer_step()
    saved = model.head.weight.detach().clone()

    eng.save_state(str(tmp_path / "resume"))

    # Mutate weights, then restore from checkpoint.
    with torch.no_grad():
        model.head.weight.add_(1.0)
    assert not torch.allclose(model.head.weight, saved)

    eng.load_state(str(tmp_path / "resume"))
    assert torch.allclose(model.head.weight, saved, atol=1e-5)

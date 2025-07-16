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

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nemo_automodel.components.utils.dist_utils as du


class _FakeDist(SimpleNamespace):
    """
    Very small façade that satisfies the subset of the torch.distributed API
    that the utilities under test rely on.  It keeps the public surface narrow
    enough to be understandable at a glance.
    """

    def __init__(self, *, rank: int = 0, world_size: int = 1) -> None:
        super().__init__()
        self._rank = rank
        self._world_size = world_size
        # The code only needs ReduceOp.SUM
        self.ReduceOp = SimpleNamespace(SUM="sum")
        # Fabricate FSDP type hierarchy deep enough for isinstance checks
        fsdp_mod = SimpleNamespace(
            _fully_shard=SimpleNamespace(_fully_shard=SimpleNamespace(FSDPModule=type("DummyFSDP", (), {})))
        )
        self.fsdp = fsdp_mod
        self._initialised = True

    def is_initialized(self) -> bool:  # noqa: D401
        return self._initialised

    def get_rank(self) -> int:  # noqa: D401
        return self._rank

    def get_world_size(self) -> int:  # noqa: D401
        return self._world_size

    # All-reduce/barrier/abort/destroy are no-ops for the purpose of unit tests
    def all_reduce(self, *_, **__):  # noqa: D401
        pass

    def barrier(self, *_, **__):  # noqa: D401
        pass

    def abort(self, *_, **__):  # noqa: D401
        raise RuntimeError("abort called (simulated)")

    def destroy_process_group(self):  # noqa: D401
        self._initialised = False


@pytest.fixture()
def patch_dist(monkeypatch):
    """
    Replace `torch.distributed` **inside the utils module** with a lightweight
    fake implementation so tests do not need an actual back-end (NCCL / Gloo).
    """
    fake = _FakeDist()
    monkeypatch.setattr(du.torch, "distributed", fake, raising=False)
    # The module keeps a short alias ``dist``; patch it as well
    monkeypatch.setattr(du, "dist", fake, raising=False)
    yield fake


def test_rank_helpers_single_process(patch_dist):
    """
    get_rank_safe / get_world_size_safe / get_local_rank_preinit must fall back
    to env-vars when the fake process group is initialised.
    """
    os.environ.pop("RANK", None)
    os.environ.pop("WORLD_SIZE", None)
    os.environ.pop("LOCAL_RANK", None)

    assert du.get_rank_safe() == 0
    assert du.get_world_size_safe() == 1
    assert du.get_local_rank_preinit() == 0


def test_first_rank_per_node_single_gpu(monkeypatch, patch_dist):
    """
    In the absence of a distributed init the context manager should behave like
    a regular `nullcontext()`, returning True for the guarded block.
    """
    # Pretend that dist is *not* initialised for this test
    patch_dist._initialised = False
    monkeypatch.setattr(du.dist, "is_initialized", lambda: False, raising=False)

    with du.FirstRankPerNode() as is_first:
        assert is_first is True


def test_append_to_progress_log(tmp_path: Path, monkeypatch, patch_dist):
    """
    Verify that `progress.txt` is created and that the line contains the
    time-stamp, (fake) Job-ID and GPU count.
    """
    # Patch helpers so the function thinks we are rank-0
    monkeypatch.setattr(du, "get_rank_safe", lambda: 0)
    monkeypatch.setattr(du, "get_world_size_safe", lambda: 4)
    os.environ["SLURM_JOB_ID"] = "424242"

    du.append_to_progress_log(str(tmp_path), "unit-test ✓", barrier=False)

    log_file = tmp_path / "progress.txt"
    content = log_file.read_text()
    assert "unit-test ✓" in content
    assert "424242" in content
    assert "# GPUs: 4" in content


def test_reduce_loss_no_dp(monkeypatch):
    """
    With dp_group=None the routine must simply sum the supplied tensors and
    construct the correct denominator.
    """
    losses = [torch.tensor(1.0), torch.tensor(3.0)]
    tokens = torch.tensor(4)

    loss, denom = du.reduce_loss(losses, tokens, per_token_loss=True, dp_group=None)
    assert torch.isclose(loss, torch.tensor(4.0)), loss
    assert torch.equal(denom, torch.tensor(4)), denom


def test_get_sync_ctx(monkeypatch, patch_dist):
    """
    If the model is neither DDP nor FSDP the utility must return a
    `nullcontext`.
    """

    class Plain(torch.nn.Linear):
        pass

    ctx = du.get_sync_ctx(Plain(2, 2), is_optim_step=False)
    # entering/exiting the context must be a no-op
    with ctx:
        pass


def test_rescale_gradients_with_dp_group(monkeypatch, patch_dist):
    """
    Verify that `rescale_gradients` uses the size reported for the *given*
    dp_group (rather than the global world size) and that the dummy all-reduce
    is still called but leaves the tensor untouched.
    """
    # create a fake process-group handle.
    dp_group = object()  # any unique sentinel is fine

    # Report a group size of 3 for *this* handle.
    def _fake_get_ws(group=None):  # noqa: D401
        return 3 if group is dp_group else 1

    monkeypatch.setattr(patch_dist, "get_world_size", _fake_get_ws, raising=False)

    # make all_reduce a harmless no-op
    monkeypatch.setattr(patch_dist, "all_reduce", lambda *_, **__: None, raising=False)

    # build a toy model and attach gradient values of 1.0 everywhere.
    model = torch.nn.Linear(4, 4, bias=False)
    for p in model.parameters():
        p.grad = torch.ones_like(p.data)

    # scaling_factor = dp_group_size / num_tokens = 3 / 6 = 0.5
    du.rescale_gradients(model, torch.tensor(6), dp_group=dp_group)

    for p in model.parameters():
        assert torch.allclose(p.grad, torch.full_like(p.grad, 0.5)), p.grad


def test_clip_gradients(monkeypatch):
    """
    The util calls two *internal* helpers that do not exist on stock PyTorch
    builds.  Patch stubs in to observe the behaviour.
    """

    # Monkey-patch torch.nn.utils.{get_total_norm,clip_grads_with_norm_}
    def _get_total_norm(grads, foreach=False):  # noqa: D401
        return torch.linalg.vector_norm(torch.stack([g.flatten() for g in grads]))

    def _clip_grads_with_norm_(params, clip_norm, total_norm, foreach=False):  # noqa: D401
        for p in params:
            if p.grad is None:
                continue
            factor = clip_norm / (total_norm + 1e-6)
            p.grad.mul_(min(1.0, factor))

    monkeypatch.setattr(torch.nn.utils, "get_total_norm", _get_total_norm, raising=False)
    monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_", _clip_grads_with_norm_, raising=False)

    # Build a toy model with exaggerated gradient values.
    model = torch.nn.Linear(2, 2, bias=False)
    for p in model.parameters():
        p.grad = torch.full_like(p.data, 10.0)

    before = torch.linalg.vector_norm(torch.stack([p.grad.flatten() for p in model.parameters()]))
    assert before > 5

    # Clip and verify.
    grad_norm = du.clip_gradients(model, clip_norm=2.0, foreach=False)
    after = torch.linalg.vector_norm(torch.stack([p.grad.flatten() for p in model.parameters()]))

    assert torch.isclose(grad_norm, before)
    assert after <= 2.0 + 1e-5

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

"""Unit tests for :pyfile:`nemo_automodel/components/distributed/cp_utils.py`.

The real implementation relies heavily on ``torch.distributed`` and GPU-specific
behavior.  These unit-tests therefore *mock* the heavyweight distributed pieces
so they can run quickly on CPU-only CI systems while still verifying the public
contract of the helper utilities.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
import torch

# Import module under test
from nemo_automodel.components.distributed import cp_utils as _cu


class _DummySubMesh:
    """A minimal stub emulating ``torch.distributed.device_mesh.DeviceMesh`` slices."""

    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:  # noqa: D401  (simple method)
        return self._size


class _DummyDeviceMesh(dict):
    """Dictionary-like container expected by :pyfunc:`make_cp_batch_and_ctx`."""

    def __init__(self, cp_size: int, tp_size: int):
        super().__init__()
        self["context_parallel"] = _DummySubMesh(cp_size)
        self["tensor_parallel"] = _DummySubMesh(tp_size)


def test_build_position_ids_adds_missing():
    """If ``position_ids`` is absent it should be generated correctly."""
    batch: dict[str, Any] = {"input_ids": torch.arange(6).view(1, -1)}
    device = torch.device("cpu")

    returned = _cu._build_position_ids(batch, device)

    # Same object returned & mutated in-place
    assert returned is batch

    assert "position_ids" in batch, "position_ids key should be added"
    expected = torch.arange(batch["input_ids"].shape[1], device=device).unsqueeze(0)
    assert torch.equal(batch["position_ids"], expected), "Generated position_ids incorrect"


def test_build_position_ids_does_not_override_existing():
    """Existing ``position_ids`` must be left untouched."""
    original_pos = torch.tensor([[5, 4, 3]])
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "position_ids": original_pos.clone(),
    }

    _cu._build_position_ids(batch, torch.device("cpu"))
    assert torch.equal(batch["position_ids"], original_pos), "position_ids should not be modified"

def test_make_cp_batch_and_ctx_no_mesh():
    """When *no* device mesh is provided the call should be a no-op."""
    input_ids = torch.tensor([[1, 2, 3]])
    batch = {"input_ids": input_ids, "position_ids": torch.tensor([[0, 1, 2]])}
    labels = torch.tensor([[1, 2, 3]])

    ctx_obj, new_batch = _cu.make_cp_batch_and_ctx(None, batch, labels, loss_mask=None)

    # Expect the nullcontext *class* (not an instantiated object)
    assert ctx_obj is contextlib.nullcontext

    # Should hand back the *same* batch object
    assert new_batch is batch

    # Entering the context manager must be a no-op
    with ctx_obj():
        pass  # nothing should happen


def test_make_cp_batch_and_ctx_with_cp(monkeypatch):
    """Verify correct interaction when Context-Parallelism *is* enabled."""

    dummy_cp_ctx = object()

    def _fake_create_ctx(**kwargs):  # noqa: D401
        """Return a sentinel object so we can verify it was passed through."""
        return dummy_cp_ctx

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)

    def _fake_get_train_ctx(enable_loss_parallel, enable_compiled_autograd, cp_ctx):  # noqa: D401
        assert cp_ctx is dummy_cp_ctx, "create_context_parallel_ctx output should feed into get_train_context"
        return "dummy_train_ctx"

    monkeypatch.setattr(_cu, "get_train_context", _fake_get_train_ctx)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # CP enabled (>1)
    batch = {"input_ids": torch.tensor([[10, 20, 30]])}
    labels = torch.tensor([[10, 20, 30]])
    loss_mask = torch.tensor([[1, 1, 1]])

    ctx_obj, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch, labels, loss_mask)

    # We expect the stub training context to be returned
    assert ctx_obj == "dummy_train_ctx"

    # The function should have injected position_ids because CP>1
    assert "position_ids" in new_batch, "position_ids should be added when CP is enabled"
    expected_pos = torch.arange(batch["input_ids"].shape[1]).unsqueeze(0)
    assert torch.equal(new_batch["position_ids"], expected_pos)

    # Buffers inside *new_batch* should alias the originals (in-place modification)
    assert new_batch is batch 
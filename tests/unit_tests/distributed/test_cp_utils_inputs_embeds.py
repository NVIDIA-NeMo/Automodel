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

"""Tests for ``make_cp_batch_and_ctx`` accepting ``inputs_embeds`` as the
primary sequence tensor (VLM-CP path).

These cover:
  - XOR contract: exactly one of ``input_ids`` / ``inputs_embeds`` in batch
  - The cp_buffers list uses ``inputs_embeds`` when present
  - position_ids synthesis works whether ``input_ids`` or ``inputs_embeds`` is the source
  - ``cp_size <= 1`` short-circuit applies regardless of which key is present
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from nemo_automodel.components.distributed import cp_utils as _cu


class _DummySubMesh:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size

    def get_group(self):
        return None


class _DummyDeviceMesh(dict):
    def __init__(self, cp_size: int, tp_size: int):
        super().__init__()
        self["cp"] = _DummySubMesh(cp_size)
        self["tp"] = _DummySubMesh(tp_size)
        self.mesh_dim_names = ["cp", "tp"]


def test_xor_assertion_neither_present(monkeypatch):
    """Batch missing both input_ids AND inputs_embeds must raise AssertionError."""
    monkeypatch.setattr(_cu, "create_context_parallel_ctx", lambda **kw: object())
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    batch = {"labels": torch.zeros(1, 4, dtype=torch.long)}  # neither present
    with pytest.raises(AssertionError, match="exactly one of"):
        _cu.make_cp_batch_and_ctx(device_mesh, batch)


def test_xor_assertion_both_present(monkeypatch):
    """Batch with BOTH input_ids and inputs_embeds must raise AssertionError."""
    monkeypatch.setattr(_cu, "create_context_parallel_ctx", lambda **kw: object())
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    batch = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "inputs_embeds": torch.zeros(1, 4, 8),
        "labels": torch.zeros(1, 4, dtype=torch.long),
    }
    with pytest.raises(AssertionError, match="exactly one of"):
        _cu.make_cp_batch_and_ctx(device_mesh, batch)


def test_inputs_embeds_path_uses_embeds_as_primary_seq_tensor(monkeypatch):
    """When inputs_embeds is the only seq input, cp_buffers[0] must be inputs_embeds."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    inputs_embeds = torch.randn(1, 8, 16)  # [B=1, S=8, H=16]
    labels = torch.zeros(1, 8, dtype=torch.long)
    batch = {"inputs_embeds": inputs_embeds, "labels": labels}

    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    cp_buffers = captured["cp_buffers"]
    assert cp_buffers[0] is inputs_embeds, "primary cp buffer must be inputs_embeds"
    assert cp_buffers[1] is labels


def test_input_ids_path_unchanged(monkeypatch):
    """Standard LLM path (input_ids only) unchanged by the inputs_embeds extension."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    input_ids = torch.zeros(1, 8, dtype=torch.long)
    labels = torch.zeros(1, 8, dtype=torch.long)
    batch = {"input_ids": input_ids, "labels": labels}

    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    cp_buffers = captured["cp_buffers"]
    assert cp_buffers[0] is input_ids


def test_position_ids_synthesized_from_inputs_embeds_seq_dim(monkeypatch):
    """When position_ids is missing AND inputs_embeds is the primary, the
    synthesized arange must use ``inputs_embeds.shape[1]`` (the seq dim of the
    embed tensor, not its hidden dim)."""
    monkeypatch.setattr(_cu, "create_context_parallel_ctx", lambda **kw: object())
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    inputs_embeds = torch.randn(1, 12, 32)  # B=1, S=12, H=32
    batch = {"inputs_embeds": inputs_embeds, "labels": torch.zeros(1, 12, dtype=torch.long)}
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert "position_ids" in batch
    pos = batch["position_ids"]
    # Must be derived from S=12, NOT H=32
    assert pos.shape == (1, 12), f"expected [1,12], got {tuple(pos.shape)}"
    assert torch.equal(pos[0], torch.arange(12))


def test_inputs_embeds_no_op_when_cp_size_le_1():
    """cp_size<=1 short-circuit must apply to the inputs_embeds path too."""
    device_mesh = _DummyDeviceMesh(cp_size=1, tp_size=1)
    inputs_embeds = torch.randn(1, 4, 8)
    batch = {"inputs_embeds": inputs_embeds, "labels": torch.zeros(1, 4, dtype=torch.long)}

    ctx, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch)
    assert ctx is contextlib.nullcontext
    assert new_batch is batch
    # Must NOT inject position_ids when CP is off
    assert "position_ids" not in batch


def test_inputs_embeds_path_preserves_padding_mask_in_cp_buffers(monkeypatch):
    """If batch has padding_mask, it should ride along under the inputs_embeds path."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    inputs_embeds = torch.randn(1, 8, 16)
    pad_mask = torch.ones(1, 8, dtype=torch.bool)
    batch = {
        "inputs_embeds": inputs_embeds,
        "labels": torch.zeros(1, 8, dtype=torch.long),
        "padding_mask": pad_mask,
    }
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    # Use id()-based check: ``pad_mask in [tensors]`` doesn't work because
    # element-wise tensor equality requires matching shapes.
    assert any(b is pad_mask for b in captured["cp_buffers"])


def test_inputs_embeds_3d_position_ids_seq_dim(monkeypatch):
    """mRoPE 3D position_ids should still pick pos_seq_dim=2 even on the
    inputs_embeds path (seq sharding for embeds is still dim 1)."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    inputs_embeds = torch.randn(1, 8, 16)
    position_ids_3d = torch.randint(0, 8, (3, 1, 8))  # [3, B, S] mRoPE
    batch = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids_3d,
        "labels": torch.zeros(1, 8, dtype=torch.long),
    }
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    cp_seq_dims = captured["cp_seq_dims"]
    # [inputs_embeds, labels, position_ids] => [1, 1, 2]
    assert cp_seq_dims == [1, 1, 2]

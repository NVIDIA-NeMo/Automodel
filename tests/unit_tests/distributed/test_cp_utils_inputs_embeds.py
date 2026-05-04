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


def test_padding_pads_all_buffers_to_cp_divisor_multiple(monkeypatch):
    """seq_len=6 with cp_size=2 (divisor=4) -> pad_len=2.  All cp_buffers must
    be padded along their seq dim, and the original (padded) versions must be
    mirrored into the batch."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # divisor = 4
    seq_len = 6  # 6 % 4 = 2 -> pad to 8
    inputs_embeds = torch.randn(1, seq_len, 16)
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
    position_ids = torch.arange(seq_len).unsqueeze(0)
    batch = {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "position_ids": position_ids,
    }
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    cp_buffers = captured["cp_buffers"]
    expected_padded_len = 8
    # All buffers padded along seq dim
    assert cp_buffers[0].shape[1] == expected_padded_len, "inputs_embeds not padded"
    assert cp_buffers[1].shape[1] == expected_padded_len, "labels not padded"
    assert cp_buffers[2].shape[1] == expected_padded_len, "position_ids not padded"

    # Batch dict mirrored to padded versions (so the model forward sees padded shapes)
    assert batch["inputs_embeds"].shape[1] == expected_padded_len
    assert batch["labels"].shape[1] == expected_padded_len
    assert batch["position_ids"].shape[1] == expected_padded_len


def test_padding_labels_use_negative_100_int_buffers_use_zero(monkeypatch):
    """labels must pad with -100 (ignore_index for CE); other int buffers
    (input_ids, position_ids) pad with 0; float buffers (inputs_embeds) pad with
    zeros."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # divisor = 4
    inputs_embeds = torch.ones(1, 6, 4)
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
    position_ids = torch.arange(6).unsqueeze(0)
    batch = {"inputs_embeds": inputs_embeds, "labels": labels, "position_ids": position_ids}
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    # Pad region: indices [6, 7]
    padded_inputs_embeds = captured["cp_buffers"][0]
    padded_labels = captured["cp_buffers"][1]
    padded_position_ids = captured["cp_buffers"][2]

    # labels: original tail [5, 6], padded tail [-100, -100]
    assert padded_labels[0, 6].item() == -100
    assert padded_labels[0, 7].item() == -100
    # position_ids (int): pad with 0
    assert padded_position_ids[0, 6].item() == 0
    assert padded_position_ids[0, 7].item() == 0
    # inputs_embeds (float): pad with zeros
    assert torch.equal(padded_inputs_embeds[0, 6], torch.zeros(4))
    assert torch.equal(padded_inputs_embeds[0, 7], torch.zeros(4))
    # original content preserved
    assert torch.equal(padded_labels[0, :6], labels[0])
    assert torch.equal(padded_inputs_embeds[0, :6], inputs_embeds[0])


def test_padding_handles_loss_mask_and_padding_mask(monkeypatch):
    """When loss_mask and padding_mask ride along, both must also be padded
    to the cp-divisor multiple."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # divisor = 4
    seq_len = 6
    inputs_embeds = torch.randn(1, seq_len, 8)
    labels = torch.zeros(1, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    loss_mask = torch.ones(1, seq_len, dtype=torch.long)
    padding_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    batch = {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "position_ids": position_ids,
        "padding_mask": padding_mask,
    }
    _cu.make_cp_batch_and_ctx(device_mesh, batch, loss_mask=loss_mask)

    expected = 8
    # cp_buffers order: [primary, labels, position_ids, loss_mask, padding_mask]
    cp_buffers = captured["cp_buffers"]
    for i, buf in enumerate(cp_buffers):
        assert buf.shape[1] == expected, f"cp_buffers[{i}] not padded to {expected}: shape={tuple(buf.shape)}"


def test_padding_mirrors_padding_mask_back_into_batch(monkeypatch):
    """When padding triggers and the batch had a ``padding_mask``, the padded
    version must be mirrored back into ``batch["padding_mask"]`` so any
    downstream consumer reading the batch sees the matched shape (avoids a
    latent shape-mismatch trap if a future model accepts padding_mask)."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # divisor = 4
    seq_len = 6  # 6 % 4 = 2 -> pad to 8
    inputs_embeds = torch.randn(1, seq_len, 8)
    labels = torch.zeros(1, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    padding_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    batch = {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "position_ids": position_ids,
        "padding_mask": padding_mask,
    }
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    expected_padded_len = 8
    # All four batch entries reflect the padded shape
    assert batch["inputs_embeds"].shape[1] == expected_padded_len
    assert batch["labels"].shape[1] == expected_padded_len
    assert batch["position_ids"].shape[1] == expected_padded_len
    assert batch["padding_mask"].shape[1] == expected_padded_len
    # And the mirror is the same object the cp_buffers hold (not a stale copy)
    pmask_idx = next(
        i for i, b in enumerate(captured["cp_buffers"]) if b is batch["padding_mask"]
    )
    assert pmask_idx >= 3, "padding_mask must come after the primary trio"


def test_padding_no_op_when_seq_already_aligned(monkeypatch):
    """seq_len already divisible by cp_size*2 -> no padding needed; buffers
    must be the original objects (identity-preserving)."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # divisor = 4
    seq_len = 8  # 8 % 4 == 0 -> no pad
    inputs_embeds = torch.randn(1, seq_len, 16)
    labels = torch.zeros(1, seq_len, dtype=torch.long)
    batch = {"inputs_embeds": inputs_embeds, "labels": labels}
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    cp_buffers = captured["cp_buffers"]
    # Identity preserved (not a new tensor from torch.cat)
    assert cp_buffers[0] is inputs_embeds
    assert cp_buffers[1] is labels


def test_padding_input_ids_path_int_padding_with_zero(monkeypatch):
    """input_ids path (no inputs_embeds): integer dtype padded with 0, NOT -100
    (only labels get -100)."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # divisor = 4
    seq_len = 6
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
    batch = {"input_ids": input_ids, "labels": labels}
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    padded_input_ids = captured["cp_buffers"][0]
    padded_labels = captured["cp_buffers"][1]
    # input_ids (int) padded with 0
    assert padded_input_ids[0, 6].item() == 0
    assert padded_input_ids[0, 7].item() == 0
    # labels still padded with -100
    assert padded_labels[0, 6].item() == -100
    assert padded_labels[0, 7].item() == -100
    # batch mirrored
    assert batch["input_ids"].shape[1] == 8
    assert batch["labels"].shape[1] == 8


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

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
    def __init__(self, size: int, local_rank: int = 0):
        self._size = size
        self._local_rank = local_rank

    def size(self) -> int:
        return self._size

    def get_group(self):
        return None

    def get_local_rank(self) -> int:
        return self._local_rank


class _DummyDeviceMesh(dict):
    def __init__(self, cp_size: int, tp_size: int, cp_rank: int = 0):
        super().__init__()
        self["cp"] = _DummySubMesh(cp_size, cp_rank)
        self["tp"] = _DummySubMesh(tp_size)
        self.mesh_dim_names = ["cp", "tp"]


class _DummyHSDPDeviceMesh(_DummyDeviceMesh):
    def __init__(self, root_rank: int, cp_rank: int):
        super().__init__(cp_size=2, tp_size=1, cp_rank=cp_rank)
        self["dp_replicate"] = _DummySubMesh(2)
        self["dp_shard"] = _DummySubMesh(2)
        self.mesh_dim_names = ["dp_replicate", "dp_shard", "cp", "tp"]
        self._root_rank = root_rank

    def get_local_rank(self) -> int:
        return self._root_rank


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


def test_inputs_embeds_with_grad_is_sharded_out_of_place(monkeypatch):
    """Grad-bearing embeds must bypass context_parallel's in-place resize."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1, cp_rank=1)
    full_sequence = torch.arange(8.0, requires_grad=True)
    inputs_embeds = full_sequence.view(1, 8, 1)
    batch = {
        "inputs_embeds": inputs_embeds,
        "labels": torch.arange(8).view(1, 8),
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert all(not buffer.requires_grad for buffer in captured["cp_buffers"])
    assert torch.equal(batch["inputs_embeds"].flatten(), torch.tensor([2.0, 3.0, 4.0, 5.0]))

    batch["inputs_embeds"].sum().backward()
    assert torch.equal(full_sequence.grad, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    ("root_rank", "cp_rank", "expected"),
    [
        (0, 0, [0.0, 1.0, 6.0, 7.0]),
        (3, 1, [2.0, 3.0, 4.0, 5.0]),
        (4, 0, [0.0, 1.0, 6.0, 7.0]),
        (7, 1, [2.0, 3.0, 4.0, 5.0]),
    ],
)
def test_inputs_embeds_grad_sharding_uses_cp_submesh_rank_under_hsdp(monkeypatch, root_rank, cp_rank, expected):
    """HSDP root coordinates must not affect the rank used for CP sharding."""
    monkeypatch.setattr(_cu, "create_context_parallel_ctx", lambda **kw: object())
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyHSDPDeviceMesh(root_rank=root_rank, cp_rank=cp_rank)
    full_sequence = torch.arange(8.0, requires_grad=True)
    batch = {
        "inputs_embeds": full_sequence.view(1, 8, 1),
        "labels": torch.arange(8).view(1, 8),
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert batch["inputs_embeds"].flatten().tolist() == expected


def test_inputs_embeds_with_grad_and_cp_padding_preserves_global_token_mean(monkeypatch):
    """Uneven valid-token counts across padded CP shards must not bias gradients."""
    # Stress the production padding branch with 200 appended rows. CP=128
    # requires a multiple of 256 (2 * cp_size), so 56 valid rows become 256.
    # This leaves 72 CP ranks with no valid labels at all.
    cp_size = 128
    sequence_length = 56
    padded_sequence_length = 256
    padding_length = padded_sequence_length - sequence_length
    hidden_size = 3
    initial_weight = torch.arange(sequence_length * hidden_size, dtype=torch.float64).view(sequence_length, hidden_size)
    initial_weight = initial_weight / 13.0
    input_ids = torch.arange(sequence_length).unsqueeze(0)
    full_labels = torch.arange(sequence_length).unsqueeze(0)

    local_losses = []
    scaled_local_gradients = []
    for cp_rank in range(cp_size):
        captured = {}

        def _fake_create_ctx(**kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
        monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

        embedding = torch.nn.Embedding(sequence_length, hidden_size, dtype=torch.float64)
        with torch.no_grad():
            embedding.weight.copy_(initial_weight)
        batch = {
            "inputs_embeds": embedding(input_ids),
            "labels": full_labels.clone(),
        }

        _cu.make_cp_batch_and_ctx(_DummyDeviceMesh(cp_size=cp_size, tp_size=1, cp_rank=cp_rank), batch)

        # The grad-bearing primary tensor is already sharded out of place. The
        # ordinary labels remain in PyTorch's buffer list; independently apply
        # the same head-tail indices that the real CP context uses.
        padded_labels = captured["cp_buffers"][0]
        assert padded_labels.shape == (1, padded_sequence_length)
        assert torch.count_nonzero(padded_labels[:, sequence_length:] != -100) == 0
        assert padded_labels[:, sequence_length:].numel() == padding_length

        shard_indices = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1])
        local_labels = padded_labels.index_select(1, shard_indices)

        local_inputs = batch["inputs_embeds"]
        assert local_inputs.shape == (1, 2, hidden_size)
        local_inputs.retain_grad()
        valid = local_labels != -100
        per_token_loss = (local_inputs.sum(dim=-1) + 0.75).square()
        local_loss = per_token_loss[valid].sum() / sequence_length
        (local_loss * cp_size).backward()

        assert torch.count_nonzero(local_inputs.grad[~valid]) == 0
        local_losses.append(local_loss.detach())
        scaled_local_gradients.append(embedding.weight.grad.detach().clone())

    # FSDP averages gradients over DP*CP. The recipe multiplies each local
    # backward loss by that same size, so averaging these scaled rank-local
    # gradients must recover the full-sequence valid-token mean.
    candidate_loss = torch.stack(local_losses).sum()
    candidate_gradient = torch.stack(scaled_local_gradients).mean(dim=0)

    reference_embedding = torch.nn.Embedding(sequence_length, hidden_size, dtype=torch.float64)
    with torch.no_grad():
        reference_embedding.weight.copy_(initial_weight)
    reference_inputs = reference_embedding(input_ids)
    reference_loss = (reference_inputs.sum(dim=-1) + 0.75).square().sum() / sequence_length
    reference_loss.backward()

    torch.testing.assert_close(candidate_loss, reference_loss)
    torch.testing.assert_close(candidate_gradient, reference_embedding.weight.grad)


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


def test_position_ids_synthesized_for_each_batch_row(monkeypatch):
    """Synthesized 2D position_ids must match the batch dimension so later CP
    sharding keeps positions aligned with each sample."""
    monkeypatch.setattr(_cu, "create_context_parallel_ctx", lambda **kw: object())
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    inputs_embeds = torch.randn(3, 8, 16)
    batch = {"inputs_embeds": inputs_embeds, "labels": torch.zeros(3, 8, dtype=torch.long)}

    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert batch["position_ids"].shape == (3, 8)
    expected = torch.arange(8).expand(3, -1)
    assert torch.equal(batch["position_ids"], expected)


def test_singleton_position_ids_expand_to_batch_size(monkeypatch):
    """A caller-provided [1, S] position tensor should expand to [B, S] before
    the CP context records buffers."""
    captured = {}

    def _fake_create_ctx(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *a, **kw: "ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    inputs_embeds = torch.randn(2, 8, 16)
    position_ids = torch.arange(8).unsqueeze(0)
    batch = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "labels": torch.zeros(2, 8, dtype=torch.long),
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert batch["position_ids"].shape == (2, 8)
    assert captured["cp_buffers"][2] is batch["position_ids"]
    assert torch.equal(batch["position_ids"], torch.arange(8).expand(2, -1))


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


def test_padding_mask_pad_value_is_True_not_False(monkeypatch):
    """Regression: cp-divisor padding extends the seq with positions that are
    semantically padding.  ``padding_mask`` (bool, ``True`` == "this position
    is pad") must be padded with ``True``, NOT the dtype-default ``0``/``False``.
    Otherwise the MoE router treats the cp-pad slots as real tokens and routes
    them to experts every layer."""
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
    # Original padding_mask: 4 real, 2 pad
    padding_mask = torch.tensor([[False, False, False, False, True, True]])
    batch = {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "position_ids": position_ids,
        "padding_mask": padding_mask,
    }
    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    padded = batch["padding_mask"]
    assert padded.shape == (1, 8)
    # Positions 0..3 unchanged
    assert torch.equal(padded[0, :6], padding_mask[0])
    # Positions 6, 7 are cp-pad slots -- MUST be True (pad), not False
    assert bool(padded[0, 6].item()) is True, "cp-pad slot 6 should be marked as padding (True)"
    assert bool(padded[0, 7].item()) is True, "cp-pad slot 7 should be marked as padding (True)"


def test_padding_attention_mask_pad_value_is_zero(monkeypatch):
    """If a future caller passes an ``attention_mask`` in the batch, it should
    pad with ``0`` (HF convention: 1=real, 0=pad) -- NOT with True/dtype-default.

    Today ``cp_utils`` strips ``attention_mask`` at the top of the function so
    this case is moot, but the PAD_FILL table is the right place to encode the
    semantic in case the strip is ever revisited.
    """
    # Just verify the PAD_FILL table itself maps attention_mask -> False
    # (the runtime code path is currently unreachable because attention_mask
    # is popped at line 272).
    src = open(_cu.__file__).read()
    assert '"attention_mask": False' in src, "PAD_FILL must explicitly map attention_mask -> False (HF: 0 = pad)"


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
    pmask_idx = next(i for i, b in enumerate(captured["cp_buffers"]) if b is batch["padding_mask"])
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

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

    def get_group(self):  # noqa: D401  (simple method)
        """Return None to simulate no distributed process group."""
        return None


class _DummyDeviceMesh(dict):
    """Dictionary-like container expected by :pyfunc:`make_cp_batch_and_ctx`."""

    def __init__(self, cp_size: int, tp_size: int):
        super().__init__()
        self["cp"] = _DummySubMesh(cp_size)
        self["tp"] = _DummySubMesh(tp_size)
        self.mesh_dim_names = ["cp", "tp"]

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


# ============================================================================
# Tests for make_cp_batch_for_te
# ============================================================================


def test_make_cp_batch_for_te_no_cp(monkeypatch):
    """Test make_cp_batch_for_te with CP size 1."""
    cp_mesh = _DummySubMesh(size=1)

    # Create simple batch with 2 sequences in BSHD format
    # Batch size: 2, max_seq_len: 5
    # Actual lengths: [3, 5]
    input_ids = torch.tensor([
        [1, 2, 3, 0, 0],
        [4, 5, 6, 7, 8],
    ])
    labels = torch.tensor([
        [10, 20, 30, -100, -100],
        [40, 50, 60, 70, 80],
    ])
    seq_lens = torch.tensor([3, 5])
    seq_lens_padded = torch.tensor([4, 8])  # Padded to multiples of 4

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }

    def mock_get_rank(group=None):
        return 0

    def mock_pad_thd(input_ids, labels, cu_seqlens, divisibility_factor, padding_token_id, padding_label_id):
        # For cp_size=1, divisibility_factor=2, sequences should be padded to multiples of 2
        return input_ids, labels, cu_seqlens

    def mock_generate_pos_ids(cu_seqlens, divisibility_factor, dtype):
        return torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])

    def mock_get_batch(cu_seqlens_padded, input_ids_padded, labels_padded, position_ids_padded, cp_size, cp_rank, qvk_format):
        return input_ids_padded, labels_padded, position_ids_padded

    import nemo_automodel.components.distributed.te_cp_utils as te_cp
    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)
    monkeypatch.setattr(te_cp, "pad_thd_sequences_for_cp", mock_pad_thd)
    monkeypatch.setattr(te_cp, "generate_positional_ids_for_cp", mock_generate_pos_ids)
    monkeypatch.setattr(te_cp, "get_batch_on_this_cp_rank", mock_get_batch)

    result = _cu.make_cp_batch_for_te(
        cp_mesh=cp_mesh,
        batch=batch,
    )

    # Should always return processed batch
    assert "input_ids" in result
    assert "labels" in result
    assert "position_ids" in result
    assert "cu_seqlens" in result
    assert "cu_seqlens_padded" in result
    assert "qkv_format" in result








def test_make_cp_batch_for_te_unsupported_format():
    """Test that unsupported qvk_format raises ValueError."""
    cp_mesh = _DummySubMesh(size=2)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[10, 20, 30, 40]])
    seq_lens = torch.tensor([4])
    seq_lens_padded = torch.tensor([4])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }

    with pytest.raises(ValueError, match="Currently only 'thd' format is supported"):
        _cu.make_cp_batch_for_te(
            cp_mesh=cp_mesh,
            batch=batch,
            qvk_format="bshd",
        )


def test_make_cp_batch_for_te_padding_and_sharding(monkeypatch):
    """Test that make_cp_batch_for_te properly calls padding and sharding functions."""
    cp_mesh = _DummySubMesh(size=2)

    # BSHD format input
    input_ids = torch.tensor([[1, 2, 0, 0, 0], [3, 4, 5, 0, 0]])
    labels = torch.tensor([[10, 20, -100, -100, -100], [30, 40, 50, -100, -100]])
    seq_lens = torch.tensor([2, 3])
    seq_lens_padded = torch.tensor([4, 4])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }

    # Track function calls
    pad_called = []
    position_ids_called = []
    shard_called = []

    def mock_pad_thd_sequences_for_cp(input_ids, labels, cu_seqlens, divisibility_factor,
                                       padding_token_id, padding_label_id):
        pad_called.append({
            "divisibility_factor": divisibility_factor,
            "padding_token_id": padding_token_id,
            "padding_label_id": padding_label_id,
        })
        # Return padded versions (for CP size 2, divisibility_factor = 4)
        # Sequence 1: length 2 -> pad to 4
        # Sequence 2: length 3 -> pad to 4
        input_ids_padded = torch.tensor([1, 2, 0, 0, 3, 4, 5, 0])
        labels_padded = torch.tensor([10, 20, -100, -100, 30, 40, 50, -100])
        cu_seqlens_padded = torch.tensor([0, 4, 8])
        return input_ids_padded, labels_padded, cu_seqlens_padded

    def mock_generate_positional_ids_for_cp(cu_seqlens, divisibility_factor, dtype):
        position_ids_called.append({
            "divisibility_factor": divisibility_factor,
            "dtype": dtype,
        })
        # Generate position IDs for the padded sequences
        return torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    def mock_get_batch_on_this_cp_rank(cu_seqlens_padded, input_ids_padded,
                                        labels_padded, position_ids_padded,
                                        cp_size, cp_rank, qvk_format):
        shard_called.append({
            "cu_seqlens_padded": cu_seqlens_padded.tolist(),
            "cp_size": cp_size,
            "cp_rank": cp_rank,
            "qvk_format": qvk_format,
        })
        # Return the same tensors (simplified for testing)
        return input_ids_padded, labels_padded, position_ids_padded

    def mock_get_rank(group=None):
        """Mock torch.distributed.get_rank to return 0."""
        return 0

    # Mock the imported functions
    import nemo_automodel.components.distributed.te_cp_utils as te_cp
    monkeypatch.setattr(te_cp, "pad_thd_sequences_for_cp", mock_pad_thd_sequences_for_cp)
    monkeypatch.setattr(te_cp, "generate_positional_ids_for_cp", mock_generate_positional_ids_for_cp)
    monkeypatch.setattr(te_cp, "get_batch_on_this_cp_rank", mock_get_batch_on_this_cp_rank)
    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)

    result = _cu.make_cp_batch_for_te(
        cp_mesh=cp_mesh,
        batch=batch,
        padding_token_id=0,
        padding_label_id=-100,
    )

    # Verify pad function was called with correct divisibility factor (2 * cp_size = 4)
    assert len(pad_called) == 1
    assert pad_called[0]["divisibility_factor"] == 4
    assert pad_called[0]["padding_token_id"] == 0
    assert pad_called[0]["padding_label_id"] == -100

    # Verify position IDs generation was called
    assert len(position_ids_called) == 1
    assert position_ids_called[0]["divisibility_factor"] == 1
    assert position_ids_called[0]["dtype"] == torch.long

    # Verify sharding function was called with cp_size and cp_rank
    assert len(shard_called) == 1
    assert shard_called[0]["cu_seqlens_padded"] == [0, 4, 8]
    assert shard_called[0]["cp_size"] == 2
    assert shard_called[0]["cp_rank"] == 0
    assert shard_called[0]["qvk_format"] == "thd"

    # Verify output structure (new format)
    assert "input_ids" in result
    assert "labels" in result
    assert "position_ids" in result
    assert "cu_seqlens" in result
    assert "cu_seqlens_padded" in result
    assert "qkv_format" in result
    assert result["qkv_format"] == "thd"




def test_make_cp_batch_for_te_2d_input(monkeypatch):
    """Test make_cp_batch_for_te with 2D input tensors."""
    cp_mesh = _DummySubMesh(size=1)

    # 2D tensors with batch dimension
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[10, 20, 30, 40, 50]])
    seq_lens = torch.tensor([5])
    seq_lens_padded = torch.tensor([8])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }

    def mock_get_rank(group=None):
        return 0

    def mock_pad_thd(input_ids, labels, cu_seqlens, divisibility_factor, padding_token_id, padding_label_id):
        return input_ids, labels, cu_seqlens

    def mock_generate_pos_ids(cu_seqlens, divisibility_factor, dtype):
        return torch.tensor([0, 1, 2, 3, 4])

    def mock_get_batch(cu_seqlens_padded, input_ids_padded, labels_padded, position_ids_padded, cp_size, cp_rank, qvk_format):
        return input_ids_padded, labels_padded, position_ids_padded

    import nemo_automodel.components.distributed.te_cp_utils as te_cp
    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)
    monkeypatch.setattr(te_cp, "pad_thd_sequences_for_cp", mock_pad_thd)
    monkeypatch.setattr(te_cp, "generate_positional_ids_for_cp", mock_generate_pos_ids)
    monkeypatch.setattr(te_cp, "get_batch_on_this_cp_rank", mock_get_batch)

    result = _cu.make_cp_batch_for_te(
        cp_mesh=cp_mesh,
        batch=batch,
    )

    # Should handle 2D input correctly
    assert "input_ids" in result
    assert "labels" in result
    assert "position_ids" in result
    assert "cu_seqlens" in result
    assert "cu_seqlens_padded" in result
    assert "qkv_format" in result


def test_make_cp_batch_for_te_with_seq_lens(monkeypatch):
    """Test make_cp_batch_for_te with seq_lens for BSHD format input."""
    cp_mesh = _DummySubMesh(size=1)

    # BSHD format: batch_size=3, max_seq_len=5
    # Actual sequence lengths: [3, 2, 4]
    input_ids = torch.tensor([
        [1, 2, 3, 0, 0],      # seq 1: [1, 2, 3], padding: [0, 0]
        [4, 5, 0, 0, 0],      # seq 2: [4, 5], padding: [0, 0, 0]
        [6, 7, 8, 9, 0],      # seq 3: [6, 7, 8, 9], padding: [0]
    ])
    labels = torch.tensor([
        [10, 20, 30, -100, -100],
        [40, 50, -100, -100, -100],
        [60, 70, 80, 90, -100],
    ])
    seq_lens = torch.tensor([3, 2, 4])
    seq_lens_padded = torch.tensor([4, 4, 4])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }

    def mock_get_rank(group=None):
        return 0

    def mock_pad_thd(input_ids, labels, cu_seqlens, divisibility_factor, padding_token_id, padding_label_id):
        # Return as-is for simplicity
        return input_ids, labels, cu_seqlens

    def mock_generate_pos_ids(cu_seqlens, divisibility_factor, dtype):
        return torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])

    def mock_get_batch(cu_seqlens_padded, input_ids_padded, labels_padded, position_ids_padded, cp_size, cp_rank, qvk_format):
        return input_ids_padded, labels_padded, position_ids_padded

    import nemo_automodel.components.distributed.te_cp_utils as te_cp
    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)
    monkeypatch.setattr(te_cp, "pad_thd_sequences_for_cp", mock_pad_thd)
    monkeypatch.setattr(te_cp, "generate_positional_ids_for_cp", mock_generate_pos_ids)
    monkeypatch.setattr(te_cp, "get_batch_on_this_cp_rank", mock_get_batch)

    result = _cu.make_cp_batch_for_te(
        cp_mesh=cp_mesh,
        batch=batch,
    )

    # Should convert to THD format and remove padding
    assert "input_ids" in result
    assert "labels" in result
    assert "position_ids" in result
    assert "cu_seqlens" in result
    assert "cu_seqlens_padded" in result
    assert "qkv_format" in result

    # Input IDs should be concatenated without padding: [1,2,3,4,5,6,7,8,9]
    expected_input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert torch.equal(result["input_ids"], expected_input_ids)

    # Labels should be concatenated without padding: [10,20,30,40,50,60,70,80,90]
    expected_labels = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90])
    assert torch.equal(result["labels"], expected_labels)

    # cu_seqlens should be [0, 3, 5, 9]
    expected_cu_seqlens = torch.tensor([0, 3, 5, 9])
    assert torch.equal(result["cu_seqlens"], expected_cu_seqlens)

    # Position IDs should restart at each sequence
    expected_position_ids = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
    assert torch.equal(result["position_ids"], expected_position_ids)




def test_make_cp_batch_for_te_requires_seqlens():
    """Test that make_cp_batch_for_te raises error when seq_lens and seq_lens_padded are not provided."""
    cp_mesh = _DummySubMesh(size=1)

    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[10, 20, 30]])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
    }

    with pytest.raises(ValueError, match="BSHD format is not supported"):
        _cu.make_cp_batch_for_te(
            cp_mesh=cp_mesh,
            batch=batch,
        )

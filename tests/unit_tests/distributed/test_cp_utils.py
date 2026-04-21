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
    labels = torch.tensor([[1, 2, 3]])
    batch = {
        "input_ids": input_ids,
        "position_ids": torch.tensor([[0, 1, 2]]),
        "labels": labels,
    }

    ctx_obj, new_batch = _cu.make_cp_batch_and_ctx(None, batch, loss_mask=None)

    # Expect the nullcontext *class* (not an instantiated object)
    assert ctx_obj is contextlib.nullcontext

    # Should hand back the *same* batch object
    assert new_batch is batch

    # Entering the context manager must be a no-op
    with ctx_obj():
        pass  # nothing should happen


def test_make_cp_batch_and_ctx_with_cp(monkeypatch):
    """CP should pad and slice the batch locally, then return nullcontext."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # CP enabled (>1)
    labels = torch.tensor([[10, 20, 30, 40, 50, 60]])
    loss_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
    batch = {
        "input_ids": torch.tensor([[10, 20, 30, 40, 50, 60]]),
        "labels": labels,
    }

    ctx_obj, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch, loss_mask)

    assert ctx_obj is contextlib.nullcontext
    assert new_batch is batch
    assert torch.equal(new_batch["input_ids"], torch.tensor([[50, 60, 0, 0]]))
    assert torch.equal(new_batch["labels"], torch.tensor([[50, 60, -100, -100]]))
    assert torch.equal(new_batch["position_ids"], torch.tensor([[4, 5, 6, 7]]))


def test_make_cp_batch_and_ctx_includes_padding_mask(monkeypatch):
    """padding_mask should be sliced alongside the other sequence tensors."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    padding_mask = torch.tensor([[True, False, True, False, True, False]])
    batch = {
        "input_ids": torch.tensor([[10, 20, 30, 40, 50, 60]]),
        "labels": torch.tensor([[10, 20, 30, 40, 50, 60]]),
        "padding_mask": padding_mask,
    }

    _ctx, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch, loss_mask=None)

    assert torch.equal(new_batch["padding_mask"], torch.tensor([[True, False, True, False]]))


def test_make_cp_batch_and_ctx_3d_mrope_position_ids(monkeypatch):
    """3D mRoPE position_ids should shard on their sequence axis."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    seq_len = 6
    # mRoPE position_ids: [3, B, S] — temporal, height, width
    position_ids_3d = torch.arange(3 * 1 * seq_len).view(3, 1, seq_len)
    batch = {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "labels": torch.arange(seq_len).unsqueeze(0),
        "position_ids": position_ids_3d,
    }

    _ctx, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert torch.equal(new_batch["position_ids"], position_ids_3d.new_tensor([[[4, 5, 6, 7]], [[10, 11, 12, 13]], [[16, 17, 18, 19]]]))


def test_make_cp_batch_and_ctx_2d_position_ids_seq_dim(monkeypatch):
    """Standard 2D position_ids should shard on dim 1."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    seq_len = 6
    batch = {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "labels": torch.arange(seq_len).unsqueeze(0),
        "position_ids": torch.arange(seq_len).unsqueeze(0),
    }

    _ctx, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert torch.equal(new_batch["position_ids"], torch.tensor([[0, 1, 2, 3]]))


def test_make_cp_batch_and_ctx_3d_mrope_with_loss_mask(monkeypatch):
    """3D mRoPE should still work when loss_mask is provided."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    seq_len = 4
    position_ids_3d = torch.arange(3 * 1 * seq_len).view(3, 1, seq_len)
    loss_mask = torch.ones(1, seq_len)
    batch = {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "labels": torch.arange(seq_len).unsqueeze(0),
        "position_ids": position_ids_3d,
    }

    _ctx, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch, loss_mask=loss_mask)

    assert torch.equal(new_batch["position_ids"], position_ids_3d[:, :, :2])


def test_make_cp_batch_and_ctx_pops_attention_mask_when_cp_enabled(monkeypatch):
    """When CP is enabled, attention_mask should be removed from the batch."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    _ctx, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert "attention_mask" not in new_batch, "attention_mask should be removed when CP > 1"


def test_make_cp_batch_and_ctx_slices_mm_token_type_ids_and_per_layer_inputs(monkeypatch):
    """CP slicing must keep multimodal metadata aligned with the local shard."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    per_layer_inputs = torch.arange(1 * 6 * 2 * 3, dtype=torch.float32).view(1, 6, 2, 3)
    batch = {
        "inputs_embeds": torch.arange(1 * 6 * 4, dtype=torch.float32).view(1, 6, 4),
        "labels": torch.arange(6).view(1, 6),
        "mm_token_type_ids": torch.tensor([[1, 1, 0, 0, 0, 0]]),
        "per_layer_inputs": per_layer_inputs,
    }

    _ctx, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert torch.equal(new_batch["mm_token_type_ids"], torch.tensor([[0, 0, 0, 0]]))
    expected_per_layer_inputs = torch.nn.functional.pad(per_layer_inputs, (0, 0, 0, 0, 0, 2))[:, 4:8]
    assert torch.equal(new_batch["per_layer_inputs"], expected_per_layer_inputs)


# ============================================================================
# Tests for attach_context_parallel_hooks
# ============================================================================


class _FakeSelfAttn(torch.nn.Module):
    """Minimal module that records the kwargs it receives."""

    def forward(self, hidden_states, **kwargs):
        self.last_kwargs = kwargs
        return hidden_states


class _FakeTransformerBlock(torch.nn.Module):
    """A toy model with a ``self_attn`` sub-module to test hook attachment."""

    def __init__(self):
        super().__init__()
        self.self_attn = _FakeSelfAttn()


class _FakeModel(torch.nn.Module):
    """Two-layer model with ``self_attn`` sub-modules."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeTransformerBlock(), _FakeTransformerBlock()])


def test_attach_context_parallel_hooks_registers_on_self_attn():
    """Hooks should be registered on every module whose name ends with 'self_attn'."""
    model = _FakeModel()

    # Count hooks before
    hooks_before = {
        name: len(mod._forward_pre_hooks) for name, mod in model.named_modules() if name.endswith("self_attn")
    }

    _cu.attach_context_parallel_hooks(model)

    for name, mod in model.named_modules():
        if name.endswith("self_attn"):
            assert len(mod._forward_pre_hooks) == hooks_before[name] + 1


def test_attach_context_parallel_hooks_strips_attention_mask():
    """The hook should replace attention_mask with None and set is_causal=True."""
    model = _FakeModel()
    _cu.attach_context_parallel_hooks(model)

    dummy_input = torch.randn(1, 4, 8)
    attn_mask = torch.ones(1, 1, 4, 4)

    model.layers[0].self_attn(dummy_input, attention_mask=attn_mask)

    kwargs = model.layers[0].self_attn.last_kwargs
    assert kwargs["attention_mask"] is None, "attention_mask should be set to None by the hook"
    assert kwargs["is_causal"] is True, "is_causal should be set to True by the hook"


def test_attach_context_parallel_hooks_no_mask_passthrough():
    """When no attention_mask kwarg is passed, the hook should be a no-op."""
    model = _FakeModel()
    _cu.attach_context_parallel_hooks(model)

    dummy_input = torch.randn(1, 4, 8)
    model.layers[0].self_attn(dummy_input, some_other_kwarg=42)

    kwargs = model.layers[0].self_attn.last_kwargs
    assert "attention_mask" not in kwargs
    assert "is_causal" not in kwargs
    assert kwargs["some_other_kwarg"] == 42


def test_attach_context_parallel_hooks_skips_non_self_attn():
    """Modules not ending with 'self_attn' should have no hooks added."""
    model = _FakeModel()
    _cu.attach_context_parallel_hooks(model)

    # The top-level model and the layers list should not get hooks
    assert len(model._forward_pre_hooks) == 0
    assert len(model.layers._forward_pre_hooks) == 0
    for layer in model.layers:
        assert len(layer._forward_pre_hooks) == 0


# ============================================================================
# Tests for make_cp_batch_for_te
# ============================================================================


def test_make_cp_batch_for_te_basic(monkeypatch):
    """Test make_cp_batch_for_te with basic input."""
    cp_mesh = _DummySubMesh(size=2)

    # Create simple batch in BSHD format
    # 2 sequences: [1,2,3,4] and [5,6,7,8]
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
    position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    seq_lens = torch.tensor([[4], [4]])  # Both sequences have length 4
    seq_lens_padded = torch.tensor([[4], [4]])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "seq_lens": seq_lens,
        "seq_lens_padded": seq_lens_padded,
    }

    def mock_get_rank(group=None):
        return 0

    # Mock tex.thd_get_partitioned_indices to return all indices (simplified)
    def mock_thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
        # For simplicity, just return all indices
        return torch.arange(total_tokens)

    # Mock transformer_engine_torch module
    class MockTex:
        @staticmethod
        def thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
            return mock_thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank)

    # Mock at the module level where it's imported
    import sys

    sys.modules["transformer_engine_torch"] = MockTex

    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)

    result = _cu.make_cp_batch_for_te(
        cp_mesh=cp_mesh,
        batch=batch,
    )

    # Should return processed batch with correct keys
    assert "input_ids" in result
    assert "labels" in result
    assert "position_ids" in result
    assert "cu_seqlens" in result
    assert "max_seqlen" in result
    assert "qkv_format" in result
    assert "padding_mask" in result

    # Verify format
    assert result["qkv_format"] == "thd"

    # Verify cu_seqlens are properly formatted
    assert result["cu_seqlens"].dtype == torch.int32


def test_shard_thd_chunk_skips_missing_padding_mask(monkeypatch):
    """Test that _shard_thd_chunk_for_te handles missing padding_mask gracefully."""
    cp_mesh = _DummySubMesh(size=2)

    def mock_get_rank(group=None):
        return 0

    class MockTex:
        @staticmethod
        def thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
            return torch.arange(total_tokens)

    import sys
    sys.modules['transformer_engine_torch'] = MockTex

    monkeypatch.setattr(torch.distributed, "get_rank", mock_get_rank)

    # Batch without padding_mask — should not raise KeyError
    batch = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([10, 20, 30, 40]),
        "position_ids": torch.tensor([0, 1, 2, 3]),
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 4], dtype=torch.int32),
    }

    result = _cu._shard_thd_chunk_for_te(batch, cp_mesh, "thd", -1000, 0)

    assert "input_ids" in result
    assert "attention_mask" not in result


def test_make_cp_batch_for_te_unsupported_format():
    """Test that unsupported qvk_format raises ValueError."""
    cp_mesh = _DummySubMesh(size=2)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[10, 20, 30, 40]])
    seq_lens = torch.tensor([[4]])
    seq_lens_padded = torch.tensor([[4]])

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
            qkv_format="bshd",
        )


def test_make_cp_batch_for_te_requires_seqlens():
    """Test that make_cp_batch_for_te raises error when seq_lens and seq_lens_padded are not provided."""
    cp_mesh = _DummySubMesh(size=1)

    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[10, 20, 30]])

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": torch.tensor([[0, 1, 2]]),
    }

    with pytest.raises(KeyError, match="seq_lens"):
        _cu.make_cp_batch_for_te(
            cp_mesh=cp_mesh,
            batch=batch,
        )

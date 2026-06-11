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

    def __init__(self, size: int, local_rank: int = 0):
        self._size = size
        self._local_rank = local_rank

    def size(self) -> int:  # noqa: D401  (simple method)
        return self._size

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_group(self):  # noqa: D401  (simple method)
        """Return None to simulate no distributed process group."""
        return None


class _DummyDeviceMesh(dict):
    """Dictionary-like container expected by :pyfunc:`make_cp_batch_and_ctx`."""

    def __init__(self, cp_size: int, tp_size: int, cp_rank: int = 0):
        super().__init__()
        self["cp"] = _DummySubMesh(cp_size, cp_rank)
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
    """Verify correct interaction when Context-Parallelism *is* enabled."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)  # CP enabled (>1)
    # seq_len=4 is divisible by cp_size*2=4 so the cp-divisor padding path is
    # not exercised here (covered by test_cp_utils_inputs_embeds.py).
    labels = torch.tensor([[10, 20, 30, 40]])
    loss_mask = torch.tensor([[1, 1, 1, 1]])
    batch = {
        "input_ids": torch.tensor([[10, 20, 30, 40]]),
        "labels": labels,
        "_cp_manual_allgather": True,
    }

    ctx_obj, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch, loss_mask)

    assert ctx_obj is contextlib.nullcontext

    # The function should have injected position_ids because CP>1
    assert "position_ids" in new_batch, "position_ids should be added when CP is enabled"
    expected_pos = torch.tensor([[0, 1]])
    assert torch.equal(new_batch["position_ids"], expected_pos)
    assert torch.equal(new_batch["input_ids"], torch.tensor([[10, 20]]))
    assert torch.equal(new_batch["labels"], torch.tensor([[10, 20]]))
    assert torch.equal(new_batch["loss_mask"], torch.tensor([[1, 1]]))

    # Buffers inside *new_batch* should alias the originals (in-place modification)
    assert new_batch is batch


def test_make_cp_batch_and_ctx_pads_to_cp_load_balance_multiple(monkeypatch):
    """CP buffers should be padded to a multiple of 2 * cp_size."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1, cp_rank=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
        "_cp_manual_allgather": True,
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch, padding_token_id=99)

    assert batch["input_ids"].shape[1] == 2
    assert batch["input_ids"][0, -1].item() == 99
    assert batch["labels"][0, -1].item() == -100
    assert batch["mm_token_type_ids"][0, -1].item() == 0


def test_make_cp_batch_and_ctx_mm_token_type_ids_do_not_select_manual_allgather(monkeypatch):
    """VLM metadata alone should not opt models into manual all-gather CP."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    calls = {}

    def fake_create_context_parallel_ctx(**kwargs):
        calls["cp_buffers"] = kwargs["cp_buffers"]
        return "cp_ctx"

    def fake_get_train_context(enable_loss_parallel, enable_compiled_autograd, cp_context=None):
        calls["cp_context"] = cp_context
        return contextlib.nullcontext

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", fake_create_context_parallel_ctx)
    monkeypatch.setattr(_cu, "get_train_context", fake_get_train_context)

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": torch.tensor([[1, 2, 3, 4]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
    }

    ctx_obj, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch, padding_token_id=99)

    assert ctx_obj is contextlib.nullcontext
    assert calls["cp_context"] == "cp_ctx"
    assert len(calls["cp_buffers"]) == 3
    assert torch.equal(new_batch["input_ids"], torch.tensor([[1, 2, 3, 4]]))
    assert torch.equal(new_batch["mm_token_type_ids"], torch.tensor([[0, 1, 1, 0]]))


def test_make_cp_batch_and_ctx_supports_inputs_embeds_and_per_layer_inputs(monkeypatch):
    """Manual all-gather CP pre-embedding should shard inputs_embeds side inputs."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    inputs_embeds = torch.randn(1, 4, 8)
    labels = torch.tensor([[1, 2, 3, 4]])
    per_layer_inputs = torch.randn(1, 4, 2, 3)
    batch = {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "per_layer_inputs": per_layer_inputs,
        "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long),
        "_cp_manual_allgather": True,
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert batch["position_ids"].shape == (1, 2)
    assert batch["inputs_embeds"].shape == (1, 2, 8)
    assert batch["per_layer_inputs"].shape == (1, 2, 2, 3)
    assert torch.equal(batch["labels"], torch.tensor([[1, 2]]))


def test_make_cp_batch_and_ctx_pads_and_slices_packed_seq_ids(monkeypatch):
    """Packed document ids should stay aligned with the local CP shard."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1, cp_rank=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
        "_packed_seq_ids": torch.tensor([[1, 1, 2]]),
        "_cp_manual_allgather": True,
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch, padding_token_id=99)

    assert torch.equal(batch["input_ids"], torch.tensor([[3, 99]]))
    assert torch.equal(batch["labels"], torch.tensor([[3, -100]]))
    assert torch.equal(batch["_packed_seq_ids"], torch.tensor([[2, 0]]))


def test_make_cp_batch_and_ctx_includes_padding_mask(monkeypatch):
    """Verify that padding_mask is included in CP buffers when present in batch."""
    captured_kwargs = {}

    def _fake_create_ctx(**kwargs):
        captured_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(_cu, "create_context_parallel_ctx", _fake_create_ctx)
    monkeypatch.setattr(_cu, "get_train_context", lambda *_args, **_kw: "dummy_train_ctx")

    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    # seq_len=4 is divisible by cp_size*2=4 (no padding triggered).
    padding_mask = torch.tensor([[True, False, True, True]])
    batch = {
        "input_ids": torch.tensor([[10, 20, 30, 40]]),
        "labels": torch.tensor([[10, 20, 30, 40]]),
        "padding_mask": padding_mask,
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch, loss_mask=None)

    # padding_mask should be in cp_buffers
    assert any(t is padding_mask for t in captured_kwargs["cp_buffers"]), "padding_mask must be included in cp_buffers"
    assert padding_mask in captured_kwargs["cp_no_restore_buffers"]


def test_make_cp_batch_and_ctx_3d_mrope_position_ids(monkeypatch):
    """Verify that 3D mRoPE position_ids [3, B, S] are sharded on dim 2 (sequence), not dim 1 (batch)."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    seq_len = 8  # divisible by cp_size*2 to skip the cp-divisor padding path
    # mRoPE position_ids: [3, B, S] — temporal, height, width
    position_ids_3d = torch.arange(3 * 1 * seq_len).view(3, 1, seq_len)
    batch = {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "labels": torch.arange(seq_len).unsqueeze(0),
        "position_ids": position_ids_3d,
        "_cp_manual_allgather": True,
    }

    ctx_obj, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert ctx_obj is contextlib.nullcontext
    assert new_batch["position_ids"].shape == (3, 1, 4)
    assert torch.equal(new_batch["position_ids"], position_ids_3d[:, :, :4])


def test_make_cp_batch_and_ctx_2d_position_ids_seq_dim(monkeypatch):
    """Verify that standard 2D position_ids [B, S] are still sharded on dim 1."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    seq_len = 6
    batch = {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "labels": torch.arange(seq_len).unsqueeze(0),
        "position_ids": torch.arange(seq_len).unsqueeze(0),
        "_cp_manual_allgather": True,
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert torch.equal(batch["position_ids"], torch.tensor([[0, 1, 2, 3]]))


def test_make_cp_batch_and_ctx_3d_mrope_with_loss_mask(monkeypatch):
    """Verify 3D mRoPE position_ids work correctly with loss_mask."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    seq_len = 4
    position_ids_3d = torch.arange(3 * 1 * seq_len).view(3, 1, seq_len)
    loss_mask = torch.ones(1, seq_len)
    batch = {
        "input_ids": torch.arange(seq_len).unsqueeze(0),
        "labels": torch.arange(seq_len).unsqueeze(0),
        "position_ids": position_ids_3d,
        "_cp_manual_allgather": True,
    }

    _cu.make_cp_batch_and_ctx(device_mesh, batch, loss_mask=loss_mask)

    assert batch["position_ids"].shape == (3, 1, 2)
    assert torch.equal(batch["loss_mask"], torch.ones(1, 2))


def test_make_cp_batch_and_ctx_pops_attention_mask_when_cp_enabled(monkeypatch):
    """When CP is enabled, attention_mask should be removed from the batch."""
    device_mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    _ctx, new_batch = _cu.make_cp_batch_and_ctx(device_mesh, batch)

    assert "attention_mask" not in new_batch, "attention_mask should be removed when CP > 1"


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


class _FakeVLM(torch.nn.Module):
    """Toy VLM with language and vision self-attention modules."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = _FakeModel()
        self.model.vision_tower = torch.nn.Module()
        self.model.vision_tower.encoder = _FakeModel()


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


def test_attach_context_parallel_hooks_skips_vision_tower_self_attn():
    """CP hooks should not alter full-sequence vision tower attention."""
    model = _FakeVLM()

    _cu.attach_context_parallel_hooks(model)

    assert len(model.model.language_model.layers[0].self_attn._forward_pre_hooks) == 1
    assert len(model.model.vision_tower.encoder.layers[0].self_attn._forward_pre_hooks) == 0


def test_cp_attention_module_name_filter_excludes_multimodal_towers():
    assert _cu._is_cp_attention_module_name("model.language_model.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.vision_tower.encoder.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.vision_model.encoder.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.image_model.encoder.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.image_tower.encoder.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.audio_tower.encoder.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.audio_model.encoder.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.video_tower.encoder.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.video_model.encoder.layers.0.self_attn")
    assert not _cu._is_cp_attention_module_name("model.visual_model.encoder.layers.0.self_attn")


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

    sys.modules["transformer_engine_torch"] = MockTex

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


# ---------------------------------------------------------------------------
# Direct unit tests for the CP batch helper functions. These call the helpers
# directly (rather than only through ``make_cp_batch_and_ctx``) so each branch
# is exercised explicitly on CPU without a real process group.
# ---------------------------------------------------------------------------


def test_is_cp_non_text_module_path():
    assert _cu._is_cp_non_text_module_path("model.vision_tower.layers.0")
    assert _cu._is_cp_non_text_module_path("audio_model")
    assert not _cu._is_cp_non_text_module_path("model.language_model.layers.0.self_attn")
    assert not _cu._is_cp_non_text_module_path("")


def test_pad_tensor_seq_dim():
    t = torch.tensor([[1, 2, 3]])
    # No-op when pad_len <= 0.
    assert _cu._pad_tensor_seq_dim_(t, 1, 0) is t
    assert _cu._pad_tensor_seq_dim_(t, 1, -5) is t
    # Pads with the requested fill value along the sequence dim.
    out = _cu._pad_tensor_seq_dim_(t, 1, 2, value=99)
    assert torch.equal(out, torch.tensor([[1, 2, 3, 99, 99]]))
    # Works on a non-trailing seq dim too.
    t3 = torch.zeros(1, 2, 4)
    assert _cu._pad_tensor_seq_dim_(t3, 1, 3).shape == (1, 5, 4)


def test_pad_position_ids_seq_dim():
    pos = torch.tensor([[0, 1, 2]])
    # No-op when pad_len <= 0.
    assert _cu._pad_position_ids_seq_dim_(pos, 1, 0) is pos
    # Continues monotonically from the last position.
    out = _cu._pad_position_ids_seq_dim_(pos, 1, 3)
    assert torch.equal(out, torch.tensor([[0, 1, 2, 3, 4, 5]]))
    # 3D mRoPE position_ids [3, B, S] pad along seq dim 2.
    pos3 = torch.arange(6).view(3, 1, 2)
    out3 = _cu._pad_position_ids_seq_dim_(pos3, 2, 2)
    assert out3.shape == (3, 1, 4)
    assert torch.equal(out3[:, 0, 2:], pos3[:, 0, -1:].expand(3, 2) + torch.tensor([1, 2]))


def test_get_submesh():
    mesh = _DummyDeviceMesh(cp_size=2, tp_size=1)
    assert _cu._get_submesh(mesh, "cp") is mesh["cp"]
    assert _cu._get_submesh(mesh, "missing") is None
    # Objects without mesh_dim_names yield None.
    assert _cu._get_submesh(object(), "cp") is None


def test_get_mesh_size():
    assert _cu._get_mesh_size(None) == 0
    assert _cu._get_mesh_size(_DummySubMesh(size=4)) == 4


def test_cp_allgather_attention_context_is_frozen():
    fields = dict(
        module=torch.nn.Identity(),
        query=torch.zeros(1),
        key=torch.zeros(1),
        value=torch.zeros(1),
        key_full=torch.zeros(1),
        value_full=torch.zeros(1),
        cp_mesh=None,
        cp_group=None,
        cp_size=2,
        cp_rank=0,
        seq_local=4,
        seq_full=8,
        seq_global_start=0,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=None,
        enable_gqa=False,
        kwargs={},
        metadata={},
    )
    ctx = _cu.CPAllGatherAttentionContext(**fields)
    assert ctx.cp_size == 2 and ctx.is_causal is True
    with pytest.raises(Exception):
        ctx.cp_size = 3  # frozen dataclass


def test_prepare_cp_batch_common_4d_attention_mask():
    cp_mesh = _DummySubMesh(size=2)
    # 4D causal mask; diagonal True means "attend" -> padding_mask False there.
    attn = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    batch = {
        "input_ids": torch.arange(4).unsqueeze(0),
        "labels": torch.arange(4).unsqueeze(0),
        "attention_mask": attn,
    }
    primary_key, _, seq_len, labels, pos, pos_dim, loss_mask = _cu._prepare_cp_batch_common(cp_mesh, None, batch, None)
    assert primary_key == "input_ids" and seq_len == 4 and pos_dim == 1
    assert "attention_mask" not in batch
    assert "padding_mask" in batch and batch["padding_mask"].dtype == torch.bool
    # position_ids injected because cp_size > 1.
    assert torch.equal(pos, torch.arange(4).unsqueeze(0))


def test_prepare_cp_batch_common_2d_attention_mask_and_pos_expand():
    cp_mesh = _DummySubMesh(size=2)
    batch = {
        "input_ids": torch.arange(8).view(2, 4),
        "labels": torch.arange(8).view(2, 4),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),  # [1, S] expands to batch.
    }
    _, _, _, _, pos, _, _ = _cu._prepare_cp_batch_common(cp_mesh, None, batch, None)
    assert batch["padding_mask"].shape == (2, 4)
    assert pos.shape == (2, 4)


def test_prepare_cp_batch_common_labels_from_loss_mask():
    cp_mesh = _DummySubMesh(size=2)
    loss_mask = torch.tensor([[1, 1, 0, 0]])
    batch = {"input_ids": torch.arange(4).unsqueeze(0)}
    _, _, _, labels, _, _, returned_loss_mask = _cu._prepare_cp_batch_common(cp_mesh, None, batch, loss_mask)
    assert labels is loss_mask
    assert returned_loss_mask is None


def test_prepare_cp_batch_common_requires_labels():
    cp_mesh = _DummySubMesh(size=2)
    batch = {"input_ids": torch.arange(4).unsqueeze(0)}
    with pytest.raises(KeyError, match="labels"):
        _cu._prepare_cp_batch_common(cp_mesh, None, batch, None)


def test_prepare_cp_batch_common_requires_exactly_one_primary():
    cp_mesh = _DummySubMesh(size=2)
    both = {
        "input_ids": torch.arange(4).unsqueeze(0),
        "inputs_embeds": torch.zeros(1, 4, 8),
        "labels": torch.arange(4).unsqueeze(0),
    }
    with pytest.raises(AssertionError):
        _cu._prepare_cp_batch_common(cp_mesh, None, both, None)


def test_make_manual_allgather_cp_batch_pads_and_slices_all_keys():
    cp_mesh = _DummySubMesh(size=2, local_rank=0)
    # seq_len=3 -> pad to multiple of 2*cp_size=4 (pad_len=1), local shard=2.
    labels = torch.tensor([[5, 6, 7]])
    position_ids = torch.tensor([[0, 1, 2]])
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
        "_packed_seq_ids": torch.tensor([[1, 1, 2]]),
        "padding_mask": torch.tensor([[True, False, True]]),
    }
    loss_mask = torch.tensor([[1, 1, 1]])
    ctx, out = _cu._make_manual_allgather_cp_batch(
        cp_mesh,
        batch,
        primary_key="input_ids",
        seq_len=3,
        labels=labels,
        position_ids=position_ids,
        pos_seq_dim=1,
        loss_mask=loss_mask,
        padding_token_id=99,
    )
    assert ctx is contextlib.nullcontext
    # rank 0 keeps the first local_seq_len=2 tokens after padding to 4.
    assert torch.equal(out["input_ids"], torch.tensor([[1, 2]]))
    assert torch.equal(out["labels"], torch.tensor([[5, 6]]))
    assert torch.equal(out["position_ids"], torch.tensor([[0, 1]]))
    assert torch.equal(out["mm_token_type_ids"], torch.tensor([[0, 1]]))
    assert torch.equal(out["_packed_seq_ids"], torch.tensor([[1, 1]]))
    assert out["padding_mask"].shape == (1, 2)
    assert torch.equal(out["loss_mask"], torch.tensor([[1, 1]]))


def test_make_manual_allgather_cp_batch_second_rank_inputs_embeds():
    cp_mesh = _DummySubMesh(size=2, local_rank=1)
    labels = torch.tensor([[5, 6, 7, 8]])
    position_ids = torch.tensor([[0, 1, 2, 3]])
    # ``_prepare_cp_batch_common`` puts position_ids in the batch upstream; the
    # manual path only re-derives them into the batch when padding is applied.
    batch = {"inputs_embeds": torch.arange(4 * 2).float().view(1, 4, 2), "position_ids": position_ids}
    ctx, out = _cu._make_manual_allgather_cp_batch(
        cp_mesh,
        batch,
        primary_key="inputs_embeds",
        seq_len=4,
        labels=labels,
        position_ids=position_ids,
        pos_seq_dim=1,
        loss_mask=None,
        padding_token_id=0,
    )
    # seq_len=4 already a multiple of 2*cp_size=4 -> no padding; rank 1 keeps last 2.
    assert out["inputs_embeds"].shape == (1, 2, 2)
    assert torch.equal(out["labels"], torch.tensor([[7, 8]]))
    assert torch.equal(out["position_ids"], torch.tensor([[2, 3]]))

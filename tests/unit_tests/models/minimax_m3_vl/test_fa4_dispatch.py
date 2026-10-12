# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU dispatch boundaries only; actual FLASH kernels have separate CUDA tests."""

from collections.abc import Callable
from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import MiniMaxM3CPSparseAttention
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3Attention


def _attention() -> MiniMaxM3CPSparseAttention:
    """Construct on CPU with SDPA, then select FA4 without importing optional kernels."""
    config = MiniMaxM3VLTextConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        rotary_dim=32,
        num_hidden_layers=1,
        num_mtp_modules=0,
        sparse_attention_config={
            "sparse_num_index_heads": 2,
            "sparse_index_dim": 64,
            "sparse_block_size": 128,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
    )
    backend = BackendConfig(attn="sdpa", linear="torch", rope_fusion=False, experts="torch", dispatcher="torch")
    module = MiniMaxM3CPSparseAttention(config, backend)
    module.backend.attn = "fa4"
    return module


@pytest.mark.parametrize(
    "unsupported", ["cpu", "float32", "float64", "block64", "head32", "fused_rope", "thd", "window"]
)
def test_fa4_sparse_rejects_unsupported_dispatch(unsupported: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported shapes/backends before calling a sparse kernel."""
    module = _attention()
    hidden = Mock(spec=torch.Tensor)
    hidden.is_cuda = unsupported != "cpu"
    hidden.dtype = {"float32": torch.float32, "float64": torch.float64}.get(unsupported, torch.bfloat16)
    if unsupported == "block64":
        module.indexer.block_size = 64
    if unsupported == "head32":
        module.head_dim = 32
    if unsupported == "fused_rope":
        module.backend.rope_fusion = True
    generic = Mock(side_effect=AssertionError("FA4 sparse input escaped into generic attention"))
    local = Mock(side_effect=AssertionError("Unsupported input entered FLASH"))
    monkeypatch.setattr(MiniMaxM3Attention, "forward", generic)
    monkeypatch.setattr(module, "_local_sparse_forward", local)
    with pytest.raises(NotImplementedError, match="local FA4 sparse attention requires"):
        module(
            hidden,
            freqs_cis=torch.empty(1, 4, 32),
            qkv_format="thd" if unsupported == "thd" else "bshd",
            window_size=(127, 0) if unsupported == "window" else (-1, 0),
        )
    generic.assert_not_called()
    local.assert_not_called()


def test_fa4_supported_dispatch_uses_local_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dispatch supported unpacked CUDA inputs to the local sparse implementation."""
    module = _attention()
    hidden = Mock(spec=torch.Tensor)
    hidden.is_cuda, hidden.dtype = True, torch.bfloat16
    expected = torch.empty(1, 4, 128)
    local = Mock(return_value=expected)
    generic = Mock(side_effect=AssertionError("Sparse attention reached generic FA4"))
    monkeypatch.setattr(module, "_local_sparse_forward", local)
    monkeypatch.setattr(MiniMaxM3Attention, "forward", generic)
    assert module(hidden, freqs_cis=torch.empty(1, 4, 32)) is expected
    local.assert_called_once()
    generic.assert_not_called()


@pytest.mark.parametrize(
    "field",
    [
        "cu_seqlens",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_padded",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "max_seqlen",
        "max_seqlen_q",
        "max_seqlen_kv",
        "packed_token_indices",
        "_packed_seq_ids",
    ],
)
def test_fa4_sparse_rejects_compact_packing(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compact metadata cannot silently enter the explicit-mask sparse kernel."""
    module = _attention()
    hidden = Mock(spec=torch.Tensor)
    hidden.is_cuda, hidden.dtype = True, torch.bfloat16
    local = Mock(return_value=torch.empty(1, 4, 128))
    monkeypatch.setattr(module, "_local_sparse_forward", local)
    metadata: torch.Tensor | int = 2 if field.startswith("max_seqlen") else torch.tensor([0, 2, 4])
    with pytest.raises(NotImplementedError, match="does not support compact packed metadata"):
        module(hidden, freqs_cis=torch.empty(1, 4, 32), **{field: metadata})
    local.assert_not_called()


def test_fa4_explicit_document_mask_and_none_metadata_remain_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing explicit 4-D masks bypass no document-isolation metadata."""
    module = _attention()
    hidden = Mock(spec=torch.Tensor)
    hidden.is_cuda, hidden.dtype = True, torch.bfloat16
    mask = torch.eye(4, dtype=torch.bool)[None, None]
    expected = torch.empty(1, 4, 128)
    local = Mock(return_value=expected)
    monkeypatch.setattr(module, "_local_sparse_forward", local)
    assert module(hidden, freqs_cis=torch.empty(1, 4, 32), attention_mask=mask, cu_seqlens=None) is expected
    assert local.call_args.kwargs["attention_mask"] is mask


@pytest.mark.parametrize(
    "major,backend,rescue,length,head_specific,query_block",
    [
        (10, "fa4", False, 17, False, 256),
        (10, "fa4", False, 129, False, 256),
        (10, "fa4", False, 257, True, 256),
        (11, "fa4", False, 511, False, 256),
        (9, "fa4", False, 257, False, 128),
        (10, "sdpa", False, 257, False, 128),
        (10, "fa4", True, 257, False, 128),
    ],
)
def test_sparse_execution_tiles_preserve_selected_keys(
    major: int,
    backend: str,
    rescue: bool,
    length: int,
    head_specific: bool,
    query_block: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check actual CPU BlockMask metadata and token membership without invoking a CUDA kernel."""
    import torch.nn.attention.flex_attention as flex

    import nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn as cp

    module = _attention()
    module.backend.attn = backend
    capability = Mock(return_value=(major, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    create_block_mask = flex.create_block_mask

    def create_cpu_mask(
        mask_mod: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        B: int,
        H: int,
        Q_LEN: int,
        KV_LEN: int,
        device: torch.device,
        BLOCK_SIZE: int | tuple[int, int],
        _compile: bool,
    ) -> flex.BlockMask:
        return create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, device=device, BLOCK_SIZE=BLOCK_SIZE)

    monkeypatch.setattr(flex, "create_block_mask", create_cpu_mask)
    captured: list[flex.BlockMask] = []

    def capture_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        block_mask: flex.BlockMask,
        scale: float,
        enable_gqa: bool,
        kernel_options: dict[str, str] | None,
    ) -> torch.Tensor:
        """Record the production mask while bypassing only the CUDA execution.

        Args:
            q: Queries [mask_batch, query_heads, sequence, head_dim].
            k: Keys [mask_batch, kv_heads, sequence, head_dim].
            v: Values with the same layout as k.
            block_mask: Sparse metadata and the per-token mask predicate.
            scale: Query-key score multiplier.
            enable_gqa: Whether query heads share key/value heads.
            kernel_options: Selected kernel backend, or default Triton options.

        Returns:
            Zero output with q's layout; numerical attention is tested on CUDA.
        """
        assert kernel_options == ({"BACKEND": "FLASH"} if backend == "fa4" and not rescue else None)
        captured.append(block_mask)
        return torch.zeros_like(q)

    monkeypatch.setattr(cp, "_get_compiled_flash_attention", lambda: capture_attention)
    monkeypatch.setattr(cp, "_get_compiled_flex_attention", lambda: capture_attention)

    key_blocks = (length + 127) // 128
    # Give the first index head only the first 128 keys, the second only the
    # last key block. The independent oracle below enumerates these key ranges.
    selected = torch.zeros(1, 2, length, key_blocks, dtype=torch.bool)
    selected[:, 0, :, 0] = True
    selected[:, 1, :, -1] = True
    positions = torch.arange(length)
    documents = positions >= 128
    keep = (documents[:, None] == documents[None, :])[None, None]
    keep = keep.expand(1, 4 if head_specific else 1, length, length).clone()
    keep[:, :, -1, :] = False
    keep[:, :, :, -1] = False
    if head_specific:
        keep[:, 1, :, ::2] = False
    module._flex_sparse_attention(
        torch.zeros(1, length, 4, 64),
        torch.zeros(1, length, 2, 64),
        torch.zeros(1, length, 2, 64),
        block_sel=selected,
        q_positions=positions,
        keep_mask=keep,
        rescue_pad_queries=rescue,
    )
    assert module.indexer.block_size == 128
    assert len(captured) == 1
    block_mask = captured[0]
    assert block_mask.BLOCK_SIZE == (query_block, 128)
    grouped = not rescue and not head_specific
    assert block_mask.kv_num_blocks.shape[:2] == ((2, 1) if grouped else (1, 4))
    if backend != "fa4" or rescue:
        capability.assert_not_called()

    padded_q = ((length + query_block - 1) // query_block) * query_block
    padded_k = key_blocks * 128
    query = torch.arange(padded_q)[:, None]
    key = torch.arange(padded_k)[None, :]
    for head in range(4):
        batch_id, mask_head = (head // 2, 0) if grouped else (0, head)
        actual = block_mask.mask_mod(torch.tensor(batch_id), torch.tensor(mask_head), query, key)
        first_key = 0 if head < 2 else (key_blocks - 1) * 128
        expected = (query < length) & (key < length) & (key <= query)
        expected &= (key >= first_key) & (key < first_key + 128)
        expected[:length, :length] &= keep[0, head if head_specific else 0]
        if rescue:
            expected |= (query == key) & (query < length)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Verify the real BlockMask's union of full/partial tiles, not only
        # the callback. Padded rows and keys must never create live tiles.
        expected_tiles = expected.reshape(padded_q // query_block, query_block, key_blocks, 128).any(dim=(1, 3))
        torch.testing.assert_close(block_mask.to_dense()[batch_id, mask_head].bool(), expected_tiles)

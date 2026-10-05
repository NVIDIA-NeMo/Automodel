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

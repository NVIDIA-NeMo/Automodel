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

"""Test flash attention packing integration.

Verifies that:
1. Dataset helpers extract per-document cu_seqlens from indexed masks
2. neat_packed_vlm_collater follows the structural packing contract
3. configure_packing patches the right functions with an injected dataset helper
"""

import pytest
import torch

from nemo_automodel.components.datasets.packing import get_seqlens_in_batch, get_unpad_data
from nemo_automodel.components.models.common.packing import configure_packing, get_packing_capabilities


def test_get_seqlens_in_batch():
    mask = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )
    seqlens = get_seqlens_in_batch(mask)
    assert seqlens.tolist() == [2, 3, 1, 2, 3]


def test_get_unpad_data():
    mask = torch.tensor(
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
    )
    indices, cu_seqlens, max_seqlen = get_unpad_data(mask)

    assert indices.tolist() == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
    assert cu_seqlens.tolist() == [0, 2, 5, 6, 8, 11]
    assert max_seqlen == 3


def test_get_unpad_data_single_doc():
    """Single document per batch element (no packing)."""
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    indices, cu_seqlens, max_seqlen = get_unpad_data(mask)

    assert indices.tolist() == [0, 1, 2, 3]
    assert cu_seqlens.tolist() == [0, 4]
    assert max_seqlen == 4


def test_collater_flash_returns_2d_mask():
    """With flash_attention_2, collater should return 2D indexed mask."""
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    batch = [
        {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "labels": torch.tensor([-100, 20, 30, 40, 50]),
            "attention_mask": torch.tensor([1, 1, 2, 2, 2]),
            "position_ids": torch.tensor([0, 1, 0, 1, 2]),
            "n_images": 0,
            "n_videos": 0,
        },
    ]
    result = neat_packed_vlm_collater(
        batch,
        packing=get_packing_capabilities("flash_attention_2"),
    )
    assert result["attention_mask"].ndim == 2  # [B, S]
    assert result["attention_mask"].shape == (1, 5)
    # Values preserved
    assert result["attention_mask"][0].tolist() == [1, 1, 2, 2, 2]


@pytest.mark.parametrize("attn_implementation", ["flash_attention_2", "flash_attention_3", "flash_attention_4"])
def test_collater_all_flash_variants_return_2d_mask(attn_implementation):
    """Every flash variant keeps the indexed mask, not just FA2.

    Regression: the collater matched ``== "flash_attention_2"`` exactly, so FA3/FA4
    silently fell through to the dense 4D branch and lost the varlen fast path.
    """
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    batch = [
        {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "labels": torch.tensor([-100, 20, 30, 40, 50]),
            "attention_mask": torch.tensor([1, 1, 2, 2, 2]),
            "position_ids": torch.tensor([0, 1, 0, 1, 2]),
            "n_images": 0,
            "n_videos": 0,
        },
    ]
    result = neat_packed_vlm_collater(
        batch,
        packing=get_packing_capabilities(attn_implementation),
    )
    assert result["attention_mask"].ndim == 2, f"{attn_implementation} fell back to a dense mask"
    assert result["attention_mask"].shape == (1, 5)
    assert result["attention_mask"][0].tolist() == [1, 1, 2, 2, 2]


def test_collater_native_fa4_emits_varlen_metadata():
    """Native FA4 receives the unpadding metadata consumed by its BSHD adapter."""
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    batch = [
        {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "labels": torch.tensor([-100, 20, 30, 40, 50]),
            "attention_mask": torch.tensor([1, 1, 2, 2, 2]),
            "position_ids": torch.tensor([0, 1, 0, 1, 2]),
            "n_images": 0,
            "n_videos": 0,
        },
        {
            "input_ids": torch.tensor([60, 70, 80]),
            "labels": torch.tensor([-100, 70, 80]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "position_ids": torch.tensor([0, 1, 2]),
            "n_images": 0,
            "n_videos": 0,
        },
    ]

    result = neat_packed_vlm_collater(
        batch,
        packing=get_packing_capabilities("fa4"),
    )

    assert result["attention_mask"].tolist() == [[1, 1, 2, 2, 2], [1, 1, 1, 0, 0]]
    assert result["_packed_seq_ids"].tolist() == result["attention_mask"].tolist()
    assert result["packed_token_indices"].tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, -1, -1]]
    assert result["cu_seqlens"].tolist() == [[0, 2, 5], [0, 3, -1]]
    assert result["max_seqlen"] == 3


def test_collater_sdpa_returns_4d_mask():
    """With sdpa, collater should return 4D block-causal mask."""
    from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

    batch = [
        {
            "input_ids": torch.tensor([10, 20, 30, 40, 50]),
            "labels": torch.tensor([-100, 20, 30, 40, 50]),
            "attention_mask": torch.tensor([1, 1, 2, 2, 2]),
            "position_ids": torch.tensor([0, 1, 0, 1, 2]),
            "n_images": 0,
            "n_videos": 0,
        },
    ]
    result = neat_packed_vlm_collater(
        batch,
        packing=get_packing_capabilities("sdpa"),
    )
    assert result["attention_mask"].ndim == 4  # [B, 1, S, S]
    assert result["attention_mask"].shape == (1, 1, 5, 5)


def test_configure_packing_patches():
    """Verify monkey-patching works."""
    import transformers.modeling_flash_attention_utils as flash_utils

    original_fn = flash_utils._get_unpad_data

    # Should not patch for sdpa
    configure_packing(attn_implementation="sdpa")
    assert flash_utils._get_unpad_data is original_fn

    # Should patch for flash_attention_2
    configure_packing(attn_implementation="flash_attention_2", unpad_data=get_unpad_data)
    assert flash_utils._get_unpad_data is get_unpad_data

    # Restore
    flash_utils._get_unpad_data = original_fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_varlen_with_indexed_mask():
    """End-to-end: verify flash_attn_varlen_func works with our cu_seqlens."""
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        pytest.skip("flash_attn not installed")

    B, S, H, D = 1, 12, 4, 64
    num_docs = 3
    doc_len = S // num_docs

    # Build indexed mask
    mask = torch.zeros(B, S, dtype=torch.long, device="cuda")
    for i in range(num_docs):
        mask[0, i * doc_len : (i + 1) * doc_len] = i + 1

    indices, cu_seqlens, max_seqlen = get_unpad_data(mask)

    # Create Q, K, V
    q = torch.randn(B * S, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B * S, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B * S, H, D, dtype=torch.bfloat16, device="cuda")

    # Should not raise
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens.to("cuda"),
        cu_seqlens_k=cu_seqlens.to("cuda"),
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )
    assert out.shape == (B * S, H, D)

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

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.mimo_v2_flash.cp import (
    _MIMO_GLOBAL_IMAGE_MASK,
    _MIMO_GLOBAL_VIDEO_MASK,
    _MIMO_THD_LOCAL_INDICES,
    make_mimo_te_cp_sharder,
    shard_batch_for_mimo_te,
)


class _FakeCPMesh:
    def __init__(self, size: int = 2):
        self._size = size
        self._group = object()

    def size(self) -> int:
        return self._size

    def get_group(self):
        return self._group


def _thd_batch(*, batch_size: int = 1, sequence: int = 8) -> dict:
    input_ids = torch.arange(batch_size * sequence, dtype=torch.long).reshape(batch_size, sequence)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "position_ids": torch.arange(sequence).expand(batch_size, -1).clone(),
        "seq_lens": torch.full((batch_size, 1), sequence, dtype=torch.long),
        "seq_lens_padded": torch.full((batch_size, 1), sequence, dtype=torch.long),
        "qkv_format": "thd",
    }


def test_mimo_te_sharder_delegates_to_framework_and_preserves_vlm_metadata():
    """The adapter must retain PP media and global-to-local VLM token maps."""
    batch = _thd_batch(batch_size=2, sequence=4)
    batch["input_ids"][0, 1] = 91
    batch["input_ids"][1, 2] = 92
    media = {"pixel_values": [torch.ones(1)]}
    batch["_vlm_pp_media_chunks"] = media
    delegated = {
        "input_ids": batch["input_ids"].clone(),
        "labels": batch["labels"].clone(),
        "position_ids": batch["position_ids"].clone(),
        "cu_seqlens": torch.tensor([[0, 4], [0, 4]], dtype=torch.int32),
        "max_seqlen": torch.tensor([4, 4], dtype=torch.int32),
        "padding_mask": torch.zeros_like(batch["input_ids"], dtype=torch.bool),
        "qkv_format": "thd",
    }

    with patch(
        "nemo_automodel.components.models.mimo_v2_flash.cp.make_cp_batch_for_te",
        return_value=delegated,
    ) as make_te:
        _, result, layout = shard_batch_for_mimo_te(
            None,
            None,
            batch,
            num_chunks=2,
            image_token_id=91,
            video_token_id=92,
        )

    make_te.assert_called_once()
    assert layout is None
    torch.testing.assert_close(result[_MIMO_THD_LOCAL_INDICES], torch.arange(4).expand(2, -1))
    torch.testing.assert_close(
        result[_MIMO_GLOBAL_IMAGE_MASK],
        torch.tensor([[False, True, False, False], [False, False, False, False]]),
    )
    torch.testing.assert_close(
        result[_MIMO_GLOBAL_VIDEO_MASK],
        torch.tensor([[False, False, False, False], [False, False, True, False]]),
    )
    assert result["_vlm_pp_media_chunks"] is media


def test_mimo_te_sharder_uses_te_dual_chunk_indices_and_reports_layout():
    """CP token ownership must come from TE's THD partition primitive."""
    batch = _thd_batch(sequence=8)
    mesh = _FakeCPMesh()
    model = torch.nn.Module()
    expected = torch.tensor([0, 1, 6, 7], dtype=torch.long)
    fake_tex = SimpleNamespace(thd_get_partitioned_indices=lambda cu, total, size, rank: expected)
    delegated = {
        "input_ids": batch["input_ids"].reshape(-1).index_select(0, expected),
        "labels": batch["labels"].reshape(-1).index_select(0, expected),
        "position_ids": batch["position_ids"].reshape(-1).index_select(0, expected),
        "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
        "max_seqlen": torch.tensor(8, dtype=torch.int32),
        "padding_mask": torch.zeros(4, dtype=torch.bool),
        "qkv_format": "thd",
    }

    with (
        patch.dict(sys.modules, {"transformer_engine_torch": fake_tex}),
        patch("torch.distributed.get_rank", return_value=0),
        patch(
            "nemo_automodel.components.models.mimo_v2_flash.cp.make_cp_batch_for_te",
            return_value=delegated,
        ),
        patch("nemo_automodel.components.models.mimo_v2_flash.cp.ensure_mimo_te_context_parallel") as ensure_cp,
    ):
        _, result, layout = shard_batch_for_mimo_te(
            mesh,
            None,
            batch,
            model=model,
            image_token_id=None,
            video_token_id=None,
        )

    ensure_cp.assert_called_once_with(model, mesh)
    torch.testing.assert_close(result[_MIMO_THD_LOCAL_INDICES], expected)
    assert layout is not None
    torch.testing.assert_close(layout.local_token_global_indices, expected)
    assert layout.input_row_shape == (1, 8)
    assert layout.padded_seq_len == 8


def test_mimo_te_sharder_rejects_neat_packed_cp_without_fallback():
    """A block-diagonal NEAT mask must never silently become ordinary causal CP."""
    batch = _thd_batch()
    batch["qkv_format"] = "bshd"

    with (
        patch("nemo_automodel.components.models.mimo_v2_flash.cp.make_cp_batch_for_te") as make_te,
        pytest.raises(ValueError, match="requires qkv_format='thd'"),
    ):
        shard_batch_for_mimo_te(_FakeCPMesh(), None, batch)

    make_te.assert_not_called()


def test_mimo_te_sharder_rejects_documents_not_divisible_by_dual_chunks():
    """TE CP requires every physical document span divisible by 2 * CP."""
    batch = _thd_batch(sequence=8)
    batch["seq_lens"] = torch.tensor([[6, 2]])
    batch["seq_lens_padded"] = torch.tensor([[6, 2]])

    with pytest.raises(ValueError, match=r"divisible by 2 \* cp_size"):
        shard_batch_for_mimo_te(_FakeCPMesh(), None, batch)


def test_mimo_te_sharder_rejects_external_loss_mask():
    batch = _thd_batch()
    with pytest.raises(ValueError, match="external loss_mask"):
        shard_batch_for_mimo_te(None, None, batch, loss_mask=torch.ones(1, 8))


def test_make_mimo_te_cp_sharder_returns_framework_contract():
    model = torch.nn.Module()
    sharder = make_mimo_te_cp_sharder(model=model, num_chunks=1, image_token_id=91, video_token_id=92)
    assert isinstance(sharder, ContextParallelSharder)
    assert sharder.local_token_global_indices is None
    assert sharder.shard_batch.keywords["model"] is model

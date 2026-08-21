# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the order-independent PP media index protocol.

Every microbatch must select its own media chunk through ``pp_media_index``,
including a microbatch that carries no media at all, because the schedule may run
microbatches out of order and additionally probes stage 0 with one extra forward
for runtime shape inference.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from nemo_automodel.components.datasets.vlm.pp_media import (
    PP_MEDIA_CHUNKS_ATTR,
    VLM_PP_MEDIA_KEY,
    build_pp_media_index,
    prepare_vlm_media_for_pp,
    stage_vlm_media_for_pp,
)
from nemo_automodel.shared.pipeline import PP_MEDIA_INDEX_KEY, pp_media_chunk


def _fake_pp():
    return SimpleNamespace(info=SimpleNamespace(has_first_stage=True))


def _split_index_per_microbatch(batch: dict, n_microbatches: int) -> list[torch.Tensor]:
    """Split the batch the way ``torch.distributed.pipelining`` splits it."""
    _, kwargs_split = split_args_kwargs_into_chunks(
        (batch["input_ids"],),
        {PP_MEDIA_INDEX_KEY: batch[PP_MEDIA_INDEX_KEY]},
        n_microbatches,
    )
    return [kwargs[PP_MEDIA_INDEX_KEY] for kwargs in kwargs_split]


def test_build_pp_media_index_matches_tensor_split_microbatches():
    """Boundaries follow torch.tensor_split, not ceil-sized microbatches.

    torch gives the first ``batch % n_microbatches`` microbatches one extra
    sample. Ceil-sizing instead front-loads full microbatches and leaves a short
    tail, which for batch=7/n=3 would place samples 5 and 6 in different chunks
    than torch puts them in.
    """
    assert build_pp_media_index(4, 2).tolist() == [0, 0, 1, 1]
    assert build_pp_media_index(4, 1).tolist() == [0, 0, 0, 0]
    # tensor_split(7, 3) -> sizes 3, 2, 2 (ceil-sizing would give 3, 3, 1)
    assert build_pp_media_index(7, 3).tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert torch.tensor_split(torch.arange(7), 3)[0].numel() == 3
    assert build_pp_media_index(4, 2).dtype == torch.int64


def test_text_only_microbatch_selects_its_own_empty_chunk():
    """Microbatch 0 has no images, microbatch 1 does; neither may read the other's chunk."""
    batch_size, n_microbatches, patch_dim = 4, 2, 8
    # Samples 0 and 1 are text-only (image orphaned by truncation -> count 0).
    n_images_per_sample = torch.tensor([0, 0, 1, 1])
    image_grid_hws = torch.tensor([[2, 2], [2, 2]])  # 4 patches per image
    pixel_values = torch.arange(8 * patch_dim, dtype=torch.float32).reshape(8, patch_dim)

    batch = {
        "input_ids": torch.arange(batch_size * 6).reshape(batch_size, 6),
        "pixel_values": pixel_values,
        "image_grid_hws": image_grid_hws,
        "n_images_per_sample": n_images_per_sample,
    }
    prepare_vlm_media_for_pp(batch, batch_size=batch_size, n_microbatches=n_microbatches)

    assert "pixel_values" not in batch
    assert batch[PP_MEDIA_INDEX_KEY].tolist() == [0, 0, 1, 1]
    chunks = batch[VLM_PP_MEDIA_KEY]
    assert len(chunks["pixel_values"]) == n_microbatches
    assert chunks["pixel_values"][0].shape[0] == 0  # empty chunk kept in position
    assert torch.equal(chunks["pixel_values"][1], pixel_values)

    index_per_mb = _split_index_per_microbatch(batch, n_microbatches)
    module = torch.nn.Module()
    with stage_vlm_media_for_pp(_fake_pp(), [module], batch):
        # Query in reverse order: chunk selection must not depend on call order.
        assert torch.equal(pp_media_chunk(module, "pixel_values", index_per_mb[1]), pixel_values)
        # The empty chunk is staged in position but reported as None so the caller
        # skips its vision path instead of encoding zero inputs.
        assert pp_media_chunk(module, "pixel_values", index_per_mb[0]) is None
        assert pp_media_chunk(module, "image_grid_hws", index_per_mb[0]) is None
        assert torch.equal(pp_media_chunk(module, "image_grid_hws", index_per_mb[1]), image_grid_hws)

    assert getattr(module, PP_MEDIA_CHUNKS_ATTR) is None
    assert VLM_PP_MEDIA_KEY not in batch


def test_media_bearing_first_microbatch_is_not_shifted():
    """The mirrored case: microbatch 0 has the image, microbatch 1 is text-only."""
    batch_size, n_microbatches, patch_dim = 4, 2, 8
    image_grid_hws = torch.tensor([[2, 2]])
    pixel_values = torch.arange(4 * patch_dim, dtype=torch.float32).reshape(4, patch_dim)

    batch = {
        "input_ids": torch.arange(batch_size * 6).reshape(batch_size, 6),
        "pixel_values": pixel_values,
        "image_grid_hws": image_grid_hws,
        "n_images_per_sample": torch.tensor([1, 0, 0, 0]),
    }
    prepare_vlm_media_for_pp(batch, batch_size=batch_size, n_microbatches=n_microbatches)

    index_per_mb = _split_index_per_microbatch(batch, n_microbatches)
    module = torch.nn.Module()
    with stage_vlm_media_for_pp(_fake_pp(), [module], batch):
        assert torch.equal(pp_media_chunk(module, "pixel_values", index_per_mb[0]), pixel_values)
        assert pp_media_chunk(module, "pixel_values", index_per_mb[1]) is None


def test_grid_thws_is_staged_instead_of_forwarded_raw():
    """kimi_k25_vl emits grid_thws next to image_grid_hws; it must not reach schedule.step."""
    batch_size, n_microbatches, patch_dim = 2, 2, 8
    grid_thws = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixel_values = torch.arange(8 * patch_dim, dtype=torch.float32).reshape(8, patch_dim)

    batch = {
        "input_ids": torch.arange(batch_size * 6).reshape(batch_size, 6),
        "pixel_values": pixel_values,
        "grid_thws": grid_thws,
        "image_grid_hws": grid_thws[:, 1:],
        "n_images_per_sample": torch.tensor([1, 1]),
    }
    prepare_vlm_media_for_pp(batch, batch_size=batch_size, n_microbatches=n_microbatches)

    assert "grid_thws" not in batch
    index_per_mb = _split_index_per_microbatch(batch, n_microbatches)
    module = torch.nn.Module()
    with stage_vlm_media_for_pp(_fake_pp(), [module], batch):
        assert torch.equal(pp_media_chunk(module, "grid_thws", index_per_mb[0]), grid_thws[:1])
        assert torch.equal(pp_media_chunk(module, "grid_thws", index_per_mb[1]), grid_thws[1:])


def test_misaligned_per_image_metadata_is_rejected():
    batch = {
        "input_ids": torch.arange(2 * 6).reshape(2, 6),
        "pixel_values": torch.zeros(8, 8),
        "image_grid_hws": torch.tensor([[2, 2], [2, 2]]),
        "grid_thws": torch.tensor([[1, 2, 2]]),  # one row short
        "n_images_per_sample": torch.tensor([1, 1]),
    }
    with pytest.raises(ValueError, match="cannot align 'grid_thws'"):
        prepare_vlm_media_for_pp(batch, batch_size=2, n_microbatches=2)


@pytest.mark.parametrize("batch_size", list(range(1, 17)))
@pytest.mark.parametrize("n_microbatches", [1, 2, 3, 4, 5])
def test_media_index_agrees_with_torch_microbatch_split(batch_size, n_microbatches):
    """Every sample in a torch microbatch must claim that microbatch's chunk.

    The pipeline schedule splits ``pp_media_index`` itself, with
    ``torch.tensor_split``. If our chunk boundaries used a different rule, one
    torch microbatch could hold samples belonging to two different media chunks,
    and the lookup -- which reads the first sample's index -- would silently pair
    those samples with another microbatch's images. This pins the two together so
    a change on either side fails here rather than as degraded training quality.
    """
    if n_microbatches > batch_size:
        pytest.skip("torch requires at least one sample per microbatch")

    split_args_kwargs_into_chunks = pytest.importorskip(
        "torch.distributed.pipelining.microbatch"
    ).split_args_kwargs_into_chunks

    media_index = build_pp_media_index(batch_size, n_microbatches)
    input_ids = torch.arange(batch_size).unsqueeze(1)

    _, kwargs_per_microbatch = split_args_kwargs_into_chunks(
        (input_ids,), {"pp_media_index": media_index}, n_microbatches
    )

    for microbatch_index, kwargs in enumerate(kwargs_per_microbatch):
        claimed = set(kwargs["pp_media_index"].tolist())
        assert claimed == {microbatch_index}, (
            f"torch microbatch {microbatch_index} holds samples claiming chunks {sorted(claimed)}"
        )

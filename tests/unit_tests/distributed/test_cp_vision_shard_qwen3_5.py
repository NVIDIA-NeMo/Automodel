# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Composition tests: the REAL (tiny) Qwen3-VL vision tower + the CP vision-shard partitioner.

The dense Qwen3.5 VLM (``Qwen3_5ForConditionalGeneration``) runs the transformers
``Qwen3VLVisionModel`` in its CP pre-embed and, when the CP vision group is published, routes
it through ``cp_vision_shard.maybe_distribute_visual``.  That helper's correctness rests on one
property of this exact tower: the per-frame attention (``cu_seqlens`` from ``grid_thw``) and
spatial-only position embeddings make frames INDEPENDENT, so

    visual(all_frames).pooler_output == concat_r visual(frames_of_rank_r).pooler_output

up to numerical precision, for BOTH ``pooler_output`` and every ``deepstack_features`` entry.

These CPU tests build a small real ``Qwen3VLVisionModel`` and assert that property using the
module's OWN contiguous partitioner (``_contiguous_balanced_bounds``) -- i.e. the tower plus the
real partition/gather-order logic together reproduce the replicated full forward.  The
collective/backward path of ``maybe_distribute_visual`` itself is covered (with real gloo
collectives) in ``test_cp_vision_shard_gloo.py``.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.distributed import cp_vision_shard as vs

# The real tower class only exists in recent transformers; skip cleanly otherwise.
_qwen3vl = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
Qwen3VLVisionConfig = pytest.importorskip("transformers.models.qwen3_vl.configuration_qwen3_vl").Qwen3VLVisionConfig
Qwen3VLVisionModel = _qwen3vl.Qwen3VLVisionModel


def _tiny_tower(deepstack: bool = True) -> Qwen3VLVisionModel:
    """Build a small real Qwen3-VL vision tower on CPU (fp32, sdpa attention)."""
    cfg = Qwen3VLVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=32,
        depth=2,
        patch_size=4,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0, 1] if deepstack else [],
    )
    torch.manual_seed(0)
    return Qwen3VLVisionModel(cfg).eval().to(torch.float32)


def _patch_feature_dim(tower: Qwen3VLVisionModel) -> int:
    c = tower.config
    return c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size


def _pixels(tower: Qwen3VLVisionModel, grid: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Random ``[total_patch_rows, patch_dim]`` pixel rows for ``grid`` ([N, 3] (t, h, w))."""
    total = int(grid.prod(dim=-1).sum())
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(total, _patch_feature_dim(tower), generator=gen)


def _assert_gathered_matches_full(tower, pixel, grid, cuts):
    """Forward each contiguous entry-slice ``[cuts[r], cuts[r+1])`` through the real tower and
    assert the concatenation equals the full forward, for pooler_output and every deepstack
    feature -- i.e. frame-independence holds and the gather preserves original entry order.

    ``pixel``/``grid`` are the module contract ([total_patch_rows, patch_dim] frame-contiguous
    rows / [N, 3] (t, h, w) rows); ``cuts`` are entry boundaries of length ``world + 1``.
    """
    full = tower(pixel, grid_thw=grid, return_dict=True)
    pix_bounds = [0] + grid.prod(dim=-1).cumsum(0).tolist()

    pooler_blocks = []
    deepstack_blocks: list[list[torch.Tensor]] = []
    for r in range(len(cuts) - 1):
        lo, hi = cuts[r], cuts[r + 1]
        out = tower(pixel[pix_bounds[lo] : pix_bounds[hi]], grid_thw=grid[lo:hi], return_dict=True)
        pooler_blocks.append(out.pooler_output)
        if out.deepstack_features is not None:
            deepstack_blocks.append(list(out.deepstack_features))

    gathered_pooler = torch.cat(pooler_blocks, dim=0)
    assert gathered_pooler.shape == full.pooler_output.shape
    torch.testing.assert_close(gathered_pooler, full.pooler_output, atol=1e-5, rtol=1e-5)

    if full.deepstack_features is not None:
        for k in range(len(full.deepstack_features)):
            gathered_k = torch.cat([blocks[k] for blocks in deepstack_blocks], dim=0)
            torch.testing.assert_close(gathered_k, full.deepstack_features[k], atol=1e-5, rtol=1e-5)


# entries of varied size; every (t*h*w) is a multiple of spatial_merge_size**2 = 4.
_IMAGE_ENTRIES = [(1, 4, 4), (1, 2, 2), (1, 4, 2), (1, 2, 4), (1, 6, 2)]


@pytest.mark.parametrize("world", [2, 3, 4])
def test_real_tower_image_entries_frame_independent(world):
    """Real tower + module partitioner: contiguous entry partition reassembles to the full
    forward (pooler + deepstack) at world 2-4."""
    tower = _tiny_tower(deepstack=True)
    grid = torch.tensor(_IMAGE_ENTRIES, dtype=torch.long)
    pixel = _pixels(tower, grid, seed=1)

    cuts = vs._contiguous_balanced_bounds(grid.prod(dim=-1), world, cost_alpha_source=tower)
    assert cuts is not None and len(cuts) == world + 1
    _assert_gathered_matches_full(tower, pixel, grid, cuts)


@pytest.mark.parametrize("world", [2, 4])
def test_real_tower_single_video_frame_split(world):
    """A single video (t>1) split across ranks at FRAME granularity still reassembles to the
    full forward -- the entry-level partitioner falls back (1 entry < world), so build per-rank
    ``(count, h, w)`` grids exactly as ``maybe_distribute_visual`` coalesces same-entry frames."""
    tower = _tiny_tower(deepstack=True)
    t, h, w = 8, 4, 4
    grid = torch.tensor([[t, h, w]], dtype=torch.long)
    pixel = _pixels(tower, grid, seed=2)
    rows_per_frame = h * w

    # frame-level partition (module falls back at entry level for 1 entry < world).
    frame_patches = torch.full((t,), rows_per_frame, dtype=torch.long)
    frame_cuts = vs._contiguous_balanced_bounds(frame_patches, world, cost_alpha_source=tower)
    assert frame_cuts is not None

    full = tower(pixel, grid_thw=grid, return_dict=True)
    pooler_blocks = []
    deepstack_blocks: list[list[torch.Tensor]] = []
    for r in range(world):
        count = frame_cuts[r + 1] - frame_cuts[r]
        lo_row = frame_cuts[r] * rows_per_frame
        hi_row = frame_cuts[r + 1] * rows_per_frame
        out = tower(
            pixel[lo_row:hi_row],
            grid_thw=torch.tensor([[count, h, w]], dtype=torch.long),
            return_dict=True,
        )
        pooler_blocks.append(out.pooler_output)
        deepstack_blocks.append(list(out.deepstack_features))

    torch.testing.assert_close(torch.cat(pooler_blocks, dim=0), full.pooler_output, atol=1e-5, rtol=1e-5)
    for k in range(len(full.deepstack_features)):
        gathered_k = torch.cat([blocks[k] for blocks in deepstack_blocks], dim=0)
        torch.testing.assert_close(gathered_k, full.deepstack_features[k], atol=1e-5, rtol=1e-5)


def test_real_tower_pooler_matches_get_image_features_contract():
    """The flat ``visual(...).pooler_output`` equals ``torch.cat(split-per-entry)`` -- the exact
    equivalence the Qwen3.5 CP wiring relies on when it swaps ``get_image_features`` for
    ``maybe_distribute_visual`` (whose pooler_output is the flat tensor)."""
    tower = _tiny_tower(deepstack=False)
    grid = torch.tensor(_IMAGE_ENTRIES, dtype=torch.long)
    pixel = _pixels(tower, grid, seed=3)

    flat = tower(pixel, grid_thw=grid, return_dict=True).pooler_output
    split_sizes = (grid.prod(dim=-1) // tower.spatial_merge_size**2).tolist()
    recombined = torch.cat(torch.split(flat, split_sizes), dim=0)
    torch.testing.assert_close(recombined, flat, atol=0, rtol=0)

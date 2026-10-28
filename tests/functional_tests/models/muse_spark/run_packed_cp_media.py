#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check packed Muse media placement and gradients with real TE CP partitions.

Run with torchrun --standalone --nproc-per-node=2. Covers CP1/CP2 and one,
two, or four PP input chunks, unequal document counts, inter-document padding,
and a media-free CP shard. The full-sequence reference checks placement,
summed feature gradients, and one SGD update without loading model weights.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.context_parallel.utils import make_cp_batch_for_te
from tests.unit_tests.models.muse_spark.conftest import tiny_muse_spark_model


def main() -> None:
    """Validate model-owned indices on two GPUs with actual TE kernels."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This test requires exactly two ranks")
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
        model = tiny_muse_spark_model()
        model.lm_head = None
        patch = model.config.patch_token_id
        # Row 0 puts its sole image on CP rank 1, leaving rank 0 media-free.
        ids = (torch.arange(64, device=device).reshape(4, 16) % 32) + 1
        for row, columns in enumerate(([1], [0, 3, 8, 9, 12, 14], [1, 2, 9, 14], [0, 4, 6, 12])):
            ids[row, columns] = patch
        real = torch.tensor([[3, 7, -1000], [7, 3, 3], [7, 7, -1000], [3, 7, 3]], device=device)
        padded = torch.tensor([[4, 12, -1000], [8, 4, 4], [8, 8, -1000], [4, 8, 4]], device=device)
        positions = torch.arange(16, device=device).expand(4, -1)
        cases = 0
        for cp_size in (1, 2):
            cp_mesh = mesh if cp_size == 2 else None
            model.cp_mesh = cp_mesh
            for num_chunks in (1, 2, 4):
                batch = {
                    "input_ids": ids.clone(),
                    "labels": ids.clone(),
                    "position_ids": positions.clone(),
                    "seq_lens": real.clone(),
                    "seq_lens_padded": padded.clone(),
                    "qkv_format": "thd",
                    "_vlm_pp_media_chunks": {"pixel_values": [torch.ones(1, device=device)] * num_chunks},
                }
                prepared = model.prepare_model_inputs_for_cp(batch, num_chunks=num_chunks)
                sharder = prepared.pop("cp_sharder")
                batch.update(prepared)
                _, sharded, _ = sharder.shard_batch(cp_mesh, None, batch)
                chunk_indices = sharded["_thd_local_indices"].reshape(num_chunks, -1)
                global_ids = sharded["input_ids"].reshape(num_chunks, -1)
                torch.testing.assert_close(global_ids, ids.reshape(num_chunks, -1))
                chunk_ids = global_ids.gather(1, chunk_indices.to(torch.long))
                global_masks = sharded["_muse_spark_global_image_mask"].reshape(num_chunks, -1)
                rows_per_chunk = 4 // num_chunks
                empty_local = False
                for chunk in range(num_chunks):
                    start, end = chunk * rows_per_chunk, (chunk + 1) * rows_per_chunk
                    # The unchanged public API still returns indices for one stream.
                    reference_batch = {
                        key: value[start:end].clone()
                        for key, value in {
                            "input_ids": ids,
                            "labels": ids,
                            "position_ids": positions,
                            "seq_lens": real,
                            "seq_lens_padded": padded,
                        }.items()
                    }
                    reference, expected_indices = make_cp_batch_for_te(
                        cp_mesh, reference_batch, return_local_indices=True
                    )
                    torch.testing.assert_close(chunk_indices[chunk], expected_indices)
                    torch.testing.assert_close(chunk_ids[chunk], reference["input_ids"].reshape(-1))
                    full_mask = (ids[start:end] == patch).reshape(-1)
                    torch.testing.assert_close(global_masks[chunk], full_mask)
                    local_mask = chunk_ids[chunk] == patch
                    empty_local |= not bool(local_mask.any())
                    n_features = int(full_mask.sum())
                    initial = torch.arange(n_features * 3, device=device, dtype=torch.float32).reshape(-1, 3)
                    initial = initial / 7 + 0.125
                    features = torch.nn.Parameter(initial.clone())
                    optimizer = torch.optim.SGD([features], lr=0.01)
                    full_stream = torch.zeros(full_mask.numel(), 3, device=device)
                    full_stream[full_mask] = initial
                    local_stream = model._splice_features(
                        torch.zeros(chunk_ids.shape[1], 3, device=device),
                        chunk_ids[chunk],
                        patch,
                        features,
                        global_mask=global_masks[chunk],
                        local_indices=chunk_indices[chunk],
                    )
                    torch.testing.assert_close(local_stream, full_stream.index_select(0, expected_indices))
                    coefficients = torch.arange(full_stream.numel(), device=device).reshape_as(full_stream).float()
                    coefficients = coefficients / 13 - 0.5
                    loss = (local_stream * coefficients.index_select(0, expected_indices)).sum()
                    loss.backward()
                    if cp_size == 2:
                        dist.all_reduce(features.grad, group=cp_mesh.get_group())
                    expected_gradient = coefficients[full_mask]
                    torch.testing.assert_close(features.grad, expected_gradient)
                    optimizer.step()
                    torch.testing.assert_close(features, initial - 0.01 * expected_gradient)
                if cp_size == 2 and num_chunks == 4 and dist.get_rank() == 0:
                    assert empty_local, "Expected a media-free shard in the multi-microbatch case"
                if dist.get_rank() == 0:
                    print(f"PASS cp={cp_size} chunks={num_chunks}: placement, gradients, SGD update", flush=True)
                cases += 1
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"MUSE PACKED CP MEDIA: PASS ({cases} configurations)", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

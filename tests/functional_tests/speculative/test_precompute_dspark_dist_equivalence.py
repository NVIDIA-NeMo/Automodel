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

"""Functional test: distributed DSpark precompute equals the single-process cache.

The single-process ``write_cache_shards`` path shipped with the offline-cache PR and
is the reference. This test runs a REAL (tiny, random) Qwen3 target through
``HFDSparkTargetModel`` twice -- once with the single-process writer, once with the
distributed writer across two genuine CPU/gloo ranks (real ``all_reduce`` step sync,
real ``barrier``) -- and asserts the two caches are numerically identical, sample by
sample. Sample counts are chosen so the ranks get UNEVEN step counts, forcing the
second rank through the dummy-batch lockstep padding path.

CPU-only and tiny (hidden=64, 10 samples): runs in seconds, writes a few hundred KB.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, Qwen3Config

from nemo_automodel.components.datasets.llm.dspark_cache import CachedDSparkDataset, write_manifest, write_shard
from nemo_automodel.components.datasets.llm.offline_cache import write_cache_shards, write_cache_shards_distributed
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel

logger = logging.getLogger(__name__)

_VOCAB = 128
_HIDDEN = 64
_SEQ = 16
_LAYERS = [1, 3, 5]
# 10 samples / shard_size 2 -> 5 shards over 2 ranks: rank0 takes 3 shards (3 steps at
# batch 2), rank1 takes 2 (2 steps) and must pad one dummy lockstep forward.
_NUM_SAMPLES = 10
_SHARD_SIZE = 2
_BATCH_SIZE = 2
_WORLD_SIZE = 2


def _tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=2 * _HIDDEN,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


class _TokenDataset(Dataset):
    """Deterministic tokenized samples, identical on every rank."""

    def __len__(self):
        return _NUM_SAMPLES

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(1000 + index)
        return {
            "input_ids": torch.randint(0, _VOCAB, (_SEQ,), generator=generator, dtype=torch.long),
            "attention_mask": torch.ones(_SEQ, dtype=torch.long),
            "loss_mask": torch.ones(_SEQ, dtype=torch.long),
        }


def _collate(features):
    return {k: torch.stack([f[k] for f in features], dim=0) for k in features[0]}


def _loader() -> DataLoader:
    return DataLoader(_TokenDataset(), batch_size=_BATCH_SIZE, shuffle=False, collate_fn=_collate, drop_last=False)


def _load_target(model_dir: str):
    target = AutoModelForCausalLM.from_pretrained(model_dir).to(dtype=torch.float32).eval()
    target.requires_grad_(False)
    return HFDSparkTargetModel(target, target_layer_ids=list(_LAYERS))


def _make_compute_batch(wrapper):
    def _compute(batch):
        with torch.no_grad():
            target_batch = wrapper.generate_batch(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
            )
        return {
            "input_ids": target_batch.input_ids.to(torch.long).cpu(),
            "loss_mask": target_batch.loss_mask.to(torch.long).cpu(),
            "target_hidden_states": target_batch.target_hidden_states.to(torch.float32).cpu(),
            "target_last_hidden_states": target_batch.target_last_hidden_states.to(torch.float32).cpu(),
        }

    return _compute


def _write_test_manifest(cache_dir: str) -> None:
    write_manifest(
        cache_dir,
        {
            "target_model": "tiny-qwen3",
            "target_model_type": "qwen3",
            "target_vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "num_hidden_layers": 6,
            "seq_length": _SEQ,
            "dtype": "fp32",
            "num_samples": _NUM_SAMPLES,
            "shard_size": _SHARD_SIZE,
            "target_hidden_dim": _HIDDEN * len(_LAYERS),
            "target_last_hidden_dim": _HIDDEN,
            "target_layer_ids": list(_LAYERS),
        },
    )


def _rank_worker(rank: int, world_size: int, init_file: str, model_dir: str, cache_dir: str) -> None:
    """One gloo rank of the distributed precompute (spawned; must be top-level)."""
    torch.set_num_threads(1)
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        wrapper = _load_target(model_dir)

        def _sync_max_steps(local_steps: int) -> int:
            tensor = torch.tensor([int(local_steps)])
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            return int(tensor.item())

        if rank == 0:
            _write_test_manifest(cache_dir)
        dist.barrier()
        write_cache_shards_distributed(
            dataloader=_loader(),
            output_dir=cache_dir,
            shard_size=_SHARD_SIZE,
            world_size=world_size,
            rank=rank,
            compute_batch=_make_compute_batch(wrapper),
            write_shard_fn=write_shard,
            logger=logger,
            sync_max_steps=_sync_max_steps,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_distributed_precompute_matches_single_process(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(0)

    # One fixed random tiny target on disk so every process loads identical weights.
    model_dir = str(tmp_path / "tiny_target")
    AutoModelForCausalLM.from_config(_tiny_config()).save_pretrained(model_dir)

    # Reference: the (already shipped and validated) single-process writer.
    ref_dir = str(tmp_path / "cache_ref")
    _write_test_manifest(ref_dir)
    write_cache_shards(
        dataloader=_loader(),
        output_dir=ref_dir,
        shard_size=_SHARD_SIZE,
        start_shard_index=0,
        compute_batch=_make_compute_batch(_load_target(model_dir)),
        write_shard_fn=write_shard,
        logger=logger,
    )

    # Distributed: two real gloo ranks with real all-reduce step sync + barriers.
    dist_dir = str(tmp_path / "cache_dist")
    os.makedirs(dist_dir, exist_ok=True)
    init_file = str(tmp_path / "dist_init")
    mp.spawn(
        _rank_worker,
        args=(_WORLD_SIZE, init_file, model_dir, dist_dir),
        nprocs=_WORLD_SIZE,
        join=True,
    )

    ref_ds, dist_ds = CachedDSparkDataset(ref_dir), CachedDSparkDataset(dist_dir)
    assert len(ref_ds) == len(dist_ds) == _NUM_SAMPLES
    for i in range(_NUM_SAMPLES):
        ref_sample, dist_sample = ref_ds[i], dist_ds[i]
        assert ref_sample.keys() == dist_sample.keys()
        for key in ref_sample:
            torch.testing.assert_close(dist_sample[key], ref_sample[key], rtol=0.0, atol=0.0)

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

"""Tiny real shard reads, offline export, process encoding and cache replay."""

import json
import shutil
from contextlib import closing
from operator import itemgetter

import pytest
import torch
from zephon import Dataset, Pipeline, StaticMixtureWorkSource

from examples.vlm_finetune.zephon.prepare_data import export_jsonl
from nemo_automodel.components.datasets.zephon.config import ZephonCacheOptions, ZephonParquetRGCacheOptions
from nemo_automodel.components.datasets.zephon.vlm import VlmEncoderConfig, merge_image_documents
from nemo_automodel.components.models.qwen2_5_vl.preprocessing import Qwen2_5_VLPositionState
from tests.integration.zephon.test_zephon_vlm_packing import _convert
from tests.utils.vlm_fixtures import embedded_inputs as embedded_inputs
from tests.utils.vlm_fixtures import inputs as inputs
from tests.utils.vlm_fixtures import processor as processor
from tests.utils.vlm_fixtures import tiny_vlm_config


def _pipeline(root, processor_path, *, cache=None, runner="inline"):
    dataset = Dataset.from_path("images", str(root))
    source = StaticMixtureWorkSource(
        [dataset], {"images": 1.0}, chunk_size=3, shuffle_shards=False, shuffle_within_shard=False
    )
    operator = VlmEncoderConfig(inject_fake_images=False, parallelism=1).build(
        pretrained_model_name_or_path=processor_path, positions=Qwen2_5_VLPositionState(tiny_vlm_config())
    )
    pipeline = Pipeline(source).fetch(parallelism=1).add_op(operator)
    pipeline.pack_sequences(max_length=41, algorithm="first_fit", num_bins=2, pack_payloads=merge_image_documents)
    pipeline.map_transform(itemgetter("packed_samples"), parallelism=1).batch(2, drop_last=False)
    pipeline.options(deterministic=True, max_workers=2, runner=runner, mtp_mode=False, shm_enabled=False)
    if cache is not None:
        pipeline.options(io_options=cache.build())
    return pipeline


def test_offline_export_cache_eviction_and_spawn_checkpoint(inputs, tmp_path):
    processor_path, documents = inputs
    source = tmp_path / "pointers.jsonl"
    source.write_text("".join(json.dumps(document) + "\n" for document in documents * 2))
    root = tmp_path / "embedded"
    assert export_jsonl(source, root, shard_bytes=1024) == 12
    # All training media must now come from fetched shards, never these files.
    for image in tmp_path.glob("image-*.png"):
        image.unlink()
    cache_root = tmp_path / "decoded-cache"
    cache = ZephonCacheOptions(
        parquet_rg_cache=ZephonParquetRGCacheOptions(root=str(cache_root), limit_bytes="1mb", min_free_bytes=0)
    )
    baseline = [_convert(batch) for batch in _pipeline(root, processor_path, cache=cache)]
    assert len(baseline) >= 2 and cache_root.exists()
    # A cold decoded cache must reconstruct the same documents from source shards.
    shutil.rmtree(cache_root)
    pipeline = _pipeline(root, processor_path, cache=cache, runner="process")
    try:
        with closing(iter(pipeline)) as iterator:
            first = _convert(next(iterator))
            state = pipeline.checkpoint()
    finally:
        pipeline.close()
    restored = _pipeline(root, processor_path, cache=cache, runner="process")
    try:
        restored.restore(state)
        actual = [first, *(_convert(batch) for batch in restored)]
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        assert sum(batch["labels"].ne(-100).sum().item() for batch in actual) == 36
    finally:
        restored.close()


def test_litdata_binary_shards_decode_embedded_images(embedded_inputs, tmp_path):
    pytest.importorskip("litdata")
    from litdata.streaming.writer import BinaryWriter

    processor_path, documents = embedded_inputs
    root = tmp_path / "litdata"
    writer = BinaryWriter(cache_dir=str(root), chunk_size=3)
    for index in range(6):
        image = documents[0]["conversation"][0]["content"][0]["image"]
        writer.add_item(index, {"image": image, "question": "Describe image", "answer": "the answer"})
    writer.done()
    writer.merge()
    batches = [_convert(batch) for batch in _pipeline(root, processor_path)]
    assert batches and sum(batch["labels"].ne(-100).sum().item() for batch in batches) == 18
    assert sum(batch["image_grid_thw"].shape[0] for batch in batches) == 6

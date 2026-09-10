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

import json

from tools.diffusion.preprocessing_multiprocess import _save_metadata_shards


def test_unsharded_preserves_original_filenames(tmp_path):
    """Default single-job usage (shard_world=1) must keep the original filenames, so
    existing caches and any external tooling reading them are unaffected."""
    _save_metadata_shards(
        all_metadata=[{"prompt": f"p{i}"} for i in range(3)],
        output_dir=tmp_path,
        processor_name="qwen_image",
        model_name="Qwen/Qwen-Image",
        model_type="qwen_image",
        shard_size=10000,
        extra_fields={"max_pixels": 65536},
    )

    index = json.loads((tmp_path / "metadata.json").read_text())
    assert index["shards"] == ["metadata_shard_0000.json"]
    assert "shard_rank" not in index
    assert "shard_world" not in index
    assert (tmp_path / "metadata_shard_0000.json").exists()


def test_sharded_namespaces_filenames_by_rank(tmp_path):
    """When shard_world > 1, filenames must be rank-namespaced so multiple jobs can share
    one output_dir without overwriting each other's shards or index files."""
    _save_metadata_shards(
        all_metadata=[{"prompt": f"p{i}"} for i in range(3)],
        output_dir=tmp_path,
        processor_name="qwen_image",
        model_name="Qwen/Qwen-Image",
        model_type="qwen_image",
        shard_size=10000,
        extra_fields={"max_pixels": 65536},
        shard_rank=2,
        shard_world=4,
    )

    index = json.loads((tmp_path / "metadata_r02.json").read_text())
    assert index["shards"] == ["metadata_shard_r02_s0000.json"]
    assert index["shard_rank"] == 2
    assert index["shard_world"] == 4
    assert (tmp_path / "metadata_shard_r02_s0000.json").exists()
    assert not (tmp_path / "metadata.json").exists()


def test_sharded_different_ranks_do_not_collide(tmp_path):
    """Two ranks writing to the same output_dir must not clobber each other's files."""
    for rank in (0, 1):
        _save_metadata_shards(
            all_metadata=[{"prompt": f"rank{rank}-p{i}"} for i in range(2)],
            output_dir=tmp_path,
            processor_name="qwen_image",
            model_name="Qwen/Qwen-Image",
            model_type="qwen_image",
            shard_size=10000,
            extra_fields={},
            shard_rank=rank,
            shard_world=2,
        )

    index0 = json.loads((tmp_path / "metadata_r00.json").read_text())
    index1 = json.loads((tmp_path / "metadata_r01.json").read_text())
    assert index0["shards"] == ["metadata_shard_r00_s0000.json"]
    assert index1["shards"] == ["metadata_shard_r01_s0000.json"]

    shard0 = json.loads((tmp_path / "metadata_shard_r00_s0000.json").read_text())
    shard1 = json.loads((tmp_path / "metadata_shard_r01_s0000.json").read_text())
    assert [item["prompt"] for item in shard0] == ["rank0-p0", "rank0-p1"]
    assert [item["prompt"] for item in shard1] == ["rank1-p0", "rank1-p1"]


def test_sharded_multiple_chunks_numbered_within_rank(tmp_path):
    """A rank whose metadata exceeds shard_size must still split into multiple chunk files,
    numbered independently within that rank's namespace."""
    _save_metadata_shards(
        all_metadata=[{"prompt": f"p{i}"} for i in range(5)],
        output_dir=tmp_path,
        processor_name="qwen_image",
        model_name="Qwen/Qwen-Image",
        model_type="qwen_image",
        shard_size=2,
        extra_fields={},
        shard_rank=1,
        shard_world=3,
    )

    index = json.loads((tmp_path / "metadata_r01.json").read_text())
    assert index["shards"] == [
        "metadata_shard_r01_s0000.json",
        "metadata_shard_r01_s0001.json",
        "metadata_shard_r01_s0002.json",
    ]
    assert index["num_shards"] == 3
    assert index["total_items"] == 5

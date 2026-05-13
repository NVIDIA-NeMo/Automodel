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

"""Unit tests for S3 / MSC object-storage support in MegatronPretraining.

All S3 interactions are mocked via ``unittest.mock``; no real network calls.
"""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_automodel.components.datasets.llm.megatron import indexed_dataset as ids
from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import (
    ObjectStorageConfig,
    _cache_index_file,
    _get_index_cache_path,
    _is_object_storage_path,
    _parse_s3_path,
    _S3BinReader,
)
from nemo_automodel.components.datasets.llm.megatron_dataset import (
    validate_dataset_asset_accessibility,
)

# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------


class TestObjectStorageConfig:
    def test_defaults(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/tmp/idx_cache")
        assert cfg.path_to_idx_cache == "/tmp/idx_cache"
        assert cfg.bin_chunk_nbytes == 256 * 1024 * 1024

    def test_custom_chunk_size(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/x", bin_chunk_nbytes=16 * 1024 * 1024)
        assert cfg.bin_chunk_nbytes == 16 * 1024 * 1024


class TestIsObjectStoragePath:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("s3://bucket/key", True),
            ("msc://profile/path", True),
            ("/local/path", False),
            ("relative/path", False),
            ("", False),
            ("file:///tmp/x", False),
        ],
    )
    def test_detection(self, path, expected):
        assert _is_object_storage_path(path) is expected


class TestParseS3Path:
    def test_bucket_and_key(self):
        assert _parse_s3_path("s3://my-bucket/path/to/object") == ("my-bucket", "path/to/object")

    def test_bucket_only(self):
        assert _parse_s3_path("s3://my-bucket") == ("my-bucket", "")

    def test_rejects_non_s3(self):
        with pytest.raises(ValueError, match="Not an S3 path"):
            _parse_s3_path("/local/path")


class TestGetIndexCachePath:
    def test_s3_path(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/cache")
        out = _get_index_cache_path("s3://bucket/dir/shard.idx", cfg)
        assert out == "/cache/bucket/dir/shard.idx"

    def test_msc_path(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/cache")
        out = _get_index_cache_path("msc://profile/dir/shard.idx", cfg)
        assert out == "/cache/profile/dir/shard.idx"

    def test_rejects_local(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/cache")
        with pytest.raises(ValueError, match="Not an object storage path"):
            _get_index_cache_path("/local/shard.idx", cfg)


# ---------------------------------------------------------------------------
# _cache_index_file: rank-0 download, others wait
# ---------------------------------------------------------------------------


class TestCacheIndexFile:
    def test_downloads_when_missing(self, tmp_path):
        local = tmp_path / "sub" / "shard.idx"
        # Stand in for boto3.client("s3") — record download_file invocation.
        fake_client = MagicMock()

        def fake_download(bucket, key, dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as f:
                f.write(b"FAKE_IDX")

        fake_client.download_file.side_effect = fake_download

        with patch.object(ids, "HAS_BOTO3", True), patch.object(ids, "boto3") as boto3_mod:
            boto3_mod.client.return_value = fake_client
            _cache_index_file("s3://bucket/dir/shard.idx", str(local))

        assert local.exists()
        fake_client.download_file.assert_called_once_with("bucket", "dir/shard.idx", str(local))

    def test_skips_when_cached(self, tmp_path):
        local = tmp_path / "shard.idx"
        local.write_bytes(b"EXISTING")
        with patch.object(ids, "HAS_BOTO3", True), patch.object(ids, "boto3") as boto3_mod:
            _cache_index_file("s3://bucket/dir/shard.idx", str(local))
            boto3_mod.client.assert_not_called()

    def test_raises_when_boto3_missing(self, tmp_path):
        local = tmp_path / "shard.idx"
        with patch.object(ids, "HAS_BOTO3", False):
            with pytest.raises(ImportError, match="boto3 is required"):
                _cache_index_file("s3://bucket/dir/shard.idx", str(local))

    def test_rejects_unknown_scheme(self, tmp_path):
        local = tmp_path / "shard.idx"
        with pytest.raises(ValueError, match="Unsupported object storage path"):
            _cache_index_file("/local/shard.idx", str(local))


# ---------------------------------------------------------------------------
# _S3BinReader: ranged GETs and in-memory chunk caching
# ---------------------------------------------------------------------------


class TestS3BinReader:
    @staticmethod
    def _make_reader(payload: bytes, chunk_nbytes: int = 32):
        """Return an _S3BinReader whose client returns ``payload`` for any GET."""
        fake_client = MagicMock()

        def fake_get_object(Bucket, Key, Range):
            # Range looks like "bytes=<start>-<end>"
            start_end = Range.split("=")[1]
            start, end = (int(x) for x in start_end.split("-"))
            chunk = payload[start : end + 1]
            return {"Body": io.BytesIO(chunk)}

        fake_client.get_object.side_effect = fake_get_object

        cfg = ObjectStorageConfig(path_to_idx_cache="/cache", bin_chunk_nbytes=chunk_nbytes)
        with patch.object(ids, "HAS_BOTO3", True), patch.object(ids, "boto3") as boto3_mod:
            boto3_mod.client.return_value = fake_client
            reader = _S3BinReader("s3://b/key", cfg)
        return reader, fake_client

    def test_read_in_chunk_uses_cache(self):
        payload = np.arange(64, dtype=np.int32).tobytes()  # 256 bytes total
        reader, fake_client = self._make_reader(payload, chunk_nbytes=128)

        # First read fetches one chunk; second read within same window must reuse cache.
        a = reader.read(np.int32, count=4, offset=0)
        b = reader.read(np.int32, count=4, offset=16)
        np.testing.assert_array_equal(a, np.arange(4, dtype=np.int32))
        np.testing.assert_array_equal(b, np.arange(4, 8, dtype=np.int32))
        assert fake_client.get_object.call_count == 1, "Second read should hit in-memory cache"

    def test_read_crossing_chunk_refetches(self):
        payload = np.arange(64, dtype=np.int32).tobytes()
        reader, fake_client = self._make_reader(payload, chunk_nbytes=32)

        reader.read(np.int32, count=4, offset=0)  # bytes [0, 16) — chunk [0,32)
        reader.read(np.int32, count=4, offset=128)  # bytes [128,144) — new chunk
        assert fake_client.get_object.call_count == 2

    def test_init_rejects_invalid_chunk_size(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/cache", bin_chunk_nbytes=0)
        with patch.object(ids, "HAS_BOTO3", True), patch.object(ids, "boto3"):
            with pytest.raises(ValueError, match="bin_chunk_nbytes must be positive"):
                _S3BinReader("s3://b/key", cfg)

    def test_init_raises_when_boto3_missing(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/cache")
        with patch.object(ids, "HAS_BOTO3", False):
            with pytest.raises(ImportError, match="boto3 is required"):
                _S3BinReader("s3://b/key", cfg)


# ---------------------------------------------------------------------------
# IndexedDataset.exists short-circuit
# ---------------------------------------------------------------------------


class TestIndexedDatasetExistsObjectStorage:
    def test_s3_prefix_returns_true_without_network_call(self):
        # exists() is a @staticmethod; should not touch boto3 at all.
        assert ids.IndexedDataset.exists("s3://b/anything") is True

    def test_msc_prefix_returns_true_without_network_call(self):
        assert ids.IndexedDataset.exists("msc://profile/anything") is True

    def test_local_prefix_still_checks_filesystem(self, tmp_path):
        prefix = tmp_path / "shard"
        # Neither .idx nor .bin exist — should return False.
        assert ids.IndexedDataset.exists(str(prefix)) is False


# ---------------------------------------------------------------------------
# validate_dataset_asset_accessibility: skip local FS check for s3://
# ---------------------------------------------------------------------------


class TestValidateAssetAccessibilityObjectStorage:
    def test_skips_s3_path_when_config_provided(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/cache")
        # Would raise FileNotFoundError if it tried Path(...).exists()
        validate_dataset_asset_accessibility("s3://bucket/does/not/exist", object_storage_config=cfg)

    def test_skips_s3_paths_in_zipped_list(self):
        cfg = ObjectStorageConfig(path_to_idx_cache="/cache")
        validate_dataset_asset_accessibility(
            ["0.5", "s3://bucket/a", "0.5", "s3://bucket/b"], object_storage_config=cfg
        )

    def test_local_path_still_validated(self, tmp_path):
        cfg = ObjectStorageConfig(path_to_idx_cache="/cache")
        missing = tmp_path / "shard"
        with pytest.raises(FileNotFoundError):
            validate_dataset_asset_accessibility(str(missing), object_storage_config=cfg)

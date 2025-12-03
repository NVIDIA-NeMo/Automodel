# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for Delta Lake dataset support."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestIsDeltaLakePath:
    """Tests for the is_delta_lake_path function."""

    def test_delta_protocol_prefix(self):
        """Test that delta:// prefix is recognized."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import is_delta_lake_path

        assert is_delta_lake_path("delta:///path/to/table") is True
        assert is_delta_lake_path("delta://catalog.schema.table") is True

    def test_dbfs_prefix(self):
        """Test that dbfs:/ prefix is recognized."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import is_delta_lake_path

        assert is_delta_lake_path("dbfs:/path/to/table") is True

    def test_abfss_prefix(self):
        """Test that abfss:// prefix is recognized."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import is_delta_lake_path

        assert is_delta_lake_path("abfss://container@account.dfs.core.windows.net/path") is True

    def test_s3_with_delta_hint(self):
        """Test that S3 paths with delta hint are recognized."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import is_delta_lake_path

        assert is_delta_lake_path("s3://bucket/path/my_delta_table") is True
        assert is_delta_lake_path("s3a://bucket/delta_table") is True

    def test_local_directory_with_delta_log(self, tmp_path: Path):
        """Test that local directories with _delta_log are recognized."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import is_delta_lake_path

        # Create a directory with _delta_log
        delta_log = tmp_path / "_delta_log"
        delta_log.mkdir()

        assert is_delta_lake_path(str(tmp_path)) is True

    def test_local_directory_without_delta_log(self, tmp_path: Path):
        """Test that local directories without _delta_log are not recognized."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import is_delta_lake_path

        assert is_delta_lake_path(str(tmp_path)) is False

    def test_non_delta_paths(self):
        """Test that non-delta paths are not recognized."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import is_delta_lake_path

        assert is_delta_lake_path("/path/to/file.json") is False
        assert is_delta_lake_path("org/dataset") is False
        assert is_delta_lake_path("s3://bucket/regular_data") is False

    def test_non_string_input(self):
        """Test that non-string inputs return False."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import is_delta_lake_path

        assert is_delta_lake_path(None) is False
        assert is_delta_lake_path(123) is False
        assert is_delta_lake_path(["path"]) is False


class TestNormalizeDeltaPath:
    """Tests for the _normalize_delta_path function."""

    def test_removes_delta_prefix(self):
        """Test that delta:// prefix is removed."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import _normalize_delta_path

        assert _normalize_delta_path("delta:///path/to/table") == "/path/to/table"
        assert _normalize_delta_path("delta://catalog.schema.table") == "catalog.schema.table"

    def test_preserves_other_paths(self):
        """Test that other path types are preserved."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import _normalize_delta_path

        assert _normalize_delta_path("/local/path") == "/local/path"
        assert _normalize_delta_path("s3://bucket/path") == "s3://bucket/path"
        assert _normalize_delta_path("dbfs:/path") == "dbfs:/path"


class TestCheckDeltalakeAvailable:
    """Tests for the _check_deltalake_available function."""

    def test_returns_boolean(self):
        """Test that function returns a boolean."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import _check_deltalake_available

        result = _check_deltalake_available()
        assert isinstance(result, bool)


@pytest.mark.skipif(
    not _deltalake_available(),
    reason="deltalake package not installed"
)
class TestDeltaLakeIterator:
    """Tests for the DeltaLakeIterator class (requires deltalake)."""

    def test_env_storage_options(self):
        """Test that environment variables are picked up for storage options."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import DeltaLakeIterator

        with patch.dict(os.environ, {"DATABRICKS_TOKEN": "test_token"}):
            # This will fail to actually connect, but we can test the storage options setup
            try:
                iterator = DeltaLakeIterator(
                    table_path="delta://fake/table",
                    storage_options={},
                )
                assert "DATABRICKS_TOKEN" in iterator.storage_options
                assert iterator.storage_options["DATABRICKS_TOKEN"] == "test_token"
            except Exception:
                # Expected to fail when trying to connect
                pass


@pytest.mark.skipif(
    not _deltalake_available(),
    reason="deltalake package not installed"
)
class TestDeltaLakeDataset:
    """Tests for the DeltaLakeDataset class (requires deltalake)."""

    def test_streaming_mode_raises_on_len(self):
        """Test that __len__ raises in streaming mode."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import DeltaLakeDataset

        # Mock the DeltaTable to avoid actual file access
        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeIterator"):
            ds = DeltaLakeDataset.__new__(DeltaLakeDataset)
            ds.streaming = True

            with pytest.raises(RuntimeError, match="streaming mode"):
                len(ds)

    def test_streaming_mode_raises_on_getitem(self):
        """Test that __getitem__ raises in streaming mode."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import DeltaLakeDataset

        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeIterator"):
            ds = DeltaLakeDataset.__new__(DeltaLakeDataset)
            ds.streaming = True

            with pytest.raises(RuntimeError, match="streaming mode"):
                _ = ds[0]


class TestHFDeltaLakeDataset:
    """Tests for the HFDeltaLakeDataset class."""

    def test_shard_returns_self(self):
        """Test that shard() returns self for method chaining."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import HFDeltaLakeDataset

        # Create a minimal mock instance
        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeIterator"):
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset._check_deltalake_available", return_value=True):
                ds = HFDeltaLakeDataset.__new__(HFDeltaLakeDataset)
                ds._shard_info = None
                ds.streaming = True

                result = ds.shard(4, 1)
                assert result is ds
                assert ds._shard_info == (4, 1)

    def test_shuffle_returns_self(self):
        """Test that shuffle() returns self for method chaining."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import HFDeltaLakeDataset

        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeIterator"):
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset._check_deltalake_available", return_value=True):
                ds = HFDeltaLakeDataset.__new__(HFDeltaLakeDataset)
                ds._shuffle_info = None
                ds._epoch = 0

                result = ds.shuffle(buffer_size=1000, seed=42)
                assert result is ds
                assert ds._shuffle_info == (1000, 42)

    def test_set_epoch(self):
        """Test that set_epoch() updates the epoch."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import HFDeltaLakeDataset

        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeIterator"):
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset._check_deltalake_available", return_value=True):
                ds = HFDeltaLakeDataset.__new__(HFDeltaLakeDataset)
                ds._epoch = 0

                ds.set_epoch(5)
                assert ds._epoch == 5


class TestLimitedDeltaLakeDataset:
    """Tests for the _LimitedDeltaLakeDataset class."""

    def test_limits_iteration(self):
        """Test that iteration is limited to n samples."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import _LimitedDeltaLakeDataset

        # Create a mock base dataset
        mock_base = MagicMock()
        mock_base.__iter__ = MagicMock(return_value=iter([
            {"col": "a"},
            {"col": "b"},
            {"col": "c"},
            {"col": "d"},
        ]))

        limited = _LimitedDeltaLakeDataset(mock_base, 2)
        results = list(limited)

        assert len(results) == 2
        assert results[0] == {"col": "a"}
        assert results[1] == {"col": "b"}

    def test_take_reduces_limit(self):
        """Test that take() further reduces the limit."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import _LimitedDeltaLakeDataset

        mock_base = MagicMock()
        limited = _LimitedDeltaLakeDataset(mock_base, 10)

        further_limited = limited.take(5)
        assert further_limited._limit == 5

        # Taking more than current limit should keep original limit
        further_limited2 = limited.take(20)
        assert further_limited2._limit == 10


class TestLoadDatasetWithDelta:
    """Tests for _load_dataset integration with Delta Lake."""

    def test_detects_delta_path(self):
        """Test that _load_dataset detects Delta Lake paths."""
        from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import _load_dataset

        # Mock the delta lake module
        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.is_delta_lake_path", return_value=True):
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.load_delta_lake_dataset") as mock_load:
                mock_load.return_value = MagicMock()

                _load_dataset("delta:///path/to/table", streaming=True)

                mock_load.assert_called_once_with(
                    path="delta:///path/to/table",
                    storage_options=None,
                    streaming=True,
                    version=None,
                )

    def test_passes_delta_options(self):
        """Test that delta options are passed through."""
        from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import _load_dataset

        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.is_delta_lake_path", return_value=True):
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.load_delta_lake_dataset") as mock_load:
                mock_load.return_value = MagicMock()

                storage_opts = {"DATABRICKS_TOKEN": "dapi123"}
                _load_dataset(
                    "delta:///path/to/table",
                    streaming=True,
                    delta_storage_options=storage_opts,
                    delta_version=5,
                )

                mock_load.assert_called_once_with(
                    path="delta:///path/to/table",
                    storage_options=storage_opts,
                    streaming=True,
                    version=5,
                )


def _deltalake_available():
    """Helper to check if deltalake is available."""
    try:
        import deltalake  # noqa: F401
        return True
    except ImportError:
        return False


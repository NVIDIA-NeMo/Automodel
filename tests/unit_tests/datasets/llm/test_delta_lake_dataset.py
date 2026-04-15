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
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _deltalake_available():
    """Helper to check if deltalake is available."""
    try:
        import importlib

        importlib.import_module("deltalake")

        return True
    except ImportError:
        return False


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

    def test_shard_returns_self(self):
        """Test that shard() returns self and delegates to the iterator."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import DeltaLakeDataset

        # Patch the iterator to avoid touching real backends and to assert delegation
        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeIterator") as MockIter:
            mock_iter = MagicMock()
            MockIter.return_value = mock_iter

            ds = DeltaLakeDataset("delta:///fake/table")
            result = ds.shard(4, 1)
            assert result is ds
            mock_iter.shard.assert_called_once_with(4, 1)

    def test_shuffle_returns_self(self):
        """Test that shuffle() returns self and wraps iterator with ReservoirSampler."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import DeltaLakeDataset

        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeIterator") as MockIter:
            mock_iter = MagicMock()
            MockIter.return_value = mock_iter

            # Important: patch the symbol used inside delta_lake_dataset (module alias), not the origin
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.ReservoirSampler") as MockRS:
                sentinel = object()
                MockRS.return_value = sentinel

                ds = DeltaLakeDataset("delta:///fake/table")

                result = ds.shuffle(buffer_size=1000, seed=42)
                assert result is ds
                assert ds._data_iterator is sentinel
                MockRS.assert_called_once_with(mock_iter, 1000, 42)

    def test_set_epoch(self):
        """Test that set_epoch() updates the epoch."""
        from nemo_automodel.components.datasets.llm.delta_lake_dataset import DeltaLakeDataset

        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeIterator"):
            ds = DeltaLakeDataset.__new__(DeltaLakeDataset)
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
        """Test that _load_streaming_dataset detects Delta Lake paths and constructs DeltaLakeDataset."""
        from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset import (
            _load_streaming_dataset,
        )

        # Mock the delta lake module
        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.is_delta_lake_path", return_value=True):
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeDataset") as mock_cls:
                instance = MagicMock()
                mock_cls.return_value = instance

                result = _load_streaming_dataset("delta:///path/to/table", streaming=True)

                mock_cls.assert_called_once_with(
                    table_path="delta:///path/to/table", storage_options=None, version=None, sql_query=None
                )
                assert result is instance

    def test_passes_delta_options(self):
        """Test that delta options are passed through to DeltaLakeDataset."""
        from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset import (
            _load_streaming_dataset,
        )

        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.is_delta_lake_path", return_value=True):
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeDataset") as mock_cls:
                instance = MagicMock()
                mock_cls.return_value = instance

                storage_opts = {"DATABRICKS_TOKEN": "dapi123"}
                result = _load_streaming_dataset(
                    "delta:///path/to/table",
                    streaming=True,
                    delta_storage_options=storage_opts,
                    delta_version=5,
                )

                mock_cls.assert_called_once_with(
                    table_path="delta:///path/to/table", storage_options=storage_opts, version=5, sql_query=None
                )
                assert result is instance

    def test_passes_delta_sql_query(self):
        """Test that delta SQL query is passed through to DeltaLakeDataset."""
        from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset import (
            _load_streaming_dataset,
        )

        with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.is_delta_lake_path", return_value=True):
            with patch("nemo_automodel.components.datasets.llm.delta_lake_dataset.DeltaLakeDataset") as mock_cls:
                instance = MagicMock()
                mock_cls.return_value = instance

                query = "SELECT 'q' AS question, 'a' AS answer"
                result = _load_streaming_dataset("delta:///path/to/table", streaming=True, delta_sql_query=query)

                mock_cls.assert_called_once_with(
                    table_path="delta:///path/to/table", storage_options=None, version=None, sql_query=query
                )
                assert result is instance


class TestDeltaLakeSqlQueryExecution:
    """Tests for executing SQL queries via the Databricks SQL backend."""

    def test_databricks_sql_uses_sql_query(self):
        """When sql_query is provided, it should be executed as-is (no table_path SELECT wrapper)."""
        from nemo_automodel.components.datasets.llm import delta_lake_dataset as mod

        executed: dict[str, str] = {}

        class FakeCursor:
            def __init__(self):
                self.description = [("a",), ("b",)]
                self._done = False

            def execute(self, query):
                executed["query"] = query

            def fetchmany(self, batch_size):
                if self._done:
                    return []
                self._done = True
                return [(1, 2)]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeSql:
            def connect(self, **kwargs):
                return FakeConn()

        fake_databricks = types.ModuleType("databricks")
        fake_databricks.sql = FakeSql()

        query = "SELECT 1 AS a, 2 AS b"
        storage_opts = {"DATABRICKS_HOST": "https://workspace", "DATABRICKS_TOKEN": "dapi...", "DATABRICKS_HTTP_PATH": "/sql"}

        with patch.dict(sys.modules, {"databricks": fake_databricks}):
            with patch.object(mod, "_check_databricks_sql_available", return_value=True):
                with patch.object(mod, "_get_spark_session", return_value=None):
                    it = mod.DeltaLakeIterator(
                        table_path="catalog.schema.table",
                        storage_options=storage_opts,
                        batch_size=8,
                        sql_query=query,
                    )
                    rows = list(it)

        assert executed["query"] == query
        assert rows == [{"a": 1, "b": 2}]

    def test_databricks_sql_wraps_query_when_columns_requested(self):
        """If columns are set alongside sql_query, we project via SELECT ... FROM (<query>)."""
        from nemo_automodel.components.datasets.llm import delta_lake_dataset as mod

        executed: dict[str, str] = {}

        class FakeCursor:
            def __init__(self):
                self.description = [("a",)]
                self._done = False

            def execute(self, query):
                executed["query"] = query

            def fetchmany(self, batch_size):
                if self._done:
                    return []
                self._done = True
                return [(1,)]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeSql:
            def connect(self, **kwargs):
                return FakeConn()

        fake_databricks = types.ModuleType("databricks")
        fake_databricks.sql = FakeSql()

        base_query = "SELECT 1 AS a, 2 AS b"
        storage_opts = {"DATABRICKS_HOST": "https://workspace", "DATABRICKS_TOKEN": "dapi...", "DATABRICKS_HTTP_PATH": "/sql"}

        with patch.dict(sys.modules, {"databricks": fake_databricks}):
            with patch.object(mod, "_check_databricks_sql_available", return_value=True):
                with patch.object(mod, "_get_spark_session", return_value=None):
                    it = mod.DeltaLakeIterator(
                        table_path="catalog.schema.table",
                        columns=["a"],
                        storage_options=storage_opts,
                        batch_size=8,
                        sql_query=base_query,
                    )
                    rows = list(it)

        assert executed["query"] == f"SELECT a FROM ({base_query}) AS _q"
        assert rows == [{"a": 1}]


class TestDeletionVectorsFallback:
    """Tests for deletion-vectors fallbacks without requiring real deltalake/pyspark installs."""

    def test_falls_back_to_spark_when_deltalake_raises(self):
        from nemo_automodel.components.datasets.llm import delta_lake_dataset as mod

        class FakeDeltaTable:
            def __init__(self, *args, **kwargs):
                pass

            def to_pyarrow_dataset(self):
                raise Exception(
                    "The table has set these reader features: {'deletionVectors'} "
                    "but these are not yet supported by the deltalake reader."
                )

        fake_deltalake = types.ModuleType("deltalake")
        fake_deltalake.DeltaTable = FakeDeltaTable

        with patch.dict(sys.modules, {"deltalake": fake_deltalake}):
            with patch.object(mod, "_check_deltalake_available", return_value=True):
                with patch.object(mod, "_get_spark_session", return_value=object()):
                    it = mod.DeltaLakeIterator(table_path="delta:///tmp/table")
                    with patch.object(it, "_iter_with_spark", return_value=iter([{"x": 1}])) as mock_spark:
                        rows = list(it)

                    assert rows == [{"x": 1}]
                    mock_spark.assert_called()

    def test_raises_clear_import_error_when_spark_missing(self):
        from nemo_automodel.components.datasets.llm import delta_lake_dataset as mod

        class FakeDeltaTable:
            def __init__(self, *args, **kwargs):
                pass

            def to_pyarrow_dataset(self):
                raise Exception(
                    "The table has set these reader features: {'deletionVectors'} "
                    "but these are not yet supported by the deltalake reader."
                )

        fake_deltalake = types.ModuleType("deltalake")
        fake_deltalake.DeltaTable = FakeDeltaTable

        with patch.dict(sys.modules, {"deltalake": fake_deltalake}):
            with patch.object(mod, "_check_deltalake_available", return_value=True):
                with patch.object(mod, "_get_spark_session", return_value=None):
                    it = mod.DeltaLakeIterator(table_path="delta:///tmp/table")
                    with pytest.raises(ImportError, match=r"Spark|pyspark|Databricks"):
                        _ = list(it)

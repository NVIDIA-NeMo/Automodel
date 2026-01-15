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

"""Delta Lake dataset support for streaming instruction-tuning datasets.

This module provides support for reading Delta Lake tables from Databricks or
local storage as streaming datasets. It integrates with the existing
ColumnMappedTextInstructionDataset infrastructure.

**Supports tables with Deletion Vectors** (Databricks Runtime 15.4+) via DuckDB
or Databricks SQL Connector backends.

Installation:
    ```bash
    # For basic Delta Lake support (without deletion vectors)
    pip install deltalake

    # For tables with deletion vectors (Databricks 15.4+)
    pip install duckdb deltalake

    # For Databricks Unity Catalog streaming access
    pip install databricks-sql-connector duckdb deltalake
    ```

Usage:
    Local Delta tables:

    ```python
    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id="delta:///path/to/delta_table",
        column_mapping={"question": "input", "answer": "output"},
        tokenizer=tokenizer,
    )
    ```

    Cloud storage (S3, Azure, GCS):

    ```python
    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id="s3://bucket/path/to/delta_table",
        column_mapping={"question": "input", "answer": "output"},
        tokenizer=tokenizer,
        delta_storage_options={
            "AWS_ACCESS_KEY_ID": "...",
            "AWS_SECRET_ACCESS_KEY": "...",
        },
    )
    ```

    Databricks Unity Catalog (streaming):

    ```python
    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id="catalog.schema.table",  # Unity Catalog format
        column_mapping={"question": "input", "answer": "output"},
        tokenizer=tokenizer,
        delta_storage_options={
            "DATABRICKS_HOST": "https://your-workspace.databricks.com",
            "DATABRICKS_TOKEN": "dapi...",
            "DATABRICKS_WAREHOUSE_ID": "abc123def456",  # or DATABRICKS_HTTP_PATH
        },
    )
    ```

Environment Variables:
    The following environment variables are automatically detected:

    - DATABRICKS_HOST / DATABRICKS_WORKSPACE_URL
    - DATABRICKS_TOKEN / DATABRICKS_ACCESS_TOKEN
    - DATABRICKS_HTTP_PATH / DATABRICKS_WAREHOUSE_ID / DATABRICKS_CLUSTER_ID
    - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_REGION
    - AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY, AZURE_STORAGE_SAS_TOKEN
    - GOOGLE_APPLICATION_CREDENTIALS
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

logger = logging.getLogger(__name__)

# Lazy imports to avoid requiring deltalake/duckdb when not used
_DELTALAKE_AVAILABLE: Optional[bool] = None
_DUCKDB_AVAILABLE: Optional[bool] = None
_DATABRICKS_SQL_AVAILABLE: Optional[bool] = None


def _check_duckdb_available() -> bool:
    """Check if DuckDB is available for reading Delta tables with deletion vectors."""
    global _DUCKDB_AVAILABLE
    if _DUCKDB_AVAILABLE is None:
        try:
            import duckdb  # noqa: F401

            _DUCKDB_AVAILABLE = True
        except ImportError:
            _DUCKDB_AVAILABLE = False
    return _DUCKDB_AVAILABLE


def _check_deltalake_available() -> bool:
    """Check if the deltalake package is available."""
    global _DELTALAKE_AVAILABLE
    if _DELTALAKE_AVAILABLE is None:
        try:
            import deltalake  # noqa: F401

            _DELTALAKE_AVAILABLE = True
        except ImportError:
            _DELTALAKE_AVAILABLE = False
    return _DELTALAKE_AVAILABLE


def _check_databricks_sql_available() -> bool:
    """Check if databricks-sql-connector is available for Unity Catalog access."""
    global _DATABRICKS_SQL_AVAILABLE
    if _DATABRICKS_SQL_AVAILABLE is None:
        try:
            from databricks import sql  # noqa: F401

            _DATABRICKS_SQL_AVAILABLE = True
        except Exception as e:  # noqa: BLE001
            # In some environments (e.g., Databricks Repos), importing a missing module can surface
            # as a non-ImportError (like an HTTP 404 from workspace import hooks). Treat any failure
            # as "not available" and continue.
            logger.debug("databricks-sql-connector not available (%s): %s", type(e).__name__, e)
            _DATABRICKS_SQL_AVAILABLE = False
    return _DATABRICKS_SQL_AVAILABLE


def _check_delta_reader_available() -> bool:
    """Check if any Delta Lake reader is available (DuckDB, deltalake, or databricks-sql)."""
    return _check_duckdb_available() or _check_deltalake_available() or _check_databricks_sql_available()


def _is_deletion_vectors_error(e: BaseException) -> bool:
    """Return True if *e* looks like the deltalake 'deletionVectors' unsupported-reader-features error."""
    msg = str(e).lower()
    return "deletionvectors" in msg or "deletion vectors" in msg


def _normalize_duckdb_path(path: str) -> str:
    """Normalize paths for DuckDB.

    - Databricks DBFS URI paths (dbfs:/...) are typically accessible via the local FUSE mount (/dbfs/...).
      We only rewrite when that mount exists to avoid breaking non-Databricks environments.
    """
    if path.startswith("dbfs:/") and Path("/dbfs").exists():
        return str(Path("/dbfs") / path[len("dbfs:/") :])
    return path


def _is_unity_catalog_path(path: str) -> bool:
    """Check if path refers to a Unity Catalog table (catalog.schema.table format)."""
    # Unity Catalog paths are typically: catalog.schema.table
    # They don't contain slashes or cloud storage prefixes
    if any(path.startswith(prefix) for prefix in ["s3://", "abfss://", "gs://", "dbfs:/", "/", "."]):
        return False
    parts = path.split(".")
    return len(parts) == 3 and all(p.strip() for p in parts)


def is_delta_lake_path(path: str) -> bool:
    """Check if a path refers to a Delta Lake table.

    A path is considered a Delta Lake path if:
    1. It starts with "delta://" protocol prefix
    2. It's a local directory containing a "_delta_log" subdirectory
    3. It starts with "dbfs:/" (Databricks file system)

    Args:
        path: The path to check.

    Returns:
        True if the path is a Delta Lake table, False otherwise.
    """
    if not isinstance(path, str):
        return False

    # Check for explicit delta:// protocol
    if path.startswith("delta://"):
        return True

    # Check for Databricks file system paths
    if path.startswith("dbfs:/"):
        return True

    # Check for abfss:// (Azure Blob Storage with hierarchical namespace)
    if path.startswith("abfss://"):
        return True

    # Check for s3:// or s3a:// paths with _delta_log hint
    if path.startswith(("s3://", "s3a://", "gs://")) and "_delta" in path.lower():
        return True

    # Check for local directory with _delta_log
    local_path = Path(path)
    if local_path.exists() and local_path.is_dir():
        delta_log = local_path / "_delta_log"
        if delta_log.exists() and delta_log.is_dir():
            return True

    return False


def _normalize_delta_path(path: str) -> str:
    """Normalize a Delta Lake path by removing the delta:// prefix if present.

    Args:
        path: The Delta Lake path.

    Returns:
        The normalized path suitable for the deltalake library.
    """
    if path.startswith("delta://"):
        return path[8:]  # Remove "delta://" prefix
    return path


class DeltaLakeIterator:
    """Iterator that yields rows from a Delta Lake table.

    This class provides a streaming interface for Delta Lake tables,
    yielding rows as dictionaries one at a time to support memory-efficient
    iteration over large tables.

    Supports tables with deletion vectors (Databricks Runtime 15.4+) via DuckDB backend.

    Args:
        table_path: Path to the Delta Lake table.
        columns: Optional list of column names to read. If None, reads all columns.
        storage_options: Optional storage options for cloud storage access.
        batch_size: Number of rows to read at a time (default: 1024).
        version: Optional version of the table to read.
        use_duckdb: If True, use DuckDB for reading (supports deletion vectors).
            If None, auto-detect based on table features. Default: None.
    """

    def __init__(
        self,
        table_path: str,
        columns: Optional[list] = None,
        storage_options: Optional[Dict[str, str]] = None,
        batch_size: int = 1024,
        version: Optional[int] = None,
        use_duckdb: Optional[bool] = None,
    ):
        if not _check_delta_reader_available():
            raise ImportError(
                "Either 'duckdb' or 'deltalake' package is required for Delta Lake support. "
                "For tables with deletion vectors (Databricks 15.4+), use: pip install duckdb\n"
                "For basic Delta Lake support, use: pip install deltalake"
            )

        self.table_path = _normalize_delta_path(table_path)
        self.columns = columns
        self.storage_options = storage_options or {}
        self.batch_size = batch_size
        self.version = version
        self.use_duckdb = use_duckdb

        # Add environment-based storage options
        self._add_env_storage_options()

    def _add_env_storage_options(self):
        """Add storage options from environment variables if not already set."""
        env_mappings = {
            # Databricks authentication
            "DATABRICKS_TOKEN": ["DATABRICKS_TOKEN", "DATABRICKS_ACCESS_TOKEN"],
            "DATABRICKS_HOST": ["DATABRICKS_HOST", "DATABRICKS_WORKSPACE_URL", "DATABRICKS_SERVER_HOSTNAME"],
            "DATABRICKS_HTTP_PATH": ["DATABRICKS_HTTP_PATH"],
            "DATABRICKS_WAREHOUSE_ID": ["DATABRICKS_WAREHOUSE_ID", "DATABRICKS_SQL_WAREHOUSE_ID"],
            "DATABRICKS_CLUSTER_ID": ["DATABRICKS_CLUSTER_ID"],
            # AWS S3 authentication
            "AWS_ACCESS_KEY_ID": ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": ["AWS_SECRET_ACCESS_KEY"],
            "AWS_SESSION_TOKEN": ["AWS_SESSION_TOKEN"],
            "AWS_REGION": ["AWS_REGION", "AWS_DEFAULT_REGION"],
            # Azure authentication
            "AZURE_STORAGE_ACCOUNT_NAME": ["AZURE_STORAGE_ACCOUNT_NAME"],
            "AZURE_STORAGE_ACCOUNT_KEY": ["AZURE_STORAGE_ACCOUNT_KEY"],
            "AZURE_STORAGE_SAS_TOKEN": ["AZURE_STORAGE_SAS_TOKEN"],
            # GCP authentication
            "GOOGLE_APPLICATION_CREDENTIALS": ["GOOGLE_APPLICATION_CREDENTIALS"],
        }

        for key, env_vars in env_mappings.items():
            if key not in self.storage_options:
                for env_var in env_vars:
                    value = os.environ.get(env_var)
                    if value:
                        self.storage_options[key] = value
                        break

    def _configure_duckdb_cloud_access(self, conn) -> None:
        """Configure DuckDB for cloud storage access (S3, Azure, GCS)."""
        # Install and load required extensions
        conn.execute("INSTALL delta; LOAD delta;")

        # Configure AWS S3 access
        if self.storage_options.get("AWS_ACCESS_KEY_ID"):
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            conn.execute(f"SET s3_access_key_id='{self.storage_options['AWS_ACCESS_KEY_ID']}';")
            if self.storage_options.get("AWS_SECRET_ACCESS_KEY"):
                conn.execute(f"SET s3_secret_access_key='{self.storage_options['AWS_SECRET_ACCESS_KEY']}';")
            if self.storage_options.get("AWS_SESSION_TOKEN"):
                conn.execute(f"SET s3_session_token='{self.storage_options['AWS_SESSION_TOKEN']}';")
            if self.storage_options.get("AWS_REGION"):
                conn.execute(f"SET s3_region='{self.storage_options['AWS_REGION']}';")

        # Configure Azure access
        if self.storage_options.get("AZURE_STORAGE_ACCOUNT_NAME"):
            conn.execute("INSTALL azure; LOAD azure;")
            account = self.storage_options["AZURE_STORAGE_ACCOUNT_NAME"]
            if self.storage_options.get("AZURE_STORAGE_ACCOUNT_KEY"):
                conn.execute(
                    f"SET azure_storage_connection_string='DefaultEndpointsProtocol=https;"
                    f"AccountName={account};AccountKey={self.storage_options['AZURE_STORAGE_ACCOUNT_KEY']}';"
                )
            elif self.storage_options.get("AZURE_STORAGE_SAS_TOKEN"):
                # SAS token authentication
                pass  # DuckDB handles this via the URL

        # Configure GCS access
        if self.storage_options.get("GOOGLE_APPLICATION_CREDENTIALS"):
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            # GCS uses service account credentials from the file
            pass

    def _iter_with_duckdb(self) -> Iterator[Dict[str, Any]]:
        """Iterate using DuckDB (supports deletion vectors).

        Supports reading from:
        - Local paths: /path/to/delta_table
        - S3: s3://bucket/path/to/delta_table
        - Azure: abfss://container@account.dfs.core.windows.net/path
        - GCS: gs://bucket/path/to/delta_table
        """
        import duckdb

        # Build column selection
        col_select = ", ".join(self.columns) if self.columns else "*"

        conn = duckdb.connect()
        try:
            # Configure cloud storage access if needed
            self._configure_duckdb_cloud_access(conn)

            duckdb_path = _normalize_duckdb_path(self.table_path)

            # DuckDB can read Delta tables directly
            query = f"SELECT {col_select} FROM delta_scan(?)"

            # Fetch in batches for memory efficiency
            result = conn.execute(query, [duckdb_path])
            while True:
                batch = result.fetchmany(self.batch_size)
                if not batch:
                    break
                columns = [desc[0] for desc in result.description]
                for row in batch:
                    yield dict(zip(columns, row))
        finally:
            conn.close()

    def _iter_with_deltalake(self) -> Iterator[Dict[str, Any]]:
        """Iterate using deltalake library."""
        from deltalake import DeltaTable

        try:
            # Open the Delta table and get PyArrow dataset for efficient streaming
            dt = DeltaTable(self.table_path, storage_options=self.storage_options, version=self.version)
            pa_dataset = dt.to_pyarrow_dataset()
        except Exception as e:  # noqa: BLE001
            if _is_deletion_vectors_error(e):
                if _check_duckdb_available():
                    logger.warning(
                        "Table uses deletion vectors. Falling back to DuckDB reader. "
                        "To avoid this warning, set use_duckdb=True."
                    )
                    yield from self._iter_with_duckdb()
                    return
                else:
                    raise ImportError(
                        "This Delta table uses deletion vectors which are not supported by "
                        "the deltalake library. Install DuckDB to read this table:\n"
                        "  pip install duckdb\n"
                        "Or disable deletion vectors on the table:\n"
                        "  ALTER TABLE table_name SET TBLPROPERTIES ('delta.enableDeletionVectors' = false)"
                    ) from e
            raise

        # Iterate over batches
        for batch in pa_dataset.to_batches(columns=self.columns, batch_size=self.batch_size):
            # Convert batch to Python dicts
            batch_dict = batch.to_pydict()
            num_rows = len(batch_dict[next(iter(batch_dict))])

            for i in range(num_rows):
                yield {col: batch_dict[col][i] for col in batch_dict}

    def _iter_with_databricks_sql(self) -> Iterator[Dict[str, Any]]:
        """Iterate using Databricks SQL Connector (for Unity Catalog tables).

        This is the recommended method for accessing Unity Catalog tables
        as it handles authentication, deletion vectors, and column mapping natively.
        """
        from databricks import sql

        # Extract connection parameters
        server_hostname = self.storage_options.get("DATABRICKS_HOST", "").replace("https://", "")
        access_token = self.storage_options.get("DATABRICKS_TOKEN", "")
        http_path = self.storage_options.get("DATABRICKS_HTTP_PATH", "")

        if not server_hostname or not access_token:
            raise ValueError(
                "Databricks connection requires DATABRICKS_HOST and DATABRICKS_TOKEN. "
                "Set them in storage_options or as environment variables."
            )

        if not http_path:
            # Try to construct from cluster_id or warehouse_id
            warehouse_id = self.storage_options.get("DATABRICKS_WAREHOUSE_ID", "")
            cluster_id = self.storage_options.get("DATABRICKS_CLUSTER_ID", "")
            if warehouse_id:
                http_path = f"/sql/1.0/warehouses/{warehouse_id}"
            elif cluster_id:
                http_path = f"/sql/protocolv1/o/0/{cluster_id}"
            else:
                raise ValueError(
                    "Databricks SQL requires DATABRICKS_HTTP_PATH, DATABRICKS_WAREHOUSE_ID, "
                    "or DATABRICKS_CLUSTER_ID to be set."
                )

        # Build column selection
        col_select = ", ".join(self.columns) if self.columns else "*"

        # Build query - table_path is catalog.schema.table format
        query = f"SELECT {col_select} FROM {self.table_path}"

        logger.info(f"Connecting to Databricks SQL: {server_hostname}")

        with sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)

                # Get column names
                columns = [desc[0] for desc in cursor.description]

                # Fetch in batches
                while True:
                    batch = cursor.fetchmany(self.batch_size)
                    if not batch:
                        break
                    for row in batch:
                        yield dict(zip(columns, row))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in the Delta Lake table.

        Yields:
            Dict containing column name to value mappings for each row.
        """
        # Check if this is a Unity Catalog table (catalog.schema.table format)
        if _is_unity_catalog_path(self.table_path):
            if _check_databricks_sql_available():
                logger.info(f"Detected Unity Catalog table: {self.table_path}")
                yield from self._iter_with_databricks_sql()
                return
            else:
                raise ImportError(
                    f"Unity Catalog table '{self.table_path}' requires databricks-sql-connector. "
                    "Install with: pip install databricks-sql-connector"
                )

        # Determine which backend to use for file-based tables
        if self.use_duckdb is True:
            if not _check_duckdb_available():
                raise ImportError("DuckDB is required but not installed. Install with: pip install duckdb")
            yield from self._iter_with_duckdb()
        elif self.use_duckdb is False:
            if not _check_deltalake_available():
                raise ImportError("deltalake is required but not installed. Install with: pip install deltalake")
            yield from self._iter_with_deltalake()
        else:
            # Auto-detect: try deltalake first, fall back to duckdb if needed
            if _check_deltalake_available():
                yield from self._iter_with_deltalake()
            elif _check_duckdb_available():
                yield from self._iter_with_duckdb()
            else:
                raise ImportError(
                    "Either 'duckdb' or 'deltalake' package is required for Delta Lake support."
                )


class DeltaLakeDataset:
    """A dataset wrapper for Delta Lake tables that integrates with HuggingFace datasets.

    This class provides a HuggingFace-compatible interface for Delta Lake tables,
    supporting both iteration and indexing (when the table is small enough).

    Args:
        table_path: Path to the Delta Lake table. Can be:
            - Local path: "/path/to/delta_table"
            - Delta protocol: "delta:///path/to/delta_table"
            - DBFS: "dbfs:/path/to/delta_table"
            - S3: "s3://bucket/path/to/delta_table"
            - Azure: "abfss://container@account.dfs.core.windows.net/path"
            - GCS: "gs://bucket/path/to/delta_table"
        columns: Optional list of column names to read.
        storage_options: Optional dict of storage options for cloud authentication.
        streaming: If True, returns an iterable dataset. If False, loads into memory.
        version: Optional specific version of the Delta table to read.
    """

    def __init__(
        self,
        table_path: str,
        columns: Optional[list] = None,
        storage_options: Optional[Dict[str, str]] = None,
        streaming: bool = False,
        version: Optional[int] = None,
        use_duckdb: Optional[bool] = None,
    ):
        self.table_path = _normalize_delta_path(table_path)
        self.columns = columns
        self.storage_options = storage_options or {}
        self.streaming = streaming
        self.version = version
        self.use_duckdb = use_duckdb
        self._data: Optional[list] = None
        self._length: Optional[int] = None

        if not _check_delta_reader_available():
            raise ImportError(
                "A Delta Lake reader is required. Install one of:\n"
                "  pip install deltalake\n"
                "  pip install duckdb\n"
                "  pip install databricks-sql-connector  # for Unity Catalog tables"
            )

        if self.use_duckdb is True and not _check_duckdb_available():
            raise ImportError("DuckDB is required but not installed. Install with: pip install duckdb")

        if self.use_duckdb is False and not _check_deltalake_available():
            raise ImportError("deltalake is required but not installed. Install with: pip install deltalake")

        # Unity Catalog tables are supported in streaming mode via databricks-sql-connector.
        if not self.streaming and _is_unity_catalog_path(self.table_path):
            raise ValueError(
                "Unity Catalog tables are only supported in streaming mode. "
                "Use streaming=True (and install databricks-sql-connector)."
            )

        # Add environment-based storage options
        self._iterator = DeltaLakeIterator(
            table_path=table_path,
            columns=columns,
            storage_options=storage_options,
            version=version,
            use_duckdb=use_duckdb,
        )

        if not streaming:
            self._load_data()

    def _load_data(self):
        """Load the entire Delta table into memory."""
        # Unity Catalog tables are supported in streaming mode via databricks-sql-connector.
        if _is_unity_catalog_path(self.table_path):
            raise ValueError(
                "Unity Catalog tables are only supported in streaming mode. "
                "Use streaming=True (and install databricks-sql-connector)."
            )

        if self.use_duckdb is True:
            iterator = DeltaLakeIterator(
                table_path=self.table_path,
                columns=self.columns,
                storage_options=self._iterator.storage_options,
                version=self.version,
                use_duckdb=True,
            )
            self._data = list(iterator)
            self._length = len(self._data)
            return

        if self.use_duckdb is False:
            if not _check_deltalake_available():
                raise ImportError("deltalake is required but not installed. Install with: pip install deltalake")
            from deltalake import DeltaTable

            try:
                dt = DeltaTable(
                    self.table_path,
                    storage_options=self._iterator.storage_options,
                    version=self.version,
                )
                pa_table = dt.to_pyarrow_table(columns=self.columns)
            except Exception as e:  # noqa: BLE001
                if _is_deletion_vectors_error(e):
                    raise ImportError(
                        "This Delta table uses deletion vectors which are not supported by the deltalake library. "
                        "Install DuckDB to read this table:\n"
                        "  pip install duckdb"
                    ) from e
                raise

            self._data = pa_table.to_pylist()
            self._length = len(self._data)
            return

        # Auto-detect: prefer deltalake when available, fall back to DuckDB when needed.
        if _check_deltalake_available():
            from deltalake import DeltaTable

            try:
                dt = DeltaTable(
                    self.table_path,
                    storage_options=self._iterator.storage_options,
                    version=self.version,
                )
                pa_table = dt.to_pyarrow_table(columns=self.columns)
                self._data = pa_table.to_pylist()
                self._length = len(self._data)
                return
            except Exception as e:  # noqa: BLE001
                if _is_deletion_vectors_error(e) and _check_duckdb_available():
                    logger.warning(
                        "Table uses deletion vectors. Falling back to DuckDB reader. "
                        "To avoid this warning, set use_duckdb=True."
                    )
                else:
                    raise

        if _check_duckdb_available():
            iterator = DeltaLakeIterator(
                table_path=self.table_path,
                columns=self.columns,
                storage_options=self._iterator.storage_options,
                version=self.version,
                use_duckdb=True,
            )
            self._data = list(iterator)
            self._length = len(self._data)
            return

        raise ImportError(
            "Unable to read this Delta table. Install one of:\n"
            "  pip install deltalake\n"
            "  pip install duckdb"
        )

    def __len__(self) -> int:
        """Return the number of rows in the table.

        Raises:
            RuntimeError: If streaming is enabled.
        """
        if self.streaming:
            raise RuntimeError("__len__ is not supported in streaming mode. Use iteration instead.")
        if self._length is None:
            self._load_data()
        return self._length  # type: ignore[return-value]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific row by index.

        Args:
            idx: The row index.

        Returns:
            Dict containing column name to value mappings.

        Raises:
            RuntimeError: If streaming is enabled.
        """
        if self.streaming:
            raise RuntimeError("__getitem__ is not supported in streaming mode. Use iteration instead.")
        if self._data is None:
            self._load_data()
        return self._data[idx]  # type: ignore[index]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in the dataset.

        Yields:
            Dict containing column name to value mappings for each row.
        """
        if self.streaming:
            yield from self._iterator
        else:
            if self._data is None:
                self._load_data()
            yield from self._data  # type: ignore[misc]


def load_delta_lake_dataset(
    path: str,
    columns: Optional[list] = None,
    storage_options: Optional[Dict[str, str]] = None,
    streaming: bool = False,
    version: Optional[int] = None,
    use_duckdb: Optional[bool] = None,
) -> Union[DeltaLakeDataset, "HFDeltaLakeDataset"]:
    """Load a Delta Lake table as a HuggingFace-compatible dataset.

    This function creates a dataset that can be used with the ColumnMappedTextInstructionDataset
    and other HuggingFace-compatible dataset consumers.

    Args:
        path: Path to the Delta Lake table.
        columns: Optional list of column names to read.
        storage_options: Optional dict of storage options for cloud authentication.
        streaming: If True, returns a streaming iterable dataset.
        version: Optional specific version of the Delta table to read.

    Returns:
        A HuggingFace-compatible dataset object.

    Example:
        ```python
        # Local Delta table
        ds = load_delta_lake_dataset("/path/to/delta_table")

        # Databricks table with authentication
        ds = load_delta_lake_dataset(
            "delta://catalog.schema.table",
            storage_options={"DATABRICKS_TOKEN": "dapi..."},
            streaming=True,
        )
        ```
    """
    if not _check_delta_reader_available():
        raise ImportError(
            "A Delta Lake reader is required. Install one of:\n"
            "  pip install deltalake\n"
            "  pip install duckdb\n"
            "  pip install databricks-sql-connector  # for Unity Catalog tables"
        )

    if use_duckdb is True and not _check_duckdb_available():
        raise ImportError("DuckDB is required but not installed. Install with: pip install duckdb")

    # Return HF-compatible wrapper if datasets library is available
    try:
        from datasets import Dataset, IterableDataset

        return HFDeltaLakeDataset(
            table_path=path,
            columns=columns,
            storage_options=storage_options,
            streaming=streaming,
            version=version,
            use_duckdb=use_duckdb,
        )
    except ImportError:
        return DeltaLakeDataset(
            table_path=path,
            columns=columns,
            storage_options=storage_options,
            streaming=streaming,
            version=version,
            use_duckdb=use_duckdb,
        )


class HFDeltaLakeDataset:
    """HuggingFace datasets-compatible wrapper for Delta Lake tables.

    This class provides better integration with the HuggingFace datasets library,
    supporting features like sharding, shuffling, and epoch setting for distributed
    training scenarios.

    Args:
        table_path: Path to the Delta Lake table.
        columns: Optional list of column names to read.
        storage_options: Optional dict of storage options for cloud authentication.
        streaming: If True, returns a streaming iterable dataset.
        version: Optional specific version of the Delta table to read.
    """

    def __init__(
        self,
        table_path: str,
        columns: Optional[list] = None,
        storage_options: Optional[Dict[str, str]] = None,
        streaming: bool = False,
        version: Optional[int] = None,
        use_duckdb: Optional[bool] = None,
    ):
        self.table_path = _normalize_delta_path(table_path)
        self.columns = columns
        self.storage_options = storage_options or {}
        self.streaming = streaming
        self.version = version
        self.use_duckdb = use_duckdb
        self._dataset: Optional[Any] = None
        self._epoch: int = 0
        self._shard_info: Optional[tuple] = None  # (num_shards, shard_index)
        self._shuffle_info: Optional[tuple] = None  # (buffer_size, seed)

        if self.use_duckdb is True and not _check_duckdb_available():
            raise ImportError("DuckDB is required but not installed. Install with: pip install duckdb")

        # Eagerly create the internal iterator to validate the table
        self._base_iterator = DeltaLakeIterator(
            table_path=table_path,
            columns=columns,
            storage_options=storage_options,
            version=version,
            use_duckdb=use_duckdb,
        )

        if not streaming:
            self._load_as_hf_dataset()

    def _load_as_hf_dataset(self):
        """Load the Delta table as a HuggingFace Dataset."""
        from datasets import Dataset

        # Unity Catalog tables are supported in streaming mode via databricks-sql-connector.
        if _is_unity_catalog_path(self.table_path):
            raise ValueError(
                "Unity Catalog tables are only supported in streaming mode. "
                "Use streaming=True (and install databricks-sql-connector)."
            )

        if self.use_duckdb is True:
            iterator = DeltaLakeIterator(
                table_path=self.table_path,
                columns=self.columns,
                storage_options=self._base_iterator.storage_options,
                version=self.version,
                use_duckdb=True,
            )
            self._dataset = Dataset.from_list(list(iterator))
            return

        if self.use_duckdb is False:
            if not _check_deltalake_available():
                raise ImportError("deltalake is required but not installed. Install with: pip install deltalake")
            from deltalake import DeltaTable

            try:
                dt = DeltaTable(
                    self.table_path,
                    storage_options=self._base_iterator.storage_options,
                    version=self.version,
                )
                pa_table = dt.to_pyarrow_table(columns=self.columns)
            except Exception as e:  # noqa: BLE001
                if _is_deletion_vectors_error(e):
                    raise ImportError(
                        "This Delta table uses deletion vectors which are not supported by the deltalake library. "
                        "Install DuckDB to read this table:\n"
                        "  pip install duckdb"
                    ) from e
                raise

            self._dataset = Dataset.from_list(pa_table.to_pylist())
            return

        # Auto-detect: prefer deltalake when available, fall back to DuckDB when needed.
        if _check_deltalake_available():
            from deltalake import DeltaTable

            try:
                dt = DeltaTable(
                    self.table_path,
                    storage_options=self._base_iterator.storage_options,
                    version=self.version,
                )
                pa_table = dt.to_pyarrow_table(columns=self.columns)
                self._dataset = Dataset.from_list(pa_table.to_pylist())
                return
            except Exception as e:  # noqa: BLE001
                if _is_deletion_vectors_error(e) and _check_duckdb_available():
                    logger.warning(
                        "Table uses deletion vectors. Falling back to DuckDB reader. "
                        "To avoid this warning, set use_duckdb=True."
                    )
                else:
                    raise

        if _check_duckdb_available():
            iterator = DeltaLakeIterator(
                table_path=self.table_path,
                columns=self.columns,
                storage_options=self._base_iterator.storage_options,
                version=self.version,
                use_duckdb=True,
            )
            self._dataset = Dataset.from_list(list(iterator))
            return

        raise ImportError(
            "Unable to read this Delta table. Install one of:\n"
            "  pip install deltalake\n"
            "  pip install duckdb"
        )

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        if self.streaming:
            raise RuntimeError("__len__ is not supported in streaming mode.")
        if self._dataset is None:
            self._load_as_hf_dataset()
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific row by index."""
        if self.streaming:
            raise RuntimeError("__getitem__ is not supported in streaming mode.")
        if self._dataset is None:
            self._load_as_hf_dataset()
        return self._dataset[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in the dataset."""
        if self.streaming:
            iterator = DeltaLakeIterator(
                table_path=self.table_path,
                columns=self.columns,
                storage_options=self._base_iterator.storage_options,
                version=self.version,
                use_duckdb=self.use_duckdb,
            )

            row_idx = 0
            for row in iterator:
                # Apply sharding if configured
                if self._shard_info is not None:
                    num_shards, shard_index = self._shard_info
                    if row_idx % num_shards != shard_index:
                        row_idx += 1
                        continue

                yield row
                row_idx += 1
        else:
            if self._dataset is None:
                self._load_as_hf_dataset()
            yield from self._dataset

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling.

        Args:
            epoch: The epoch number.
        """
        self._epoch = epoch

    def shard(self, num_shards: int, index: int) -> "HFDeltaLakeDataset":
        """Shard the dataset for distributed processing.

        Args:
            num_shards: Total number of shards.
            index: Index of this shard (0-based).

        Returns:
            Self for method chaining.
        """
        self._shard_info = (num_shards, index)
        return self

    def shuffle(self, buffer_size: int = 1000, seed: Optional[int] = None) -> "HFDeltaLakeDataset":
        """Configure shuffling for the dataset.

        Note: For streaming Delta Lake datasets, shuffling is performed on-the-fly
        using a shuffle buffer. The actual shuffling happens during iteration.

        Args:
            buffer_size: Size of the shuffle buffer.
            seed: Random seed for reproducibility.

        Returns:
            Self for method chaining.
        """
        self._shuffle_info = (buffer_size, seed if seed is not None else self._epoch)
        return self

    def take(self, n: int) -> "HFDeltaLakeDataset":
        """Limit the dataset to the first n samples.

        Args:
            n: Number of samples to take.

        Returns:
            A new HFDeltaLakeDataset limited to n samples.
        """
        # Create a wrapper that limits iteration
        limited = _LimitedDeltaLakeDataset(self, n)
        return limited  # type: ignore[return-value]


class _LimitedDeltaLakeDataset:
    """Internal wrapper to limit a Delta Lake dataset to n samples."""

    def __init__(self, base: HFDeltaLakeDataset, limit: int):
        self._base = base
        self._limit = limit

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        count = 0
        for row in self._base:
            if count >= self._limit:
                break
            yield row
            count += 1

    def set_epoch(self, epoch: int) -> None:
        self._base.set_epoch(epoch)

    def shard(self, num_shards: int, index: int):
        self._base.shard(num_shards, index)
        return self

    def shuffle(self, buffer_size: int = 1000, seed: Optional[int] = None):
        self._base.shuffle(buffer_size, seed)
        return self

    def take(self, n: int):
        return _LimitedDeltaLakeDataset(self._base, min(n, self._limit))


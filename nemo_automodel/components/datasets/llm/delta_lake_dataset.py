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

**Supports tables with Deletion Vectors** (Databricks Runtime 15.4+) via Spark
(Databricks runtime) and optionally via Databricks SQL Connector for Unity Catalog
access outside of Spark.

Installation:
    ```bash
    # For basic Delta Lake support (without deletion vectors)
    pip install deltalake

    # For Databricks Unity Catalog access without Spark (optional)
    pip install databricks-sql-connector deltalake
    ```

Usage:
    Local Delta tables:

    ```python
    from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset import (
        ColumnMappedTextInstructionIterableDataset,
    )

    ds = ColumnMappedTextInstructionIterableDataset(
        path_or_dataset_id="delta:///path/to/delta_table",
        column_mapping={"question": "input", "answer": "output"},
        tokenizer=tokenizer,
    )
    ```

    Cloud storage (S3, Azure, GCS):

    ```python
    ds = ColumnMappedTextInstructionIterableDataset(
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
    ds = ColumnMappedTextInstructionIterableDataset(
        path_or_dataset_id="delta://catalog.schema.table",  # Unity Catalog format
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
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union
from nemo_automodel.components.datasets.reservoir_sampler import ReservoirSampler

logger = logging.getLogger(__name__)

# Lazy imports to avoid requiring deltalake/pyspark when not used
_DELTALAKE_AVAILABLE: Optional[bool] = None
_DATABRICKS_SQL_AVAILABLE: Optional[bool] = None
_PYSPARK_AVAILABLE: Optional[bool] = None


def _check_deltalake_available() -> bool:
    """Check if the deltalake package is available."""
    global _DELTALAKE_AVAILABLE
    if _DELTALAKE_AVAILABLE is None:
        try:
            import importlib

            importlib.import_module("deltalake")

            _DELTALAKE_AVAILABLE = True
        except ImportError:
            _DELTALAKE_AVAILABLE = False
    return _DELTALAKE_AVAILABLE


def _check_pyspark_available() -> bool:
    """Check if PySpark is available (used as a fallback on Databricks for deletion vectors)."""
    global _PYSPARK_AVAILABLE
    if _PYSPARK_AVAILABLE is None:
        try:
            import importlib

            importlib.import_module("pyspark")
            importlib.import_module("pyspark.sql")

            _PYSPARK_AVAILABLE = True
        except Exception:  # noqa: BLE001
            _PYSPARK_AVAILABLE = False
    return _PYSPARK_AVAILABLE


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
    """Check if any Delta Lake reader is available (deltalake, Spark, or databricks-sql)."""
    return _check_deltalake_available() or _check_pyspark_available() or _check_databricks_sql_available()


def _is_deletion_vectors_error(e: BaseException) -> bool:
    """Return True if *e* looks like the deltalake 'deletionVectors' unsupported-reader-features error."""
    msg = str(e).lower()
    return "deletionvectors" in msg or "deletion vectors" in msg


def _get_spark_session() -> Optional[Any]:
    """Get an active Spark session if available (Databricks notebooks/jobs).

    Returns:
        A SparkSession instance if available, else None.
    """
    if not _check_pyspark_available():
        return None
    try:
        import importlib

        SparkSession = importlib.import_module("pyspark.sql").SparkSession

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as e:  # noqa: BLE001
        logger.debug("SparkSession not available (%s): %s", type(e).__name__, e)
        return None


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

    # @akoumparouli: we allow on purpose abfss,s3,s3a,gs paths.

    # Check for abfss:// (Azure Blob Storage with hierarchical namespace)
    if path.startswith("abfss://"):
        return True

    # Check for s3:// or s3a:// paths
    if path.startswith(("s3://", "s3a://", "gs://")):
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


_UNITY_STORAGE_TABLE_PATH_RE = re.compile(
    r"__unitystorage/catalogs/(?P<catalog_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
    r"/tables/(?P<table_id>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def _parse_unity_storage_ids(path: str) -> Optional[Dict[str, str]]:
    """Parse Unity Catalog managed storage IDs from a __unitystorage path.

    Databricks Unity Catalog managed tables use internal cloud locations like:
      .../__unitystorage/catalogs/<catalog_uuid>/tables/<table_uuid>
    Direct path access to these locations is blocked on Databricks ("LOCATION_OVERLAP").
    """
    m = _UNITY_STORAGE_TABLE_PATH_RE.search(path)
    if not m:
        return None
    return {"catalog_id": m.group("catalog_id"), "table_id": m.group("table_id")}


def _quote_sql_ident(ident: str) -> str:
    """Quote an identifier for Spark SQL (handles embedded backticks)."""
    return f"`{ident.replace('`', '``')}`"


def _build_uc_table_fqn(catalog: str, schema: str, table: str) -> str:
    """Build a fully-qualified UC table name with safe quoting."""
    return f"{_quote_sql_ident(catalog)}.{_quote_sql_ident(schema)}.{_quote_sql_ident(table)}"


def _try_resolve_uc_table_from_system_tables(
    spark: Any,
    *,
    table_id: Optional[str] = None,
    storage_location: Optional[str] = None,
) -> Optional[str]:
    """Best-effort reverse lookup of a UC table name via Databricks system tables."""
    candidates: list[str] = []
    if table_id:
        safe_table_id = table_id.replace("'", "''")
        for col in ("table_id", "id", "table_uuid"):
            candidates.append(
                "SELECT TABLE_CATALOG AS table_catalog, TABLE_SCHEMA AS table_schema, TABLE_NAME AS table_name "
                f"FROM system.information_schema.tables WHERE {col} = '{safe_table_id}'"
            )
    if storage_location:
        safe_loc = storage_location.replace("'", "''")
        # Unity Catalog information schema uses STORAGE_PATH for table locations.
        for col in ("storage_path", "storage_location", "location"):
            candidates.append(
                "SELECT TABLE_CATALOG AS table_catalog, TABLE_SCHEMA AS table_schema, TABLE_NAME AS table_name "
                f"FROM system.information_schema.tables WHERE {col} = '{safe_loc}'"
            )

    for query in candidates:
        try:
            rows = spark.sql(query).take(1)
        except Exception:  # noqa: BLE001
            continue
        if not rows:
            continue
        row = rows[0]
        row_dict = row.asDict(recursive=True) if hasattr(row, "asDict") else dict(row)
        cat = row_dict.get("table_catalog")
        sch = row_dict.get("table_schema")
        name = row_dict.get("table_name")
        if cat and sch and name:
            return _build_uc_table_fqn(str(cat), str(sch), str(name))
    return None


def _resolve_uc_table_from_unity_storage_path(spark: Any, path: str) -> Optional[str]:
    """If *path* looks like UC managed storage, try to resolve to catalog.schema.table."""
    ids = _parse_unity_storage_ids(path)
    if ids is None:
        return None
    # Try a few storage path variants (some APIs include/omit trailing slash).
    loc_variants = {path, path.rstrip("/"), f"{path.rstrip('/')}/"}
    for loc in loc_variants:
        resolved = _try_resolve_uc_table_from_system_tables(spark, table_id=ids.get("table_id"), storage_location=loc)
        if resolved is not None:
            return resolved
    return None


def _is_location_overlap_error(e: BaseException) -> bool:
    """Return True if *e* looks like Databricks UC managed-storage overlap."""
    msg = str(e).lower()
    return "location_overlap" in msg or "checkpathaccess" in msg or "overlaps with managed storage" in msg


class DeltaLakeIterator:
    """Iterator that yields rows from a Delta Lake table.

    This class provides a streaming interface for Delta Lake tables,
    yielding rows as dictionaries one at a time to support memory-efficient
    iteration over large tables.

    Supports tables with deletion vectors (Databricks Runtime 15.4+) via Spark backend
    (recommended when running in Databricks notebooks/jobs).

    Args:
        table_path: Path to the Delta Lake table.
        columns: Optional list of column names to read. If None, reads all columns.
        storage_options: Optional storage options for cloud storage access.
        batch_size: Number of rows to read at a time (default: 1024).
        version: Optional version of the table to read.
        sql_query: Optional SQL query to read the table and/or create alias columns.
        shard_info: Optional sharding configuration ``(num_shards, shard_index)``.
            When provided, only rows where ``row_idx % num_shards == shard_index`` are yielded.
    """

    def __init__(
        self,
        table_path: str,
        columns: Optional[list] = None,
        storage_options: Optional[Dict[str, str]] = None,
        batch_size: int = 1024,
        version: Optional[int] = None,
        sql_query: Optional[str] = None,
        shard_info: Optional[tuple[int, int]] = None,
    ):
        if not _check_delta_reader_available():
            raise ImportError(
                "A Delta Lake reader is required. Install 'deltalake' (for tables without deletion vectors), "
                "or run in a Spark environment (Databricks), or install 'databricks-sql-connector' for "
                "Unity Catalog access without Spark."
            )

        self.table_path = _normalize_delta_path(table_path)
        self.columns = columns
        self.storage_options = storage_options or {}
        self.batch_size = batch_size
        self.version = version
        self.sql_query = sql_query
        self._shard_info: Optional[tuple[int, int]] = None

        if shard_info is not None:
            num_shards, shard_index = shard_info
            num_shards = int(num_shards)
            shard_index = int(shard_index)
            if num_shards <= 0:
                raise ValueError(f"num_shards must be > 0, got {num_shards}")
            if shard_index < 0 or shard_index >= num_shards:
                raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
            # Normalize the no-op shard case to None to keep checks cheap.
            self._shard_info = None if num_shards == 1 else (num_shards, shard_index)

        # SQL query execution requires an engine (Spark or Databricks SQL). The `deltalake`
        # reader does not support arbitrary SQL query execution on its own.
        if self.sql_query is not None and not (_check_pyspark_available() or _check_databricks_sql_available()):
            raise ImportError(
                "Delta Lake `sql_query` requires either Spark (pyspark / Databricks runtime) "
                "or `databricks-sql-connector`. A deltalake-only install cannot execute SQL queries."
            )

        # Add environment-based storage options
        self._add_env_storage_options()

    def _iter_all_rows(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all rows (no sharding)."""
        # Custom SQL query: prefer Spark when available, fall back to Databricks SQL connector.
        if self.sql_query is not None:
            if _get_spark_session() is not None:
                logger.info("Executing Delta SQL query via Spark.")
                yield from self._iter_with_spark()
                return
            if _check_databricks_sql_available():
                logger.info("Executing Delta SQL query via Databricks SQL connector.")
                yield from self._iter_with_databricks_sql()
                return
            raise ImportError(
                "Delta Lake `sql_query` requires either Spark (Databricks runtime / pyspark) "
                "or `databricks-sql-connector`."
            )

        # Check if this is a Unity Catalog table (catalog.schema.table format)
        if _is_unity_catalog_path(self.table_path):
            # Prefer Spark when available (Databricks notebooks/jobs) to avoid extra dependencies.
            if _get_spark_session() is not None:
                logger.info(f"Detected Unity Catalog table (Spark): {self.table_path}")
                yield from self._iter_with_spark()
                return
            if _check_databricks_sql_available():
                logger.info(f"Detected Unity Catalog table (Databricks SQL): {self.table_path}")
                yield from self._iter_with_databricks_sql()
                return
            raise ImportError(
                f"Unity Catalog table '{self.table_path}' requires either Spark (Databricks runtime) "
                "or databricks-sql-connector."
            )

        # File-based tables: prefer deltalake when possible, fall back to Spark (for deletion vectors).
        if _check_deltalake_available():
            yield from self._iter_with_deltalake()
            return
        if _get_spark_session() is not None:
            yield from self._iter_with_spark()
            return
        raise ImportError(
            "Unable to read this Delta table. Install 'deltalake' or run inside a Spark environment (Databricks)."
        )

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

    def _iter_with_spark(self) -> Iterator[Dict[str, Any]]:
        """Iterate using Spark (supports deletion vectors on Databricks).

        This backend requires a working SparkSession (e.g., Databricks notebooks/jobs).
        It is the recommended fallback for Delta tables that use deletion vectors.
        """
        spark = _get_spark_session()
        if spark is None:
            raise ImportError(
                "PySpark/SparkSession is required to read this Delta table (e.g., on Databricks). "
                "Install pyspark or run inside a Spark environment."
            )

        try:
            if self.sql_query is not None:
                query = str(self.sql_query).strip().rstrip(";").strip()
                df = spark.sql(query)
                if self.columns:
                    df = df.select(*self.columns)
            else:
                effective_path = self.table_path

                # Unity Catalog managed tables have internal storage locations under __unitystorage.
                # Databricks blocks direct reads of these locations (LOCATION_OVERLAP); resolve to a UC table name if possible.
                if not _is_unity_catalog_path(effective_path):
                    ids = _parse_unity_storage_ids(effective_path)
                    if ids is not None:
                        resolved = _resolve_uc_table_from_unity_storage_path(spark, effective_path)
                        if resolved is not None:
                            logger.info(
                                "Resolved Unity Catalog managed storage path to table: %s (from %s)",
                                resolved,
                                effective_path,
                            )
                            effective_path = resolved
                            self.table_path = resolved
                        else:
                            raise RuntimeError(
                                "This looks like a Unity Catalog managed-table storage location (contains '__unitystorage'). "
                                "Databricks blocks direct path access to UC managed storage. "
                                "Pass the Unity Catalog table name in `catalog.schema.table` format (or "
                                "`delta://catalog.schema.table`) instead of the underlying cloud path.\n"
                                f"Provided: {effective_path}\n"
                                f"Detected table_id: {ids.get('table_id')}"
                            )

                if _is_unity_catalog_path(effective_path):
                    if self.version is not None:
                        col_select = ", ".join([f"`{c}`" for c in self.columns]) if self.columns else "*"
                        df = spark.sql(f"SELECT {col_select} FROM {effective_path} VERSION AS OF {int(self.version)}")
                    else:
                        df = spark.table(effective_path)
                        if self.columns:
                            df = df.select(*self.columns)
                else:
                    reader = spark.read.format("delta")
                    if self.version is not None:
                        reader = reader.option("versionAsOf", str(int(self.version)))
                    df = reader.load(effective_path)
                    if self.columns:
                        df = df.select(*self.columns)
        except Exception as e:  # noqa: BLE001
            # Common Databricks failure when users pass the managed storage path for a UC table.
            if _is_location_overlap_error(e) and not _is_unity_catalog_path(self.table_path):
                raise RuntimeError(
                    "Spark refused to read this Delta location because it overlaps with Unity Catalog managed storage "
                    "(LOCATION_OVERLAP / CheckPathAccess). If this Delta table is managed by Unity Catalog, pass the "
                    "table name in `catalog.schema.table` format (or `delta://catalog.schema.table`) instead of the "
                    "underlying cloud path.\n"
                    f"Provided: {self.table_path}"
                ) from e
            raise RuntimeError(f"Failed to read Delta table via Spark: {e}") from e

        for row in df.toLocalIterator():
            yield row.asDict(recursive=True)

    def _iter_with_deltalake(self) -> Iterator[Dict[str, Any]]:
        """Iterate using deltalake library."""
        if self.sql_query is not None:
            raise RuntimeError(
                "Delta Lake `sql_query` is not supported with the `deltalake` backend. "
                "Run with Spark (Databricks / pyspark) or install `databricks-sql-connector`."
            )
        import importlib

        DeltaTable = importlib.import_module("deltalake").DeltaTable

        try:
            # Open the Delta table and get PyArrow dataset for efficient streaming
            dt = DeltaTable(self.table_path, storage_options=self.storage_options, version=self.version)
            pa_dataset = dt.to_pyarrow_dataset()
        except Exception as e:  # noqa: BLE001
            if _is_deletion_vectors_error(e):
                spark = _get_spark_session()
                if spark is not None:
                    logger.warning("Table uses deletion vectors. Falling back to Spark reader.")
                    yield from self._iter_with_spark()
                    return
                raise ImportError(
                    "This Delta table uses deletion vectors which are not supported by the deltalake library. "
                    "Run inside a Spark environment (e.g., Databricks), or disable deletion vectors on the table:\n"
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

        if self.sql_query is not None:
            base_query = str(self.sql_query).strip().rstrip(";").strip()
            # Optional projection on top of the provided query.
            if self.columns:
                query = f"SELECT {col_select} FROM ({base_query}) AS _q"
            else:
                query = base_query
        else:
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
        base_iter = self._iter_all_rows()
        if self._shard_info is None:
            num_shards = 1
            shard_index = 0
        else:
            num_shards, shard_index = self._shard_info

        for row_idx, row in enumerate(base_iter):
            if row_idx % num_shards == shard_index:
                yield row

    def shard(self, num_shards: int, index: int) -> "DeltaLakeIterator":
        """Shard the iterator for distributed processing.

        Args:
            num_shards: Total number of shards.
            index: Index of this shard (0-based).
        """
        assert num_shards > 0, "num_shards must be > 0"
        assert index >= 0 and index < num_shards, "index must be in [0, num_shards)"
        self._shard_info = (num_shards, index)
        return self

class DeltaLakeDataset:
    """HuggingFace datasets-compatible wrapper for Delta Lake tables.

    This class provides better integration with the HuggingFace datasets library,
    supporting features like sharding, shuffling, and epoch setting for distributed
    training scenarios.

    Args:
        table_path: Path to the Delta Lake table.
        columns: Optional list of column names to read.
        storage_options: Optional dict of storage options for cloud authentication.
        version: Optional specific version of the Delta table to read.
    """

    def __init__(
        self,
        table_path: str,
        columns: Optional[list] = None,
        storage_options: Optional[Dict[str, str]] = None,
        version: Optional[int] = None,
        sql_query: Optional[str] = None,
    ):
        self.table_path = _normalize_delta_path(table_path)
        self.columns = columns
        self.storage_options = storage_options or {}
        self.version = version
        self.sql_query = sql_query
        self._epoch: int = 0
        self._shard_info: Optional[tuple] = None  # (num_shards, shard_index)
        self._shuffle_info: Optional[tuple] = None  # (buffer_size, seed)

        # Eagerly create the internal iterator to validate the table
        self._data_iterator = DeltaLakeIterator(
            table_path=table_path,
            columns=columns,
            storage_options=storage_options,
            version=version,
            sql_query=sql_query,
        )

    def __len__(self) -> int:
        """Return the number of rows in the table."""
        raise RuntimeError("__len__ is not supported in streaming mode.")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a specific row by index."""
        raise RuntimeError("__getitem__ is not supported in streaming mode.")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in the dataset."""
        yield from self._data_iterator

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling.

        Args:
            epoch: The epoch number.
        """
        self._epoch = epoch

    def shard(self, num_shards: int, index: int) -> "DeltaLakeDataset":
        """Shard the dataset for distributed processing.

        Args:
            num_shards: Total number of shards.
            index: Index of this shard (0-based).

        Returns:
            Self for method chaining.
        """
        self._base_iterator.shard(num_shards, index)
        return self

    def shuffle(self, buffer_size: int = 1000, seed: Optional[int] = None) -> "DeltaLakeDataset":
        """Configure shuffling for the dataset.

        Note: For streaming Delta Lake datasets, shuffling is performed on-the-fly
        using a shuffle buffer. The actual shuffling happens during iteration.

        Args:
            buffer_size: Size of the shuffle buffer.
            seed: Random seed for reproducibility.

        Returns:
            Self for method chaining.
        """
        self._data_iterator = ReservoirSampler(self._data_iterator, buffer_size, seed)
        return self

    def take(self, n: int) -> "DeltaLakeDataset":
        """Limit the dataset to the first n samples.

        Args:
            n: Number of samples to take.

        Returns:
            A new DeltaLakeDataset limited to n samples.
        """
        # Create a wrapper that limits iteration
        limited = _LimitedDeltaLakeDataset(self, n)
        return limited  # type: ignore[return-value]

class _LimitedDeltaLakeDataset:
    """Internal wrapper to limit a Delta Lake dataset to n samples."""

    def __init__(self, base: DeltaLakeDataset, limit: int):
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

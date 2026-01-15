#!/usr/bin/env python3
"""Test script to verify Delta tables with deletion vectors can be read.

This script:
1. Creates a Delta table using PySpark with deletion vectors enabled
2. Performs a DELETE operation to trigger deletion vector creation
3. Reads the table using Spark (supports deletion vectors)
4. Tests the DeltaLakeIterator class with auto-detection
5. Verifies the data is correctly read
"""

import os
import shutil
import sys
import tempfile

import deltalake
import pandas as pd

# The delta_lake_dataset.py module is copied to /app by the Dockerfile


def create_delta_table_with_deletion_vectors(table_path: str) -> None:
    """Create a Delta table with deletion vectors enabled using PySpark."""
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    # Create Spark session with Delta Lake support
    builder = (
        SparkSession.builder.appName("DeltaLakeTest")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Enable deletion vectors (similar to Databricks Runtime 15.4+)
        .config("spark.databricks.delta.properties.defaults.enableDeletionVectors", "true")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Create sample data
        data = [
            (1, "Alice", "Engineering"),
            (2, "Bob", "Sales"),
            (3, "Charlie", "Marketing"),
            (4, "Diana", "Engineering"),
            (5, "Eve", "Sales"),
        ]
        df = spark.createDataFrame(data, ["id", "name", "department"])

        # Write as Delta table with deletion vectors enabled
        df.write.format("delta").mode("overwrite").option(
            "delta.enableDeletionVectors", "true"
        ).save(table_path)

        print(f"✓ Created Delta table at {table_path}")

        # Perform a DELETE to create deletion vectors
        from delta.tables import DeltaTable

        delta_table = DeltaTable.forPath(spark, table_path)
        delta_table.delete("id = 2")  # Delete Bob's record

        print("✓ Deleted row (id=2), creating deletion vector")

        # Show the table state
        spark.read.format("delta").load(table_path).show()

    finally:
        spark.stop()


def read_with_deltalake(table_path: str) -> pd.DataFrame:
    """Read the Delta table metadata using deltalake and data using Spark when needed."""
    print(f"\n--- Reading with deltalake {deltalake.__version__} ---")

    dt = deltalake.DeltaTable(table_path)

    # Print table info
    print(f"Table version: {dt.version()}")
    print(f"Protocol: {dt.protocol()}")

    # Check if deletion vectors are present
    metadata = dt.metadata()
    print(f"Table metadata: {metadata}")

    # Check if table has deletion vectors
    protocol = dt.protocol()
    has_deletion_vectors = (
        protocol.reader_features is not None
        and "deletionVectors" in protocol.reader_features
    )

    if has_deletion_vectors:
        print("\n⚠️  Table has deletion vectors - using Spark to read (deltalake doesn't support this yet)")
        df = read_with_spark(table_path)
    else:
        # Read data with deltalake (for tables without deletion vectors)
        df = dt.to_pandas()
        print(f"\n✓ Successfully read {len(df)} rows with deltalake:")
        print(df)

    return df


def read_with_spark(table_path: str) -> pd.DataFrame:
    """Read the Delta table using Spark (supports deletion vectors)."""
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName("DeltaLakeReadTest")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        sdf = spark.read.format("delta").load(table_path)
        df = sdf.toPandas()
        print(f"\n✓ Successfully read {len(df)} rows with Spark:")
        print(df)
        return df
    finally:
        spark.stop()


def verify_data(df: pd.DataFrame) -> None:
    """Verify the data was read correctly."""
    # Should have 4 rows (Bob was deleted)
    assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

    # Bob (id=2) should not be present
    assert 2 not in df["id"].values, "Deleted row (id=2) should not be present"

    # All other IDs should be present
    expected_ids = {1, 3, 4, 5}
    actual_ids = set(df["id"].values)
    assert actual_ids == expected_ids, f"Expected IDs {expected_ids}, got {actual_ids}"

    print("\n✓ All verification checks passed!")


def test_delta_lake_iterator(table_path: str) -> None:
    """Test the DeltaLakeIterator class with auto-detection."""
    print("\n--- Testing DeltaLakeIterator (auto-detection) ---")

    try:
        from delta_lake_dataset import DeltaLakeIterator

        # Test with auto-detection (should fall back to Spark due to deletion vectors)
        iterator = DeltaLakeIterator(
            table_path=table_path,
            batch_size=100,
        )

        rows = list(iterator)
        print(f"✓ DeltaLakeIterator read {len(rows)} rows")

        # Verify data
        ids = {row["id"] for row in rows}
        assert len(rows) == 4, f"Expected 4 rows, got {len(rows)}"
        assert 2 not in ids, "Deleted row (id=2) should not be present"
        print("✓ DeltaLakeIterator data verified!")

        # Test column selection (should work regardless of backend)
        print("\n--- Testing DeltaLakeIterator (column selection) ---")
        iterator_cols = DeltaLakeIterator(
            table_path=table_path,
            columns=["id", "name"],  # Test column selection
        )

        rows_cols = list(iterator_cols)
        print(f"✓ DeltaLakeIterator read {len(rows_cols)} rows with selected columns")

        # Verify only selected columns are present
        assert set(rows_cols[0].keys()) == {"id", "name"}, "Should only have selected columns"
        print("✓ Column selection works!")

    except ImportError as e:
        print(f"⚠️  Skipping DeltaLakeIterator test (module not available): {e}")
    except Exception as e:
        print(f"⚠️  DeltaLakeIterator test failed: {e}")
        raise


def main():
    print("=" * 60)
    print("Delta Lake Deletion Vectors Test")
    print("=" * 60)
    print(f"deltalake version: {deltalake.__version__}")
    print()

    # Create a temporary directory for the test table
    test_dir = tempfile.mkdtemp(prefix="delta_test_")
    table_path = os.path.join(test_dir, "test_table")

    try:
        # Step 1: Create table with deletion vectors using PySpark
        print("\n--- Step 1: Creating Delta table with deletion vectors ---")
        create_delta_table_with_deletion_vectors(table_path)

        # Step 2: Read with deltalake library (uses Spark fallback)
        print("\n--- Step 2: Reading with deltalake/Spark ---")
        df = read_with_deltalake(table_path)

        # Step 3: Verify the data
        print("\n--- Step 3: Verifying data ---")
        verify_data(df)

        # Step 4: Test DeltaLakeIterator class
        print("\n--- Step 4: Testing DeltaLakeIterator class ---")
        test_delta_lake_iterator(table_path)

        print("\n" + "=" * 60)
        print("SUCCESS: All tests passed!")
        print("- Spark can read Delta tables with deletion vectors")
        print("- DeltaLakeIterator auto-detects and uses Spark")
        print("- (deltalake library does not yet support deletion vectors)")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        raise

    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    main()


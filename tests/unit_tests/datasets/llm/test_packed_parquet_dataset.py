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

import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nemo_automodel.components.datasets.llm.packed_parquet_dataset import PackedParquetDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet(path, rows, row_group_size=None):
    """Write a list of row dicts to a Parquet file."""
    cols = {
        "input_ids": [r["input_ids"] for r in rows],
        "loss_mask": [r["loss_mask"] for r in rows],
        "seq_start_id": [r["seq_start_id"] for r in rows],
    }
    table = pa.table(
        {
            "input_ids": pa.array(cols["input_ids"], type=pa.list_(pa.int32())),
            "loss_mask": pa.array(cols["loss_mask"], type=pa.list_(pa.uint8())),
            "seq_start_id": pa.array(cols["seq_start_id"], type=pa.list_(pa.int32())),
        }
    )
    kwargs = {}
    if row_group_size is not None:
        kwargs["row_group_size"] = row_group_size
    pq.write_table(table, str(path), **kwargs)


# A simple pack: 8 tokens, 3 sequences at positions [0, 3, 6]
SIMPLE_ROW = {
    "input_ids": [10, 20, 30, 40, 50, 60, 70, 80],
    "loss_mask": [1, 1, 0, 1, 1, 1, 0, 0],
    "seq_start_id": [0, 3, 6],
}


@pytest.fixture
def single_file(tmp_path):
    """Single Parquet file with one row."""
    path = tmp_path / "data.idx.parquet"
    _write_parquet(path, [SIMPLE_ROW])
    return tmp_path, path


@pytest.fixture
def multi_row_file(tmp_path):
    """Single Parquet file with multiple rows."""
    rows = [
        SIMPLE_ROW,
        {
            "input_ids": [1, 2, 3, 4, 5],
            "loss_mask": [1, 1, 1, 1, 1],
            "seq_start_id": [0, 3],
        },
        {
            "input_ids": [100, 200],
            "loss_mask": [0, 1],
            "seq_start_id": [0],
        },
    ]
    path = tmp_path / "data.idx.parquet"
    _write_parquet(path, rows)
    return tmp_path, path, rows


@pytest.fixture
def multi_file_dir(tmp_path):
    """Directory with two Parquet shard files."""
    rows_a = [SIMPLE_ROW]
    rows_b = [
        {
            "input_ids": [1, 2, 3],
            "loss_mask": [1, 1, 1],
            "seq_start_id": [0],
        }
    ]
    _write_parquet(tmp_path / "shard_000.idx.parquet", rows_a)
    _write_parquet(tmp_path / "shard_001.idx.parquet", rows_b)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBasicRead:
    def test_len(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        assert len(ds) == 1

    def test_output_keys(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        sample = ds[0]
        assert set(sample.keys()) == {"input_ids", "labels", "position_ids", "seq_lens", "seq_lens_padded"}

    def test_input_ids_padded(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        sample = ds[0]
        ids = sample["input_ids"]
        assert ids[:8] == [10, 20, 30, 40, 50, 60, 70, 80]
        assert ids[8:] == [0, 0]  # padded with padding_idx=0
        assert len(ids) == 10

    def test_values_are_lists(self, single_file):
        """Values are lists (matching HF Dataset contract) for collater compatibility."""
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        sample = ds[0]
        for key in ("input_ids", "labels", "position_ids", "seq_lens", "seq_lens_padded"):
            assert isinstance(sample[key], list), f"{key} should be a list"


class TestLossMaskToLabels:
    def test_masked_positions_get_ignore_idx(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        labels = ds[0]["labels"]
        # loss_mask = [1, 1, 0, 1, 1, 1, 0, 0] → positions 2, 6, 7 → -100
        assert labels[0] == 10
        assert labels[1] == 20
        assert labels[2] == -100  # loss_mask=0
        assert labels[3] == 40
        assert labels[4] == 50
        assert labels[5] == 60
        assert labels[6] == -100  # loss_mask=0
        assert labels[7] == -100  # loss_mask=0
        # Padding also gets -100
        assert labels[8] == -100
        assert labels[9] == -100


class TestSeqStartIdToSeqLens:
    def test_seq_lens(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        seq_lens = ds[0]["seq_lens"]
        # seq_start_id = [0, 3, 6], len = 8 → seq_lens = [3, 3, 2]
        assert seq_lens == [3, 3, 2]

    def test_single_sequence(self, tmp_path):
        """Single sequence spanning entire pack."""
        row = {"input_ids": [1, 2, 3, 4], "loss_mask": [1, 1, 1, 1], "seq_start_id": [0]}
        path = tmp_path / "data.parquet"
        _write_parquet(path, [row])
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=6)
        assert ds[0]["seq_lens"] == [4]


class TestPositionIds:
    def test_reset_at_boundaries(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        pos = ds[0]["position_ids"]
        # seq_start_id = [0, 3, 6]: three sequences of lengths 3, 3, 2
        # Positions reset at each boundary:
        assert pos[:3] == [0, 1, 2]  # seq 1
        assert pos[3:6] == [0, 1, 2]  # seq 2
        assert pos[6:8] == [0, 1]  # seq 3


class TestPadding:
    def test_seq_lens_padded_sum_equals_pack_size(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        seq_lens_padded = ds[0]["seq_lens_padded"]
        assert sum(seq_lens_padded) == 10

    def test_seq_lens_padded_last_includes_padding(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        seq_lens = ds[0]["seq_lens"]
        seq_lens_padded = ds[0]["seq_lens_padded"]
        # Non-last elements are the same
        assert seq_lens_padded[:-1] == seq_lens[:-1]
        # Last element includes pack padding
        assert seq_lens_padded[-1] >= seq_lens[-1]

    def test_exact_fit_no_padding(self, tmp_path):
        """When row exactly fills pack_size, no padding needed."""
        row = {"input_ids": [1, 2, 3, 4, 5], "loss_mask": [1, 1, 1, 1, 1], "seq_start_id": [0, 3]}
        path = tmp_path / "data.parquet"
        _write_parquet(path, [row])
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=5)
        sample = ds[0]
        assert sample["input_ids"] == [1, 2, 3, 4, 5]
        assert sample["seq_lens"] == [3, 2]
        assert sample["seq_lens_padded"] == [3, 2]


class TestOversizedRow:
    def test_raises_value_error(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=5)
        with pytest.raises(ValueError, match="packed_sequence_size"):
            ds[0]


class TestMultiRow:
    def test_len(self, multi_row_file):
        _, path, rows = multi_row_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        assert len(ds) == 3

    def test_independent_getitem(self, multi_row_file):
        _, path, rows = multi_row_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        # Each row has different input_ids
        s0 = ds[0]["input_ids"][:8]
        s1 = ds[1]["input_ids"][:5]
        s2 = ds[2]["input_ids"][:2]
        assert s0 == [10, 20, 30, 40, 50, 60, 70, 80]
        assert s1 == [1, 2, 3, 4, 5]
        assert s2 == [100, 200]


class TestMultiFile:
    def test_directory_resolution(self, multi_file_dir):
        ds = PackedParquetDataset(data_path=str(multi_file_dir), packed_sequence_size=10)
        assert len(ds) == 2

    def test_cross_file_reads(self, multi_file_dir):
        ds = PackedParquetDataset(data_path=str(multi_file_dir), packed_sequence_size=10)
        s0 = ds[0]["input_ids"][:8]
        s1 = ds[1]["input_ids"][:3]
        assert s0 == [10, 20, 30, 40, 50, 60, 70, 80]
        assert s1 == [1, 2, 3]

    def test_glob_pattern(self, multi_file_dir):
        pattern = str(multi_file_dir / "shard_*.idx.parquet")
        ds = PackedParquetDataset(data_path=pattern, packed_sequence_size=10)
        assert len(ds) == 2


class TestErrorHandling:
    def test_no_files_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PackedParquetDataset(data_path=str(tmp_path / "nonexistent"), packed_sequence_size=10)

    def test_missing_columns_raises_value_error(self, tmp_path):
        table = pa.table({"input_ids": pa.array([[1, 2, 3]], type=pa.list_(pa.int32()))})
        path = tmp_path / "bad.parquet"
        pq.write_table(table, str(path))
        with pytest.raises(ValueError, match="missing required columns"):
            PackedParquetDataset(data_path=str(path), packed_sequence_size=10)

    def test_negative_indexing(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        last = ds[-1]
        first = ds[0]
        assert last["input_ids"] == first["input_ids"]
    def test_index_out_of_range(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        with pytest.raises(IndexError):
            ds[1]


class TestPickleSafety:
    def test_pickle_roundtrip(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        # Pickle and unpickle (simulates DataLoader worker spawn)
        ds2 = pickle.loads(pickle.dumps(ds))
        sample = ds2[0]
        assert sample["input_ids"][:8] == [10, 20, 30, 40, 50, 60, 70, 80]


class TestCPAwarePadding:
    def test_cp_size_2(self, tmp_path):
        """With cp_size=2, seq_lens_padded values should be divisible by 2*cp_size=4."""
        row = {
            "input_ids": [1, 2, 3, 4, 5, 6, 7],
            "loss_mask": [1, 1, 1, 1, 1, 1, 1],
            "seq_start_id": [0, 3],
        }
        path = tmp_path / "data.parquet"
        _write_parquet(path, [row])
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=16, cp_size=2)
        sample = ds[0]
        seq_lens = sample["seq_lens"]
        seq_lens_padded = sample["seq_lens_padded"]
        # Original: [3, 4]
        assert seq_lens == [3, 4]
        # _pad_pack applies CP rounding then adds pack-level padding to last seq.
        # CP divisibility_factor = 2 * cp_size = 4
        # Seq 1: 3 → 4 (rounded up)
        # Seq 2: 4 → 4 (already divisible) + 9 pack padding = 13
        # The last element absorbs pack padding so is NOT necessarily CP-divisible.
        assert seq_lens_padded[0] == 4
        # Sum is 4 + 13 = 17 (not pack_size) because CP rounding expands
        # individual seq_lens beyond the actual token count, and pack padding
        # is added on top of the last CP-rounded length.
        assert seq_lens_padded == [4, 13]


class TestIsPrePacked:
    def test_attribute_exists(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        assert ds.is_pre_packed is True

    def test_getattr_works(self, single_file):
        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        assert getattr(ds, "is_pre_packed", False) is True


class TestRowGroupCaching:
    def test_multiple_reads_same_rg(self, tmp_path):
        """Multiple rows in same row group should use cached read."""
        rows = [
            {"input_ids": [1, 2, 3], "loss_mask": [1, 1, 1], "seq_start_id": [0]},
            {"input_ids": [4, 5, 6], "loss_mask": [1, 1, 1], "seq_start_id": [0]},
            {"input_ids": [7, 8, 9], "loss_mask": [1, 1, 1], "seq_start_id": [0]},
        ]
        path = tmp_path / "data.parquet"
        _write_parquet(path, rows, row_group_size=3)
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=5)
        # Read all three rows
        assert ds[0]["input_ids"][:3] == [1, 2, 3]
        assert ds[1]["input_ids"][:3] == [4, 5, 6]
        assert ds[2]["input_ids"][:3] == [7, 8, 9]
        # Verify internal cache hit (same rg_flat_idx)
        assert ds._reader_state._cached_rg_flat_idx == 0


class TestCollaterCompatibility:
    def test_works_with_thd_collater(self, single_file):
        """End-to-end: DataLoader batch via packed_sequence_thd_collater."""
        from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

        _, path = single_file
        ds = PackedParquetDataset(data_path=str(path), packed_sequence_size=10)
        batch = packed_sequence_thd_collater([ds[0]])
        assert "input_ids" in batch
        assert "labels" in batch
        assert "position_ids" in batch
        assert "seq_lens" in batch
        assert "seq_lens_padded" in batch
        assert batch["qkv_format"] == "thd"
        assert batch["input_ids"].shape == (1, 10)


class TestListPath:
    def test_list_of_files(self, multi_file_dir):
        """Support list of file paths."""
        files = [
            str(multi_file_dir / "shard_000.idx.parquet"),
            str(multi_file_dir / "shard_001.idx.parquet"),
        ]
        ds = PackedParquetDataset(data_path=files, packed_sequence_size=10)
        assert len(ds) == 2

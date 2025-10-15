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
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import get_worker_info

from nemo_automodel.components.datasets.llm.nanogpt_dataset import (
    NanogptDataset,
    MAGIC,
    VERSION,
    load_bin_shard,
    _get_start_end_pos_single_file,
    _get_next_bos_position,
    _get_worker_id_and_total_workers
)


def _make_fake_shard(tmpdir: Path, tokens: np.ndarray) -> Path:
    """Create a binary shard with the required header and *tokens* (uint16)."""
    shard_path = tmpdir / "shard.bin"
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens)
    header[3] = 2
    with open(shard_path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())
    return shard_path


def test_nanogpt_dataset_iteration():
    # Create a tiny synthetic shard: BOS + 4 tokens → exactly one sample when seq_len=4
    bos = 50256
    toks = np.array([bos, 1, 2, 3, 4], dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmp:
        shard = _make_fake_shard(Path(tmp), toks)
        ds = NanogptDataset(str(shard), seq_len=4, align_to_bos=True, bos_token=bos)
        samples = []
        # Take only one sample since it's an infinite iterator
        for i, sample in enumerate(ds):
            if i >= 1:
                break
            samples.append(sample)

        assert len(samples) == 1
        sample = samples[0]
        assert isinstance(sample, dict)
        assert "input_ids" in sample
        assert "labels" in sample

        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Data should be lists of integers
        assert isinstance(input_ids, list)
        assert isinstance(labels, list)

        # Inputs/labels length must equal seq_len
        assert len(input_ids) == 4 and len(labels) == 4

        # Check shifting logic: labels[0] should equal input_ids[1] in original token stream
        assert labels[0] == 1 and labels[-1] == 4
        assert input_ids == [bos, 1, 2, 3]  # BOS + first 3 tokens
        assert labels == [1, 2, 3, 4]       # Next 4 tokens (shifted by 1)


def test_nanogpt_dataset_len():
    # Test that __len__ raises NotImplementedError
    bos = 50256
    toks = np.concatenate([[bos], np.arange(1, 9, dtype=np.uint16)])

    with tempfile.TemporaryDirectory() as tmp:
        shard = _make_fake_shard(Path(tmp), toks)
        ds = NanogptDataset(str(shard), seq_len=4, align_to_bos=False)
        try:
            len(ds)
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass  # Expected


def test_load_bin_shard():
    """Test the load_bin_shard function directly."""
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmp:
        shard_path = _make_fake_shard(Path(tmp), tokens)
        loaded_tokens = load_bin_shard(shard_path)

        # Should be a torch tensor
        assert isinstance(loaded_tokens, torch.Tensor)
        assert loaded_tokens.dtype == torch.uint16
        assert loaded_tokens.shape == (5,)

        # Values should match
        assert torch.equal(loaded_tokens, torch.from_numpy(tokens))


def test_nanogpt_dataset_getitem():
    """Test that __getitem__ raises NotImplementedError."""
    tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmp:
        shard_path = _make_fake_shard(Path(tmp), tokens)
        ds = NanogptDataset(str(shard_path), seq_len=4, align_to_bos=False)

        # Test that __getitem__ raises NotImplementedError
        try:
            ds[0]
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass  # Expected


def test_nanogpt_dataset_error_conditions():
    """Test various error conditions."""
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmp:
        shard_path = _make_fake_shard(Path(tmp), tokens)

        # Test that align_to_bos=True requires bos_token
        try:
            ds = NanogptDataset(str(shard_path), seq_len=4, align_to_bos=True, bos_token=None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "bos_token must be provided when align_to_bos is True" in str(e)


def test_nanogpt_dataset_no_files_error():
    """Test FileNotFoundError when no files match pattern."""
    try:
        ds = NanogptDataset("/nonexistent/path/*.bin", seq_len=4)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "No files matched pattern" in str(e)


def test_get_start_end_pos_single_file():
    """Test _get_start_end_pos_single_file function."""

    # Test evenly divisible case
    start, end = _get_start_end_pos_single_file(1000, 4, 0)  # worker 0 of 4
    assert start == 0
    assert end == 250

    start, end = _get_start_end_pos_single_file(1000, 4, 1)  # worker 1 of 4
    assert start == 250
    assert end == 500

    start, end = _get_start_end_pos_single_file(1000, 4, 2)  # worker 2 of 4
    assert start == 500
    assert end == 750

    # Last worker gets remaining tokens
    start, end = _get_start_end_pos_single_file(1000, 4, 3)  # worker 3 of 4 (last)
    assert start == 750
    assert end == 1000

    # Test non-evenly divisible case
    start, end = _get_start_end_pos_single_file(1003, 4, 0)  # worker 0 of 4
    assert start == 0
    assert end == 250  # 1003 // 4 = 250

    start, end = _get_start_end_pos_single_file(1003, 4, 3)  # last worker gets remainder
    assert start == 750
    assert end == 1003  # Gets all remaining tokens

    # Single worker case
    start, end = _get_start_end_pos_single_file(1000, 1, 0)
    assert start == 0
    assert end == 1000

    # Edge case: more workers than tokens
    start, end = _get_start_end_pos_single_file(2, 4, 0)
    assert start == 0
    assert end == 0  # 2 // 4 = 0

    start, end = _get_start_end_pos_single_file(2, 4, 3)  # last worker
    assert start == 0
    assert end == 2


def test_get_next_bos_position():
    """Test _get_next_bos_position function."""
    bos_token = 50256

    # Create test tokens: [1, BOS, 2, 3, BOS, 4, 5]
    tokens = torch.tensor([1, bos_token, 2, 3, bos_token, 4, 5], dtype=torch.int32)

    # Test with index file (bos_positions provided)
    bos_positions = np.array([1, 4])  # BOS at positions 1 and 4

    # Starting from position 0, should find BOS at position 1
    pos = _get_next_bos_position(tokens, bos_token, bos_positions, 0, len(tokens))
    assert pos == 1

    # Starting from position 2, should find BOS at position 4
    pos = _get_next_bos_position(tokens, bos_token, bos_positions, 2, len(tokens))
    assert pos == 4

    # Starting from position 5, should return max_pos (no more BOS tokens)
    pos = _get_next_bos_position(tokens, bos_token, bos_positions, 5, len(tokens))
    assert pos == len(tokens)

    # Test with max_pos limiting search
    pos = _get_next_bos_position(tokens, bos_token, bos_positions, 0, 3)
    assert pos == 1  # Found BOS at position 1, which is < 3

    pos = _get_next_bos_position(tokens, bos_token, bos_positions, 2, 4)
    assert pos == 4  # max_pos is 4, BOS at position 4 is not included (< max_pos), so returns max_pos

    # Test without index file (linear search)
    pos = _get_next_bos_position(tokens, bos_token, None, 0, len(tokens))
    assert pos == 1

    pos = _get_next_bos_position(tokens, bos_token, None, 2, len(tokens))
    assert pos == 4

    pos = _get_next_bos_position(tokens, bos_token, None, 5, len(tokens))
    assert pos == len(tokens)  # No BOS found after position 5

    # Test with tokens that don't contain BOS
    no_bos_tokens = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    pos = _get_next_bos_position(no_bos_tokens, bos_token, None, 0, len(no_bos_tokens))
    assert pos == len(no_bos_tokens)

    # Test empty bos_positions array
    empty_bos_positions = np.array([])
    pos = _get_next_bos_position(tokens, bos_token, empty_bos_positions, 0, len(tokens))
    assert pos == len(tokens)


def test_get_worker_id_and_total_workers():
    """Test _get_worker_id_and_total_workers function."""

    # Test case 1: No DataLoader worker, no distributed training
    mock_worker = Mock()
    mock_worker.num_workers = 1
    mock_worker.id = 0

    with patch('torch.distributed.is_initialized', return_value=False):
        worker_id, total_workers = _get_worker_id_and_total_workers(mock_worker)
        assert worker_id == 0
        assert total_workers == 1

    # Test case 2: DataLoader workers but no distributed training
    mock_worker.num_workers = 4
    mock_worker.id = 2

    with patch('torch.distributed.is_initialized', return_value=False):
        worker_id, total_workers = _get_worker_id_and_total_workers(mock_worker)
        assert worker_id == 2
        assert total_workers == 4

    # Test case 3: Distributed training but no DataLoader workers
    mock_worker.num_workers = 1
    mock_worker.id = 0

    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=3), \
         patch('torch.distributed.get_rank', return_value=1):
        worker_id, total_workers = _get_worker_id_and_total_workers(mock_worker)
        assert worker_id == 1  # rank 1 * 1 DL worker + 0 DL worker id
        assert total_workers == 3  # 3 ranks * 1 DL worker each

    # Test case 4: Both distributed training and DataLoader workers
    mock_worker.num_workers = 2
    mock_worker.id = 1

    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=3), \
         patch('torch.distributed.get_rank', return_value=2):
        worker_id, total_workers = _get_worker_id_and_total_workers(mock_worker)
        assert worker_id == 5  # rank 2 * 2 DL workers + 1 DL worker id = 5
        assert total_workers == 6  # 3 ranks * 2 DL workers each = 6

    # Test case 5: No worker info (single-threaded DataLoader)
    with patch('torch.distributed.is_initialized', return_value=False):
        worker_id, total_workers = _get_worker_id_and_total_workers(None)
        assert worker_id == 0
        assert total_workers == 1

    # Test case 6: Exception handling when import fails
    mock_worker.num_workers = 2
    mock_worker.id = 1

    # Mock the import to raise an exception
    def mock_import(name, *args, **kwargs):
        if name == 'torch.distributed':
            raise ImportError("No module named 'torch.distributed'")
        return __builtins__['__import__'](name, *args, **kwargs)

    with patch('builtins.__import__', side_effect=mock_import):
        worker_id, total_workers = _get_worker_id_and_total_workers(mock_worker)
        assert worker_id == 1  # Should fall back to DL worker id
        assert total_workers == 2  # Should fall back to DL num workers

    # Test case 7: Distributed available but not initialized
    mock_worker.num_workers = 3
    mock_worker.id = 2

    with patch('torch.distributed.is_initialized', return_value=False):
        worker_id, total_workers = _get_worker_id_and_total_workers(mock_worker)
        assert worker_id == 2
        assert total_workers == 3

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

import numpy as np
import torch

from nemo_automodel.components.datasets.llm.nanogpt_dataset import NanogptDataset, MAGIC, VERSION, load_bin_shard


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
        ds = NanogptDataset(str(shard), seq_len=4, align_to_bos=True, infinite=False, bos_token=bos)
        samples = list(iter(ds))
        # Should yield exactly 1 sample dictionary
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
    # 9 tokens total → with seq_len=4 ⇒ floor((9-1)/4)=2 samples
    bos = 50256
    toks = np.concatenate([[bos], np.arange(1, 9, dtype=np.uint16)])

    with tempfile.TemporaryDirectory() as tmp:
        shard = _make_fake_shard(Path(tmp), toks)
        ds = NanogptDataset(str(shard), seq_len=4, align_to_bos=False, infinite=False, drop_last=True)
        assert len(ds) == 2


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
    """Test random access via __getitem__."""
    tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmp:
        shard_path = _make_fake_shard(Path(tmp), tokens)
        ds = NanogptDataset(str(shard_path), seq_len=4, align_to_bos=False, infinite=False, drop_last=True)

        # Should have 2 samples: (9-1)//4 = 2
        assert len(ds) == 2

        # Test first sample
        inputs, labels = ds[0]
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert inputs.dtype == torch.int32
        assert labels.dtype == torch.int64
        assert len(inputs) == 4
        assert len(labels) == 4
        assert torch.equal(inputs, torch.tensor([1, 2, 3, 4], dtype=torch.int32))
        assert torch.equal(labels, torch.tensor([2, 3, 4, 5], dtype=torch.int64))

        # Test second sample
        inputs, labels = ds[1]
        assert torch.equal(inputs, torch.tensor([5, 6, 7, 8], dtype=torch.int32))
        assert torch.equal(labels, torch.tensor([6, 7, 8, 9], dtype=torch.int64))


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

        # Test that __len__ raises TypeError when infinite=True
        ds = NanogptDataset(str(shard_path), seq_len=4, infinite=True)
        try:
            len(ds)
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert "`__len__` is undefined when `infinite=True`" in str(e)

        # Test that __getitem__ raises TypeError when infinite=True
        try:
            ds[0]
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert "Random access is not supported when `infinite=True`" in str(e)

        # Test that __getitem__ raises NotImplementedError when align_to_bos=True
        ds = NanogptDataset(str(shard_path), seq_len=4, align_to_bos=True, bos_token=50256, infinite=False)
        try:
            ds[0]
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError as e:
            assert "__getitem__ with align_to_bos=True is not implemented" in str(e)


def test_nanogpt_dataset_no_files_error():
    """Test FileNotFoundError when no files match pattern."""
    try:
        ds = NanogptDataset("/nonexistent/path/*.bin", seq_len=4)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "No files matched pattern" in str(e)

import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from nemo_automodel.components.datasets.llm.bin_dataset import BinTokenDataset, MAGIC, VERSION


def _make_fake_shard(tmpdir: Path, tokens: np.ndarray) -> Path:
    """Create a binary shard with the required header and *tokens* (uint16)."""
    shard_path = tmpdir / "shard.bin"
    header = np.zeros(256, dtype=np.int32)
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens)
    with open(shard_path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())
    return shard_path


def test_bintoken_dataset_iteration():
    # Create a tiny synthetic shard: BOS + 4 tokens → exactly one sample when seq_len=4
    bos = 50256
    toks = np.array([bos, 1, 2, 3, 4], dtype=np.uint16)

    with tempfile.TemporaryDirectory() as tmp:
        shard = _make_fake_shard(Path(tmp), toks)
        ds = BinTokenDataset(str(shard), seq_len=4, align_to_bos=True, infinite=False)
        samples = list(iter(ds))
        # Should yield exactly 1 (inputs, targets) pair
        assert len(samples) == 1
        inp, tgt = samples[0]
        assert inp.dtype == torch.int32
        assert tgt.dtype == torch.int64
        # Inputs/tgt length must equal seq_len
        assert inp.shape[0] == 4 and tgt.shape[0] == 4
        # Check shifting logic: target[0] should equal input[1] in original token stream
        assert tgt[0].item() == 1 and tgt[-1].item() == 4


def test_bintoken_dataset_len():
    # 9 tokens total → with seq_len=4 ⇒ floor((9-1)/4)=2 samples
    bos = 50256
    toks = np.concatenate([[bos], np.arange(1, 9, dtype=np.uint16)])

    with tempfile.TemporaryDirectory() as tmp:
        shard = _make_fake_shard(Path(tmp), toks)
        ds = BinTokenDataset(str(shard), seq_len=4, align_to_bos=False, infinite=False, drop_last=True)
        assert len(ds) == 2 
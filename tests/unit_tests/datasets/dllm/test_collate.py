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

"""Tests for DLLMCollator (two-stage block-aligned padding)."""

from nemo_automodel.components.datasets.dllm.collate import DLLMCollator


def _make_sample(length, pad_token_id=0):
    """Create a minimal unshifted sample dict."""
    return {
        "input_ids": list(range(1, length + 1)),
        "loss_mask": [1] * length,
        "attention_mask": [1] * length,
    }


# ---------------------------------------------------------------------------
# Basic collation
# ---------------------------------------------------------------------------


class TestDLLMCollatorBasic:
    def test_output_keys(self):
        collator = DLLMCollator()
        batch = [_make_sample(10)]
        out = collator(batch)
        assert set(out.keys()) == {"input_ids", "loss_mask", "attention_mask", "input_lengths"}

    def test_output_shapes(self):
        collator = DLLMCollator()
        batch = [_make_sample(10), _make_sample(15)]
        out = collator(batch)
        B = 2
        L = out["input_ids"].shape[1]
        assert out["input_ids"].shape == (B, L)
        assert out["loss_mask"].shape == (B, L)
        assert out["attention_mask"].shape == (B, L)
        assert out["input_lengths"].shape == (B,)

    def test_content_preserved(self):
        collator = DLLMCollator()
        batch = [_make_sample(5)]
        out = collator(batch)
        assert out["input_ids"][0, :5].tolist() == [1, 2, 3, 4, 5]

    def test_input_lengths_correct(self):
        collator = DLLMCollator()
        batch = [_make_sample(7), _make_sample(12)]
        out = collator(batch)
        assert out["input_lengths"].tolist() == [7, 12]

    def test_pads_to_longest(self):
        collator = DLLMCollator()
        batch = [_make_sample(5), _make_sample(10)]
        out = collator(batch)
        assert out["input_ids"].shape[1] >= 10


# ---------------------------------------------------------------------------
# Block-aligned padding
# ---------------------------------------------------------------------------


class TestDLLMCollatorBlockAlign:
    def test_output_length_divisible_by_block_size(self):
        collator = DLLMCollator(block_size=8)
        batch = [_make_sample(10)]  # 10 -> aligned to 16
        out = collator(batch)
        assert out["input_ids"].shape[1] % 8 == 0

    def test_block_padding_uses_eos_token(self):
        collator = DLLMCollator(block_size=8, eos_token_id=2, pad_token_id=0)
        batch = [_make_sample(5)]  # content=5, block-aligned=8
        out = collator(batch)
        # Positions 5-7 should be eos (block padding)
        assert (out["input_ids"][0, 5:8] == 2).all()

    def test_block_padding_loss_mask_is_one(self):
        """Block padding (eos fill) should have loss_mask=1."""
        collator = DLLMCollator(block_size=8, eos_token_id=2)
        batch = [_make_sample(5)]
        out = collator(batch)
        # Content positions 0-4: loss_mask=1, block pad 5-7: loss_mask=1
        assert (out["loss_mask"][0, :8] == 1).all()

    def test_global_padding_loss_mask_is_zero(self):
        """Global padding should have loss_mask=0."""
        collator = DLLMCollator(block_size=8, pad_token_id=0)
        batch = [_make_sample(5), _make_sample(14)]
        out = collator(batch)
        # For the short sample (len=5, block-aligned=8), positions 8+ are global padding
        assert (out["loss_mask"][0, 8:] == 0).all()

    def test_attention_mask_zero_for_all_padding(self):
        """attention_mask should be 0 for both block and global padding."""
        collator = DLLMCollator(block_size=8)
        batch = [_make_sample(5)]
        out = collator(batch)
        # Content: 5 positions with attn=1, rest should be 0
        assert out["attention_mask"][0, :5].sum() == 5
        assert out["attention_mask"][0, 5:].sum() == 0


# ---------------------------------------------------------------------------
# pad_seq_len_divisible
# ---------------------------------------------------------------------------


class TestDLLMCollatorSeqLenDivisible:
    def test_divisible_alignment(self):
        collator = DLLMCollator(block_size=32, pad_seq_len_divisible=1024)
        batch = [_make_sample(100)]
        out = collator(batch)
        L = out["input_ids"].shape[1]
        assert L % 1024 == 0

    def test_lcm_alignment(self):
        """Output should be divisible by lcm(block_size, pad_seq_len_divisible)."""
        import math

        collator = DLLMCollator(block_size=32, pad_seq_len_divisible=48)
        batch = [_make_sample(10)]
        out = collator(batch)
        L = out["input_ids"].shape[1]
        assert L % math.lcm(32, 48) == 0


# ---------------------------------------------------------------------------
# max_seq_len
# ---------------------------------------------------------------------------


class TestDLLMCollatorMaxSeqLen:
    def test_caps_output_length(self):
        collator = DLLMCollator(max_seq_len=64)
        batch = [_make_sample(100)]
        out = collator(batch)
        assert out["input_ids"].shape[1] <= 64

    def test_caps_with_block_alignment(self):
        collator = DLLMCollator(block_size=32, max_seq_len=64)
        batch = [_make_sample(50)]
        out = collator(batch)
        assert out["input_ids"].shape[1] <= 64

    def test_content_truncated_to_max_seq_len(self):
        collator = DLLMCollator(max_seq_len=10)
        batch = [_make_sample(20)]
        out = collator(batch)
        # Only first 10 tokens should be copied
        assert out["input_ids"][0, :10].tolist() == list(range(1, 11))


# ---------------------------------------------------------------------------
# supervise_padding
# ---------------------------------------------------------------------------


class TestDLLMCollatorSupervisePadding:
    def test_global_padding_loss_mask_zero_by_default(self):
        """Default: global padding has loss_mask=0."""
        collator = DLLMCollator(pad_token_id=0)
        batch = [_make_sample(5), _make_sample(10)]
        out = collator(batch)
        # Short sample: positions beyond content are global padding
        assert (out["loss_mask"][0, 5:] == 0).all()

    def test_global_padding_loss_mask_one_when_enabled(self):
        """With supervise_padding=True, global padding has loss_mask=1."""
        collator = DLLMCollator(pad_token_id=0, supervise_padding=True)
        batch = [_make_sample(5), _make_sample(10)]
        out = collator(batch)
        # Short sample: global padding positions should now be 1
        assert (out["loss_mask"][0, 5:] == 1).all()

    def test_content_loss_mask_unaffected(self):
        """supervise_padding should not change loss_mask for content positions."""
        collator_off = DLLMCollator(pad_token_id=0, supervise_padding=False)
        collator_on = DLLMCollator(pad_token_id=0, supervise_padding=True)
        batch = [_make_sample(8)]
        out_off = collator_off(batch)
        out_on = collator_on(batch)
        # Content region should be identical
        assert (out_off["loss_mask"][0, :8] == out_on["loss_mask"][0, :8]).all()

    def test_with_block_padding(self):
        """Block padding always has loss_mask=1; supervise_padding only affects global padding."""
        collator = DLLMCollator(block_size=8, eos_token_id=2, pad_token_id=0, supervise_padding=True)
        batch = [_make_sample(5), _make_sample(14)]
        out = collator(batch)
        L = out["input_ids"].shape[1]
        # For short sample (len=5, block-aligned=8):
        # block padding (5-7): loss_mask=1 (always)
        assert (out["loss_mask"][0, 5:8] == 1).all()
        # global padding (8+): loss_mask=1 (because supervise_padding=True)
        if L > 8:
            assert (out["loss_mask"][0, 8:] == 1).all()

    def test_no_padding_no_difference(self):
        """When all samples are the same length, supervise_padding has no effect."""
        collator = DLLMCollator(pad_token_id=0, supervise_padding=True)
        batch = [_make_sample(10), _make_sample(10)]
        out = collator(batch)
        # No padding needed, all positions are content with loss_mask=1
        assert (out["loss_mask"] == 1).all()

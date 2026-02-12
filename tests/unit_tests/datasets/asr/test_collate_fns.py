# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_automodel.components.datasets.asr.collate_fns import (
    COLLATE_FNS,
    parakeet_collate_fn,
    shift_tokens_right,
    whisper_collate_fn,
)


class TestShiftTokensRight:
    """Test the shift_tokens_right helper function behaviors."""

    def test_shift_tokens_right_shifts_content(self):
        """Verify tokens are shifted right by one position."""
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        pad_token_id = 0
        decoder_start_token_id = 50

        result = shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id)

        # First token should be decoder_start_token_id
        assert (result[:, 0] == decoder_start_token_id).all()

        # Rest should be shifted from input (output[1:] == input[:-1])
        assert torch.equal(result[:, 1:], input_ids[:, :-1])

    def test_shift_tokens_right_replaces_minus_100_with_pad_token(self):
        """Verify -100 labels are replaced with pad_token_id (HuggingFace convention)."""
        input_ids = torch.tensor([[1, 2, -100, 4], [5, -100, 7, -100]])
        pad_token_id = 0
        decoder_start_token_id = 50

        result = shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id)

        # No -100 values should remain in output
        assert (result != -100).all()

        # Verify specific positions that had -100 now have pad_token_id
        # After shifting: input[i] moves to result[i+1]
        # input[0, 2] = -100 → result[0, 3] should be pad_token_id
        # input[1, 1] = -100 → result[1, 2] should be pad_token_id
        # (Note: input[1, 3] = -100 gets dropped during right shift)
        assert result[0, 3] == pad_token_id  # Was input[0, 2] = -100
        assert result[1, 2] == pad_token_id  # Was input[1, 1] = -100

    def test_shift_tokens_right_with_all_valid_tokens(self):
        """Verify shifting works correctly when no -100 labels present."""
        input_ids = torch.tensor([[10, 20, 30], [40, 50, 60]])
        pad_token_id = 0
        decoder_start_token_id = 100

        result = shift_tokens_right(input_ids, pad_token_id, decoder_start_token_id)

        # Expected: [[100, 10, 20], [100, 40, 50]]
        expected = torch.tensor([[100, 10, 20], [100, 40, 50]])
        assert torch.equal(result, expected)


class TestWhisperCollateFn:
    """Test whisper_collate_fn behavior."""

    def test_whisper_collate_fn_produces_correct_output_structure(
        self, dummy_whisper_processor, dummy_audio_samples
    ):
        """Verify whisper_collate_fn returns dict with required keys."""
        result = whisper_collate_fn(dummy_audio_samples, dummy_whisper_processor)

        # Output must have these three keys
        assert "input_features" in result
        assert "decoder_input_ids" in result
        assert "labels" in result

        # All values should be tensors
        assert isinstance(result["input_features"], torch.Tensor)
        assert isinstance(result["decoder_input_ids"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)

    def test_whisper_collate_fn_produces_correct_shapes(self, dummy_whisper_processor, dummy_audio_samples):
        """Verify output tensor shapes are correct."""
        batch_size = len(dummy_audio_samples)
        result = whisper_collate_fn(dummy_audio_samples, dummy_whisper_processor)

        # input_features should be (batch_size, 80, 3000)
        # 80 mel channels, 3000 time steps for Whisper
        assert result["input_features"].shape[0] == batch_size
        assert result["input_features"].shape[1] == 80
        assert result["input_features"].shape[2] == 3000

        # decoder_input_ids and labels should have same shape
        assert result["decoder_input_ids"].shape == result["labels"].shape

        # Text dimension should be same across batch (due to padding)
        assert result["labels"].shape[0] == batch_size

    def test_whisper_collate_fn_shifts_decoder_inputs_from_labels(
        self, dummy_whisper_processor, dummy_audio_samples
    ):
        """Verify decoder_input_ids are shifted right from labels (teacher forcing)."""
        result = whisper_collate_fn(dummy_audio_samples, dummy_whisper_processor)

        decoder_start_token_id = dummy_whisper_processor.tokenizer.convert_tokens_to_ids(
            "<|startoftranscript|>"
        )

        # First token of decoder_input_ids should be decoder_start_token_id
        assert (result["decoder_input_ids"][:, 0] == decoder_start_token_id).all()

        # Verify shifting: decoder_input_ids[i] should equal labels[i-1] for non-masked positions
        labels = result["labels"]
        decoder_input_ids = result["decoder_input_ids"]
        pad_token_id = dummy_whisper_processor.tokenizer.pad_token_id

        # Check first few positions where labels are not padding
        for batch_idx in range(labels.shape[0]):
            for pos in range(1, min(5, labels.shape[1])):  # Check first 5 positions
                label_prev = labels[batch_idx, pos - 1]
                # For non-padding labels, verify they were shifted correctly
                if label_prev != pad_token_id:
                    assert (
                        decoder_input_ids[batch_idx, pos] == label_prev
                    ), f"Batch {batch_idx}, position {pos}: expected {label_prev}, got {decoder_input_ids[batch_idx, pos]}"

    def test_whisper_collate_fn_handles_different_text_fields(
        self, dummy_whisper_processor, dummy_audio_samples_with_sentence_field
    ):
        """Verify collation works with both 'text' and 'sentence' fields."""
        # Test with 'sentence' field (Common Voice format)
        result_sentence = whisper_collate_fn(
            dummy_audio_samples_with_sentence_field, dummy_whisper_processor
        )

        # Should produce same structure regardless of field name
        assert "input_features" in result_sentence
        assert "decoder_input_ids" in result_sentence
        assert "labels" in result_sentence

        # Shapes should still be correct
        batch_size = len(dummy_audio_samples_with_sentence_field)
        assert result_sentence["input_features"].shape[0] == batch_size
        assert result_sentence["labels"].shape[0] == batch_size

    def test_whisper_collate_fn_respects_max_length(self, dummy_whisper_processor, dummy_audio_samples):
        """Verify max_length parameter truncates text sequences."""
        # Test with small max_length
        max_length = 10
        result = whisper_collate_fn(dummy_audio_samples, dummy_whisper_processor, max_length=max_length)

        # labels sequence length should not exceed max_length
        assert result["labels"].shape[1] <= max_length

    def test_whisper_collate_fn_with_single_example(self, dummy_whisper_processor):
        """Verify collation works with batch size of 1."""
        single_example = [
            {
                "audio": {"array": torch.randn(16000).numpy(), "sampling_rate": 16000},
                "text": "hello",
            }
        ]

        result = whisper_collate_fn(single_example, dummy_whisper_processor)

        # Should still have batch dimension
        assert result["input_features"].shape[0] == 1
        assert result["labels"].shape[0] == 1


class TestParakeetCollateFn:
    """Test parakeet_collate_fn behavior."""

    def test_parakeet_collate_fn_produces_correct_output_structure(
        self, dummy_parakeet_processor, dummy_audio_samples
    ):
        """Verify parakeet_collate_fn returns dict with required keys."""
        result = parakeet_collate_fn(dummy_audio_samples, dummy_parakeet_processor)

        # Output must have these keys (CTC format)
        assert "input_features" in result
        assert "attention_mask" in result
        assert "labels" in result

        # All values should be tensors
        assert isinstance(result["input_features"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
        assert isinstance(result["labels"], torch.Tensor)

    def test_parakeet_collate_fn_produces_correct_shapes(
        self, dummy_parakeet_processor, dummy_audio_samples
    ):
        """Verify output tensor shapes are correct for CTC models."""
        batch_size = len(dummy_audio_samples)
        result = parakeet_collate_fn(dummy_audio_samples, dummy_parakeet_processor)

        # input_features should have batch_size as first dimension
        assert result["input_features"].shape[0] == batch_size

        # attention_mask should match time dimension of input_features
        # Shape: (batch, time)
        assert result["attention_mask"].shape[0] == batch_size
        assert result["attention_mask"].shape[1] == result["input_features"].shape[2]

        # labels should have batch dimension
        assert result["labels"].shape[0] == batch_size

    def test_parakeet_collate_fn_attention_mask_is_binary(
        self, dummy_parakeet_processor, dummy_audio_samples
    ):
        """Verify attention mask contains only 0s and 1s."""
        result = parakeet_collate_fn(dummy_audio_samples, dummy_parakeet_processor)

        # attention_mask should be binary (0 for padding, 1 for valid)
        unique_values = torch.unique(result["attention_mask"])
        assert all(val in [0, 1] for val in unique_values)

    def test_parakeet_collate_fn_handles_different_text_fields(
        self, dummy_parakeet_processor, dummy_audio_samples_with_sentence_field
    ):
        """Verify collation works with both 'text' and 'sentence' fields."""
        result = parakeet_collate_fn(dummy_audio_samples_with_sentence_field, dummy_parakeet_processor)

        # Should produce same structure regardless of field name
        assert "input_features" in result
        assert "attention_mask" in result
        assert "labels" in result

        batch_size = len(dummy_audio_samples_with_sentence_field)
        assert result["input_features"].shape[0] == batch_size

    def test_parakeet_collate_fn_with_single_example(self, dummy_parakeet_processor):
        """Verify collation works with batch size of 1."""
        single_example = [
            {
                "audio": {"array": torch.randn(16000).numpy(), "sampling_rate": 16000},
                "text": "test",
            }
        ]

        result = parakeet_collate_fn(single_example, dummy_parakeet_processor)

        # Should still have batch dimension
        assert result["input_features"].shape[0] == 1
        assert result["attention_mask"].shape[0] == 1
        assert result["labels"].shape[0] == 1


class TestCollateFnsDispatchTable:
    """Test the COLLATE_FNS dispatch table."""

    def test_collate_fns_contains_required_keys(self):
        """Verify dispatch table has expected processor type mappings."""
        # Should have mappings for both processor types
        assert "WhisperProcessor" in COLLATE_FNS
        assert "ParakeetProcessor" in COLLATE_FNS
        assert "default" in COLLATE_FNS

    def test_collate_fns_maps_to_correct_functions(self):
        """Verify dispatch table maps to correct collation functions."""
        # WhisperProcessor should map to whisper_collate_fn
        assert COLLATE_FNS["WhisperProcessor"] == whisper_collate_fn

        # ParakeetProcessor should map to parakeet_collate_fn
        assert COLLATE_FNS["ParakeetProcessor"] == parakeet_collate_fn

        # default should map to whisper_collate_fn
        assert COLLATE_FNS["default"] == whisper_collate_fn

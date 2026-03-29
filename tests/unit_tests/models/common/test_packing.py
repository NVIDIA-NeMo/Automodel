# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_automodel.components.models.common.packing."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.models.common.packing import (
    _passthrough_create_causal_mask,
    configure_packing,
    get_attn_implementation,
    get_seqlens_in_batch,
    get_unpad_data,
)


# ---------------------------------------------------------------------------
# get_seqlens_in_batch
# ---------------------------------------------------------------------------


class TestGetSeqlensInBatch:
    def test_single_sequence(self):
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = get_seqlens_in_batch(mask)
        assert result.tolist() == [3]

    def test_packed_sequences(self):
        mask = torch.tensor([[1, 1, 2, 2, 2, 0]])
        result = get_seqlens_in_batch(mask)
        assert sorted(result.tolist()) == [2, 3]

    def test_no_padding(self):
        mask = torch.tensor([[1, 1, 1]])
        result = get_seqlens_in_batch(mask)
        assert result.tolist() == [3]


# ---------------------------------------------------------------------------
# get_unpad_data
# ---------------------------------------------------------------------------


class TestGetUnpadData:
    def test_basic(self):
        mask = torch.tensor([[1, 1, 0]])
        indices, cu_seqlens, max_seqlen = get_unpad_data(mask)
        assert max_seqlen == 2
        assert cu_seqlens.tolist() == [0, 2]

    def test_packed(self):
        mask = torch.tensor([[1, 1, 2, 2, 0]])
        indices, cu_seqlens, max_seqlen = get_unpad_data(mask)
        assert max_seqlen == 2
        assert indices.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# _passthrough_create_causal_mask
# ---------------------------------------------------------------------------


class TestPassthroughCreateCausalMask:
    def test_passthrough_4d_mask(self):
        """4D masks (already block-causal from sdpa collater) are returned as-is."""
        mask = torch.ones(2, 1, 8, 8)
        result = _passthrough_create_causal_mask(attention_mask=mask)
        assert result is mask

    def test_passthrough_indexed_packed_mask(self):
        """Indexed masks with values > 1 (packed sequences) are returned as-is."""
        mask = torch.tensor([[1, 1, 2, 2, 0]])
        result = _passthrough_create_causal_mask(attention_mask=mask)
        assert result is mask

    def test_delegates_to_original_for_normal_mask(self):
        """Normal 2D mask (max<=1) delegates to the original HF create_causal_mask."""
        from unittest.mock import patch

        mask = torch.tensor([[1, 1, 1, 0, 0]])
        with patch("transformers.masking_utils.create_causal_mask", return_value="delegated") as mock_cm:
            result = _passthrough_create_causal_mask(
                attention_mask=mask,
                config=None,
                inputs_embeds=torch.zeros(1, 5, 64),
                cache_position=torch.arange(5),
            )
        assert result == "delegated"
        mock_cm.assert_called_once()

    def test_handles_extra_kwargs(self):
        """Extra kwargs don't break — indexed mask still passes through."""
        mask = torch.tensor([[1, 1, 2, 2, 0]])
        result = _passthrough_create_causal_mask(
            attention_mask=mask, or_mask_function=None, and_mask_function=None
        )
        assert result is mask


# ---------------------------------------------------------------------------
# get_attn_implementation
# ---------------------------------------------------------------------------


class TestGetAttnImplementation:
    def test_from_backend_config(self):
        cfg = SimpleNamespace(backend=SimpleNamespace(attn="te"))
        assert get_attn_implementation(cfg) == "te"

    def test_from_attn_implementation(self):
        cfg = MagicMock()
        del cfg.backend
        cfg.get.return_value = "flash_attention_2"
        assert get_attn_implementation(cfg) == "flash_attention_2"

    def test_default_sdpa(self):
        assert get_attn_implementation(None) == "sdpa"

    def test_backend_takes_precedence(self):
        cfg = SimpleNamespace(backend=SimpleNamespace(attn="te"))
        cfg.get = MagicMock(return_value="flash_attention_2")
        assert get_attn_implementation(cfg) == "te"


# ---------------------------------------------------------------------------
# configure_packing
# ---------------------------------------------------------------------------


class TestConfigurePacking:
    def test_noop_for_sdpa(self):
        """configure_packing should do nothing for non-FA2 backends."""
        configure_packing("sdpa")  # should not raise

    def test_patches_flash_attention_utils(self, monkeypatch):
        """configure_packing should patch _get_unpad_data for flash_attention_2."""
        import transformers.modeling_flash_attention_utils as fa_utils

        original = fa_utils._get_unpad_data
        try:
            configure_packing("flash_attention_2")
            assert fa_utils._get_unpad_data is get_unpad_data
        finally:
            fa_utils._get_unpad_data = original

    def test_patches_loaded_model_modules(self):
        """configure_packing should patch create_causal_mask on loaded modules."""
        import transformers.modeling_flash_attention_utils as fa_utils

        original_unpad = fa_utils._get_unpad_data
        # Create a fake module with create_causal_mask
        fake_mod = MagicMock()
        fake_mod.create_causal_mask = MagicMock()
        fake_mod_name = "transformers.models.qwen3_vl.modeling_qwen3_vl"
        sys.modules[fake_mod_name] = fake_mod
        try:
            configure_packing("flash_attention_2")
            assert fake_mod.create_causal_mask is _passthrough_create_causal_mask
        finally:
            fa_utils._get_unpad_data = original_unpad
            del sys.modules[fake_mod_name]

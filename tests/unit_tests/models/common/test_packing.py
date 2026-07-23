# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_automodel.components.models.common.packing."""

from __future__ import annotations

import importlib
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.models.common.packing import (
    _passthrough_create_causal_mask,
    _patch_preprocess_mask_arguments_for_packing,
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

    def test_fa2_passthrough_for_normal_mask(self):
        """FA2 config with normal 2D mask still passes through (FA2 handles masking)."""
        config = SimpleNamespace(_attn_implementation="flash_attention_2")
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = _passthrough_create_causal_mask(config=config, attention_mask=mask)
        assert result is mask

    def test_delegates_to_original_for_non_fa2(self):
        """Non-FA2 config with normal 2D mask delegates to HF create_causal_mask."""
        from unittest.mock import patch

        config = SimpleNamespace(_attn_implementation="sdpa")
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        with patch("transformers.masking_utils.create_causal_mask", return_value="delegated") as mock_cm:
            result = _passthrough_create_causal_mask(
                attention_mask=mask,
                config=config,
                inputs_embeds=torch.zeros(1, 5, 64),
                cache_position=torch.arange(5),
            )
        assert result == "delegated"
        mock_cm.assert_called_once()
        assert "inputs_embeds" in mock_cm.call_args.kwargs
        assert "input_embeds" not in mock_cm.call_args.kwargs

    def test_handles_extra_kwargs(self):
        """Extra kwargs don't break — indexed mask still passes through."""
        mask = torch.tensor([[1, 1, 2, 2, 0]])
        result = _passthrough_create_causal_mask(attention_mask=mask, or_mask_function=None, and_mask_function=None)
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
    @pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
    def test_noop_for_unsupported_backends(self, attn_implementation, monkeypatch):
        """configure_packing should not install flash-attn shims for unsupported backends."""
        patch_preprocess = MagicMock()
        monkeypatch.setattr(
            "nemo_automodel.components.models.common.packing._patch_preprocess_mask_arguments_for_packing",
            patch_preprocess,
        )

        configure_packing(attn_implementation)

        patch_preprocess.assert_not_called()

    def test_patches_flash_attention_utils(self):
        """configure_packing should patch _get_unpad_data for flash_attention_2."""
        import transformers.masking_utils as masking_utils
        import transformers.modeling_flash_attention_utils as fa_utils

        original = fa_utils._get_unpad_data
        original_preprocess = masking_utils._preprocess_mask_arguments
        original_flag = getattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", None)
        try:
            configure_packing("flash_attention_2")
            assert fa_utils._get_unpad_data is get_unpad_data
        finally:
            fa_utils._get_unpad_data = original
            masking_utils._preprocess_mask_arguments = original_preprocess
            if original_flag is None:
                if hasattr(masking_utils, "_nemo_automodel_packing_preprocess_patched"):
                    delattr(masking_utils, "_nemo_automodel_packing_preprocess_patched")
            else:
                masking_utils._nemo_automodel_packing_preprocess_patched = original_flag

    def test_patches_preprocess_mask_arguments_for_indexed_mask(self):
        """configure_packing should preserve integer indexed masks for FA2."""
        import transformers.masking_utils as masking_utils
        import transformers.modeling_flash_attention_utils as fa_utils

        original_unpad = fa_utils._get_unpad_data
        pre_test_preprocess = masking_utils._preprocess_mask_arguments
        original_flag = getattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", None)
        # Reload so the reference call below exercises the pristine installed
        # Transformers implementation regardless of test order.
        importlib.reload(masking_utils)
        if hasattr(masking_utils, "_nemo_automodel_packing_preprocess_patched"):
            delattr(masking_utils, "_nemo_automodel_packing_preprocess_patched")
        original_preprocess = masking_utils._preprocess_mask_arguments

        config = SimpleNamespace(_attn_implementation="flash_attention_2")
        input_embeds = torch.zeros(1, 5, 8)

        def build_args(attention_mask, cache_position):
            """Build positional args matching the installed _preprocess_mask_arguments signature.

            Args:
                attention_mask: Tensor of shape [batch, sequence] containing
                    indexed document IDs, or a boolean tensor of shape [batch,
                    heads, query_sequence, key_sequence] for the reference call.
                cache_position: Optional tensor of shape [sequence] containing
                    absolute token positions.

            Returns:
                Positional argument list for the installed private Transformers
                function. Tensor layouts are unchanged.
            """
            values = {
                "config": config,
                "input_embeds": input_embeds,
                "inputs_embeds": input_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "layer_idx": 0,
            }
            args = []
            for name, param in inspect.signature(original_preprocess).parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                if name in values:
                    args.append(values[name])
                elif param.default is not inspect.Parameter.empty:
                    args.append(param.default)
                else:
                    args.append(None)
            return args

        try:
            configure_packing("flash_attention_2")
            mask = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
            probe_mask = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
            expected_template = original_preprocess(*build_args(probe_mask, torch.arange(5)))
            for cache_position in (torch.arange(5), None):
                result = masking_utils._preprocess_mask_arguments(*build_args(mask, cache_position))
                assert result[0] is expected_template[0]
                assert result[1] is mask
                assert result[2:] == expected_template[2:]

            binary_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
            expected_binary = original_preprocess(*build_args(binary_mask, None))
            result_binary = masking_utils._preprocess_mask_arguments(*build_args(binary_mask, None))
            assert result_binary[0] is expected_binary[0]
            assert result_binary[1].dtype == torch.bool
            assert torch.equal(result_binary[1], expected_binary[1])
            assert result_binary[2:] == expected_binary[2:]
        finally:
            fa_utils._get_unpad_data = original_unpad
            masking_utils._preprocess_mask_arguments = pre_test_preprocess
            if original_flag is None:
                if hasattr(masking_utils, "_nemo_automodel_packing_preprocess_patched"):
                    delattr(masking_utils, "_nemo_automodel_packing_preprocess_patched")
            else:
                masking_utils._nemo_automodel_packing_preprocess_patched = original_flag

    def test_qwen3_preserves_indexed_mask(self):
        """Qwen3 should preserve indexed masks after packing is configured."""
        import transformers.masking_utils as masking_utils
        import transformers.modeling_flash_attention_utils as fa_utils
        import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

        original_unpad = fa_utils._get_unpad_data
        original_preprocess = masking_utils._preprocess_mask_arguments
        original_flag = getattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", None)
        try:
            configure_packing("flash_attention_2")
            mask = torch.tensor([[1, 1, 2, 2, 0]], dtype=torch.long)
            result = modeling_qwen3.create_causal_mask(
                config=SimpleNamespace(_attn_implementation="flash_attention_2"),
                inputs_embeds=torch.zeros(1, 5, 8),
                attention_mask=mask,
                past_key_values=None,
                position_ids=torch.arange(5).unsqueeze(0),
            )

            assert result is mask
        finally:
            fa_utils._get_unpad_data = original_unpad
            masking_utils._preprocess_mask_arguments = original_preprocess
            if original_flag is None:
                if hasattr(masking_utils, "_nemo_automodel_packing_preprocess_patched"):
                    delattr(masking_utils, "_nemo_automodel_packing_preprocess_patched")
            else:
                masking_utils._nemo_automodel_packing_preprocess_patched = original_flag

    def test_fails_when_preprocess_shim_cannot_install(self, monkeypatch):
        """A missing private hook must fail before training can mix packed documents."""
        import transformers.masking_utils as masking_utils

        monkeypatch.setattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", False, raising=False)
        monkeypatch.delattr(masking_utils, "_preprocess_mask_arguments", raising=False)

        with pytest.raises(RuntimeError, match="Cannot enable FA2 neat packing.*_preprocess_mask_arguments"):
            _patch_preprocess_mask_arguments_for_packing()

    def test_fails_on_incompatible_preprocess_result(self, monkeypatch):
        """An incompatible private return contract must fail instead of guessing tuple fields."""
        import transformers.masking_utils as masking_utils

        def incompatible_preprocess(**kwargs):
            """Return a non-early-exit result for the 4D contract probe."""
            return False, kwargs["attention_mask"]

        monkeypatch.setattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", False, raising=False)
        monkeypatch.setattr(masking_utils, "_preprocess_mask_arguments", incompatible_preprocess)

        with pytest.raises(RuntimeError, match="incompatible _preprocess_mask_arguments early-exit result"):
            _patch_preprocess_mask_arguments_for_packing()

    def test_patches_loaded_model_modules(self):
        """configure_packing should patch create_causal_mask on loaded modules."""
        import transformers.masking_utils as masking_utils
        import transformers.modeling_flash_attention_utils as fa_utils

        original_unpad = fa_utils._get_unpad_data
        original_preprocess = masking_utils._preprocess_mask_arguments
        original_flag = getattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", None)
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
            masking_utils._preprocess_mask_arguments = original_preprocess
            if original_flag is None:
                if hasattr(masking_utils, "_nemo_automodel_packing_preprocess_patched"):
                    delattr(masking_utils, "_nemo_automodel_packing_preprocess_patched")
            else:
                masking_utils._nemo_automodel_packing_preprocess_patched = original_flag
            del sys.modules[fake_mod_name]

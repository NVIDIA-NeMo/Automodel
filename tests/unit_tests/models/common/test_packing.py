# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_automodel.components.models.common.packing."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.models.common.packing import (
    get_attn_implementation,
    get_seqlens_in_batch,
    get_unpad_data,
    validate_flash_packing_support,
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
# get_unpad_data  (pure helper used by in-tree custom models)
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

    def test_built_model_wins_over_stale_config(self):
        """A packed run force-switches the model to flash; the config keeps saying sdpa."""
        cfg = MagicMock()
        del cfg.backend
        cfg.get.return_value = "sdpa"
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flash_attention_2"))
        assert get_attn_implementation(cfg, model=model) == "flash_attention_2"

    def test_backend_config_wins_over_built_model(self):
        """Custom models keep naming their backend; ``te`` inits through sdpa."""
        cfg = SimpleNamespace(backend=SimpleNamespace(attn="te"))
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="sdpa"))
        assert get_attn_implementation(cfg, model=model) == "te"

    def test_reads_through_ddp_wrapper(self):
        """DDP holds the model as ``.module`` and does not proxy attribute access."""
        cfg = MagicMock()
        del cfg.backend
        cfg.get.return_value = "sdpa"
        inner = SimpleNamespace(config=SimpleNamespace(_attn_implementation="flash_attention_2"))
        assert get_attn_implementation(cfg, model=SimpleNamespace(module=inner)) == "flash_attention_2"

    def test_kernels_hub_id_maps_back_to_mainline_flash(self):
        """Transformers records a kernels-hub id when only ``kernels`` provides FA2."""
        cfg = MagicMock()
        del cfg.backend
        cfg.get.return_value = "flash_attention_2"
        model = SimpleNamespace(config=SimpleNamespace(_attn_implementation="kernels-community/flash-attn2"))
        assert get_attn_implementation(cfg, model=model) == "flash_attention_2"

    @pytest.mark.parametrize(
        "model",
        [
            SimpleNamespace(),
            SimpleNamespace(config=SimpleNamespace()),
            SimpleNamespace(config=SimpleNamespace(_attn_implementation=None)),
            # A dispatch key naming no layout packing knows about must not select one.
            SimpleNamespace(config=SimpleNamespace(_attn_implementation="some_future_backend")),
        ],
    )
    def test_falls_back_to_config_when_model_names_no_known_backend(self, model):
        cfg = MagicMock()
        del cfg.backend
        cfg.get.return_value = "eager"
        assert get_attn_implementation(cfg, model=model) == "eager"


# ---------------------------------------------------------------------------
# validate_flash_packing_support
# ---------------------------------------------------------------------------


class TestValidateFlashPackingSupport:
    @pytest.mark.parametrize("attn_implementation", ["sdpa", "eager", "te"])
    def test_noop_for_non_flash_backends(self, attn_implementation):
        """Non-flash backends use the 4D block-causal mask and need no varlen contract."""
        validate_flash_packing_support(attn_implementation)  # must not raise

    @pytest.mark.parametrize("impl", ["flash_attention_2", "flash_attention_3", "flash_attention_4"])
    def test_passes_when_varlen_kwargs_supported(self, impl):
        """The installed transformers exposes the public FlashAttentionKwargs contract."""
        validate_flash_packing_support(impl)  # must not raise

    def test_installs_no_global_patch(self):
        """Validation must be side-effect free: no monkeypatching of private functions."""
        import transformers.modeling_flash_attention_utils as fa_utils

        original_unpad = fa_utils._get_unpad_data
        validate_flash_packing_support("flash_attention_2")
        assert fa_utils._get_unpad_data is original_unpad

    def test_raises_when_varlen_kwargs_missing(self, monkeypatch):
        """A transformers build without the varlen kwargs must fail loudly, not silently pack."""

        def _legacy_flash_attention_forward(query, key, value, attention_mask, **kwargs):
            """Legacy signature lacking cu_seq_lens_q/max_length_q varlen kwargs."""
            return query

        monkeypatch.setattr(
            "transformers.modeling_flash_attention_utils._flash_attention_forward",
            _legacy_flash_attention_forward,
        )
        with pytest.raises(RuntimeError, match="varlen FlashAttention kwargs"):
            validate_flash_packing_support("flash_attention_2")

    def test_accepts_model_with_varkwargs(self):
        """An HF-style forward with **kwargs can receive the FlashAttentionKwargs."""

        class _Model:
            def forward(self, input_ids, position_ids=None, **kwargs):
                """Threads FlashAttentionKwargs through **kwargs."""

        validate_flash_packing_support("flash_attention_2", model=_Model())  # must not raise

    def test_accepts_model_with_packed_seq_ids_param(self):
        """A custom-model forward that names _packed_seq_ids consumes the contract."""

        class _CustomModel:
            def forward(self, input_ids, position_ids=None, _packed_seq_ids=None):
                """Custom model reads the per-document map explicitly."""

        validate_flash_packing_support("flash_attention_2", model=_CustomModel())  # must not raise

    def test_accepts_model_behind_ddp_wrapper(self):
        """The check must unwrap DDP's .module, which does not proxy attribute access."""

        class _Inner:
            def forward(self, input_ids, **kwargs):
                """Consumes via **kwargs."""

        class _DDP:
            def __init__(self, module):
                self.module = module

            def forward(self, *args, **kwargs):
                """DDP's own forward is not the packing-consuming one."""

        validate_flash_packing_support("flash_attention_2", model=_DDP(_Inner()))  # must not raise

    def test_rejects_model_that_cannot_consume_contract(self):
        """A forward with no **kwargs, no varlen params, no _packed_seq_ids must fail loudly."""

        class _BlindModel:
            def forward(self, input_ids, position_ids=None, attention_mask=None):
                """Cannot receive the typed packing metadata."""

        with pytest.raises(RuntimeError, match="_packed_seq_ids"):
            validate_flash_packing_support("flash_attention_2", model=_BlindModel())

    def test_accepts_model_with_all_four_varlen_kwargs(self):
        """A forward naming all four cumulative-length kwargs consumes the contract."""

        class _VarlenModel:
            def forward(self, input_ids, cu_seq_lens_q=None, cu_seq_lens_k=None, max_length_q=None, max_length_k=None):
                """Names the full varlen kwarg set explicitly."""

        validate_flash_packing_support("flash_attention_2", model=_VarlenModel())  # must not raise

    def test_rejects_model_with_partial_varlen_kwargs(self):
        """A forward exposing only some varlen kwargs must be rejected: HF needs all four,
        and filter_forward_kwargs would drop the rest, so the packing would silently break.
        """

        class _PartialModel:
            def forward(self, input_ids, cu_seq_lens_q=None):
                """Names one of four varlen kwargs; the other three would be dropped."""

        with pytest.raises(RuntimeError, match="all four varlen"):
            validate_flash_packing_support("flash_attention_2", model=_PartialModel())

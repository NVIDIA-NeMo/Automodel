# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_automodel.components.models.common.packing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common.packing import (
    configure_packing,
    flatten_packed_sequence_metadata,
    get_model_attn_implementation,
    get_packing_capabilities,
    validate_flash_packing_support,
)
from nemo_automodel.components.models.common.utils import BackendConfig

# ---------------------------------------------------------------------------
# model-derived packing and metadata
# ---------------------------------------------------------------------------


class TestGetAttnImplementation:
    def test_from_backend_config(self):
        model = torch.nn.Module()
        model.backend = BackendConfig(attn="te")
        assert get_model_attn_implementation(model) == "te"

    def test_from_attn_implementation(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        assert get_model_attn_implementation(model) == "flash_attention_2"

    def test_default_sdpa(self):
        assert get_model_attn_implementation(torch.nn.Module()) == "sdpa"

    def test_backend_takes_precedence(self):
        model = torch.nn.Module()
        model.backend = BackendConfig(attn="te")
        model.config = SimpleNamespace(_attn_implementation="sdpa")
        assert get_model_attn_implementation(model) == "te"

    def test_declared_hf_fa4_dispatch_takes_precedence(self):
        model = torch.nn.Module()
        model.backend = BackendConfig(attn="fa4")
        model.config = SimpleNamespace(_attn_implementation="flash_attention_4")
        model._uses_hf_attention = True

        assert get_model_attn_implementation(model) == "flash_attention_4"

    def test_declared_hf_dispatch_uses_live_hf_backend(self):
        model = torch.nn.Module()
        model.backend = BackendConfig(attn="fa4")
        model.config = SimpleNamespace(_attn_implementation="sdpa")
        model._uses_hf_attention = True

        assert get_model_attn_implementation(model) == "sdpa"

    def test_native_fa4_consumer_uses_typed_backend(self):
        model = torch.nn.Module()
        model._uses_native_fa4 = True
        model.backend = BackendConfig(attn="fa4")
        model.config = SimpleNamespace(_attn_implementation="flash_attention_4")

        assert get_model_attn_implementation(model) == "fa4"

    def test_reads_through_ddp_wrapper(self):
        """DDP holds the model as ``.module`` and does not proxy attribute access."""
        inner = torch.nn.Module()
        inner.config = SimpleNamespace(_attn_implementation="flash_attention_2")
        wrapper = torch.nn.Module()
        wrapper.module = inner
        assert get_model_attn_implementation(wrapper) == "flash_attention_2"

    def test_kernels_hub_id_maps_back_to_mainline_flash(self):
        """Transformers records a kernels-hub id when only ``kernels`` provides FA2."""
        model = torch.nn.Module()
        model.config = SimpleNamespace(_attn_implementation="kernels-community/flash-attn2")
        assert get_model_attn_implementation(model) == "flash_attention_2"

    @pytest.mark.parametrize("implementation", ["magi", "some_future_backend"])
    def test_preserves_live_dispatch_key(self, implementation):
        model = torch.nn.Module()
        model.config = SimpleNamespace(_attn_implementation=implementation)
        assert get_model_attn_implementation(model) == implementation

    def test_requires_built_model(self):
        with pytest.raises(TypeError, match="built torch.nn.Module"):
            get_model_attn_implementation(SimpleNamespace())


# ---------------------------------------------------------------------------
# configure_packing
# ---------------------------------------------------------------------------


class TestConfigurePacking:
    def test_model_semantics_select_document_ids_and_explicit_metadata(self):
        model = SimpleNamespace(
            packed_mask_type="document_ids",
            requires_packed_sequence_metadata=True,
        )

        capabilities = get_packing_capabilities("sdpa", model=model)

        assert capabilities.packed_mask_type == "document_ids"
        assert capabilities.requires_packed_sequence_metadata is True

    def test_fa4_requires_explicit_native_consumer_capability(self):
        hf_dispatched_model = torch.nn.Module()
        native_model = torch.nn.Module()
        native_model._uses_native_fa4 = True

        with pytest.raises(ValueError, match="declaring native FA4 support"):
            get_packing_capabilities("fa4", model=hf_dispatched_model)
        native_capabilities = get_packing_capabilities("fa4", model=native_model)

        assert native_capabilities.requires_packed_sequence_metadata is True
        assert native_capabilities.uses_native_fa4 is True

    def test_batch_major_metadata_flattens_after_microbatch_splitting(self):
        indices, cu_seqlens = flatten_packed_sequence_metadata(
            torch.tensor([[0, 1, 2, -1]]),
            torch.tensor([[0, 1, 3]], dtype=torch.int32),
            batch_size=1,
            sequence_length=4,
        )

        assert indices.tolist() == [0, 1, 2]
        assert cu_seqlens.tolist() == [0, 1, 3]

    def test_native_fa4_flattens_metadata_once_per_model_forward(self):
        class NativeFA4Model(torch.nn.Module):
            _uses_native_fa4 = True

            def forward(self, input_ids: torch.Tensor, **kwargs):
                """Capture normalized packed metadata.

                Args:
                    input_ids: Token IDs of shape [batch, sequence].
                    **kwargs: Model inputs containing packed token indices of
                        shape [tokens] and cumulative lengths of shape
                        [documents + 1].

                Returns:
                    The received keyword-input mapping with tensor layouts
                    unchanged.
                """
                del input_ids
                return kwargs

        model = NativeFA4Model()
        configure_packing("fa4", model=model)
        configure_packing("fa4", model=model)
        assert len(model._forward_pre_hooks) == 1
        with pytest.raises(ValueError, match="pre-packed THD"):
            model(torch.ones(2, 4, dtype=torch.long), qkv_format="thd")

        output = model(
            torch.ones(2, 4, dtype=torch.long),
            packed_token_indices=torch.tensor([[0, 1, 2, -1], [0, 1, -1, -1]]),
            cu_seqlens=torch.tensor([[0, 1, 3], [0, 2, -1]], dtype=torch.int32),
            max_seqlen=2,
        )

        assert output["packed_token_indices"].tolist() == [0, 1, 2, 4, 5]
        assert output["cu_seqlens"].tolist() == [0, 1, 3, 5]

    def test_native_fa4_rejects_incomplete_metadata_at_model_entry(self):
        class NativeFA4Model(torch.nn.Module):
            _uses_native_fa4 = True

            def forward(self, **kwargs):
                """Return the received model inputs unchanged.

                Args:
                    **kwargs: Optional cu_seqlens of shape [documents + 1] or
                        [batch, max_documents + 1], supplied without token indices.

                Returns:
                    The input mapping, unchanged; the entry hook rejects this call.
                """
                return kwargs

        model = NativeFA4Model()
        configure_packing("fa4", model=model)

        with pytest.raises(ValueError, match="must be tensors supplied together"):
            model(cu_seqlens=torch.tensor([0, 2], dtype=torch.int32))


def test_non_metadata_consumer_preserves_legacy_thd_boundaries():
    class LegacyModel(torch.nn.Module):
        def forward(self, **kwargs):
            """Return legacy THD boundaries unchanged.

            Args:
                **kwargs: ``cu_seqlens`` of shape [documents + 1] and scalar format options.

            Returns:
                The original boundary tensor of shape [documents + 1].
            """
            return kwargs["cu_seqlens"]

    model = LegacyModel()
    configure_packing("sdpa", model=model)
    boundaries = torch.tensor([0, 2, 5], dtype=torch.int32)
    assert model(cu_seqlens=boundaries, qkv_format="thd") is boundaries


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
        contract = configure_packing("flash_attention_2")
        assert contract.packed_mask_type == "flash_varlen"
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

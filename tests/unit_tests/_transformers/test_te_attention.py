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

"""Unit tests for te_attention.py."""

from unittest import mock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel._transformers.te_attention import (
    _infer_attn_params,
    _make_te_sdpa,
    inject_te_attention,
    inject_te_attention_into_module,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_self_attn(num_heads=8, num_kv_heads=4, head_dim=64):
    """Return a minimal module that looks like a Llama-style self_attn."""
    module = nn.Module()
    module.q_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim, bias=False)
    module.k_proj = nn.Linear(num_heads * head_dim, num_kv_heads * head_dim, bias=False)
    module.v_proj = nn.Linear(num_heads * head_dim, num_kv_heads * head_dim, bias=False)
    module.o_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim, bias=False)
    module.num_heads = num_heads
    module.num_key_value_heads = num_kv_heads
    module.head_dim = head_dim
    return module


def _make_mock_model(num_layers=2, **attn_kwargs):
    """Return a tiny mock model with self_attn submodules."""

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _make_mock_self_attn(**attn_kwargs)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(num_layers)])

    return _Model()


# ---------------------------------------------------------------------------
# _infer_attn_params
# ---------------------------------------------------------------------------


class TestInferAttnParams:
    def test_standard_layout(self):
        module = _make_mock_self_attn(num_heads=16, num_kv_heads=8, head_dim=128)
        params = _infer_attn_params(module)
        assert params == {
            "num_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "window_size": (-1, 0),
            "softmax_scale": 128**-0.5,
        }

    def test_mha_layout(self):
        """MHA: num_kv_heads defaults to num_heads."""
        module = _make_mock_self_attn(num_heads=8, num_kv_heads=8, head_dim=64)
        params = _infer_attn_params(module)
        assert params["num_kv_heads"] == 8

    def test_missing_q_proj_returns_none(self):
        module = nn.Module()
        module.num_heads = 8
        module.head_dim = 64
        assert _infer_attn_params(module) is None

    def test_infers_num_heads_from_proj_when_attr_missing(self):
        """When num_heads attribute is absent, infer from q_proj.out_features // head_dim."""
        module = nn.Module()
        module.q_proj = nn.Linear(512, 512, bias=False)
        module.k_proj = nn.Linear(512, 256, bias=False)
        module.v_proj = nn.Linear(512, 256, bias=False)
        module.head_dim = 64
        params = _infer_attn_params(module)
        assert params is not None
        assert params["num_heads"] == 8  # 512 // 64
        assert params["num_kv_heads"] == 4  # 256 // 64

    def test_missing_head_dim_returns_none(self):
        module = nn.Module()
        module.q_proj = nn.Linear(512, 512, bias=False)
        module.k_proj = nn.Linear(512, 512, bias=False)
        module.v_proj = nn.Linear(512, 512, bias=False)
        module.num_heads = 8
        assert _infer_attn_params(module) is None

    def test_num_attention_heads_alias(self):
        """Some HF models use num_attention_heads instead of num_heads."""
        module = nn.Module()
        module.q_proj = nn.Linear(512, 512, bias=False)
        module.k_proj = nn.Linear(512, 512, bias=False)
        module.v_proj = nn.Linear(512, 512, bias=False)
        module.num_attention_heads = 8
        module.head_dim = 64
        params = _infer_attn_params(module)
        assert params is not None
        assert params["num_heads"] == 8

    def test_sliding_window_sets_te_window_size(self):
        """sliding_window=512 should yield window_size=(511, 0) for TE."""
        module = _make_mock_self_attn(num_heads=8, num_kv_heads=4, head_dim=64)
        module.sliding_window = 512
        params = _infer_attn_params(module)
        assert params is not None
        assert params["window_size"] == (511, 0)

    def test_no_sliding_window_gives_unbounded(self):
        """No sliding_window attribute means unbounded context: (-1, 0)."""
        module = _make_mock_self_attn(num_heads=8, num_kv_heads=4, head_dim=64)
        params = _infer_attn_params(module)
        assert params is not None
        assert params["window_size"] == (-1, 0)


# ---------------------------------------------------------------------------
# _make_te_sdpa
# ---------------------------------------------------------------------------


class TestTeSdpa:
    def _make_te_sdpa_with_mock(self, num_heads=8, num_kv_heads=4):
        te_module = mock.MagicMock()
        # TE returns [B, S, H, D]
        te_module.return_value = torch.zeros(2, 10, num_heads, 64)
        original_sdpa = mock.MagicMock(return_value=torch.zeros(2, num_heads, 10, 64))
        te_sdpa = _make_te_sdpa(te_module, num_heads=num_heads, num_kv_heads=num_kv_heads, original_sdpa=original_sdpa)
        return te_sdpa, te_module, original_sdpa

    def test_causal_transpose_and_output_shape(self):
        """Verifies Q/K/V are transposed for TE and output is transposed back."""
        B, H, S, D = 2, 8, 10, 64
        Hkv = 4
        te_sdpa, te_module, _ = self._make_te_sdpa_with_mock(H, Hkv)

        q = torch.randn(B, H, S, D)
        k = torch.randn(B, Hkv, S, D)
        v = torch.randn(B, Hkv, S, D)

        out = te_sdpa(q, k, v, is_causal=True, enable_gqa=True)

        assert out.shape == (B, H, S, D)
        call_args = te_module.call_args
        assert call_args.kwargs["attn_mask_type"] == "causal"
        # TE received [B, S, H, D]
        q_te, k_te, v_te = call_args.args
        assert q_te.shape == (B, S, H, D)
        assert k_te.shape == (B, S, Hkv, D)

    def test_no_mask_type_when_not_causal(self):
        te_sdpa, te_module, _ = self._make_te_sdpa_with_mock()
        B, H, S, D = 1, 8, 5, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, 4, S, D)
        v = torch.randn(B, 4, S, D)
        te_sdpa(q, k, v, is_causal=False, enable_gqa=True)
        assert te_module.call_args.kwargs["attn_mask_type"] == "no_mask"

    def test_repeat_kv_undone_for_gqa(self):
        """When K/V heads are repeated (enable_gqa=False), undo repeat_kv."""
        num_heads, num_kv_heads = 8, 2
        te_sdpa, te_module, _ = self._make_te_sdpa_with_mock(num_heads, num_kv_heads)
        te_module.return_value = torch.zeros(1, 4, num_heads, 32)

        B, H, S, D = 1, num_heads, 4, 32
        q = torch.randn(B, H, S, D)
        # K/V have been repeat_kv'd to num_heads
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        te_sdpa(q, k, v, is_causal=True, enable_gqa=False)

        _, k_te, v_te = te_module.call_args.args
        # After undoing repeat_kv and transposing: [B, S, Hkv, D]
        assert k_te.shape == (B, S, num_kv_heads, D)
        assert v_te.shape == (B, S, num_kv_heads, D)

    def test_non_null_mask_falls_back_to_original_sdpa(self):
        """Non-trivial attn_mask should fall back to the original SDPA."""
        te_sdpa, te_module, original_sdpa = self._make_te_sdpa_with_mock()
        B, H, S, D = 1, 8, 4, 64
        original_sdpa.return_value = torch.zeros(B, H, S, D)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, 4, S, D)
        v = torch.randn(B, 4, S, D)
        mask = torch.zeros(B, 1, S, S)

        te_sdpa(q, k, v, attn_mask=mask, enable_gqa=True)

        te_module.assert_not_called()
        original_sdpa.assert_called_once()


# ---------------------------------------------------------------------------
# inject_te_attention_into_module
# ---------------------------------------------------------------------------


class TestInjectTEAttentionIntoModule:
    def test_sets_attn_module(self):
        with mock.patch("nemo_automodel._transformers.te_attention._create_te_dot_product_attention") as mock_create:
            fake_te = mock.MagicMock()
            mock_create.return_value = fake_te

            module = _make_mock_self_attn()
            result = inject_te_attention_into_module(module)

        assert result is True
        assert hasattr(module, "attn_module")
        assert module.attn_module is fake_te

    def test_patches_forward(self):
        with mock.patch("nemo_automodel._transformers.te_attention._create_te_dot_product_attention") as mock_create:
            mock_create.return_value = mock.MagicMock()
            module = _make_mock_self_attn()
            inject_te_attention_into_module(module)

        # Instance-level forward should now shadow the class-level one.
        assert "forward" in module.__dict__

    def test_returns_false_for_incompatible_module(self):
        module = nn.Linear(32, 32)
        result = inject_te_attention_into_module(module)
        assert result is False

    def test_skips_already_patched_module(self):
        with mock.patch("nemo_automodel._transformers.te_attention._create_te_dot_product_attention") as mock_create:
            mock_create.return_value = mock.MagicMock()
            module = _make_mock_self_attn()
            # Manually set attn_module to simulate prior injection
            module.attn_module = mock.MagicMock()
            old_attn_module = module.attn_module

            inject_te_attention(mock.MagicMock(**{"named_modules.return_value": [("self_attn", module)]}))

        # Should not have replaced the existing attn_module
        assert module.attn_module is old_attn_module


# ---------------------------------------------------------------------------
# inject_te_attention (model-level)
# ---------------------------------------------------------------------------


class TestInjectTEAttention:
    def test_injects_into_all_self_attn_modules(self):
        model = _make_mock_model(num_layers=3)
        with mock.patch("nemo_automodel._transformers.te_attention._create_te_dot_product_attention") as mock_create:
            mock_create.return_value = mock.MagicMock()
            inject_te_attention(model)

        for layer in model.layers:
            assert hasattr(layer.self_attn, "attn_module"), "attn_module should be set"

    def test_sets_model_flag(self):
        model = _make_mock_model(num_layers=1)
        with mock.patch("nemo_automodel._transformers.te_attention._create_te_dot_product_attention") as mock_create:
            mock_create.return_value = mock.MagicMock()
            inject_te_attention(model)

        assert getattr(model, "_te_attention_injected", False) is True

    def test_skips_already_patched_self_attn(self):
        model = _make_mock_model(num_layers=2)
        existing_te = mock.MagicMock()
        model.layers[0].self_attn.attn_module = existing_te

        with mock.patch("nemo_automodel._transformers.te_attention._create_te_dot_product_attention") as mock_create:
            new_te = mock.MagicMock()
            mock_create.return_value = new_te
            inject_te_attention(model)

        # Layer 0's attn_module should remain unchanged
        assert model.layers[0].self_attn.attn_module is existing_te
        # Layer 1's attn_module should be the newly created one
        assert model.layers[1].self_attn.attn_module is new_te

    def test_no_flag_when_no_compatible_modules(self):
        model = nn.Linear(32, 32)  # No self_attn submodules
        inject_te_attention(model)
        assert not getattr(model, "_te_attention_injected", False)


# ---------------------------------------------------------------------------
# Forward monkey-patching round-trip
# ---------------------------------------------------------------------------


class TestForwardPatch:
    """Verify that the patched forward swaps F.scaled_dot_product_attention."""

    def test_forward_uses_te_sdpa(self):
        """The patched forward must call the TE SDPA replacement, not the original."""

        te_sdpa_calls = []

        def fake_te_sdpa(*args, **kwargs):
            te_sdpa_calls.append((args, kwargs))
            # Return a tensor that matches the expected [B, H, S, D] shape
            return torch.zeros(1, 4, 5, 16)

        class _MinimalAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Identity()
                self.k_proj = nn.Identity()
                self.v_proj = nn.Identity()
                self.num_heads = 4
                self.num_key_value_heads = 4
                self.head_dim = 16

            def forward(self, x):
                q = x.view(1, 4, 5, 16)
                # Calls F.scaled_dot_product_attention — should be intercepted
                return F.scaled_dot_product_attention(q, q, q, is_causal=True)

        module = _MinimalAttn()

        with mock.patch("nemo_automodel._transformers.te_attention._create_te_dot_product_attention") as mock_create:
            mock_te = mock.MagicMock()
            mock_te.return_value = torch.zeros(1, 5, 4, 16)  # [B, S, H, D]
            mock_create.return_value = mock_te
            inject_te_attention_into_module(module)

        # During forward, F.scaled_dot_product_attention should reach the TE path.
        x = torch.randn(1, 4, 5, 16)
        original_sdpa = F.scaled_dot_product_attention
        try:
            module.forward(x)
        finally:
            # Ensure restoration happened
            assert F.scaled_dot_product_attention is original_sdpa, "SDPA must be restored after forward"

        # TE module should have been called
        assert mock_te.called, "TE DotProductAttention should have been called"

    def test_sdpa_restored_after_forward(self):
        """F.scaled_dot_product_attention must be restored even if forward raises."""

        class _RaisingAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Identity()
                self.k_proj = nn.Identity()
                self.v_proj = nn.Identity()
                self.num_heads = 4
                self.num_key_value_heads = 4
                self.head_dim = 16

            def forward(self, x):
                _ = F.scaled_dot_product_attention(x, x, x)
                raise RuntimeError("intentional error")

        module = _RaisingAttn()
        with mock.patch("nemo_automodel._transformers.te_attention._create_te_dot_product_attention") as mock_create:
            mock_te = mock.MagicMock()
            mock_te.return_value = torch.zeros(1, 2, 4, 16)
            mock_create.return_value = mock_te
            inject_te_attention_into_module(module)

        original_sdpa = F.scaled_dot_product_attention
        with pytest.raises(RuntimeError, match="intentional error"):
            module.forward(torch.randn(1, 4, 2, 16))

        assert F.scaled_dot_product_attention is original_sdpa, "SDPA must be restored after exception"

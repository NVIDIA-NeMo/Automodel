# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Hopper execution regression for standard Gemma4 MoE FFPA mask routing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from nemo_automodel.components.attention import ffpa_attention as ffpa
from nemo_automodel.components.models.gemma4_moe.model import (
    _build_unpacked_gemma4_causal_mask_mapping,
)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="FFPA 0.2.4 CuTe D512 backward requires a Hopper GPU",
)
@pytest.mark.parametrize("batch,sequence", [(2, 128), (1, 2048)])
def test_gemma4_text_mask_dispatches_stock_cute_with_strided_qkv(monkeypatch, batch, sequence):
    pytest.importorskip("ffpa_attn", reason="requires the optional ffpa extra")
    from ffpa_attn.functional import CuTeDSLBackend

    ffpa.register_ffpa_attention()
    config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=512,
        sliding_window=64,
        layer_types=["sliding_attention", "full_attention"],
        use_bidirectional_attention="vision",
    )
    config._attn_implementation = "ffpa"
    embeddings = torch.zeros(batch, sequence, 32, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(sequence, device="cuda").unsqueeze(0).expand(batch, -1)
    token_types = torch.zeros(batch, sequence, device="cuda", dtype=torch.long)
    masks = _build_unpacked_gemma4_causal_mask_mapping(
        config,
        embeddings,
        torch.ones_like(token_types),
        None,
        positions,
        token_types,
        None,
        is_training=True,
    )
    assert masks["full_attention"] is None
    assert isinstance(masks["sliding_attention"], torch.nn.attention.flex_attention.BlockMask)

    real_function, backend = ffpa._get_ffpa_high_level()
    assert isinstance(backend, CuTeDSLBackend)
    called = Mock(wraps=real_function)
    monkeypatch.setattr(ffpa, "_FFPA_HIGH_LEVEL", (called, backend))
    torch.manual_seed(42)
    # Gemma projects BSHD and transposes to BHND; do not conceal that stride contract.
    q, k, v = [
        (torch.randn(batch, sequence, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1)
        .transpose(1, 2)
        .requires_grad_()
        for heads in (4, 2, 2)
    ]
    assert not q.is_contiguous()
    module = SimpleNamespace(head_dim=512, num_key_value_groups=2, training=True, is_causal=True)
    output, weights = ffpa.ffpa_attention_forward(module, q, k, v, masks["full_attention"], scaling=1.0)
    called.assert_called_once()
    assert called.call_args.kwargs["backend"] is backend
    assert weights is None
    reference_inputs = [tensor.detach().float().requires_grad_() for tensor in (q, k, v)]
    reference = torch.nn.functional.scaled_dot_product_attention(
        *reference_inputs, is_causal=True, enable_gqa=True, scale=1.0
    ).transpose(1, 2)
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (q, k, v), upstream)
    reference_grads = torch.autograd.grad(reference, reference_inputs, upstream.float())
    torch.cuda.synchronize()
    # BF16 fused attention rounds probabilities differently from FP32 reference.
    assert (output.float() - reference).norm() / reference.norm() < 0.01
    for actual, expected in zip(actual_grads, reference_grads, strict=True):
        assert torch.isfinite(actual).all()
        assert (actual.float() - expected).norm() / expected.norm() < 0.02

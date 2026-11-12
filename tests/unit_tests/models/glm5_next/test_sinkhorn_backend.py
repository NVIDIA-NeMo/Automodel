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

"""CPU contracts for the optional GLM Sinkhorn backend and mHC precision."""

from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.glm5_next.backend import Glm5NextBackendConfig
from nemo_automodel.components.models.glm5_next.cp import Glm5NextPackedContext
from nemo_automodel.components.models.glm5_next.layers import Glm5NextDecoderLayer, Glm5NextHyperConnection
from nemo_automodel.components.models.glm5_next.model import (
    Glm5NextForConditionalGeneration,
    build_glm5_next_moe_config,
)
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config


class _IdentityUpdate(nn.Module):
    def forward(self, hidden: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return hidden


def _reference_hc(
    hidden: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    hc: int,
    norm_eps: float,
    eps: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent, explicit mathematical reference with the original cast sites.

    Args:
        hidden: Tensor of shape [batch, sequence, streams, hidden], where
            streams is ``hc``.
        fn: Tensor of shape [streams * (2 + streams), streams * hidden].
        base: FP32 tensor of shape [streams * (2 + streams)], split into
            pre, post and row-major comb biases.
        scale: FP32 tensor of shape [3] holding the pre, post and comb scales.
        hc: Number of residual streams.
        norm_eps: RMS normalization epsilon.
        eps: Sinkhorn and ``pre`` stabilizer.
        iterations: Number of Sinkhorn rounds.

    Returns:
        FP32 ``post`` of shape [batch, sequence, streams], FP32 ``comb`` of
        shape [batch, sequence, streams, streams] indexed [source_stream,
        destination_stream], and ``collapsed`` of shape [batch, sequence,
        hidden] in ``hidden``'s dtype.
    """
    flat = hidden.flatten(start_dim=2).float()
    flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre_w, post_w, comb_w = F.linear(flat, fn.float()).split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
    comb = (comb_w.reshape(*comb_w.shape[:-1], hc, hc) * scale[2] + comb_b.reshape(hc, hc)).softmax(-1)
    comb = comb + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    collapsed = (pre.unsqueeze(-1) * hidden).sum(2).to(hidden.dtype)
    return post, comb, collapsed


@pytest.mark.parametrize("sinkhorn", ["torch", "tilelang"])
def test_model_threads_independent_sinkhorn_backend(sinkhorn: str) -> None:
    backend = Glm5NextBackendConfig(**{**asdict(tiny_backend()), "attn": "cudnn", "sinkhorn": sinkhorn})
    config = tiny_glm5_next_config()
    config.text_config.kv_lora_rank = 512
    model = Glm5NextForConditionalGeneration(config, backend=backend)
    assert model.backend.attn == "cudnn"
    for layer in model.language_model.layers.values():
        assert layer.attn_hc._fp32_params.sinkhorn_backend == sinkhorn
        assert layer.ffn_hc._fp32_params.sinkhorn_backend == sinkhorn
        assert layer.attn_hc.base.dtype == torch.float32
        assert layer.attn_hc.scale.dtype == torch.float32


def test_existing_shared_backend_preserves_torch_sinkhorn() -> None:
    model = Glm5NextForConditionalGeneration(tiny_glm5_next_config(), backend=tiny_backend())
    assert all(layer.attn_hc._fp32_params.sinkhorn_backend == "torch" for layer in model.language_model.layers.values())
    assert Glm5NextBackendConfig(**asdict(tiny_backend())).sinkhorn == "torch"


def test_invalid_sinkhorn_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported sinkhorn"):
        Glm5NextBackendConfig(**{**asdict(tiny_backend()), "sinkhorn": "invalid"})


def test_explicit_tilelang_does_not_fall_back_on_cpu() -> None:
    module = Glm5NextHyperConnection(tiny_glm5_next_config().text_config, sinkhorn_backend="tilelang")
    module.init_weights(torch.device("cpu"), 0.02)
    with pytest.raises(RuntimeError, match="Sinkhorn|sinkhorn"):
        module(torch.randn(1, 3, 2, 16))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("iterations", [1, 20])
def test_full_mhc_decoder_preserves_rounding_and_gradients(dtype: torch.dtype, iterations: int) -> None:
    """Exercise actual decoder expand sites, both mHCs, projection, collapse and VJP."""
    torch.manual_seed(191)
    config = tiny_glm5_next_config().text_config
    config.torch_dtype = dtype
    config.hc_sinkhorn_iters = iterations
    layer = Glm5NextDecoderLayer(config, 0, build_glm5_next_moe_config(config, dtype), tiny_backend())
    # Isolate the mHC transforms while retaining the production decoder's two
    # expansion expressions. These stubs are not attention/MLP validation.
    layer.self_attn = _IdentityUpdate()
    layer.mlp = _IdentityUpdate()
    for hc in (layer.attn_hc, layer.ffn_hc):
        hc.init_weights(torch.device("cpu"), 0.1)
        with torch.no_grad():
            hc.base.normal_(std=0.3)
            hc.scale.copy_(torch.tensor([0.7, -0.4, 1.3]))
    hidden = torch.randn(1, 7, 2, 16, dtype=dtype, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    parameters = dict(layer.named_parameters())
    expected_parameters = {name: value.detach().clone().requires_grad_() for name, value in parameters.items()}
    expected = expected_hidden
    for name, norm_name in (("attn_hc", "input_layernorm"), ("ffn_hc", "post_attention_layernorm")):
        post, comb, collapsed = _reference_hc(
            expected,
            expected_parameters[f"{name}.fn"],
            expected_parameters[f"{name}._fp32_params.base"],
            expected_parameters[f"{name}._fp32_params.scale"],
            config.hc_mult,
            config.rms_norm_eps,
            config.hc_eps,
            config.hc_sinkhorn_iters,
        )
        update = collapsed.float()
        update = update * torch.rsqrt(update.square().mean(-1, keepdim=True) + config.rms_norm_eps)
        update = expected_parameters[f"{norm_name}.weight"] * update.to(dtype)
        # Cast operands before BF16 multiply/matmul, then round each result and
        # the final sum. Delaying those casts changes the checkpoint's math.
        expected = post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), expected
        )
    context = Glm5NextPackedContext(doc_ids=torch.ones(1, 7, dtype=torch.int32), original_seq_len=7)
    actual = layer(hidden, packed_context=context)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn(*actual.shape[:-2], actual.shape[-1], actual.shape[-2], dtype=dtype).transpose(-1, -2)
    assert not upstream.is_contiguous()
    actual_grads = torch.autograd.grad(actual, (hidden, *parameters.values()), upstream)
    expected_grads = torch.autograd.grad(expected, (expected_hidden, *expected_parameters.values()), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert layer.attn_hc.fn.dtype == dtype
    assert layer.attn_hc.base.dtype == torch.float32
    assert layer.attn_hc.scale.dtype == torch.float32

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

from functools import partial

import torch
import torch.nn.functional as F

from nemo_automodel.components.models.hy_v4 import layers as hy_v4_layers
from nemo_automodel.components.models.hy_v4.config import HyV4Config
from nemo_automodel.components.models.hy_v4.hc import HyV4HCHead, HyV4HCPost, HyV4HCPre
from nemo_automodel.components.models.hy_v4.model import HyV4ForCausalLM


def _rms_rsqrt(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Return a new FP32 inverse RMS ``[..., 1]`` for ``x[..., features]``."""
    return torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)


def _vllm_indexer_math(
    q: torch.Tensor,
    k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    topk: int,
    **_kwargs,
) -> torch.Tensor:
    """Small exact-math stand-in for vLLM's sparse indexer kernel.

    Args:
        q: Query features ``[tokens, heads, index_dim]``.
        k: Key features ``[tokens, index_dim]``.
        head_weights: Per-token head weights ``[tokens, heads]``.
        cu_seqlens: Packed boundaries ``[documents + 1]``.
        topk: Fixed output width.
        **_kwargs: Ignored optimized-kernel metadata.

    Returns:
        New int32 indices ``[tokens, 1, topk]`` with ``-1`` padding.
    """
    scores = torch.einsum("qhd,kd->qhk", q.float(), k.float()).relu()
    scores = (scores * head_weights.float().unsqueeze(-1)).sum(dim=1)
    indices = torch.full((q.shape[0], 1, topk), -1, dtype=torch.int32, device=q.device)
    boundaries = cu_seqlens.reshape(-1).tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for query in range(start, end):
            width = min(topk, query - start + 1)
            selected = scores[query, start : query + 1].topk(width).indices + start
            indices[query, 0, :width] = selected.to(torch.int32)
    return indices


def _vllm_sparse_mla_math(
    q: torch.Tensor,
    latent_kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    *,
    attn_sink: torch.Tensor | None = None,
    value_head_dim: int,
    **_kwargs,
) -> torch.Tensor:
    """Evaluate the unquantized sparse-MQA math used by vLLM HY4.

    Args:
        q: Absorbed queries ``[tokens, heads, latent_dim]``.
        latent_kv: Shared latent keys/values ``[tokens, 1, latent_dim]``.
        indices: Sparse key indices ``[tokens, 1, sparse_width]``.
        softmax_scale: Query/key score scale.
        attn_sink: Optional FP32 sink logits ``[heads]``.
        value_head_dim: Leading latent width interpreted as values.
        **_kwargs: Ignored optimized-kernel metadata.

    Returns:
        New sparse attention values ``[tokens, heads, value_head_dim]``.
    """
    keys = latent_kv.squeeze(1)
    rows = []
    for query in range(q.shape[0]):
        selected = indices[query, 0]
        selected = selected[(selected >= 0) & (selected < keys.shape[0])].long()
        selected_keys = keys.index_select(0, selected)
        scores = torch.matmul(q[query].float(), selected_keys.float().T) * softmax_scale
        if attn_sink is not None:
            sink = attn_sink.float().unsqueeze(-1)
            probs = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1)
        selected_values = selected_keys[..., :value_head_dim]
        rows.append(torch.matmul(probs, selected_values.float()).to(q.dtype))
    return torch.stack(rows)


def test_ihc_pre_and_head_match_explicit_fp32_reference():
    config = HyV4Config(
        hidden_size=3,
        num_hidden_layers=1,
        mlp_layer_types=["dense"],
        indexer_types=["full"],
        hc_mult=2,
        hc_magnitude=2.0,
        hc_eps=1e-6,
    )
    pre = HyV4HCPre(config)
    head = HyV4HCHead(config)
    torch.manual_seed(7)
    with torch.no_grad():
        for parameter in (*pre.parameters(), *head.parameters()):
            parameter.normal_(mean=0.0, std=0.2)

    streams = torch.randn(2, 4, 2, 3, requires_grad=True)
    reduced, post = pre(streams)
    flat = streams.flatten(start_dim=-2)
    mixes = F.linear(flat.float(), pre.hc_fn) * _rms_rsqrt(flat, config.rms_norm_eps)
    pre_raw, post_raw = mixes.chunk(2, dim=-1)
    expected_pre = torch.sigmoid(pre_raw * pre.hc_scale[0] + pre.hc_base[:2]) + config.hc_eps
    expected_post = 2.0 * torch.sigmoid(post_raw * pre.hc_scale[1] + pre.hc_base[2:]) + config.hc_eps
    expected_reduced = (expected_pre.unsqueeze(-1) * streams).sum(dim=-2)

    torch.testing.assert_close(reduced, expected_reduced)
    torch.testing.assert_close(post, expected_post)

    head_out = head(streams)
    expected_head_mix = F.linear(flat.float(), head.hc_head_fn) * _rms_rsqrt(flat, config.rms_norm_eps)
    expected_head_gate = torch.sigmoid(expected_head_mix * head.hc_head_scale + head.hc_head_base) + config.hc_eps
    expected_head = (expected_head_gate.unsqueeze(-1) * streams).sum(dim=-2)
    torch.testing.assert_close(head_out, expected_head)

    (reduced.square().mean() + post.square().mean() + head_out.square().mean()).backward()
    assert streams.grad is not None and torch.isfinite(streams.grad).all()
    assert all(parameter.grad is not None for parameter in (*pre.parameters(), *head.parameters()))


def test_ihc_post_matches_vllm_fp32_reference_forward_and_backward():
    torch.manual_seed(19)
    post_layer = HyV4HCPost()
    sublayer_output = torch.randn(2, 5, 7, requires_grad=True)
    residual = torch.randn(2, 5, 4, 7, requires_grad=True)
    post_gates = torch.randn(2, 5, 4, requires_grad=True)
    output_gradient = torch.randn_like(residual)

    actual = post_layer(sublayer_output, residual, post_gates)
    actual_grads = torch.autograd.grad(
        actual,
        (sublayer_output, residual, post_gates),
        output_gradient,
    )

    reference_output = (post_gates.float().unsqueeze(-1) * sublayer_output.float().unsqueeze(-2) + residual.float()).to(
        sublayer_output.dtype
    )
    reference_grads = torch.autograd.grad(
        reference_output,
        (sublayer_output, residual, post_gates),
        output_gradient,
    )

    torch.testing.assert_close(actual, reference_output, rtol=0, atol=0)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)


def test_indexshare_layers_reuse_the_previous_full_indexer(tiny_hy_v4_model):
    layers = tiny_hy_v4_model.model.layers

    assert layers["0"].self_attn.indexer is not None
    assert layers["1"].self_attn.indexer is not None
    assert layers["2"].self_attn.indexer is None


def test_tiny_model_forward_backward_covers_ihc_moe_sink_and_mtp(monkeypatch, tiny_hy_v4_model):
    monkeypatch.setattr(hy_v4_layers, "is_cudnn_dsa_available", lambda: True)
    monkeypatch.setattr(hy_v4_layers, "cudnn_indexer_topk", _vllm_indexer_math)
    model = tiny_hy_v4_model.train()
    monkeypatch.setattr(
        hy_v4_layers,
        "cudnn_sparse_attention",
        partial(_vllm_sparse_mla_math, value_head_dim=model.config.kv_lora_rank),
    )
    input_ids = torch.tensor([[1, 3, 5, 7]])
    output = model(
        input_ids=input_ids,
        position_ids=torch.arange(4).unsqueeze(0),
        qkv_format="thd",
        cu_seqlens=torch.tensor([[0, 4]], dtype=torch.int32),
        output_hidden_states=True,
    )

    assert output.logits.shape == (1, 4, 32)
    assert output.logits.dtype is torch.float32
    assert output.hidden_states.shape == (1, 4, 8)
    assert output.mtp_per_depth_h is not None
    assert len(output.mtp_per_depth_h) == 1
    assert output.mtp_per_depth_h[0].shape == (1, 4, 8)
    assert output.mtp_loss_scaling_factor == 0.1

    loss = output.logits.square().mean() + 0.1 * output.mtp_per_depth_h[0].square().mean()
    loss.backward()

    parameters = dict(model.named_parameters())
    expected_gradients = (
        "model.layers.0.hc_attn_layer.hc_pre.hc_fn",
        "model.layers.0.self_attn.learnable_sink_param.weight",
        "model.layers.1.mlp.gate.weight",
        "model.mtp_layers.0.eh_proj.weight",
        "model.mtp_layers.0.mlp.gate.weight",
    )
    for name in expected_gradients:
        assert parameters[name].grad is not None, name
        assert torch.isfinite(parameters[name].grad).all(), name


def test_reference_compute_keeps_ihc_sink_and_router_state_in_fp32(tiny_hy_v4_config, tiny_backend):
    config_dict = tiny_hy_v4_config.to_dict()
    config_dict["dtype"] = "bfloat16"
    config_dict.pop("torch_dtype", None)
    config = HyV4Config.from_dict(config_dict)
    model = HyV4ForCausalLM(config, backend=tiny_backend)
    state = model.state_dict()

    assert state["model.layers.0.hc_attn_layer.hc_pre.hc_fn"].dtype is torch.float32
    assert state["model.layers.0.self_attn.learnable_sink_param.weight"].dtype is torch.float32
    assert state["model.layers.1.mlp.gate.e_score_correction_bias"].dtype is torch.float32
    assert state["model.mtp_layers.0.self_attn.learnable_sink_param.weight"].dtype is torch.float32
    assert state["model.mtp_layers.0.mlp.gate.e_score_correction_bias"].dtype is torch.float32
    assert state["model.mtp_layers.0.eh_proj.weight"].dtype is torch.bfloat16
    assert state["lm_head.weight"].dtype is torch.bfloat16

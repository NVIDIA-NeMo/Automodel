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

"""Independent floating equations: token-wise attention/convolution and per-token experts.

This oracle does not call any Inkling2 production layer or common MoE forward.
It validates synthetic floating weights, not the quantized serving arithmetic.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from nemo_automodel.components.models.inkling2.config import Inkling2TextConfig


@dataclass
class ReferenceOutput:
    """FP32 logits [B,T,V], floating partial blocks [B,T,H], and routing IDs [B*T,K]."""

    logits: torch.Tensor
    partials: list[torch.Tensor]
    routing_ids: list[torch.Tensor]


def rms(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Normalize the last axis.

    Args:
        x: Tensor of shape [..., hidden].
        weight: Optional tensor of shape [hidden].
        eps: Stabilizer.

    Returns:
        FP32 tensor of shape [..., hidden].
    """
    out = x.float() / torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return out if weight is None else out * weight.float()


def mix(sources: list[torch.Tensor], query: torch.Tensor, normalize: bool) -> torch.Tensor:
    """Evaluate one depth query without the production einsum implementation.

    Args:
        sources: Tensors of shape [batch, sequence, hidden].
        query: Tensor of shape [hidden].
        normalize: Whether values are RMS-normalized.

    Returns:
        Tensor of shape [batch, sequence, hidden].
    """
    keys = [rms(source, None, 1e-6) for source in sources]
    logits = torch.stack([(key * query).sum(-1) / query.numel() ** 0.5 for key in keys], -1)
    probabilities = logits.softmax(-1)
    values = keys if normalize else [source.float() for source in sources]
    return sum(value * probabilities[..., i, None] for i, value in enumerate(values))


def convolution(
    x: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    prefix: str,
    *,
    width: int,
    valid: torch.Tensor,
    dynamic_input: torch.Tensor | None = None,
    heads: int = 1,
) -> torch.Tensor:
    """Token-wise convolution with the source's static/dynamic tap orientation.

    Args:
        x: Tensor of shape [batch, sequence, channels].
        weights: Parameter tensors keyed by native names.
        prefix: Convolution's native name.
        width: Number of taps.
        valid: Boolean tensor of shape [batch, sequence].
        dynamic_input: Optional tensor of shape [batch, sequence, hidden].
        heads: Number of generated filters.

    Returns:
        Tensor of shape [batch, sequence, channels].
    """
    x = x * valid[..., None]
    results = []
    for t in range(x.shape[1]):
        out = x[:, t].float()
        for lag in range(min(width, t + 1)):
            if dynamic_input is None:
                coefficient = weights[prefix + ".weight"][:, 0, width - 1 - lag].float()
            else:
                generated = F.linear(dynamic_input[:, t], weights[prefix + ".weight_proj.weight"]).float()
                generated = generated.reshape(x.shape[0], heads, width)[:, :, lag]
                generated = torch.cat(
                    [generated[:, h : h + 1].expand(-1, x.shape[-1] // heads) for h in range(heads)],
                    -1,
                )
                coefficient = weights[prefix + ".static_w"][lag].float() + generated
            out = out + x[:, t - lag].float() * coefficient
        denominator = (width + 1 if dynamic_input is None else 2 * width + 1) ** 0.5
        results.append(out / denominator)
    return torch.stack(results, 1).to(x.dtype) * valid[..., None]


def clip(x: torch.Tensor, scale: torch.Tensor | None, threshold: float) -> torch.Tensor:
    """Reference clipping equation.

    Args:
        x: Tensor of shape [batch, sequence, heads, head_dim].
        scale: Optional tensor of shape [heads, head_dim].
        threshold: Maximum L2 length before optional length scaling.

    Returns:
        Tensor with x's shape.
    """
    value = x.float() if scale is None else x.float() * scale.float()
    denominator = torch.maximum(
        (value.square().sum(-1, keepdim=True).sqrt() + 1e-6) / threshold,
        torch.ones_like(value[..., :1]),
    )
    return (value / denominator).to(x.dtype)


def attention(
    cfg: Inkling2TextConfig,
    weights: Mapping[str, torch.Tensor],
    prefix: str,
    sources: list[torch.Tensor],
    *,
    local: bool,
    valid: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Per-query/head attention using the explicit LSE gate/sink equation.

    Args:
        cfg: Text configuration.
        weights: Named parameter tensors.
        prefix: Attention module name.
        sources: Three tensors of shape [batch, sequence, hidden].
        local: Whether to apply sliding-window attention.
        valid: Boolean tensor of shape [batch, sequence].
        positions: Integer tensor of shape [batch, sequence].

    Returns:
        Tensor of shape [batch, sequence, hidden].
    """
    heads = cfg.swa_num_attention_heads if local else cfg.num_attention_heads
    kv_heads = cfg.swa_num_key_value_heads if local else cfg.num_key_value_heads
    dim = cfg.swa_head_dim if local else cfg.head_dim
    extent = cfg.sliding_window_size if local else cfg.rel_extent
    sizes = [
        heads * dim,
        kv_heads * dim,
        kv_heads * dim,
        heads * cfg.d_rel,
        heads * dim,
        heads,
    ]
    projections = []
    start = 0
    for index, size in enumerate(sizes):
        source = sources[1] if index == 1 else sources[2] if index == 2 else sources[0]
        bias = weights.get(prefix + ".qkvr.bias")
        projections.append(
            F.linear(
                source,
                weights[prefix + ".qkvr.weight"][start : start + size],
                None if bias is None else bias[start : start + size],
            )
        )
        start += size
    q, k, v, r, gate, sink = projections
    k = convolution(
        k,
        weights,
        prefix + ".k_sconv",
        width=cfg.sconv_kernel_size,
        valid=valid,
        dynamic_input=None if local else sources[1],
        heads=kv_heads,
    )
    v = convolution(
        v,
        weights,
        prefix + ".v_sconv",
        width=cfg.sconv_kernel_size,
        valid=valid,
        dynamic_input=None if local else sources[2],
        heads=kv_heads,
    )
    batch, length, _ = q.shape
    q = clip(
        q.reshape(batch, length, heads, dim),
        weights[prefix + ".q_l2_clip.scale"],
        cfg.qk_l2_clip_threshold,
    )
    k = clip(k.reshape(batch, length, kv_heads, dim), None, cfg.qk_l2_clip_threshold)
    r = clip(
        r.reshape(batch, length, heads, cfg.d_rel),
        weights[prefix + ".rel_l2_clip.scale"],
        cfg.rel_l2_clip_threshold,
    )
    if not local:
        tau = 1 + cfg.log_scaling_alpha * torch.log(
            torch.maximum(
                (positions.float() + 1) / cfg.log_scaling_n_floor,
                torch.ones_like(positions, dtype=torch.float32),
            )
        )
        q = (q.float() * tau[..., None, None]).to(q.dtype)
        r = (r.float() * tau[..., None, None]).to(r.dtype)
    v = v.reshape(batch, length, kv_heads, dim)
    if cfg.use_value_norm_no_gamma:
        v = rms(v, None, cfg.rms_norm_eps).to(v.dtype)
    batches = []
    for b in range(batch):
        tokens = []
        for t in range(length):
            head_values = []
            for h in range(heads):
                kv = h // (heads // kv_heads)
                keys = [
                    s
                    for s in range(t + 1)
                    if valid[b, s]
                    and (not local or 0 <= int(positions[b, t] - positions[b, s]) < cfg.sliding_window_size)
                ]
                if not valid[b, t] or not keys:
                    head_values.append(torch.zeros_like(q[b, t, h]))
                    continue
                scores = []
                for s in keys:
                    distance = int(positions[b, t] - positions[b, s])
                    relative = (
                        (r[b, t, h].float() * weights[prefix + ".relative_table"][:, distance].float())
                        .sum()
                        .to(r.dtype)
                        .float()
                        if (0 <= distance < extent)
                        else q.new_zeros(())
                    )
                    scores.append((q[b, t, h].float() * k[b, s, kv].float()).sum() / dim**0.5 + relative)
                scores = torch.stack(scores)
                context = (scores.softmax(0)[:, None] * v[b, keys, kv].float()).sum(0)
                context = context * torch.sigmoid(gate[b, t].float().reshape(heads, dim)[h])
                head_values.append(context * torch.sigmoid(torch.logsumexp(scores, 0) - sink[b, t, h].float()))
            tokens.append(torch.cat(head_values))
        batches.append(torch.stack(tokens))
    return (
        F.linear(
            torch.stack(batches).to(sources[0].dtype),
            weights[prefix + ".wo_ud.weight"],
            weights.get(prefix + ".wo_ud.bias"),
        )
        * valid[..., None]
    )


def experts(
    cfg: Inkling2TextConfig,
    weights: Mapping[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
    routing_ids: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Per-token/top-k expert computation with explicit gate and up branches.

    Args:
        cfg: Text configuration.
        weights: Named parameters, including [experts, hidden, 2 * intermediate] interleaved weights.
        prefix: MoE module name.
        x: Tensor of shape [batch, sequence, hidden].
        routing_ids: Optional collector for this layer's [batch * sequence, top-k] expert IDs.

    Returns:
        Tensor of shape [batch, sequence, hidden].
    """
    outputs, selected = [], []
    for token in x.reshape(-1, x.shape[-1]):
        scores = torch.sigmoid(F.linear(token.float(), weights[prefix + ".gate.weight"].float()))
        choice = scores + weights.get(prefix + ".gate.e_score_correction_bias", torch.zeros_like(scores))
        ids = torch.argsort(choice, descending=True)[: cfg.num_experts_per_tok]
        selected.append(ids)
        gamma = cfg.route_scale * scores[ids] / scores[ids].sum()
        result = torch.zeros_like(token, dtype=torch.float32)
        for slot, expert in enumerate(ids):
            fused = weights[prefix + ".experts.gate_and_up_projs"][expert]
            gate = (token @ fused[:, 0::2]).float()
            up = (token @ fused[:, 1::2]).float()
            activation = cfg.mlp_gate_softcap * torch.tanh(gate / cfg.mlp_gate_softcap) * torch.sigmoid(gate)
            activation = activation * cfg.mlp_up_softcap * torch.tanh(up / cfg.mlp_up_softcap)
            down = (gamma[slot] * activation).to(token.dtype) @ weights[prefix + ".experts.down_projs"][expert]
            result = result + down.float()
        outputs.append(result.to(token.dtype))
    if routing_ids is not None:
        routing_ids.append(torch.stack(selected))
    return torch.stack(outputs).reshape_as(x)


def forward(
    cfg: Inkling2TextConfig,
    weights: Mapping[str, torch.Tensor],
    input_ids: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> ReferenceOutput:
    """Functional decoder oracle with separate depth-state transitions.

    Args:
        cfg: Text configuration.
        weights: Named parameters/buffers with FP32-sensitive groups, independently differentiable if required.
        input_ids: Integer tensor of shape [batch, sequence].
        valid: Optional boolean tensor of shape [batch, sequence].
        positions: Optional integer tensor of shape [batch, sequence].

    Returns:
        ReferenceOutput containing [batch, sequence, vocab] logits and one
        [batch, sequence, hidden] partial block per layer.
    """
    valid = torch.ones_like(input_ids, dtype=torch.bool) if valid is None else valid
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0) if positions is None else positions
    dtype = weights["embed_tokens.weight"].dtype
    current = rms(
        F.embedding(input_ids, weights["embed_tokens.weight"]),
        weights["embed_norm.weight"],
        cfg.rms_norm_eps,
    )
    current = current.to(dtype) * valid[..., None]
    completed = []
    partials = []
    routing_ids = []
    for i in range(cfg.num_hidden_layers):
        prefix = f"layers.{i}"
        local = i in cfg.local_layer_ids
        count = 1 if local else 3
        inputs = (
            [current] * count
            if i == 0
            else [
                mix(completed + [current], query, cfg.attn_res_normalize_values)
                for query in weights[prefix + ".attn_query.weight"]
            ]
        )
        inputs = [
            rms(value, weights[f"{prefix}.attn_norms.{j}.weight"], cfg.rms_norm_eps).to(dtype)
            for j, value in enumerate(inputs)
        ]
        attn = attention(
            cfg,
            weights,
            prefix + ".attn",
            inputs * 3 if local else inputs,
            local=local,
            valid=valid,
            positions=positions,
        )
        attn = convolution(
            attn,
            weights,
            prefix + ".attn_sconv",
            width=cfg.sconv_kernel_size,
            valid=valid,
        )
        if i % cfg.n_layers_per_attnres_block == 0:
            completed.append(current)
            current = attn.float()
        else:
            current = current.float() + attn.float()
        mlp_input = mix(
            completed + [current],
            weights[prefix + ".mlp_query.weight"][0],
            cfg.attn_res_normalize_values,
        )
        mlp_input = rms(mlp_input, weights[prefix + ".mlp_norm.weight"], cfg.rms_norm_eps).to(dtype)
        out = experts(cfg, weights, prefix + ".mlp", mlp_input, routing_ids)
        current = (
            current
            + convolution(
                out,
                weights,
                prefix + ".mlp_sconv",
                width=cfg.sconv_kernel_size,
                valid=valid,
            ).float()
        )
        current = current.to(dtype)
        partials.append(current)
    hidden = rms(
        mix(
            completed + [current],
            weights["output_query.weight"][0],
            cfg.attn_res_normalize_values,
        ),
        weights["norm.weight"],
        cfg.rms_norm_eps,
    )
    logits = F.linear(hidden.to(dtype).float(), weights["lm_head.weight"].float())
    logits = cfg.final_logit_softcapping * torch.tanh(logits / cfg.final_logit_softcapping)
    logits = logits.masked_fill(
        torch.arange(cfg.vocab_size, device=logits.device) >= cfg.unpadded_vocab_size,
        float("-inf"),
    )
    return ReferenceOutput(logits, partials, routing_ids)

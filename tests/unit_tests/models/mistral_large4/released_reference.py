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

"""Bounded independent equations for released-weight text comparisons.

This oracle uses source tensor names and complex rotary arithmetic. Attention is
explicit matrix softmax, never production SDPA. Tiny tests cross-check it against
the original token/head oracle before it is used for released configurations.
"""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from tests.unit_tests.models.mistral_large4.math_reference import rms, rotate


@dataclass
class ReferenceBlock:
    """Independent outputs: hidden [tokens, hidden], KV and discrete selections."""

    hidden: torch.Tensor
    kv: tuple[torch.Tensor, torch.Tensor]
    indices: torch.Tensor
    experts: torch.Tensor | None


def reference_block(
    state: dict[str, torch.Tensor],
    x: torch.Tensor,
    positions: torch.Tensor,
    c: MistralLarge4Config,
    *,
    layer: int,
    shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    indices: torch.Tensor | None = None,
) -> ReferenceBlock:
    """Evaluate one original layer independently, preserving source identities.

    Args:
        state: Source-key weights [output, input], norms/biases [channels], and
            optional learned rotary scale [rotary_dim / 2], independently loaded.
        x: Floating [tokens, hidden] for one unpadded document.
        positions: Integer [tokens] rotary positions; storage order is causal.
        c: Original architecture, including full-depth sharing schedules.
        layer: Original zero-based layer identity.
        shared_kv: Optional latent [tokens, latent_dim] and rotated keys
            [tokens, rotary_dim], or [tokens, 0] for NoPE, owned by this oracle.
        indices: Previous selector's integer [tokens, min(topk, tokens)] with
            -1 padding, used only for IndexShare layers.

    Returns:
        Hidden [tokens, hidden], KV pair in the layouts above, selected indices,
        and optional routed experts [tokens, experts_per_token]. KV and indices
        may alias this oracle's inputs; neither production state nor its helpers
        participate in this calculation.
    """
    n = x.shape[0]
    p = f"layers.{layer}."
    a = p + "attention."
    alpha = state.get("lerope.alpha")
    frequency_mode = "float32" if x.dtype == torch.bfloat16 else "scalar"
    y = rms(x, state[p + "attention_norm.weight"], c.rms_norm_eps)
    q = F.linear(y, state[a + "wq.weight"]).reshape(n, c.num_attention_heads, c.head_dim)
    rope = not c.nope_layer_flags[layer]
    nope = c.qk_nope_head_dim if rope else c.head_dim
    if c.kv_source_layers[layer] == layer:
        raw = F.linear(y, state[a + "wkv_a_with_mqa.weight"])
        latent = rms(raw[:, : c.kv_lora_rank], state[a + "kv_a_norm.weight"], c.rms_norm_eps)
        pe = (
            rotate(raw[:, c.kv_lora_rank :], positions, alpha, c, frequency_mode=frequency_mode)
            if rope
            else x.new_empty((n, 0))
        )
    else:
        if shared_kv is None:
            raise ValueError(f"Reference layer {layer} needs its source KV")
        latent, pe = shared_kv
    if rope:
        q = torch.cat((q[..., :nope], rotate(q[..., nope:], positions, alpha, c, frequency_mode=frequency_mode)), -1)
    if c.ssmax_scaling:
        q = (
            q
            * (
                (
                    (positions.float() + 1).log().to(q.dtype)[:, None] * state[a + "ssmax_scaling.s"]
                    + state[a + "ssmax_scaling.b"]
                )[..., None]
            )
        )
    if c.index_sharing[layer] == "full":
        with torch.no_grad():
            ip = a + "indexer."
            iq = F.layer_norm(
                y,
                (c.hidden_size,),
                state[ip + "q_input_norm.weight"],
                state[ip + "q_input_norm.bias"],
                c.indexer_norm_eps,
            )
            iq = F.linear(iq, state[ip + "wq_b.weight"]).reshape(n, c.index_n_heads, c.index_head_dim)
            ik = F.linear(y, state[ip + "wk.weight"])
            ik = F.layer_norm(
                ik.float(),
                (c.index_head_dim,),
                state[ip + "k_norm.weight"].float(),
                state[ip + "k_norm.bias"].float(),
                c.indexer_norm_eps,
            ).to(y.dtype)
            iw = F.linear(y, state[ip + "weights_proj.weight"]).float() / math.sqrt(c.index_head_dim * c.index_n_heads)
            if c.indexer_ssmax_scaling:
                iq = (
                    iq
                    * (
                        (
                            (positions.float() + 1).log().to(iq.dtype)[:, None] * state[ip + "ssmax_scaling.s"]
                            + state[ip + "ssmax_scaling.b"]
                        )[..., None]
                    )
                )
            if not c.indexer_nope_layer_flags[layer]:
                d = c.qk_rope_head_dim
                iq = torch.cat(
                    (rotate(iq[..., :d], positions, alpha, c, frequency_mode=frequency_mode), iq[..., d:]), -1
                )
                ik = torch.cat(
                    (rotate(ik[..., :d], positions, alpha, c, frequency_mode=frequency_mode), ik[..., d:]), -1
                )
            rows = []
            keys = torch.arange(n, device=x.device)
            # Sum explicit head contributions; the production selector uses an
            # einsum/head reduction. This avoids a shared implementation path.
            for start in range(0, n, 64):
                stop = min(start + 64, n)
                scores = sum(
                    (iq[start:stop, h].float() @ ik.float().T).relu() * iw[start:stop, h, None]
                    for h in range(c.index_n_heads)
                )
                scores = scores.masked_fill(
                    keys[None] > torch.arange(start, stop, device=x.device)[:, None], -torch.inf
                )
                order = scores.argsort(dim=-1, descending=True, stable=True)[:, : min(c.index_topk, n)]
                rows.append(order.masked_fill(~scores.gather(1, order).isfinite(), -1))
            indices = torch.cat(rows)
    if indices is None:
        raise ValueError("Reference IndexShare has no preceding selection")
    kv = F.linear(latent, state[a + "wkv_b.weight"]).reshape(n, c.num_attention_heads, nope + c.v_head_dim)
    k, v = kv[..., :nope], kv[..., nope:]
    if rope:
        k = torch.cat((k, pe[:, None].expand(-1, c.num_attention_heads, -1)), -1)
    q, k, v = q.float(), k.float(), v.float()
    attention = []
    for start in range(0, n, 32):
        stop = min(start + 32, n)
        chosen = indices[start:stop]
        selected_k = k[chosen.clamp_min(0)].permute(0, 2, 1, 3)
        selected_v = v[chosen.clamp_min(0)].permute(0, 2, 1, 3)
        scores = (selected_k * q[start:stop, :, None]).sum(-1) / math.sqrt(c.head_dim)
        probabilities = scores.masked_fill(chosen[:, None] < 0, -torch.inf).softmax(-1)
        attention.append((probabilities[..., None] * selected_v).sum(-2).flatten(-2))
    x = x + F.linear(torch.cat(attention).to(y.dtype), state[a + "wo.weight"])
    y = rms(x, state[p + "ffn_norm.weight"], c.rms_norm_eps)

    def mlp(prefix: str) -> torch.Tensor:
        """Return [tokens, hidden] from independent source SwiGLU weights."""
        return F.linear(
            F.silu(F.linear(y, state[prefix + "w1.weight"])) * F.linear(y, state[prefix + "w3.weight"]),
            state[prefix + "w2.weight"],
        )

    experts = None
    if layer < c.first_k_dense_replace or layer % c.moe_layer_freq:
        update = mlp(p + "feed_forward.")
    else:
        probabilities = F.linear(y.float(), state[p + "gate.weight"].float()).softmax(-1)
        weights, experts = probabilities.topk(c.num_experts_per_tok)
        weights = weights / weights.sum(-1, keepdim=True) * c.routed_scaling_factor
        # Full matrix contractions preserve the BF16 gradient accumulation boundary.
        # Output selection is mathematically cross-checked by the token oracle.
        selected = torch.zeros(n, c.num_experts_per_tok, c.hidden_size, device=y.device, dtype=torch.float32)
        for expert in range(c.n_routed_experts):
            value = mlp(p + f"experts.{expert}.").float()
            selected = selected + value[:, None] * (experts == expert)[..., None]
        update = (selected * weights[..., None]).sum(1).to(y.dtype)
        if c.n_shared_experts:
            update = update + mlp(p + "shared_experts.")
    return ReferenceBlock(x + update, (latent, pe), indices, experts)

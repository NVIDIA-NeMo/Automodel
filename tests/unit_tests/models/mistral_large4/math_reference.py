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

"""Independent scalar/token oracle from the pinned vendor architecture equations.

Uses released parameter names, complex pair rotations, token-wise sparse attention,
and selected-expert dispatch. It does not invoke production model math/helpers.
This is a mathematical oracle, not an official HF implementation or vLLM runtime.
"""

import math
from typing import Literal

import torch
from torch.nn import functional as F

from nemo_automodel.components.models.mistral_large4.config import ConfigValue, MistralLarge4Config


def tiny_config(**overrides: ConfigValue | torch.dtype) -> MistralLarge4Config:
    values = dict(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        vocab_size=31,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=3,
        n_routed_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        dtype="float32",
        nope_layer_flags=[False, False, False, True],
        kv_source_layers=[0, 1, 2, 3],
        index_sharing=["full", "full", "shared", "full"],
    )
    values.update(overrides)
    return MistralLarge4Config(**values)


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Independent RMS normalization.

    Args:
        x: Floating [..., hidden], arbitrary leading axes.
        weight: Floating [hidden] scale.
        eps: Variance epsilon.

    Returns:
        Independent normalized tensor with x's shape and dtype.
    """
    # Share one FP32 node so numerator and variance gradients combine before
    # returning to BF16. Separate x.float() calls round each branch gradient
    # independently and do not model an FP32 normalization backward.
    x_float = x.float()
    if x.dtype == torch.bfloat16:
        # Preserve native FP32 reduction/rounding before the BF16 cast. PyTorch
        # autograd remains independent of AutoModel's custom analytic backward.
        # Explicit scalar equations remain the FP32 oracle; their different
        # rounding can cross BF16 boundaries at the released hidden width.
        return F.rms_norm(x_float, (x.shape[-1],), weight.float(), eps).to(x.dtype)
    return (x_float / torch.sqrt((x_float**2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


def rotate(
    x: torch.Tensor,
    positions: torch.Tensor,
    alpha: torch.Tensor | None,
    c: MistralLarge4Config,
    *,
    frequency_mode: Literal["scalar", "float32"] = "scalar",
) -> torch.Tensor:
    """Complex-number rotary oracle with scalar frequency/ramp construction.

    Args:
        x: Floating [tokens, heads, rotary_dim] or [tokens, rotary_dim].
        positions: Integer [tokens].
        alpha: Optional FP32 [rotary_dim / 2] learned log-frequency scale.
        c: Architecture configuration.
        frequency_mode: Scalar high-precision frequency oracle, or explicit
            FP32 frequency operations for the vendor's BF16 rounding contract.
            Rotation remains independent complex multiplication in both modes.

    Returns:
        Independent rotated tensor, same shape and dtype as x.
    """
    d = c.qk_rope_head_dim
    lo = max(
        0,
        math.floor(
            d
            * math.log(c.yarn_original_max_position_embeddings / (2 * math.pi * c.yarn_beta_fast))
            / (2 * math.log(c.rope_theta))
        ),
    )
    hi = min(
        d - 1,
        math.ceil(
            d
            * math.log(c.yarn_original_max_position_embeddings / (2 * math.pi * c.yarn_beta_slow))
            / (2 * math.log(c.rope_theta))
        ),
    )
    if frequency_mode == "scalar":
        frequencies = []
        for i in range(d // 2):
            ramp = min(1, max(0, (i - lo) / (hi - lo if hi != lo else 0.001)))
            base = c.rope_theta ** (-2 * i / d)
            frequencies.append(base * ((1 - ramp) + ramp / c.yarn_factor))
        frequency = torch.tensor(frequencies, device=x.device, dtype=torch.float32)
        if alpha is not None:
            frequency = frequency * alpha.exp()
    else:
        # Match documented FP32 frequency rounding before BF16 rotations. At
        # large positions a one-ULP frequency change crosses BF16 rounding bins
        # and can strongly perturb a cancellation-sensitive LeRoPE gradient.
        channel = torch.arange(d // 2, device=x.device, dtype=torch.float32)
        ramp = ((channel - lo) / (hi - lo if hi != lo else 0.001)).clamp(0, 1)
        wavelength = torch.pow(c.rope_theta, 2 * channel / d)
        scale = 1.0 if alpha is None else alpha.exp()
        frequency = (1 / (c.yarn_factor * wavelength) * scale) * ramp + (1 / wavelength * scale) * (1 - ramp)
    phase = positions.float()[:, None] * frequency
    multiplier = torch.polar(torch.ones_like(phase), phase)
    if x.ndim == 3:
        multiplier = multiplier[:, None]
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(complex_x * multiplier).flatten(-2).to(x.dtype)


def reference_logits(
    state: dict[str, torch.Tensor],
    ids: torch.Tensor,
    c: MistralLarge4Config,
    positions: torch.Tensor | None = None,
    *,
    inputs_embeds: torch.Tensor | None = None,
    expert_batching: Literal["token", "grouped"] = "token",
) -> torch.Tensor:
    """One-document oracle with released names and explicit sparse attention.

    Args:
        state: Released-key tensors, matrices [output, input], norm/bias [channels].
            Independent leaf tensors may require gradients.
        ids: Integer [tokens] input IDs for one document.
        c: Architecture configuration.
        positions: Optional integer [tokens] document positions.
        inputs_embeds: Optional floating [tokens, hidden] fused image/text embeddings.
        expert_batching: Token-wise selected-expert evaluation, which rounds each
            contribution, or grouped batching, which concatenates gate/up weights and
            selects tokens before each expert GEMM, matching shared MoE BF16
            accumulation. Both independently implement the same FP32 equations.

    Returns:
        Capped FP32 logits [tokens, vocab], preserving a differentiable graph.
    """
    n = ids.numel()
    positions = torch.arange(n) if positions is None else positions
    alpha = state.get("lerope.alpha")
    frequency_mode = "float32" if state["tok_embeddings.weight"].dtype == torch.bfloat16 else "scalar"
    x = F.embedding(ids, state["tok_embeddings.weight"]) if inputs_embeds is None else inputs_embeds
    caches = {}
    selected = None
    for layer in range(c.num_hidden_layers):
        p = f"layers.{layer}."
        a = p + "attention."
        y = rms(x, state[p + "attention_norm.weight"], c.rms_norm_eps)
        q = F.linear(y, state[a + "wq.weight"]).reshape(n, c.num_attention_heads, c.head_dim)
        rope = not c.nope_layer_flags[layer]
        nope = c.qk_nope_head_dim if rope else c.head_dim
        source = c.kv_source_layers[layer]
        if source == layer:
            raw = F.linear(y, state[a + "wkv_a_with_mqa.weight"])
            latent = rms(raw[:, : c.kv_lora_rank], state[a + "kv_a_norm.weight"], c.rms_norm_eps)
            pe = rotate(raw[:, c.kv_lora_rank :], positions, alpha, c, frequency_mode=frequency_mode) if rope else None
            caches[layer] = (latent, pe)
        else:
            latent, pe = caches[source]
        if rope:
            q = torch.cat(
                (q[..., :nope], rotate(q[..., nope:], positions, alpha, c, frequency_mode=frequency_mode)), -1
            )
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
                    1e-6,
                )
                iq = F.linear(iq, state[ip + "wq_b.weight"]).reshape(n, c.index_n_heads, c.index_head_dim)
                ik = F.linear(y, state[ip + "wk.weight"])
                ik = F.layer_norm(
                    ik.float(),
                    (c.index_head_dim,),
                    state[ip + "k_norm.weight"].float(),
                    state[ip + "k_norm.bias"].float(),
                    1e-6,
                ).to(y.dtype)
                iw = F.linear(y, state[ip + "weights_proj.weight"]).float() / math.sqrt(
                    c.index_head_dim * c.index_n_heads
                )
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
                selected = []
                for t in range(n):
                    score = sum(
                        F.relu(ik[: t + 1].float() @ iq[t, h].float()) * iw[t, h] for h in range(c.index_n_heads)
                    )
                    selected.append(torch.argsort(score, descending=True, stable=True)[: c.index_topk])
        # Explicit projection followed by token/head attention: no production SDPA or masks.
        kv = F.linear(latent, state[a + "wkv_b.weight"]).reshape(n, c.num_attention_heads, nope + c.v_head_dim)
        k, v = kv[..., :nope], kv[..., nope:]
        if rope:
            k = torch.cat((k, pe[:, None, :].expand(-1, c.num_attention_heads, -1)), -1)
        # Cast before token selection: the FP32 attention operation must also
        # accumulate its K/V gradients in FP32 before returning to BF16 projections.
        q, k, v = q.float(), k.float(), v.float()
        outputs = []
        for t in range(n):
            heads = []
            for h in range(c.num_attention_heads):
                score = k[selected[t], h].float() @ q[t, h].float() / math.sqrt(c.head_dim)
                heads.append(score.softmax(0) @ v[selected[t], h].float())
            outputs.append(torch.cat(heads))
        x = x + F.linear(torch.stack(outputs).to(y.dtype), state[a + "wo.weight"])
        y = rms(x, state[p + "ffn_norm.weight"], c.rms_norm_eps)

        def mlp(z: torch.Tensor, prefix: str) -> torch.Tensor:
            """Apply a source SwiGLU.

            Args:
                z: Floating [tokens, hidden] activations.
                prefix: Released parameter namespace.

            Returns:
                Independent floating [tokens, hidden] activations.
            """
            return F.linear(
                F.silu(F.linear(z, state[prefix + "w1.weight"])) * F.linear(z, state[prefix + "w3.weight"]),
                state[prefix + "w2.weight"],
            )

        if layer < c.first_k_dense_replace:
            x = x + mlp(y, p + "feed_forward.")
        else:
            scores = F.linear(y.float(), state[p + "gate.weight"].float()).softmax(-1)
            weights, experts = scores.topk(c.num_experts_per_tok)
            weights = weights / weights.sum(-1, keepdim=True) * c.routed_scaling_factor
            if expert_batching == "grouped":
                # Match one gate/up GEMM per selected expert, including the
                # single BF16 cast of its input gradient. Use only source
                # matrices and primitive ops; no production MoE/dispatch code.
                slots = y.new_zeros((n * c.num_experts_per_tok, y.shape[-1]), dtype=torch.float32)
                for expert in range(c.n_routed_experts):
                    rows, choices = torch.where(experts == expert)
                    prefix = p + f"experts.{expert}."
                    projection = torch.cat((state[prefix + "w1.weight"], state[prefix + "w3.weight"]), dim=0)
                    gate, up = F.linear(y[rows], projection).chunk(2, dim=-1)
                    output = F.linear(F.silu(gate) * up, state[prefix + "w2.weight"])
                    slots = slots.index_add(
                        0, rows * c.num_experts_per_tok + choices, output.float() * weights[rows, choices, None]
                    )
                update = slots.view(n, c.num_experts_per_tok, -1).sum(1).to(y.dtype)
            else:
                routed = []
                for t in range(n):
                    value = sum(
                        mlp(y[t : t + 1], p + f"experts.{int(experts[t, j])}.").float() * weights[t, j]
                        for j in range(c.num_experts_per_tok)
                    )
                    routed.append(value)
                update = torch.cat(routed).to(y.dtype)
            if c.n_shared_experts:
                update = update + mlp(y, p + "shared_experts.")
            x = x + update
    x = rms(x, state["norm.weight"], c.rms_norm_eps)
    logits = F.linear(x, state["output.weight"]).float()
    return c.final_logit_softcapping * torch.tanh(logits / c.final_logit_softcapping)

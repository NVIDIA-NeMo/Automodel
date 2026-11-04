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

"""Test-owned decoder math; no production attention, routing, or expert helpers."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn


def tiny_config(dtype: torch.dtype = torch.float32) -> FlintConfig:
    """Return asymmetric dense/full, MoE/sliding, and MoE/full tiny settings."""
    return FlintConfig(
        vocab_size=31,
        hidden_size=12,
        intermediate_size=20,
        moe_intermediate_size=5,
        shared_expert_intermediate_size=7,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0],
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        sliding_window=3,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        router_aux_loss_coef=0.0,
        pad_token_id=2,
        eos_token_id=[2, 3, 4],
        tie_word_embeddings=False,
        dtype=dtype,
    )


def torch_backend() -> BackendConfig:
    """Use actual eager/torch math with the learned input-dtype router."""
    return BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        rope="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        gate_precision=None,
        enable_fsdp_optimizations=False,
    )


def full_native_tensors(model: torch.nn.Module, *, gradients: bool = False) -> dict[str, torch.Tensor]:
    """Materialize canonical native model tensors in collective parameter order.

    Args:
        model: Native model with ordinary parameters or FSDP DTensor shards.
            Global native matrices are [out,in], fused experts [experts,hidden,2*intermediate]
            and [experts,intermediate,hidden], and norms/sinks are vectors.
        gradients: Read matching .grad tensors instead of parameters. Every
            trainable parameter must have a gradient, including zero rows.
            Frozen parameters are excluded only when reading gradients.

    Returns:
        Detached independent tensors in the same global shapes/dtypes/devices.
        Sharded DTensors are gathered by every rank in the same order; callers
        must all enter this function. Names omit activation-checkpoint wrappers.
    """
    result = {}
    for name, parameter in model.named_parameters():
        if gradients and not parameter.requires_grad:
            continue
        tensor = parameter.grad if gradients else parameter
        if tensor is None:
            raise AssertionError(f"Missing gradient: {name}")
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        result[canonical_parameter_fqn(name)] = tensor.detach().clone()
    return result


def rms(values: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute reference RMS normalization with FP32 accumulation.

    Args:
        values: FP32/BF16 tensor [...,hidden], arbitrary leading axes.
        scale: Trainable scale [hidden] on values' device, BF16 or FP32.
        eps: Positive denominator epsilon.

    Returns:
        New tensor [...,hidden] in values' dtype with gradients to both inputs.
    """
    x = values.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * scale.float()).to(values.dtype)


class _AnalyticSwiGLU(torch.autograd.Function):
    """Independent routed-activation math with one low-precision adjoint cast."""

    @staticmethod
    def forward(ctx: Any, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU in FP32 before rounding the activation once.

        Args:
            ctx: Autograd context owning saved tensors for this invocation.
            gate: BF16/FP32 tensor [selected_tokens,intermediate] on CPU/CUDA,
                potentially a noncontiguous half of a fused gate/up projection.
            up: Same shape/dtype/device as gate, representing its other half.

        Returns:
            Independent [selected_tokens,intermediate] tensor in gate's dtype.
            Original inputs are retained without mutation for the adjoint.
        """
        ctx.save_for_backward(gate, up)
        a, b = gate.float(), up.float()
        return (a * a.sigmoid() * b).to(gate.dtype)

    @staticmethod
    def backward(ctx: Any, upstream: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the analytical FP32 activation adjoint before one cast.

        Args:
            ctx: Context holding original gate/up [selected_tokens,intermediate]
                inputs in BF16/FP32 on CPU/CUDA.
            upstream: Output adjoint [selected_tokens,intermediate] in the
                output's dtype/device. FP32 route-weight ones promote this
                adjoint in the supported compiled routed-expert contract.

        Returns:
            Independent gate/up adjoints [selected_tokens,intermediate],
            evaluated in FP32 then rounded once to each input's dtype/device.
            No production activation or backward helper is used.
        """
        gate, up = ctx.saved_tensors
        a, b, g = gate.float(), up.float(), upstream.float()
        sigmoid = a.sigmoid()
        gate_grad = g * b * sigmoid * (1 + a * (1 - sigmoid))
        up_grad = g * a * sigmoid
        return gate_grad.to(gate.dtype), up_grad.to(up.dtype)


def rope(values: torch.Tensor, theta: float) -> torch.Tensor:
    """Apply independent NeoX half-channel pairing with FP32 frequencies.

    Args:
        values: FP32/BF16 [batch,sequence,heads,head_dim] tensor with even head_dim.
        theta: Positive RoPE frequency base.

    Returns:
        New tensor with the same shape/device/dtype and preserved autograd history.
    """
    half = values.shape[-1] // 2
    exponent = torch.arange(half, device=values.device, dtype=torch.float32) / half
    angle = torch.arange(values.shape[1], device=values.device, dtype=torch.float32)[:, None] / theta**exponent
    cosine, sine = angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]
    first, second = values.float()[..., :half], values.float()[..., half:]
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1).to(values.dtype)


def reference_logits(
    weights: dict[str, torch.Tensor],
    config: FlintConfig,
    batch: dict[str, torch.Tensor],
    *,
    compute_dtype: torch.dtype = torch.float32,
    parameter_dtype: torch.dtype | None = None,
    routed_precision_contract: bool = False,
) -> tuple[torch.Tensor, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Evaluate independent full-LM math with explicit projection rounding.

    Args:
        weights: Independent BF16/FP32 leaves in expanded HF [out,in] layout,
            on one CPU/CUDA device. Embedding/head [vocab,hidden], Q/K/V
            [query_heads*head_dim,hidden]/[kv_heads*head_dim,hidden], output
            [hidden,query_heads*head_dim], norms [hidden] or [head_dim], sinks
            [query_heads]. Each MLP has [intermediate,hidden] gate/up and
            [hidden,intermediate] down matrices, with independent shared width.
        config: Tiny validated architecture and layer schedules.
        batch: int64 input_ids and boolean/int64 binary attention_mask [batch,sequence]
            on the weights' device. Any extra fields are ignored.
        compute_dtype: FP32 or BF16 projection arithmetic. Embedding/residual
            storage follows weight dtype; sink attention and head math stay FP32.
        parameter_dtype: Optional FSDP bulk parameter compute dtype. Cast bulk
            embedding/norm/projection weights to this dtype while preserving
            the sink/head FP32 units. Original leaves keep their storage dtype.
        routed_precision_contract: For strict BF16 update comparisons, gather
            selected token rows before one fused gate/up GEMM and use an
            independently derived FP32 SwiGLU activation/adjoint rounded once
            to BF16. Weight expert outputs in FP32 and reduce top-k slots.
            The ordinary separate-projection/eager-autograd oracle remains a
            separate forward/every-gradient/norm gate at the original tolerance.

    Returns:
        FP32 logits [batch,sequence,vocab] and per-MoE-layer (weights,IDs), both
        [batch*sequence,top_k], with FP32 weights and int64 expert IDs. Outputs
        retain autograd history and do not alias/mutate inputs. Padded-query
        logits/routes are unspecified and excluded from numerical comparisons.
        This mathematical reference does not reproduce serving-kernel rounding.
    """
    if parameter_dtype is not None:
        weights = {
            name: tensor if name == "lm_head.weight" or name.endswith("softmax_offset") else tensor.to(parameter_dtype)
            for name, tensor in weights.items()
        }
    x = F.embedding(batch["input_ids"], weights["model.embed_tokens.weight"])
    b, s, h = x.shape
    qh, kh, d = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    positions = torch.arange(s, device=x.device)
    distances = positions[:, None] - positions[None, :]
    valid = batch["attention_mask"].bool()
    routes = {}

    def project(values: torch.Tensor, name: str) -> torch.Tensor:
        """Project [...,in] values with [out,in] weights on their shared device.

        Args:
            values: BF16/FP32 activations [...,in], with arbitrary leading axes.
            name: Key of the enclosing weights mapping's [out,in] matrix.

        Returns:
            Independent [...,out] tensor in compute_dtype with autograd history.
        """
        return F.linear(values.to(compute_dtype), weights[name].to(compute_dtype))

    def mlp(values: torch.Tensor, name: str) -> torch.Tensor:
        """Apply named reference SwiGLU without a production MLP helper.

        Args:
            values: BF16/FP32 activations [...,hidden] with arbitrary leading axes.
            name: Prefix of gate/up [intermediate,hidden] and down [hidden,intermediate].

        Returns:
            New tensor [...,hidden] in compute_dtype on the input device.
        """
        gate, up = project(values, name + ".gate_proj.weight"), project(values, name + ".up_proj.weight")
        return project(F.silu(gate) * up, name + ".down_proj.weight")

    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        attn = prefix + ".self_attn"
        normalized = rms(x, weights[prefix + ".input_layernorm.weight"], config.rms_norm_eps)
        q = project(normalized, attn + ".q_proj.weight").reshape(b, s, qh, d)
        k = project(normalized, attn + ".k_proj.weight").reshape(b, s, kh, d)
        v = project(normalized, attn + ".v_proj.weight").reshape(b, s, kh, d)
        q, k = (
            rms(q, weights[attn + ".q_norm.weight"], config.rms_norm_eps),
            rms(k, weights[attn + ".k_norm.weight"], config.rms_norm_eps),
        )
        if index not in config.no_rope_layer_indices:
            q, k = (
                rope(q, float(config.rope_parameters["rope_theta"])),
                rope(k, float(config.rope_parameters["rope_theta"])),
            )
        allowed = distances >= 0
        if config.layer_types[index] == "sliding_attention":
            allowed &= distances < config.sliding_window
        allowed = allowed[None] & valid[:, None, :] & valid[:, :, None]
        heads = []
        for head in range(qh):
            kv = head // (qh // kh)
            scores = (q[:, :, head].float() @ k[:, :, kv].float().transpose(-1, -2)) / math.sqrt(d)
            sink = weights[attn + ".softmax_offset"][head].float().expand(b, s, 1)
            probability = torch.cat((scores.masked_fill(~allowed, -torch.inf), sink), -1).softmax(-1)[..., :-1]
            heads.append(probability @ v[:, :, kv].float())
        attended = torch.stack(heads, 2).reshape(b, s, qh * d).to(compute_dtype)
        x = x + project(attended, attn + ".o_proj.weight")
        normalized = rms(x, weights[prefix + ".post_attention_layernorm.weight"], config.rms_norm_eps)
        name = prefix + ".mlp"
        if index in config.mlp_only_layers:
            output = mlp(normalized, name)
        else:
            flat = normalized.reshape(-1, h)
            scores = torch.sigmoid(project(flat, name + ".gate.weight").float())
            selected, ids = scores.topk(config.num_experts_per_tok, dim=-1, sorted=False)
            probabilities = selected / (selected.sum(-1, keepdim=True) + 1e-20)
            routes[index] = probabilities, ids
            if routed_precision_contract:
                slots = torch.zeros(
                    flat.shape[0] * config.num_experts_per_tok, h, dtype=torch.float32, device=flat.device
                )
                for expert in range(config.num_experts):
                    expert_name = name + f".experts.{expert}"
                    rows, positions = ((ids == expert) & valid.flatten()[:, None]).nonzero(as_tuple=True)
                    gate_weight, up_weight, down_weight = (
                        weights[expert_name + ".gate_proj.weight"],
                        weights[expert_name + ".up_proj.weight"],
                        weights[expert_name + ".down_proj.weight"],
                    )
                    if rows.numel() == 0:
                        slots = slots + (gate_weight.sum() + up_weight.sum() + down_weight.sum()) * 0
                        continue
                    selected_values = flat.index_select(0, rows).to(compute_dtype)
                    fused_weight = torch.cat((gate_weight.T, up_weight.T), dim=-1).to(compute_dtype)
                    gate, up = (selected_values @ fused_weight).chunk(2, dim=-1)
                    activated = _AnalyticSwiGLU.apply(gate, up)
                    values = activated @ down_weight.T.to(compute_dtype)
                    contribution = values.float() * probabilities[rows, positions, None]
                    slots = slots.index_add(0, rows * config.num_experts_per_tok + positions, contribution)
                routed = slots.reshape(flat.shape[0], config.num_experts_per_tok, h).sum(dim=1)
            else:
                routed = torch.zeros_like(flat, dtype=torch.float32)
                for expert in range(config.num_experts):
                    coefficient = ((ids == expert) * probabilities).sum(-1, keepdim=True)
                    routed = routed + mlp(flat, name + f".experts.{expert}").float() * coefficient
            output = routed.to(flat.dtype) + mlp(flat, name + ".shared_expert")
            output = output.reshape_as(normalized)
        x = x + output
    normalized = rms(x, weights["model.norm.weight"], config.rms_norm_eps)
    return F.linear(normalized.float(), weights["lm_head.weight"].float()), routes

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

# The router and softcapped expert activation follow the HuggingFace Inkling 2 implementation (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Source: https://github.com/huggingface/transformers/pull/50682
# Revision: 109a7d19bf667168837f32929e00b791e195d14d
# Original file: src/transformers/models/inkling2/modeling_inkling2.py

"""Routed-expert feed-forward for Inkling 2.

The routing and expert equations follow the Transformers ``inkling2`` reference
(huggingface/transformers commit 109a7d19bf667168837f32929e00b791e195d14d,
``Inkling2TopKRouter`` / ``Inkling2SoftcappedSwiGLU`` / ``Inkling2Experts``, Apache-2.0).
Routing reuses AutoModel's :class:`Gate` (``sigmoid_with_bias``), and the experts reuse
AutoModel's grouped / expert-parallel implementations with an Inkling 2 activation.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import torch

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .configuration import Inkling2TextConfig


def inkling2_moe_config(config: Inkling2TextConfig, dtype: torch.dtype) -> MoEConfig:
    """Describe the Inkling 2 routed experts as an AutoModel :class:`MoEConfig`.

    The router computes FP32 logits and sigmoid scores, selects the top-k experts on
    ``scores + e_score_correction_bias`` and returns the selected unbiased scores
    normalized to sum to one and multiplied by ``route_scale``. The routing weights stay
    FP32 and multiply each expert's output after its down projection, and the top-k sum is
    accumulated in FP32, as in the reference. There are no shared experts.

    Args:
        config: Inkling 2 text configuration.
        dtype: Storage dtype of the expert weights. The router weight and the correction
            bias are always FP32.

    Returns:
        The routed-expert configuration.
    """
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.moe_intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=0,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid_with_bias",
        route_scale=config.route_scale,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        apply_router_weight_after_down=True,
        router_weights_fp32=True,
        force_e_score_correction_bias=True,
        dtype=dtype,
        gate_dtype=torch.float32,
    )


def inkling2_softcapped_swiglu(
    gate_and_up: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    gate_softcap: float,
    up_softcap: float,
) -> torch.Tensor:
    """Apply Inkling 2's softcapped SwiGLU to an interleaved gate/up projection.

    Computes ``softcap(g, gate_softcap) * sigmoid(g) * softcap(u, up_softcap)`` in FP32,
    where ``softcap(x, c) = tanh(x / c) * c``. The sigmoid sees the raw gate.

    Args:
        gate_and_up: Tensor of shape ``[..., 2 * intermediate]`` with arbitrary leading
            dimensions, whose last axis interleaves gate and up channels as
            ``[g0, u0, g1, u1, ...]`` (the released ``w13_weight`` row order).
        routing_weights: Tensor broadcastable to ``[..., intermediate]`` scaling the
            activation; all ones when the router weight is applied after the down projection.

    Returns:
        Tensor of shape ``[..., intermediate]`` in ``gate_and_up.dtype``.
    """
    gate = gate_and_up[..., 0::2].float()
    up = gate_and_up[..., 1::2].float()
    gate_factor = torch.tanh(gate / gate_softcap) * gate_softcap
    capped_up = torch.tanh(up / up_softcap) * up_softcap
    activated = gate_factor * torch.sigmoid(gate) * capped_up
    return (activated * routing_weights.float()).to(gate_and_up.dtype)


class Inkling2MoE(MoE):
    """Inkling 2 sparse feed-forward on AutoModel's router and grouped experts.

    Subclasses :class:`~nemo_automodel.components.moe.layers.MoE`, so the expert-parallel
    parallelizer finds it and shards ``experts``. The base forward is reused unchanged.

    Native parameters:
        ``gate.weight``: FP32 tensor of shape ``[experts, hidden]`` (checkpoint ``mlp.gate.weight``).
        ``gate.e_score_correction_bias``: FP32 buffer of shape ``[experts]`` (checkpoint
        ``mlp.gate.bias``); it only biases expert selection and receives no gradient.
        ``experts.gate_and_up_projs``: Tensor of shape ``[experts, hidden, 2 * intermediate]``
        whose last axis interleaves gate and up channels, the transpose of the checkpoint's
        ``w13_weight`` ``[experts, 2 * intermediate, hidden]``.
        ``experts.down_projs``: Tensor of shape ``[experts, intermediate, hidden]``, the
        transpose of the checkpoint's ``w2_weight``.
    """

    def __init__(
        self,
        config: Inkling2TextConfig,
        backend: BackendConfig,
        moe_config: MoEConfig | None = None,
    ) -> None:
        """Build the router and experts.

        Args:
            config: Inkling 2 text configuration (softcaps, and the expert shapes when
                ``moe_config`` is omitted).
            backend: Backend configuration. ``gate_precision`` must be unset or FP32,
                since the reference routes in FP32.
            moe_config: Routed-expert configuration; defaults to :func:`inkling2_moe_config`
                with the dtype of ``config.torch_dtype`` (BF16 when unset).

        Raises:
            ValueError: If the backend requests a non-FP32 gate precision or an expert
                implementation whose parameters the state-dict adapter cannot map (TE grouped
                experts, MoK).
        """
        if backend.gate_precision not in (None, torch.float32):
            raise ValueError(f"Inkling 2 routes in FP32; got gate_precision={backend.gate_precision}")
        if moe_config is None:
            moe_config = inkling2_moe_config(config, get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16))
        super().__init__(moe_config, dataclasses.replace(backend, gate_precision=torch.float32))

        activation = partial(
            inkling2_softcapped_swiglu,
            gate_softcap=config.mlp_gate_softcap,
            up_softcap=config.mlp_up_softcap,
        )
        # Only expert implementations that keep ``gate_and_up_projs`` / ``down_projs`` (the names
        # the state-dict adapter maps) are accepted; they name their pluggable activation differently.
        if isinstance(self.experts, GroupedExperts):
            self.experts.expert_activation_grouped = activation
        elif isinstance(self.experts, GroupedExpertsDeepEP):
            self.experts.expert_activation = activation
        else:
            raise ValueError(
                f"Inkling 2 does not support the {type(self.experts).__name__} expert backend; with the "
                "deepep/hybridep/uccl_ep dispatchers set backend.experts to torch_mm (or gmm / torch_mm_mxfp8)"
            )


__all__ = ["Inkling2MoE", "inkling2_moe_config", "inkling2_softcapped_swiglu"]

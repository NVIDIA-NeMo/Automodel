# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared checkpoint helpers for fp32 GatedDeltaNet (GDN) params.

GDN layers carry intrinsically-fp32 parameters (``A_log`` and ``dt_bias``) that
feed the decay gate ``g = -exp(A_log) * softplus(a + dt_bias)``. Under FSDP2 mixed
precision with fp32 master weights, the bulk of a model computes in bf16
(``param_dtype=bf16``) while these parameters must stay in fp32 -- ``A_log`` is
exponentiated, so bf16 rounding becomes a proportional error on the decay rate
that the recurrence compounds across the sequence.

Each model owns these parameters as plain fp32 attributes under their HF names
(``linear_attn.A_log``, ``linear_attn.dt_bias``), so native state-dict keys equal
HF keys. This module centralizes the checkpoint contract: recognize those keys,
upcast them to fp32 when checkpoint tensors arrive in a lower precision, and force
fp32 on HF export.
"""

from __future__ import annotations

import torch

# Intrinsically-fp32 GatedDeltaNet parameter names (HF names on the GDN module).
FP32_GDN_PARAM_NAMES = ("A_log", "dt_bias")
GDN_FP32_CHECKPOINT_ARCHITECTURES = frozenset(
    (
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    )
)


def is_gated_delta_net_fp32_param_key(key: str, param_names: tuple[str, ...] = FP32_GDN_PARAM_NAMES) -> bool:
    """Return whether ``key`` names an intrinsically-fp32 GDN parameter."""
    return key.endswith(param_names) and ".linear_attn." in key


def has_gated_delta_net_fp32_checkpoint_contract(hf_config: object) -> bool:
    """Return whether ``hf_config`` belongs to an architecture with fp32 GDN params."""
    architectures = getattr(hf_config, "architectures", None) or ()
    return any(arch in GDN_FP32_CHECKPOINT_ARCHITECTURES for arch in architectures)


def upcast_gated_delta_net_fp32_state_tensor(
    key: str, tensor: object, param_names: tuple[str, ...] = FP32_GDN_PARAM_NAMES
) -> object:
    """Cast loaded GDN fp32-param tensors to fp32 while leaving other state untouched.

    Construction-time upcasting is not enough for checkpoint and HF load paths that
    replace or carry tensor values from disk. This helper preserves the fp32 GDN
    contract at adapter boundaries before tensors enter the live model state dict.
    """
    if not is_gated_delta_net_fp32_param_key(key, param_names):
        return tensor
    if getattr(tensor, "dtype", None) == torch.float32:
        return tensor
    is_floating_point = getattr(tensor, "is_floating_point", None)
    if callable(is_floating_point) and is_floating_point():
        return tensor.to(dtype=torch.float32)
    return tensor


def forced_gated_delta_net_fp32_dtype_mapping(
    state_dict: dict[str, object], param_names: tuple[str, ...] = FP32_GDN_PARAM_NAMES
) -> dict[str, str]:
    """Return HF export dtype overrides for intrinsically-fp32 GDN tensors."""
    forced: dict[str, str] = {}
    for key, tensor in state_dict.items():
        if not is_gated_delta_net_fp32_param_key(key, param_names):
            continue
        is_floating_point = getattr(tensor, "is_floating_point", None)
        if callable(is_floating_point) and is_floating_point():
            forced[key] = "F32"
    return forced

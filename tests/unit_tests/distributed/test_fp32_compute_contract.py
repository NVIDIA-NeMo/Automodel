# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""End-to-end contract: every parameter of a block computes in the dtype that
precision correctness requires, inside the block's single FSDP unit.

These are CPU unit tests (no real FSDP): we run the production
``fully_shard_by_dtype`` with ``fully_shard`` mocked to record the unit's policy,
then resolve each parameter's compute dtype as
``param_dtype_override_fn(param) or mp_policy.param_dtype``. This exercises the
full resolution path (pinned -> HF-recorded -> mp_policy.param_dtype) across the
model archetypes that use ``fully_shard_by_dtype`` (NemotronH dense layers,
Qwen3.5 / Qwen3-Next hybrid mixers, Qwen3.5-MoE) under fp32 master weights.
"""

from dataclasses import fields

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy

import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype

_HAS_OVERRIDE_FIELD = "param_dtype_override_fn" in {field.name for field in fields(MixedPrecisionPolicy)}
requires_param_dtype_override = pytest.mark.skipif(
    not _HAS_OVERRIDE_FIELD,
    reason="MixedPrecisionPolicy.param_dtype_override_fn requires PyTorch >= 2.15",
)

FP32_TOKENS = ("linear_attn.A_log", "linear_attn.dt_bias")


def _mp_policy(param_dtype: torch.dtype | None = torch.bfloat16) -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=torch.float32)


def _compute_dtype(policy: MixedPrecisionPolicy | None, param: nn.Parameter) -> torch.dtype:
    """Compute dtype FSDP would use for ``param`` under ``policy``."""
    if policy is None or policy.param_dtype is None:
        return param.dtype
    if _HAS_OVERRIDE_FIELD and policy.param_dtype_override_fn is not None:
        return policy.param_dtype_override_fn(param) or policy.param_dtype
    return policy.param_dtype


def _resident_compute_dtypes(model, mp_policy, fp32_compute_module_names, monkeypatch):
    """Shard ``model`` (mocked) as one unit and return ``{param_name: compute dtype}``."""
    calls: list[tuple[nn.Module, MixedPrecisionPolicy | None]] = []

    def fake_fully_shard(module, *, mesh, mp_policy, offload_policy, **kwargs):
        calls.append((module, mp_policy))

    monkeypatch.setattr(parallelizer_utils, "fully_shard", fake_fully_shard, raising=True)

    fully_shard_by_dtype(
        model,
        mesh=object(),
        mp_policy=mp_policy,
        offload_policy=object(),
        fp32_compute_module_names=fp32_compute_module_names,
    )

    assert [module for module, _ in calls] == [model], "a block must be exactly one FSDP unit"
    policy = calls[0][1]
    return {name: _compute_dtype(policy, param) for name, param in model.named_parameters()}


def _tag_hf(tensor, dtype):
    tensor._hf_compute_dtype = dtype


# --------------------------------------------------------------------------- #
# Model archetypes (tiny) that use fully_shard_by_dtype in production.
# --------------------------------------------------------------------------- #


class DenseLayer(nn.Module):
    """NemotronH-style dense layer: only ordinary projection weights."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.attn = nn.Linear(8, 8, bias=False).to(dtype)
        self.mlp = nn.Linear(8, 8, bias=False).to(dtype)


class GatedDeltaNet(nn.Module):
    """Qwen3.5-style mixer: projections plus bare fp32 ``A_log``/``dt_bias`` (HF names).

    The decay parameters are constructed fp32 regardless of the model dtype.
    """

    def __init__(self, dtype=torch.float32, n=4):
        super().__init__()
        self.in_proj = nn.Linear(8, 8, bias=False).to(dtype)
        self.out_proj = nn.Linear(8, 8, bias=False).to(dtype)
        self.A_log = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(n, dtype=torch.float32))


class HybridLayer(nn.Module):
    """Qwen3.5 / Qwen3-Next linear-attention decoder layer."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.linear_attn = GatedDeltaNet(dtype)
        self.mlp = nn.Linear(8, 8, bias=False).to(dtype)


class MoEHybridLayer(nn.Module):
    """Qwen3.5-MoE linear-attention layer (#3327 archetype) under fp32 master weights:
    projections, ``shared_expert_gate`` and experts are all stored fp32 and compute
    bf16; only the pinned ``A_log``/``dt_bias`` compute fp32."""

    def __init__(self):
        super().__init__()
        self.linear_attn = GatedDeltaNet(torch.float32)
        self.shared_expert_gate = nn.Linear(8, 1, bias=False).to(torch.float32)
        self.experts = nn.Linear(8, 8, bias=False).to(torch.float32)


# --------------------------------------------------------------------------- #
# Dense archetype
# --------------------------------------------------------------------------- #


def test_dense_master_weights_compute_bf16(monkeypatch):
    """fp32 master weights, no fp32 params -> the whole dense layer computes bf16."""
    layer = DenseLayer(dtype=torch.float32)
    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), (), monkeypatch)

    assert resolved == {"attn.weight": torch.bfloat16, "mlp.weight": torch.bfloat16}


def test_dense_genuine_fp32_policy_computes_fp32(monkeypatch):
    """An explicit fp32 compute policy keeps the dense layer in fp32."""
    layer = DenseLayer(dtype=torch.float32)
    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.float32), (), monkeypatch)

    assert resolved == {"attn.weight": torch.float32, "mlp.weight": torch.float32}


# --------------------------------------------------------------------------- #
# Hybrid archetype -- the dtype-source scenarios
# --------------------------------------------------------------------------- #

HYBRID_EXPECTED = {
    "linear_attn.in_proj.weight": torch.bfloat16,
    "linear_attn.out_proj.weight": torch.bfloat16,
    "linear_attn.A_log": torch.float32,
    "linear_attn.dt_bias": torch.float32,
    "mlp.weight": torch.bfloat16,
}


@requires_param_dtype_override
def test_hybrid_master_weights_pinned_keeps_decay_fp32(monkeypatch):
    """From-scratch master weights: the pin keeps A_log/dt_bias fp32, bulk computes bf16."""
    layer = HybridLayer(dtype=torch.float32)
    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), FP32_TOKENS, monkeypatch)

    assert resolved == HYBRID_EXPECTED


@requires_param_dtype_override
def test_hybrid_master_weights_hf_recorded_keeps_decay_fp32(monkeypatch):
    """Loaded-from-checkpoint master weights: HF records keep A_log fp32 with no pin."""
    layer = HybridLayer(dtype=torch.float32)
    for name, param in layer.named_parameters():
        _tag_hf(param, torch.float32 if name.endswith(("A_log", "dt_bias")) else torch.bfloat16)

    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), (), monkeypatch)

    assert resolved == HYBRID_EXPECTED


@requires_param_dtype_override
def test_hybrid_bf16_load_with_restored_fp32_decay_is_rejected(monkeypatch):
    """bf16 load: projections stored bf16 beside fp32 decay params cannot share one FSDP unit."""
    layer = HybridLayer(dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="one storage dtype per unit.*model.dtype"):
        _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), FP32_TOKENS, monkeypatch)


@requires_param_dtype_override
def test_hybrid_pin_overrides_hf_recorded_dtype(monkeypatch):
    """The pin wins even when an HF record would say bf16."""
    layer = HybridLayer(dtype=torch.float32)
    for param in layer.parameters():
        _tag_hf(param, torch.bfloat16)

    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), FP32_TOKENS, monkeypatch)

    assert resolved == HYBRID_EXPECTED


@requires_param_dtype_override
def test_moe_hybrid_master_weights_gate_computes_bf16(monkeypatch):
    """#3327 under fp32 master weights: ``shared_expert_gate`` and experts compute bf16
    like every other projection; only the pinned decay parameters stay fp32."""
    layer = MoEHybridLayer()
    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), FP32_TOKENS, monkeypatch)

    assert resolved == {
        "linear_attn.in_proj.weight": torch.bfloat16,
        "linear_attn.out_proj.weight": torch.bfloat16,
        "linear_attn.A_log": torch.float32,
        "linear_attn.dt_bias": torch.float32,
        "shared_expert_gate.weight": torch.bfloat16,
        "experts.weight": torch.bfloat16,
    }


@requires_param_dtype_override
def test_hybrid_stack_of_layers_master_weights(monkeypatch):
    """Strategies shard one layer at a time: every layer resolves identically."""
    stack = nn.ModuleList([HybridLayer(dtype=torch.float32) for _ in range(3)])

    for layer in stack:
        assert _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), FP32_TOKENS, monkeypatch) == HYBRID_EXPECTED


def test_no_mixed_precision_policy_falls_back_to_storage(monkeypatch):
    """Without a policy, compute dtype is the storage dtype (fp32 master weights)."""
    layer = HybridLayer(dtype=torch.float32)
    resolved = _resident_compute_dtypes(layer, None, FP32_TOKENS, monkeypatch)

    assert set(resolved.values()) == {torch.float32}


def test_pinned_param_below_fp32_storage_is_rejected(monkeypatch):
    """A model cast wholesale to bf16 cannot satisfy the fp32 contract."""
    layer = HybridLayer(dtype=torch.bfloat16).to(torch.bfloat16)

    with pytest.raises(ValueError, match=r"model\.dtype"):
        _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), FP32_TOKENS, monkeypatch)


def test_intrinsic_fp32_holder_models_declare_strict_pin():
    """Hybrid models must pin their fp32 decay parameters by concrete HF name.

    ``fully_shard_by_dtype`` keys fp32 compute on ``_keep_in_fp32_modules_strict``;
    a missing token silently drops A_log/dt_bias to bf16 compute.
    """
    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
    from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
    from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM

    for cls in (Qwen3_5MoeForConditionalGeneration, Qwen3NextForCausalLM, Qwen3_8_FlashNextForConditionalGeneration):
        strict = set(cls._keep_in_fp32_modules_strict)
        assert "_fp32_params" not in strict, cls.__name__
        assert set(FP32_TOKENS) <= strict, cls.__name__

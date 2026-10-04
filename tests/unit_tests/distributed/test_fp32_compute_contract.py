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

These are CPU unit tests (no real FSDP): the base ``ModelParallelizer`` shards
with ``fully_shard`` mocked to record each unit's policy, then each parameter's
compute dtype is resolved as ``param_dtype_override_fn(param) or
mp_policy.param_dtype``. This exercises the full resolution path (strict pin ->
mp_policy.param_dtype) across the model archetypes (NemotronH dense layers,
Qwen3.5 / Qwen3-Next hybrid mixers, Qwen3.5-MoE, an fp32 ``lm_head``) under
fp32 master weights.
"""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy

import nemo_automodel.components.distributed.parallelizer as parallelizer_mod
import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer

_HAS_OVERRIDE_FIELD = "param_dtype_override_fn" in {field.name for field in fields(MixedPrecisionPolicy)}
requires_param_dtype_override = pytest.mark.requires_param_dtype_override

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


class _FakeFSDPModule(nn.Module):
    """Marker mixed into modules the fake ``fully_shard`` has wrapped."""


def _record_fully_shard(monkeypatch) -> list[tuple[nn.Module, dict]]:
    """Replace the base parallelizer's ``fully_shard`` with a ``(module, kwargs)`` recorder.

    Like the real primitive, the fake turns the module into an FSDP unit so later
    units (the root) no longer own its parameters.
    """
    calls: list[tuple[nn.Module, dict]] = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))
        module.__class__ = type(type(module).__name__, (type(module), _FakeFSDPModule), {})
        module.set_modules_to_forward_prefetch = MagicMock()
        module.set_modules_to_backward_prefetch = MagicMock()
        return module

    monkeypatch.setattr(parallelizer_mod, "fully_shard", fake_fully_shard, raising=True)
    monkeypatch.setattr(parallelizer_mod, "FSDPModule", _FakeFSDPModule)
    monkeypatch.setattr(parallelizer_utils, "FSDPModule", _FakeFSDPModule)
    return calls


def _resident_compute_dtypes(block, mp_policy, fp32_compute_module_names, monkeypatch):
    """Shard ``block`` (mocked) as one unit of a bound model and return ``{param_name: compute dtype}``."""
    calls = _record_fully_shard(monkeypatch)
    model = nn.Module()
    model.layers = nn.ModuleList([block])
    model._keep_in_fp32_modules_strict = list(fp32_compute_module_names)
    parallelizer = ModelParallelizer()

    with parallelizer._bind_model(model):
        parallelizer._fully_shard_module(block, mesh=object(), mp_policy=mp_policy, offload_policy=object())

    assert [module for module, _ in calls] == [block], "a block must be exactly one FSDP unit"
    policy = calls[0][1]["mp_policy"]
    return {name: _compute_dtype(policy, param) for name, param in block.named_parameters()}


# --------------------------------------------------------------------------- #
# Model archetypes (tiny) sharded one block per FSDP unit in production.
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
def test_hybrid_bf16_load_with_restored_fp32_decay_is_rejected(monkeypatch):
    """bf16 load: projections stored bf16 beside fp32 decay params cannot share one FSDP unit."""
    layer = HybridLayer(dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="one storage dtype per unit.*model.dtype"):
        _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), FP32_TOKENS, monkeypatch)


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

    FSDP sharding keys fp32 compute on ``_keep_in_fp32_modules_strict``; a
    missing token silently drops A_log/dt_bias to bf16 compute.
    """
    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
    from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
    from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM

    for cls in (Qwen3_5MoeForConditionalGeneration, Qwen3NextForCausalLM, Qwen3_8_FlashNextForConditionalGeneration):
        strict = set(cls._keep_in_fp32_modules_strict)
        assert "_fp32_params" not in strict, cls.__name__
        assert set(FP32_TOKENS) <= strict, cls.__name__


# --------------------------------------------------------------------------- #
# The base ModelParallelizer applies the contract to every unit it creates.
# --------------------------------------------------------------------------- #


class TinyHybridLM(nn.Module):
    """Untied-embedding hybrid LM: layers, final norm and an fp32 ``lm_head``."""

    _keep_in_fp32_modules_strict = [*FP32_TOKENS, "lm_head"]

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=2, hidden_size=8)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(16, 8)
        self.model.layers = nn.ModuleList([HybridLayer(torch.float32) for _ in range(2)])
        self.model.norm = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _dense_apply(model, monkeypatch, mp_policy):
    calls = _record_fully_shard(monkeypatch)
    tp_mesh = MagicMock()
    tp_mesh.size.return_value = 1
    device_mesh = MagicMock(spec=DeviceMesh)
    device_mesh.__getitem__.side_effect = lambda key: tp_mesh
    device_mesh.mesh_dim_names = ("dp",)
    dp_mesh = MagicMock()
    dp_mesh.mesh_dim_names = ("dp",)
    monkeypatch.setattr(parallelizer_mod, "get_fsdp_dp_mesh", lambda *args, **kwargs: dp_mesh)
    monkeypatch.setattr(parallelizer_mod, "_patch_fsdp_accumulated_grad_guard", lambda: None)
    ModelParallelizer()._apply(model=model, device_mesh=device_mesh, mp_policy=mp_policy)
    return {id(module): kwargs["mp_policy"] for module, kwargs in calls}, calls


@requires_param_dtype_override
def test_base_parallelizer_applies_contract_to_layers_root_and_lm_head(monkeypatch):
    model = TinyHybridLM()
    mp_policy = _mp_policy(torch.bfloat16)

    policies, calls = _dense_apply(model, monkeypatch, mp_policy)

    assert [module for module, _ in calls] == [*model.model.layers, model.model.embed_tokens, model.lm_head, model]
    for layer in model.model.layers:
        resolved = {name: _compute_dtype(policies[id(layer)], param) for name, param in layer.named_parameters()}
        assert resolved == HYBRID_EXPECTED
    # Nothing in the embedding table or the root's leftovers (final norm) is pinned.
    assert policies[id(model.model.embed_tokens)] is mp_policy
    assert policies[id(model)] is mp_policy
    # A leaf unit made only of contract parameters computes in fp32 end to end.
    lm_head_policy = policies[id(model.lm_head)]
    assert lm_head_policy.param_dtype == torch.float32
    assert lm_head_policy.reduce_dtype == torch.float32
    assert lm_head_policy.output_dtype == torch.float32
    assert lm_head_policy.cast_forward_inputs is True
    assert lm_head_policy.param_dtype_override_fn is None


@requires_param_dtype_override
def test_base_parallelizer_contract_survives_full_layer_checkpointing(monkeypatch):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    model = TinyHybridLM()
    for index, layer in enumerate(model.model.layers):
        model.model.layers[index] = checkpoint_wrapper(layer)

    policies, _ = _dense_apply(model, monkeypatch, _mp_policy(torch.bfloat16))

    for wrapper in model.model.layers:
        layer = wrapper._checkpoint_wrapped_module
        assert policies[id(wrapper)].param_dtype_override_fn(layer.linear_attn.A_log) == torch.float32
        assert policies[id(wrapper)].param_dtype_override_fn(layer.mlp.weight) is None


def test_base_parallelizer_fp32_policy_passes_through(monkeypatch):
    model = TinyHybridLM()
    mp_policy = _mp_policy(torch.float32)

    policies, _ = _dense_apply(model, monkeypatch, mp_policy)

    assert all(policy is mp_policy for policy in policies.values())


@requires_param_dtype_override
def test_base_parallelizer_moe_block_excludes_ep_experts(monkeypatch):
    """The MoE path hands EP-owned experts over as ``ignored_params``; the block still pins its own."""
    block = MoEHybridLayer()
    block.router = nn.Linear(8, 4, bias=False)
    model = nn.Module()
    model.layers = nn.ModuleList([block])
    model._keep_in_fp32_modules_strict = [*FP32_TOKENS, "router.weight"]
    calls = _record_fully_shard(monkeypatch)
    parallelizer = ModelParallelizer()
    experts = set(block.experts.parameters())

    with parallelizer._bind_model(model):
        parallelizer._fully_shard_module(
            block,
            mesh=object(),
            mp_policy=_mp_policy(torch.bfloat16),
            ignored_params=experts,
            reshard_after_forward=True,
        )

    (kwargs,) = [kwargs for _, kwargs in calls]
    assert kwargs["ignored_params"] == experts
    assert kwargs["reshard_after_forward"] is True
    override = kwargs["mp_policy"].param_dtype_override_fn
    assert override(block.router.weight) == torch.float32
    assert override(block.linear_attn.A_log) == torch.float32
    assert override(block.experts.weight) is None
    assert override(block.shared_expert_gate.weight) is None


def test_base_parallelizer_rejects_foreign_module_while_bound():
    model = nn.Module()
    model.layer = nn.Linear(4, 4)
    parallelizer = ModelParallelizer()

    with parallelizer._bind_model(model):
        with pytest.raises(RuntimeError, match="not part of the model"):
            parallelizer._fsdp_unit_mp_policy(nn.Linear(4, 4), _mp_policy(torch.bfloat16), None)
    # Unbound, the hook degrades to a plain unit with no model-level contract.
    mp_policy = _mp_policy(torch.bfloat16)
    assert parallelizer._fsdp_unit_mp_policy(nn.Linear(4, 4), mp_policy, None) is mp_policy


@requires_param_dtype_override
def test_strict_lm_head_pin_equals_lm_head_precision_fp32(monkeypatch):
    """``lm_head_precision: float32`` and ``"lm_head"`` in the strict list produce the same unit policy."""
    model = TinyHybridLM()
    calls = _record_fully_shard(monkeypatch)
    parallelizer = ModelParallelizer()
    explicit = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32)

    with parallelizer._bind_model(model):
        parallelizer._fully_shard_module(model.lm_head, mesh=object(), mp_policy=_mp_policy(torch.bfloat16))
        parallelizer._fully_shard_module(model.lm_head, mesh=object(), mp_policy=explicit)

    pinned_policy, explicit_policy = (kwargs["mp_policy"] for _, kwargs in calls)
    assert explicit_policy is explicit
    assert (pinned_policy.param_dtype, pinned_policy.reduce_dtype, pinned_policy.output_dtype) == (
        explicit.param_dtype,
        explicit.reduce_dtype,
        explicit.output_dtype,
    )
    assert pinned_policy.cast_forward_inputs is explicit.cast_forward_inputs is True

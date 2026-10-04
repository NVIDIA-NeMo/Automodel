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

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import MixedPrecisionPolicy

import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.fsdp_patches import (
    patch_fsdp_uniform_reduce_dtype,
    patch_fsdp_unused_param_reduction,
)
from nemo_automodel.components.distributed.parallelizer_utils import (
    configure_fsdp_unused_param_reduction,
    fully_shard_by_dtype,
    get_internal_fsdp_mp_policy,
    reject_unsupported_mtp_cp,
    reject_unsupported_mtp_cp_pp,
    with_fp32_compute_override,
)


def test_reject_unsupported_mtp_cp_pp_allows_disabled_model():
    model = nn.Linear(2, 2)
    model.supports = SimpleNamespace(mtp_enabled=False, supports_mtp_cp_pp=False)
    reject_unsupported_mtp_cp_pp(model)


def test_reject_unsupported_mtp_cp_rejects_enabled_unsupported_model():
    model = nn.Module()
    model.mtp_config = SimpleNamespace(enabled=True)
    model.supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=False)

    with pytest.raises(RuntimeError, match="does not support MTP with context parallelism"):
        reject_unsupported_mtp_cp(model)


def test_reject_unsupported_mtp_cp_allows_supported_or_disabled_model():
    model = nn.Module()
    model.mtp_config = SimpleNamespace(enabled=True)
    model.supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=True)
    reject_unsupported_mtp_cp(model)

    model.mtp_config.enabled = False
    model.supports.mtp_enabled = False
    model.supports.supports_mtp_cp = False
    reject_unsupported_mtp_cp(model)


def test_configure_fsdp_unused_param_reduction_uses_public_fsdp_api(monkeypatch):
    from nemo_automodel.components.distributed import parallelizer_utils

    class FakeFSDPModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def set_reduce_scatter_unused_params(self, enabled, *, recurse):
            self.calls.append((enabled, recurse))

    install_fallback = Mock()
    monkeypatch.setattr(parallelizer_utils, "FSDPModule", FakeFSDPModule)
    monkeypatch.setattr(parallelizer_utils, "_patch_fsdp_unused_param_reduction", install_fallback)
    model = nn.Sequential(FakeFSDPModule(), nn.Sequential(FakeFSDPModule()))

    assert configure_fsdp_unused_param_reduction(model) == 2
    install_fallback.assert_not_called()
    assert model[0].calls == [(True, False)]
    assert model[1][0].calls == [(True, False)]


def test_configure_fsdp_unused_param_reduction_uses_legacy_fallback(monkeypatch):
    from nemo_automodel.components.distributed import parallelizer_utils

    class LegacyFSDPModule(nn.Module):
        pass

    install_fallback = Mock()
    monkeypatch.setattr(parallelizer_utils, "FSDPModule", LegacyFSDPModule)
    monkeypatch.setattr(parallelizer_utils, "_patch_fsdp_unused_param_reduction", install_fallback)
    model = nn.Sequential(LegacyFSDPModule(), nn.Sequential(LegacyFSDPModule()))

    assert configure_fsdp_unused_param_reduction(model) == 2
    install_fallback.assert_called_once_with()


def test_legacy_fsdp_unused_param_reduction_fills_missing_local_grad(monkeypatch):
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    calls = []

    def original_post_backward(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return "post-backward-result"

    monkeypatch.setattr(FSDPParamGroup, "post_backward", original_post_backward)
    patch_fsdp_unused_param_reduction()
    patched_post_backward = FSDPParamGroup.post_backward

    param = torch.nn.Parameter(torch.ones(2))
    fsdp_param = SimpleNamespace(
        _unsharded_param=param,
        unsharded_accumulated_grad=None,
        unsharded_param=param,
    )
    param_group = SimpleNamespace(
        reduce_grads=True,
        _training_state=TrainingState.PRE_BACKWARD,
        fsdp_params=[fsdp_param, SimpleNamespace()],
    )

    result = patched_post_backward(param_group, "arg", flag=True)
    patch_fsdp_unused_param_reduction()

    assert result == "post-backward-result"
    assert torch.equal(param.grad, torch.zeros_like(param))
    assert calls == [(param_group, ("arg",), {"flag": True})]
    assert FSDPParamGroup.post_backward is patched_post_backward


def _install_uniform_reduce_dtype(monkeypatch, recorder):
    """Install the patch over a stub foreach_reduce that records what it receives."""
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group

    def stub(fsdp_params, unsharded_grads, *args, **kwargs):
        recorder.append([g.dtype for g in unsharded_grads])
        return "reduced"

    monkeypatch.setattr(collectives, "foreach_reduce", stub)
    monkeypatch.setattr(param_group, "foreach_reduce", stub)
    patch_fsdp_uniform_reduce_dtype()
    return collectives


def test_uniform_reduce_dtype_widens_mixed_group(monkeypatch):
    """A bf16 straggler is widened to match its fp32 peers before the reduce."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)

    grads = [torch.ones(2, dtype=torch.float32), torch.full((2,), 5.0, dtype=torch.bfloat16)]
    result = collectives.foreach_reduce(["p0", "p1"], grads)

    assert result == "reduced"
    assert seen == [[torch.float32, torch.float32]]
    # Mutated in place so foreach_reduce's list.clear() still frees the caller's refs.
    assert [g.dtype for g in grads] == [torch.float32, torch.float32]
    assert torch.equal(grads[1], torch.full((2,), 5.0))


def test_uniform_reduce_dtype_localizes_residual_dtensor(monkeypatch):
    """The old public unused-param zero is localized before ``chunk_cat``."""
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group
    import torch.distributed.tensor as tensor_module

    class FakeDTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, tensor):
            return torch.Tensor._make_subclass(cls, tensor, False)

        def to_local(self):
            # Model an EP-local tensor with half of the global expert storage.
            return self.as_subclass(torch.Tensor)[:2]

    seen = []

    def stub(fsdp_params, unsharded_grads, *args, **kwargs):
        seen.append([(type(grad), grad.numel()) for grad in unsharded_grads])
        return "reduced"

    monkeypatch.setattr(tensor_module, "DTensor", FakeDTensor)
    monkeypatch.setattr(collectives, "foreach_reduce", stub)
    monkeypatch.setattr(param_group, "foreach_reduce", stub)
    patch_fsdp_uniform_reduce_dtype()

    grads = [torch.ones(2), FakeDTensor(torch.ones(4))]
    result = collectives.foreach_reduce(["used", "unused"], grads)

    assert result == "reduced"
    assert seen == [[(torch.Tensor, 2), (torch.Tensor, 2)]]
    assert all(type(grad) is torch.Tensor for grad in grads)


def test_uniform_reduce_dtype_leaves_uniform_group_untouched(monkeypatch):
    """Uniform groups pass straight through, preserving upstream's own checks."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)

    grads = [torch.ones(2, dtype=torch.bfloat16), torch.ones(2, dtype=torch.bfloat16)]
    original = [g for g in grads]
    collectives.foreach_reduce(["p0", "p1"], grads)

    assert seen == [[torch.bfloat16, torch.bfloat16]]
    assert all(a is b for a, b in zip(grads, original))


def test_uniform_reduce_dtype_ignores_non_float_mixtures(monkeypatch):
    """Non-float gradients are left alone so the upstream assertion still fires."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)

    grads = [torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.int32)]
    collectives.foreach_reduce(["p0", "p1"], grads)

    assert seen == [[torch.float32, torch.int32]]


def test_uniform_reduce_dtype_patch_is_idempotent(monkeypatch):
    """Re-installing must not stack a second wrapper."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)
    wrapped = collectives.foreach_reduce

    patch_fsdp_uniform_reduce_dtype()

    assert collectives.foreach_reduce is wrapped


def test_configure_fsdp_unused_param_reduction_installs_dtype_alignment_first(monkeypatch):
    """The zero fill must wrap the alignment so filled zeros are aligned too."""
    from nemo_automodel.components.distributed import parallelizer_utils

    class LegacyFSDPModule(nn.Module):
        pass

    order = []
    monkeypatch.setattr(parallelizer_utils, "FSDPModule", LegacyFSDPModule)
    monkeypatch.setattr(parallelizer_utils, "_patch_fsdp_uniform_reduce_dtype", lambda: order.append("uniform_dtype"))
    monkeypatch.setattr(parallelizer_utils, "_patch_fsdp_unused_param_reduction", lambda: order.append("zero_fill"))

    assert configure_fsdp_unused_param_reduction(nn.Sequential(LegacyFSDPModule())) == 1
    assert order == ["uniform_dtype", "zero_fill"]


# --------------------------------------------------------------------------- #
# fp32 compute inside a single FSDP unit
# --------------------------------------------------------------------------- #

_HAS_OVERRIDE_FIELD = "param_dtype_override_fn" in {field.name for field in fields(MixedPrecisionPolicy)}
requires_param_dtype_override = pytest.mark.skipif(
    not _HAS_OVERRIDE_FIELD,
    reason="MixedPrecisionPolicy.param_dtype_override_fn requires PyTorch >= 2.15",
)

FP32_TOKENS = ("linear_attn.A_log", "linear_attn.dt_bias")


def _make_mp_policy(param_dtype: torch.dtype | None = torch.bfloat16) -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
    )


def _record_fully_shard(monkeypatch) -> list[tuple[nn.Module, dict]]:
    """Replace ``parallelizer_utils.fully_shard`` with a recorder of ``(module, kwargs)``."""
    calls: list[tuple[nn.Module, dict]] = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    monkeypatch.setattr(parallelizer_utils, "fully_shard", fake_fully_shard, raising=True)
    return calls


def _override_of(policy: MixedPrecisionPolicy):
    return policy.param_dtype_override_fn


class GatedDeltaNet(nn.Module):
    """Linear-attention mixer: projections plus bare fp32 ``A_log``/``dt_bias`` (HF names)."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.in_proj = nn.Linear(4, 4, bias=False).to(dtype)
        self.out_proj = nn.Linear(4, 4, bias=False).to(dtype)
        self.A_log = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))


class HybridBlock(nn.Module):
    """Decoder block whose ``linear_attn`` owns the fp32-contract parameters."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.linear_attn = GatedDeltaNet(dtype)
        self.mlp = nn.Linear(4, 4, bias=False).to(dtype)


def _assert_policy_fields_preserved(policy: MixedPrecisionPolicy, source: MixedPrecisionPolicy) -> None:
    assert policy.param_dtype == source.param_dtype
    assert policy.reduce_dtype == source.reduce_dtype
    assert policy.output_dtype == source.output_dtype
    assert policy.cast_forward_inputs == source.cast_forward_inputs


def test_internal_fsdp_mp_policy_drops_only_output_dtype():
    mp_policy = _make_mp_policy()

    internal_policy = get_internal_fsdp_mp_policy(mp_policy)

    assert internal_policy is not mp_policy
    assert internal_policy.param_dtype == torch.bfloat16
    assert internal_policy.reduce_dtype == torch.float32
    assert internal_policy.cast_forward_inputs is False
    assert internal_policy.output_dtype is None
    assert get_internal_fsdp_mp_policy(None) is None


def test_fully_shard_by_dtype_no_params(monkeypatch):
    """A parameterless module is still sharded exactly once with the caller's policy."""
    calls = _record_fully_shard(monkeypatch)
    mp_policy = _make_mp_policy()

    model = nn.Identity()
    fully_shard_by_dtype(model, mesh=object(), mp_policy=mp_policy, offload_policy=None)

    assert [module for module, _ in calls] == [model]
    assert calls[0][1]["mp_policy"] is mp_policy


@requires_param_dtype_override
def test_fully_shard_by_dtype_fp32_masters_pinned_params_one_unit(monkeypatch):
    """fp32 master weights + pinned params: one unit, fp32 override only for the pins."""
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.float32)
    mp_policy = _make_mp_policy()
    mesh, offload = object(), object()

    fully_shard_by_dtype(
        block,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload,
        fp32_compute_module_names=FP32_TOKENS,
        reshard_after_forward=True,
    )

    assert [module for module, _ in calls] == [block]
    kwargs = calls[0][1]
    assert kwargs["mesh"] is mesh
    assert kwargs["offload_policy"] is offload
    assert kwargs["reshard_after_forward"] is True
    assert "ignored_params" not in kwargs
    policy = kwargs["mp_policy"]
    assert policy is not mp_policy
    _assert_policy_fields_preserved(policy, mp_policy)
    override = _override_of(policy)
    assert override(block.linear_attn.A_log) == torch.float32
    assert override(block.linear_attn.dt_bias) == torch.float32
    assert override(block.linear_attn.in_proj.weight) is None
    assert override(block.linear_attn.out_proj.weight) is None
    assert override(block.mlp.weight) is None


@requires_param_dtype_override
def test_fully_shard_by_dtype_hf_recorded_fp32_param_without_pin(monkeypatch):
    """A checkpoint-recorded fp32 dtype keeps the parameter fp32 with no strict pin."""
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.float32)
    block.linear_attn.A_log._hf_compute_dtype = torch.float32
    block.linear_attn.dt_bias._hf_compute_dtype = torch.float32
    block.mlp.weight._hf_compute_dtype = torch.bfloat16

    fully_shard_by_dtype(block, mesh=object(), mp_policy=_make_mp_policy(), offload_policy=None)

    override = _override_of(calls[0][1]["mp_policy"])
    assert override(block.linear_attn.A_log) == torch.float32
    assert override(block.linear_attn.dt_bias) == torch.float32
    assert override(block.mlp.weight) is None
    assert override(block.linear_attn.in_proj.weight) is None


@requires_param_dtype_override
def test_fully_shard_by_dtype_pin_beats_bf16_hf_record(monkeypatch):
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.float32)
    block.linear_attn.A_log._hf_compute_dtype = torch.bfloat16

    fully_shard_by_dtype(
        block,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=None,
        fp32_compute_module_names=FP32_TOKENS,
    )

    override = _override_of(calls[0][1]["mp_policy"])
    assert override(block.linear_attn.A_log) == torch.float32
    assert override(block.linear_attn.in_proj.weight) is None


def test_fully_shard_by_dtype_nothing_pinned_passes_policy_through(monkeypatch):
    """Without fp32-contract parameters the caller's policy object is forwarded as is."""
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.float32)
    mp_policy = _make_mp_policy()

    fully_shard_by_dtype(block, mesh=object(), mp_policy=mp_policy, offload_policy=None)

    assert [module for module, _ in calls] == [block]
    assert calls[0][1]["mp_policy"] is mp_policy


@pytest.mark.parametrize("mp_policy", [None, _make_mp_policy(torch.float32), _make_mp_policy(None)])
def test_fully_shard_by_dtype_fp32_or_absent_policy_passes_through(monkeypatch, mp_policy):
    """``None``, fp32 and dtype-less policies never need an override, even with pins."""
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.float32)

    fully_shard_by_dtype(
        block,
        mesh=object(),
        mp_policy=mp_policy,
        offload_policy=None,
        fp32_compute_module_names=FP32_TOKENS,
    )

    assert [module for module, _ in calls] == [block]
    assert calls[0][1]["mp_policy"] is mp_policy


def test_fully_shard_by_dtype_rejects_bf16_storage_for_pinned_param(monkeypatch):
    """A pinned parameter stored below fp32 cannot compute in fp32; the error names the fix."""
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.bfloat16).to(torch.bfloat16)

    with pytest.raises(ValueError, match=r"linear_attn\.A_log must compute in fp32.*model\.dtype"):
        fully_shard_by_dtype(
            block,
            mesh=object(),
            mp_policy=_make_mp_policy(),
            offload_policy=None,
            fp32_compute_module_names=FP32_TOKENS,
        )
    assert calls == []


@requires_param_dtype_override
def test_fully_shard_by_dtype_excludes_ep_params(monkeypatch):
    """EP experts stay outside the fp32 contract and only the block's own ignored params are forwarded."""

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32))

    class MoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = GatedDeltaNet(torch.float32)
            self.gate = Router()
            # Experts are owned by the EP unit, stored bf16: pinning them would be a ValueError.
            self.experts = nn.Linear(4, 4, bias=False).to(torch.bfloat16)

    calls = _record_fully_shard(monkeypatch)
    block = MoEBlock()
    expert_params = set(block.experts.parameters())
    foreign_param = nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))

    fully_shard_by_dtype(
        block,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=None,
        fp32_compute_module_names=(*FP32_TOKENS, "gate.weight", "experts"),
        reshard_after_forward=False,
        ignored_params=expert_params | {foreign_param},
    )

    assert [module for module, _ in calls] == [block]
    kwargs = calls[0][1]
    assert kwargs["ignored_params"] == expert_params
    assert kwargs["reshard_after_forward"] is False
    override = _override_of(kwargs["mp_policy"])
    assert override(block.gate.weight) == torch.float32
    assert override(block.linear_attn.A_log) == torch.float32
    assert override(block.experts.weight) is None
    assert override(block.linear_attn.in_proj.weight) is None


def test_fully_shard_by_dtype_omits_foreign_only_ignored_params(monkeypatch):
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.float32)
    foreign_param = nn.Parameter(torch.zeros(2))

    fully_shard_by_dtype(
        block,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=None,
        ignored_params={foreign_param},
    )

    assert "ignored_params" not in calls[0][1]


def test_fully_shard_by_dtype_does_not_expand_modulelist(monkeypatch):
    """Callers shard one block at a time; a ModuleList is sharded as the single unit it is."""
    calls = _record_fully_shard(monkeypatch)
    layers = nn.ModuleList([HybridBlock(torch.float32), HybridBlock(torch.float32)])
    mp_policy = _make_mp_policy()

    fully_shard_by_dtype(layers, mesh=object(), mp_policy=mp_policy, offload_policy=None)

    assert [module for module, _ in calls] == [layers]
    assert calls[0][1]["mp_policy"] is mp_policy


def test_fully_shard_by_dtype_omits_none_reshard_kwarg(monkeypatch):
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.float32)

    fully_shard_by_dtype(block, mesh=object(), mp_policy=_make_mp_policy(), offload_policy=None)
    fully_shard_by_dtype(
        block, mesh=object(), mp_policy=_make_mp_policy(), offload_policy=None, reshard_after_forward=4
    )

    assert "reshard_after_forward" not in calls[0][1]
    assert calls[1][1]["reshard_after_forward"] == 4


@requires_param_dtype_override
def test_compute_dtype_pins_logical_names_through_activation_checkpointing():
    """Checkpoint wrappers preserve strict fp32 parameter matching on canonical names."""
    attention = nn.Module()
    attention.sinks_param = nn.Linear(4, 4, bias=False)
    attention.proj = nn.Linear(4, 4, bias=False)
    model = nn.Module()
    model.attn = checkpoint_wrapper(attention)

    policy = with_fp32_compute_override(model, _make_mp_policy(), ("attn.sinks_param",))

    override = _override_of(policy)
    assert override(attention.sinks_param.weight) == torch.float32
    assert override(attention.proj.weight) is None


def test_fully_shard_by_dtype_uses_model_parallelizer_primitive(monkeypatch):
    calls = _record_fully_shard(monkeypatch)
    block = HybridBlock(torch.float32)
    mp_policy = _make_mp_policy()
    model_parallelizer = Mock()
    model_parallelizer._fully_shard_module = Mock()
    mesh = object()

    fully_shard_by_dtype(
        block,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=None,
        reshard_after_forward=False,
        model_parallelizer=model_parallelizer,
    )

    assert calls == []
    model_parallelizer._fully_shard_module.assert_called_once_with(
        block, mesh=mesh, mp_policy=mp_policy, offload_policy=None, reshard_after_forward=False
    )


def test_fully_shard_by_dtype_requires_override_api_for_pins(monkeypatch):
    """Older PyTorch cannot keep pinned fp32 params in fp32 under a bf16 policy."""
    calls = _record_fully_shard(monkeypatch)
    monkeypatch.setattr(parallelizer_utils, "_HAS_PARAM_DTYPE_OVERRIDE", False)
    block = HybridBlock(torch.float32)

    with pytest.raises(RuntimeError, match="param_dtype_override_fn"):
        fully_shard_by_dtype(
            block,
            mesh=object(),
            mp_policy=_make_mp_policy(),
            offload_policy=None,
            fp32_compute_module_names=FP32_TOKENS,
        )
    assert calls == []

    # Without pins the old API suffices: the policy passes through.
    fully_shard_by_dtype(block, mesh=object(), mp_policy=_make_mp_policy(), offload_policy=None)
    assert len(calls) == 1

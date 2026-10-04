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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.fsdp_patches import (
    patch_fsdp_uniform_reduce_dtype,
    patch_fsdp_unused_param_reduction,
)
from nemo_automodel.components.distributed.parallelizer_utils import (
    configure_fsdp_unused_param_reduction,
    fsdp_unit_named_parameters,
    get_internal_fsdp_mp_policy,
    reject_unsupported_mtp_cp,
    reject_unsupported_mtp_cp_pp,
    with_fp32_compute_override,
)

# The two fsdp_patches version boundaries: PyTorch >= 2.15 promotes mixed gradient
# dtypes inside FSDP2 and reduces unused parameters through its public API.
_UPSTREAM_PROMOTES_GRAD_DTYPES = hasattr(FSDPParamGroup, "_get_reduce_dtype")
_UPSTREAM_REDUCES_UNUSED_PARAMS = hasattr(FSDPModule, "set_reduce_scatter_unused_params")
legacy_reduce_dtype = pytest.mark.skipif(
    _UPSTREAM_PROMOTES_GRAD_DTYPES, reason="PyTorch >= 2.15 promotes mixed FSDP2 gradient dtypes itself"
)
legacy_unused_params = pytest.mark.skipif(
    _UPSTREAM_REDUCES_UNUSED_PARAMS, reason="PyTorch >= 2.15 has FSDPModule.set_reduce_scatter_unused_params"
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


@legacy_unused_params
def test_legacy_fsdp_unused_param_reduction_fills_missing_local_grad(monkeypatch):
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState

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


@pytest.mark.skipif(not _UPSTREAM_REDUCES_UNUSED_PARAMS, reason="legacy PyTorch needs the zero-fill backport")
def test_unused_param_reduction_patch_is_noop_with_public_api():
    original = FSDPParamGroup.post_backward
    patch_fsdp_unused_param_reduction()
    assert FSDPParamGroup.post_backward is original


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


@legacy_reduce_dtype
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

    # Two 16-bit floats promote to fp32, never to one of them.
    half_grads = [torch.ones(2, dtype=torch.float16), torch.ones(2, dtype=torch.bfloat16)]
    collectives.foreach_reduce(["p0", "p1"], half_grads)
    assert seen[-1] == [torch.float32, torch.float32]


@legacy_reduce_dtype
def test_uniform_reduce_dtype_leaves_uniform_group_untouched(monkeypatch):
    """Uniform groups pass straight through, preserving upstream's own checks."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)

    grads = [torch.ones(2, dtype=torch.bfloat16), torch.ones(2, dtype=torch.bfloat16)]
    original = [g for g in grads]
    collectives.foreach_reduce(["p0", "p1"], grads)

    assert seen == [[torch.bfloat16, torch.bfloat16]]
    assert all(a is b for a, b in zip(grads, original))


@legacy_reduce_dtype
def test_uniform_reduce_dtype_ignores_non_float_mixtures(monkeypatch):
    """Non-float gradients are left alone so the upstream assertion still fires."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)

    grads = [torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.int32)]
    collectives.foreach_reduce(["p0", "p1"], grads)

    assert seen == [[torch.float32, torch.int32]]


@legacy_reduce_dtype
def test_uniform_reduce_dtype_patch_is_idempotent(monkeypatch):
    """Re-installing must not stack a second wrapper."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)
    wrapped = collectives.foreach_reduce

    patch_fsdp_uniform_reduce_dtype()

    assert collectives.foreach_reduce is wrapped


@pytest.mark.skipif(not _UPSTREAM_PROMOTES_GRAD_DTYPES, reason="legacy PyTorch needs the widening patch")
def test_uniform_reduce_dtype_patch_is_noop_when_upstream_promotes():
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives

    original = collectives.foreach_reduce
    patch_fsdp_uniform_reduce_dtype()
    assert collectives.foreach_reduce is original


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

requires_param_dtype_override = pytest.mark.requires_param_dtype_override

FP32_TOKENS = ("linear_attn.A_log", "linear_attn.dt_bias")


def _make_mp_policy(param_dtype: torch.dtype | None = torch.bfloat16) -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
    )


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


def test_fsdp_unit_named_parameters_skips_sharded_descendants_and_dedupes_ties(monkeypatch):
    class Sharded(nn.Linear):
        pass

    monkeypatch.setattr(parallelizer_utils, "FSDPModule", Sharded)
    model = nn.Module()
    model.embed = nn.Embedding(4, 4)
    model.layer = Sharded(4, 4, bias=False)
    model.head = nn.Linear(4, 4, bias=False)
    model.head.weight = model.embed.weight

    names = [name for name, _ in fsdp_unit_named_parameters(model)]

    assert names == ["embed.weight"]


@requires_param_dtype_override
def test_override_fp32_masters_pins_only_contract_params():
    """fp32 master weights + pinned params: the override selects only the pins."""
    block = HybridBlock(torch.float32)
    mp_policy = _make_mp_policy()

    policy = with_fp32_compute_override(block, mp_policy, FP32_TOKENS)

    assert policy is not mp_policy
    _assert_policy_fields_preserved(policy, mp_policy)
    override = _override_of(policy)
    assert override(block.linear_attn.A_log) == torch.float32
    assert override(block.linear_attn.dt_bias) == torch.float32
    assert override(block.linear_attn.in_proj.weight) is None
    assert override(block.linear_attn.out_proj.weight) is None
    assert override(block.mlp.weight) is None


def test_override_nothing_pinned_passes_policy_through():
    """Without fp32-contract parameters the caller's policy object is returned as is."""
    block = HybridBlock(torch.float32)
    mp_policy = _make_mp_policy()

    assert with_fp32_compute_override(block, mp_policy, ()) is mp_policy
    assert with_fp32_compute_override(nn.Identity(), mp_policy, FP32_TOKENS) is mp_policy


@pytest.mark.parametrize("mp_policy", [None, _make_mp_policy(torch.float32), _make_mp_policy(None)])
def test_override_fp32_or_absent_policy_passes_through(mp_policy):
    """``None``, fp32 and dtype-less policies never need an override, even with pins."""
    assert with_fp32_compute_override(HybridBlock(torch.float32), mp_policy, FP32_TOKENS) is mp_policy


def test_override_rejects_bf16_storage_for_pinned_param():
    """A pinned parameter stored below fp32 cannot compute in fp32; the error names the fix."""
    block = HybridBlock(torch.bfloat16).to(torch.bfloat16)

    with pytest.raises(ValueError, match=r"linear_attn\.A_log must compute in fp32.*model\.dtype"):
        with_fp32_compute_override(block, _make_mp_policy(), FP32_TOKENS)


def test_override_rejects_mixed_trainable_storage_dtypes():
    block = HybridBlock(torch.bfloat16)

    with pytest.raises(ValueError, match="one storage dtype per unit.*model.dtype"):
        with_fp32_compute_override(block, _make_mp_policy(), FP32_TOKENS)


@requires_param_dtype_override
def test_override_excludes_ignored_params():
    """EP experts handed over as ``ignored_params`` stay outside the contract and the dtype check."""

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

    block = MoEBlock()

    override = _override_of(
        with_fp32_compute_override(
            block, _make_mp_policy(), (*FP32_TOKENS, "gate.weight", "experts"), set(block.experts.parameters())
        )
    )

    assert override(block.gate.weight) == torch.float32
    assert override(block.linear_attn.A_log) == torch.float32
    assert override(block.experts.weight) is None
    assert override(block.linear_attn.in_proj.weight) is None


@requires_param_dtype_override
def test_override_skips_params_of_sharded_descendants(monkeypatch):
    """The root unit resolves only its own leftovers, not its already-sharded layers."""

    class Sharded(HybridBlock):
        pass

    monkeypatch.setattr(parallelizer_utils, "FSDPModule", Sharded)
    model = nn.Module()
    model.layer = Sharded(torch.bfloat16)  # mixed storage inside: FSDP already owns it
    model.norm = nn.Linear(4, 4, bias=False)

    mp_policy = _make_mp_policy()
    assert with_fp32_compute_override(model, mp_policy, FP32_TOKENS) is mp_policy


@requires_param_dtype_override
def test_override_matches_model_level_tokens_through_module_name():
    model = nn.Module()
    model.lm_head = nn.Linear(4, 4, bias=False)
    model.layer = nn.Linear(4, 4, bias=False)

    override = _override_of(
        with_fp32_compute_override(model.lm_head, _make_mp_policy(), ("lm_head",), module_name="lm_head")
    )

    assert override(model.lm_head.weight) == torch.float32
    assert with_fp32_compute_override(model.lm_head, _make_mp_policy(), ("lm_head",)) is not None


@requires_param_dtype_override
def test_override_pins_logical_names_through_activation_checkpointing():
    """Checkpoint wrappers preserve strict fp32 parameter matching on canonical names."""
    attention = nn.Module()
    attention.sinks_param = nn.Linear(4, 4, bias=False)
    attention.proj = nn.Linear(4, 4, bias=False)
    model = nn.Module()
    model.attn = checkpoint_wrapper(attention)

    override = _override_of(with_fp32_compute_override(model, _make_mp_policy(), ("attn.sinks_param",)))

    assert override(attention.sinks_param.weight) == torch.float32
    assert override(attention.proj.weight) is None


def test_override_requires_override_api_for_pins(monkeypatch):
    """Older PyTorch cannot keep pinned fp32 params in fp32 under a bf16 policy."""
    monkeypatch.setattr(parallelizer_utils, "_HAS_PARAM_DTYPE_OVERRIDE", False)
    block = HybridBlock(torch.float32)

    with pytest.raises(RuntimeError, match="param_dtype_override_fn"):
        with_fp32_compute_override(block, _make_mp_policy(), FP32_TOKENS)

    # Without pins the old API suffices: the policy passes through.
    mp_policy = _make_mp_policy()
    assert with_fp32_compute_override(block, mp_policy, ()) is mp_policy

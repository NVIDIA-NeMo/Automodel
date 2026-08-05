# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""CPU unit tests for the Megatron-FSDP per-parameter attribute helpers.

Covers :func:`fully_shard_optimizer`, :func:`snapshot_distributed_param_attrs`,
and :func:`restore_distributed_param_attrs` without a real Megatron-FSDP wrap:
the module globals ``MegatronFSDP``/``HAS_MEGATRON_FSDP``/
``megatron_fsdp_fully_shard_optimizer`` are monkeypatched to lightweight fakes
so the ``isinstance(model, MegatronFSDP)`` fast-path is taken on CPU.
"""

import pytest
import torch.nn as nn

from nemo_automodel.components.distributed import megatron_fsdp as mfsdp


class _FakeMegatronFSDP:
    """Lightweight stand-in for ``megatron_fsdp.MegatronFSDP``.

    Holds the real wrapped module under ``.module`` (mirroring the wheel's
    layout) so the snapshot/restore/re-stamp helpers can walk its parameters on
    CPU without a distributed wrap. ``_replace_param_with_distributed_if_needed``
    is a call-counting no-op and ``parameters`` forwards to the wrapped module.
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.replace_calls = 0

    def _replace_param_with_distributed_if_needed(self) -> None:
        self.replace_calls += 1

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)


class _TiedTinyModel(nn.Module):
    """Tiny model with ``lm_head.weight`` tied to ``embed_tokens.weight``.

    Reproduces the weight tie the recipe re-applies after wrapping so the
    ``remove_duplicate=False`` snapshot keying and the tied-``_is_shared``
    restore branch are exercised. ``mlp`` contributes a distinct (non-tied)
    weight plus a bias that is intentionally left unstamped.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 3)
        self.lm_head = nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tie the alias
        self.mlp = nn.Linear(3, 2)


def _patch_globals(monkeypatch, *, has_megatron_fsdp: bool = True) -> list:
    """Point the module globals at the CPU fakes and record fully_shard calls.

    Returns the list that the patched ``megatron_fsdp_fully_shard_optimizer``
    appends each received optimizer to, letting callers assert it ran.
    """
    fully_shard_calls: list = []

    def _fake_fully_shard_optimizer(optimizer):
        fully_shard_calls.append(optimizer)
        return optimizer

    monkeypatch.setattr(mfsdp, "MegatronFSDP", _FakeMegatronFSDP, raising=True)
    monkeypatch.setattr(mfsdp, "HAS_MEGATRON_FSDP", has_megatron_fsdp, raising=True)
    monkeypatch.setattr(mfsdp, "megatron_fsdp_fully_shard_optimizer", _fake_fully_shard_optimizer, raising=True)
    return fully_shard_calls


def _stamp_distributed_attrs(wrapped: _FakeMegatronFSDP) -> None:
    """Mimic ``MegatronFSDP.__init__`` decorating each distributed param.

    Stamps ``_megatron_fsdp_model`` and ``orig_param`` on the weight parameters
    and ``_is_shared`` on the tied alias; ``mlp.bias`` is left untouched so its
    snapshot entry is empty (exercising the restore skip branch).
    """
    module = wrapped.module
    for name, param in module.named_parameters(remove_duplicate=False):
        if name.endswith(".bias"):
            continue  # leave one param unstamped -> empty snapshot entry
        param._megatron_fsdp_model = wrapped
        param.orig_param = f"orig::{name}"
    # Megatron-FSDP marks the tied alias so grads route through the root hook.
    module.embed_tokens.weight._is_shared = True


def _rebuild_params(module: nn.Module) -> None:
    """Simulate the from_pretrained reload + re-tie that drops plain-attr state.

    Replaces each weight/bias with a fresh ``nn.Parameter`` cloned from the
    existing data (so ``Parameter.__dict__`` is emptied) and re-ties
    ``lm_head.weight`` to the new ``embed_tokens.weight`` object.
    """
    new_embed = nn.Parameter(module.embed_tokens.weight.data.clone())
    module.embed_tokens.weight = new_embed
    module.lm_head.weight = new_embed  # re-tie to the same fresh object
    module.mlp.weight = nn.Parameter(module.mlp.weight.data.clone())
    module.mlp.bias = nn.Parameter(module.mlp.bias.data.clone())


def test_snapshot_captures_tied_alias_and_stamped_attrs(monkeypatch):
    """Snapshot keys the tied alias under both names and captures stamped attrs."""
    _patch_globals(monkeypatch)
    wrapped = _FakeMegatronFSDP(_TiedTinyModel())
    _stamp_distributed_attrs(wrapped)

    snapshot = mfsdp.snapshot_distributed_param_attrs(wrapped)

    # remove_duplicate=False -> the tied param appears under BOTH names.
    assert "embed_tokens.weight" in snapshot
    assert "lm_head.weight" in snapshot
    # The _is_shared marker on the tied alias is captured under both names.
    assert snapshot["embed_tokens.weight"]["_is_shared"] is True
    assert snapshot["lm_head.weight"]["_is_shared"] is True
    # Back-ref and orig_param are captured on the distinct (non-tied) param too.
    assert snapshot["embed_tokens.weight"]["_megatron_fsdp_model"] is wrapped
    assert snapshot["mlp.weight"]["_megatron_fsdp_model"] is wrapped
    assert snapshot["mlp.weight"]["orig_param"] == "orig::mlp.weight"
    # The deliberately unstamped bias yields an empty entry (not dropped).
    assert snapshot["mlp.bias"] == {}


def test_snapshot_returns_none_when_not_megatron_fsdp(monkeypatch):
    """Snapshot no-ops (returns None) for a plain, unwrapped module."""
    _patch_globals(monkeypatch)
    assert mfsdp.snapshot_distributed_param_attrs(_TiedTinyModel()) is None


def test_restore_reapplies_dropped_attrs_and_preserves_rederived(monkeypatch):
    """Restore re-applies only missing attrs, keeps tied _is_shared, no clobber."""
    _patch_globals(monkeypatch)
    wrapped = _FakeMegatronFSDP(_TiedTinyModel())
    _stamp_distributed_attrs(wrapped)
    snapshot = mfsdp.snapshot_distributed_param_attrs(wrapped)

    # Rebuild params (drops the plain attrs) and re-derive one attr on mlp.weight
    # so the "only restore what is missing" branch can be checked for clobbering.
    _rebuild_params(wrapped.module)
    wrapped.module.mlp.weight.orig_param = "rederived::mlp.weight"
    assert not hasattr(wrapped.module.embed_tokens.weight, "_is_shared")

    mfsdp.restore_distributed_param_attrs(wrapped, snapshot)

    module = wrapped.module
    # _is_shared survives on the tied alias (same object under both names).
    assert module.embed_tokens.weight._is_shared is True
    assert module.lm_head.weight._is_shared is True
    # Back-ref restored on every stamped param (tied + distinct).
    assert module.embed_tokens.weight._megatron_fsdp_model is wrapped
    assert module.mlp.weight._megatron_fsdp_model is wrapped
    # orig_param restored on the tied param; the re-derived value is NOT clobbered.
    assert module.embed_tokens.weight.orig_param == "orig::lm_head.weight"
    assert module.mlp.weight.orig_param == "rederived::mlp.weight"
    # The unstamped bias had an empty snapshot entry -> nothing was applied.
    assert not hasattr(module.mlp.bias, "_megatron_fsdp_model")


def test_restore_is_noop_when_snapshot_none(monkeypatch):
    """Restore returns without touching params when the snapshot is None."""
    _patch_globals(monkeypatch)
    wrapped = _FakeMegatronFSDP(_TiedTinyModel())
    # Should not raise and should leave params bare.
    mfsdp.restore_distributed_param_attrs(wrapped, None)
    assert not hasattr(wrapped.module.embed_tokens.weight, "_megatron_fsdp_model")


def test_fully_shard_optimizer_restamps_and_returns_optimizer(monkeypatch):
    """fully_shard_optimizer re-stamps every param and returns the optimizer."""
    fully_shard_calls = _patch_globals(monkeypatch)
    wrapped = _FakeMegatronFSDP(_TiedTinyModel())
    optimizer = object()

    result = mfsdp.fully_shard_optimizer(wrapped, optimizer)

    assert result is optimizer
    assert fully_shard_calls == [optimizer]
    assert wrapped.replace_calls == 1
    # Every param (including the tied alias) carries the back-ref.
    for _name, param in wrapped.module.named_parameters(remove_duplicate=False):
        assert param._megatron_fsdp_model is wrapped


def test_fully_shard_optimizer_passthrough_when_not_megatron_fsdp(monkeypatch):
    """A non-Megatron-FSDP model returns the optimizer unchanged, no sharding."""
    fully_shard_calls = _patch_globals(monkeypatch)
    optimizer = object()

    result = mfsdp.fully_shard_optimizer(_TiedTinyModel(), optimizer)

    assert result is optimizer
    assert fully_shard_calls == []


def test_fully_shard_optimizer_raises_when_megatron_fsdp_unavailable(monkeypatch):
    """A wrapped model but a missing install raises ImportError before sharding."""
    _patch_globals(monkeypatch, has_megatron_fsdp=False)
    wrapped = _FakeMegatronFSDP(_TiedTinyModel())

    with pytest.raises(ImportError, match="MegatronFSDP is not installed"):
        mfsdp.fully_shard_optimizer(wrapped, object())

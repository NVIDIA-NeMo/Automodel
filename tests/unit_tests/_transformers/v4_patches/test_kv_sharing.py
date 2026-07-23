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

"""Unit tests for ``nemo_automodel._transformers.v4_patches.kv_sharing``.

These exercise the AM-454 fix on CPU (no GPU/FSDP needed) by simulating the
behaviour that breaks HF cross-layer KV sharing: FSDP2's forward-input cast runs
``tree_map`` over the per-layer kwargs, which reconstructs a plain ``dict`` and
therefore hands each layer its own empty copy of ``shared_kv_states``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from nemo_automodel._transformers.v4_patches.kv_sharing import (
    _SharedKVStates,
    install_kv_sharing_holder,
    should_install_kv_sharing_holder,
)


class _Attn(nn.Module):
    """Mimics the HF gemma3n KV-sharing contract for one layer."""

    def __init__(self, layer_idx, is_shared, kv_idx, store_full):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kv_shared_layer = is_shared
        self.kv_shared_layer_index = kv_idx
        self.store_full_length_kv = store_full

    def forward(self, x, shared_kv_states):
        if self.is_kv_shared_layer:
            # Reads K/V written by an earlier layer -> KeyError if dict not shared.
            k, v = shared_kv_states[self.kv_shared_layer_index]
            return x + k + v
        k = v = x
        if self.store_full_length_kv:
            shared_kv_states[self.layer_idx] = (k, v)
        return x


class _Layer(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn

    def forward(self, x, shared_kv_states=None):
        return self.self_attn(x, shared_kv_states)


class _TextModel(nn.Module):
    """Tiny stand-in for Gemma3nTextModel: stores K/V at layer 1, reads at 2."""

    def __init__(self, num_kv_shared_layers=2, model_type="gemma3n_text"):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=3, num_kv_shared_layers=num_kv_shared_layers, model_type=model_type
        )
        self.layers = nn.ModuleList(
            [
                _Layer(_Attn(0, is_shared=False, kv_idx=None, store_full=False)),
                _Layer(_Attn(1, is_shared=False, kv_idx=None, store_full=True)),
                _Layer(_Attn(2, is_shared=True, kv_idx=1, store_full=False)),
            ]
        )

    def forward(self, x):
        # The model owns a single dict, exactly like HF gemma3n.
        shared_kv_states = {}
        for layer in self.layers:
            x = layer(x, shared_kv_states=shared_kv_states)
        return x


def _simulate_fsdp_cast(model):
    """Register a pre-hook on each layer that copies kwargs via ``tree_map``.

    This reproduces FSDP2's ``cast_forward_inputs`` behaviour, which reconstructs
    the ``shared_kv_states`` dict per layer and breaks the sharing.
    """

    def _hook(_mod, args, kwargs):
        return pytree.tree_map(lambda t: t, args), pytree.tree_map(lambda t: t, kwargs)

    for layer in model.layers:
        layer.register_forward_pre_hook(_hook, with_kwargs=True)


def test_shared_kv_states_is_dict_like():
    holder = _SharedKVStates()
    assert 1 not in holder
    holder[1] = ("k", "v")
    assert 1 in holder
    assert holder[1] == ("k", "v")
    assert list(holder.keys()) == [1]
    assert len(holder) == 1
    assert holder.get(2) is None
    holder.clear()
    assert len(holder) == 0


def test_holder_is_pytree_opaque():
    """tree_map must return the *same* instance (a dict would be rebuilt)."""
    holder = _SharedKVStates()
    holder[1] = ("k", "v")
    mapped = pytree.tree_map(lambda t: t, {"shared_kv_states": holder})
    assert mapped["shared_kv_states"] is holder

    plain = {1: ("k", "v")}
    mapped_plain = pytree.tree_map(lambda t: t, {"shared_kv_states": plain})
    assert mapped_plain["shared_kv_states"] is not plain  # demonstrates the bug


def test_fsdp_cast_breaks_sharing_without_fix():
    model = _TextModel()
    _simulate_fsdp_cast(model)
    with pytest.raises(KeyError):
        model(torch.zeros(2))


def test_install_kv_sharing_holder_restores_sharing():
    model = _TextModel()
    _simulate_fsdp_cast(model)

    n = install_kv_sharing_holder([model])
    assert n == 3  # all three layers hooked

    # No KeyError, and the shared K/V (zeros) flow through.
    out = model(torch.zeros(2))
    assert torch.allclose(out, torch.zeros(2))


def test_holder_resets_between_forwards():
    model = _TextModel()
    install_kv_sharing_holder([model])
    model(torch.zeros(2))
    # Layer 0 (first) clears the holder each forward, so no stale keys leak in.
    holder = model.layers[0]._forward_pre_hooks  # presence check only
    assert holder is not None
    # A second forward must still succeed (holder reset, then re-populated).
    out = model(torch.ones(2))
    assert out.shape == (2,)


def test_should_install_detects_kv_sharing():
    assert should_install_kv_sharing_holder([_TextModel(num_kv_shared_layers=2)]) is True


def test_should_install_false_without_kv_sharing():
    model = _TextModel(num_kv_shared_layers=0)
    assert should_install_kv_sharing_holder([model]) is False
    assert install_kv_sharing_holder([model]) == 0


def test_skips_non_gemma3n_kv_sharing_models():
    """Other kv-sharing models (e.g. gemma4) manage their own FSDP2-safe store
    in-model and may thread a caller-supplied store (the drafter). Injecting our
    holder there would clobber it (regressed test_..._gemma4_joint_drafter), so
    such models must be skipped."""
    model = _TextModel(num_kv_shared_layers=2, model_type="gemma4_text")
    assert should_install_kv_sharing_holder([model]) is False
    assert install_kv_sharing_holder([model]) == 0
    # Layers are untouched: the caller-supplied dict still flows through.
    assert not getattr(model.layers[0], "_kv_sharing_holder_installed", False)


def test_install_is_idempotent_per_stack():
    model = _TextModel()
    first = install_kv_sharing_holder([model])
    second = install_kv_sharing_holder([model])
    assert first == 3
    assert second == 0  # already installed -> skipped


def test_apply_runtime_compatibility_fixes_installs_holder():
    """The fix is wired into _apply_runtime_compatibility_fixes (the prod path)."""
    from nemo_automodel._transformers.infrastructure import _apply_runtime_compatibility_fixes

    model = _TextModel()
    _simulate_fsdp_cast(model)
    returned = _apply_runtime_compatibility_fixes(model)
    assert returned is model
    # Holder is installed -> sharing works despite the simulated FSDP cast.
    out = model(torch.zeros(2))
    assert torch.allclose(out, torch.zeros(2))


def test_apply_runtime_compatibility_fixes_noop_without_kv_sharing():
    from nemo_automodel._transformers.infrastructure import _apply_runtime_compatibility_fixes

    model = _TextModel(num_kv_shared_layers=0)
    _apply_runtime_compatibility_fixes(model)
    assert not getattr(model.layers[0], "_kv_sharing_holder_installed", False)

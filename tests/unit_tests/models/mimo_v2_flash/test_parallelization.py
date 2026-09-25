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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2FlashAttention
from nemo_automodel.components.models.mimo_v2_flash.parallelization import (
    ensure_mimo_te_context_parallel,
    setup_mimo_te_context_parallel,
)


class _FakeDotProductAttention(torch.nn.Module):
    def __init__(self, query_heads=8, kv_heads=4):
        super().__init__()
        self.num_attention_heads = query_heads
        self.num_gqa_groups = kv_heads
        self.calls = []

    def set_context_parallel_group(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _mimo_attention(attn_module, *, query_heads=8, kv_heads=4, backend="te"):
    attention = MiMoV2FlashAttention.__new__(MiMoV2FlashAttention)
    torch.nn.Module.__init__(attention)
    attention.attn_module = attn_module
    attention.backend = SimpleNamespace(attn=backend)
    attention.num_attention_heads = query_heads
    attention.num_key_value_heads = kv_heads
    attention.layer_idx = 0
    return attention


class _Block(torch.nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn


class _Model(torch.nn.Module):
    def __init__(self, attentions):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(attn) for attn in attentions])
        self.cp_mesh = None


def _cp_mesh(size=2, group=None):
    mesh = Mock()
    mesh.size.return_value = size
    mesh.get_group.return_value = group if group is not None else object()
    return mesh


def _setup_patches(stream):
    return (
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        patch("torch.cuda.Stream", return_value=stream),
    )


def test_setup_mimo_te_cp_forces_a2a_on_every_attention():
    dpa0 = _FakeDotProductAttention()
    dpa1 = _FakeDotProductAttention()
    model = _Model([_mimo_attention(dpa0), _mimo_attention(dpa1)])
    mesh = _cp_mesh()
    stream = object()
    ranks_patch, stream_patch = _setup_patches(stream)

    with ranks_patch, stream_patch:
        result = setup_mimo_te_context_parallel(model, mesh)

    assert result is None
    assert model._mimo_te_cp_configured_group is mesh.get_group()
    for dpa in (dpa0, dpa1):
        assert len(dpa.calls) == 1
        args, kwargs = dpa.calls[0]
        assert args == (mesh.get_group.return_value, [0, 1], stream)
        assert kwargs == {"cp_comm_type": "a2a"}


def test_setup_mimo_te_cp_unwraps_checkpointed_attention():
    dpa = _FakeDotProductAttention()
    wrapper = torch.nn.Module()
    wrapper._checkpoint_wrapped_module = _mimo_attention(dpa)
    model = _Model([wrapper])
    stream = object()
    ranks_patch, stream_patch = _setup_patches(stream)

    with ranks_patch, stream_patch:
        result = setup_mimo_te_context_parallel(model, _cp_mesh())

    assert result is None
    assert dpa.calls[0][1]["cp_comm_type"] == "a2a"


def test_setup_mimo_te_cp_rejects_non_te_attention():
    model = _Model([_mimo_attention(torch.nn.Identity(), backend="torch")])
    stream = object()
    ranks_patch, stream_patch = _setup_patches(stream)

    with ranks_patch, stream_patch:
        with pytest.raises(ValueError, match="requires backend.attn='te'"):
            setup_mimo_te_context_parallel(model, _cp_mesh())


@pytest.mark.parametrize(
    ("query_heads", "kv_heads"),
    [
        (7, 4),
        (8, 3),
    ],
)
def test_setup_mimo_te_cp_validates_query_and_kv_head_divisibility(query_heads, kv_heads):
    dpa = _FakeDotProductAttention(query_heads=query_heads, kv_heads=kv_heads)
    model = _Model([_mimo_attention(dpa, query_heads=query_heads, kv_heads=kv_heads)])
    stream = object()
    ranks_patch, stream_patch = _setup_patches(stream)

    with ranks_patch, stream_patch:
        with pytest.raises(ValueError, match="requires both query and key/value head counts to be divisible"):
            setup_mimo_te_context_parallel(model, _cp_mesh(size=2))

    assert dpa.calls == []


def test_setup_mimo_te_cp_is_noop_without_active_cp():
    model = _Model([torch.nn.Identity()])

    assert setup_mimo_te_context_parallel(model, _cp_mesh(size=1)) is None
    assert model._mimo_te_cp_configured_group is None


def test_ensure_mimo_te_cp_configures_a2a_only_once():
    dpa = _FakeDotProductAttention()
    model = _Model([_mimo_attention(dpa)])
    group = object()
    first_mesh = _cp_mesh(group=group)
    second_mesh = _cp_mesh(group=group)
    stream = object()
    ranks_patch, stream_patch = _setup_patches(stream)

    with ranks_patch, stream_patch:
        ensure_mimo_te_context_parallel(model, first_mesh)
        ensure_mimo_te_context_parallel(model, second_mesh)

    assert len(dpa.calls) == 1

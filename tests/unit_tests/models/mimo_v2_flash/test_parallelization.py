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

from unittest.mock import Mock, patch

import pytest
import torch

from nemo_automodel.components.models.mimo_v2_flash.parallelization import (
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


class _FakeSelfAttention(torch.nn.Module):
    def __init__(self, attn_module, query_heads=8, kv_heads=4):
        super().__init__()
        self.attn_module = attn_module
        self.num_attention_heads = query_heads
        self.num_key_value_heads = kv_heads


class _Block(torch.nn.Module):
    def __init__(self, self_attn):
        super().__init__()
        self.self_attn = self_attn


class _Model(torch.nn.Module):
    def __init__(self, attentions):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(attn) for attn in attentions])
        self.cp_mesh = None


def _cp_mesh(size=2):
    mesh = Mock()
    mesh.size.return_value = size
    mesh.get_group.return_value = object()
    return mesh


def _setup_patches(stream):
    return (
        patch(
            "nemo_automodel.components.models.mimo_v2_flash.parallelization.safe_import_from",
            return_value=(True, _FakeDotProductAttention),
        ),
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        patch("torch.cuda.Stream", return_value=stream),
    )


def test_setup_mimo_te_cp_forces_a2a_on_every_attention():
    dpa0 = _FakeDotProductAttention()
    dpa1 = _FakeDotProductAttention()
    model = _Model([_FakeSelfAttention(dpa0), _FakeSelfAttention(dpa1)])
    mesh = _cp_mesh()
    stream = object()
    import_patch, ranks_patch, stream_patch = _setup_patches(stream)

    with import_patch, ranks_patch, stream_patch:
        configured = setup_mimo_te_context_parallel(model, mesh)

    assert configured == 2
    assert model.cp_mesh is mesh
    for dpa in (dpa0, dpa1):
        assert len(dpa.calls) == 1
        args, kwargs = dpa.calls[0]
        assert args == (mesh.get_group.return_value, [0, 1], stream)
        assert kwargs == {"cp_comm_type": "a2a"}


def test_setup_mimo_te_cp_unwraps_checkpointed_attention():
    dpa = _FakeDotProductAttention()
    wrapper = torch.nn.Module()
    wrapper._checkpoint_wrapped_module = _FakeSelfAttention(dpa)
    model = _Model([wrapper])
    stream = object()
    import_patch, ranks_patch, stream_patch = _setup_patches(stream)

    with import_patch, ranks_patch, stream_patch:
        configured = setup_mimo_te_context_parallel(model, _cp_mesh())

    assert configured == 1
    assert dpa.calls[0][1]["cp_comm_type"] == "a2a"


def test_setup_mimo_te_cp_rejects_non_te_attention():
    model = _Model([_FakeSelfAttention(torch.nn.Identity())])
    stream = object()
    import_patch, ranks_patch, stream_patch = _setup_patches(stream)

    with import_patch, ranks_patch, stream_patch:
        with pytest.raises(ValueError, match="only supports Transformer Engine DotProductAttention"):
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
    model = _Model([_FakeSelfAttention(dpa, query_heads=query_heads, kv_heads=kv_heads)])
    stream = object()
    import_patch, ranks_patch, stream_patch = _setup_patches(stream)

    with import_patch, ranks_patch, stream_patch:
        with pytest.raises(ValueError, match="query and KV head counts divisible by cp_size"):
            setup_mimo_te_context_parallel(model, _cp_mesh(size=2))

    assert dpa.calls == []


def test_setup_mimo_te_cp_is_noop_without_active_cp():
    model = _Model([_FakeSelfAttention(torch.nn.Identity())])

    assert setup_mimo_te_context_parallel(model, _cp_mesh(size=1)) == 0
    assert model.cp_mesh is None

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

"""CP exchange selection for sliding-window layers: a2a for packed (thd) input, all_gather for batched input."""

import pytest
import torch

from nemo_automodel.components.models.beam.layers import BeamAttention


class _FakeTECPModule(torch.nn.Module):
    """Records TE DotProductAttention.set_context_parallel_group calls as apply_cp leaves them."""

    def __init__(self, cp_comm_type):
        super().__init__()
        self.cp_group, self.cp_global_ranks, self.cp_stream = object(), [0, 1], object()
        self.cp_comm_type = cp_comm_type
        self.calls = []

    def set_context_parallel_group(self, cp_group, cp_global_ranks, cp_stream, cp_comm_type="p2p"):
        self.calls.append((cp_group, cp_global_ranks, cp_stream, cp_comm_type))
        self.cp_group, self.cp_global_ranks, self.cp_stream = cp_group, cp_global_ranks, cp_stream
        self.cp_comm_type = cp_comm_type


def _attention(beam_config, backend_config, layer_idx, cp_comm_type, cp_size, monkeypatch):
    attn = BeamAttention(beam_config, layer_idx, backend_config)
    attn.attn_module = _FakeTECPModule(cp_comm_type)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: cp_size)
    return attn


def test_sliding_layer_uses_a2a_for_packed_and_all_gather_for_batched(beam_config, backend_config, monkeypatch):
    attn = _attention(beam_config, backend_config, 0, "all_gather", 2, monkeypatch)
    te = attn.attn_module
    group, ranks, stream = te.cp_group, te.cp_global_ranks, te.cp_stream

    attn._set_sliding_cp_transport("thd")
    assert te.calls == [(group, ranks, stream, "a2a")]
    attn._set_sliding_cp_transport("thd")  # already a2a: no second call
    assert len(te.calls) == 1

    attn._set_sliding_cp_transport("bshd")
    assert te.calls[-1] == (group, ranks, stream, "all_gather")
    assert len(te.calls) == 2


def test_full_attention_layer_keeps_p2p(beam_config, backend_config, monkeypatch):
    attn = _attention(beam_config, backend_config, 1, "p2p", 2, monkeypatch)
    attn._set_sliding_cp_transport("thd")
    assert attn.attn_module.calls == [] and attn.attn_module.cp_comm_type == "p2p"


def test_no_cp_group_is_a_no_op(beam_config, backend_config, monkeypatch):
    attn = _attention(beam_config, backend_config, 0, "all_gather", 2, monkeypatch)
    attn.attn_module.cp_group = None
    attn._set_sliding_cp_transport("thd")
    assert attn.attn_module.calls == []


def test_packed_cp_size_must_divide_head_counts(beam_config, backend_config, monkeypatch):
    # tiny config: 4 query heads, 2 KV heads -> CP4 does not divide the KV heads
    attn = _attention(beam_config, backend_config, 0, "all_gather", 4, monkeypatch)
    with pytest.raises(ValueError, match="a2a"):
        attn._set_sliding_cp_transport("thd")
    assert attn.attn_module.calls == []
    attn._set_sliding_cp_transport("bshd")  # all_gather has no head-count constraint
    assert attn.attn_module.calls == []

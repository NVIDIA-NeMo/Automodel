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

"""CPU unit tests for EAGLE-3 target-model context parallelism.

Covers the pieces verifiable without multiple GPUs: the target wrapper's
``cp_mesh`` handling + self_attn hook attachment, and the recipe-level CP gates.
The real ring-attention forward + sequence gather (``make_target_cp_ctx`` /
``gather_cp_seq``) requires a multi-GPU NCCL run and is validated on the server.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel
from nemo_automodel.recipes.llm.train_eagle3 import _validate_cp_gates


def _tiny_target(num_hidden_layers: int = 4) -> LlamaForCausalLM:
    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=32,
    )
    config.torch_dtype = torch.float32
    return LlamaForCausalLM(config).to(torch.float32).eval()


def _self_attn_modules(target: HFEagle3TargetModel):
    return [m for name, m in target.model.named_modules() if name.endswith("self_attn")]


# --------------------------------------------------------------------------- #
# Target wrapper cp_mesh handling
# --------------------------------------------------------------------------- #
def test_cp_mesh_none_is_single_cp_and_attaches_no_hooks():
    target = HFEagle3TargetModel(_tiny_target(), aux_layer_ids=[1, 2, 3])
    assert target.cp_mesh is None
    assert target._cp_size == 1
    assert all(len(m._forward_pre_hooks) == 0 for m in _self_attn_modules(target))


def test_cp_mesh_size_one_does_not_attach_hooks():
    mesh = SimpleNamespace(size=lambda: 1)
    target = HFEagle3TargetModel(_tiny_target(), aux_layer_ids=[1, 2, 3], cp_mesh=mesh)
    assert target._cp_size == 1
    assert all(len(m._forward_pre_hooks) == 0 for m in _self_attn_modules(target))


def test_cp_mesh_size_gt_one_attaches_cp_hooks_on_every_self_attn():
    mesh = SimpleNamespace(size=lambda: 2)
    target = HFEagle3TargetModel(_tiny_target(num_hidden_layers=4), aux_layer_ids=[1, 2, 3], cp_mesh=mesh)
    assert target._cp_size == 2
    attns = _self_attn_modules(target)
    assert len(attns) == 4
    assert all(len(m._forward_pre_hooks) >= 1 for m in attns)


# --------------------------------------------------------------------------- #
# cp_utils helpers exist and are wired
# --------------------------------------------------------------------------- #
def test_cp_utils_helpers_importable():
    from nemo_automodel.components.distributed.cp_utils import gather_cp_seq, make_target_cp_ctx

    assert callable(make_target_cp_ctx)
    assert callable(gather_cp_seq)


# --------------------------------------------------------------------------- #
# Recipe CP gates
# --------------------------------------------------------------------------- #
def test_cp_gate_allows_cp_size_one():
    # cp_size==1 must never raise, even with packing or the remote backend.
    _validate_cp_gates(cp_size=1, backend="remote", packed_sequence_size=4)
    _validate_cp_gates(cp_size=1, backend="colocated", packed_sequence_size=0)


def test_cp_gate_rejects_remote_backend():
    with pytest.raises(NotImplementedError, match="remote backend runs the target out-of-process"):
        _validate_cp_gates(cp_size=2, backend="remote", packed_sequence_size=0)


def test_cp_gate_rejects_sequence_packing():
    with pytest.raises(NotImplementedError, match="sequence packing"):
        _validate_cp_gates(cp_size=2, backend="colocated", packed_sequence_size=4)

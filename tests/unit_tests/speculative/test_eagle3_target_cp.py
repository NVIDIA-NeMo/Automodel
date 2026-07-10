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
``cp_mesh`` handling + self_attn hook attachment, the recipe-level CP gates, and
the sequence shard/gather helpers (``make_target_cp_ctx`` / ``gather_cp_seq``)
with the torch context-parallel primitives mocked. The real multi-rank ring
attention is validated on the server.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import nemo_automodel.components.speculative.target_cp as target_cp
import nemo_automodel.recipes.llm.train_eagle3 as train_eagle3
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
# target_cp helpers exist and are wired
# --------------------------------------------------------------------------- #
def test_target_cp_helpers_importable():
    from nemo_automodel.components.speculative.target_cp import gather_cp_seq, make_target_cp_ctx

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


# --------------------------------------------------------------------------- #
# make_target_cp_ctx: sequence padding + buffer construction (CP primitive mocked)
# --------------------------------------------------------------------------- #
def test_make_target_cp_ctx_pads_and_generates_position_ids(monkeypatch):
    """T not divisible by cp_size is right-padded; position_ids default to an arange."""
    seen = {}

    def fake_cp(cp_mesh, buffers, buffer_seq_dims, no_restore_buffers):
        seen["dims"] = buffer_seq_dims
        seen["n_buffers"] = len(buffers)
        return contextlib.nullcontext()

    monkeypatch.setattr("torch.distributed.tensor.experimental.context_parallel", fake_cp)
    mesh = SimpleNamespace(size=lambda: 4)
    input_ids = torch.arange(10).view(1, 10)  # 10 -> pad to 12 (multiple of 4)
    ctx, ids_buf, pos_buf, orig_len = target_cp.make_target_cp_ctx(mesh, input_ids, position_ids=None)

    assert orig_len == 10
    assert ids_buf.shape == (1, 12)
    assert pos_buf.shape == (1, 12)
    assert torch.equal(pos_buf[0, :10], torch.arange(10))
    assert seen == {"dims": [1, 1], "n_buffers": 2}
    with ctx:
        pass


def test_make_target_cp_ctx_no_pad_clones_and_expands_position_ids(monkeypatch):
    """Already-aligned T is cloned (fresh buffers); a [1, T] position_ids expands to batch."""
    monkeypatch.setattr(
        "torch.distributed.tensor.experimental.context_parallel",
        lambda cp_mesh, **kw: contextlib.nullcontext(),
    )
    mesh = SimpleNamespace(size=lambda: 2)
    input_ids = torch.arange(16).view(2, 8)  # 8 already a multiple of 2 -> no pad
    pos = torch.arange(8).view(1, 8)
    ctx, ids_buf, pos_buf, orig_len = target_cp.make_target_cp_ctx(mesh, input_ids, position_ids=pos)

    assert orig_len == 8
    assert ids_buf.shape == (2, 8)
    assert pos_buf.shape == (2, 8)  # expanded from [1, 8]
    assert ids_buf is not input_ids  # cloned, so the caller's tensor is not sharded in place
    with ctx:
        pass


# --------------------------------------------------------------------------- #
# gather_cp_seq: unshard + slice back to the pre-pad length (CP primitive mocked)
# --------------------------------------------------------------------------- #
def test_gather_cp_seq_unshards_and_slices_to_orig_len(monkeypatch):
    monkeypatch.setattr(
        "torch.distributed.tensor.experimental._attention.context_parallel_unshard",
        lambda cp_mesh, local_tensors, seq_dims: list(local_tensors),
    )
    mesh = SimpleNamespace(size=lambda: 2)
    aux = torch.randn(1, 8, 4)
    logits = torch.randn(1, 8, 6)
    out = target_cp.gather_cp_seq(mesh, [aux, logits], seq_dim=1, orig_len=6)

    assert out[0].shape == (1, 6, 4)
    assert out[1].shape == (1, 6, 6)


# --------------------------------------------------------------------------- #
# generate_batch CP branch: shard -> forward -> gather, with the CP helpers mocked
# --------------------------------------------------------------------------- #
def test_generate_batch_cp_branch_gathers_full_sequence(monkeypatch):
    target = HFEagle3TargetModel(
        _tiny_target(num_hidden_layers=4),
        aux_layer_ids=[1, 2, 3],
        cp_mesh=SimpleNamespace(size=lambda: 2),
    )

    def fake_make_ctx(cp_mesh, input_ids, position_ids):
        pos = position_ids if position_ids is not None else torch.arange(input_ids.shape[1]).unsqueeze(0)
        return contextlib.nullcontext(), input_ids, pos, input_ids.shape[1]

    def fake_gather(cp_mesh, tensors, seq_dim, orig_len):
        return [t.narrow(seq_dim, 0, orig_len).contiguous() for t in tensors]

    monkeypatch.setattr(target_cp, "make_target_cp_ctx", fake_make_ctx)
    monkeypatch.setattr(target_cp, "gather_cp_seq", fake_gather)

    b, t = 1, 8
    input_ids = torch.randint(0, 64, (b, t))
    attention_mask = torch.ones(b, t, dtype=torch.long)
    loss_mask = torch.ones(b, t, dtype=torch.long)
    out = target.generate_batch(input_ids, attention_mask, loss_mask)

    # aux = concat of the 3 captured layers (hidden_size 16 each) -> 48
    assert out.aux_hidden_states.shape == (b, t, 48)
    assert out.logits.shape[:2] == (b, t)


def test_check_captured_raises_on_layer_count_mismatch():
    target = HFEagle3TargetModel(_tiny_target(), aux_layer_ids=[1, 2, 3])
    with pytest.raises(RuntimeError, match="Expected 3 captured aux layers"):
        target._check_captured({1: torch.zeros(1)})


# --------------------------------------------------------------------------- #
# _submesh_or_none: None mesh, present axis, and a missing axis (KeyError -> None)
# --------------------------------------------------------------------------- #
def test_submesh_or_none_handles_none_present_and_missing(monkeypatch):
    assert train_eagle3._submesh_or_none(None, "cp") is None

    monkeypatch.setattr(train_eagle3, "get_flat_mesh", lambda mesh, name: ("submesh", name))
    assert train_eagle3._submesh_or_none(object(), "dp") == ("submesh", "dp")

    def _raise(mesh, name):
        raise KeyError(name)

    monkeypatch.setattr(train_eagle3, "get_flat_mesh", _raise)
    assert train_eagle3._submesh_or_none(object(), "cp") is None

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

"""CPU unit tests for EAGLE-3 target-model tensor parallelism.

Covers the pieces verifiable without multiple GPUs: the recipe-level TP gates and
the target wrapper's gather of a tensor-parallel (vocab-sharded ``DTensor``)
logits output back to a plain tensor for the draft. The real multi-rank TP
sharding is validated on the server.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel, _to_full_tensor
from nemo_automodel.recipes.llm.train_eagle3 import _validate_tp_gates


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


@pytest.fixture
def single_rank_pg():
    """A single-rank gloo process group so DTensor / TP primitives run on CPU."""
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    already = dist.is_initialized()
    if not already:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield
    finally:
        if not already:
            dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Recipe TP gates
# --------------------------------------------------------------------------- #
def test_tp_gate_allows_tp_size_one():
    # tp_size==1 must never raise, regardless of backend or cp_size.
    _validate_tp_gates(tp_size=1, backend="remote", cp_size=4)
    _validate_tp_gates(tp_size=1, backend="colocated", cp_size=1)


def test_tp_gate_allows_colocated():
    _validate_tp_gates(tp_size=2, backend="colocated", cp_size=1)


def test_tp_gate_rejects_remote_backend():
    with pytest.raises(NotImplementedError, match="only supported with the colocated target backend"):
        _validate_tp_gates(tp_size=2, backend="remote", cp_size=1)


def test_tp_gate_rejects_combined_with_cp():
    with pytest.raises(NotImplementedError, match="context parallelism"):
        _validate_tp_gates(tp_size=2, backend="colocated", cp_size=2)


# --------------------------------------------------------------------------- #
# _to_full_tensor: gather a TP-sharded DTensor, no-op on a plain tensor
# --------------------------------------------------------------------------- #
def test_to_full_tensor_is_noop_on_plain_tensor():
    plain = torch.randn(2, 3)
    assert _to_full_tensor(plain) is plain


def test_to_full_tensor_gathers_vocab_sharded_dtensor(single_rank_pg):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    full = torch.randn(2, 4, 6)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    sharded = distribute_tensor(full, mesh, [Shard(-1)])  # mimics a column-parallel lm_head
    assert hasattr(sharded, "full_tensor")

    out = _to_full_tensor(sharded)
    assert not hasattr(out, "full_tensor")  # now a plain tensor
    torch.testing.assert_close(out, full)


# --------------------------------------------------------------------------- #
# generate_batch under a tensor-parallel (DTensor) lm_head
# --------------------------------------------------------------------------- #
def test_generate_batch_gathers_tp_sharded_logits(single_rank_pg):
    """A column-parallel lm_head yields vocab-sharded DTensor logits; the wrapper
    must hand the draft a plain (gathered) tensor, not a DTensor."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    torch.manual_seed(0)
    model = _tiny_target(num_hidden_layers=4)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    # Tensor-parallel only the lm_head -> vocab-sharded DTensor logits (the TP plan
    # leaves the decoder residual stream replicated, so aux hidden states stay plain).
    parallelize_module(model, mesh, {"lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)})
    assert hasattr(model(input_ids=torch.randint(0, 64, (1, 8))).logits, "full_tensor")

    wrapper = HFEagle3TargetModel(model, aux_layer_ids=[1, 2, 3])
    b, t = 1, 8
    batch = wrapper.generate_batch(
        input_ids=torch.randint(0, 64, (b, t)),
        attention_mask=torch.ones(b, t, dtype=torch.long),
        loss_mask=torch.ones(b, t, dtype=torch.long),
    )

    assert not hasattr(batch.logits, "full_tensor")  # gathered to a plain tensor
    assert batch.logits.shape[:2] == (b, t)
    assert torch.isfinite(batch.logits).all()
    # aux hidden states are 3 concatenated layers of hidden_size 16 -> 48.
    assert batch.aux_hidden_states.shape == (b, t, 48)

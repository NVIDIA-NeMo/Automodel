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

"""CPU unit tests for EAGLE-1 / EAGLE-2 target-model tensor parallelism.

Covers the pieces verifiable without multiple GPUs: the target wrapper's gather
of a tensor-parallel (vocab-sharded ``DTensor``) logits output back to a plain
tensor, and the ``dp`` submesh resolution the recipe keys its draft DDP group /
dataloader sampler on. The real multi-rank TP sharding is validated on the server.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

import nemo_automodel.recipes.llm.train_eagle1 as train_eagle1
from nemo_automodel.components.speculative.eagle.target_v12 import HFEagleTargetModel, _to_full_tensor


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
# _to_full_tensor
# --------------------------------------------------------------------------- #
def test_to_full_tensor_is_noop_on_plain_tensor():
    plain = torch.randn(2, 3)
    assert _to_full_tensor(plain) is plain


def test_to_full_tensor_gathers_vocab_sharded_dtensor(single_rank_pg):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    full = torch.randn(2, 4, 6)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    sharded = distribute_tensor(full, mesh, [Shard(-1)])
    assert hasattr(sharded, "full_tensor")

    out = _to_full_tensor(sharded)
    assert not hasattr(out, "full_tensor")
    torch.testing.assert_close(out, full)


# --------------------------------------------------------------------------- #
# generate_batch under a tensor-parallel (DTensor) lm_head
# --------------------------------------------------------------------------- #
def test_generate_batch_gathers_tp_sharded_logits(single_rank_pg):
    """A column-parallel lm_head yields vocab-sharded DTensor logits; the wrapper
    must hand the draft a plain (gathered) tensor. Hidden states stay plain."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    torch.manual_seed(0)
    model = _tiny_target(num_hidden_layers=4)
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    parallelize_module(model, mesh, {"lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)})
    assert hasattr(model(input_ids=torch.randint(0, 64, (1, 8))).logits, "full_tensor")

    wrapper = HFEagleTargetModel(model)
    b, t = 1, 8
    batch = wrapper.generate_batch(
        input_ids=torch.randint(0, 64, (b, t)),
        attention_mask=torch.ones(b, t, dtype=torch.long),
        loss_mask=torch.ones(b, t, dtype=torch.long),
    )

    assert not hasattr(batch.target_logits, "full_tensor")  # gathered to plain
    assert torch.isfinite(batch.target_logits).all()
    assert batch.target_logits.shape[:2] == (b, t)
    # Hidden states are replicated under the default plan -> plain tensors.
    assert not hasattr(batch.input_hidden_states, "full_tensor")
    assert not hasattr(batch.target_hidden_states, "full_tensor")


# --------------------------------------------------------------------------- #
# _submesh_or_none: None mesh, present axis, missing axis (KeyError -> None)
# --------------------------------------------------------------------------- #
def test_submesh_or_none_handles_none_present_and_missing(monkeypatch):
    assert train_eagle1._submesh_or_none(None, "dp") is None

    monkeypatch.setattr(train_eagle1, "get_flat_mesh", lambda mesh, name: ("submesh", name))
    assert train_eagle1._submesh_or_none(object(), "dp") == ("submesh", "dp")

    def _raise(mesh, name):
        raise KeyError(name)

    monkeypatch.setattr(train_eagle1, "get_flat_mesh", _raise)
    assert train_eagle1._submesh_or_none(object(), "dp") is None

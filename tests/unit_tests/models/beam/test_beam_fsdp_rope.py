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

"""RoPE precision and sharing across real FSDP2 boundaries and checkpoint recomputation."""

import copy

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed.activation_checkpointing import apply_full_layer_checkpointing_to_layers
from nemo_automodel.components.models.beam.parallelization import PARALLELIZER, beam_fully_shard_unit

from .utils import capture_rope_calls, hf_model_and_checkpoint, native_from_hf

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 needs a CUDA device")

# Whichever FSDP test runs first in a process (test selection decides which one) creates the single-rank NCCL
# communicator on its first collective; measured up to 6.5 s on H200, the warm tests take 0.2-1.4 s.
first_nccl_collective = pytest.mark.runtime_budget(
    20, reason="The first FSDP test in a process creates the NCCL communicator on its first collective."
)


@pytest.fixture(scope="module")
def mesh():
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
    finally:
        if created:
            dist.destroy_process_group()


def _policy(param_dtype=torch.bfloat16):
    # the default strategy's resolved policy (cast_forward_inputs defaults to True)
    return MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=torch.float32)


def _build(cfg, backend, storage_dtype, ac):
    cfg = copy.deepcopy(cfg)
    cfg.torch_dtype = storage_dtype
    _, sd = hf_model_and_checkpoint(cfg, torch.device("cuda"))
    model = native_from_hf(cfg, backend, sd, torch.device("cuda"))
    if storage_dtype == torch.bfloat16:
        model = model.to(torch.bfloat16)
        for layer in model.model.layers.values():  # keep the fp32 router contract
            if hasattr(layer.mlp, "gate"):
                layer.mlp.gate.float()
    model.train()
    if ac:
        apply_full_layer_checkpointing_to_layers(model, list(model.model.layers.values()))
    return model


def _shard(model, mesh, policy):
    """Shard through Beam's dense FSDP2 path (dtype-aware blocks), then shard the root."""
    PARALLELIZER._apply_fsdp_sharding(model, mesh, policy)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
    return model


def _run(model, ids):
    """Capture rotary calls through forward and backward.

    Args:
        model: Native Beam model, optionally sharded and checkpointed.
        ids: Integer token tensor [batch, sequence].

    Returns:
        Detached logits [batch, sequence, vocab], observed calls, and the forward call count.
    """
    calls = capture_rope_calls(model)
    out = model(ids).logits.float()
    n_forward = len(calls)
    F.cross_entropy(out[:, :-1].reshape(-1, out.shape[-1]), ids[:, 1:].reshape(-1)).backward()
    return out.detach(), calls, n_forward


@first_nccl_collective
@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float32], ids=["bf16_storage", "fp32_master_weights"])
@pytest.mark.parametrize("ac", [False, True], ids=["no_ac", "ac"])
def test_tables_stay_fp32_shared_and_exact_across_fsdp_boundary(beam_config, backend_config, mesh, storage, ac):
    model = _shard(_build(beam_config, backend_config, storage, ac), mesh, _policy())
    ids = torch.randint(0, beam_config.vocab_size, (2, 16), generator=torch.Generator().manual_seed(5)).cuda()
    out, calls, n_forward = _run(model, ids)
    n_rope = sum(1 for lt in beam_config.layer_types if lt == "sliding_attention")
    assert n_forward == n_rope and len(calls) == (2 * n_rope if ac else n_rope)  # AC: the recompute calls again
    for rec in calls:
        assert rec.x_dtype == torch.bfloat16, rec  # compute dtype, also with fp32 master weights
        assert rec.dtypes == (torch.float32, torch.float32), rec
        assert rec.exact, rec
    assert len({rec.storages for rec in calls}) == 1  # one pair, shared by every RoPE layer, also in recompute
    for block in model.model.layers.values():
        inner = getattr(block, "_checkpoint_wrapped_module", block)
        assert inner._fsdp_compute_dtype == torch.bfloat16
        if hasattr(inner.mlp, "gate"):  # MoE block: the fp32 router is its own FSDP unit
            assert isinstance(inner.mlp.gate, FSDPModule)
            assert inner.mlp.gate.weight.dtype == torch.float32
    assert torch.isfinite(out).all()


@first_nccl_collective
def test_fsdp_logits_equal_the_unsharded_bf16_model(beam_config, backend_config, mesh):
    """Single-rank FSDP with the Beam units computes the same numbers as the plain bf16 model (bit-exact)."""
    ref = _build(beam_config, backend_config, torch.bfloat16, False).eval()
    model = _shard(copy.deepcopy(ref), mesh, _policy()).eval()
    ids = torch.randint(0, beam_config.vocab_size, (2, 16), generator=torch.Generator().manual_seed(6)).cuda()
    with torch.no_grad():
        assert torch.equal(model(ids).logits.float(), ref(ids).logits.float())


@first_nccl_collective
def test_no_policy_means_no_cast(beam_config, backend_config, mesh):
    """Without mixed precision FSDP casts nothing and the block records no compute dtype (uniform fp32 storage, so the
    blocks are plain units; the dtype-aware helper itself requires a policy and is never called without one by the
    recipe paths)."""
    model = _build(beam_config, backend_config, torch.float32, False)
    policy = MixedPrecisionPolicy()  # FSDP2's "no mixed precision": param_dtype None
    for block in model.model.layers.values():
        beam_fully_shard_unit(block, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
    ids = torch.randint(0, beam_config.vocab_size, (2, 16), generator=torch.Generator().manual_seed(7)).cuda()
    _, calls, _ = _run(model, ids)
    assert all(rec.x_dtype == torch.float32 and rec.exact for rec in calls)
    assert all(
        getattr(getattr(b, "_checkpoint_wrapped_module", b), "_fsdp_compute_dtype", None) is None
        for b in model.model.layers.values()
    )

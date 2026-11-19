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

"""Multi-rank checks of ERNIE 5.5 under its parallelization sidecar: Gloo ranks on CPU, no GPU needed.

``test_two_rank_gradient_presence_and_reduced_gradients`` (two ranks) runs a small FP32 model with one HCA layer
and a dense MLP. A row of 64 tokens has no complete compression window and does not use the HCA compressor; a row
of 160 tokens does.

1. Gradient presence. FSDP2 reduces a parameter only on the ranks that hold a gradient for it, so every rank must
   hold a gradient for the same parameters. On models that are not sharded, where no collective can block, one
   rank runs a short row and the other a long row. With the synchronization group that the sidecar sets on the
   compressor, both ranks hold a gradient for the same parameters; without the group the check reports the
   compressor parameters, which shows that it detects a gradient that exists on some ranks only.
2. Reduced gradients. The modules are sharded through the sidecar's ``_fully_shard_module`` in the order of the
   shared expert-parallel flow, and the reduced gradients agree with those of one process that sees both rows to
   the rounding of FP32, for a step with a short and a long row and for a step with two short rows.

The two checkpoint tests run ``parallelize_model`` with a mesh context built as a recipe builds it, on a BF16
model with every layer kind (HCA with a dense MLP, HCA with an MoE, DSA with an MoE) and ``ep_size`` 2. A Hugging
Face layout checkpoint is loaded through the ``Checkpointer`` into the parallelized model, every native tensor is
compared with the source, the model is saved in the raw format, zeroed and restored, and compared again. Before
that, each test asserts the layout that the round trip depends on: the mesh and placements of the routed-expert
weights, and that the key and value projections of the DSA layer are sharded over all ranks of the job.

* ``test_two_rank_checkpoint_round_trip_with_one_rank_per_expert_group`` (two ranks): the routed-expert weights
  are on the one-dimensional ``ep`` mesh.
* ``test_four_rank_checkpoint_round_trip_with_two_ranks_per_expert_group`` (four ranks): they are on the
  ``(ep_shard, ep)`` mesh.
"""

import copy
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed import FSDP2Config, MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.hca import Ernie5_5HCAAttention
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.parallelization import PARALLELIZER

SHORT, LONG, VOCAB, WORLD_SIZE = 64, 160, 64, 2
# Modules of a decoder block that compute in FP32 and are FSDP units of their own in the released BF16 model.
FP32_UNITS = (
    "attn_hc",
    "ffn_hc",
    "attention.vha",
    "attention.sinks_param",
    "attention.compressor.wkv",
    "attention.compressor.wgate",
    "attention.compressor.ape_param",
)
# Tolerances of the reduced gradients against one process that sees both rows. Both sides run the same FP32 graph
# and the mean over two ranks is exact, so a step of two short rows equals the reference except where a halved
# value underflows (differences below 1e-44). In a step with a short and a long row, the rank with the short row
# also runs the compressor on a masked window, which keeps the collectives of the two ranks the same and changes
# the order of the FP32 operations of that row: measured on two ranks, 1,709 of 5,785 gradient elements differ, by
# at most 9.6e-7, and the comparison needs rtol=8.4e-7 at this atol. The bound is about 24 times that. A missing
# rank contribution or a wrong divisor is an error of the order of the gradient itself.
GRADIENT_RTOL, GRADIENT_ATOL = 2e-5, 1e-7


def _model(ratios: tuple[int, ...], dtype: str) -> Ernie5_5_MoeForCausalLM:
    """Build a small model on the eager attention path.

    Args:
        ratios: Compression ratio of each decoder layer: 128 for an HCA layer, -2 for a DSA layer. The first layer
            has a dense MLP, the others an MoE with four routed experts.
        dtype: Model dtype, ``"float32"`` or ``"bfloat16"``.
    """
    config = Ernie5_5Config(
        vocab_size=VOCAB,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=len(ratios),
        max_position_embeddings=1024,
        compress_ratios=list(ratios),
        num_nextn_predict_layers=0,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=6,
        o_lora_rank=4,
        o_groups=2,
        sliding_window=8,
        # Four DSA heads, so that two and four ranks hold whole heads of the fused key and value projection.
        hybrid_mla_num_attention_heads=4,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=4,
        hybrid_mla_qk_nope_head_dim=4,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        rope_interleave=True,
        hc_mult=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        moe_latent_size=8,
        pad_token_id=0,
        torch_dtype=dtype,
    )
    backend = BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")
    return Ernie5_5_MoeForCausalLM(config, backend)


def _tiny_model() -> Ernie5_5_MoeForCausalLM:
    """Build an FP32 model with one HCA layer, a dense MLP, two residual streams and seeded weights."""
    model = _model((128,), "float32")
    torch.manual_seed(5583)
    model.initialize_weights(dtype=torch.float32)
    return model


def _loss(model: Ernie5_5_MoeForCausalLM, rank: int, length: int) -> torch.Tensor:
    """Run one row of ``length`` tokens that differs per rank and return a scalar loss.

    Args:
        model: Model to run; its logits have shape [1, length, vocab].
        rank: Rank whose row is used.
        length: Number of tokens in the row.

    Returns:
        Sum of the squared logits.
    """
    tokens = torch.randint(1, VOCAB, (1, length), generator=torch.Generator().manual_seed(100 + rank))
    return model(tokens).logits.square().sum()


def _assert_gradients_exist_on_every_rank_or_none(presence: list[dict[str, bool]]) -> None:
    """Fail when a trainable parameter holds a gradient on some ranks and none on others.

    Args:
        presence: Per rank, whether each trainable parameter holds a gradient after backward.
    """
    partial = sorted(name for name in presence[0] if len({flags[name] for flags in presence}) > 1)
    assert not partial, f"Parameters with a gradient on some ranks only: {partial}"


def _check_gradient_presence(rank: int) -> None:
    for group, uniform in ((dist.group.WORLD, True), (None, False)):
        model = _tiny_model()
        for module in model.modules():
            if isinstance(module, Ernie5_5HCAAttention):
                module.compressor._set_hca_param_sync_group(group)
        _loss(model, rank, (SHORT, LONG)[rank]).backward()
        local = {name: parameter.grad is not None for name, parameter in model.named_parameters()}
        presence: list[dict[str, bool]] = [{} for _ in range(WORLD_SIZE)]
        dist.all_gather_object(presence, local)
        if uniform:
            _assert_gradients_exist_on_every_rank_or_none(presence)
            assert all(local.values())
        else:
            with pytest.raises(AssertionError, match=r"on some ranks only: \['model\.layers\.0\.attention\.compressor"):
                _assert_gradients_exist_on_every_rank_or_none(presence)


def _check_reduced_gradients(rank: int) -> None:
    mesh = init_device_mesh("cpu", (WORLD_SIZE,), mesh_dim_names=("dp_shard",))
    policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
    for lengths in ((SHORT, LONG), (SHORT, SHORT)):
        model = _tiny_model()
        reference = copy.deepcopy(model)
        block = model.model.layers[0]
        # The call sequence of the shared flow: the FP32 modules inside the block, the block, the embedding, the
        # output head, the inner root and the outer root.
        units = [block.get_submodule(name) for name in FP32_UNITS]
        units += [block, model.model.embed_tokens, model.model.lm_head, model.model, model]
        for unit in units:
            PARALLELIZER._fully_shard_module(unit, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        assert all(isinstance(unit, FSDPModule) for unit in units)
        assert isinstance(model.model.hyper_head, FSDPModule)
        assert block.attention.compressor._hca_param_sync_group is mesh.get_group()

        for source in range(WORLD_SIZE):
            (_loss(reference, source, lengths[source]) / WORLD_SIZE).backward()
        _loss(model, rank, lengths[rank]).backward()

        expected = dict(reference.named_parameters())
        unused = set()
        for name, parameter in model.named_parameters():
            if expected[name].grad is None:
                assert parameter.grad is None, name
                unused.add(name)
            else:
                reduced = parameter.grad.full_tensor()
                torch.testing.assert_close(reduced, expected[name].grad, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)
        compressor = {name for name in expected if ".compressor." in name}
        assert len(compressor) == 4
        # A step with one long row gives every rank a compressor gradient; an all-short step gives none.
        assert unused == (set() if LONG in lengths else compressor)


def _worker(rank: int, init_method: str) -> None:
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=WORLD_SIZE, timeout=timedelta(seconds=60)
    )
    try:
        torch.set_num_threads(1)
        _check_gradient_presence(rank)
        _check_reduced_gradients(rank)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.timeout(300)
def test_two_rank_gradient_presence_and_reduced_gradients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        # Both ranks run on this host.
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=WORLD_SIZE, join=True)


EP_SIZE = 2
EXPERT_WEIGHTS = ("moe.experts.gate_and_up_projs", "moe.experts.down_projs")
# Mesh dimension names and placements of the routed-expert weights, by the number of ranks per expert group.
EXPERT_LAYOUT = {1: (("ep",), (Shard(0),)), 2: (("ep_shard", "ep"), (Shard(1), Shard(0)))}
# Key and value projections of a DSA layer, which a checkpoint stores as one matrix with whole heads per rank.
DSA_PROJECTIONS = ("attention.k_b_proj", "attention.v_b_proj")


def _source_model() -> Ernie5_5_MoeForCausalLM:
    """Build the three-layer BF16 model and fill every tensor with seeded values that BF16 represents exactly."""
    model = _model((128, 128, -2), "bfloat16")
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            tensor.copy_(torch.randn(tensor.shape, generator=generator).to(torch.bfloat16).to(tensor.dtype))
    return model


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the part of a tensor that this rank holds.

    Args:
        tensor: Plain tensor or DTensor of any shape.

    Returns:
        The local shard of a DTensor, which shares its storage, or ``tensor`` itself.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _assert_holds(model: Ernie5_5_MoeForCausalLM, expected: dict[str, torch.Tensor]) -> None:
    """Require that this rank's part of every native tensor has the expected dtype and values.

    Args:
        model: Parallelized model.
        expected: Native names mapped to this rank's part of the source tensor, of any shape.
    """
    state = model.state_dict()
    assert set(state) == set(expected)
    for name, value in expected.items():
        local = _local(state[name])
        assert local.dtype == value.dtype, name
        assert torch.equal(local, value), name


def _round_trip_worker(rank: int, world_size: int, root_text: str) -> None:
    root = Path(root_text)
    dist.init_process_group(
        "gloo",
        init_method=(root / "rendezvous").as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_num_threads(1)
        source = _source_model()
        checkpoint = root / "hf_checkpoint"
        if rank == 0:
            converted = source.state_dict_adapter.to_hf(source.state_dict())
            checkpoint.mkdir(parents=True)
            save_file(
                {name: value.to(torch.bfloat16).contiguous() for name, value in converted.items()},
                str(checkpoint / "model.safetensors"),
            )
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": dict.fromkeys(converted, "model.safetensors")})
            )
            source.config.save_pretrained(checkpoint)
        dist.barrier()

        model = _model((128, 128, -2), "bfloat16")
        context = MeshContext.build(FSDP2Config(), ParallelismSizes(ep_size=EP_SIZE), world_size=world_size)
        model = parallelize_model(model, context)
        state = model.state_dict()
        names, placements = EXPERT_LAYOUT[world_size // EP_SIZE]
        experts = [tensor for name, tensor in state.items() if name.endswith(EXPERT_WEIGHTS)]
        assert len(experts) == 4
        for tensor in experts:
            assert tuple(tensor.device_mesh.mesh_dim_names) == names
            assert tuple(tensor.placements) == placements
        # The DSA projections are sharded over all ranks of the job. The sidecar admits a number of ranks only if
        # it divides the number of DSA heads, which is what the checkpoint adapter requires of this mesh.
        projections = [tensor for name, tensor in state.items() if name.endswith(DSA_PROJECTIONS)]
        assert len(projections) == 2
        for tensor in projections:
            assert tensor.device_mesh.size() == world_size
            assert tuple(tensor.placements) == (Shard(0),)

        # This rank's part of every source tensor, under the mesh and placements the parallelizer gave the tensor.
        expected = {}
        for name, value in source.state_dict().items():
            tensor = state[name]
            assert isinstance(tensor, DTensor) and tuple(tensor.shape) == tuple(value.shape), name
            shard = distribute_tensor(value.detach().clone().contiguous(), tensor.device_mesh, list(tensor.placements))
            expected[name] = shard.to_local().clone()
        with torch.no_grad():
            for tensor in state.values():
                _local(tensor).fill_(float("nan"))

        base = CheckpointingConfig(
            enabled=False,
            save_consolidated=False,
            model_save_format="safetensors",
            model_repo_id="test/ernie5_5-tiny",
            checkpoint_dir=str(root / "run"),
        ).build(rank, 0, 0)
        try:
            base.load_base_model(model, torch.device("cpu"), str(root), str(checkpoint), load_base_model=True)
        finally:
            base.close()
        _assert_holds(model, expected)

        trained = CheckpointingConfig(
            enabled=True,
            save_consolidated=False,
            is_async=False,
            model_save_format="safetensors",
            model_repo_id=str(checkpoint),
            checkpoint_dir=str(root / "save_run"),
        ).build(rank, 0, 0)
        try:
            trained.save_model(model, str(root / "saved"), is_final_checkpoint=True)
            dist.barrier()
            with torch.no_grad():
                for tensor in model.state_dict().values():
                    _local(tensor).zero_()
            trained.load_model(model, model_path=str(root / "saved" / "model"), is_init_step=False)
        finally:
            trained.close()
        _assert_holds(model, expected)
    finally:
        dist.destroy_process_group()


def _run_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, world_size: int) -> None:
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        # All ranks run on this host.
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(_round_trip_worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.timeout(300)
def test_two_rank_checkpoint_round_trip_with_one_rank_per_expert_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two ranks with ``ep_size`` 2: the routed-expert weights are on the one-dimensional ``ep`` mesh."""
    _run_round_trip(monkeypatch, tmp_path, world_size=2)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.timeout(300)
def test_four_rank_checkpoint_round_trip_with_two_ranks_per_expert_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Four ranks with ``ep_size`` 2: the routed-expert weights are on the ``(ep_shard, ep)`` mesh."""
    _run_round_trip(monkeypatch, tmp_path, world_size=4)

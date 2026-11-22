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

"""Unit tests for the distributed-checkpoint destinations and export views of ERNIE 5.5.

Most tests use single-rank DTensors on CPU, so that layouts, aliasing and the dtype contract are checked in one
process. One test runs the checkpointer on two and on four Gloo ranks, where the shards differ between ranks: two
ranks with one rank per expert group (experts on an ``(ep,)`` mesh), four ranks with an ``(ep_shard, ep)`` mesh.
"""

import collections
import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5.checkpoint_io import (
    DsaKVDcpLoadPart,
    MhcDcpLoadPart,
    common_hf_dcp_views,
    expert_hf_dcp_views,
    merge_dsa_kv_for_hf,
    mhc_hf_export_views,
    model_hf_dcp_load_destinations,
    model_hf_distributed_export,
)
from nemo_automodel.components.models.ernie5_5.checkpoint_layout import map_mhc_to_hf, pack_experts
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM

pytestmark = pytest.mark.cpu_only

VOCAB, HIDDEN, STREAMS = 64, 16, 2
EXPERTS, MOE_LATENT, MOE_INTERMEDIATE = 4, 8, 12
# Four DSA heads, so that two and four ranks hold whole heads.
HEADS, KV_LATENT, KEY_CHANNELS, VALUE_CHANNELS = 4, 4, 4, 3
EXPERT_NAMES = ("moe.experts.gate_and_up_projs", "moe.experts.down_projs")
# Part of the Hugging Face name of every per-expert matrix, and of no other tensor.
EXPERT_HF_MARK = ".block_sparse_moe.experts."
FUSED_KV = "model.layers.2.self_attn.kv_b_proj.weight"
# Hugging Face keys of the tiny model: 8 model-level tensors, 30 of the HCA + dense layer, 49 of the HCA + MoE
# layer and 50 of the DSA + MoE layer (four routed experts, three matrices each).
HF_KEYS = 137
COMMON_HF_NAMES = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
    "model.lm_head.multimax_ranges",
    "model.lm_head.multimax_ts",
    "model.mhc_contract.hc_head_fn",
    "model.mhc_contract.hc_head_base",
    "model.mhc_contract.hc_head_scale",
]
Meshes = tuple[DeviceMesh, DeviceMesh]


def _backend() -> BackendConfig:
    return BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")


def _tiny_config(dtype: str = "float32", **overrides) -> Ernie5_5Config:
    """Return a three-layer configuration with every layer kind: HCA + dense MLP, HCA + MoE, DSA + MoE."""
    fields = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=24,
        num_hidden_layers=3,
        max_position_embeddings=1024,
        compress_ratios=[128, 128, -2],
        num_nextn_predict_layers=0,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=6,
        o_lora_rank=4,
        o_groups=2,
        sliding_window=8,
        hybrid_mla_num_attention_heads=HEADS,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=KV_LATENT,
        hybrid_mla_qk_nope_head_dim=KEY_CHANNELS,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=VALUE_CHANNELS,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        rope_interleave=True,
        hc_mult=STREAMS,
        n_routed_experts=EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=MOE_INTERMEDIATE,
        moe_latent_size=MOE_LATENT,
        torch_dtype=dtype,
    )
    return Ernie5_5Config(**{**fields, **overrides})


def _random_model(dtype: str, seed: int, stored_dtype: torch.dtype | None = None) -> Ernie5_5_MoeForCausalLM:
    """Build the registered model and fill every parameter and buffer with seeded random values.

    Args:
        dtype: Model dtype.
        seed: Seed of the values; the same seed gives the same values in every process.
        stored_dtype: When given, the values are rounded to this dtype first, so that a checkpoint that stores
            them in this dtype loses nothing.
    """
    model = Ernie5_5_MoeForCausalLM(_tiny_config(dtype), _backend())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            values = torch.randn(tensor.shape, generator=generator)
            if stored_dtype is not None:
                values = values.to(stored_dtype).to(torch.float32)
            tensor.copy_(values.to(tensor.dtype))
    return model


def _meshes(world_size: int, one_rank_per_expert_group: bool = False) -> Meshes:
    """Return the ``dp`` mesh of all ranks and the mesh of the routed experts.

    Args:
        world_size: Number of ranks.
        one_rank_per_expert_group: Give every rank whole experts on a one-dimensional ``(ep,)`` mesh, the layout
            of a run whose expert-parallel size equals its data-parallel size. Otherwise the expert mesh is
            ``(ep_shard, ep)`` with two expert groups (one for a single rank).
    """
    dp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
    if one_rank_per_expert_group:
        return dp_mesh, init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
    ep_size = min(world_size, 2)
    return dp_mesh, init_device_mesh("cpu", (world_size // ep_size, ep_size), mesh_dim_names=("ep_shard", "ep"))


@pytest.fixture
def meshes():
    """Single-rank meshes in this process, on a process group that the test owns."""
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield _meshes(1)
    finally:
        dist.destroy_process_group()


def _shard(name: str, value: torch.Tensor, meshes: Meshes) -> DTensor:
    """Distribute one native tensor the way the model is sharded.

    Args:
        name: Native name of the tensor.
        value: Complete tensor of any shape; it is copied.
        meshes: The ``dp`` mesh and the expert mesh, ``(ep_shard, ep)`` or ``(ep,)``.

    Returns:
        The grouped expert tensors as DTensors on the expert mesh, ``(Shard(1), Shard(0))`` on ``(ep_shard, ep)``
        or ``(Shard(0),)`` on ``(ep,)``; every other tensor as a ``(Shard(0),)`` DTensor on the ``dp`` mesh.
    """
    dp_mesh, ep_mesh = meshes
    if not name.endswith(EXPERT_NAMES):
        return distribute_tensor(value.detach().clone().contiguous(), dp_mesh, [Shard(0)])
    placements = [Shard(0)] if ep_mesh.mesh_dim_names == ("ep",) else [Shard(1), Shard(0)]
    return distribute_tensor(value.detach().clone().contiguous(), ep_mesh, placements)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return this rank's part of a destination or exported tensor.

    Args:
        tensor: DTensor of any global shape and placements, or a plain tensor of any shape.

    Returns:
        The local shard of a DTensor, in this rank's local shape, or the plain tensor itself. The result
        aliases the storage of ``tensor``.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _shard_state(state: dict[str, torch.Tensor], meshes: Meshes) -> dict[str, DTensor]:
    """Return a sharded copy of a native state.

    Args:
        state: Native names mapped to complete tensors of any shape.
        meshes: The ``dp`` mesh and the expert mesh, ``(ep_shard, ep)`` or ``(ep,)``.
    """
    return {name: _shard(name, value, meshes) for name, value in state.items()}


def _shard_model(
    model: Ernie5_5_MoeForCausalLM, values: dict[str, torch.Tensor], meshes: Meshes
) -> dict[str, torch.Tensor]:
    """Replace the parameters and buffers of ``model`` by DTensor shards filled with NaN.

    Args:
        model: Unsharded model; modified in place.
        values: Native names mapped to complete tensors in the shapes of the model's parameters and buffers.
        meshes: The ``dp`` mesh and the expert mesh, ``(ep_shard, ep)`` or ``(ep,)``.

    Returns:
        Native names mapped to this rank's shard of ``values``: what the model must hold after a load.
    """
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    expected = {}
    for name, value in values.items():
        shard = _shard(name, value, meshes)
        expected[name] = shard.to_local().clone()
        shard.to_local().fill_(float("nan"))
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        if name in parameters:
            setattr(parent, attribute, torch.nn.Parameter(shard, requires_grad=parameters[name].requires_grad))
        else:
            assert name in buffers, name
            parent._buffers[attribute] = shard
    return expected


def _write_hf_checkpoint(model: Ernie5_5_MoeForCausalLM, directory: Path, stored_dtype: torch.dtype) -> None:
    """Write the model as a Hugging Face format directory with every tensor stored in ``stored_dtype``."""
    converted = model.state_dict_adapter.to_hf(model.state_dict())
    directory.mkdir(parents=True)
    save_file(
        {name: value.to(stored_dtype).contiguous() for name, value in converted.items()},
        str(directory / "model.safetensors"),
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": dict.fromkeys(converted, "model.safetensors")})
    )
    model.config.save_pretrained(directory)


def _assert_local_state(model: Ernie5_5_MoeForCausalLM, expected: dict[str, torch.Tensor]) -> None:
    """Require that every local shard of the model holds the expected values in the expected dtype.

    Args:
        model: Sharded model.
        expected: Native names mapped to this rank's expected shard of each tensor.
    """
    state = model.state_dict()
    assert set(state) == set(expected)
    for name, value in expected.items():
        local = state[name].to_local()
        assert local.dtype == value.dtype, name
        assert torch.equal(local, value), name


def _load_save_and_restore(
    model: Ernie5_5_MoeForCausalLM, expected: dict[str, torch.Tensor], checkpoint: Path, root: Path, rank: int
) -> None:
    """Load a Hugging Face checkpoint into a sharded model, save the model and restore the saved shards.

    Args:
        model: Sharded model whose local shards hold NaN.
        expected: Native names mapped to this rank's shard of the checkpoint values.
        checkpoint: Hugging Face format directory.
        root: Scratch directory shared by all ranks.
        rank: Rank of this process.
    """
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
    _assert_local_state(model, expected)
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
        if dist.get_world_size() > 1:
            dist.barrier()
        with torch.no_grad():
            for value in model.state_dict().values():
                value.to_local().zero_()
        trained.load_model(model, model_path=str(root / "saved" / "model"), is_init_step=False)
    finally:
        trained.close()
    _assert_local_state(model, expected)


# Destinations of single components


def test_common_views_alias_model_storage() -> None:
    model = _random_model("float32", seed=1)
    state = {name.removeprefix("model."): value for name, value in model.state_dict().items()}
    views = common_hf_dcp_views(state, vocab=VOCAB, hidden=HIDDEN, streams=STREAMS)
    assert list(views) == COMMON_HF_NAMES
    assert views["model.mhc_contract.hc_head_fn"].shape == (STREAMS * HIDDEN, STREAMS)
    with torch.no_grad():
        views["model.mhc_contract.hc_head_fn"].copy_(torch.arange(STREAMS * HIDDEN * STREAMS).reshape(-1, STREAMS))
        views["lm_head.weight"].zero_()
    assert torch.equal(
        state["hyper_head.hc_fn"], torch.arange(STREAMS * HIDDEN * STREAMS, dtype=torch.float32).reshape(-1, STREAMS).T
    )
    assert not state["lm_head.weight"].any()
    with pytest.raises(ValueError, match="Missing common target keys"):
        common_hf_dcp_views({}, vocab=VOCAB, hidden=HIDDEN, streams=STREAMS)
    with pytest.raises(ValueError, match="embed_tokens.weight has shape"):
        common_hf_dcp_views(state, vocab=VOCAB + 1, hidden=HIDDEN, streams=STREAMS)
    on_meta = {name: torch.empty_like(value, device="meta") for name, value in state.items()}
    with pytest.raises(ValueError, match="must be a materialized tensor"):
        common_hf_dcp_views(on_meta, vocab=VOCAB, hidden=HIDDEN, streams=STREAMS)


def test_mhc_load_part_installs_the_alphas_into_the_scale(meshes: Meshes) -> None:
    dp_mesh, _ = meshes
    mix = (2 + STREAMS) * STREAMS
    fn = distribute_tensor(torch.zeros(mix, STREAMS * HIDDEN), dp_mesh, [Shard(0)])
    base = distribute_tensor(torch.zeros(mix), dp_mesh, [Shard(0)])
    scale = distribute_tensor(torch.zeros(3), dp_mesh, [Shard(0)])
    part = MhcDcpLoadPart(fn, base, scale, streams=STREAMS, hidden=HIDDEN, hf_prefix="unit.")
    assert list(part.targets) == [
        "unit.mapping_proj.weight",
        "unit.bias",
        "unit.alpha_pre",
        "unit.alpha_post",
        "unit.alpha_res",
    ]
    projection = torch.randn(STREAMS * HIDDEN, mix)
    part.targets["unit.mapping_proj.weight"].to_local().copy_(projection)
    part.targets["unit.bias"].to_local().fill_(2.0)
    for name, alpha in (("alpha_pre", 0.25), ("alpha_post", 0.5), ("alpha_res", 0.75)):
        assert part.targets[f"unit.{name}"].dtype == torch.float32
        part.targets[f"unit.{name}"].to_local().fill_(alpha)
    # The projection and the bias are written in place; the alphas wait in temporaries.
    assert torch.equal(fn.to_local(), projection.T) and (base.to_local() == 2).all()
    assert not scale.to_local().any()
    part.finish()
    assert torch.equal(scale.to_local(), torch.tensor([0.25, 0.5, 0.75]))
    with pytest.raises(RuntimeError, match="already been finished"):
        part.finish()
    with pytest.raises(TypeError, match="mHC parameters must be DTensors"):
        MhcDcpLoadPart(
            torch.zeros(mix, STREAMS * HIDDEN),
            torch.zeros(mix),
            torch.zeros(3),
            streams=STREAMS,
            hidden=HIDDEN,
            hf_prefix="unit.",
        )


def _split_by_head(
    fused: torch.Tensor, *, heads: int, key_channels: int, value_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the released fused K/V projection head by head, with explicit loops.

    Args:
        fused: Tensor of shape [heads * (key_channels + value_channels), latent]; for each head in turn, its
            key rows followed by its value rows.
        heads: Number of attention heads.
        key_channels: Key rows per head.
        value_channels: Value rows per head.

    Returns:
        Keys of shape [heads * latent, key_channels], each head's key block transposed, and values of shape
        [heads * value_channels, latent].
    """
    rows = key_channels + value_channels
    keys, values = [], []
    for head in range(heads):
        block = fused[head * rows : (head + 1) * rows]
        keys.append(block[:key_channels].T)
        values.append(block[key_channels:])
    return torch.cat(keys), torch.cat(values)


@pytest.mark.parametrize("native_dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fused_kv_is_read_through_an_fp32_temporary(meshes: Meshes, native_dtype: torch.dtype) -> None:
    """The temporary is FP32 whatever the native dtype, so that a read never rounds before ``finish`` casts."""
    dp_mesh, _ = meshes
    key = distribute_tensor(torch.zeros(HEADS * KV_LATENT, KEY_CHANNELS, dtype=native_dtype), dp_mesh, [Shard(0)])
    value = distribute_tensor(torch.zeros(HEADS * VALUE_CHANNELS, KV_LATENT, dtype=native_dtype), dp_mesh, [Shard(0)])
    dimensions = dict(heads=HEADS, latent=KV_LATENT, key_channels=KEY_CHANNELS, value_channels=VALUE_CHANNELS)
    part = DsaKVDcpLoadPart(key, value, **dimensions)
    assert part.target.dtype == torch.float32
    assert part.target.shape == (HEADS * (KEY_CHANNELS + VALUE_CHANNELS), KV_LATENT)
    fused = torch.randn(HEADS * (KEY_CHANNELS + VALUE_CHANNELS), KV_LATENT, generator=torch.Generator().manual_seed(3))
    # The values need more precision than BF16 and FP16 have: a narrower temporary would change them.
    assert not torch.equal(fused, fused.to(torch.bfloat16).to(torch.float32))
    assert not torch.equal(fused, fused.to(torch.float16).to(torch.float32))
    part.target.to_local().copy_(fused)
    assert not key.to_local().any() and not value.to_local().any()
    part.finish()
    expected_key, expected_value = _split_by_head(
        fused, heads=HEADS, key_channels=KEY_CHANNELS, value_channels=VALUE_CHANNELS
    )
    assert torch.equal(key.to_local(), expected_key.to(native_dtype))
    assert torch.equal(value.to_local(), expected_value.to(native_dtype))
    with pytest.raises(RuntimeError, match="already been finished"):
        part.finish()
    with pytest.raises(TypeError, match="K and V must be DTensors"):
        DsaKVDcpLoadPart(key.to_local(), value.to_local(), **dimensions)


@pytest.mark.parametrize("one_rank_per_expert_group", [False, True], ids=["ep_shard_ep", "ep"])
def test_expert_views_alias_the_grouped_storage(meshes: Meshes, one_rank_per_expert_group: bool) -> None:
    """Both accepted layouts: ``(Shard(1), Shard(0))`` on ``(ep_shard, ep)`` and ``(Shard(0),)`` on ``(ep,)``."""
    if one_rank_per_expert_group:
        ep_mesh, placements = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",)), [Shard(0)]
    else:
        ep_mesh, placements = meshes[1], [Shard(1), Shard(0)]
    gate_up = distribute_tensor(torch.zeros(EXPERTS, MOE_LATENT, 2 * MOE_INTERMEDIATE), ep_mesh, placements)
    down = distribute_tensor(torch.zeros(EXPERTS, MOE_INTERMEDIATE, MOE_LATENT), ep_mesh, placements)
    dimensions = dict(layer=1, experts=EXPERTS, latent=MOE_LATENT, intermediate=MOE_INTERMEDIATE)
    views = expert_hf_dcp_views(gate_up, down, **dimensions)
    assert len(views) == 3 * EXPERTS
    checkpoint = {}
    for expert in range(EXPERTS):
        for projection, shape in ((1, (MOE_INTERMEDIATE, MOE_LATENT)), (2, (MOE_LATENT, MOE_INTERMEDIATE))):
            checkpoint[f"experts.{expert}.w{projection}.weight"] = torch.randn(shape)
        checkpoint[f"experts.{expert}.w3.weight"] = torch.randn(MOE_INTERMEDIATE, MOE_LATENT)
    for name, value in checkpoint.items():
        target = views[f"model.layers.1.block_sparse_moe.{name}"]
        # Whole experts of one rank are plain tensors; experts that are also split over ``ep_shard`` are DTensors.
        assert isinstance(target, DTensor) is not one_rank_per_expert_group
        assert tuple(target.shape) == tuple(value.shape)
        _local(target).copy_(value)
    expected_gate_up, expected_down = pack_experts(checkpoint, range(EXPERTS))
    assert torch.equal(gate_up.to_local(), expected_gate_up) and torch.equal(down.to_local(), expected_down)
    with pytest.raises(TypeError, match="Expert parameters must be DTensors"):
        expert_hf_dcp_views(gate_up.to_local(), down.to_local(), **dimensions)


def test_expert_views_reject_other_layouts(meshes: Meshes) -> None:
    dp_mesh, ep_mesh = meshes
    dimensions = dict(layer=1, experts=EXPERTS, latent=MOE_LATENT, intermediate=MOE_INTERMEDIATE)
    gate_up, down = (
        torch.zeros(EXPERTS, MOE_LATENT, 2 * MOE_INTERMEDIATE),
        torch.zeros(EXPERTS, MOE_INTERMEDIATE, MOE_LATENT),
    )
    accepted = r"\(ep_shard, ep\) mesh or \(Shard\(0\),\) DTensors on an \(ep,\) mesh"
    # Experts sharded like every other parameter, on the data-parallel mesh.
    with pytest.raises(ValueError, match=accepted):
        expert_hf_dcp_views(
            distribute_tensor(gate_up, dp_mesh, [Shard(0)]), distribute_tensor(down, dp_mesh, [Shard(0)]), **dimensions
        )
    # The right mesh with the two axes exchanged.
    with pytest.raises(ValueError, match=accepted):
        expert_hf_dcp_views(
            distribute_tensor(gate_up, ep_mesh, [Shard(0), Shard(1)]),
            distribute_tensor(down, ep_mesh, [Shard(0), Shard(1)]),
            **dimensions,
        )
    # One accepted layout for each tensor, but not the same one.
    one_dimensional = init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))
    with pytest.raises(ValueError, match=accepted):
        expert_hf_dcp_views(
            distribute_tensor(gate_up, ep_mesh, [Shard(1), Shard(0)]),
            distribute_tensor(down, one_dimensional, [Shard(0)]),
            **dimensions,
        )


# Export views of single components


def test_fused_kv_export_merges_whole_heads(meshes: Meshes) -> None:
    dp_mesh, _ = meshes
    heads, latent, key_channels, value_channels = 2, 4, 3, 2
    key_full = torch.arange(heads * latent * key_channels, dtype=torch.float32).reshape(heads * latent, key_channels)
    value_full = torch.arange(heads * value_channels * latent, dtype=torch.float32).reshape(-1, latent)
    key = distribute_tensor(key_full, dp_mesh, [Shard(0)])
    value = distribute_tensor(value_full, dp_mesh, [Shard(0)])
    dimensions = dict(heads=heads, latent=latent, key_channels=key_channels, value_channels=value_channels)
    actual = merge_dsa_kv_for_hf(key, value, **dimensions)
    assert actual.shape == (heads * (key_channels + value_channels), latent) and actual.dtype == torch.float32
    # Splitting the fused matrix head by head must give back the two projections.
    split_key, split_value = _split_by_head(
        actual.to_local(), heads=heads, key_channels=key_channels, value_channels=value_channels
    )
    assert torch.equal(split_key, key_full) and torch.equal(split_value, value_full)
    with pytest.raises(ValueError, match="matching"):
        merge_dsa_kv_for_hf(key, value.to(torch.bfloat16), **dimensions)
    with pytest.raises(ValueError, match="Invalid heads"):
        merge_dsa_kv_for_hf(key, value, **{**dimensions, "heads": True})


def test_mhc_export_views_match_the_eager_conversion(meshes: Meshes) -> None:
    dp_mesh, _ = meshes
    mix = (2 + STREAMS) * STREAMS
    full = {
        "fn": torch.arange(mix * STREAMS * HIDDEN, dtype=torch.float32).reshape(mix, STREAMS * HIDDEN),
        "base": torch.arange(mix, dtype=torch.float32),
        "scale": torch.tensor([1.0, 2.0, 3.0]),
    }
    native = {name: distribute_tensor(value, dp_mesh, [Shard(0)]) for name, value in full.items()}
    dimensions = dict(streams=STREAMS, hidden=HIDDEN)
    actual = mhc_hf_export_views(native["fn"], native["base"], native["scale"], hf_prefix="layer.", **dimensions)
    expected = {"layer." + name: value for name, value in map_mhc_to_hf(full).items()}
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name].to_local(), value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="prefix"):
        mhc_hf_export_views(native["fn"], native["base"], native["scale"], hf_prefix="invalid", **dimensions)


def test_decoder_export_rejects_an_incomplete_layer() -> None:
    model = _random_model("float32", seed=1)
    # The model-level tensors only: every tensor of every decoder layer is missing.
    model_level = {
        name.removeprefix("model."): value for name, value in model.state_dict().items() if ".layers." not in name
    }
    assert len(model_level) == len(COMMON_HF_NAMES)
    with pytest.raises(ValueError, match=r"Invalid native keys: missing=\[.+\], unexpected=\[\]"):
        model_hf_distributed_export(model_level, model.config)


# Whole model: load session and export of a sharded state through the adapter


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_sharded_load_session_installs_the_checkpoint_values(meshes: Meshes, dtype: str) -> None:
    model, other = _random_model(dtype, seed=1), _random_model(dtype, seed=2)
    adapter = model.state_dict_adapter
    keys = adapter.get_hf_state_dict_keys(model.state_dict())
    checkpoint = other.state_dict_adapter.to_hf(other.state_dict())
    state = _shard_state(model.state_dict(), meshes)
    for _ in range(2):  # A finished session leaves the adapter ready for the next one.
        destinations = adapter.to_hf(state, for_checkpoint_load=True)
        assert set(destinations) == set(keys) and len(destinations) == len(keys) == HF_KEYS
        assert list(destinations)[: len(COMMON_HF_NAMES)] == COMMON_HF_NAMES
        # Only the fused K/V projection has a dtype of its own; it is FP32 whatever the model dtype.
        assert destinations[FUSED_KV].dtype == torch.float32
        for name, target in destinations.items():
            if name != FUSED_KV:
                assert target.dtype == checkpoint[name].dtype, name
            target.to_local().copy_(checkpoint[name])
        loaded = adapter.from_hf(destinations)
        assert list(loaded) == list(state) and all(loaded[name] is tensor for name, tensor in state.items())
        for name, value in other.state_dict().items():
            assert state[name].dtype == value.dtype, name
            assert torch.equal(state[name].to_local(), value), name
        for tensor in state.values():
            tensor.to_local().zero_()


@pytest.mark.parametrize("replacement", ["copy", "other_adapter", "finished_session"])
def test_sharded_session_fails_closed(meshes: Meshes, replacement: str) -> None:
    model, other = _random_model("float32", seed=1), _random_model("float32", seed=2)
    adapter = model.state_dict_adapter
    state = _shard_state(model.state_dict(), meshes)
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.to_hf(state)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.get_hf_state_dict_keys(model.state_dict())
    if replacement == "copy":
        wrong = dict(destinations)
    elif replacement == "other_adapter":
        wrong = other.state_dict_adapter.to_hf(other.state_dict(), for_checkpoint_load=True)
    else:
        adapter.from_hf(destinations)
        wrong = destinations
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(wrong)
    with pytest.raises(RuntimeError, match="Missing or replaced load session"):
        adapter.from_hf(destinations)
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(RuntimeError, match="Abandoned or active load"):
        adapter.to_hf(state)


def test_key_planning_rejects_a_sharded_state(meshes: Meshes) -> None:
    """Keys are planned before the model is sharded; a sharded state is refused, not planned by accident."""
    model = _random_model("float32", seed=1)
    adapter = model.state_dict_adapter
    keys = adapter.get_hf_state_dict_keys(model.state_dict())
    assert len(keys) == HF_KEYS
    sharded = _shard_state(model.state_dict(), meshes)
    with pytest.raises(ValueError, match="before the model is sharded"):
        adapter.get_hf_state_dict_keys(sharded)
    # The refusal is not a failed load: the adapter still plans, exports and loads.
    assert adapter.get_hf_state_dict_keys(model.state_dict()) == keys
    assert len(adapter.to_hf(sharded)) == HF_KEYS
    assert len(adapter.to_hf(sharded, for_checkpoint_load=True)) == HF_KEYS


def test_unsharded_state_with_a_device_mesh_is_rejected(meshes: Meshes) -> None:
    """Loading into an unsharded model is supported only without a device mesh."""
    _, ep_mesh = meshes
    model = _random_model("float32", seed=1)
    adapter = model.state_dict_adapter
    state = model.state_dict()
    converted = adapter.to_hf(state)
    with pytest.raises(ValueError, match="unsharded model with a device mesh is not supported"):
        adapter.from_hf(converted, device_mesh=ep_mesh)
    # Outside a load session the refusal leaves the adapter usable.
    assert set(adapter.from_hf(converted)) == set(state)
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    with pytest.raises(ValueError, match="unsharded model with a device mesh is not supported"):
        adapter.from_hf(destinations, device_mesh=ep_mesh)
    # Inside a load session it is a failed load.
    with pytest.raises(RuntimeError, match="Abandoned or concurrent load"):
        adapter.to_hf(state, for_checkpoint_load=True)


def test_load_destinations_reject_incomplete_and_unsharded_states(meshes: Meshes) -> None:
    model = _random_model("float32", seed=1)
    config = model.config
    native = {name.removeprefix("model."): value for name, value in _shard_state(model.state_dict(), meshes).items()}
    targets, finishers = model_hf_dcp_load_destinations(native, config)
    # Two mHC units per layer and the fused K/V projection of the DSA layer are installed after the read.
    assert len(finishers) == 2 * 3 + 1 and len(targets) == HF_KEYS
    without_norm = {name: value for name, value in native.items() if name != "norm.weight"}
    with pytest.raises(ValueError, match="Missing common model tensors"):
        model_hf_dcp_load_destinations(without_norm, config)
    without_gate = {name: value for name, value in native.items() if name != "layers.1.moe.gate.weight"}
    with pytest.raises(ValueError, match=r"Invalid layer 1: missing=\['moe.gate.weight'\]"):
        model_hf_dcp_load_destinations(without_gate, config)
    with pytest.raises(ValueError, match="Unexpected native model keys"):
        model_hf_dcp_load_destinations({**native, "layers.3.extra": native["norm.weight"]}, config)
    with pytest.raises(TypeError, match="mHC parameters must be DTensors"):
        model_hf_dcp_load_destinations({**native, "layers.0.attn_hc.scale": torch.zeros(3)}, config)
    with pytest.raises(ValueError, match="Only explicit MTP=0 with untied embeddings"):
        model_hf_dcp_load_destinations(native, _tiny_config(tie_word_embeddings=True))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_sharded_export_matches_the_unsharded_conversion(meshes: Meshes, dtype: str) -> None:
    model = _random_model(dtype, seed=3)
    adapter = model.state_dict_adapter
    expected = adapter.to_hf(model.state_dict())
    exported = adapter.to_hf(_shard_state(model.state_dict(), meshes))
    assert set(exported) == set(expected) and len(exported) == len(expected)
    assert list(exported)[: len(COMMON_HF_NAMES)] == COMMON_HF_NAMES
    for name, value in expected.items():
        tensor = exported[name]
        assert isinstance(tensor, DTensor), name
        assert tensor.dtype == value.dtype and tuple(tensor.shape) == tuple(value.shape), name
        assert torch.equal(tensor.to_local(), value), name


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_one_dimensional_expert_layout_loads_and_exports_through_the_adapter(meshes: Meshes, dtype: str) -> None:
    """Experts on an ``(ep,)`` mesh: their destinations and exported tensors are plain tensors of this rank."""
    dp_mesh, _ = meshes
    one_dimensional = (dp_mesh, init_device_mesh("cpu", (1,), mesh_dim_names=("ep",)))
    model, other = _random_model(dtype, seed=1), _random_model(dtype, seed=2)
    adapter = model.state_dict_adapter
    keys = adapter.get_hf_state_dict_keys(model.state_dict())
    checkpoint = other.state_dict_adapter.to_hf(other.state_dict())
    state = _shard_state(model.state_dict(), one_dimensional)
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    assert set(destinations) == set(keys) and len(destinations) == HF_KEYS
    for name, target in destinations.items():
        # Only the per-expert matrices are plain tensors; every other destination is a DTensor as before.
        assert isinstance(target, DTensor) is (EXPERT_HF_MARK not in name), name
        _local(target).copy_(checkpoint[name])
    adapter.from_hf(destinations)
    for name, value in other.state_dict().items():
        assert state[name].dtype == value.dtype, name
        assert torch.equal(state[name].to_local(), value), name
    exported = adapter.to_hf(state)
    assert set(exported) == set(keys) and len(exported) == HF_KEYS
    for name, value in checkpoint.items():
        tensor = exported[name]
        assert isinstance(tensor, DTensor) is (EXPERT_HF_MARK not in name), name
        assert tensor.dtype == value.dtype and tuple(tensor.shape) == tuple(value.shape), name
        assert torch.equal(_local(tensor), value), name


def test_export_rejects_mixed_incomplete_and_mtp_states(meshes: Meshes) -> None:
    model = _random_model("float32", seed=3)
    state = model.state_dict()
    sharded = _shard_state(state, meshes)
    with pytest.raises(ValueError, match="Mixed local and distributed tensor export"):
        model.state_dict_adapter.to_hf({**sharded, "model.norm.weight": state["model.norm.weight"]})
    native = {name.removeprefix("model."): value for name, value in sharded.items()}
    without_gate = {name: value for name, value in native.items() if name != "layers.1.moe.gate.weight"}
    with pytest.raises(ValueError, match=r"Invalid native keys: missing=\['moe.gate.weight'\]"):
        model_hf_distributed_export(without_gate, model.config)
    with pytest.raises(ValueError, match="Unexpected native model tensors"):
        model_hf_distributed_export({**native, "layers.3.extra": native["norm.weight"]}, model.config)
    with_mtp = _tiny_config(num_nextn_predict_layers=1, compress_ratios=[128, 128, -2, -2])
    with pytest.raises(NotImplementedError, match="MTP export is unsupported"):
        model_hf_distributed_export(native, with_mtp)
    dimensions = dict(heads=HEADS, latent=KV_LATENT, key_channels=KEY_CHANNELS, value_channels=VALUE_CHANNELS)
    with pytest.raises(TypeError, match="K and V must be DTensors"):
        merge_dsa_kv_for_hf(
            state["model.layers.2.attention.k_b_proj"], state["model.layers.2.attention.v_b_proj"], **dimensions
        )
    with pytest.raises(TypeError, match="mHC parameters must be DTensors"):
        mhc_hf_export_views(
            state["model.layers.0.attn_hc.fn"],
            state["model.layers.0.attn_hc.base"],
            state["model.layers.0.attn_hc.scale"],
            streams=STREAMS,
            hidden=HIDDEN,
            hf_prefix="unit.",
        )


# The checkpointer on a sharded model


@pytest.mark.parametrize("stored_dtype", [torch.bfloat16, torch.float16, torch.float32], ids=["bf16", "fp16", "fp32"])
def test_checkpointer_reads_a_checkpoint_of_any_float_dtype_exactly(
    meshes: Meshes, tmp_path: Path, stored_dtype: torch.dtype
) -> None:
    """An FP32 model receives BF16, FP16 and FP32 checkpoints bit for bit, the fused K/V projection included."""
    source = _random_model("float32", seed=7, stored_dtype=None if stored_dtype == torch.float32 else stored_dtype)
    kv = source.state_dict()["model.layers.2.attention.k_b_proj"]
    if stored_dtype == torch.float32:
        # The FP32 checkpoint holds values that BF16 cannot represent: a BF16 temporary would round them.
        assert not torch.equal(kv, kv.to(torch.bfloat16).to(torch.float32))
    checkpoint = tmp_path / "hf_checkpoint"
    _write_hf_checkpoint(source, checkpoint, stored_dtype)
    model = Ernie5_5_MoeForCausalLM(_tiny_config("float32"), _backend())
    expected = _shard_model(model, source.state_dict(), meshes)
    _load_save_and_restore(model, expected, checkpoint, tmp_path, rank=0)


def _checkpointer_worker(rank: int, world_size: int, one_rank_per_expert_group: bool, rendezvous: str) -> None:
    """Load, save and restore a sharded BF16 model on one of several Gloo ranks, then check the exported keys."""
    # The ranks share the cores of one machine: one thread each, and a rendezvous that cannot wait forever.
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        root = Path(rendezvous).parent
        # Values that BF16 represents exactly, also in the tensors that the model keeps in FP32.
        source = _random_model("bfloat16", seed=11, stored_dtype=torch.bfloat16)
        checkpoint = root / "hf_checkpoint"
        if rank == 0:
            _write_hf_checkpoint(source, checkpoint, torch.bfloat16)
        dist.barrier()
        model = Ernie5_5_MoeForCausalLM(_tiny_config("bfloat16"), _backend())
        adapter = model.state_dict_adapter
        # Keys are planned before sharding; afterwards the parameters are DTensors and planning is refused.
        keys = adapter.get_hf_state_dict_keys(model.state_dict())
        assert len(keys) == HF_KEYS
        meshes = _meshes(world_size, one_rank_per_expert_group)
        expected = _shard_model(model, source.state_dict(), meshes)
        with pytest.raises(ValueError, match="before the model is sharded"):
            adapter.get_hf_state_dict_keys(model.state_dict())
        _load_save_and_restore(model, expected, checkpoint, root, rank)
        # Export: every rank names all other tensors, and of the per-expert matrices only those that it holds.
        exported = adapter.to_hf(model.state_dict())
        own_experts = [name for name in exported if EXPERT_HF_MARK in name]
        assert set(exported) - set(own_experts) == {name for name in keys if EXPERT_HF_MARK not in name}
        gathered: list[list[str]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, own_experts)
        holders = collections.Counter(name for names in gathered for name in names)
        assert set(holders) == {name for name in keys if EXPERT_HF_MARK in name}
        # On an (ep,) mesh one rank exports an expert; on (ep_shard, ep) the ranks of its group, two groups in all.
        assert set(holders.values()) == {1 if one_rank_per_expert_group else world_size // 2}
        if one_rank_per_expert_group:
            # Whole experts are exported as plain tensors: the values of this rank's experts under their global names.
            converted = source.state_dict_adapter.to_hf(source.state_dict())
            assert all(torch.equal(exported[name], converted[name]) for name in own_experts)
        # A head count that the ranks cannot split into whole heads is refused for a load and for a save, and
        # the message names the number of ranks and the head count.
        heads = world_size + 1
        key = distribute_tensor(torch.zeros(heads * KV_LATENT, KEY_CHANNELS), meshes[0], [Shard(0)])
        value = distribute_tensor(torch.zeros(heads * VALUE_CHANNELS, KV_LATENT), meshes[0], [Shard(0)])
        dimensions = dict(heads=heads, latent=KV_LATENT, key_channels=KEY_CHANNELS, value_channels=VALUE_CHANNELS)
        refusal = rf"{world_size} data-parallel ranks.*must divide hybrid_mla_num_attention_heads={heads}"
        with pytest.raises(ValueError, match=refusal):
            DsaKVDcpLoadPart(key, value, **dimensions)
        with pytest.raises(ValueError, match=refusal):
            merge_dsa_kv_for_hf(key, value, **dimensions)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world_size", "one_rank_per_expert_group"), [(2, True), (4, False)], ids=["2_ranks_ep", "4_ranks_ep_shard_ep"]
)
def test_checkpointer_loads_saves_and_restores_on_several_ranks(
    tmp_path: Path, world_size: int, one_rank_per_expert_group: bool
) -> None:
    """Two ranks hold whole experts on an ``(ep,)`` mesh; four ranks split them over ``ep`` and ``ep_shard``."""
    mp.spawn(
        _checkpointer_worker,
        args=(world_size, one_rank_per_expert_group, str(tmp_path / "rendezvous")),
        nprocs=world_size,
        join=True,
    )

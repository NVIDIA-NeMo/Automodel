#!/usr/bin/env python
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

"""Standalone multi-GPU checks for model expansion under tensor and data parallelism.

Three checks, run separately because they fail differently.

Tensor parallelism reaches the expansion weight only if the plan knows to distribute it;
``translate_to_expanded`` is what closes that. The TP check also runs the stock styles and
asserts they do *not* work, so the positive check cannot pass vacuously.

FSDP2 reduce-scatters gradients across the data-parallel axis, so the check that matters is
whether the expansion weights receive the gradient of the *global* batch rather than of one
rank's shard. Every unit here also holds a mix of frozen and trainable parameters, the usual
source of missing gradient hooks, and ``fully_shard`` patches ``__class__`` the same way the
expansion does -- which is what turned an earlier ``super(type(self), self)`` into infinite
recursion.

The ``deferred`` check covers the ordering a real training run is forced into:
allocate the expansion weights before sharding, give them values after the checkpoint
load. Function preservation has to hold there too.

The gradient check perturbs the expansion weights first. Left at their initial values the
output projections are zero, most expansion gradients are zero with them, and a comparison
of zeros passes no matter what the collectives did.

Any world size from 2 up works; the model is sized from it (see :func:`model_config`).

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/expansion/run_expansion_parallel.py

    # Optional: more ranks, or one mechanism only
    torchrun --nproc_per_node=4 tests/functional_tests/expansion/run_expansion_parallel.py \\
        --mode tp

This is also the implementation behind ``test_expansion_parallel.py``, which spawns the
same checks under pytest. It stays runnable on its own because the training containers do
not all carry pytest.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

# Importable from a source checkout with the package uninstalled, the way the pytest
# conftest puts the repo root on the path. Without this, running the script directly out
# of a clone fails with a bare ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nemo_automodel.components.expansion import (  # noqa: E402
    ExpansionConfig,
    apply_expansion,
    expansion_parameters,
    freeze_non_expansion_parameters,
    initialize_expansion,
    is_expansion_parameter,
)
from nemo_automodel.components.expansion.parallel_styles import translate_to_expanded  # noqa: E402

VOCAB, HEAD_DIM, LAYERS, SEQ = 64, 8, 4, 8
LOCAL_BATCH = 2
#: World size the pytest wrapper spawns. CI caps functional tests at 2 GPUs; the checks
#: themselves run at any world size >= 2.
WORLD_SIZE = 2
EXPANDED_LAYERS = [1, 2]

#: Leaf projections of one decoder layer and the style each takes. The defaults compose:
#: colwise hands the next module a locally-sharded tensor, rowwise declares its input
#: sharded on the feature dimension and returns a replicated one.
TP_STYLES = {
    "self_attn.q_proj": ColwiseParallel,
    "self_attn.k_proj": ColwiseParallel,
    "self_attn.v_proj": ColwiseParallel,
    "self_attn.o_proj": RowwiseParallel,
    "mlp.gate_proj": ColwiseParallel,
    "mlp.up_proj": ColwiseParallel,
    "mlp.down_proj": RowwiseParallel,
}


def model_config(world_size: int) -> LlamaConfig:
    """A tiny Llama sized so tensor parallelism divides it evenly.

    The head counts scale with the world size because tensor parallelism shards the
    attention projections by head. Fixed counts would silently cap the runnable world
    size: two KV heads over four ranks gives each rank half a head, and attention's
    ``view(batch, seq, -1, head_dim)`` then fails inside transformers, several frames away
    from anything to do with expansion.

    Args:
        world_size: Number of ranks the model will be sharded over.

    Returns:
        A config with ``world_size`` KV heads and twice as many attention heads.
    """
    attention_heads = 2 * world_size
    hidden = attention_heads * HEAD_DIM
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=LAYERS,
        num_attention_heads=attention_heads,
        num_key_value_heads=world_size,
        max_position_embeddings=SEQ * 2,
        attention_dropout=0.0,
    )


def build(
    world_size: int, layers: list[int] | None = None, perturb: float = 0.0, initialize: bool = True
) -> LlamaForCausalLM:
    """A tiny Llama on the current CUDA device, optionally expanded and perturbed.

    Args:
        world_size: Number of ranks, which sets the head counts. See :func:`model_config`.
        layers: Decoder-layer indices to expand, or ``None`` to leave the model unexpanded.
        perturb: Standard deviation of noise added to every expansion weight. Non-zero
            makes stream B observable; at zero the output projections discard it.
        initialize: Give the expansion weights their values now. ``False`` allocates them
            only, leaving :func:`initialize_expansion` to run after sharding.

    Returns:
        The model, in eval mode, identical on every rank.
    """
    torch.manual_seed(0)
    model = LlamaForCausalLM(model_config(world_size)).cuda().eval()
    if layers is not None:
        apply_expansion(model, ExpansionConfig(enabled=True, layers=layers), initialize=initialize)
    if perturb:
        generator = torch.Generator(device="cuda").manual_seed(3)
        with torch.no_grad():
            for _, param in expansion_parameters(model):
                param.add_(torch.randn(param.shape, generator=generator, device="cuda") * perturb)
    return model


def global_input_ids(world_size: int) -> torch.Tensor:
    """Returns: ``[world_size * LOCAL_BATCH, SEQ]`` token ids, identical on every rank."""
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (world_size * LOCAL_BATCH, SEQ), device="cuda")


def full(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a possibly-distributed tensor. Collective when the input is a DTensor."""
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def logits(model: LlamaForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return full(model(input_ids=input_ids, use_cache=False).logits)


def parallelize(model: LlamaForCausalLM, mesh, expansion_aware: bool) -> LlamaForCausalLM:
    """Apply a per-layer TP plan, optionally translated to its expansion-aware equivalent."""
    plan = {}
    for index in range(LAYERS):
        for suffix, style in TP_STYLES.items():
            instance = style()
            plan[f"model.layers.{index}.{suffix}"] = translate_to_expanded(instance) if expansion_aware else instance
    parallelize_module(model, mesh, plan)
    return model


def shard(model: LlamaForCausalLM, mesh) -> LlamaForCausalLM:
    """Shard every decoder layer and the root, the grouping the recipes use."""
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    return fully_shard(model, mesh=mesh)


def grads_of(model: LlamaForCausalLM) -> dict[str, torch.Tensor]:
    return {name: param.grad for name, param in model.named_parameters() if param.grad is not None}


def reference_grads(input_ids: torch.Tensor, world_size: int) -> dict[str, torch.Tensor]:
    """Gradients of the same loss over the whole batch, computed without any parallelism."""
    model = build(world_size, layers=EXPANDED_LAYERS, perturb=0.05)
    freeze_non_expansion_parameters(model)
    model.train()
    model(input_ids=input_ids, use_cache=False).logits.square().mean().backward()
    return grads_of(model)


def check_tensor_parallel(mesh, rank: int, world_size: int) -> None:
    """Function preservation and expansion-weight distribution under a TP plan."""
    input_ids = global_input_ids(world_size)

    # 1. Sharding does not perturb the model at initialization. Bit-exactness is the
    # assertion because the two paths are algebraically identical while the output
    # projections are zero; a near miss would mean TP changed the arithmetic.
    parent = logits(parallelize(build(world_size), mesh, expansion_aware=False), input_ids)
    expanded = logits(parallelize(build(world_size, layers=EXPANDED_LAYERS), mesh, expansion_aware=True), input_ids)
    assert torch.equal(expanded, parent), f"max abs diff {(expanded - parent).abs().max().item():.3e}"

    # 2. The expansion weight takes its base weight's placement. This is where expansion is
    # simpler than LoRA: same shape as the base, so the same placement, with no need for
    # the per-factor handling ColwiseParallelLora needs.
    model = parallelize(build(world_size, layers=EXPANDED_LAYERS), mesh, expansion_aware=True)
    params = dict(model.named_parameters())
    expansion = dict(expansion_parameters(model))
    assert expansion
    for name, param in expansion.items():
        base = params[name.replace(".expansion.weight", ".weight")]
        assert isinstance(param, DTensor), f"{name} was left replicated"
        assert param.placements == base.placements, f"{name}: {param.placements} vs base {base.placements}"
        assert param.to_local().numel() < param.numel(), f"{name} is a DTensor but not sharded"

    # 3. Negative control: the stock styles do not handle an expanded linear, so the
    # assertions above are load-bearing rather than true of any plan at all. Torch's own
    # partition function iterates `named_parameters()` recursively and then calls
    # `register_parameter` with what it finds, so the nested `expansion.weight` makes it
    # raise outright. The exception type is not pinned -- a future torch might instead
    # leave the weight replicated, which is equally a failure and equally caught here.
    try:
        stock = parallelize(build(world_size, layers=EXPANDED_LAYERS), mesh, expansion_aware=False)
    except Exception as error:
        if rank == 0:
            print(f"  stock TP plan rejected the expanded linear, as expected: {type(error).__name__}")
    else:
        assert any(not isinstance(param, DTensor) for _, param in expansion_parameters(stock)), (
            "the stock TP plan handled the expansion weight; the expansion-aware styles are untested"
        )


def check_fsdp(mesh, rank: int, world_size: int) -> None:
    """Function preservation, weight sharding and data-parallel gradient correctness."""
    input_ids = global_input_ids(world_size)
    local_input_ids = input_ids.chunk(world_size)[rank]

    # 1. Sharding does not perturb the model at initialization.
    parent = logits(shard(build(world_size), mesh), local_input_ids)
    expanded = logits(shard(build(world_size, layers=EXPANDED_LAYERS), mesh), local_input_ids)
    assert torch.equal(expanded, parent), f"max abs diff {(expanded - parent).abs().max().item():.3e}"

    # 2. fully_shard treats the expansion weight like any other parameter. Frozen before
    # sharding, the order a recipe uses, so each unit holds a mix of frozen and trainable
    # parameters.
    model = build(world_size, layers=EXPANDED_LAYERS, perturb=0.05)
    freeze_non_expansion_parameters(model)
    model = shard(model, mesh)
    expansion = dict(expansion_parameters(model))
    assert expansion
    for name, param in expansion.items():
        assert isinstance(param, DTensor), f"{name} was left replicated"
        assert param.to_local().numel() < param.numel(), f"{name} is a DTensor but not sharded"

    # 3. The gradient each rank ends up with is the gradient of the *global* batch, not of
    # its own shard. Each rank runs a different slice of the batch and its local mean loss;
    # FSDP2 averages the gradients, which is the gradient of the global mean only when the
    # reduction actually happens.
    model.train()
    model(input_ids=local_input_ids, use_cache=False).logits.square().mean().backward()

    grads = grads_of(model)
    assert grads, "no parameter received a gradient"
    assert all(is_expansion_parameter(name) for name in grads), "a frozen weight received a gradient"

    reference = reference_grads(input_ids, world_size)
    assert set(grads) == set(reference), sorted(set(grads) ^ set(reference))
    for name, grad in grads.items():
        assert grad.abs().sum() > 0, f"{name} received a zero gradient; the comparison would be vacuous"
        torch.testing.assert_close(full(grad), reference[name], rtol=1e-4, atol=1e-6, msg=name)


def check_deferred_initialization(mesh, rank: int, world_size: int) -> None:
    """Allocate the expansion weights before sharding, give them values after.

    This is the order a parallel run is forced into: its weights are materialized only
    once the model is already sharded, so the copy from the pretrained weight has to
    happen against distributed tensors. Function preservation has to survive that, or
    whether the expanded model equals its parent would depend on the launch topology.
    """
    input_ids = global_input_ids(world_size)
    parent = logits(parallelize(build(world_size), mesh, expansion_aware=False), input_ids)

    model = parallelize(build(world_size, layers=EXPANDED_LAYERS, initialize=False), mesh, expansion_aware=True)
    expansion = dict(expansion_parameters(model))
    assert expansion
    assert all(isinstance(param, DTensor) for param in expansion.values()), "allocation did not survive sharding"

    assert initialize_expansion(model) == len(expansion)
    assert torch.equal(logits(model, input_ids), parent)


#: Mesh dimension name per mode. The name is cosmetic -- every mesh here is 1-D over all
#: ranks -- but it keeps the intent readable in a mesh repr.
CHECKS = {
    "tp": ("tp", check_tensor_parallel),
    "fsdp": ("dp", check_fsdp),
    "deferred": ("tp", check_deferred_initialization),
}


def run_check(mode: str, rank: int, world_size: int) -> None:
    """Build the mesh for ``mode`` and run its checks. Assumes an initialized process group."""
    mesh_dim, check = CHECKS[mode]
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=(mesh_dim,))
    check(mesh, rank, world_size)
    dist.barrier()
    if rank == 0:
        print(f"[{mode}] OK", flush=True)


def spawn_worker(rank: int, world_size: int, init_file: str, mode: str) -> None:
    """One rank, rendezvous by file. Used by the pytest wrapper; must stay top-level."""
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        run_check(mode, rank, world_size)
    finally:
        dist.destroy_process_group()


def main() -> None:
    """Entry point under ``torchrun``, which supplies the rank environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[*CHECKS, "both"], default="both")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise SystemExit(
            f"these checks need at least 2 ranks to shard anything, got {world_size}; "
            "launch with torchrun --nproc_per_node=2 or more"
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl")
    try:
        for mode in CHECKS if args.mode == "both" else [args.mode]:
            run_check(mode, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

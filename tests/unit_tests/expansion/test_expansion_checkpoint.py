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

"""Checkpointing an expanded model, and the ordering that makes it work.

Expansion adds ordinary parameters on ordinarily-named modules, so the checkpoint path
needs no special handling and a saved expanded model reloads bit-exactly. What does need
pinning down is *when* each half of expansion may run. Allocating the expansion weight has
to precede sharding, because the tensor-parallel plan is what distributes it; giving it a
value has to follow the checkpoint load, because it copies the pretrained weight. A single
process does both at once, and a parallel run cannot -- it materializes its weights only
after sharding -- which is why the two are separable.

The meta-device case is the one worth a guard rather than a comment. Left unchecked it
succeeds silently -- the copy is a no-op on meta, the weights are later filled with
arbitrary memory, and every function-preservation check still passes because the
zero-initialized output projections discard stream B until training starts.

Tests run single-process with ``torch.distributed`` reported as uninitialized, which is
the pattern ``tests/unit_tests/checkpoint`` uses for a CPU round trip.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.expansion import (
    ExpansionConfig,
    apply_expansion,
    expansion_parameters,
    initialize_expansion,
)

VOCAB, HIDDEN, LAYERS, SEQ, BATCH = 64, 32, 4, 8, 2
EXPANDED_LAYERS = [1, 2]


def _config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=SEQ * 2,
        attention_dropout=0.0,
    )


def _build(expand: bool = True, perturb: float = 0.0) -> LlamaForCausalLM:
    """A tiny Llama, optionally expanded and perturbed away from its initial values."""
    torch.manual_seed(0)
    model = LlamaForCausalLM(_config()).eval()
    if expand:
        apply_expansion(model, ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS))
    if perturb:
        generator = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for _, param in expansion_parameters(model):
                param.add_(torch.randn(param.shape, generator=generator) * perturb)
    return model


def _checkpointer(directory: Path, model_save_format: str = "safetensors") -> Checkpointer:
    config = CheckpointingConfig(
        checkpoint_dir=str(directory),
        model_save_format=model_save_format,
        model_cache_dir=str(directory / "cache"),
        model_repo_id="test/model",
        save_consolidated=False,
    )
    with patch("torch.distributed.is_initialized", return_value=False):
        return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0)


def _logits(model: LlamaForCausalLM, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=input_ids, use_cache=False).logits


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, SEQ), generator=torch.Generator().manual_seed(1))


@pytest.fixture
def single_rank_mesh():
    """A one-rank CPU mesh, torn down so the group does not leak into later tests."""
    dist.init_process_group("gloo", init_method=f"file://{tempfile.mktemp()}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("model_save_format", ["safetensors", "torch_save"])
def test_expanded_checkpoint_round_trips(tmp_path, input_ids, model_save_format):
    """Save then load reproduces the model exactly, expansion weights included.

    The weights are perturbed first: at their initial values the expansion weights are a
    copy of the pretrained weight or zero, both of which a broken load could reproduce by
    accident.
    """
    model = _build(perturb=0.05)
    expected_logits = _logits(model, input_ids)
    expected_weights = {name: param.detach().clone() for name, param in expansion_parameters(model)}

    checkpointer = _checkpointer(tmp_path, model_save_format)
    checkpointer.save_model(model, str(tmp_path / "step_1"))

    restored = _build(perturb=0.0)
    checkpointer.load_model(restored, str(tmp_path / "step_1" / "model"))

    assert expected_weights
    for name, param in expansion_parameters(restored):
        assert torch.equal(param, expected_weights[name]), name
    assert torch.equal(_logits(restored, input_ids), expected_logits)


def test_expansion_weights_reach_the_checkpoint_under_their_own_keys(tmp_path):
    """The on-disk keys are the module paths, so no adapter has to know about expansion."""
    model = _build(perturb=0.05)
    checkpointer = _checkpointer(tmp_path)
    checkpointer.save_model(model, str(tmp_path / "step_1"))

    shard = next(p for p in (tmp_path / "step_1" / "model").iterdir() if p.suffix == ".safetensors")
    saved_keys = set(load_file(str(shard)))
    expected_keys = {name for name, _ in expansion_parameters(model)}
    assert expected_keys
    assert expected_keys <= saved_keys


def test_base_checkpoint_loads_into_an_expanded_model(tmp_path):
    """A pre-expansion checkpoint has no expansion keys; the subset load keeps ours.

    This is the resume-from-parent case. ``allow_checkpoint_key_subset`` is what makes it
    work, and the expansion weights must come through untouched rather than zeroed.
    """
    checkpointer = _checkpointer(tmp_path)
    checkpointer.save_model(_build(expand=False), str(tmp_path / "base"))

    expanded = _build(perturb=0.05)
    expected_weights = {name: param.detach().clone() for name, param in expansion_parameters(expanded)}
    pretrained_weight = expanded.model.layers[0].self_attn.q_proj.weight.detach().clone()
    with torch.no_grad():
        expanded.model.layers[0].self_attn.q_proj.weight.zero_()

    checkpointer.load_model(expanded, str(tmp_path / "base" / "model"), allow_checkpoint_key_subset=True)

    assert torch.equal(expanded.model.layers[0].self_attn.q_proj.weight, pretrained_weight)
    for name, param in expansion_parameters(expanded):
        assert torch.equal(param, expected_weights[name]), name


def test_base_checkpoint_without_the_subset_flag_is_refused(tmp_path):
    """Without the flag the missing expansion keys are an error, not a silent skip.

    ``CheckpointException`` derives from ``BaseException``, so a ``pytest.raises(Exception)``
    here would let the failure through and the test would report the wrong thing.
    """
    checkpointer = _checkpointer(tmp_path)
    checkpointer.save_model(_build(expand=False), str(tmp_path / "base"))

    with pytest.raises(CheckpointException, match=r"expansion\.weight"):
        checkpointer.load_model(_build(), str(tmp_path / "base" / "model"))


def test_expanding_a_meta_device_model_is_refused():
    """Copying a meta weight is a no-op, and zero-init would hide the resulting garbage."""
    with torch.device("meta"):
        model = LlamaForCausalLM(_config())

    with pytest.raises(RuntimeError, match="meta device"):
        apply_expansion(model, ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS))


def test_deferred_initialization_matches_the_eager_path(input_ids):
    """Splitting allocation from initialization must not change the result.

    Deferring is what lets a parallel run expand a model whose weights arrive later, so
    the two paths have to agree exactly or function preservation depends on the launch
    topology.
    """
    eager = _build()

    deferred = _build(expand=False)
    apply_expansion(deferred, ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS), initialize=False)
    assert initialize_expansion(deferred) == len(list(expansion_parameters(deferred)))

    eager_weights = dict(expansion_parameters(eager))
    for name, param in expansion_parameters(deferred):
        assert torch.equal(param, eager_weights[name]), name
    assert torch.equal(_logits(deferred, input_ids), _logits(eager, input_ids))


def test_allocation_on_a_meta_device_model_is_allowed():
    """Allocation reads nothing from the pretrained weight, so meta is fine.

    This is the ordering a parallel run needs: allocate before sharding, when the weights
    are still meta, and initialize after the checkpoint load.
    """
    with torch.device("meta"):
        model = LlamaForCausalLM(_config())

    apply_expansion(model, ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS), initialize=False)
    assert list(expansion_parameters(model))

    with pytest.raises(RuntimeError, match="meta device"):
        initialize_expansion(model)


def test_initialization_works_once_both_weights_are_sharded(single_rank_mesh):
    """The case the split exists for: initialize after the model has been parallelized.

    Sharding distributes the expansion weight alongside its base weight, so by the time
    the checkpoint has loaded the copy is a local operation on each rank's shard. Both
    weights are distributed here the way ``ColwiseParallelExpanded`` would.
    """
    model = _build(expand=False)
    apply_expansion(model, ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS), initialize=False)

    for index in EXPANDED_LAYERS:
        linear = model.model.layers[index].self_attn.q_proj
        for owner in (linear, linear.expansion):
            owner.weight = torch.nn.Parameter(distribute_tensor(owner.weight.data, single_rank_mesh, [Shard(0)]))

    initialize_expansion(model)

    linear = model.model.layers[EXPANDED_LAYERS[0]].self_attn.q_proj
    assert torch.equal(linear.expansion.weight.full_tensor(), linear.weight.full_tensor())


def test_initializing_a_half_sharded_linear_is_refused(single_rank_mesh):
    """Expanding in the middle of sharding leaves the two weights incompatible."""
    model = _build(expand=False)
    apply_expansion(model, ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS), initialize=False)

    linear = model.model.layers[EXPANDED_LAYERS[0]].self_attn.q_proj
    linear.weight = torch.nn.Parameter(distribute_tensor(linear.weight.data, single_rank_mesh, [Shard(0)]))

    with pytest.raises(RuntimeError, match="disagree about being distributed"):
        initialize_expansion(model)


def test_initializing_an_unexpanded_model_is_rejected():
    """Initializing a model that was never expanded would silently do nothing."""
    with pytest.raises(ValueError, match="apply_expansion"):
        initialize_expansion(_build(expand=False))


def test_expanding_an_already_sharded_model_is_refused(single_rank_mesh):
    """Sharding runs after expansion, because it is what distributes the expansion weight."""
    model = _build(expand=False)
    linear = model.model.layers[EXPANDED_LAYERS[0]].self_attn.q_proj
    linear.weight = torch.nn.Parameter(distribute_tensor(linear.weight.data, single_rank_mesh, [Shard(0)]))

    with pytest.raises(RuntimeError, match="already distributed"):
        apply_expansion(model, ExpansionConfig(enabled=True, layers=EXPANDED_LAYERS))


def test_a_pipeline_stage_tolerates_the_other_stages_checkpoint_keys(tmp_path):
    """A stage owns only its layers, so the checkpoint carries keys it does not have.

    ``allow_checkpoint_key_subset`` normally also asserts the reverse direction -- every
    checkpoint key must exist in the model -- to catch a model built from the wrong
    architecture. A pipeline stage violates that by construction, and nothing inside the
    load can tell the two apart, so the caller declares it.
    """
    checkpointer = _checkpointer(tmp_path)
    checkpointer.save_model(_build(expand=False), str(tmp_path / "base"))

    stage = _build(perturb=0.05)
    # Keep the second half of the stack, the way pipeline splitting leaves a later stage.
    stage.model.layers = torch.nn.ModuleList(list(stage.model.layers)[LAYERS // 2 :])
    expected_weights = {name: param.detach().clone() for name, param in expansion_parameters(stage)}

    with pytest.raises(RuntimeError, match="absent from the built model"):
        checkpointer.load_model(stage, str(tmp_path / "base" / "model"), allow_checkpoint_key_subset=True)

    checkpointer.load_model(
        stage,
        str(tmp_path / "base" / "model"),
        allow_checkpoint_key_subset=True,
        model_is_pipeline_stage=True,
    )
    for name, param in expansion_parameters(stage):
        assert torch.equal(param, expected_weights[name]), name

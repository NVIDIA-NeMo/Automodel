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

"""Two-rank pipeline parity for GLM-5.3 with a linear CE loss owning the lm_head projection.

Rank 0 holds the embedding and the first two layers, rank 1 the last two layers, the final norm and
``lm_head``. The last stage emits hidden states (``_pp_return_hidden_states``) and the production
``PipelineCausalLMLoss`` + ``ChunkedCrossEntropy`` computes the loss from them. Everything must match a
single-process reference that projects logits and applies a summed cross entropy on the same labels:
the summed loss, every participating parameter gradient (including ``lm_head.weight``), which
parameters receive no gradient, the global gradient norm, and the parameters after one SGD step.
"""

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.distributed.pipelining.functional import (
    _precompute_stage_shapes,
    _use_static_pipeline_stage_metadata,
    pipeline_model,
)
from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.loss.mtp import PipelineCausalLMLoss
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM

pytestmark = pytest.mark.cpu_only

RTOL, ATOL = 2e-5, 2e-6
# All-full indexers, and an IndexShare split whose second stage begins with a "shared" layer, so the
# float32 top-k carry crosses the stage boundary and must carry a defined (zero) gradient.
INDEXER_LAYOUTS = (["full", "full", "full", "full"], ["full", "full", "shared", "shared"])


def _config(indexer_types: list[str]) -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=128,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=256,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        mlp_layer_types=["dense", "dense", "sparse", "sparse"],
        indexer_types=indexer_types,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


def _build_model(config: GlmMoeDsaConfig) -> GlmMoeDsaForCausalLM:
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(1234)
    model = GlmMoeDsaForCausalLM(config, backend=backend)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model.train()


def _check_pp2_matches_reference(rank: int, indexer_types: list[str]) -> None:
    config = _config(indexer_types)
    reference = _build_model(config)
    model = copy.deepcopy(reference)

    torch.manual_seed(7)
    batch, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch, seq_len))
    labels[0, -1] = -100  # an ignored target, so normalization is not a pure token count

    # Single-process reference: logits + summed cross entropy, the semantics ChunkedCrossEntropy implements.
    logits = reference(input_ids).logits.float()
    ref_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
    )
    ref_loss.backward()
    ref_params = dict(reference.named_parameters())

    # Production pipeline wiring: split, schedule, and the hidden-state loss boundary.
    loss_fn = ChunkedCrossEntropy(compile=False)
    mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("pp",))
    schedule, model_parts, has_first, has_last, stages = pipeline_model(
        model,
        world_mesh=mesh,
        moe_mesh=None,
        pp_axis_name="pp",
        dp_axis_names=(),
        layers_per_stage=2,
        pipeline_parallel_schedule_csv=None,
        pipeline_parallel_schedule="1f1b",
        microbatch_size=1,
        local_batch_size=batch,
        device=torch.device("cpu"),
        # As in the recipe: the schedule is built with the configured loss (which also enables its
        # backward pass) and then wrapped by PipelineCausalLMLoss for the last stage.
        loss_fn=loss_fn,
        patch_inner_model=False,
        patch_causal_lm_model=False,
    )
    assert len(model_parts) == 1
    part = model_parts[0]
    assert has_first == (rank == 0) and has_last == (rank == 1)
    if has_last:
        part._pp_return_hidden_states = True
        schedule._loss_fn = PipelineCausalLMLoss(loss_fn, part)
    # The recipe installs the model-declared stage metas (``seq_len``) so torch never runs its
    # shape-inference forward on uninitialized buffers (the IndexShare carry is an index tensor).
    # ``pipeline_model``'s ``seq_len`` path also warms up NCCL neighbours with a CUDA synchronize, so
    # on CPU the two helpers it uses are called directly.
    _precompute_stage_shapes(stages, part.config, microbatch_size=1, seq_len=seq_len, tensor_dtype=torch.float32)
    _use_static_pipeline_stage_metadata(schedule, stages)

    losses: list[torch.Tensor] = []
    if has_first:
        schedule.step(input_ids, target=labels, losses=losses)
    else:
        schedule.step(target=labels, losses=losses)

    if has_last:
        total = torch.stack(losses).sum()
        torch.testing.assert_close(total, ref_loss.detach(), rtol=RTOL, atol=ATOL)

    # Every parameter of this stage: same gradient (or the same absence of one) as the reference.
    compared, local_sq = 0, torch.zeros(())
    for name, param in part.named_parameters():
        ref_param = ref_params[name]
        assert (param.grad is None) == (ref_param.grad is None), name
        if param.grad is None:
            continue
        torch.testing.assert_close(param.grad, ref_param.grad, rtol=RTOL, atol=ATOL, msg=lambda m, n=name: f"{n}: {m}")
        local_sq = local_sq + param.grad.float().pow(2).sum()
        compared += 1
    assert compared > 0
    if has_last:
        assert part.lm_head.weight.grad is not None
    dist.all_reduce(local_sq)
    ref_norm = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in reference.parameters() if p.grad is not None))
    torch.testing.assert_close(torch.sqrt(local_sq), ref_norm, rtol=RTOL, atol=ATOL)

    # One optimizer step on both sides must leave this stage's parameters equal to the reference's.
    torch.optim.SGD(reference.parameters(), lr=0.1).step()
    torch.optim.SGD(part.parameters(), lr=0.1).step()
    for name, param in part.named_parameters():
        torch.testing.assert_close(param, ref_params[name], rtol=RTOL, atol=ATOL, msg=lambda m, n=name: f"{n}: {m}")


def _worker(rank: int, init_method: str, indexer_types: list[str]) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        _check_pp2_matches_reference(rank, indexer_types)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("indexer_types", INDEXER_LAYOUTS, ids=["all_full", "indexshare_split_on_shared"])
@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="Two spawned CPU/Gloo ranks run the real pipeline split, schedule and linear-CE loss boundary.",
)
def test_pp2_chunked_ce_matches_single_process_reference(tmp_path, indexer_types):
    torch.multiprocessing.spawn(
        _worker,
        args=((tmp_path / "pp2_rendezvous").as_uri(), indexer_types),
        nprocs=2,
        join=True,
    )

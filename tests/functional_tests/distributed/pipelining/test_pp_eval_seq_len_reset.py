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

"""PP x FSDP2 evaluation stays in lockstep when DP ranks change sequence length at different batches.

``AutoPipeline.update_seq_len`` re-initializes the pipeline stages when the batch length
changes, and torch's stage initialization reshards every FSDP module of the stage. In
evaluation, FSDP units that keep their weights gathered after forward stay gathered
between batches unless the schedule reshards the stage at the end of a step. A
rank-local re-initialization then makes that rank issue all-gathers its DP peers never
post (NVBug 7125304).

One four-rank spawn evaluates a ~10M-parameter Llama with PP2 x DP-shard2 for every
combination of stage-root wrapping and schedule, and compares the last stage's logits
with a single-process reference. Ranks share two GPUs when only two are visible: every
PP and DP-shard group spans both GPUs, and no NCCL communicator spans all four ranks.
"""

import datetime
import faulthandler
import functools
import gc
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.components.distributed.pipelining.autopipeline import AutoPipeline

pytestmark = [
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs"),
    # The first test also pays for the shared four-rank spawn.
    pytest.mark.timeout(360),
]

_PP_SIZE = 2
_DP_SHARD_SIZE = 2
_WORLD_SIZE = _PP_SIZE * _DP_SHARD_SIZE
# Bounds the rendezvous, which waits for every rank's CUDA context, and the Gloo barriers.
_RENDEZVOUS_TIMEOUT = datetime.timedelta(minutes=3)
# Ends a rank stuck in a collective that its peers never post. DeviceMesh subgroups keep
# NCCL's default timeout, which outlasts the pytest timeout.
_HANG_TIMEOUT_SECONDS = 240
_BATCH_SIZE = 2
_VOCAB_SIZE = 8192
# Sequence length of each evaluation batch, per DP-shard rank. Batches 1 and 3 change the
# length only on DP rank 0, and batch 2 only on DP rank 1.
_SEQ_LENS = ((64, 128, 128, 64), (64, 64, 96, 96))
_STAGE_ROOTS = ("causal_lm", "inner_model")
_SCHEDULES = ("1f1b", "interleaved1f1b")
_CASES = [(stage_root, schedule) for stage_root in _STAGE_ROOTS for schedule in _SCHEDULES]
# Both sides run identical float32 kernels on identical shapes; FSDP and PP copy tensors exactly.
_LOGIT_TOLERANCE = 1e-4


def _config() -> LlamaConfig:
    """Return a Llama config with about 10.5M parameters."""
    return LlamaConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _build_model() -> LlamaForCausalLM:
    """Build the float32 model on the CPU with fixed weights, so every rank and the reference agree."""
    torch.manual_seed(0)
    return LlamaForCausalLM(_config()).to(dtype=torch.float32)


def _parallelize(
    model: nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None = None,
    *,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    stage_root: str,
) -> None:
    """Shard one pipeline stage the way the dense or the MoE parallelizer lays it out.

    Every unit keeps its weights gathered after forward, as with pipeline parallelism or
    ``moe.reshard_after_forward: false``.

    Args:
        model: Pipeline stage of a ``LlamaForCausalLM``; modules owned by other stages are None.
        world_mesh: Device mesh with the DP-shard axis.
        moe_mesh: Unused.
        dp_axis_names: DP axis names; the first one shards the stage.
        cp_axis_name: Unused.
        tp_axis_name: Unused.
        ep_axis_name: Unused.
        ep_shard_axis_names: Unused.
        stage_root: ``"causal_lm"`` wraps the stage root, like the dense parallelizer.
            ``"inner_model"`` wraps the embeddings, the LM head and the inner model but not
            the stage root, like the MoE parallelizer with ``wrap_outer_model: false``.
    """
    dp_mesh = world_mesh[dp_axis_names[0]]
    shard = functools.partial(fully_shard, mesh=dp_mesh, reshard_after_forward=False)
    for layer in model.model.layers.values():
        shard(layer)
    if stage_root == "causal_lm":
        shard(model)
        return
    for module in (model.model.embed_tokens, model.lm_head):
        if module is not None:
            shard(module)
    shard(model.model)


def _evaluate(mesh: DeviceMesh, device: torch.device, stage_root: str, schedule: str) -> dict[str, object]:
    """Run the evaluation batches of one case and return this rank's parity record.

    Args:
        mesh: Four-rank mesh with axes ``("pp", "dp_shard")``.
        device: This rank's CUDA device.
        stage_root: Which module is the FSDP root of each stage; see ``_parallelize``.
        schedule: Pipeline schedule name.

    Returns:
        Whether this rank owns the last stage, and the largest absolute logit difference
        from the reference over all batches (0.0 on other ranks).
    """
    dp_rank = mesh.get_local_rank("dp_shard")
    captured = []

    def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Record a microbatch's logits and return its token-mean cross-entropy loss.

        Args:
            logits: Tensor of shape [batch, sequence, vocab], the last stage's float32 output.
            target: Tensor of shape [batch, sequence] holding token IDs.

        Returns:
            Scalar float32 tensor.
        """
        captured.append(logits.detach().clone())
        return F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1))

    pipeline = AutoPipeline(
        world_mesh=mesh,
        pp_axis_name="pp",
        dp_axis_names=("dp_shard",),
        pp_schedule=schedule,
        pp_microbatch_size=1,
        pp_batch_size=_BATCH_SIZE,
        # Two stages per rank for the interleaved schedule.
        layers_per_stage=2 if schedule == "interleaved1f1b" else None,
        device=device,
        dtype=torch.float32,
    ).build(
        # Stage splitting deep-copies the model on the CPU; FSDP moves each stage's shards to the GPU.
        _build_model(),
        loss_fn=loss_fn,
        parallelize_fn=functools.partial(_parallelize, stage_root=stage_root),
    )
    for part in pipeline.parts:
        part.eval()
    has_first, has_last = pipeline.info.has_first_stage, pipeline.info.has_last_stage
    reference = _build_model().to(device).eval() if has_last else None

    max_abs_err = 0.0
    with torch.no_grad():
        for batch_idx, seq_len in enumerate(_SEQ_LENS[dp_rank]):
            generator = torch.Generator().manual_seed(1000 * batch_idx + dp_rank)
            input_ids = torch.randint(_VOCAB_SIZE, (_BATCH_SIZE, seq_len), generator=generator).to(device)
            captured.clear()
            # Same calls as the recipe's validation step.
            pipeline.update_seq_len(seq_len)
            pipeline.info.schedule.eval(
                *((input_ids,) if has_first else ()),
                target=input_ids if has_last else None,
                losses=[] if has_last else None,
            )
            if has_last:
                logits = torch.cat(captured)
                expected = torch.cat([reference(input_ids=row[None]).logits for row in input_ids])
                assert logits.shape == expected.shape, (logits.shape, expected.shape)
                max_abs_err = max(max_abs_err, (logits - expected).abs().max().item())
        # The validation epoch ends with a DP all-reduce of the loss totals.
        total = torch.ones(1, device=device)
        dist.all_reduce(total, group=mesh["dp_shard"].get_group())
        assert total.item() == mesh["dp_shard"].size()

    del pipeline, reference
    gc.collect()
    torch.cuda.empty_cache()
    return {"has_last_stage": has_last, "max_abs_logit_err": max_abs_err}


def _run_cases(rank: int, results_dir: str) -> None:
    """Evaluate every case on this rank, writing its records after each case."""
    # A rank stuck past the hang timeout writes its stacks here and exits; the test reports them.
    with open(Path(results_dir, f"rank{rank}.stacks"), "w") as stacks:
        faulthandler.dump_traceback_later(_HANG_TIMEOUT_SECONDS, exit=True, file=stacks)
        try:
            _run_cases_on_gpu(rank, results_dir)
        finally:
            faulthandler.cancel_dump_traceback_later()


def _run_cases_on_gpu(rank: int, results_dir: str) -> None:
    """Join the four-rank group on this rank's GPU and evaluate every case."""
    # The mesh is row-major: rank = pp_index * _DP_SHARD_SIZE + dp_index. With two GPUs,
    # rank r runs on GPU (pp_index + dp_index) % 2, so every PP group (a mesh column) and
    # every DP-shard group (a mesh row) spans both GPUs. Keep the standard layout: process
    # groups order their ranks by global rank, and FSDP's all-gather assumes that order
    # matches the mesh coordinates.
    pp_index, dp_index = divmod(rank, _DP_SHARD_SIZE)
    gpu = rank if torch.cuda.device_count() >= _WORLD_SIZE else (pp_index + dp_index) % 2
    # NCCL selects unbound subgroup devices from LOCAL_RANK during initialization.
    os.environ["LOCAL_RANK"] = str(gpu)
    device = torch.device("cuda", gpu)
    torch.cuda.set_device(device)
    # Create the CUDA context before any rank starts GPU work: context creation on a shared
    # GPU serializes, and other ranks' kernels would otherwise wait behind it.
    torch.zeros(1, device=device)
    # The four-rank world group uses CPU tensors and may span ranks sharing a GPU.
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{results_dir}/store",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=_RENDEZVOUS_TIMEOUT,
    )
    try:
        dist.all_reduce(torch.zeros(1))
        # Each PP and DP-shard group spans distinct GPUs and can use NCCL.
        mesh = init_device_mesh(
            "cuda",
            (_PP_SIZE, _DP_SHARD_SIZE),
            mesh_dim_names=("pp", "dp_shard"),
            backend_override={"pp": "nccl", "dp_shard": "nccl"},
        )
        records = {}
        for stage_root, schedule in _CASES:
            records[f"{stage_root}-{schedule}"] = _evaluate(mesh, device, stage_root, schedule)
            Path(results_dir, f"rank{rank}.json").write_text(json.dumps(records))
            # Start every case together.
            dist.all_reduce(torch.zeros(1))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def case_records(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[dict], str | None]:
    """Spawn four ranks once.

    Returns:
        Each rank's records, keyed by case, and a report of the spawn failure with the
        stacks of any rank that hung, or None when every rank finished.
    """
    results_dir = tmp_path_factory.mktemp("pp_eval_seq_len_reset")
    failure = None
    try:
        mp.spawn(_run_cases, args=(str(results_dir),), nprocs=_WORLD_SIZE, join=True)
    except Exception as error:  # Reported by the cases the failure cut short.
        hung = [(rank, results_dir / f"rank{rank}.stacks") for rank in range(_WORLD_SIZE)]
        stacks = [f"rank {rank} stacks:\n{path.read_text()}" for rank, path in hung if path.stat().st_size]
        failure = "\n".join([f"{type(error).__name__}: {error}", *stacks])
    per_rank = []
    for rank in range(_WORLD_SIZE):
        path = results_dir / f"rank{rank}.json"
        per_rank.append(json.loads(path.read_text()) if path.exists() else {})
    return per_rank, failure


@pytest.mark.parametrize(("stage_root", "schedule"), _CASES)
def test_pp_eval_with_rank_local_seq_len_changes(
    case_records: tuple[list[dict], str | None], stage_root: str, schedule: str
) -> None:
    """Every rank finishes all batches, and the last stage's logits match the reference."""
    per_rank, failure = case_records
    case = f"{stage_root}-{schedule}"
    missing = [rank for rank, records in enumerate(per_rank) if case not in records]
    assert not missing, (
        f"ranks {missing} did not finish {case}; a hang here means DP peers issued different FSDP "
        f"collectives after a rank-local stage reset.\n{failure}"
    )
    last_stage = [records[case] for records in per_rank if records[case]["has_last_stage"]]
    assert len(last_stage) == _DP_SHARD_SIZE, last_stage
    for record in last_stage:
        assert record["max_abs_logit_err"] <= _LOGIT_TOLERANCE, record

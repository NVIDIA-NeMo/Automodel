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

"""Native, TE, no-PP and composed PP/CP/EP checkpoint continuation gates."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import socket
import traceback
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
from nemo_automodel.components.distributed.config import MoEParallelizerConfig
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.loss import MaskedCrossEntropyConfig
from nemo_automodel.components.loss.utils import calculate_loss
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.optim import build_optimizer
from nemo_automodel.components.optim.optimizer import AdamWConfig
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from tests.functional_tests.models.flint.reference import full_native_tensors, tiny_config, torch_backend
from tests.functional_tests.models.flint.run_parallel_composition import _config, _observe_parts, _packed
from tests.functional_tests.models.flint.test_attention import _provenance, attention_backend, te_config
from tests.functional_tests.models.flint.test_parallel_training import _runtime_receipt


def _snapshot(value: Any) -> Any:
    """Copy nested training state without retaining live tensors or shards.

    Args:
        value: Optimizer/scheduler/RNG tree containing arbitrary-shape CPU/CUDA
            tensors. DTensors use native global layouts and existing placements.
            All ranks must traverse the same DTensor-containing tree in order.

    Returns:
        Independent CPU tensor copies in the same tree. DTensors are gathered
        collectively into ordinary global tensors; scalar metadata is unchanged.
    """
    if isinstance(value, torch.Tensor):
        if isinstance(value, DTensor):
            value = value.full_tensor()
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return value


def _compare(actual: Any, expected: Any, *, atol: float, rtol: float, path: str = "state") -> float:
    """Compare independent state trees and return their largest tensor error.

    Args:
        actual: Snapshot tree whose tensor leaves are ordinary CPU tensors with
            arbitrary native global shapes, dtypes, and no autograd history.
        expected: Matching snapshot tree. Both trees remain unchanged.
        atol: Absolute tolerance for floating model parameters/Adam moments.
        rtol: Relative tolerance for floating model parameters/Adam moments.
        path: Diagnostic tree path; Adam step counters always compare exactly.

    Returns:
        Largest absolute floating-tensor error. Integer/RNG tensors, Adam step
        counters, shapes, dtypes, and all non-tensor metadata must match exactly.
    """
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
        if not expected.is_floating_point() or path.endswith(".step"):
            assert torch.equal(actual, expected), path
            return 0.0
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda detail: f"{path}\n{detail}")
        return float((actual - expected).abs().max()) if actual.numel() else 0.0
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        return max(
            (_compare(actual[key], item, atol=atol, rtol=rtol, path=f"{path}.{key}") for key, item in expected.items()),
            default=0.0,
        )
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected), path
        return max(
            (
                _compare(left, right, atol=atol, rtol=rtol, path=f"{path}.{index}")
                for index, (left, right) in enumerate(zip(actual, expected))
            ),
            default=0.0,
        )
    assert actual == expected, path
    return 0.0


def _new_training_state(
    setup: DistributedSetup,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    """Build a fresh public model before binding the optimizer to its shards.

    Args:
        setup: Real two-rank FSDP setup; both ranks enter model construction.

    Returns:
        Training model with FP32 CUDA DTensor parameters in native global
        layouts, AdamW bound to those parameters, and its StepLR scheduler.
    """
    model = NeMoAutoModelForCausalLM.from_config(
        tiny_config(torch.float32),
        dtype=torch.float32,
        attn_implementation="eager",
        backend=torch_backend(),
        use_liger_kernel=False,
        use_sdpa_patching=False,
        distributed_setup=setup,
    ).train()
    assert all(isinstance(parameter, DTensor) and parameter.dtype == torch.float32 for parameter in model.parameters())
    optimizer = AdamWConfig(lr=0.004, weight_decay=0.02, eps=1e-6).build(model)[0]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)
    return model, optimizer, scheduler


def _random_batch(device: torch.device) -> dict[str, torch.Tensor]:
    """Consume every saved RNG family to construct already-shifted ragged data.

    Args:
        device: This rank's allocated CUDA device.

    Returns:
        Independent CUDA input_ids/labels int64 [1,5] and attention_mask bool
        [1,5]. Python/NumPy/Torch CPU draws affect tokens/length, while token
        draws use the rank's CUDA generator. Labels are shifted exactly once;
        invalid labels are -100 and a valid EOS=2 is retained despite pad=2.
    """
    offset = random.randrange(31) + int(np.random.randint(31)) + int(torch.randint(31, ()).item())
    tokens = (torch.randint(31, (6,), device=device) + offset) % 31
    tokens[1] = 2
    length = random.randrange(2, 6)
    valid = torch.arange(5, device=device) < length
    ids, labels = tokens[:-1].clone(), tokens[1:].clone()
    ids[~valid], labels[~valid] = 2, -100
    return {"input_ids": ids[None], "labels": labels[None], "attention_mask": valid[None]}


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    setup: DistributedSetup,
    device: torch.device,
) -> dict[str, Any]:
    """Take one real globally normalized, clipped two-rank AdamW update.

    Args:
        model: FSDP model with FP32 CUDA DTensor parameters in native layouts.
        optimizer: AdamW bound after model sharding; its moments remain FP32.
        scheduler: Actual per-update StepLR associated with optimizer.
        setup: Distributed mesh used for the production global gradient norm.
        device: This rank's allocated CUDA device.

    Returns:
        Scalar global loss/norm and copied CPU int64/bool [1,5] batch tensors.
        The optimizer/scheduler advance once; model gradients are cleared so
        snapshots represent an update boundary rather than accumulated grads.
    """
    batch = _random_batch(device)
    total = (batch["labels"] != -100).sum()
    dist.all_reduce(total)
    optimizer.zero_grad(set_to_none=True)
    prepare_for_grad_accumulation([model])
    prepare_for_final_backward([model])
    with get_sync_ctx(model, True, defer_fsdp_grad_sync=True):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
        loss = calculate_loss(
            MaskedCrossEntropyConfig().build(), logits=logits, labels=batch["labels"], num_label_tokens=total
        )
        (loss * 2).backward()  # Undo FSDP averaging after global-token normalization, as the recipe does.
    norm = scale_grads_and_clip_grad_norm(
        0.3, [model], device_mesh=setup.mesh_context.device_mesh, foreach=True, grad_norm_backend="triton"
    )
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    global_loss = loss.detach().clone()
    dist.all_reduce(global_loss)
    optimizer.step()
    scheduler.step()
    assert all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values())
    optimizer.zero_grad(set_to_none=True)
    return {"loss": global_loss.item(), "norm": norm.item(), "batch": _snapshot(batch)}


def _worker(rank: int, port: int, directory: str) -> None:
    """Save after two updates, then compare two uninterrupted/restored updates."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=5))
    checkpoint = None
    try:
        rng = StatefulRNG(146, ranked=True)
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
                ),
                defer_fsdp_grad_sync=True,
                enable_fsdp2_prefetch=False,
            ),
            parallelism_sizes=ParallelismSizes(dp_size=2, tp_size=1, cp_size=1, pp_size=1, ep_size=1),
            activation_checkpointing=False,
            world_size=2,
        )
        model, optimizer, scheduler = _new_training_state(setup)
        for _ in range(2):
            _step(model, optimizer, scheduler, setup, device)
        saved_model = _snapshot(full_native_tensors(model))
        saved_optimizer = _snapshot(optimizer.state_dict())
        saved_scheduler = _snapshot(scheduler.state_dict())
        path = str(Path(directory) / "step_2")
        checkpoint = CheckpointingConfig(
            checkpoint_dir=directory, model_save_format="torch_save", save_consolidated=False, is_async=False
        ).build(dp_rank=rank, tp_rank=0, pp_rank=0, process_group=dist.group.WORLD)
        checkpoint.save_model(model, path)
        checkpoint.save_optimizer(optimizer, model, path, scheduler)
        checkpoint.save_on_global_ranks(rng, "rng", path)
        saved_rng = _snapshot(rng.state_dict())
        dist.barrier()

        continuous = [_step(model, optimizer, scheduler, setup, device) for _ in range(2)]
        final_model = _snapshot(full_native_tensors(model))
        final_optimizer = _snapshot(optimizer.state_dict())
        final_scheduler = _snapshot(scheduler.state_dict())
        final_rng = _snapshot(rng.state_dict())

        restored, restored_optimizer, restored_scheduler = _new_training_state(setup)
        checkpoint.load_model(restored, str(Path(path) / "model"))
        checkpoint.load_optimizer(restored_optimizer, restored, path, restored_scheduler)
        # Fresh initialization and checkpoint setup can consume RNG; restore it last.
        checkpoint.load_on_global_ranks(rng, "rng", path)
        _compare(_snapshot(full_native_tensors(restored)), saved_model, atol=0, rtol=0)
        _compare(_snapshot(restored_optimizer.state_dict()), saved_optimizer, atol=0, rtol=0)
        _compare(restored_scheduler.state_dict(), saved_scheduler, atol=0, rtol=0)
        _compare(_snapshot(rng.state_dict()), saved_rng, atol=0, rtol=0)

        resumed = [_step(restored, restored_optimizer, restored_scheduler, setup, device) for _ in range(2)]
        resumed_model = _snapshot(full_native_tensors(restored))
        resumed_optimizer = _snapshot(restored_optimizer.state_dict())
        _compare(restored_scheduler.state_dict(), final_scheduler, atol=0, rtol=0)
        _compare(_snapshot(rng.state_dict()), final_rng, atol=0, rtol=0)
        for left, right in zip(resumed, continuous):
            _compare(left["batch"], right["batch"], atol=0, rtol=0)
        # CUDA collective scheduling may change last-bit accumulation. Record
        # actual errors and require FP32 closeness rather than bitwise CUDA math.
        atol, rtol = 2e-6, 2e-5
        model_error = _compare(resumed_model, final_model, atol=atol, rtol=rtol)
        optimizer_error = _compare(resumed_optimizer, final_optimizer, atol=atol, rtol=rtol)
        losses = torch.tensor([result["loss"] for result in continuous], dtype=torch.float64)
        resumed_losses = torch.tensor([result["loss"] for result in resumed], dtype=torch.float64)
        torch.testing.assert_close(resumed_losses, losses, atol=atol, rtol=rtol)
        norms = torch.tensor([result["norm"] for result in continuous], dtype=torch.float64)
        resumed_norms = torch.tensor([result["norm"] for result in resumed], dtype=torch.float64)
        torch.testing.assert_close(resumed_norms, norms, atol=atol, rtol=rtol)
        if rank == 0:
            Path(directory, "checkpoint_resume_metrics.json").write_text(
                json.dumps(
                    {
                        "world_size": 2,
                        "dtype": "float32",
                        "checkpoint_format": "native_torch_save",
                        "saved_after_updates": 2,
                        "continuation_updates": 2,
                        "restored_boundary_state_exact": True,
                        "rank_local_rng_and_continuation_batches_exact": True,
                        "scheduler_and_adam_steps_exact": True,
                        "continuous_losses": losses.tolist(),
                        "resumed_losses": resumed_losses.tolist(),
                        "loss_max_abs_error": float((resumed_losses - losses).abs().max()),
                        "global_norm_max_abs_error": float((resumed_norms - norms).abs().max()),
                        "parameter_max_abs_error": model_error,
                        "optimizer_state_max_abs_error": optimizer_error,
                        "atol": atol,
                        "rtol": rtol,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
    finally:
        if checkpoint is not None:
            checkpoint.close()
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_rank_native_checkpoint_continuation(tmp_path: Path) -> None:
    """Use exactly two allocated GPUs and actual DCP model/optimizer restore."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=2, join=True)
    evidence = tmp_path / "checkpoint_resume_metrics.json"
    assert evidence.is_file()
    print(f"FLINT_CHECKPOINT_EVIDENCE={evidence}")


def _optimizer_snapshot(value: Any) -> Any:
    """Copy each rank's exact state without unsupported int16 NCCL gathers.

    Args:
        value: Optimizer state tree with BF16 moments, int16 master remainders
            and arbitrary global/native DTensor shapes and placements.

    Returns:
        Independent CPU copies of every local byte, accompanied by each
        DTensor's global shape, mesh names and placements. Every rank compares
        its own restored shards; no optimizer value is cast or gathered.
    """
    if isinstance(value, DTensor):
        return {
            "local": value.to_local().detach().cpu().clone(),
            "global_shape": tuple(value.shape),
            "mesh_names": value.device_mesh.mesh_dim_names,
            "placements": tuple(str(placement) for placement in value.placements),
        }
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _optimizer_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_optimizer_snapshot(item) for item in value)
    return value


def _new_state(
    setup: DistributedSetup,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    """Construct actual public BF16/F32 FSDP model and recipe-style TE optimizer.

    Args:
        setup: Real two-rank DP/FSDP mesh with BF16 bulk compute and FP32 reductions.

    Returns:
        Native model with BF16 bulk and protected FP32 head/sink DTensors,
        TE FusedAdam with FP32 masters represented by int16 remainders for BF16
        parameters and BF16 moments, and a per-update StepLR scheduler.
    """
    model = NeMoAutoModelForCausalLM.from_config(
        te_config(),
        dtype=torch.bfloat16,
        backend=attention_backend("te"),
        distributed_setup=setup,
        attn_implementation="eager",
        use_liger_kernel=False,
        use_sdpa_patching=False,
    ).train()
    for name, parameter in model.named_parameters():
        assert isinstance(parameter, DTensor), name
        protected = "._fp32_params." in name or name == "lm_head.weight"
        assert parameter.dtype == (torch.float32 if protected else torch.bfloat16), name
    optimizer = build_optimizer(
        model,
        (
            "transformer_engine.pytorch.optimizers.FusedAdam",
            dict(
                lr=0.003,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                adam_w_mode=True,
                master_weights=True,
                master_weight_dtype="float32",
                store_param_remainders=True,
                exp_avg_dtype="bfloat16",
                exp_avg_sq_dtype="bfloat16",
            ),
        ),
        device_mesh=setup.mesh_context.device_mesh,
    )[0]
    return model, optimizer, torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)


def _optimizer_precision(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Validate initialized local/global state and count each protected format.

    Args:
        optimizer: Actual TE FusedAdam bound to native arbitrary-shape DTensors.

    Returns:
        Counts of BF16 bulk states, FP32 protected states and int16 remainder
        tensors. State tensors keep existing global shapes/placements.
    """
    counts = {"bf16_parameters": 0, "fp32_parameters": 0, "int16_remainders": 0}
    for parameter, state in optimizer.state.items():
        if not state:
            continue
        counts["bf16_parameters" if parameter.dtype == torch.bfloat16 else "fp32_parameters"] += 1
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].dtype == torch.bfloat16, name
            local = state[name].to_local() if isinstance(state[name], DTensor) else state[name]
            assert torch.isfinite(local).all(), name
        if parameter.dtype == torch.bfloat16:
            assert state["master_param"].dtype == torch.int16
            counts["int16_remainders"] += 1
    assert all(counts.values()), counts
    return counts


def _te_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    setup: DistributedSetup,
    device: torch.device,
) -> dict[str, Any]:
    """Take one globally normalized actual CUDA optimizer update.

    Args:
        model: Public FSDP Flint with BF16 bulk and protected FP32 head/sinks.
        optimizer: Recipe-built TE FusedAdam with mixed state formats.
        scheduler: Scheduler advanced exactly once after the optimizer step.
        setup: DP mesh used for production gradient normalization/clipping.
        device: Rank-local CUDA device for already-shifted ragged [1,5] batches.

    Returns:
        Copied batch tensors, scalar global loss/norm, and observed state-format
        counts. Inputs consume Python, NumPy, CPU and CUDA RNG families.
    """
    batch = _random_batch(device)
    total = (batch["labels"] != -100).sum()
    dist.all_reduce(total)
    optimizer.zero_grad(set_to_none=True)
    prepare_for_grad_accumulation([model])
    prepare_for_final_backward([model])
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten(), reduction="sum") / total
    (loss * 2).backward()
    norm = scale_grads_and_clip_grad_norm(
        0.3, [model], device_mesh=setup.mesh_context.device_mesh, foreach=True, grad_norm_backend="triton"
    )
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    global_loss = loss.detach().clone()
    dist.all_reduce(global_loss)
    optimizer.step()
    scheduler.step()
    counts = _optimizer_precision(optimizer)
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(global_loss), "norm": float(norm), "batch": _snapshot(batch), "precision_counts": counts}


def _te_worker(rank: int, port: int, directory: str) -> None:
    """Save model/optimizer/scheduler/RNG after two updates and continue twice."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=8))
    checkpointer = None
    try:
        provenance = _provenance()
        rng = StatefulRNG(437528, ranked=True)
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                    cast_forward_inputs=False,
                ),
                enable_fsdp2_prefetch=False,
                defer_fsdp_grad_sync=False,
            ),
            parallelism_sizes=ParallelismSizes(dp_size=2, ep_size=1, pp_size=1, cp_size=1, tp_size=1),
            activation_checkpointing=False,
            world_size=2,
        )
        model, optimizer, scheduler = _new_state(setup)
        initial = _snapshot(full_native_tensors(model))
        for _ in range(2):
            _te_step(model, optimizer, scheduler, setup, torch.device("cuda", rank))
        saved_model = _snapshot(full_native_tensors(model))
        saved_optimizer = _optimizer_snapshot(optimizer.state_dict())
        saved_scheduler = _snapshot(scheduler.state_dict())
        assert any(
            not torch.equal(value, initial[name])
            for name, value in saved_model.items()
            if value.dtype == torch.bfloat16
        ), "BF16 resident updates must be observable"
        path = str(Path(directory) / "step_2")
        checkpointer = CheckpointingConfig(
            checkpoint_dir=directory,
            model_save_format="torch_save",
            save_consolidated=False,
            is_async=False,
        ).build(dp_rank=rank, tp_rank=0, pp_rank=0, process_group=dist.group.WORLD)
        checkpointer.save_model(model, path)
        checkpointer.save_optimizer(optimizer, model, path, scheduler)
        checkpointer.save_on_global_ranks(rng, "rng", path)
        checkpointer.async_wait()
        saved_rng = _snapshot(rng.state_dict())
        dist.barrier()
        continuous = [_te_step(model, optimizer, scheduler, setup, torch.device("cuda", rank)) for _ in range(2)]
        final_model = _snapshot(full_native_tensors(model))
        final_optimizer = _optimizer_snapshot(optimizer.state_dict())
        final_scheduler = _snapshot(scheduler.state_dict())
        final_rng = _snapshot(rng.state_dict())

        restored, restored_optimizer, restored_scheduler = _new_state(setup)
        checkpointer.load_model(restored, str(Path(path) / "model"))
        checkpointer.load_optimizer(restored_optimizer, restored, path, restored_scheduler)
        checkpointer.load_on_global_ranks(rng, "rng", path)
        _compare(_snapshot(full_native_tensors(restored)), saved_model, atol=0, rtol=0)
        _compare(_optimizer_snapshot(restored_optimizer.state_dict()), saved_optimizer, atol=0, rtol=0)
        _compare(_snapshot(restored_scheduler.state_dict()), saved_scheduler, atol=0, rtol=0)
        _compare(_snapshot(rng.state_dict()), saved_rng, atol=0, rtol=0)
        _optimizer_precision(restored_optimizer)
        resumed = [
            _te_step(restored, restored_optimizer, restored_scheduler, setup, torch.device("cuda", rank))
            for _ in range(2)
        ]
        model_error = _compare(_snapshot(full_native_tensors(restored)), final_model, atol=0, rtol=0)
        optimizer_error = _compare(
            _optimizer_snapshot(restored_optimizer.state_dict()), final_optimizer, atol=0, rtol=0
        )
        _compare(_snapshot(restored_scheduler.state_dict()), final_scheduler, atol=0, rtol=0)
        _compare(_snapshot(rng.state_dict()), final_rng, atol=0, rtol=0)
        for actual_step, expected_step in zip(resumed, continuous, strict=True):
            _compare(actual_step["batch"], expected_step["batch"], atol=0, rtol=0)
            _compare(actual_step["loss"], expected_step["loss"], atol=0, rtol=0)
            _compare(actual_step["precision_counts"], expected_step["precision_counts"], atol=0, rtol=0)
        norms = torch.tensor([item["norm"] for item in continuous], dtype=torch.float64)
        resumed_norms = torch.tensor([item["norm"] for item in resumed], dtype=torch.float64)
        assert _provenance()["source_sha256"] == provenance["source_sha256"]
        record = {
            "status": "measured_before_final_scalar_checks",
            "scope": "actual two-rank public Checkpointer/DCP model and TE FusedAdam optimizer continuation",
            "provenance": provenance,
            "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "world_size": 2,
            "rank": rank,
            "resident_bulk_dtype": "bfloat16",
            "protected_dtype": "float32",
            "optimizer_master_precision": "float32 represented by BF16 resident + int16 remainder",
            "optimizer_moments_dtype": "bfloat16",
            "precision_counts": resumed[0]["precision_counts"],
            "saved_after_updates": 2,
            "continuation_updates": 2,
            "restored_boundary_exact": True,
            "model_max_abs": model_error,
            "optimizer_state_max_abs": optimizer_error,
            "losses": [item["loss"] for item in continuous],
            "resumed_losses": [item["loss"] for item in resumed],
            "norms": [item["norm"] for item in continuous],
            "resumed_norms": [item["norm"] for item in resumed],
            "scheduler_rng_batches_and_all_optimizer_states_exact": True,
            "loss_max_abs": max(abs(a["loss"] - b["loss"]) for a, b in zip(resumed, continuous, strict=True)),
            "norm_max_abs": max(abs(a["norm"] - b["norm"]) for a, b in zip(resumed, continuous, strict=True)),
            "strict_bitwise_norm_equal": torch.equal(resumed_norms, norms),
            "norm_protocol": {
                "atol": 2e-6,
                "rtol": 2e-5,
                "source": "tests/functional_tests/models/flint/test_checkpoint_resume.py",
                "scope": "auxiliary reduced norm logs only; all state, batch and loss assertions remain bitwise",
            },
        }
        rank_record = Path(directory, f"te_dcp_resume_rank_{rank}.json")
        rank_record.write_text(json.dumps(record, indent=2) + "\n")
        # Match the existing checkpoint protocol's reduced-norm comparison:
        # parallel CUDA reductions can differ in their last accumulated bits.
        # Unlike that older test, every state tensor and loss above stays exact.
        torch.testing.assert_close(resumed_norms, norms, atol=2e-6, rtol=2e-5)
        record["status"] = "rank_passed"
        rank_record.write_text(json.dumps(record, indent=2) + "\n")
        dist.barrier()
        if rank == 0:
            record["status"] = "passed"
            Path(directory, "te_dcp_resume.json").write_text(json.dumps(record, indent=2) + "\n")
    finally:
        if checkpointer is not None:
            checkpointer.close()
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_rank_te_mixed_optimizer_public_checkpoint(tmp_path: Path) -> None:
    """Require exact public DCP continuation; norm logs follow the existing protocol."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    directory = Path(os.environ.get("FLINT_BACKEND_EVIDENCE", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    mp.spawn(_te_worker, args=(port, str(directory)), nprocs=2, join=True)
    assert (directory / "te_dcp_resume.json").is_file()


@dataclass
class _State:
    """One public no-pipeline model, its actual optimizer and observed precision."""

    model: torch.nn.Module
    setup: DistributedSetup
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.StepLR
    observations: list[dict[str, Any]]
    handles: list[Any]


def _nopp_new_state(source: str, *, activation_checkpointing: bool) -> _State:
    """Build BF16 native shards and protected FP32 units through the public loader.

    Args:
        source: Test-owned three-layer HF checkpoint with canonical expert keys.
        activation_checkpointing: Apply the actual shared decoder checkpoint wrapper.

    Returns:
        Public PP1/CP2/EP2 model with native arbitrary-shape DTensor shards,
        TE FusedAdam with BF16 live moments/int16 remainders, and StepLR.
    """
    setup = DistributedSetup.build(
        strategy=FSDP2Config(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=False,
            ),
            enable_fsdp2_prefetch=False,
            defer_fsdp_grad_sync=False,
        ),
        parallelism_sizes=ParallelismSizes(dp_size=1, cp_size=2, ep_size=2, pp_size=1, tp_size=1),
        moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=True),
        activation_checkpointing=activation_checkpointing,
        world_size=2,
    )
    model = NeMoAutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        backend=replace(attention_backend("te"), experts="torch_mm", dispatcher="deepep"),
        distributed_setup=setup,
        attn_implementation="eager",
        use_liger_kernel=False,
        use_sdpa_patching=False,
        local_files_only=True,
    ).train()
    for name, parameter in model.named_parameters():
        protected = "._fp32_params." in name or name == "lm_head.weight"
        assert parameter.dtype == (torch.float32 if protected else torch.bfloat16), name
    optimizer = build_optimizer(
        model,
        (
            "transformer_engine.pytorch.optimizers.FusedAdam",
            dict(
                lr=0.003,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                adam_w_mode=True,
                master_weights=True,
                master_weight_dtype="float32",
                store_param_remainders=True,
                exp_avg_dtype="bfloat16",
                exp_avg_sq_dtype="bfloat16",
            ),
        ),
        device_mesh=setup.mesh_context.device_mesh,
    )[0]
    observations: list[dict[str, Any]] = []
    return _State(
        model,
        setup,
        optimizer,
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7),
        observations,
        _observe_parts([model], observations),
    )


def _batch(device: torch.device) -> dict[str, Any]:
    """Draw two document-aware packs while consuming all checkpointed RNG families.

    Args:
        device: Rank-local CUDA device.

    Returns:
        Identical global IDs/labels/positions [2,128] on both CP ranks, real and
        padded lengths [2,2]. Labels shift once inside documents of length 5/7;
        valid EOS2 and physical pad30 are distinct positional contracts.
    """
    batch = _packed(device, flatten=False)
    batch = {name: value[:2].clone() if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
    offset = random.randrange(29) + int(np.random.randint(29)) + int(torch.randint(29, ()).item())
    for row in range(2):
        for start, length in ((0, 5), (64, 7)):
            values = (torch.randint(29, (length,), device=device) + offset) % 29
            values[1] = 2
            dist.broadcast(values, src=0)
            batch["input_ids"][row, start : start + length] = values
            batch["labels"][row, start : start + length - 1] = values[1:]
    return batch


def _live_optimizer(state: _State) -> list[list[dict[str, Any]]]:
    """Snapshot raw local optimizer representations in stable parameter-group order.

    Args:
        state: Native optimizer containing arbitrary global/native DTensor shapes.

    Returns:
        Independent local BF16 moment/int16 remainder and scale-state bytes,
        plus parameter global shapes/dtypes; no unsupported int16 gather occurs.
    """
    return [
        [
            {
                "parameter_dtype": str(parameter.dtype),
                "global_shape": tuple(parameter.shape),
                "state": _optimizer_snapshot(state.optimizer.state.get(parameter, {})),
            }
            for parameter in group["params"]
        ]
        for group in state.optimizer.param_groups
    ]


def _nopp_snapshot(state: _State) -> dict[str, Any]:
    """Copy actual local model and public/raw optimizer trees without value casts.

    Args:
        state: Public native model with arbitrary-shape local/DTensor parameters.

    Returns:
        Independent CPU local shards and global shape/placement metadata,
        public FP32 optimizer transport, raw live BF16/int16 storage and scheduler.
        Only inactive empty non-FP8 TE extra state is omitted by native safetensors.
    """
    tensors = {}
    for name, value in state.model.state_dict().items():
        if name.endswith("_extra_state"):
            assert value is None or (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.device.type == "cpu"
                and value.numel() == 0
            ), name
        else:
            tensors[name] = value
    return {
        "model": _optimizer_snapshot(tensors),
        "optimizer_public": _optimizer_snapshot(state.optimizer.state_dict()),
        "optimizer_live": _live_optimizer(state),
        "scheduler": _optimizer_snapshot(state.scheduler.state_dict()),
    }


def _gradients(state: _State) -> dict[str, Any]:
    """Copy every finite local parameter gradient with unchanged DTensor metadata.

    Args:
        state: Model after actual CP/EP backward and recipe expert normalization.

    Returns:
        CPU local gradients keyed by FQN. Missing/nonfinite gradients fail;
        trainable FP32 sinks must have a nonzero adjoint on their local owner.
    """
    result = {}
    for name, parameter in state.model.named_parameters():
        gradient = parameter.grad
        assert gradient is not None, name
        local = gradient.to_local() if isinstance(gradient, DTensor) else gradient
        assert torch.isfinite(local).all(), name
        if "._fp32_params." in name:
            assert gradient.dtype == torch.float32
            assert local.numel() == 0 or torch.count_nonzero(local) > 0, name
        result[name] = _optimizer_snapshot(gradient)
    return result


def _nopp_step(state: _State, batch: dict[str, Any], *, update: bool = True) -> dict[str, Any]:
    """Run actual TE/DeepEP backward, recipe normalization and optional Adam update.

    Args:
        state: Public PP1/CP2/EP2 model and recipe-style mixed TE optimizer.
        batch: Global packed IDs/labels [2,128], dispatched once by the shared sharder.
        update: Advance optimizer/scheduler when True; False isolates repeat backward.

    Returns:
        Exact CPU global batch, local valid-query full-vocabulary logits [Tvalid,V],
        global loss, pre/post-clip local gradients, auxiliary norm and state-format counts.
        Physical padding never contributes to loss or logit acceptance.
    """
    state.optimizer.zero_grad(set_to_none=True)
    sharder = ContextParallelSharder(state.model, state.setup.mesh_context.device_mesh, batch, padding_token_id=30)
    context, local = sharder.shard(batch)
    labels = local.pop("labels")
    for key in ("seq_lens", "seq_lens_padded", "loss_mask"):
        local.pop(key, None)
    valid = ~local["padding_mask"].reshape(-1)
    total = (labels != -100).sum()
    dist.all_reduce(total)
    assert int(total) == 20
    prepare_for_grad_accumulation([state.model])
    prepare_for_final_backward([state.model])
    with context():
        logits = state.model(**local).logits
        assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum") / total
        (loss * 2).backward()
    options = dict(
        device_mesh=state.setup.mesh_context.device_mesh,
        moe_mesh=state.setup.mesh_context.moe_mesh,
        ep_axis_name="ep",
        foreach=True,
        grad_norm_backend="triton",
    )
    # Match the no-PP recipe: loss was normalized before backward, then expert
    # gradients receive the EP divisor exactly once before capture/clipping.
    scale_grads_and_clip_grad_norm(float("inf"), [state.model], dp_group_size=2, **options)
    gradients = _gradients(state)
    norm = clip_grad_norm(0.3, [state.model], **options)
    clipped = _gradients(state)
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    global_loss = loss.detach().clone()
    dist.all_reduce(global_loss)
    if update:
        state.optimizer.step()
        state.scheduler.step()
    result = {
        "batch": _optimizer_snapshot(batch),
        "logits": logits.detach().flatten(0, 1)[valid].cpu().clone(),
        "loss": float(global_loss),
        "gradients": gradients,
        "clipped_gradients": clipped,
        "norm": float(norm),
        "precision_counts": _optimizer_precision(state.optimizer),
    }
    state.optimizer.zero_grad(set_to_none=True)
    return result


def _differences(actual: Any, expected: Any, path: str = "state") -> list[dict[str, Any]]:
    """Describe exact byte/structure differences without changing acceptance precision.

    Args:
        actual: Independent CPU tensor/scalar/container snapshot.
        expected: Matching saved snapshot, including local DTensor layout metadata.
        path: Component path used in first-divergence diagnostics.

    Returns:
        Difference records. Floating tensors are compared by raw bytes, including
        signed zero; float64 diagnostic arithmetic never participates in acceptance.
    """
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return [
                {"path": path, "reason": "shape_or_dtype", "actual": str(actual.dtype), "expected": str(expected.dtype)}
            ]
        left, right = actual.contiguous(), expected.contiguous()
        if torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)):
            return []
        delta = left.double() - right.double()
        return [
            {
                "path": path,
                "reason": "tensor_bytes",
                "different_values": int(torch.count_nonzero(left != right)),
                "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
            }
        ]
    if type(actual) is not type(expected):
        return [{"path": path, "reason": "type", "actual": str(type(actual)), "expected": str(type(expected))}]
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return [{"path": path, "reason": "keys"}]
        return [item for key in actual for item in _differences(actual[key], expected[key], f"{path}.{key}")]
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            return [{"path": path, "reason": "length"}]
        return [
            item
            for index, (a, b) in enumerate(zip(actual, expected, strict=True))
            for item in _differences(a, b, f"{path}.{index}")
        ]
    return (
        []
        if actual == expected
        else [{"path": path, "reason": "value", "actual": repr(actual), "expected": repr(expected)}]
    )


def _require_exact(actual: Any, expected: Any, *, name: str, record: dict[str, Any], output: Path) -> None:
    """Persist all rank-local first divergences before a symmetric exact assertion.

    Args:
        actual: Actual local CPU state tree, with arbitrary tensor shapes.
        expected: Expected local snapshot with identical structural metadata.
        name: Named checkpoint/continuation phase.
        record: Mutable compact evidence record.
        output: Rank-local JSON path, written before the collective assertion.
    """
    differences = _differences(actual, expected, name)
    summary = {
        "bitwise_equal": not differences,
        "difference_count": len(differences),
        "first_differences": differences[:32],
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, summary)
    record.setdefault("comparisons", {})[name] = gathered
    output.write_text(json.dumps(record, indent=2) + "\n")
    assert all(item["bitwise_equal"] for item in gathered), {"component": name, "ranks": gathered}


def _checkpoint(state: _State, directory: str) -> Checkpointer:
    """Build native-safetensors model/DCP optimizer transport on the real EP mesh."""
    return CheckpointingConfig(
        checkpoint_dir=directory,
        model_save_format="safetensors",
        save_consolidated=False,
        is_async=False,
        v4_compatible=False,
    ).build(
        dp_rank=dist.get_rank(),
        tp_rank=0,
        pp_rank=0,
        moe_mesh=state.setup.mesh_context.moe_mesh,
        process_group=dist.group.WORLD,
    )


def _nopp_worker(rank: int, port: int, source: str, directory: str, activation_checkpointing: bool) -> None:
    """Require exact restore and two continued updates on a genuine no-PP topology."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(minutes=12))
    output = Path(directory, f"nopp_checkpoint_rank_{rank}.json")
    record: dict[str, Any] = {"status": "running", "rank": rank, "phase": "construction"}
    checkpointer = restored_checkpointer = None
    try:
        provenance = _runtime_receipt()
        root = Path(__file__).resolve().parents[4]
        for relative in (
            "components/checkpoint/checkpointing.py",
            "components/checkpoint/config.py",
            "components/optim/optimizer.py",
        ):
            path = root / "nemo_automodel" / relative
            provenance["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        record.update(
            provenance=provenance,
            test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            topology={"world": 2, "pp": 1, "cp": 2, "ep": 2, "dp": 1, "tp": 1},
            activation_checkpointing=activation_checkpointing,
            determinism_environment={
                key: os.environ.get(key) for key in ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "CUBLAS_WORKSPACE_CONFIG")
            },
            checkpoint={"model_save_format": "safetensors", "save_consolidated": False},
            norm_protocol={
                "atol": 2e-6,
                "rtol": 2e-5,
                "source": "tests/functional_tests/models/flint/test_checkpoint_resume.py",
                "scope": "auxiliary norm scalar only",
            },
        )
        rng = StatefulRNG(437529, ranked=True)
        state = _nopp_new_state(source, activation_checkpointing=activation_checkpointing)
        device = torch.device("cuda", rank)
        initial = _nopp_snapshot(state)["model"]
        for _ in range(2):
            _nopp_step(state, _batch(device))
        saved = _nopp_snapshot(state)
        expert_names = [name for name in saved["model"] if ".mlp.experts." in name]
        assert expert_names and any(_differences(saved["model"][name], initial[name]) for name in expert_names)

        record["phase"] = "duplicate_uninterrupted_backward"
        probe = _batch(device)
        probe_rng = _optimizer_snapshot(rng.state_dict())
        duplicate_a = _nopp_step(state, probe, update=False)
        rng.load_state_dict(probe_rng)
        duplicate_b = _nopp_step(state, probe, update=False)
        rng.load_state_dict(probe_rng)
        duplicate_diffs = _differences(
            {k: v for k, v in duplicate_b.items() if k != "norm"},
            {k: v for k, v in duplicate_a.items() if k != "norm"},
            "duplicate_backward",
        )
        record["duplicate_uninterrupted_backward"] = {
            "bitwise_equal": not duplicate_diffs,
            "difference_count": len(duplicate_diffs),
            "first_differences": duplicate_diffs[:32],
            "norms": [duplicate_a["norm"], duplicate_b["norm"]],
            "scope": "diagnostic; does not substitute for exact checkpoint trajectory assertions",
        }
        output.write_text(json.dumps(record, indent=2) + "\n")
        del duplicate_a, duplicate_b

        record["phase"] = "save"
        checkpointer = _checkpoint(state, directory)
        path = str(Path(directory, "step_2"))
        checkpointer.save_model(state.model, path)
        checkpointer.save_optimizer(state.optimizer, state.model, path, state.scheduler)
        checkpointer.save_on_global_ranks(rng, "rng", path)
        checkpointer.async_wait()
        saved_rng = _optimizer_snapshot(rng.state_dict())
        saved = _nopp_snapshot(state)
        dist.barrier()
        continuous = []
        for _ in range(2):
            step = _nopp_step(state, _batch(device))
            step["state"] = _nopp_snapshot(state)
            step["rng"] = _optimizer_snapshot(rng.state_dict())
            continuous.append(step)

        record["phase"] = "fresh_public_rebuild_and_restore"
        restored = _nopp_new_state(source, activation_checkpointing=activation_checkpointing)
        restored_checkpointer = _checkpoint(restored, directory)
        restored_checkpointer.load_model(restored.model, str(Path(path, "model")))
        restored_checkpointer.load_optimizer(restored.optimizer, restored.model, path, restored.scheduler)
        restored_checkpointer.load_on_global_ranks(rng, "rng", path)
        _require_exact(_nopp_snapshot(restored), saved, name="restore_boundary", record=record, output=output)
        _require_exact(
            _optimizer_snapshot(rng.state_dict()), saved_rng, name="restore_rng", record=record, output=output
        )
        norm_pairs = []
        for index, expected in enumerate(continuous):
            record["phase"] = f"continued_update_{index + 1}"
            actual = _nopp_step(restored, _batch(device))
            actual["state"] = _nopp_snapshot(restored)
            actual["rng"] = _optimizer_snapshot(rng.state_dict())
            norm_pairs.append([expected["norm"], actual["norm"]])
            record["norm_pairs"] = norm_pairs
            # Compare the pre-update objective/adjoints before post-update state,
            # preserving the first causal divergence in numerical diagnostics.
            for component in (
                "batch",
                "logits",
                "loss",
                "gradients",
                "clipped_gradients",
                "state",
                "rng",
                "precision_counts",
            ):
                _require_exact(
                    actual[component],
                    expected[component],
                    name=f"step_{index + 1}.{component}",
                    record=record,
                    output=output,
                )
        norms = torch.tensor(norm_pairs, dtype=torch.float64)
        record.update(
            strict_bitwise_norm_equal=torch.equal(norms[:, 0], norms[:, 1]),
            norm_max_abs=float((norms[:, 0] - norms[:, 1]).abs().max()),
        )
        output.write_text(json.dumps(record, indent=2) + "\n")
        torch.testing.assert_close(norms[:, 0], norms[:, 1], atol=2e-6, rtol=2e-5)
        assert _runtime_receipt()["source_sha256"].items() <= provenance["source_sha256"].items()
        record.update(status="passed", phase="complete", saved_after_updates=2, continued_updates=2)
        output.write_text(json.dumps(record, indent=2) + "\n")
        dist.barrier()
        if rank == 0:
            Path(directory, "nopp_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
    except BaseException:
        record.update(status="failed", traceback=traceback.format_exc())
        output.write_text(json.dumps(record, indent=2) + "\n")
        raise
    finally:
        for value in (checkpointer, restored_checkpointer):
            if value is not None:
                value.close()
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs and actual TE/DeepEP")
@pytest.mark.parametrize("activation_checkpointing", [False, True], ids=["plain", "activation_checkpointed"])
def test_two_gpu_nopp_checkpoint_continuation(tmp_path: Path, activation_checkpointing: bool) -> None:
    """Require exact actual CP2/EP2 checkpoint continuation with no pipeline axis."""
    destination = Path(os.environ.get("FLINT_NOPP_CHECKPOINT_EVIDENCE", str(tmp_path)))
    directory = destination / ("activation_checkpointed" if activation_checkpointing else "plain")
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "source"
    source.mkdir()
    torch.manual_seed(437529)
    # Retain the already-proven grouped-GEMM/DeepEP aligned synthetic widths.
    config = replace(
        te_config(),
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=192,
    )
    model = config.build(backend=attention_backend("eager"))
    config.save_pretrained(source)
    save_file(model.state_dict_adapter.to_hf(model.state_dict()), source / "model.safetensors")
    del model
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_nopp_worker, args=(port, str(source), str(directory), activation_checkpointing), nprocs=2, join=True)
    assert json.loads(Path(directory, "nopp_checkpoint.json").read_text())["status"] == "passed"


@dataclass
class _TrainingState:
    """Actual local stages, per-stage optimizers and per-step observations."""

    model: AutoPipeline
    setup: DistributedSetup
    optimizers: list[torch.optim.Optimizer]
    schedulers: list[torch.optim.lr_scheduler.StepLR]
    losses: list[torch.Tensor]
    logits: list[torch.Tensor]
    observations: list[dict[str, Any]]
    handles: list[Any]

    @property
    def stage_ids(self) -> list[int]:
        """Return global stage indices aligned with local parts/optimizers."""
        return [stage.stage_index for stage in self.model.info.stages]


def _composed_new_state(source: str) -> _TrainingState:
    """Build the long recipe's public topology with small aligned dimensions.

    Args:
        source: Test-owned local HF checkpoint containing four tiny layers.

    Returns:
        BF16 resident bulk, FP32 head/sinks/residuals, actual TE attention and
        TorchMM/DeepEP experts on PP2/CP2/EP2 with four virtual stages. One real
        TE FusedAdam and scheduler are bound to each local model part.
    """
    losses, logits = [], []

    def loss_fn(prediction: Any, target: torch.Tensor) -> torch.Tensor:
        """Return raw CE sum for [1,Tlocal,V]; shared PP normalization follows."""
        value = prediction.logits if hasattr(prediction, "logits") else prediction
        assert value.dtype == torch.float32 and torch.isfinite(value).all()
        logits.append(value.detach().cpu().clone())
        loss = F.cross_entropy(value.flatten(0, 1), target.flatten(), reduction="sum")
        losses.append(loss.detach())
        return loss

    setup = DistributedSetup.build(
        strategy=FSDP2Config(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=False,
            ),
            enable_fsdp2_prefetch=False,
            defer_fsdp_grad_sync=False,
        ),
        parallelism_sizes=ParallelismSizes(dp_size=1, pp_size=2, cp_size=2, ep_size=2),
        pipeline_config=PipelineConfig(
            pp_schedule="interleaved1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            layers_per_stage=1,
            scale_grads_in_schedule=False,
            patch_inner_model=False,
            patch_causal_lm_model=False,
            loss_fn=loss_fn,
        ),
        moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=True),
        activation_checkpointing=False,
        world_size=4,
    )
    model = NeMoAutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        backend=replace(attention_backend("te"), experts="torch_mm", dispatcher="deepep"),
        distributed_setup=setup,
        attn_implementation="eager",
        use_liger_kernel=False,
        use_sdpa_patching=False,
        local_files_only=True,
    )
    assert isinstance(model, AutoPipeline) and len(model.parts) == 2
    for part in model.parts:
        part.train()
        for name, parameter in part.named_parameters():
            protected = "._fp32_params." in name or name == "lm_head.weight"
            assert parameter.dtype == (torch.float32 if protected else torch.bfloat16), name
    optimizers = build_optimizer(
        model,
        (
            "transformer_engine.pytorch.optimizers.FusedAdam",
            dict(
                lr=0.003,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
                adam_w_mode=True,
                master_weights=True,
                master_weight_dtype="float32",
                store_param_remainders=True,
                exp_avg_dtype="bfloat16",
                exp_avg_sq_dtype="bfloat16",
            ),
        ),
        device_mesh=setup.mesh_context.device_mesh,
    )
    assert len(optimizers) == len(model.parts)
    observations: list[dict[str, Any]] = []
    return _TrainingState(
        model,
        setup,
        optimizers,
        [torch.optim.lr_scheduler.StepLR(item, step_size=2, gamma=0.7) for item in optimizers],
        losses,
        logits,
        observations,
        _observe_parts(model.parts, observations),
    )


def _composed_batch(device: torch.device) -> dict[str, torch.Tensor | str]:
    """Consume all rank-local RNG families and broadcast two physical packs.

    Args:
        device: This process's CUDA device.

    Returns:
        IDs/once-shifted labels/positions [2,128], real/padded lengths [2,2]
        with real lengths 5/7 and physical document slots 64/64. Every PP/CP/EP
        rank receives identical tokens; valid EOS=2 is independent of padding=30.
    """
    batch = _packed(device, flatten=False)
    batch = {name: value[:2].clone() if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
    offset = random.randrange(29) + int(np.random.randint(29)) + int(torch.randint(29, ()).item())
    for row in range(2):
        for start, length in ((0, 5), (64, 7)):
            values = (torch.randint(29, (length,), device=device) + offset) % 29
            values[1] = 2
            dist.broadcast(values, src=0)
            batch["input_ids"][row, start : start + length] = values
            batch["labels"][row, start : start + length - 1] = values[1:]
    assert int((batch["labels"] != -100).sum()) == 20
    return batch


def _composed_step(state: _TrainingState, device: torch.device) -> dict[str, Any]:
    """Run one actual packed pipeline update with shared PP/EP normalization.

    Args:
        state: Live composed stages and their optimizers/schedulers.
        device: This process's CUDA device.

    Returns:
        Independent CPU copies of input/target tensors and local complete
        microbatch logits, global scalar CE/norm and validated state-format counts.
    """
    batch = _composed_batch(device)
    state.losses.clear()
    state.logits.clear()
    for optimizer in state.optimizers:
        optimizer.zero_grad(set_to_none=True)
    sharder = ContextParallelSharder(
        state.model.parts[0],
        state.setup.mesh_context.device_mesh,
        batch,
        num_chunks=2,
        padding_token_id=30,
    )
    _, local = sharder.shard(batch)
    labels = local.pop("labels")
    for name in ("seq_lens", "seq_lens_padded", "loss_mask"):
        local.pop(name, None)
    prepare_for_grad_accumulation(state.model.parts, pp_enabled=True)
    prepare_for_final_backward(state.model.parts, pp_enabled=True)
    state.model.update_seq_len(128)
    state.model.step(
        local.pop("input_ids"),
        target=labels if state.model.info.has_last_stage else None,
        losses=[] if state.model.info.has_last_stage else None,
        **local,
    )
    gradient_count = 0
    for part in state.model.parts:
        for name, parameter in part.named_parameters():
            assert parameter.grad is not None, name
            gradient = parameter.grad
            gradient = gradient.to_local() if isinstance(gradient, DTensor) else gradient
            assert torch.isfinite(gradient).all(), name
            if "._fp32_params." in name or name == "lm_head.weight":
                assert gradient.dtype == torch.float32, name
            if name.endswith("softmax_offset"):
                assert torch.count_nonzero(gradient) > 0, name
            gradient_count += 1
    context = state.setup.mesh_context
    norm = scale_grads_and_clip_grad_norm(
        0.3,
        state.model.parts,
        pp_enabled=True,
        device_mesh=context.device_mesh,
        moe_mesh=context.moe_mesh,
        pp_axis_name="pp",
        ep_axis_name="ep",
        foreach=True,
        num_label_tokens=20,
        dp_group_size=2,
        grad_norm_backend="triton",
    )
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    assert torch.isfinite(norm) and float(norm) > 0
    loss = sum(state.losses, torch.zeros((), device=device)) / 20
    dist.all_reduce(loss)
    assert torch.isfinite(loss) and float(loss) > 0
    counts = []
    for optimizer, scheduler in zip(state.optimizers, state.schedulers, strict=True):
        optimizer.step()
        scheduler.step()
        counts.append(_optimizer_precision(optimizer))
        optimizer.zero_grad(set_to_none=True)
    return {
        "batch": _snapshot(batch),
        "logits": _snapshot(state.logits),
        "loss": float(loss),
        "norm": float(norm),
        "precision_counts": counts,
        "local_parameter_gradient_count": gradient_count,
    }


def _model_snapshot(value: Any) -> Any:
    """Copy local tensor shards and the complete immutable TE extra-state bytes.

    Args:
        value: Model state tree whose TE extra state can contain BytesIO.

    Returns:
        Independent CPU tensor copies with DTensor shape/placement metadata.
        BytesIO buffers become byte contents plus cursor positions, avoiding
        identity comparisons or live mutable references across continuation.
    """
    if isinstance(value, io.BytesIO):
        return {"bytesio_contents": value.getvalue(), "position": value.tell()}
    if isinstance(value, dict):
        return {key: _model_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_model_snapshot(item) for item in value)
    return _optimizer_snapshot(value)


def _state_snapshot(state: _TrainingState) -> dict[str, Any]:
    """Copy every local model/optimizer byte without int16 NCCL collectives.

    Args:
        state: Composed runtime with native arbitrary-shape DTensor shards.

    Returns:
        State trees keyed by global stage index. DTensor local bytes, global
        shapes, mesh names and placements are retained without casts/gathers.
        Public TE serialization promotes moments to FP32 transport. Raw live
        moment representations are measured separately because loading rebuilds
        their scaled storage through the installed optimizer's public contract.
    """
    return {
        str(stage): {
            # Safetensors deliberately excludes TE module metadata, matching
            # Checkpointer.save_model's public non-FP8 weight contract.
            "model": _model_snapshot(
                {name: value for name, value in part.state_dict().items() if not name.endswith("_extra_state")}
            ),
            "optimizer": _optimizer_snapshot(optimizer.state_dict()),
            "scheduler": _snapshot(scheduler.state_dict()),
        }
        for stage, part, optimizer, scheduler in zip(
            state.stage_ids,
            state.model.parts,
            state.optimizers,
            state.schedulers,
            strict=True,
        )
    }


def _live_optimizer_snapshot(state: _TrainingState) -> dict[str, Any]:
    """Measure raw scaled storage in stable stage/group/parameter order.

    Args:
        state: Real per-stage TE optimizers with arbitrary local DTensor shapes.

    Returns:
        Exact local BF16 moment/remainder/scaling-state bytes as a diagnostic.
        Public serialized values and subsequent trajectory remain the strict
        gate; checkpoint loading is allowed to reconstruct scaled representation.
    """
    return {
        str(stage): [
            [
                {
                    "parameter_dtype": str(parameter.dtype),
                    "parameter_global_shape": tuple(parameter.shape),
                    "state": _optimizer_snapshot(optimizer.state.get(parameter, {})),
                }
                for parameter in group["params"]
            ]
            for group in optimizer.param_groups
        ]
        for stage, optimizer in zip(state.stage_ids, state.optimizers, strict=True)
    }


def _extra_state_snapshot(state: _TrainingState) -> dict[str, Any]:
    """Require inactive non-FP8 TE metadata before safetensors omits it.

    Installed TE returns None or an empty CPU uint8 tensor when FP8 checkpoint,
    compute and calibration are disabled. Reject active/nonempty metadata: this
    test does not certify FP8 extra-state restoration through a weights format.
    """
    snapshots = {}
    for stage, part in zip(state.stage_ids, state.model.parts, strict=True):
        extra = {name: value for name, value in part.state_dict().items() if name.endswith("_extra_state")}
        for name, value in extra.items():
            assert value is None or (
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.device.type == "cpu"
                and value.numel() == 0
            ), f"Active/unsupported TE extra state cannot be omitted by this safetensors test: {name}"
        snapshots[str(stage)] = _model_snapshot(extra)
    return snapshots


def _raw_storage_diagnostic(actual: Any, expected: Any) -> dict[str, Any]:
    """Report bit equality without imposing an extra storage canonicalization gate."""
    try:
        _compare(actual, expected, atol=0, rtol=0)
    except AssertionError as error:
        return {"bitwise_equal": False, "first_difference": str(error)}
    return {"bitwise_equal": True, "first_difference": None}


def _composed_checkpoint(state: _TrainingState, directory: str) -> Checkpointer:
    """Build the real recipe Checkpointer with PP and MoE mesh ownership."""
    mesh = state.setup.mesh_context.device_mesh
    return CheckpointingConfig(
        checkpoint_dir=directory,
        model_save_format="safetensors",
        save_consolidated=False,
        is_async=False,
        v4_compatible=False,
    ).build(
        dp_rank=0,
        tp_rank=0,
        pp_rank=mesh["pp"].get_local_rank(),
        moe_mesh=state.setup.mesh_context.moe_mesh,
        pp_group=mesh["pp"].get_group(),
        process_group=dist.group.WORLD,
    )


def _composed_worker(rank: int, port: int, source: str, directory: str) -> None:
    """Save after two updates, rebuild publicly, restore and continue twice."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE="4", LOCAL_RANK=str(rank)
    )
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=4, timeout=timedelta(minutes=12))
    checkpoint = restored_checkpoint = None
    record: dict[str, Any] = {"status": "running", "rank": rank, "phase": "construction"}
    output = Path(directory, f"composed_checkpoint_rank_{rank}.json")
    try:
        provenance = _runtime_receipt()
        root = Path(__file__).resolve().parents[4]
        for relative in (
            "components/checkpoint/checkpointing.py",
            "components/checkpoint/config.py",
            "components/optim/optimizer.py",
        ):
            path = root / "nemo_automodel" / relative
            provenance["source_sha256"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance["test_dependency_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "run_parallel_composition.py",
                "test_attention.py",
                "test_parallel_training.py",
                "reference.py",
                "sdpa_reference.py",
            )
        }
        provenance["source_fixture_sha256"] = {
            name: hashlib.sha256((Path(source) / name).read_bytes()).hexdigest()
            for name in ("config.json", "model.safetensors")
        }
        record.update(
            provenance=provenance,
            test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            topology={"world": 4, "dp": 1, "pp": 2, "cp": 2, "ep": 2, "virtual_stages": 4},
            scope="tiny same-topology native checkpoint continuation; not released-model composition parity",
            checkpoint_contract={
                "model_save_format": "safetensors",
                "save_consolidated": False,
                "model": "every serialized parameter/buffer exact; TE _extra_state is omitted by this format",
                "te_extra_state": "separate diagnostic; no FP8 policy is enabled",
            },
            optimizer_state_contract={
                "public_state_dict": "TE FP32 moment transport plus exact int16 master remainders",
                "live_state": "actual BF16 moment storage and int16 remainders, FP32 protected parameters",
                "comparison": "public serialized state and continuation bitwise; raw scaled-storage equality diagnostic",
            },
        )
        rng = StatefulRNG(437529, ranked=True)
        state = _composed_new_state(source)
        record["optimizer_part_ids"] = state.stage_ids
        assert state.stage_ids in ([0, 2], [1, 3]), state.stage_ids
        initial_bulk = [
            (parameter, (parameter.to_local() if isinstance(parameter, DTensor) else parameter).detach().cpu().clone())
            for part in state.model.parts
            for parameter in part.parameters()
            if parameter.dtype == torch.bfloat16
        ]
        for _ in range(2):
            _composed_step(state, torch.device("cuda", rank))
        changed_bulk = sum(
            not torch.equal(
                (parameter.to_local() if isinstance(parameter, DTensor) else parameter).detach().cpu(), initial
            )
            for parameter, initial in initial_bulk
        )
        assert changed_bulk > 0, "BF16 resident optimizer updates must be observable"
        record["locally_changed_bf16_parameter_tensors_after_two_updates"] = changed_bulk
        del initial_bulk
        path = str(Path(directory, "step_2"))
        record["phase"] = "save"
        checkpoint = _composed_checkpoint(state, directory)
        checkpoint.save_model(state.model.parts, path)
        checkpoint.save_optimizer(
            state.optimizers, state.model.parts, path, state.schedulers, optimizer_part_ids=state.stage_ids
        )
        checkpoint.save_on_global_ranks(rng, "rng", path)
        checkpoint.async_wait()
        saved = _state_snapshot(state)
        saved_live = _live_optimizer_snapshot(state)
        saved_extra = _extra_state_snapshot(state)
        saved_rng = _snapshot(rng.state_dict())
        dist.barrier()
        record["phase"] = "continuous_updates"
        continuous = [_composed_step(state, torch.device("cuda", rank)) for _ in range(2)]
        final, final_rng = _state_snapshot(state), _snapshot(rng.state_dict())
        final_live = _live_optimizer_snapshot(state)
        final_extra = _extra_state_snapshot(state)
        record["phase"] = "rebuild_and_restore"
        restored = _composed_new_state(source)
        restored_checkpoint = _composed_checkpoint(restored, directory)
        restored_checkpoint.load_model(restored.model.parts, str(Path(path, "model")))
        restored_checkpoint.load_optimizer(
            restored.optimizers, restored.model.parts, path, restored.schedulers, optimizer_part_ids=restored.stage_ids
        )
        restored_checkpoint.load_on_global_ranks(rng, "rng", path)
        for optimizer in restored.optimizers:
            _optimizer_precision(optimizer)
        boundary_error = _compare(_state_snapshot(restored), saved, atol=0, rtol=0)
        boundary_live = _raw_storage_diagnostic(_live_optimizer_snapshot(restored), saved_live)
        boundary_extra = _raw_storage_diagnostic(_extra_state_snapshot(restored), saved_extra)
        _compare(_snapshot(rng.state_dict()), saved_rng, atol=0, rtol=0)
        record["phase"] = "resumed_updates"
        resumed = [_composed_step(restored, torch.device("cuda", rank)) for _ in range(2)]
        final_error = _compare(_state_snapshot(restored), final, atol=0, rtol=0)
        final_live_diagnostic = _raw_storage_diagnostic(_live_optimizer_snapshot(restored), final_live)
        final_extra_diagnostic = _raw_storage_diagnostic(_extra_state_snapshot(restored), final_extra)
        _compare(_snapshot(rng.state_dict()), final_rng, atol=0, rtol=0)
        for actual, expected in zip(resumed, continuous, strict=True):
            for key in ("batch", "logits", "loss", "precision_counts", "local_parameter_gradient_count"):
                _compare(actual[key], expected[key], atol=0, rtol=0, path=key)
        norms = torch.tensor([step["norm"] for step in continuous], dtype=torch.float64)
        resumed_norms = torch.tensor([step["norm"] for step in resumed], dtype=torch.float64)
        record.update(
            status="measured_before_norm_check",
            phase="norm_check",
            saved_after_updates=2,
            continuation_updates=2,
            restored_boundary_max_abs=boundary_error,
            final_all_state_max_abs=final_error,
            raw_scaled_optimizer_storage_diagnostic={"boundary": boundary_live, "final": final_live_diagnostic},
            omitted_te_extra_state_diagnostic={"boundary": boundary_extra, "final": final_extra_diagnostic},
            model_optimizer_scheduler_rng_batches_logits_losses_exact=True,
            losses=[step["loss"] for step in continuous],
            resumed_losses=[step["loss"] for step in resumed],
            norms=norms.tolist(),
            resumed_norms=resumed_norms.tolist(),
            strict_bitwise_norm_equal=torch.equal(norms, resumed_norms),
            norm_max_abs=float((norms - resumed_norms).abs().max()),
            norm_protocol={
                "atol": 2e-6,
                "rtol": 2e-5,
                "source": "tests/functional_tests/models/flint/test_checkpoint_resume.py",
                "scope": "auxiliary reduced norms only; every other comparison is bitwise",
            },
            precision_counts=resumed[0]["precision_counts"],
            local_parameter_gradient_count=resumed[0]["local_parameter_gradient_count"],
            observed_dtypes=restored.observations,
        )
        output.write_text(json.dumps(record, indent=2) + "\n")
        torch.testing.assert_close(resumed_norms, norms, atol=2e-6, rtol=2e-5)
        for relative, digest in provenance["source_sha256"].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
        for name, digest in provenance["test_dependency_sha256"].items():
            assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == record["test_sha256"]
        for name, digest in provenance["source_fixture_sha256"].items():
            assert hashlib.sha256((Path(source) / name).read_bytes()).hexdigest() == digest, name
        record.update(status="rank_passed", phase="complete")
        output.write_text(json.dumps(record, indent=2) + "\n")
        dist.barrier()
        if rank == 0:
            record["status"] = "passed"
            Path(directory, "composed_checkpoint.json").write_text(json.dumps(record, indent=2) + "\n")
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        output.write_text(json.dumps(record, indent=2) + "\n")
        raise
    finally:
        for item in (checkpoint, restored_checkpoint):
            if item is not None:
                item.close()
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("FLINT_RUN_COMPOSED_CHECKPOINT") != "1" or torch.cuda.device_count() < 4,
    reason="scheduled four-GPU test; set FLINT_RUN_COMPOSED_CHECKPOINT=1",
)
def test_four_rank_composed_mixed_optimizer_checkpoint(tmp_path: Path) -> None:
    """Require genuine composed Checkpointer continuation on four reserved GPUs."""
    directory = Path(os.environ.get("FLINT_COMPOSED_CHECKPOINT_EVIDENCE", str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "source"
    source.mkdir()
    torch.manual_seed(437529)
    config = FlintConfig(
        **{
            **_config().to_dict(),
            "dtype": "bfloat16",
            "hidden_size": 128,
            "intermediate_size": 256,
            "moe_intermediate_size": 128,
            "shared_expert_intermediate_size": 192,
            "residual_dtype": "fp32",
            "attention_compute_dtype": None,
        }
    )
    model = config.build(backend=attention_backend("eager"))
    config.save_pretrained(source)
    save_file(model.state_dict_adapter.to_hf(model.state_dict()), source / "model.safetensors")
    del model
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_composed_worker, args=(port, str(source), str(directory)), nprocs=4, join=True)
    assert json.loads((directory / "composed_checkpoint.json").read_text())["status"] == "passed"

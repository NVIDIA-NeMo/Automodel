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

"""Real dense Qwen3.5 mixed-dtype parity, checkpoint, and composed-mesh tests.

The small VLM retains alternating GatedDeltaNet/full-attention layers and a
frozen vision tower. Synthetic text exercises the language side of the VLM;
vision execution and the full-size recipe's peak memory are not measured here.
The reference bypasses FSDP and uses differentiable transient BF16 parameter
copies, retaining the original resident weights for gradients/optimizer state.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.func import functional_call
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.context_parallel.utils import attach_context_parallel_hooks
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.distributed.tp_replicas import synchronize_tp_replica_gradients
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration
from nemo_automodel.components.optim.optimizer import AdamWConfig, FusedAdamConfig
from nemo_automodel.components.training.utils import clip_grad_norm, scale_grads_and_clip_grad_norm
from nemo_automodel.recipes.kd_utils import (
    RUN_TEACHER,
    KDMeshBridge,
    create_kd_distributed_setups,
    materialize_teacher_logits,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

VOCAB = 256
SEQ = 32
MICROBATCHES = 2
MAX_NORM = 0.1


def _model(device: torch.device, dtype: torch.dtype, seed: int = 2026) -> torch.nn.Module:
    torch.manual_seed(seed)
    text = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=128,
        layer_types=["linear_attention", "full_attention"] * 2,
        tie_word_embeddings=False,
        use_cache=False,
        dtype=dtype,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=64,
    )
    config = Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=252,
        video_token_id=253,
        vision_start_token_id=254,
        vision_end_token_id=255,
        architectures=["Qwen3_5ForConditionalGeneration"],
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForConditionalGeneration(
        config,
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False),
    ).to(device)
    model.initialize_weights(buffer_device=device, dtype=dtype)
    model.model.visual.requires_grad_(False)
    return model.train()


def _forward_reference(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    params = {
        name: parameter if "_fp32_params" in name else parameter.to(torch.bfloat16)
        for name, parameter in model.named_parameters()
    }
    return functional_call(model, params, (), batch).logits.float()


def _batch(dp_rank: int, step: int, device: torch.device, *, packed: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(7000 + 100 * step + dp_rank)
    ids = torch.randint(1, 240, (MICROBATCHES, SEQ), generator=generator, device=device)
    labels = torch.roll(ids, -1, dims=1)
    labels[:, -1] = -100
    batch = {"input_ids": ids, "labels": labels}
    if packed:
        # Unequal documents, with a boundary crossing the middle of the sequence.
        batch["attention_mask"] = torch.tensor([1] * 11 + [2] * 21, device=device).expand_as(ids).clone()
        batch["position_ids"] = torch.cat((torch.arange(11), torch.arange(21))).to(device).expand_as(ids).clone()
        batch["labels"][:, 10] = -100
    return batch


def _named(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {canonical_parameter_fqn(name): parameter for name, parameter in model.named_parameters()}


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float()


def _assert_gradients(parts: list[torch.nn.Module], reference: torch.nn.Module) -> None:
    expected = _named(reference)
    used = 0
    gates = 0
    for part in parts:
        for name, parameter in _named(part).items():
            if not parameter.requires_grad:
                continue
            ref_grad = expected[name].grad
            if ref_grad is None:
                assert parameter.grad is None, f"unused parameter acquired a gradient: {name}"
                continue
            assert parameter.grad is not None, f"missing gradient: {name}"
            actual = _full(parameter.grad)
            want = ref_grad.detach().float()
            # Compare relative tensor error as well as elementwise tolerance;
            # near-zero BF16 entries should not dominate the relative metric.
            torch.testing.assert_close(actual, want, rtol=0.08, atol=2e-3, msg=lambda msg: f"{name}: {msg}")
            relative_error = torch.linalg.vector_norm(actual - want) / torch.linalg.vector_norm(want).clamp_min(1e-8)
            assert relative_error < 0.05, f"{name}: relative gradient error {relative_error.item()}"
            used += 1
            gates += "_fp32_params" in name
    assert used > 0 and gates > 0, "parity test must exercise trainable FP32 gates and bulk parameters"


def _assert_weights(parts: list[torch.nn.Module], reference: torch.nn.Module) -> None:
    expected = _named(reference)
    for part in parts:
        for name, parameter in _named(part).items():
            if parameter.requires_grad:
                # BF16 moments/updates can cross a quantization boundary for a
                # near-zero gradient. FP32 gate updates use a tighter bound.
                atol = 2e-5 if "_fp32_params" in name else (5e-4 if parameter.dtype == torch.bfloat16 else 3e-4)
                torch.testing.assert_close(
                    _full(parameter),
                    expected[name].detach().float(),
                    rtol=0,
                    atol=atol,
                    msg=lambda msg: f"{name}: {msg}",
                )


def _parallelize(model: torch.nn.Module, mesh: MeshContext) -> torch.nn.Module:
    model = parallelize_model(model, mesh)
    if mesh.dp_size > 1:
        assert isinstance(model, FSDPModule), "the test must apply production FSDP wrapping"
    if mesh.tp_size > 1:
        assert any(
            isinstance(parameter, DTensor)
            and "tp" in parameter.device_mesh.mesh_dim_names
            and parameter.placements[parameter.device_mesh.mesh_dim_names.index("tp")].is_shard()
            for parameter in model.parameters()
        ), "the test must actually shard parameters across TP"
    if mesh.cp_size > 1:
        model.cp_mesh = mesh.device_mesh["cp"]
        attach_context_parallel_hooks(model)
    return model


def _optimizer(model: torch.nn.Module, te: bool):
    config = FusedAdamConfig(lr=1e-4, master_weights=True) if te else AdamWConfig(lr=1e-4)
    return config.build(model)[0]


def _assert_te_master_ownership(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        state = optimizer.state[parameter]
        if parameter.dtype == torch.bfloat16:
            assert "master_param" in state, f"missing BF16 master: {name}"
            assert state["master_param"].dtype == torch.float32, f"master is not FP32: {name}"
        else:
            assert parameter.dtype == torch.float32, f"unexpected resident dtype: {name}"
            assert "master_param" not in state, f"redundant FP32 master: {name}"


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")


def _backward(model: torch.nn.Module, mesh: MeshContext, step: int, *, packed: bool = False) -> torch.Tensor:
    batch = _batch(mesh.device_mesh["dp"].get_local_rank(), step, next(model.parameters()).device, packed=packed)
    local_loss = torch.zeros((), device=batch["input_ids"].device)
    token_count = batch["labels"].ne(-100).sum()
    for microbatch in range(MICROBATCHES):
        sync = microbatch == MICROBATCHES - 1
        if isinstance(model, FSDPModule):
            model.set_requires_gradient_sync(sync)
        inputs = {name: value[microbatch : microbatch + 1].clone() for name, value in batch.items()}
        sharder = ContextParallelSharder(model, mesh.device_mesh, inputs)
        train_ctx, inputs = sharder.shard(inputs)
        labels = inputs.pop("labels")
        with train_ctx():
            logits = model(**inputs).logits
            loss = _loss(logits, labels) * mesh.cp_size / token_count
            loss.backward()
        local_loss += loss.detach()
    synchronize_tp_replica_gradients([model], mesh.device_mesh)
    group = mesh.device_mesh["dp_cp"].get_group()
    dist.all_reduce(local_loss, group=group)
    return local_loss / dist.get_world_size(group)


def _reference_backward(
    reference: torch.nn.Module,
    dp_size: int,
    step: int,
    *,
    packed: bool = False,
    normalize_after_backward: bool = False,
) -> torch.Tensor:
    loss = torch.zeros((), device=next(reference.parameters()).device)
    for dp_rank in range(dp_size):
        batch = _batch(dp_rank, step, loss.device, packed=packed)
        labels = batch.pop("labels")
        # Match the microbatch shape so BF16 kernel selection/rounding does not
        # become an unrelated batch-size comparison. No distributed hooks run.
        for microbatch in range(MICROBATCHES):
            inputs = {name: tensor[microbatch : microbatch + 1] for name, tensor in batch.items()}
            value = _loss(_forward_reference(reference, inputs), labels[microbatch : microbatch + 1])
            normalizer = labels.ne(-100).sum() * dp_size
            if not normalize_after_backward:
                value = value / normalizer
            value.backward()
            loss += value.detach()
    if normalize_after_backward:
        # The PP recipe normalizes accumulated gradients after backward.
        # Moving this into the loss changes BF16 gradient rounding.
        for parameter in reference.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(normalizer)
        loss /= normalizer
    return loss


def _run_dense(mesh: MeshContext, dtype: torch.dtype, *, packed: bool, te: bool, resume: bool) -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    reference = _model(device, dtype)
    model = _parallelize(copy.deepcopy(reference), mesh)
    optimizer = _optimizer(model, te)
    reference_optimizer = _optimizer(reference, te)
    holders = [p for name, p in model.named_parameters() if "_fp32_params" in name]
    assert holders and all(p.dtype is torch.float32 for p in holders)
    for parameter in holders:
        if isinstance(parameter, DTensor):
            assert parameter.device_mesh.mesh_dim_names == ("tp",), "FP32 gates must remain outside DP sharding"
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        actual_loss = _backward(model, mesh, step, packed=packed)
        expected_loss = _reference_backward(reference, mesh.dp_size, step, packed=packed)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0.01, atol=0.01)
        _assert_gradients([model], reference)
        norm = clip_grad_norm(MAX_NORM, [model], device_mesh=mesh.device_mesh)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), MAX_NORM)
        torch.testing.assert_close(norm.float(), expected_norm.float(), rtol=0.05, atol=1e-3)
        _assert_gradients([model], reference)
        optimizer.step()
        reference_optimizer.step()
        if te:
            _assert_te_master_ownership(model, optimizer)
        _assert_weights([model], reference)
        if step == 0 and resume:
            model, optimizer = _resume(model, optimizer, mesh, dtype, te)
    print(
        json.dumps(
            {
                "case": "dense",
                "rank": dist.get_rank(),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                "resident_dtype": str(dtype),
                "te": te,
                "resume_next_step_parity": resume,
            }
        ),
        flush=True,
    )


def _resume(model, optimizer, mesh: MeshContext, dtype: torch.dtype, te: bool):
    directory = [tempfile.mkdtemp(prefix="qwen35-ownership-") if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(directory, src=0)
    checkpoint = Checkpointer(
        CheckpointingConfig(checkpoint_dir=directory[0], model_save_format="torch_save", is_async=False),
        dp_rank=mesh.device_mesh["dp"].get_local_rank(),
        tp_rank=mesh.device_mesh["tp"].get_local_rank(),
        pp_rank=0,
    )
    try:
        checkpoint.save_model(model, directory[0])
        checkpoint.save_optimizer(optimizer, model, directory[0])
        # Different initialization prevents a no-op load from passing.
        restored = _parallelize(_model(next(model.parameters()).device, dtype, seed=8080), mesh)
        restored_optimizer = _optimizer(restored, te)
        checkpoint.load_model(restored, str(Path(directory[0]) / "model"))
        checkpoint.load_optimizer(restored_optimizer, restored, directory[0])
        if te:
            _assert_te_master_ownership(restored, restored_optimizer)
        current = _named(model)
        for name, parameter in _named(restored).items():
            torch.testing.assert_close(_full(parameter), _full(current[name]), rtol=0, atol=0)
            if parameter.requires_grad:
                source_state = optimizer.state[current[name]]
                target_state = restored_optimizer.state[parameter]
                assert source_state.keys() == target_state.keys(), f"optimizer state keys changed: {name}"
                for key, value in source_state.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(_full(target_state[key]), _full(value), rtol=0, atol=0)
                    else:
                        assert target_state[key] == value
    finally:
        checkpoint.close()
        dist.barrier()
        if dist.get_rank() == 0:
            import shutil

            shutil.rmtree(directory[0])
    return restored, restored_optimizer


def _run_pipeline(mesh: MeshContext, dtype: torch.dtype) -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    reference = _model(device, dtype)

    def shard_stage(part, world_mesh, moe_mesh, **kwargs):
        del world_mesh, moe_mesh, kwargs
        return _parallelize(part, mesh)

    pipeline = AutoPipeline(
        world_mesh=mesh.device_mesh,
        moe_mesh=None,
        pp_axis_name="pp",
        dp_axis_names=("dp_replicate", "dp_shard_cp"),
        cp_axis_name="cp",
        tp_axis_name="tp",
        pp_schedule="1f1b",
        pp_microbatch_size=1,
        pp_batch_size=MICROBATCHES,
        device=device,
        dtype=torch.bfloat16,
        pp_seq_len=SEQ,
        scale_grads_in_schedule=False,
        defer_fsdp_grad_sync=True,
    ).build(copy.deepcopy(reference), loss_fn=_loss, parallelize_fn=shard_stage)
    optimizers = [_optimizer(part, False) for part in pipeline.parts]
    reference_optimizer = _optimizer(reference, False)
    for step in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        batch = _batch(mesh.device_mesh["dp"].get_local_rank(), step, device)
        labels = batch.pop("labels")
        losses = [] if pipeline.info.has_last_stage else None
        if pipeline.info.has_first_stage:
            pipeline.info.schedule.step(batch.pop("input_ids"), target=labels, losses=losses, **batch)
        else:
            pipeline.info.schedule.step(target=labels, losses=losses)
        synchronize_tp_replica_gradients(pipeline.parts, mesh.device_mesh)
        norm = scale_grads_and_clip_grad_norm(
            float("inf"),
            pipeline.parts,
            pp_enabled=True,
            device_mesh=mesh.device_mesh,
            pp_axis_name="pp",
            num_label_tokens=int(labels.ne(-100).sum()) * mesh.dp_size,
            dp_group_size=mesh.dp_size,
        )
        expected_loss = _reference_backward(reference, mesh.dp_size, step, normalize_after_backward=True)
        _assert_gradients(pipeline.parts, reference)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), float("inf"))
        torch.testing.assert_close(norm.float(), expected_norm.float(), rtol=0.05, atol=1e-3)
        if losses is not None:
            actual_loss = torch.stack(losses).sum() / labels.ne(-100).sum()
            dist.all_reduce(actual_loss, group=mesh.device_mesh["dp_cp"].get_group())
            actual_loss /= mesh.dp_size
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0.01, atol=0.01)
        for optimizer in optimizers:
            optimizer.step()
        reference_optimizer.step()
        _assert_weights(pipeline.parts, reference)
    print(
        f"PASS: TP{mesh.tp_size}/PP{mesh.pp_size}/FSDP{mesh.dp_size} per-parameter, norm, loss, and update parity",
        flush=True,
    )


def _run_kd(teacher_axis: str) -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    teacher_topology = {
        "strategy": "fsdp2",
        "dp_size": 2 if teacher_axis == "dp" else 1,
        "tp_size": 2 if teacher_axis == "tp" else 1,
        "cp_size": 2 if teacher_axis == "cp" else 1,
    }
    setups = create_kd_distributed_setups(
        {
            "separate_meshes": True,
            "distributed": {"strategy": "fsdp2", "dp_size": 2},
            "teacher_distributed": teacher_topology,
        },
        world_size=dist.get_world_size(),
    )
    bridge = KDMeshBridge(setups, device=device)
    setup = setups.student if bridge.is_student else setups.teacher
    reference = _model(device, torch.float32)
    teacher_reference = _model(device, torch.bfloat16, seed=2027).eval()
    model = _parallelize(copy.deepcopy(reference if bridge.is_student else teacher_reference), setup.mesh_context)
    optimizer = _optimizer(model, False) if bridge.is_student else None
    reference_optimizer = _optimizer(reference, False) if bridge.is_student else None
    kd_loss = KDLoss()
    for step in range(2):
        if bridge.is_student:
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        for microbatch in range(MICROBATCHES):
            batch = None
            if bridge.is_student:
                batch = {
                    k: v[microbatch : microbatch + 1]
                    for k, v in _batch(setup.mesh_context.device_mesh["dp"].get_local_rank(), step, device).items()
                }
            bridge.broadcast_command(RUN_TEACHER if bridge.is_student else None)
            received = None
            for wave in range(bridge.num_waves):
                teacher_batch = bridge.send_batch(wave, batch)
                logits = None
                if bridge.is_teacher:
                    inputs = {k: v for k, v in teacher_batch.items() if k != "labels"}
                    sharder = ContextParallelSharder(model, setup.mesh_context.device_mesh, inputs)
                    ctx, inputs = sharder.shard(inputs)
                    with torch.no_grad(), ctx():
                        logits = materialize_teacher_logits(
                            model(**inputs).logits,
                            device_mesh=setup.mesh_context.device_mesh,
                            sequence_length=SEQ,
                        )
                result = bridge.send_logits(wave, logits)
                if result is not None:
                    received = result
            if bridge.is_student:
                with torch.no_grad():
                    expected_teacher = _forward_reference(teacher_reference, {"input_ids": batch["input_ids"]})
                torch.testing.assert_close(received.float(), expected_teacher, rtol=0.05, atol=0.01)
                model.set_requires_gradient_sync(microbatch == MICROBATCHES - 1)
                output = model(input_ids=batch["input_ids"]).logits
                loss = (
                    0.5
                    * (
                        _loss(output, batch["labels"]) / batch["labels"].ne(-100).sum()
                        + kd_loss(output, received, batch["labels"])
                    )
                    / MICROBATCHES
                )
                loss.backward()
        if bridge.is_student:
            for dp_rank in range(2):
                batch = _batch(dp_rank, step, device)
                labels = batch.pop("labels")
                for microbatch in range(MICROBATCHES):
                    inputs = {name: tensor[microbatch : microbatch + 1] for name, tensor in batch.items()}
                    micro_labels = labels[microbatch : microbatch + 1]
                    with torch.no_grad():
                        teacher_logits = _forward_reference(teacher_reference, inputs)
                    logits = _forward_reference(reference, inputs)
                    loss = (
                        0.5
                        * (
                            _loss(logits, micro_labels) / micro_labels.ne(-100).sum()
                            + kd_loss(logits, teacher_logits, micro_labels)
                        )
                        / (2 * MICROBATCHES)
                    )
                    loss.backward()
            _assert_gradients([model], reference)
            norm = clip_grad_norm(MAX_NORM, [model], device_mesh=setup.mesh_context.device_mesh)
            ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), MAX_NORM)
            torch.testing.assert_close(norm.float(), ref_norm.float(), rtol=0.05, atol=1e-3)
            optimizer.step()
            reference_optimizer.step()
            _assert_weights([model], reference)
    print(f"PASS: separate-mesh Qwen3.5 KD student DP2 / teacher {teacher_axis}2", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("packed-resume", "cp", "tp", "pp-tp", "kd"), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--te", action="store_true")
    parser.add_argument("--teacher-axis", choices=("dp", "tp", "cp"), default="tp")
    parser.add_argument("--tp-size", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        if args.case == "kd":
            assert dist.get_world_size() == 4, "separate KD requires two student and two teacher GPUs"
            _run_kd(args.teacher_axis)
        else:
            policy = FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16 if args.case == "cp" else torch.float32,
                    output_dtype=torch.bfloat16,
                ),
                enable_fsdp2_prefetch=True,
            )
            sizes = ParallelismSizes(
                tp_size=args.tp_size if args.case in ("tp", "pp-tp") else 1,
                cp_size=2 if args.case == "cp" else 1,
                pp_size=2 if args.case == "pp-tp" else 1,
            )
            mesh = MeshContext.build(policy, sizes, activation_checkpointing=args.case != "pp-tp")
            dtype = getattr(torch, args.dtype)
            if args.case == "pp-tp":
                _run_pipeline(mesh, dtype)
            else:
                _run_dense(
                    mesh, dtype, packed=args.case == "packed-resume", te=args.te, resume=args.case == "packed-resume"
                )
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"PASS: {args.case}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

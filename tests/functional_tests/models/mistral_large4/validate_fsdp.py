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

"""Validate real FSDP/HSDP against an unsharded FP32 model.

Run with PYTHONPATH=. python <this-file> --output <directory> --world-size 2.
Use --world-size 4 --replicate-size 2 for composed HSDP. CUDA uses NCCL.
"""

import argparse
import copy
import json
import tempfile
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed import FSDP2Config, MeshContext
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.unit_tests.models.mistral_large4.test_checkpoint import make_optimizer, peft_config
from tests.unit_tests.models.mistral_large4.test_model import make_model


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """Read global values without changing parameter ownership.

    Args:
        tensor: Tensor or DTensor with arbitrary parameter/gradient axes. FSDP
            shards its leading axis and HSDP also replicates it on mesh axis zero.

    Returns:
        Detached full-shaped CPU tensor in the same axis order.
    """
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().cpu()


def _batch(rank: int, step: int, micro: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unequal per-rank sequences and a rank-local empty supervision case.

    Returns:
        Integer IDs and labels [1, sequence]; labels use -100 for ignored targets.
    """
    ids = ((torch.arange(7 + rank % 2, device=device) * 3 + rank + step + micro) % 29 + 1)[None]
    labels = ids.clone()
    if rank == 0 and step == 1:
        labels.fill_(-100)
    return ids, labels


def _global_batch(context, rank, iteration, micro, device):
    ids, labels = _batch(rank // context.cp_size, iteration, micro, device)
    batch = {"input_ids": ids, "labels": labels}
    if context.cp_size > 1:
        # Two documents with unequal left/right padding and custom nonmonotonic
        # positions distinguish causal storage order from rotary positions.
        ids = torch.cat((ids, ids.flip(1)), 0)
        labels = torch.cat((labels, labels.flip(1)), 0)
        mask = torch.ones_like(ids, dtype=torch.bool)
        mask[0, :2] = False
        mask[1, -2:] = False
        batch = {"input_ids": ids, "labels": labels, "attention_mask": mask}
        if micro == 1:
            batch["position_ids"] = torch.arange(ids.shape[1], device=device).flip(0)[None].expand_as(ids)
    return batch


def _local_batch(model, context, batch):
    if context.cp_size == 1:
        return batch, None
    sharder = ContextParallelSharder(model=model, device_mesh=context.device_mesh, batch=batch)
    _, local = sharder.shard(batch)
    return local, sharder


def _parity_case(
    mesh: MeshContext, rank: int, world: int, device: torch.device, *, lora: bool, ac: bool, depth: int
) -> dict:
    model = make_model(
        n_routed_experts=4, num_hidden_layers=depth, nope_layer_flags=None, kv_source_layers=None, index_sharing=None
    ).to(device)
    if lora:
        peft = peft_config()
        peft.dropout = 0.0
        apply_lora_to_linear_modules(model, peft)
        # Nonzero B factors exercise gradients through both adapter matrices.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.01)
    reference = copy.deepcopy(model)
    mesh.activation_checkpointing = ac
    model = parallelize_model(model, mesh)
    assert world == 1 or isinstance(model, FSDPModule)
    units = sum(isinstance(module, FSDPModule) for module in model.modules())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.003)
    ref_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.003)
    maximum = {"logits": 0.0, "gradients": 0.0, "parameters": 0.0, "norm": 0.0}
    # Full accumulation windows of one and three, then a partial window of two.
    for iteration, accumulation in enumerate((1, 3, 2)):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        prepare_for_grad_accumulation([model])
        for micro in range(accumulation):
            if micro == accumulation - 1:
                prepare_for_final_backward([model])
            batch = _global_batch(mesh, rank, iteration, micro, device)
            expected = reference(**batch)
            local, sharder = _local_batch(model, mesh, dict(batch))
            actual = model(**local)
            expected_logits = expected.logits
            actual_logits = actual.logits
            if sharder is not None:
                length = actual.logits.shape[1]
                start = mesh.device_mesh["cp"].get_local_rank() * length
                keep = max(0, min(length, expected.logits.shape[1] - start))
                expected_logits = expected.logits[:, start : start + keep]
                actual_logits = actual.logits[:, :keep]
            torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=2e-5)
            maximum["logits"] = max(maximum["logits"], (actual_logits - expected_logits).abs().max().item())
            (actual.loss / accumulation).backward()
            # Average each rank's mean loss, matching the existing FSDP DP contract.
            for peer in range(world):
                peer_batch = _global_batch(mesh, peer, iteration, micro, device)
                (reference(**peer_batch).loss / (world * accumulation)).backward()
        scale_grads_and_clip_grad_norm(
            float("inf"), [model], moe_mesh=mesh.moe_mesh, ep_axis_name="ep", dp_group_size=world, foreach=False
        )
        reference_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            expected = reference_params[canonical_parameter_fqn(name)]
            if not parameter.requires_grad:
                assert parameter.grad is None and expected.grad is None
                continue
            actual_grad = _full(parameter.grad)
            expected_grad = expected.grad.cpu()
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=4e-5, msg=name)
            maximum["gradients"] = max(maximum["gradients"], (actual_grad - expected_grad).abs().max().item())
        clip = 0.25 if iteration == 1 else float("inf")
        actual_norm = clip_grad_norm(clip, [model], moe_mesh=mesh.moe_mesh, ep_axis_name="ep", foreach=False)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), clip, foreach=False)
        torch.testing.assert_close(
            torch.as_tensor(actual_norm).float().cpu(), expected_norm.float().cpu(), atol=2e-6, rtol=4e-5
        )
        maximum["norm"] = max(maximum["norm"], abs(float(actual_norm) - float(expected_norm)))
        optimizer.step()
        ref_optimizer.step()
        for name, parameter in model.named_parameters():
            value = _full(parameter)
            torch.testing.assert_close(
                value, reference_params[canonical_parameter_fqn(name)].cpu(), atol=2e-6, rtol=4e-5, msg=name
            )
            maximum["parameters"] = max(
                maximum["parameters"],
                (value - reference_params[canonical_parameter_fqn(name)].cpu()).abs().max().item(),
            )
    return {
        "lora": lora,
        "activation_checkpointing": ac,
        "depth": depth,
        "fsdp_units": units,
        "max_abs_error": maximum,
        "passed": True,
    }


def _lifecycle_case(
    context: MeshContext, rank: int, device: torch.device, output: Path, *, lora: bool, mixed: bool
) -> dict:
    context.activation_checkpointing = True
    context.strategy_config.mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if mixed else torch.float32,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    peft = peft_config() if lora else None
    # FP32 master storage for full-parameter training, BF16 compute if selected.
    model = make_model(
        n_routed_experts=4,
    ).to(device)
    if peft is not None:
        apply_lora_to_linear_modules(model, peft)
    model = parallelize_model(model, context)
    # Inspect tensors inside real module calls, after FSDP unshards/casts them.
    # V4's precision policy distinguishes FP32 masters from FP32 compute islands;
    # assert both sides of that contract, including the self-casting key norm.
    observed_dtypes = {}
    handles = []
    expected_body_dtype = torch.bfloat16 if mixed else torch.float32

    def capture_dtype(name, expected, *, rotary=False, preserves_input=False):
        def check(module, inputs, result):
            parameter = module.alpha if rotary else module.weight
            assert parameter.dtype == expected, (name, parameter.dtype, expected)
            if rotary:
                assert all(value.dtype == torch.float32 for value in result)
            elif preserves_input:
                assert result.dtype == inputs[0].dtype == expected_body_dtype
            elif isinstance(result, tuple):
                # The common router returns FP32 weights, integer expert IDs,
                # and an optional auxiliary loss instead of raw gate logits.
                assert result[0].dtype == expected
                assert result[1].dtype == torch.int64
            else:
                assert result.dtype == expected
            observed_dtypes[name] = str(parameter.dtype)

        return check

    for name, module in model.named_modules():
        canonical = canonical_parameter_fqn(name)
        if canonical.endswith("rotary_emb"):
            handles.append(module.register_forward_hook(capture_dtype(canonical, torch.float32, rotary=True)))
        elif canonical.endswith("mlp.gate"):
            handles.append(module.register_forward_hook(capture_dtype(canonical, torch.float32)))
        elif canonical.endswith("indexer.k_norm"):
            handles.append(module.register_forward_hook(capture_dtype(canonical, torch.float32, preserves_input=True)))
        elif canonical.endswith("self_attn.kv_a_proj_with_mqa"):
            handles.append(module.register_forward_hook(capture_dtype(canonical, expected_body_dtype)))
    assert handles
    optimizer, scheduler = make_optimizer(model)
    rng = StatefulRNG(seed=781 + rank)

    def update(model, optimizer, scheduler, iteration):
        optimizer.zero_grad(set_to_none=True)
        prepare_for_final_backward([model])
        batch = _global_batch(context, rank, iteration, 0, device)
        local, _ = _local_batch(model, context, batch)
        loss = model(**local).loss
        loss.backward()
        scale_grads_and_clip_grad_norm(
            0.5,
            [model],
            moe_mesh=context.moe_mesh,
            ep_axis_name="ep",
            dp_group_size=dist.get_world_size(),
            foreach=False,
        )
        optimizer.step()
        scheduler.step()
        assert bool(torch.isfinite(loss))
        return loss.detach()

    for iteration in range(2):
        update(model, optimizer, scheduler, iteration)
    assert any(name.endswith("rotary_emb") for name in observed_dtypes)
    assert any(name.endswith("mlp.gate") for name in observed_dtypes)
    assert any(name.endswith("indexer.k_norm") for name in observed_dtypes)
    assert any(name.endswith("kv_a_proj_with_mqa") for name in observed_dtypes)
    for handle in handles:
        handle.remove()
    checkpoint = output / (f"resume-{'lora' if lora else 'sft'}-{'mixed' if mixed else 'fp32'}")
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(checkpoint),
        model_save_format="torch_save",
        save_consolidated=False,
        is_peft=lora,
        cpu_offload=True,
    ).build(dp_rank=rank, tp_rank=0, pp_rank=0)
    try:
        checkpointer.save_model(model, str(checkpoint), peft_config=peft)
        checkpointer.save_optimizer(optimizer, model, str(checkpoint), scheduler)
        checkpointer.save_on_global_ranks(rng, "rng", str(checkpoint))
        saved_parameters = {name: _full(value).clone() for name, value in model.state_dict().items()}
        saved_optimizer = [
            {name: _full(value).clone() for name, value in state.items()} for state in optimizer.state.values()
        ]
        expected = torch.stack([update(model, optimizer, scheduler, i) for i in (2, 3)])
        restored = make_model(n_routed_experts=4).to(device)
        if peft is not None:
            apply_lora_to_linear_modules(restored, peft)
        restored = parallelize_model(restored, context)
        new_optimizer, new_scheduler = make_optimizer(restored)
        checkpointer.load_model(restored, str(checkpoint / "model"), is_init_step=False)
        checkpointer.load_optimizer(new_optimizer, restored, str(checkpoint), new_scheduler)
        checkpointer.load_on_global_ranks(rng, "rng", str(checkpoint))
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(
                _full(value), saved_parameters[name], atol=0, rtol=0, msg="loaded parameter " + name
            )
        for index, (saved_state, loaded_state) in enumerate(zip(saved_optimizer, new_optimizer.state.values())):
            for name, value in saved_state.items():
                torch.testing.assert_close(
                    _full(loaded_state[name]), value, atol=0, rtol=0, msg=f"loaded optimizer {index} {name}"
                )
        actual = torch.stack([update(restored, new_optimizer, new_scheduler, i) for i in (2, 3)])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(_full(restored.state_dict()[name]), _full(value), atol=0, rtol=0, msg=name)
        for left, right in zip(optimizer.state.values(), new_optimizer.state.values()):
            for name in left:
                torch.testing.assert_close(_full(left[name]), _full(right[name]), atol=0, rtol=0)
        assert scheduler.state_dict() == new_scheduler.state_dict()
    finally:
        checkpointer.close()
    return {
        "lora": lora,
        "bf16_compute_fp32_storage": mixed,
        "activation_checkpointing": True,
        "resume_exact": True,
        "runtime_parameter_dtypes": observed_dtypes,
        "continuation_losses": actual.tolist(),
    }


def _worker(
    rank: int, world: int, replicate: int, ep_size: int, cp_size: int, rendezvous: str, output: str, use_cuda: bool
) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )
    report = {
        "rank": rank,
        "world_size": world,
        "replicate_size": replicate,
        "ep_size": ep_size,
        "cp_size": cp_size,
        "device": str(device),
        "torch": torch.__version__,
        "cases": [],
    }
    target = Path(output) / f"rank-{rank}.json"
    try:
        mesh = init_device_mesh(
            device.type,
            (1, replicate, world // (replicate * cp_size), cp_size, 1),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp"),
        )
        mesh[("dp_shard", "cp")]._flatten(mesh_dim_name="dp_shard_cp")
        moe_mesh = (
            init_device_mesh(device.type, (1, world // ep_size, ep_size), mesh_dim_names=("pp", "ep_shard", "ep"))
            if ep_size > 1
            else None
        )
        context = MeshContext(
            moe_mesh=moe_mesh,
            device_mesh=mesh,
            strategy_config=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False
                )
            ),
        )
        for lora, ac, depth in (
            (False, False, 4),
            (False, True, 4),
            (True, False, 4),
            (True, True, 4),
            (False, True, 40),
        ):
            report["cases"].append(_parity_case(context, rank, world, device, lora=lora, ac=ac, depth=depth))
            target.write_text(json.dumps(report, indent=2))
        report["lifecycle"] = []
        for lora, mixed in ((False, False), (True, False), (False, True), (True, True)):
            report["lifecycle"].append(_lifecycle_case(context, rank, device, Path(output), lora=lora, mixed=mixed))
            target.write_text(json.dumps(report, indent=2))
        report["passed"] = True
        target.write_text(json.dumps(report, indent=2))
    except BaseException:
        (Path(output) / f"rank-{rank}-failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--replicate-size", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    args = parser.parse_args()
    if (
        min(args.world_size, args.replicate_size, args.ep_size, args.cp_size) < 1
        or args.world_size % (args.replicate_size * args.cp_size)
        or args.world_size % args.ep_size
    ):
        parser.error("world size must be positive and divisible by replication * CP and by EP")
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        mp.spawn(
            _worker,
            args=(
                args.world_size,
                args.replicate_size,
                args.ep_size,
                args.cp_size,
                "file://" + str(Path(temporary).resolve() / "rendezvous"),
                str(args.output.resolve()),
                args.cuda,
            ),
            nprocs=args.world_size,
            join=True,
        )
    print(json.dumps({"passed": True, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

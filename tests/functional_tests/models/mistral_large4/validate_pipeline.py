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

"""Real GPU PP/EP/FSDP forward, gradient, clipping and optimizer parity.

Default PR-sized execution uses two CUDA ranks and PP2. Explicit four-rank
execution covers PP2/EP2 and PP4 for extended/scheduled validation.
Run with torchrun --standalone --nproc-per-node=2 <this-file> --output <dir>.
"""

import argparse
import copy
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
from nemo_automodel.components.training.utils import (
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.unit_tests.models.mistral_large4.test_checkpoint import peft_config
from tests.unit_tests.models.mistral_large4.test_context_parallel import make_sparse_model
from tests.unit_tests.models.mistral_large4.test_hybridep import grouped_backend, use_transport_rounding_reference
from tests.unit_tests.models.mistral_large4.test_model import assert_bf16_error, tensor_error
from tests.unit_tests.models.mistral_large4.test_multimodal import image_batch
from tests.unit_tests.models.mistral_large4.test_pipeline import make_pipeline_model


def full_value(value: torch.Tensor) -> torch.Tensor:
    """Read a global parameter or gradient with arbitrary matrix/vector shape.

    Args:
        value: Tensor or leading-axis-sharded FSDP DTensor; EP-owned experts
            retain globally named local expert identity.

    Returns:
        Detached CPU tensor with the full parameter's shape and original dtype.
    """
    return (value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu()


def make_batch(
    peer: int, iteration: int, microbatch_count: int, *, multimodal: bool, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[torch.Tensor]]]:
    """Return IDs/masks/shifted labels [microbatches, sequence] and ragged image lists."""
    if multimodal:
        image_ids, images = image_batch(device)
        rows, media = [], []
        for micro in range(microbatch_count):
            if (peer + micro + iteration) % 2:
                rows.append(image_ids[0])
                media.append([p * (1 + 0.02 * (peer + iteration + micro)) for p in images])
            else:
                rows.append((torch.arange(image_ids.shape[1], device=device) % 8 + 1).long())
                media.append([])
        ids = torch.stack(rows)
    else:
        ids = (
            torch.arange(7, device=device)[None]
            + torch.arange(microbatch_count, device=device)[:, None] * 3
            + peer
            + iteration
        ) % 29 + 1
        media = [[] for _ in range(microbatch_count)]
    mask = torch.ones_like(ids, dtype=torch.bool)
    if not multimodal:
        mask[0, :2] = False
        mask[-1, -2:] = False
    labels = ids.roll(-1, dims=1)
    labels[:, -1] = -100
    labels.masked_fill_(~mask, -100)
    if multimodal:
        labels.masked_fill_((labels == 10) | (labels == 12) | (labels == 13), -100)
    if iteration == 1:
        labels[0].fill_(-100)
    positions = torch.arange(ids.shape[1], device=device).flip(0)[None].expand_as(ids)
    return ids, mask, labels, positions, media


def align_reference_weights(
    parts: list[torch.nn.Module], reference: torch.nn.Module, pp_group: dist.ProcessGroup | None = None
) -> None:
    """Copy current distributed parameters to the numerical reference before a step.

    Each value retains its model parameter shape and storage dtype. PP peers
    exchange full CPU stage dictionaries; this is only for bounded fixtures.
    Optimizer updates are compared before this copy, so the check measures
    each update from equal weights without claiming trajectory equivalence.
    """
    local = {
        canonical_parameter_fqn(name): full_value(parameter).clone()
        for part in parts
        for name, parameter in part.named_parameters()
    }
    if pp_group is not None:
        gathered = [None] * dist.get_world_size(pp_group)
        dist.all_gather_object(gathered, local, group=pp_group)
        local = {name: value for stage in gathered for name, value in stage.items()}
    parameters = dict(reference.named_parameters())
    assert set(parameters) == set(local)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(local[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--schedule", choices=("1f1b", "interleaved1f1b"), default="1f1b")
    parser.add_argument("--ac", action="store_true")
    parser.add_argument(
        "--reduce-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="FSDP gradient reduction dtype; float32 matches the common FSDP2 recipe default",
    )
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--packed", action="store_true", help="Compare packed THD CP/PP against unsharded THD")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--hybridep", action="store_true")
    parser.add_argument(
        "--full-index", action="store_true", help="Select all causal keys to isolate smooth gradient parity"
    )
    parser.add_argument(
        "--match-transport-rounding",
        action="store_true",
        help="Use a single-process reference with HybridEP BF16 weighted-output rounding",
    )
    parser.add_argument(
        "--align-step-weights", action="store_true", help="Compare each update from identical current weights"
    )
    parser.add_argument("--attn", choices=("sdpa", "tilelang"), default="sdpa")
    args = parser.parse_args()
    if args.attn == "tilelang" and args.dtype != "bfloat16":
        raise ValueError("TileLang validation requires --dtype bfloat16")
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    dtype = getattr(torch, args.dtype)
    reduce_dtype = getattr(torch, args.reduce_dtype)
    setup = DistributedSetup.build(
        strategy=FSDP2Config(
            mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=reduce_dtype, output_dtype=dtype),
            defer_fsdp_grad_sync=False,
            reshard_after_forward=True,
        ),
        parallelism_sizes=ParallelismSizes(pp_size=args.pp_size, ep_size=args.ep_size, cp_size=args.cp_size),
        activation_checkpointing=args.ac,
        world_size=world,
    )
    context = setup.mesh_context
    dp_cp = world // args.pp_size
    dp = dp_cp // args.cp_size
    peer = context.device_mesh["dp"].get_local_rank()
    model = (
        make_sparse_model(dtype=dtype, multimodal=args.multimodal)
        if args.attn == "tilelang"
        else make_pipeline_model(dtype=dtype, multimodal=args.multimodal)
    ).to(device)
    if args.lora:
        peft = peft_config()
        peft.dropout = 0.0
        apply_lora_to_linear_modules(model, peft)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.01)
    if args.hybridep:
        if args.ep_size < 2 or dtype != torch.bfloat16 or args.lora:
            raise ValueError("HybridEP validation requires EP>1, BF16 and full fine-tuning")
        config = copy.deepcopy(model.config)
        if args.full_index:
            config.index_topk = 64
        config.hidden_size = 256
        config.intermediate_size = 384
        config.moe_intermediate_size = 256
        reference = type(model)(config, replace(grouped_backend(), attn=args.attn)).to(device)
        reference.initialize_weights()
        if args.match_transport_rounding:
            use_transport_rounding_reference(reference)
        model = type(model)(config, replace(grouped_backend("hybridep"), attn=args.attn)).to(device)
        model.load_state_dict(reference.state_dict())
    else:
        reference = copy.deepcopy(model)
    microbatches = 4
    sequence = 11 if args.multimodal else 7
    captured = []

    def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Sum CE for FP32 logits [1, sequence, vocab] and shifted labels [1, sequence].

        Returns:
            Scalar summed cross entropy; global normalization happens after backward.
        """
        captured.append(logits.detach().cpu())
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")

    def shard(part: torch.nn.Module, **runtime) -> torch.nn.Module:
        del runtime
        return parallelize_model(part, context)

    pp = AutoPipeline(
        world_mesh=context.device_mesh,
        moe_mesh=context.moe_mesh,
        **context.pipeline_axis_kwargs(),
        pp_schedule=args.schedule,
        pp_microbatch_size=1,
        pp_batch_size=microbatches,
        layers_per_stage=1 if args.schedule == "interleaved1f1b" else None,
        patch_inner_model=False,
        patch_causal_lm_model=False,
        defer_fsdp_grad_sync=False,
        device=device,
        dtype=dtype,
        pp_seq_len=sequence,
    )
    pp.build(model, loss_fn=loss_fn, parallelize_fn=shard)
    parts = pp.parts
    if args.hybridep:
        experts = [module for part in parts for module in part.modules() if isinstance(module, GroupedExpertsDeepEP)]
        assert experts and all(
            module.dispatcher_backend == "hybridep" and module.ep_size == args.ep_size for module in experts
        )
        assert all(module.token_dispatcher is not None for module in experts)
    parameters = [p for part in parts for p in part.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.003)
    reference_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.003)
    errors = {"logits_max_abs": 0.0, "gradient_max_abs": 0.0, "norm_max_abs": 0.0, "update_max_abs": 0.0}
    reference_parameters = dict(reference.named_parameters())
    for iteration, accumulation in enumerate((1, 2)):
        if args.align_step_weights:
            align_reference_weights(parts, reference, context.device_mesh["pp"].get_group())
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        captured.clear()
        reference_gradient_sums = (
            {
                name: torch.zeros_like(p, dtype=torch.float32)
                for name, p in reference_parameters.items()
                if p.requires_grad
            }
            if args.hybridep
            else None
        )
        expected_logits = []
        token_count = 0
        prepare_for_grad_accumulation(parts, pp_enabled=True)
        for window in range(accumulation):
            if window == accumulation - 1:
                prepare_for_final_backward(parts, pp_enabled=True)
            ids, mask, labels, positions, media = make_batch(
                peer, iteration + window, microbatches, multimodal=args.multimodal, device=device
            )
            for candidate_peer in range(dp):
                ri, rm, rl, rp, rmedia = make_batch(
                    candidate_peer, iteration + window, microbatches, multimodal=args.multimodal, device=device
                )
                token_count += int(rl.ne(-100).sum())
                for micro in range(microbatches):
                    kwargs = {"pixel_values": rmedia[micro]} if args.multimodal else {}
                    if args.packed:
                        kwargs.update(
                            qkv_format="thd",
                            cu_seqlens=torch.tensor(
                                [0, sequence] if args.multimodal else [0, 3, sequence], device=device, dtype=torch.int32
                            ),
                        )
                    logits = reference(
                        ri[micro : micro + 1],
                        attention_mask=rm[micro : micro + 1],
                        position_ids=rp[micro : micro + 1],
                        **kwargs,
                    ).logits
                    if candidate_peer == peer:
                        expected_logits.append(logits.detach().cpu())
                    F.cross_entropy(
                        logits.flatten(0, 1), rl[micro : micro + 1].flatten(), ignore_index=-100, reduction="sum"
                    ).backward()
                    if reference_gradient_sums is not None:
                        # A single-rank oracle otherwise rounds a longer serial BF16
                        # sum than the distributed reduction tree. Accumulate the
                        # independent microbatch gradients in FP32 to reduce oracle error.
                        for name, parameter in reference_parameters.items():
                            if parameter.grad is not None:
                                reference_gradient_sums[name].add_(parameter.grad.float())
                                parameter.grad = None
            if args.multimodal and pp.info.has_first_stage:
                parts[0]._vlm_pixel_values_chunks = media
                parts[0]._vlm_chunk_idx = 0
            losses = [] if pp.info.has_last_stage else None
            batch = dict(input_ids=ids, labels=labels, attention_mask=mask, position_ids=positions)
            if args.packed:
                lengths = torch.tensor([sequence] if args.multimodal else [3, sequence - 3], device=device)
                batch.update(
                    qkv_format="thd",
                    seq_lens=lengths[None].expand(microbatches, -1),
                    seq_lens_padded=lengths[None].expand(microbatches, -1),
                )
            sharder = ContextParallelSharder(parts[0], context.device_mesh, batch, num_chunks=microbatches)
            train_ctx, batch = sharder.shard(batch)
            local_targets = batch.pop("labels")
            primary = batch.pop("input_ids")
            pp.update_seq_len(primary.shape[1])
            with train_ctx():
                pp.step(
                    primary,
                    target=local_targets if pp.info.has_last_stage else None,
                    losses=losses,
                    **batch,
                )
            if args.multimodal and pp.info.has_first_stage:
                assert parts[0]._vlm_chunk_idx == microbatches
                parts[0]._vlm_pixel_values_chunks = None
        for name, parameter in reference_parameters.items():
            if reference_gradient_sums is not None and parameter.requires_grad:
                parameter.grad = (reference_gradient_sums[name] / token_count).to(parameter.dtype)
            elif parameter.grad is not None:
                parameter.grad.div_(token_count)
        actual_norm = scale_grads_and_clip_grad_norm(
            float("inf"),
            parts,
            pp_enabled=True,
            device_mesh=context.device_mesh,
            pp_axis_name="pp",
            moe_mesh=context.moe_mesh,
            ep_axis_name="ep" if args.ep_size > 1 else None,
            dp_group_size=dp_cp,
            num_label_tokens=token_count,
            foreach=False,
        )
        atol, rtol = (4e-6, 1e-4) if dtype == torch.float32 else (0.003, 0.04)
        if pp.info.has_last_stage:
            assert len(captured) == len(expected_logits)
            for actual, expected in zip(captured, expected_logits, strict=True):
                if args.cp_size > 1:
                    start = context.device_mesh["cp"].get_local_rank() * actual.shape[1]
                    keep = min(actual.shape[1], max(0, expected.shape[1] - start))
                    expected = expected[:, start : start + keep]
                    actual = actual[:, :keep]
                errors["logits_max_abs"] = max(errors["logits_max_abs"], float((actual - expected).abs().max()))
                if args.hybridep:
                    assert_bf16_error(tensor_error("logits", actual, expected))
                else:
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        gradient_errors = []
        for part in parts:
            for name, parameter in part.named_parameters():
                ref = reference_parameters[canonical_parameter_fqn(name)]
                if not parameter.requires_grad:
                    assert parameter.grad is None
                    continue
                assert parameter.grad is not None, name
                actual, expected = full_value(parameter.grad), ref.grad.detach().cpu()
                assert torch.isfinite(actual).all(), name
                errors["gradient_max_abs"] = max(errors["gradient_max_abs"], float((actual - expected).abs().max()))
                if args.hybridep:
                    gradient_errors.append(tensor_error(name, actual, expected))
                else:
                    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)
        if gradient_errors:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / f"gradients-step{iteration}-rank{rank}.json").write_text(
                json.dumps(
                    [
                        {
                            "name": error.name,
                            "relative_l2": error.relative_l2,
                            "rms_error": error.rms_error,
                            "reference_rms": error.reference_rms,
                            "max_absolute": error.max_absolute,
                        }
                        for error in gradient_errors
                    ],
                    indent=2,
                )
            )
            for error in gradient_errors:
                assert_bf16_error(error)
            totals = torch.tensor(
                [sum(e.squared_error for e in gradient_errors), sum(e.reference_squared_norm for e in gradient_errors)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(totals, group=context.device_mesh["pp"].get_group())
            relative_l2 = float((totals[0] / totals[1]).sqrt())
            assert relative_l2 < 0.005, relative_l2
            errors["gradient_relative_l2"] = max(errors.get("gradient_relative_l2", 0.0), relative_l2)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), float("inf"), foreach=False)
        errors["norm_max_abs"] = max(errors["norm_max_abs"], abs(float(actual_norm) - float(expected_norm)))
        torch.testing.assert_close(actual_norm.float(), expected_norm.float(), atol=atol, rtol=rtol)
        # Apply the same clipping after comparing unmodified gradients.
        factor = min(1.0, 0.25 / (float(actual_norm) + 1e-6))
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25, foreach=False)
        optimizer.step()
        reference_optimizer.step()
        for part in parts:
            for name, parameter in part.named_parameters():
                actual = full_value(parameter)
                expected = reference_parameters[canonical_parameter_fqn(name)].detach().cpu()
                errors["update_max_abs"] = max(errors["update_max_abs"], float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=name)
    result = {
        "passed": True,
        "rank": rank,
        "world_size": world,
        "pp_size": args.pp_size,
        "ep_size": args.ep_size,
        "cp_size": args.cp_size,
        "dtype": args.dtype,
        "reduce_dtype": args.reduce_dtype,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "schedule": args.schedule,
        "activation_checkpointing": args.ac,
        "multimodal": args.multimodal,
        "lora": args.lora,
        "hybridep": args.hybridep,
        "attention_backend": args.attn,
        "packed": args.packed,
        "full_index_fixture": args.full_index,
        "reference_matches_transport_rounding": args.match_transport_rounding,
        "reference_weights_aligned_each_step": args.align_step_weights,
        "reference_gradient_accumulation_dtype": "float32" if args.hybridep else args.dtype,
        "microbatches": microbatches,
        "accumulation_windows": [1, 2],
        "optimizer_updates": 2,
        "errors": errors,
        "reference": "Unsplit model, identical initial weights and compute dtype; FP32 cases use FP32 oracle",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

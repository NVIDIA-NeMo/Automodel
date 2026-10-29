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

"""Two-rank CP parity: nonpacked and packed THD image/text, plus the native sparse-MLA kernel.

Run with torchrun --standalone --nproc-per-node=2 <this-file> --output <dir> [--packed | --kernel-only].
The default nonpacked mode uses CUDA/FSDP2; --cpu isolates CP collectives with replicated weights.
Packed mode compares THD CP with an unsharded reference; --attn tilelang uses native sparse MLA.
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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.nn import functional as F

from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model
from nemo_automodel.components.models.mistral_large4.attention import sparse_mla
from nemo_automodel.components.models.mistral_large4.context_parallel import gather_tokens
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.functional_tests.models.mistral_large4.validate_pipeline import full_value
from tests.unit_tests.models.mistral_large4.test_context_parallel import dense_latent_oracle, make_sparse_model
from tests.unit_tests.models.mistral_large4.test_multimodal import image_batch
from tests.unit_tests.models.mistral_large4.test_pipeline import make_pipeline_model


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Compare identically shaped logits, parameters or gradients in FP32."""
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return {"relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)), "max_abs": float((a - b).abs().max())}


def kernel_relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Compare equal-shaped kernel outputs or gradients in FP32.

    Args:
        actual: Arbitrary-shape floating output/gradient tensor.
        reference: Independent same-shape FP32 reference.

    Returns:
        Relative L2 error; finite tensors are required.
    """
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    return float(
        (actual.detach().float() - reference.detach().float()).norm()
        / reference.detach().float().norm().clamp_min(1e-12)
    )


def validate_sparse_kernel(output: Path) -> None:
    """Check native MLA and CP gradients, including empty-query peers."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    rank, world = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))["cp"]
    tokens, heads = 32, 64
    local = tokens // world
    records = []
    for case in ("rope", "nope", "empty-query-rank", "all-empty"):
        torch.manual_seed(719)
        q = (torch.randn(tokens, heads, 576, device=device) * 0.2).bfloat16()
        kv = (torch.randn(tokens, 576, device=device) * 0.2).bfloat16()
        weight = (torch.randn(heads, 128, 512, device=device) / 512**0.5).bfloat16()
        if case == "nope":
            q[..., 512:] = 0
            kv[..., 512:] = 0
        # Two packed documents. Only up to four selected causal keys per query;
        # the kernel must not include other keys or cross document boundaries.
        selected = torch.full((tokens, 4), -1, device=device, dtype=torch.long)
        for i in range(tokens):
            choices = torch.arange(max((i // 16) * 16, i - 3), i + 1, device=device)
            selected[i, : choices.numel()] = choices
        if case == "empty-query-rank":
            selected[tokens // 2 :] = -1
        if case == "all-empty":
            selected.fill_(-1)
        start, stop = rank * local, (rank + 1) * local
        local_q = q[start:stop].clone().requires_grad_()
        local_kv = kv[start:stop].clone().requires_grad_()
        local_w = weight.clone().requires_grad_()
        global_kv = gather_tokens(local_kv, mesh, differentiable=True)
        actual = sparse_mla(local_q, global_kv, selected[start:stop], local_w, scale=128**-0.5)
        ref_q, ref_kv, ref_w = [t.float().requires_grad_() for t in (q, kv, weight)]
        expected = dense_latent_oracle(ref_q, ref_kv, selected, ref_w, scale=128**-0.5)
        upstream = torch.randn_like(expected)
        actual.backward(upstream[start:stop].to(actual.dtype))
        # Same rounded upstream in the reference isolates kernel arithmetic.
        expected.backward(upstream.bfloat16().float())
        weight_grad = local_w.grad.float()
        dist.all_reduce(weight_grad)
        record = dict(
            case=case,
            logits_relative_l2=kernel_relative_error(actual, expected[start:stop]),
            query_gradient_relative_l2=kernel_relative_error(local_q.grad, ref_q.grad[start:stop]),
            kv_gradient_relative_l2=kernel_relative_error(local_kv.grad, ref_kv.grad[start:stop]),
            value_weight_gradient_relative_l2=kernel_relative_error(weight_grad, ref_w.grad),
        )
        record["passed"] = all(value <= 0.005 for name, value in record.items() if name.endswith("relative_l2"))
        records.append(record)
    output.mkdir(parents=True, exist_ok=True)
    passed = all(record["passed"] for record in records)
    (output / f"rank{rank}.json").write_text(
        json.dumps(dict(passed=passed, cp_size=world, cases=records), indent=2) + "\n"
    )
    dist.destroy_process_group()
    assert passed, records


def validate_packed(args: argparse.Namespace) -> None:
    """Compare packed THD CP with an unsharded reference over three local updates."""
    if args.attn == "tilelang" and args.dtype != "bfloat16":
        raise ValueError("TileLang validation requires --dtype bfloat16")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    dtype = getattr(torch, args.dtype)
    setup = DistributedSetup.build(
        strategy=FSDP2Config(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=dtype, reduce_dtype=torch.float32, output_dtype=dtype, cast_forward_inputs=False
            ),
            defer_fsdp_grad_sync=False,
        ),
        parallelism_sizes=ParallelismSizes(cp_size=world),
        activation_checkpointing=args.ac,
        world_size=world,
    )
    context = setup.mesh_context
    model = (
        make_sparse_model(dtype=dtype, multimodal=True)
        if args.attn == "tilelang"
        else make_pipeline_model(dtype=dtype, multimodal=True)
    ).to(device)
    if args.reference_attn == "sdpa":
        reference = type(model)(copy.deepcopy(model.config), replace(model.backend, attn="sdpa")).to(device)
        reference.load_state_dict(model.state_dict())
    else:
        reference = copy.deepcopy(model)
    model = parallelize_model(model, context)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.003)
    reference_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.003)
    reference_parameters = dict(reference.named_parameters())
    results = []
    for step in range(3):
        # Each update starts from identical weights. The preceding update is checked
        # before alignment; this is a local-update check, not trajectory acceptance.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                reference_parameters[canonical_parameter_fqn(name)].copy_(full_value(parameter).to(device))
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        image_ids, images = image_batch(device)
        text = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]], device=device)
        ids = torch.cat((text, image_ids, torch.zeros(1, 1, device=device, dtype=torch.long)), 1)
        lengths = [7, image_ids.shape[1]]
        padded = [8, image_ids.shape[1] + 1]
        positions = torch.cat([torch.arange(n, device=device) for n in padded])[None]
        padding = torch.zeros_like(ids, dtype=torch.bool)
        padding[:, [7, ids.shape[1] - 1]] = True
        labels = ids.roll(-1, 1)
        labels[:, [6, 7, ids.shape[1] - 2, ids.shape[1] - 1]] = -100
        labels.masked_fill_((labels == 10) | (labels == 12) | (labels == 13), -100)
        if step == 1:
            labels[:, ids.shape[1] // 2 :] = -100
        if step == 2:
            labels.fill_(-100)
        count = labels.ne(-100).sum().clamp_min(1)
        pixels = [p.to(dtype).requires_grad_() for p in images]
        ref_pixels = [p.detach().clone().requires_grad_() for p in pixels]
        expected = reference(
            ids,
            pixel_values=ref_pixels,
            position_ids=positions,
            padding_mask=padding,
            qkv_format="thd",
            cu_seqlens=torch.tensor([0, padded[0], sum(padded)], device=device),
        ).logits
        batch = dict(
            input_ids=ids,
            pixel_values=pixels,
            labels=labels,
            position_ids=positions,
            seq_lens=torch.tensor([lengths], device=device),
            seq_lens_padded=torch.tensor([padded], device=device),
            qkv_format="thd",
        )
        sharder = ContextParallelSharder(model, context.device_mesh, batch)
        train_ctx, local = sharder.shard(batch)
        local_labels = local.pop("labels")
        with train_ctx():
            actual = model(**local).logits
            loss = F.cross_entropy(actual.flatten(0, 1), local_labels.flatten(), reduction="sum", ignore_index=-100)
            (loss * world / count).backward()
        ref_loss = F.cross_entropy(expected.flatten(0, 1), labels.flatten(), reduction="sum", ignore_index=-100) / count
        ref_loss.backward()
        combined = sharder.gather_token_tensor(actual.detach(), trim=True)
        record = {"step": step, "logits": error(combined, expected)}
        if dtype == torch.float32:
            torch.testing.assert_close(combined, expected, atol=2e-6, rtol=2e-5)
        else:
            assert record["logits"]["relative_l2"] <= 0.005, record
        loss = loss.detach() / count
        dist.all_reduce(loss)
        torch.testing.assert_close(loss, ref_loss.detach(), atol=2e-6, rtol=2e-5)
        grad_diff, grad_norm = 0.0, 0.0
        for name, parameter in model.named_parameters():
            ref = reference_parameters[canonical_parameter_fqn(name)]
            if not parameter.requires_grad:
                assert parameter.grad is None and ref.grad is None
                continue
            assert parameter.grad is not None and ref.grad is not None, name
            a, b = full_value(parameter.grad).float(), ref.grad.detach().float().cpu()
            assert torch.isfinite(a).all(), name
            if dtype == torch.float32:
                torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4, msg=name)
            grad_diff += float((a - b).square().sum())
            grad_norm += float(b.square().sum())
        record["gradient_relative_l2"] = (grad_diff / max(grad_norm, 1e-24)) ** 0.5
        assert record["gradient_relative_l2"] <= (0.005 if dtype == torch.bfloat16 else 1e-4), record
        record["pixels"] = []
        for pixel, ref_pixel in zip(pixels, ref_pixels, strict=True):
            a = pixel.grad.detach().float().clone()
            dist.all_reduce(a)
            a /= world
            item = error(a, ref_pixel.grad)
            record["pixels"].append(item)
            if dtype == torch.float32:
                torch.testing.assert_close(a, ref_pixel.grad, atol=2e-5, rtol=1e-4)
            else:
                assert item["relative_l2"] <= 0.005, item
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in model.named_parameters():
            ref = reference_parameters[canonical_parameter_fqn(name)]
            a, b = full_value(parameter).float(), ref.detach().float().cpu()
            torch.testing.assert_close(
                a, b, atol=2e-6 if dtype == torch.float32 else 0.003, rtol=2e-5 if dtype == torch.float32 else 0.04
            )
        results.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(
        json.dumps(
            dict(
                passed=True,
                attention_backend=args.attn,
                reference_backend=args.attn if args.reference_attn == "same" else args.reference_attn,
                cp_size=world,
                pp_size=1,
                dtype=args.dtype,
                activation_checkpointing=args.ac,
                reference_weights_aligned_each_step=True,
                cases=results,
            ),
            indent=2,
        )
        + "\n"
    )
    dist.destroy_process_group()


def validate_nonpacked(args: argparse.Namespace) -> None:
    """Compare CP1 with distributed image/text execution and optimizer steps."""
    torch.set_num_threads(1)
    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if not args.cpu:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("gloo" if args.cpu else "nccl", timeout=timedelta(minutes=5))
    world, rank = dist.get_world_size(), dist.get_rank()
    model = make_pipeline_model(multimodal=True).to(device)
    if args.freeze_vision:
        model.vision_encoder.requires_grad_(False)
    reference = copy.deepcopy(model)
    if args.cpu:
        if args.ep_size != 1:
            raise ValueError("CPU transport test uses EP1; use CUDA for FSDP/EP composition")
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("cp",))
        model.cp_mesh = mesh["cp"]
        context = None
        if args.ac:
            for i, block in enumerate(model.model.layers):
                model.model.layers[i] = checkpoint_wrapper(block)
    else:
        setup = DistributedSetup.build(
            strategy=FSDP2Config(
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32
                ),
                defer_fsdp_grad_sync=False,
            ),
            parallelism_sizes=ParallelismSizes(cp_size=world, ep_size=args.ep_size),
            activation_checkpointing=args.ac,
            world_size=world,
        )
        context = setup.mesh_context
        mesh = context.device_mesh
        model = parallelize_model(model, context)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.003)
    ref_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.003)
    errors = dict(logits=0.0, loss=0.0, gradients=0.0, pixels=0.0, norm=0.0, update=0.0)
    for step, accumulation in enumerate((1, 2, 1)):
        optimizer.zero_grad(set_to_none=True)
        ref_optimizer.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            image_ids, pixels = image_batch(device)
            # One image-bearing document and one text-only document; the second
            # image crosses the CP2 boundary between global token positions 5/6.
            ids = torch.cat((image_ids, torch.arange(11, device=device)[None] % 8 + 1), 0)
            keep = torch.ones_like(ids, dtype=torch.bool)
            keep[0, 0] = False
            keep[1, -2:] = False
            labels = ids.clone()
            if step == 1:
                labels[:, 6:] = -100  # One CP rank owns no supervised targets.
            if step == 2:
                labels.fill_(-100)  # Both ranks must still enter all collectives.
            pixels = [(p * (1 + 0.02 * micro)).requires_grad_() for p in pixels]
            ref_pixels = [p.detach().clone().requires_grad_() for p in pixels]
            batch = dict(input_ids=ids, labels=labels, attention_mask=keep, pixel_values=pixels)
            if micro:
                batch["position_ids"] = torch.arange(ids.shape[1], device=device).flip(0)[None].expand_as(ids)
            expected = reference(**{**batch, "pixel_values": ref_pixels})
            sharder = ContextParallelSharder(model, mesh, dict(batch))
            train_ctx, local = sharder.shard(dict(batch))
            with train_ctx():
                actual = model(**local)
                (actual.loss / accumulation).backward()
            (expected.loss / accumulation).backward()
            start = rank * actual.logits.shape[1] if world > 1 else 0
            length = min(actual.logits.shape[1], ids.shape[1] - start)
            a, b = actual.logits[:, :length], expected.logits[:, start : start + length]
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5, msg=f"step={step} micro={micro}")
            errors["logits"] = max(errors["logits"], float((a - b).detach().abs().max()))
            loss = actual.loss.detach().clone()
            dist.all_reduce(loss)
            loss /= world
            torch.testing.assert_close(loss, expected.loss, atol=1e-6, rtol=1e-5)
            errors["loss"] = max(errors["loss"], float((loss - expected.loss.detach()).abs()))
            for pixel, ref_pixel in zip(pixels, ref_pixels, strict=True):
                grad = pixel.grad.clone()
                dist.all_reduce(grad)
                grad /= world
                torch.testing.assert_close(grad, ref_pixel.grad, atol=2e-5, rtol=1e-4)
                errors["pixels"] = max(errors["pixels"], float((grad - ref_pixel.grad).abs().max()))
        if args.cpu:
            for parameter in parameters:
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad /= world
            norm = torch.nn.utils.clip_grad_norm_(parameters, float("inf"), foreach=False)
        else:
            norm = scale_grads_and_clip_grad_norm(
                float("inf"),
                [model],
                moe_mesh=context.moe_mesh,
                ep_axis_name="ep" if args.ep_size > 1 else None,
                dp_group_size=world,
                foreach=False,
            )
        ref_params = dict(reference.named_parameters())
        for name, parameter in model.named_parameters():
            ref = ref_params[canonical_parameter_fqn(name)]
            if not parameter.requires_grad:
                assert parameter.grad is None and ref.grad is None
                continue
            assert parameter.grad is not None and ref.grad is not None, name
            actual_grad, expected_grad = full_value(parameter.grad), ref.grad.detach().cpu()
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=1e-4, msg=name)
            errors["gradients"] = max(errors["gradients"], float((actual_grad - expected_grad).abs().max()))
        ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), float("inf"), foreach=False)
        torch.testing.assert_close(torch.as_tensor(norm).float().cpu(), ref_norm.float().cpu(), atol=2e-5, rtol=1e-4)
        errors["norm"] = max(errors["norm"], abs(float(norm) - float(ref_norm)))
        clip = min(1.0, 0.25 / (float(norm) + 1e-6)) if step == 1 else 1.0
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(clip)
        if step == 1:
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25, foreach=False)
        optimizer.step()
        ref_optimizer.step()
        for name, parameter in model.named_parameters():
            a, b = full_value(parameter), ref_params[canonical_parameter_fqn(name)].detach().cpu()
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5, msg=name)
            errors["update"] = max(errors["update"], float((a - b).detach().abs().max()))
    result = dict(
        passed=True,
        rank=rank,
        cp_size=world,
        ep_size=args.ep_size,
        ac=args.ac,
        freeze_vision=args.freeze_vision,
        device=device.type,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32 if not args.cpu else None,
        errors=errors,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packed", action="store_true", help="Check packed THD CP instead of nonpacked CP")
    parser.add_argument("--kernel-only", action="store_true", help="Check native sparse MLA kernel and CP gradients")
    parser.add_argument("--ac", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Nonpacked mode only")
    parser.add_argument("--freeze-vision", action="store_true", help="Nonpacked mode only")
    parser.add_argument("--ep-size", type=int, default=1, help="Nonpacked mode only")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32", help="Packed mode only")
    parser.add_argument("--attn", choices=("sdpa", "tilelang"), default="sdpa", help="Packed mode only")
    parser.add_argument("--reference-attn", choices=("same", "sdpa"), default="same", help="Packed mode only")
    args = parser.parse_args()
    if args.kernel_only:
        validate_sparse_kernel(args.output)
    elif args.packed:
        validate_packed(args)
    else:
        validate_nonpacked(args)


if __name__ == "__main__":
    main()

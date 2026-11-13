#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distributed PP numerical regression for the full forty-layer CSA2 dependency graph.

Run with torchrun --standalone --nproc-per-node=2, optionally --checkpoint.
This component test isolates the pipeline boundary from FSDP/EP and native
checkpoint loading. It compares logits, every gradient, clipping and SGD against
the same unsplit model and CP topology, including multi-image/text-only
microbatches. Keeping CP fixed isolates pipeline state transfer from changes in
CP reduction order.
"""

from __future__ import annotations

import argparse
import copy
import os
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.datasets.vlm.pp_media import prepare_vlm_media_for_pp, stage_vlm_media_for_pp
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.deepseek_v41.cp import shard_cp_batch
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.deepseek_v41.test_vision_packing import _batch, _model, _sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--cuts", type=int, nargs="+", default=[14])
    parser.add_argument("--schedule", default="1f1b")
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--recv-pool", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(483)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    # Construct before distributed initialization: Engram tables are unsharded
    # in this PP-only test. Native PP/CP/EP coverage exercises owner rebinding.
    model = _model(full_schedule=True, engram=True).to(device)
    reference = copy.deepcopy(model)
    if args.checkpoint:
        apply_ac(reference, ignore_router=True)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=3))
    try:
        pp_size = dist.get_world_size() // args.cp_size
        mesh = init_device_mesh("cuda", (pp_size, args.cp_size), mesh_dim_names=("pp", "cp"))
        cp_mesh = mesh["cp"] if args.cp_size > 1 else None
        reference.cp_mesh = cp_mesh
        microbatches = max(2, pp_size)
        bounds = (0, *args.cuts, 40)
        if list(bounds) != sorted(set(bounds)):
            raise ValueError("Cuts must be strictly increasing between layers 0 and 40")
        modules = []
        for start, end in zip(bounds, bounds[1:]):
            names = [f"model.layers.{i}" for i in range(start, end)] + ["model.engram_hash"]
            if start == 0:
                names += [
                    "model.embed_tokens",
                    "model.vision",
                    "model.aligner",
                    "model.image_start",
                    "model.image_end",
                    "model.image_newline",
                ]
            if end == 40:
                names += ["model.norm", "lm_head"]
            modules.append(names)
        # Initialize every mesh communicator before P2P and CP collectives can
        # overlap, matching native recipe communication prewarming.
        for axis in ("pp", "cp"):
            dist.all_reduce(torch.zeros((), device=device), group=mesh[axis].get_group())
        torch.cuda.synchronize()
        total_tokens = 1

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """Normalize logits [batch, local_sequence, vocab] against labels [batch, local_sequence]."""
            return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum") / total_tokens

        def parallelize(part: torch.nn.Module, **kwargs) -> None:
            del kwargs
            part.cp_mesh = cp_mesh
            if part.model.embed_tokens is None:
                for name in ("image_start", "image_end", "image_newline"):
                    part.model.register_parameter(name, None)
            if args.checkpoint:
                apply_ac(part, ignore_router=True)

        pipeline = AutoPipeline(
            world_mesh=mesh,
            dp_axis_names=(),
            cp_axis_name="cp" if cp_mesh is not None else None,
            pp_schedule=args.schedule,
            module_fqns_per_model_part=modules,
            pp_recv_buffer_pool=args.recv_pool,
            pp_microbatch_size=1,
            pp_batch_size=microbatches,
            patch_inner_model=False,
            patch_causal_lm_model=False,
            scale_grads_in_schedule=False,
            defer_fsdp_grad_sync=False,
            device=device,
            dtype=torch.float32,
            pp_seq_len=80 // args.cp_size,
        )
        pipeline.build(model, loss_fn=loss_fn, parallelize_fn=parallelize)
        del model
        parameters = {name: parameter for part in pipeline.parts for name, parameter in part.named_parameters()}
        expected_parameters = dict(reference.named_parameters())
        optimizer = torch.optim.SGD(parameters.values(), lr=0.01)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        # Change image presence without changing shape, then change physical length.
        for images, length, packed in ((True, 80, True), (False, 80, True), (True, 96, True), (False, 80, False)):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            rows = []
            torch.manual_seed(77 + length)
            for microbatch in range(microbatches):
                grids = ((3, 5), (1, 1)) if images and microbatch == 0 else ()
                samples = [_sample(grids, prefix=3), _sample((), prefix=2), _sample(())]
                if packed:
                    _, row, _ = shard_cp_batch(None, None, _batch(samples, length=length), packed_alignment=2)
                else:
                    ids = (torch.arange(length) % 7 + 1).unsqueeze(0)
                    mask = torch.ones_like(ids)
                    if microbatch == 0:
                        mask[:, -3:] = 0
                    row = {
                        "input_ids": ids,
                        "labels": ids.masked_fill(mask == 0, -100),
                        "position_ids": torch.arange(length).expand_as(ids),
                        "attention_mask": mask,
                        "vision_token_types": torch.full_like(ids, -1),
                    }
                rows.append(
                    {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in row.items()}
                )
            total_tokens = sum(int((row["labels"] != -100).sum()) for row in rows)
            if cp_mesh is not None:
                rows = [shard_cp_batch(cp_mesh, None, dict(row), pad_multiple=2, packed_alignment=2)[1] for row in rows]
            expected_logits = []
            expected_losses = []
            for row in rows:
                inputs = {key: value for key, value in row.items() if key not in ("labels", "n_images_per_sample")}
                logits = reference(**inputs).logits
                loss = loss_fn(logits, row["labels"])
                loss.backward()
                expected_logits.append(logits.detach())
                expected_losses.append(loss.detach())
            reference_loss = torch.stack(expected_losses).sum()
            if cp_mesh is not None:
                # Match the pipeline replica reduction without turning unused
                # parameters into used zero gradients in text-only microbatches.
                dist.all_reduce(reference_loss, group=cp_mesh.get_group())
                ref_params = [p for p in expected_parameters.values() if p.requires_grad]
                used = torch.tensor([p.grad is not None for p in ref_params], device=device, dtype=torch.int32)
                dist.all_reduce(used, op=dist.ReduceOp.MAX, group=cp_mesh.get_group())
                for param, active in zip(ref_params, used.tolist()):
                    if active:
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)
                        dist.all_reduce(param.grad, group=cp_mesh.get_group())
            token_keys = (
                "input_ids",
                "labels",
                "position_ids",
                "attention_mask",
                "packed_seq_ids",
                "vision_token_types",
            )
            batch = {key: torch.cat([row[key] for row in rows], dim=0) for key in token_keys if key in rows[0]}
            if images:
                batch["pixel_values"] = rows[0]["pixel_values"]
                batch["image_grid_hws"] = rows[0]["image_grid_hws"]
                batch["n_images_per_sample"] = torch.tensor([2] + [0] * (microbatches - 1), device=device)
            prepare_vlm_media_for_pp(batch, batch_size=microbatches, n_microbatches=microbatches)
            ids, labels = batch.pop("input_ids"), batch.pop("labels")
            pipeline.update_seq_len(ids.shape[1])
            losses = [] if pipeline.info.has_last_stage else None
            with stage_vlm_media_for_pp(pipeline, pipeline.parts, batch):
                output = pipeline.step(
                    ids, target=labels if pipeline.info.has_last_stage else None, losses=losses, **batch
                )
            logit_error = torch.zeros((), device=device)
            if pipeline.info.has_last_stage:
                expected_output = torch.cat(expected_logits)
                actual_loss = torch.stack(losses).sum()
                if cp_mesh is not None:
                    dist.all_reduce(actual_loss, group=cp_mesh.get_group())
                logit_error = (output - expected_output).abs().max()
                torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual_loss, reference_loss, rtol=1e-5, atol=1e-6)
            dist.all_reduce(logit_error, op=dist.ReduceOp.MAX)
            grad_error = torch.zeros((), device=device)
            local_squared_norm = torch.zeros((), device=device, dtype=torch.float64)
            for name, parameter in parameters.items():
                expected_grad = expected_parameters[name].grad
                if cp_mesh is not None and expected_grad is not None:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad, group=cp_mesh.get_group())
                if expected_grad is None:
                    assert parameter.grad is None, name
                else:
                    assert parameter.grad is not None, name
                    torch.testing.assert_close(
                        parameter.grad, expected_grad, rtol=2e-4, atol=2e-6, msg=lambda message: f"{name}: {message}"
                    )
                    grad_error = torch.maximum(grad_error, (parameter.grad - expected_grad).abs().max())
                    local_squared_norm += parameter.grad.double().square().sum()
            dist.all_reduce(grad_error, op=dist.ReduceOp.MAX)
            dist.all_reduce(local_squared_norm, group=mesh["pp"].get_group())
            actual_norm = local_squared_norm.sqrt()
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
            torch.testing.assert_close(actual_norm.float(), expected_norm, rtol=1e-5, atol=1e-6)
            scale = (0.3 / (actual_norm + 1e-6)).clamp(max=1)
            for parameter in parameters.values():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale.to(parameter.grad.dtype))
            optimizer.step()
            ref_optimizer.step()
            for name, parameter in parameters.items():
                torch.testing.assert_close(parameter, expected_parameters[name], rtol=1e-5, atol=1e-6, msg=name)
            if dist.get_rank() == 0:
                print(
                    f"PASS cuts={args.cuts} cp={args.cp_size} checkpoint={args.checkpoint} "
                    f"packed={packed} images={images} length={length} "
                    f"loss={reference_loss.item():.9f} norm={actual_norm.item():.9f} "
                    f"logit_max_abs={logit_error.item():.3e} grad_max_abs={grad_error.item():.3e}",
                    flush=True,
                )
        dist.barrier()
    except Exception:
        traceback.print_exc()
        raise
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    with torch.compiler.set_stance("force_eager"):
        main()

#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU PP2 numerical regression for the full forty-layer CSA2 dependency graph.

Run with torchrun --standalone --nproc-per-node=2, optionally --checkpoint.
This component test isolates the pipeline boundary from FSDP/EP and native
checkpoint loading. It compares logits, every gradient, clipping and SGD against
the same unsplit model, including multi-image/text-only microbatches.
"""

from __future__ import annotations

import argparse
import copy
import os
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
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("This regression requires two GPUs")
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("pp",))
        total_tokens = 1

        def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum") / total_tokens

        def parallelize(part: torch.nn.Module, **kwargs) -> None:
            del kwargs
            if part.model.embed_tokens is None:
                for name in ("image_start", "image_end", "image_newline"):
                    part.model.register_parameter(name, None)
            if args.checkpoint:
                apply_ac(part, ignore_router=True)

        pipeline = AutoPipeline(
            world_mesh=mesh,
            dp_axis_names=(),
            cp_axis_name=None,
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            patch_inner_model=False,
            patch_causal_lm_model=False,
            scale_grads_in_schedule=False,
            defer_fsdp_grad_sync=False,
            device=device,
            dtype=torch.float32,
            pp_seq_len=80,
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
            for microbatch in range(2):
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
            expected_logits = []
            expected_losses = []
            for row in rows:
                inputs = {key: value for key, value in row.items() if key not in ("labels", "n_images_per_sample")}
                logits = reference(**inputs).logits
                loss = loss_fn(logits, row["labels"])
                loss.backward()
                expected_logits.append(logits.detach())
                expected_losses.append(loss.detach())
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
                batch["n_images_per_sample"] = torch.tensor([2, 0], device=device)
            prepare_vlm_media_for_pp(batch, batch_size=2, n_microbatches=2)
            ids, labels = batch.pop("input_ids"), batch.pop("labels")
            pipeline.update_seq_len(length)
            losses = [] if pipeline.info.has_last_stage else None
            with stage_vlm_media_for_pp(pipeline, pipeline.parts, batch):
                output = pipeline.step(
                    ids, target=labels if pipeline.info.has_last_stage else None, losses=losses, **batch
                )
            if pipeline.info.has_last_stage:
                torch.testing.assert_close(output, torch.cat(expected_logits), rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    torch.stack(losses).sum(), torch.stack(expected_losses).sum(), rtol=1e-5, atol=1e-6
                )
            local_squared_norm = torch.zeros((), device=device, dtype=torch.float64)
            for name, parameter in parameters.items():
                expected_grad = expected_parameters[name].grad
                if expected_grad is None:
                    assert parameter.grad is None, name
                else:
                    assert parameter.grad is not None, name
                    torch.testing.assert_close(parameter.grad, expected_grad, rtol=2e-4, atol=2e-6, msg=name)
                    local_squared_norm += parameter.grad.double().square().sum()
            dist.all_reduce(local_squared_norm)
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
                    f"PASS checkpoint={args.checkpoint} packed={packed} images={images} length={length} "
                    f"loss={torch.stack(expected_losses).sum().item():.9f} norm={actual_norm.item():.9f}",
                    flush=True,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with torch.compiler.set_stance("force_eager"):
        main()

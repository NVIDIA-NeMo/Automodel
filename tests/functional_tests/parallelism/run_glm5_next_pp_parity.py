# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-GPU GLM PP transport/parity test; run with torchrun --nproc_per_node=2.

Uses the native splitter, 1F1B schedule, VLM media staging and gradient scaling.
This is random-initialized model parity, not checkpoint-loading validation.
"""

import argparse
import copy
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.datasets.vlm.pp_media import prepare_vlm_media_for_pp, stage_vlm_media_for_pp
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
from nemo_automodel.components.moe.parallelizer import apply_ac
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config


def main() -> None:
    """Compare [B,S,V] outputs, every gradient, global norm and SGD updates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (dist.get_world_size(), 1), mesh_dim_names=("pp", "dp"))
    torch.manual_seed(103)
    config = tiny_glm5_next_config()
    config.text_config.linear_head_dim = 32
    config.text_config.hc_mult = 4
    model = Glm5NextForConditionalGeneration(config, backend=tiny_backend(adapter=False))
    model.initialize_weights(torch.device("cpu"), dtype=torch.bfloat16)
    model.to(device)
    reference = copy.deepcopy(model)

    def loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Sum CE for logits [B,S,V] and labels [B,S], including ignored labels."""
        return F.cross_entropy(output.float().flatten(0, 1), target.flatten(), reduction="sum")

    def parallelize(stage: Glm5NextForConditionalGeneration, *_args, **_kwargs) -> None:
        """Use native decoder checkpointing without changing parameter ownership."""
        if args.activation_checkpointing:
            apply_ac(stage, ignore_router=True)

    pp = AutoPipeline(
        world_mesh=mesh,
        moe_mesh=None,
        pp_axis_name="pp",
        dp_axis_names=("dp",),
        pp_schedule="1f1b",
        pp_microbatch_size=1,
        pp_batch_size=4,
        device=device,
        dtype=torch.bfloat16,
        pp_seq_len=args.seq_len,
        scale_grads_in_schedule=False,
    ).build(model, loss_fn=loss_fn, parallelize_fn=parallelize)
    part = pp.parts[0]
    assert (part.model.visual is not None) == pp.info.has_first_stage
    assert (part.language_model.embed_tokens is not None) == pp.info.has_first_stage
    assert (part.lm_head is not None) == pp.info.has_last_stage
    assert (part.language_model.norm is not None) == pp.info.has_last_stage
    optimizer = torch.optim.SGD(part.parameters(), lr=0.125)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.125)
    captured = []
    handle = part.register_forward_hook(lambda _m, _a, output: captured.append(output.detach().clone()))
    for iteration, length in enumerate((args.seq_len, args.seq_len + 4)):
        torch.manual_seed(307 + iteration)
        ids = torch.randint(1, 60, (4, length), device=device)
        docs = torch.stack(
            [
                torch.cat((torch.ones(boundary), torch.full((length - boundary,), 2)))
                for boundary in (length // 2, length // 3, length // 2 - 1, length // 2 + 1)
            ]
        ).to(device=device, dtype=torch.int32)
        # The second schedule has no media, detecting stale stage-zero state.
        pixels = torch.randn(8, 24, device=device, dtype=torch.bfloat16) if iteration == 0 else None
        grid = torch.tensor([[1, 2, 2], [1, 2, 2]], device=device)
        if pixels is not None:
            ids[0, 1] = ids[2, 1] = 63
        labels = ids.roll(-1, dims=1)
        labels[:, -1] = -100
        labels[0, 4:6] = -100
        labels[1, 2:] = -100
        labels[3] = -100
        count = int((labels != -100).sum())
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        expected_outputs = []
        reference_loss = torch.zeros((), device=device)
        for row in range(4):
            media = {}
            if pixels is not None and row in (0, 2):
                start = 0 if row == 0 else 4
                media = {"pixel_values": pixels[start : start + 4], "image_grid_thw": grid[:1]}
            output = reference(ids[row : row + 1], attention_mask=docs[row : row + 1], **media).logits
            expected_outputs.append(output.detach())
            loss = loss_fn(output, labels[row : row + 1])
            reference_loss += loss.detach()
            loss.backward()
        batch = {"input_ids": ids.clone(), "labels": labels.clone(), "_packed_seq_ids": docs}
        if pixels is not None:
            batch.update(
                pixel_values=pixels, image_grid_thw=grid, n_images_per_sample=torch.tensor([1, 0, 1, 0], device=device)
            )
            prepare_vlm_media_for_pp(batch, batch_size=4, n_microbatches=4)
        sharder = part.prepare_model_inputs_for_cp(batch, num_chunks=4)["cp_sharder"]
        _, batch, _ = sharder.shard_batch(None, None, batch)
        targets = batch.pop("labels")
        inputs = batch.pop("input_ids")
        pp.update_seq_len(length)
        losses = [] if pp.info.has_last_stage else None
        captured.clear()
        with stage_vlm_media_for_pp(pp, pp.parts, batch):
            pp.info.schedule.step(
                *((inputs,) if pp.info.has_first_stage else ()),
                target=targets if pp.info.has_last_stage else None,
                losses=losses,
                **batch,
            )
        assert part._vlm_chunk_idx is None
        if pp.info.has_last_stage:
            assert len(captured) == 4
            torch.testing.assert_close(torch.cat(captured), torch.cat(expected_outputs), rtol=0, atol=0)
            torch.testing.assert_close(torch.stack(losses).sum(), reference_loss)
        else:
            assert all(value.shape == (1, length, 4, 16) for value in captured)
        for parameter in reference.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(count)
        # Independent sum-of-squares oracle: torch's default per-parameter
        # BF16 norms lose precision and would change the next step's weights.
        expected_norm = (
            torch.stack(
                [
                    parameter.grad.double().square().sum()
                    for parameter in reference.parameters()
                    if parameter.grad is not None
                ]
            )
            .sum()
            .sqrt()
            .float()
        )
        coefficient = (0.5 / (expected_norm + 1e-6)).clamp(max=1.0)
        for parameter in reference.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)
        actual_norm = scale_grads_and_clip_grad_norm(
            0.5,
            pp.parts,
            pp_enabled=True,
            device_mesh=mesh,
            pp_axis_name="pp",
            num_label_tokens=count,
            dp_group_size=1,
        )
        torch.testing.assert_close(actual_norm.float(), expected_norm.float(), rtol=1e-6, atol=1e-6)
        expected_parameters = dict(reference.named_parameters())
        maximum_gradient_error = 0.0
        for name, parameter in part.named_parameters():
            expected = expected_parameters[name.replace("_checkpoint_wrapped_module.", "")]
            assert (parameter.grad is None) == (expected.grad is None), name
            if parameter.grad is not None:
                maximum_gradient_error = max(
                    maximum_gradient_error, float((parameter.grad.float() - expected.grad.float()).abs().max())
                )
                torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-4, atol=1e-6, msg=name)
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in part.named_parameters():
            torch.testing.assert_close(
                parameter,
                expected_parameters[name.replace("_checkpoint_wrapped_module.", "")],
                rtol=0.02,
                atol=5e-4,
                msg=name,
            )
        print(
            f"PASS rank={dist.get_rank()} iteration={iteration} seq={length} microbatches=4 ac={args.activation_checkpointing} "
            f"loss={float(reference_loss) / count:.8f} norm={float(actual_norm):.8f} reference_norm={float(expected_norm):.8f} "
            f"max_grad_error={maximum_gradient_error:.8g}",
            flush=True,
        )
    handle.remove()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

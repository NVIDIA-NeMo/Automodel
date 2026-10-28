#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Muse's model-owned CP/PP boundary with real TE and interleaved PP.

Default: torchrun --standalone --nproc-per-node=2 <this file>
Composed topology (manual/scheduled): add --cp-size=2 with four ranks.
Both runs exercise two virtual chunks per rank, ordinary/packed transitions at
the same input length, changed lengths, backward, and SGD against a non-pipelined
TE model with matching CP. An unsharded FP32 SDPA reference independently measures
the native backend/CP reference discrepancy; PP must not increase that difference.
This test establishes PP parity at a fixed CP degree, not CP1-vs-CP2 parity.
No recipe-specific model setters are used.
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

from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.distributed.context_parallel.sharder import round_robin_local_indices
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.models.muse_spark.model import MuseSpark1xOmniForCausalLM
from nemo_automodel.components.moe.parallelizer import apply_cp
from tests.unit_tests.models.muse_spark.conftest import tiny_backend, tiny_muse_spark_config


def main() -> None:
    """Compare PP outputs, every active parameter gradient, norm, and SGD update."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp-size", type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    try:
        if dist.get_world_size() != 2 * args.cp_size:
            raise ValueError("Expected PP2 x CP ranks")
        mesh = init_device_mesh("cuda", (2, args.cp_size), mesh_dim_names=("pp", "cp"))
        cp_mesh = mesh["cp"]
        torch.manual_seed(20260916)
        config = tiny_muse_spark_config()
        config.hidden_size = 128
        config.head_dim = 32
        config.torch_dtype = torch.bfloat16
        config.use_fp32_for_output = True
        # Random-init Muse starts its final norm at zero; use a nonzero gain
        # here so the test exercises gradients through every decoder layer.
        config.output_norm_gain_center_type = "one"
        config.moe["num_experts"] = 2  # topk=2: all experts, stable across precision.
        backend = tiny_backend(adapter=False)
        backend.attn = "te"
        model = MuseSpark1xOmniForCausalLM(config, backend=backend).to(device)
        model.initialize_weights(device, dtype=torch.bfloat16)
        model.lm_head.float()  # Honor use_fp32_for_output without the FSDP wrapper.
        with torch.no_grad():
            for layer in model.model.layers:
                layer.add_attn.gate.fill_(0.125)
                layer.add_ffn.gate.fill_(0.125)
        initial = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
            if not name.endswith("._extra_state")
        }
        ref_config = copy.deepcopy(config)
        ref_config.torch_dtype = torch.float32
        reference = MuseSpark1xOmniForCausalLM(ref_config, backend=tiny_backend(adapter=False)).to(device)
        reference.load_state_dict(initial)
        reference.train()
        reference_parameters = dict(reference.named_parameters())
        native_reference = copy.deepcopy(model)
        if args.cp_size > 1:
            apply_cp(native_reference, cp_mesh)
        native_parameters = dict(native_reference.named_parameters())
        total_tokens = 1

        def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute a local contribution to the globally normalized loss.

            Args:
                logits: Local predictions [batch, local_sequence, vocab].
                target: Local labels [batch, local_sequence], with -100 ignored.

            Returns:
                Scalar loss normalized by the global valid-token count.
            """
            return (
                F.cross_entropy(
                    logits.float().reshape(-1, config.vocab_size),
                    target.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                / total_tokens
            )

        def parallelize(part: torch.nn.Module, **kwargs) -> None:
            """Configure each actual model chunk with the standard CP parallelizer."""
            del kwargs
            if args.cp_size > 1:
                apply_cp(part, cp_mesh)

        pipeline = AutoPipeline(
            world_mesh=mesh,
            dp_axis_names=(),
            cp_axis_name="cp",
            pp_schedule="interleaved1f1b",
            pp_microbatch_size=1,
            pp_batch_size=4,
            layers_per_stage=1,
            patch_inner_model=False,
            patch_causal_lm_model=False,
            defer_fsdp_grad_sync=True,
            device=device,
            dtype=torch.float32,
            pp_seq_len=16,
        )
        pipeline.build(model, loss_fn=loss_fn, parallelize_fn=parallelize)
        del model
        assert len(pipeline.parts) == 2
        parameters = {name: parameter for part in pipeline.parts for name, parameter in part.named_parameters()}
        learning_rate = 0.001
        optimizer = torch.optim.SGD(parameters.values(), lr=learning_rate)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=learning_rate)
        native_optimizer = torch.optim.SGD(native_reference.parameters(), lr=learning_rate)
        # Same-length transitions specifically catch stale per-chunk mode state.
        for packed, length in ((True, 16), (False, 16), (True, 16), (False, 32), (True, 32)):
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(initial[name])
            reference.load_state_dict(initial)
            with torch.no_grad():
                for name, parameter in native_parameters.items():
                    parameter.copy_(initial[name])
            native_reference.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            ids = (torch.arange(4 * length, device=device).reshape(4, length) % 48) + 1
            labels = (ids + 3) % config.vocab_size
            positions = torch.arange(length, device=device).expand(4, -1)
            batch = {"input_ids": ids, "labels": labels, "position_ids": positions}
            if packed:
                # Distinct documents and inter-document padding, all fitting the
                # sliding window. Padding must not become supervised content.
                docs = length // 8
                real = torch.full((4, docs), 7, device=device, dtype=torch.long)
                real[::2, 0] = 6
                padded = torch.full_like(real, 8)
                labels = labels.clone()
                for row in range(4):
                    for doc in range(docs):
                        labels[row, doc * 8 + int(real[row, doc]) : (doc + 1) * 8] = -100
                positions = (torch.arange(length, device=device) % 8).expand(4, -1)
                batch.update(
                    labels=labels, position_ids=positions, seq_lens=real, seq_lens_padded=padded, qkv_format="thd"
                )
            total_tokens = int((labels != -100).sum())
            if dist.get_rank() == 0:
                print(f"RUN cp={args.cp_size} packed={packed} length={length}", flush=True)
            # Independent reference: each packed document is an ordinary causal
            # SDPA call; no TE partition or packed preprocessing is reused.
            reference_logits = torch.zeros(4, length, config.vocab_size, device=device)
            for row in range(4):
                spans = [(d * 8, int(real[row, d])) for d in range(length // 8)] if packed else [(0, length)]
                for start, size in spans:
                    logits = reference(input_ids=ids[row : row + 1, start : start + size]).logits
                    loss_fn(logits, labels[row : row + 1, start : start + size]).backward()
                    reference_logits[row, start : start + size] = logits.detach().squeeze(0)
            # Measure the native backend/CP-vs-FP32 error independently of PP.
            # This gives each parameter its own measured reference allowance, rather than
            # relaxing all gradient tolerances to the worst cancellation case.
            native_batch = {
                key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()
            }
            native_sharder = ContextParallelSharder(native_reference, mesh, native_batch)
            native_ctx, native_batch = native_sharder.shard(native_batch)
            native_labels = native_batch.pop("labels")
            with native_ctx():
                native_logits = native_reference(**native_batch).logits
                loss_fn(native_logits, native_labels).backward()
            native_logits = native_logits.detach().reshape(4, length // args.cp_size, config.vocab_size)
            if args.cp_size > 1:
                for parameter in native_parameters.values():
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, group=cp_mesh.get_group())
            sharder = ContextParallelSharder(
                pipeline.parts[0],
                mesh,
                batch,
                num_chunks=4,
                padding_token_id=0,
            )
            train_ctx, local_batch = sharder.shard(batch)
            local_ids = local_batch.pop("input_ids")
            local_labels = local_batch.pop("labels")
            torch.testing.assert_close(local_ids, ids)
            pipeline.update_seq_len(local_ids.shape[1])
            losses = [] if pipeline.info.has_last_stage else None
            with train_ctx():
                output = pipeline.step(
                    local_ids,
                    target=local_labels if pipeline.info.has_last_stage else None,
                    losses=losses,
                    **local_batch,
                )
            if pipeline.info.has_last_stage:
                if packed:
                    indices = local_batch["_thd_local_indices"].long()
                else:
                    indices = round_robin_local_indices(cp_mesh, length, device=device).expand(4, -1)
                expected = reference_logits.gather(1, indices.unsqueeze(-1).expand(-1, -1, config.vocab_size))
                valid = local_labels != -100
                native_expected = native_logits
                torch.testing.assert_close(output[valid], native_expected[valid], atol=0.003, rtol=0.02)
                native_reference_error = (native_expected[valid] - expected[valid]).norm()
                assert (output[valid] - expected[valid]).norm() <= (
                    native_reference_error + 0.02 * expected[valid].norm()
                )
            squared_norm = torch.zeros((), device=device)
            worst_gradient_error = 0.0
            active_parameters = 0
            for name, parameter in parameters.items():
                ref_grad = reference_parameters[name].grad
                if ref_grad is None:
                    assert parameter.grad is None, name
                    continue
                assert parameter.grad is not None, name
                if args.cp_size > 1:
                    dist.all_reduce(parameter.grad, group=cp_mesh.get_group())
                gradient = parameter.grad.float()
                assert torch.isfinite(gradient).all(), name
                native_gradient = native_parameters[name].grad.float()
                native_error = (gradient - native_gradient).norm()
                native_scale = native_gradient.norm().clamp_min(1e-8)
                relative_error = float(native_error / native_scale)
                assert native_error <= 0.03 * native_scale + 1e-7, (name, relative_error)
                reference_error = (native_gradient - ref_grad).norm()
                assert (gradient - ref_grad).norm() <= (reference_error + 0.03 * ref_grad.norm() + 1e-7), name
                if float(native_scale) > 1e-7:
                    worst_gradient_error = max(worst_gradient_error, relative_error)
                squared_norm += gradient.square().sum()
                active_parameters += 1
            dist.all_reduce(squared_norm, group=mesh["pp"].get_group())
            ref_norm = (
                torch.stack([p.grad.square().sum() for p in reference.parameters() if p.grad is not None]).sum().sqrt()
            )
            native_norm = (
                torch.stack(
                    [p.grad.float().square().sum() for p in native_reference.parameters() if p.grad is not None]
                )
                .sum()
                .sqrt()
            )
            torch.testing.assert_close(squared_norm.sqrt(), native_norm, atol=1e-5, rtol=0.02)
            assert (squared_norm.sqrt() - ref_norm).abs() <= ((native_norm - ref_norm).abs() + 0.02 * ref_norm)
            optimizer.step()
            reference_optimizer.step()
            native_optimizer.step()
            changed = 0
            for name, parameter in parameters.items():
                actual = parameter.float()
                native = native_parameters[name].float()
                fp32 = reference_parameters[name]
                update_scale = (native - initial[name].float()).norm()
                # BF16 SGD rounds its stored weights. Account for that rounding
                # alongside the gradient tolerance already checked above.
                rounding = torch.finfo(parameter.dtype).eps * native.norm()
                allowed_error = 0.03 * update_scale + rounding + 1e-7
                assert (actual - native).norm() <= allowed_error, name
                assert (actual - fp32).norm() <= (native - fp32).norm() + allowed_error, name
                changed += not torch.equal(parameter, initial[name])
            assert changed > 0
            for part in pipeline.parts:
                for layer in part.model.layers.values():
                    if args.cp_size > 1:
                        expected_transport = "p2p" if packed or layer.self_attn.sliding_window is None else "all_gather"
                        assert layer.self_attn.attn_module.cp_comm_type == expected_transport
            evidence = torch.tensor([worst_gradient_error, active_parameters], device=device)
            dist.all_reduce(evidence, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                print(
                    f"PASS cp={args.cp_size} pp=2 chunks/rank=2 packed={packed} length={length} "
                    f"max_grad_rel_l2={evidence[0].item():.6f} active_params/rank={int(evidence[1])}",
                    flush=True,
                )
        dist.barrier()
        if dist.get_rank() == 0:
            print("MUSE MODEL-OWNED CP/PP: PASS", flush=True)
    except Exception:
        # Print before group teardown, which can wait for a peer's NCCL timeout.
        traceback.print_exc()
        raise
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

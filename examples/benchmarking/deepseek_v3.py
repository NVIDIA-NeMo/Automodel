# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.optim import Adam

# from transformer_engine.pytorch.optimizers import FusedAdam
from transformers import AutoConfig

from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.pipelining.functional import pipeline_model
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM
from nemo_automodel.components.models.deepseek_v3.parallelizer import parallelize_model
from nemo_automodel.components.moe.utils import BackendConfig
from nemo_automodel.components.training.timers import Timers
from nemo_automodel.components.utils.flops_utils import deepseekv3_flops

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    initialize_distributed(backend="nccl", timeout_minutes=10)

    parser = argparse.ArgumentParser(description="Multinode PP DSv3 training script")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--pp", type=int, default=4, help="Pipeline parallelism degree")
    parser.add_argument("--ep", type=int, default=2, help="Expert parallelism degree")
    parser.add_argument("--dp-shard", type=int, default=2, help="Data parallel shard degree")
    parser.add_argument("--dp-replicate", type=int, default=1, help="Data parallel replicate degree")
    parser.add_argument("--model-id", type=str, default="moonshotai/Moonlight-16B-A3B", help="HuggingFace model ID")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--global-batch-size", type=int, default=64, help="Global batch size")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--local-batch-size", type=int, default=None, help="Local batch size per pipeline stage")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size per pipeline stage")
    parser.add_argument("--nsys-start", type=int, default=-1, help="Iteration to start nsys profiling")
    parser.add_argument("--nsys-end", type=int, default=-1, help="Iteration to end nsys profiling")
    parser.add_argument("--iters", type=int, default=30, help="Number of iterations")
    parser.add_argument("--use-fake-gate", action="store_true", default=False, help="Use fake gate")
    args = parser.parse_args()

    # Assuming 8 GPUs per node
    world_size = args.num_nodes * 8

    if args.local_batch_size is None:
        args.local_batch_size = args.pp

    assert args.local_batch_size // args.micro_batch_size >= args.pp, (
        f"local_batch_size // micro_batch_size must be greater than or equal to pp * 2, but got {args.local_batch_size} // {args.micro_batch_size} < {args.pp} * 2"
    )

    # Initialize timers
    timers = Timers(log_level=2, log_option="minmax")

    # Setup phase with timer
    with timers("setup", log_level=1):
        # Build meshes directly
        pp = args.pp
        dp_replicate = args.dp_replicate
        dp_shard = args.dp_shard
        cp = 1
        tp = 1
        ep = args.ep

        # Create default mesh
        mesh = {}
        mesh["default"] = DeviceMesh(
            device_type="cuda",
            mesh=torch.arange(world_size).reshape(pp, dp_replicate, dp_shard, cp, tp),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp"),
        )

        # Flatten dp_shard and cp dimensions if needed
        dp_shard_cp_mesh_dim_names = []
        if dp_shard > 1:
            dp_shard_cp_mesh_dim_names.append("dp_shard")
        if cp > 1:
            dp_shard_cp_mesh_dim_names.append("cp")
        if dp_shard_cp_mesh_dim_names:
            mesh["default"][tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")

        # Create MoE mesh if ep > 1
        if ep > 1:
            # Calculate dp_shard_with_ep
            dp_shard_with_ep = dp_shard // ep
            mesh["moe"] = DeviceMesh(
                device_type="cuda",
                mesh=torch.arange(world_size).reshape(pp, dp_shard_with_ep, ep),
                mesh_dim_names=("pp", "dp_shard_with_ep", "ep"),
            )

        rank = torch.distributed.get_rank()

        # Seed RNGs per DP rank so random tokens differ across DP ranks and are deterministic per-rank
        seed = 1234
        dp_rank = None
        default_mesh = mesh["default"]
        dp_replicate_enabled = dp_replicate > 1
        dp_axis_names = ("dp_replicate", "dp_shard") if dp_replicate_enabled else ("dp_shard",)
        dp_group = default_mesh[dp_axis_names].get_group()
        dp_rank = torch.distributed.get_rank(dp_group)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

        if dp_rank is None:
            dp_rank = rank

        config = AutoConfig.from_pretrained(args.model_id)
        if args.num_layers is not None:
            config.num_hidden_layers = args.num_layers

        backend = BackendConfig()
        backend.attn = "te"
        backend.linear = "te"
        backend.rms_norm = "te"
        backend.enable_deepep = True
        backend.fake_balanced_gate = args.use_fake_gate

        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        if rank == 0:
            print(f"Rank {rank} | Config: {config}")

        flops = deepseekv3_flops(config, gbs=args.global_batch_size, seq_len=args.seq_len)
        tflops = flops / (10**12)
        if rank == 0:
            print(f"TFLOPs/GPU: {tflops:.6f}")

        with torch.device("meta"):
            model = DeepseekV3ForCausalLM(config, backend=backend)

        # Simple CE loss used by the last stage
        def ce_loss_fn(pred, labels):
            return F.cross_entropy(pred.view(-1, pred.size(-1)), labels.view(-1))

        pp_schedule, model_parts, has_first_stage, has_last_stage, stages = pipeline_model(
            model,
            world_mesh=mesh["default"],
            moe_mesh=mesh["moe"] if "moe" in mesh else None,
            pp_axis_name="pp",
            dp_axis_names=("dp_replicate", "dp_shard") if dp_replicate_enabled else ("dp_shard",),
            cp_axis_name=None,
            tp_axis_name=None,
            ep_axis_name="ep",
            ep_shard_axis_names=("dp_shard_with_ep",) if ep > 1 else None,
            layers_per_stage=2,
            pipeline_parallel_schedule_csv=None,
            pipeline_parallel_schedule="interleaved1f1b",
            parallelize_fn=parallelize_model,
            microbatch_size=args.micro_batch_size,
            local_batch_size=args.local_batch_size,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
            loss_fn=ce_loss_fn,
            patch_inner_model=False,
            patch_causal_lm_model=False,
            round_to_pp_multiple="up",
            # module_fqns_per_model_part=[
            #     ["embed_tokens", "layers.0", "layers.1"],
            #     ["layers.2", "layers.3", "norm", "lm_head"],
            # ],
        )

        # Allocate parameters for each stage on its GPU and init
        device = torch.cuda.current_device()
        for mp in model_parts:
            to_empty_parameters_only(mp, device=device, dtype=torch.bfloat16)
            mp.initialize_weights(buffer_device=torch.device(f"cuda:{device}"))
            mp.train()

        optimizer = Adam(
            [
                {
                    "params": mp.parameters(),
                    "name": f"rank_{rank}_stage_{i}",
                }
                for i, mp in enumerate(model_parts)
            ],
            lr=1e-4,
        )

    # Log setup time
    rank = torch.distributed.get_rank()
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    # Training loop over dummy data
    for i in range(args.iters):
        if i == args.nsys_start and torch.distributed.get_rank() in (0, torch.distributed.get_world_size() - 1):
            print(f"Rank {rank} | Starting nsys profiling")
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        dp_size = args.dp_shard * args.dp_replicate
        ga_steps = args.global_batch_size // (args.local_batch_size * dp_size)
        if rank == 0:
            print(
                f"Rank {rank} | Iteration {i} | GA steps={ga_steps} | dp_size={args.dp_shard * args.dp_replicate} | local_bs={args.local_batch_size} | global_bs={args.global_batch_size}"
            )
        optimizer.zero_grad()
        # Time the entire iteration
        with timers("iteration", log_level=1):
            for _ga_step_idx in range(ga_steps):
                tokens = torch.randint(0, config.vocab_size, (args.local_batch_size, args.seq_len), device=device)
                labels = torch.cat(
                    [tokens[:, 1:], torch.full((args.local_batch_size, 1), -100, device=device, dtype=tokens.dtype)],
                    dim=1,
                )
                padding_mask = None
                position_ids = (
                    torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
                )

                torch.cuda.nvtx.range_push(f"iteration_{i}_ga_step_{_ga_step_idx}")
                # Run one forward+backward pipeline step
                targets, losses = (labels, []) if has_last_stage else (None, None)

                # Time the actual pipeline step
                with timers(f"forward_backward_{_ga_step_idx}", log_level=2):
                    if has_first_stage:
                        pp_schedule.step(
                            tokens, target=targets, losses=losses, position_ids=position_ids, padding_mask=padding_mask
                        )
                    else:
                        pp_schedule.step(
                            target=targets, losses=losses, position_ids=position_ids, padding_mask=padding_mask
                        )

                if has_last_stage and dp_rank == 0:
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                    print(
                        f"Rank {rank} | Iteration {i} | GA step {_ga_step_idx} | Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB | loss={loss.detach().item():.4f}"
                    )
                else:
                    ...
                torch.cuda.nvtx.range_pop()

            with timers("optimizer", log_level=2):
                optimizer.step()
                logger.debug("Optimizer step")

            # Log timing every iteration
            if i % 1 == 0:  # Log every iteration
                max_iter_time = timers._get_global_min_max_time(
                    ["iteration"], reset=False, barrier=False, normalizer=1.0
                )["iteration"][1]
                if rank == 0:
                    mfu = tflops / (torch.distributed.get_world_size() * max_iter_time)
                    mfu = mfu * 2 / 1979.0
                    print(f"\nMFU: {mfu * 100:.6f}%")

                timers.log(
                    names=["iteration", "optimizer"]
                    + [f"forward_backward_{_ga_step_idx}" for _ga_step_idx in range(ga_steps)],
                    rank=0,  # Only log on rank 0
                    normalizer=1000.0,  # s
                    reset=True,  # Reset timers after logging
                    barrier=True,  # Synchronize before collecting times
                )

        if i == args.nsys_end and torch.distributed.get_rank() in (0, torch.distributed.get_world_size() - 1):
            print(f"Rank {rank} | Stopping nsys profiling")
            torch.cuda.cudart().cudaProfilerStop()

    # Final summary
    torch.distributed.barrier()
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Training Summary")
        print(f"{'=' * 60}")

    # Get active times for overall summary
    setup_time = timers._timers["setup"].active_time() if "setup" in timers._timers else 0
    iter_time = timers._timers["iteration"].active_time() if "iteration" in timers._timers else 0

    # Calculate average iteration time excluding first 10 warmup iterations
    total_iters = args.iters
    warmup_iters = 10

    if rank == 0:
        print(f"Total setup time: {setup_time:.2f} seconds")
        print(f"Total iteration time: {iter_time:.2f} seconds")

        # Only calculate average if we have more iterations than warmup
        if total_iters > warmup_iters:
            # Note: This assumes all iterations are timed equally, which isn't perfect
            # since we can't easily separate warmup timing from the total
            avg_iter_time = iter_time / total_iters
            iterations_averaged = total_iters - warmup_iters
            estimated_avg_without_warmup = (iter_time - (avg_iter_time * warmup_iters)) / iterations_averaged
            print(
                f"Average iteration time (excluding first {warmup_iters} warmup iterations): {estimated_avg_without_warmup:.3f} seconds"
            )
            print(f"Averaged over {iterations_averaged} iterations")
        else:
            print(f"Average iteration time: {iter_time / total_iters:.3f} seconds (all {total_iters} iterations)")
        print(f"{'=' * 60}\n")

    torch.distributed.destroy_process_group()

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

import logging

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from itertools import cycle
from nemo_automodel.components.datasets.llm.hellaswag import HellaSwag

from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.pipelining.functional import pipeline_model
from nemo_automodel.components.training.rng import init_all_rng
from nemo_automodel.components.training.timers import Timers
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def calculate_mfu(tflops, world_size, time_seconds, reference_mfu=1979.0):
    mfu = tflops / (world_size * time_seconds)
    mfu = mfu / reference_mfu
    return mfu * 100


def run_benchmark(cfg):
    """Run the benchmark with the given configuration."""
    initialize_distributed(backend="nccl", timeout_minutes=10)

    local_batch_size = cfg.training.get("local_batch_size", None)
    if local_batch_size is None:
        local_batch_size = cfg.distributed.pp_size
        cfg.training.local_batch_size = local_batch_size

    assert local_batch_size // cfg.pipelining.pp_microbatch_size >= cfg.distributed.pp_size, (
        f"local_batch_size // pp_microbatch_size must be greater than or equal to pp * 2, but got {local_batch_size} // {cfg.pipelining.pp_microbatch_size} < {cfg.distributed.pp} * 2"
    )

    # Initialize timers
    timers = Timers(log_level=2, log_option="minmax")

    # Setup phase with timer
    with timers("setup", log_level=1):
        # Build meshes directly
        dp_replicate_enabled = cfg.distributed.dp_replicate_size > 1
        dp_axis_names = ("dp_replicate", "dp_shard") if dp_replicate_enabled else ("dp_shard",)

        if cfg.distributed.get("dp_shard_size", None) is None:
            world_size = torch.distributed.get_world_size()
            # Calculate dp_size to ensure dp_size * tp_size * cp_size == world_size
            total_parallel_ranks = cfg.distributed.dp_replicate_size * cfg.distributed.pp_size
            if world_size % total_parallel_ranks != 0:
                raise ValueError(
                    f"world_size ({world_size}) must be divisible by (dp_replicate_size * pp_size) "
                    f"({cfg.distributed.dp_replicate_size} * {cfg.distributed.pp_size} = {total_parallel_ranks})"
                )
            cfg.distributed.dp_shard_size = world_size // total_parallel_ranks

        # Create default mesh
        mesh = {}
        mesh["default"] = init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                cfg.distributed.pp_size,
                cfg.distributed.dp_replicate_size,
                cfg.distributed.dp_shard_size,
                1,
                1,
            ),
            mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp"),
        )

        # Create MoE mesh if ep > 1
        if cfg.distributed.ep_size > 1:
            mesh["moe"] = init_device_mesh(
                device_type="cuda",
                mesh_shape=(
                    cfg.distributed.pp_size,
                    cfg.distributed.dp_shard_size // cfg.distributed.ep_size,
                    cfg.distributed.ep_size,
                ),
                mesh_dim_names=("pp", "ep_shard", "ep"),
            )

        rank = torch.distributed.get_rank()

        seed = cfg.training.seed
        default_mesh = mesh["default"]
        dp_group = default_mesh[dp_axis_names].get_group()
        dp_rank = torch.distributed.get_rank(dp_group)

        init_all_rng(seed, ranked=False)

        config = AutoConfig.from_pretrained(
            cfg.model.pretrained_model_name_or_path, trust_remote_code=cfg.model.get("trust_remote_code", False)
        )
        if cfg.model.num_layers is not None:
            config.num_hidden_layers = cfg.model.num_layers

        # Build tokenizer and dataset (HellaSwag)
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            trust_remote_code=cfg.model.get("trust_remote_code", False),
        )

        backend = cfg.model.backend.instantiate()
        kwargs = {
            "pretrained_model_name_or_path": config,
            "backend": backend,
        }
        if "num_layers" in cfg.model:
            del cfg.model.__dict__["num_layers"]
        if "trust_remote_code" in cfg.model:
            del cfg.model.__dict__["trust_remote_code"]

        with torch.device("meta"):
            model = cfg.model.instantiate(**kwargs)

        # Prepare HellaSwag dataset and dataloader
        hellaswag = HellaSwag(
            path_or_dataset="rowan/hellaswag",
            tokenizer=tokenizer,
            split="train",
        )
        hellaswag_ds = hellaswag.dataset
        # Return tensors for "input_ids" and "labels"
        try:
            hellaswag_ds.set_format(type="torch", columns=["input_ids", "labels"])
        except Exception:
            pass
        data_loader = DataLoader(
            hellaswag_ds,
            batch_size=cfg.training.local_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        data_iter = cycle(data_loader)

        flops_formula = get_flops_formula_for_hf_config(config)
        flops = flops_formula(config, gbs=cfg.training.global_batch_size, seq_len=cfg.training.seq_len)

        tflops = flops / (10**12)
        if rank == 0:
            print(f"Rank {rank} | Config: {config}")
            print(f"TFLOPs/GPU: {tflops:.6f}")

        # Simple CE loss used by the last stage
        def ce_loss_fn(pred, labels):
            return F.cross_entropy(pred.view(-1, pred.size(-1)), labels.view(-1))

        if cfg.distributed.pp_size > 1:
            pp_schedule, model_parts, has_first_stage, has_last_stage, stages = pipeline_model(
                model,
                world_mesh=mesh["default"],
                moe_mesh=mesh["moe"],
                pp_axis_name="pp",
                dp_axis_names=("dp_replicate", "dp_shard") if dp_replicate_enabled else ("dp_shard",),
                cp_axis_name=None,
                tp_axis_name=None,
                ep_axis_name="ep",
                ep_shard_axis_names=("ep_shard",) if cfg.distributed.ep_size > 1 else None,
                layers_per_stage=cfg.pipelining.layers_per_stage,
                pipeline_parallel_schedule_csv=None,
                pipeline_parallel_schedule=cfg.pipelining.pipeline_parallel_schedule,
                parallelize_fn=cfg.parallelize_fn,
                microbatch_size=cfg.pipelining.pp_microbatch_size,
                local_batch_size=cfg.training.local_batch_size,
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
                loss_fn=ce_loss_fn,
                patch_inner_model=cfg.pipelining.patch_inner_model,
                patch_causal_lm_model=cfg.pipelining.patch_causal_lm_model,
                round_to_pp_multiple=cfg.pipelining.round_to_pp_multiple,
            )
        else:
            cfg.parallelize_fn(
                model,
                world_mesh=mesh["default"],
                moe_mesh=mesh["moe"],
                pp_enabled=False,
                dp_axis_names=("dp_shard",),
                cp_axis_name="cp",
                tp_axis_name="tp",
                ep_axis_name="ep",
                ep_shard_axis_names=("ep_shard",),
            )
            model_parts = [model]
            has_first_stage = True
            has_last_stage = True
            pp_schedule = None

        # Allocate parameters for each stage on its GPU and init
        device = torch.cuda.current_device()
        for mp in model_parts:
            to_empty_parameters_only(mp, device=device, dtype=torch.bfloat16)
            mp.initialize_weights(buffer_device=torch.device(f"cuda:{device}"))
            mp.train()

        optimizer = cfg.optimizer.instantiate(
            params=[
                {
                    "params": mp.parameters(),
                    "name": f"rank_{rank}_stage_{i}",
                }
                for i, mp in enumerate(model_parts)
            ]
        )

    # Log setup time
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    # Training loop over dummy data
    for i in range(cfg.training.steps):
        if i == cfg.profiling.nsys_start and torch.distributed.get_rank() in (
            0,
            torch.distributed.get_world_size() - 1,
        ):
            print(f"Rank {rank} | Starting nsys profiling")
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        dp_size = cfg.distributed.dp_shard_size * cfg.distributed.dp_replicate_size
        ga_steps = cfg.training.global_batch_size // (cfg.training.local_batch_size * dp_size)
        assert ga_steps > 0, "Global batch size must be divisible by local batch size * dp_size"
        if rank == 0:
            print(
                f"Rank {rank} | Iteration {i} | {ga_steps=} | {dp_size=} | {cfg.training.local_batch_size=} | {cfg.training.global_batch_size=}"
            )
        optimizer.zero_grad()
        # Time the entire iteration
        iter_timer = "iteration_warmup" if i < cfg.training.warmup_steps else "iteration"
        with timers(iter_timer, log_level=1):
            for _ga_step_idx in range(ga_steps):
                batch = next(data_iter)
                tokens = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                position_ids = (
                    torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)
                )

                torch.cuda.nvtx.range_push(f"iteration_{i}_ga_step_{_ga_step_idx}")
                # Run one forward+backward pipeline step
                targets, losses = (labels, []) if has_last_stage else (None, None)

                # Time the actual pipeline step
                with timers(f"forward_backward_{_ga_step_idx}", log_level=2):
                    if cfg.distributed.pp_size > 1:
                        if has_first_stage:
                            pp_schedule.step(tokens, target=targets, losses=losses, position_ids=position_ids)
                        else:
                            pp_schedule.step(target=targets, losses=losses, position_ids=position_ids)
                    else:
                        logits = model_parts[0](tokens, position_ids=position_ids)
                        loss = ce_loss_fn(logits, labels)
                        loss.backward()
                        losses.append(loss)

                if has_last_stage and dp_rank == 0:
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                    print(
                        f"Rank {rank} | Iteration {i} | GA step {_ga_step_idx} | Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB | loss={loss.detach().item():.4f}"
                    )

                torch.cuda.nvtx.range_pop()

            with timers("optimizer", log_level=2):
                optimizer.step()
                logger.debug("Optimizer step")

            # Log timing every iteration
            max_iter_time = timers._get_global_min_max_time([iter_timer], reset=False, barrier=False, normalizer=1.0)[
                iter_timer
            ][1]
            if rank == 0:
                mfu = calculate_mfu(
                    tflops,
                    torch.distributed.get_world_size(),
                    max_iter_time,
                    reference_mfu=cfg.get("peak_tflops", 989),
                )
                print(f"Max iter time: {max_iter_time:.6f} seconds")
                print(f"\nMFU: {mfu:.6f}%")

            timers.log(
                names=[iter_timer, "optimizer"]
                + [f"forward_backward_{_ga_step_idx}" for _ga_step_idx in range(ga_steps)],
                rank=0,  # Only log on rank 0
                normalizer=1000.0,  # s
                reset=True,  # Reset timers after logging
                barrier=True,  # Synchronize before collecting times
            )

        if i == cfg.profiling.nsys_end and torch.distributed.get_rank() in (0, torch.distributed.get_world_size() - 1):
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
    warmup_time = timers._timers["iteration_warmup"].active_time() if "iteration_warmup" in timers._timers else 0

    # Calculate average iteration time excluding first 10 warmup iterations
    total_iters = cfg.training.steps
    warmup_iters = cfg.training.warmup_steps

    if rank == 0:
        print(f"Total setup time: {setup_time:.2f} seconds")
        print(f"Total warmup time ({cfg.training.warmup_steps} steps): {warmup_time:.2f} seconds")
        print(f"Total iteration time ({cfg.training.steps - cfg.training.warmup_steps} steps): {iter_time:.2f} seconds")

        # Only calculate average if we have more iterations than warmup
        if total_iters > warmup_iters:
            # Note: This assumes all iterations are timed equally, which isn't perfect
            # since we can't easily separate warmup timing from the total
            avg_iter_time = iter_time / (total_iters - warmup_iters)
        else:
            avg_iter_time = iter_time / total_iters

        print(
            f"Average iteration time: {avg_iter_time:.3f} seconds"
            + f" (excluding first {warmup_iters} warmup iterations)"
            if total_iters > warmup_iters
            else f" (all {total_iters} iterations)"
        )
        mfu = calculate_mfu(
            tflops, torch.distributed.get_world_size(), avg_iter_time, reference_mfu=cfg.get("peak_tflops", 989)
        )
        print(
            f"Average MFU: {mfu:.6f}%" + f" (excluding first {warmup_iters} warmup iterations)"
            if total_iters > warmup_iters
            else f" (all {total_iters} iterations)"
        )
        print(f"{'=' * 60}\n")

    torch.distributed.destroy_process_group()


def main(default_config_path="examples/benchmarking/configs/moonlight_16b_torch.yaml"):
    """Main entry point for the benchmarking script."""
    initialize_distributed(backend="nccl", timeout_minutes=10)

    # Load configuration from YAML file with default path
    cfg = parse_args_and_load_config(default_config_path)

    # Run the benchmarking
    run_benchmark(cfg)


if __name__ == "__main__":
    main()

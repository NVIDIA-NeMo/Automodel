# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import pathlib

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.training.timers import Timers
from nemo_automodel.components.utils.flops_utils import calculate_mfu, get_flops_formula_for_hf_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

logger = logging.getLogger(__name__)


class BenchmarkingRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction):
    """Benchmarking recipe for next-token prediction.

    This class extends TrainFinetuneRecipeForNextTokenPrediction to provide
    a simplified benchmarking-focused training loop with timers and profiling support.
    It reuses the setup() and _forward_backward_step() methods from the parent class.
    """

    def __init__(self, cfg):
        """Initialize the benchmarking recipe.

        Args:
            cfg: Configuration dictionary/object for benchmarking.
        """
        # Store benchmarking-specific parameters from benchmark section
        bench_cfg = cfg.benchmark
        self._bench_warmup_steps = bench_cfg.warmup_steps
        self._bench_peak_tflops = bench_cfg.peak_tflops
        self._bench_nsys_start = bench_cfg.nsys_start
        self._bench_nsys_end = bench_cfg.nsys_end
        self._bench_nsys_ranks = bench_cfg.nsys_ranks

        # Infer max_steps from step_scheduler
        self._bench_steps = cfg.step_scheduler.max_steps

        # Get seq_len from dataset config
        self._bench_seq_len = cfg.dataset.seq_len

        # Infer vocab_size from model config and inject it into dataset config
        if hasattr(cfg, "dataset") and hasattr(cfg, "model"):
            # Get vocab_size from model config
            if hasattr(cfg.model, "config") and hasattr(cfg.model.config, "pretrained_model_name_or_path"):
                from transformers import AutoConfig

                model_config = AutoConfig.from_pretrained(cfg.model.config.pretrained_model_name_or_path)
                vocab_size = model_config.vocab_size
                # Inject vocab_size into dataset config
                cfg.dataset.vocab_size = vocab_size
                if logger.isEnabledFor(logging.INFO):
                    logger.info(f"Inferred vocab_size={vocab_size} from model config")

        # Inject batch_size from step_scheduler into dataset config
        if hasattr(cfg, "dataset") and hasattr(cfg, "step_scheduler"):
            local_batch_size = getattr(cfg.step_scheduler, "local_batch_size", 1)
            cfg.dataset.batch_size = local_batch_size
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Using batch_size={local_batch_size} from step_scheduler.local_batch_size")

        super().__init__(cfg)
        self.timers = Timers(log_level=2, log_option="minmax")

    def setup(self):
        """Setup the benchmarking environment.

        This method calls the parent's setup() but adapts it for benchmarking purposes.
        It skips validation dataloader, checkpointing, and other training-specific features.
        """
        with self.timers("setup", log_level=1):
            # Call parent setup
            super().setup()

        # Clear validation dataloader (not needed for benchmarking)
        self.val_dataloader = None

        # Get step_scheduler config
        seq_len = self._bench_seq_len
        global_batch_size = self.cfg.step_scheduler.global_batch_size

        # Calculate FLOPs
        flops_formula = get_flops_formula_for_hf_config(self.model_parts[0].config)
        flops = flops_formula(self.model_parts[0].config, gbs=global_batch_size, seq_len=seq_len)
        self.tflops = flops / (10**12)

        if self.dist_env.is_main:
            logger.info(f"TFLOPs/GPU: {self.tflops:.6f}")

        self.timers.log(
            names=["setup"],
            rank=0,
            normalizer=1000.0,  # Convert to seconds
            reset=True,
            barrier=True,
        )

    def run_benchmark(self):
        """Run the benchmarking loop.

        This method implements a simplified training loop focused on benchmarking
        with timers and profiling support, similar to the original benchmarking script.
        """
        rank = self.dist_env.rank
        device = self.dist_env.device

        # Get benchmarking config
        steps = self._bench_steps
        warmup_steps = self._bench_warmup_steps
        local_batch_size = self.cfg.step_scheduler.local_batch_size
        global_batch_size = self.cfg.step_scheduler.global_batch_size

        nsys_start = self._bench_nsys_start
        nsys_end = self._bench_nsys_end
        nsys_ranks = self._bench_nsys_ranks

        peak_tflops = self._bench_peak_tflops

        # Set models to training mode
        for mp in self.model_parts:
            mp.train()

        # Calculate gradient accumulation steps
        dp_size = self._get_dp_group_size()
        ga_steps = global_batch_size // (local_batch_size * dp_size)
        assert ga_steps > 0, "Global batch size must be divisible by local batch size * dp_size"

        if rank == 0:
            logger.info(f"Running {steps} iterations with {warmup_steps} warmup steps")
            logger.info(
                f"GA steps: {ga_steps}, DP size: {dp_size}, Local batch size: {local_batch_size}, Global batch size: {global_batch_size}"
            )

        # Create dataloader iterator
        dataloader_iter = iter(self.dataloader)

        # Main benchmarking loop
        for i in range(steps):
            # Start nsys profiling if configured
            if i == nsys_start and rank in nsys_ranks:
                logger.info(f"Rank {rank} | Starting nsys profiling")
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

            if rank == 0:
                logger.info(f"Rank {rank} | Iteration {i}")

            # Zero gradients
            for opt in self.optimizer:
                opt.zero_grad()

            # Time the iteration
            iter_timer = "iteration_warmup" if i < warmup_steps else "iteration"
            with self.timers(iter_timer, log_level=1):
                # Gradient accumulation loop
                num_label_tokens = 0
                loss_buffer = []
                for mp in self.model_parts:
                    if hasattr(mp, "set_fsdp_states_for_first_forward"):
                        mp.set_fsdp_states_for_first_forward()

                for ga_step_idx in range(ga_steps):
                    if ga_step_idx == ga_steps - 1 and not self.pp_enabled:
                        for mp in self.model_parts:
                            if hasattr(mp, "set_fsdp_states_for_last_backward"):
                                mp.set_fsdp_states_for_last_backward()

                    # Get batch from dataloader
                    batch = next(dataloader_iter)
                    torch.cuda.nvtx.range_push(f"iteration_{i}_ga_step_{ga_step_idx}")

                    num_label_tokens += (batch["labels"] != -100).sum().item()

                    with self.timers(f"forward_backward_{ga_step_idx}", log_level=2):
                        self._forward_backward_step(
                            ga_step_idx,
                            batch,
                            loss_buffer=loss_buffer,
                            num_label_tokens=None,
                            num_batches=ga_steps,
                            is_train=True,
                        )

                if self.pp_enabled:
                    for mp in self.model_parts:
                        if hasattr(mp, "finalize_fsdp_states_post_backward"):
                            mp.finalize_fsdp_states_post_backward()

                loss = (
                    torch.sum(torch.stack(loss_buffer)).to(device)
                    if loss_buffer
                    else torch.tensor([-1.0], device=device)
                )
                if self.pp_enabled:
                    src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
                    if self.dist_env.rank == src_rank:
                        torch.distributed.send(loss, dst=0)
                    elif self.dist_env.is_main:
                        torch.distributed.recv(loss, src=src_rank)

                if rank == 0:
                    loss = loss / num_label_tokens
                    logger.info(
                        f"Rank {rank} | Iteration {i} | num_label_tokens={num_label_tokens} | "
                        f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB | "
                        f"loss={loss.detach().item():.4f}"
                    )

                    torch.cuda.nvtx.range_pop()

                # Optimizer step
                with self.timers("optimizer", log_level=2):
                    for opt in self.optimizer:
                        opt.step()
                    logger.debug("Optimizer step")

            # Calculate and log MFU
            self._log_iteration_metrics(iter_timer, ga_steps, peak_tflops, rank)

            # Stop nsys profiling if configured
            if i == nsys_end and rank in nsys_ranks:
                logger.info(f"Rank {rank} | Stopping nsys profiling")
                torch.cuda.cudart().cudaProfilerStop()

        # Final summary
        self._log_benchmark_summary(steps, warmup_steps, peak_tflops, rank)

    def _log_iteration_metrics(self, iter_timer, ga_steps, peak_tflops, rank):
        max_iter_time = self.timers._get_global_min_max_time([iter_timer], reset=False, barrier=False, normalizer=1.0)[
            iter_timer
        ][1]

        if rank == 0:
            mfu = calculate_mfu(
                self.tflops,
                self.dist_env.world_size,
                max_iter_time,
                reference_mfu=peak_tflops,
            )
            logger.info(f"Max iter time: {max_iter_time:.6f} seconds")
            logger.info(f"MFU: {mfu:.6f}%")

        # Log detailed timers
        self.timers.log(
            names=[iter_timer, "optimizer"] + [f"forward_backward_{ga_step_idx}" for ga_step_idx in range(ga_steps)],
            rank=0,
            normalizer=1000.0,  # Convert to seconds
            reset=True,
            barrier=True,
        )

    def _log_benchmark_summary(self, steps, warmup_steps, peak_tflops, rank):
        torch.distributed.barrier()
        if rank == 0:
            logger.info(f"{'=' * 60}")
            logger.info("Benchmarking Summary")
            logger.info(f"{'=' * 60}")

        # Get active times for summary
        setup_time = self.timers._timers["setup"].active_time() if "setup" in self.timers._timers else 0
        iter_time = self.timers._timers["iteration"].active_time() if "iteration" in self.timers._timers else 0
        warmup_time = (
            self.timers._timers["iteration_warmup"].active_time() if "iteration_warmup" in self.timers._timers else 0
        )

        if rank == 0:
            logger.info(f"Total setup time: {setup_time:.2f} seconds")
            logger.info(f"Total warmup time ({warmup_steps} steps): {warmup_time:.2f} seconds")
            logger.info(f"Total iteration time ({steps - warmup_steps} steps): {iter_time:.2f} seconds")

            # Calculate average iteration time
            if steps > warmup_steps:
                avg_iter_time = iter_time / (steps - warmup_steps)
            else:
                avg_iter_time = iter_time / steps

            logger.info(
                f"Average iteration time: {avg_iter_time:.3f} seconds"
                + (
                    f" (excluding first {warmup_steps} warmup iterations)"
                    if steps > warmup_steps
                    else f" (all {steps} iterations)"
                )
            )

            mfu = calculate_mfu(self.tflops, self.dist_env.world_size, avg_iter_time, reference_mfu=peak_tflops)
            logger.info(
                f"Average MFU: {mfu:.6f}%"
                + (
                    f" (excluding first {warmup_steps} warmup iterations)"
                    if steps > warmup_steps
                    else f" (all {steps} iterations)"
                )
            )
            logger.info(f"{'=' * 60}\n")


def main(config_path=None):
    """Main entry point for the benchmarking recipe.

    Loads the configuration, sets up the recipe, and runs the benchmark.
    """
    if config_path is None:
        # Default to moonlight_16b_torch.yaml in examples/benchmark/configs
        config_path = (
            pathlib.Path(__file__).parent.parent.parent.resolve()
            / "examples"
            / "benchmark"
            / "configs"
            / "moonlight_16b_torch.yaml"
        )

    cfg = parse_args_and_load_config(config_path)
    recipe = BenchmarkingRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_benchmark()


if __name__ == "__main__":
    main()

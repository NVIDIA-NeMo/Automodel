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

import json
import logging
import pathlib

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.training.timers import Timers
from nemo_automodel.components.training.utils import (
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
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
        self._bench_json_output_path = getattr(bench_cfg, "json_output_path", None)
        self._wandb_enabled = cfg.get("wandb", None) is not None

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

        # Store wandb run object if initialized by parent
        import wandb

        self.wandb_run = wandb.run

        # Clear validation dataloader (not needed for benchmarking)
        self.val_dataloader = None

        # Get step_scheduler config
        seq_len = self._bench_seq_len
        global_batch_size = self.cfg.step_scheduler.global_batch_size

        # Calculate FLOPs
        flops_formula = get_flops_formula_for_hf_config(self.model_parts[0].config)
        flops = flops_formula(self.model_parts[0].config, gbs=global_batch_size, seq_len=seq_len)
        self.tflops = flops / (10**12)

        if hasattr(self.cfg, "peft"):
            # Calculate trainable vs non-trainable parameters without lora
            lora_params = self.param_info["trainable_params"]
            total_params = self.param_info["total_params"]
            frozen_params = total_params - lora_params

            # Adjust TFLOPS for PEFT: training has 3 computational phases:
            # 1) Forward pass: processes all parameters (frozen + LoRA)
            # 2) Backward to inputs: processes all parameters
            # 3) Backward to parameters: linear in size of trainable weights (lora params)
            #    - For full fine-tuning: phases 1, 2, 3 are roughly equal (1/3 each)
            #    - For LoRA: phases 1, 2 are the same, but phase 3 scales with lora/frozen ratio
            # Since base TFLOPS assumes all params trainable, we scale by:
            # (2 * frozen_flops + lora_flops) / (3 * frozen_flops) = (2 + lora/frozen) / 3
            param_ratio = lora_params / frozen_params
            tflops_multiplier = (2 + param_ratio) / 3
            self.tflops *= tflops_multiplier
            if self.dist_env.is_main:
                logger.info(
                    f"PEFT params - lora_params: {lora_params:,}, frozen_params: {frozen_params:,}, param_ratio (lora/frozen): {param_ratio:.4f}"
                )
                logger.info(f"TFLOPS multiplier for PEFT: (2 + {param_ratio:.4f}) / 3 = {tflops_multiplier:.4f}")

        if self.dist_env.is_main:
            logger.info(f"TFLOPs/GPU: {self.tflops:.6f}")

        # Log setup time to wandb
        if self._wandb_enabled:
            self.timers.write_to_wandb(
                names=["setup"],
                writer=self.wandb_run,
                iteration=0,
                normalizer=1.0,
                reset=False,
                barrier=True,
            )

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
                prepare_for_grad_accumulation(self.model_parts, pp_enabled=self.pp_enabled)

                for ga_step_idx in range(ga_steps):
                    if ga_step_idx == ga_steps - 1:
                        prepare_for_final_backward(self.model_parts, pp_enabled=self.pp_enabled)

                    # Get batch from dataloader
                    batch = next(dataloader_iter)
                    torch.cuda.nvtx.range_push(f"iteration_{i}_ga_step_{ga_step_idx}")

                    # Accumulate label tokens locally
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

                    torch.cuda.nvtx.range_pop()

                # Optimizer step
                with self.timers("optimizer", log_level=2):
                    for opt in self.optimizer:
                        opt.step()
                    logger.debug("Optimizer step")

            # Synchronize num_label_tokens across DP ranks
            num_label_tokens_tensor = torch.tensor(num_label_tokens, dtype=torch.long, device=device)
            num_label_tokens_tensor = self._dp_allreduce(num_label_tokens_tensor)
            num_label_tokens = num_label_tokens_tensor.item()

            # Calculate loss - following exact train_ft.py:1059-1071 pattern
            reporting_loss = torch.sum(torch.stack(loss_buffer))
            reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
            reporting_loss = reporting_loss.to(torch.float32) / num_label_tokens

            if self.pp_enabled:
                reporting_loss = reporting_loss.to(self.dist_env.device)
                # Send loss to first rank if pp group rank is 0
                src_rank = self.device_mesh.mesh.reshape(-1)[-1].item()
                if self.dist_env.rank == src_rank:
                    torch.distributed.send(reporting_loss, dst=0)
                elif self.dist_env.is_main:
                    torch.distributed.recv(reporting_loss, src=src_rank)

            reporting_loss = reporting_loss.cpu().item()

            if rank == 0:
                print(f"num_label_tokens={num_label_tokens} | loss={reporting_loss:.4f}")
                logger.info(
                    f"Rank {rank} | Iteration {i} | num_label_tokens={num_label_tokens} | "
                    f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB | "
                    f"loss={reporting_loss:.4f}"
                )

            # Calculate and log MFU
            self._log_iteration_metrics(iter_timer, ga_steps, peak_tflops, rank, i)

            # Stop nsys profiling if configured
            if i == nsys_end and rank in nsys_ranks:
                logger.info(f"Rank {rank} | Stopping nsys profiling")
                torch.cuda.cudart().cudaProfilerStop()

        # Final summary
        self._log_benchmark_summary(steps, warmup_steps, peak_tflops, rank)

    def _log_iteration_metrics(self, iter_timer, ga_steps, peak_tflops, rank, iteration):
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
        timer_names = [iter_timer, "optimizer"] + [f"forward_backward_{ga_step_idx}" for ga_step_idx in range(ga_steps)]
        # Log timers to wandb
        if self._wandb_enabled:
            self.timers.write_to_wandb(
                names=timer_names,
                writer=self.wandb_run,
                iteration=iteration,
                normalizer=1.0,
                reset=False,
                barrier=False,
            )

        self.timers.log(
            names=timer_names,
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

            # Prepare summary data
            summary_data = {
                "total_steps": steps,
                "warmup_steps": warmup_steps,
                "training_steps": steps - warmup_steps,
                "setup_time_seconds": setup_time,
                "warmup_time_seconds": warmup_time,
                "training_time_seconds": iter_time,
                "avg_iter_time_seconds": avg_iter_time,
                "avg_mfu_percent": mfu,
                "tflops_per_gpu": self.tflops,
                "peak_tflops": peak_tflops,
                "world_size": self.dist_env.world_size,
                "global_batch_size": self.cfg.step_scheduler.global_batch_size,
                "local_batch_size": self.cfg.step_scheduler.local_batch_size,
                "seq_len": self._bench_seq_len,
            }

            # Log to wandb as table
            if self.wandb_run is not None:
                import wandb

                # Create a table with the summary data (use raw numeric values)
                summary_table = wandb.Table(
                    columns=["Metric", "Value"],
                    data=[
                        ["Total Steps", steps],
                        ["Warmup Steps", warmup_steps],
                        ["Training Steps", steps - warmup_steps],
                        ["Setup Time (s)", setup_time],
                        ["Warmup Time (s)", warmup_time],
                        ["Training Time (s)", iter_time],
                        ["Avg Iteration Time (s)", avg_iter_time],
                        ["Avg MFU (%)", mfu],
                        ["TFLOPs/GPU/s", peak_tflops * mfu / 100],
                        ["Peak TFLOPs", peak_tflops],
                        ["World Size", self.dist_env.world_size],
                        ["Global Batch Size", self.cfg.step_scheduler.global_batch_size],
                        ["Local Batch Size", self.cfg.step_scheduler.local_batch_size],
                        ["Sequence Length", self._bench_seq_len],
                    ],
                )
                wandb.log({"benchmark_summary": summary_table})

                # Also log as scalar metrics for easy filtering
                wandb.log(
                    {
                        "summary/avg_iter_time_seconds": avg_iter_time,
                        "summary/avg_mfu_percent": mfu,
                        "summary/training_time_seconds": iter_time,
                        "summary/tflops_per_gpu": self.tflops,
                    }
                )

            # Save summary to JSON file if output path is provided
            if self._bench_json_output_path is not None:
                summary_file = pathlib.Path(self._bench_json_output_path)
                # Create parent directory if it doesn't exist
                summary_file.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_file, "w") as f:
                    json.dump(summary_data, f, indent=2)
                logger.info(f"Benchmark summary saved to {summary_file.absolute()}")

        # Finish wandb run
        if self.wandb_run is not None and rank == 0:
            import wandb

            wandb.finish()


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

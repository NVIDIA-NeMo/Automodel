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
import time

import torch
from transformers import AutoConfig

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.training.timers import Timers
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_mfu(tflops, world_size, time_seconds, reference_mfu=1979.0):
    """Calculate Model FLOPs Utilization (MFU)."""
    mfu = tflops / (world_size * time_seconds)
    mfu = mfu / reference_mfu
    return mfu * 100


class BenchmarkRecipeForNextTokenPrediction(TrainFinetuneRecipeForNextTokenPrediction):
    """Benchmarking recipe that extends the base training recipe with mock data generation."""

    def __init__(self, cfg):
        """Initialize the benchmarking recipe with configuration."""
        super().__init__(cfg)
        self.timers = None
        self.config = None
        self.tflops = None

    def setup(self):
        """Setup benchmarking components, reusing the parent setup logic."""
        # Initialize timers for benchmarking
        self.timers = Timers(log_level=2, log_option="minmax")

        # Setup phase with timer
        with self.timers("setup", log_level=1):
            # Call parent setup to build all components
            super().setup()

            # Get model config for FLOP calculations
            self.config = AutoConfig.from_pretrained(
                self.cfg.model.pretrained_model_name_or_path,
                trust_remote_code=self.cfg.model.get("trust_remote_code", False),
            )
            if self.cfg.model.num_layers is not None:
                self.config.num_hidden_layers = self.cfg.model.num_layers

            # Calculate FLOPs for MFU
            flops_formula = get_flops_formula_for_hf_config(self.config)
            flops = flops_formula(
                self.config, gbs=self.cfg.step_scheduler.global_batch_size, seq_len=self.cfg.get("seq_len", 2048)
            )
            self.tflops = flops / (10**12)

            if self.dist_env.is_main:
                print(f"Config: {self.config}")
                print(f"TFLOPs/GPU: {self.tflops:.6f}")

        # Log setup time after barrier
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    def generate_mock_batch(self, local_batch_size, seq_len):
        """Generate a mock batch of data for benchmarking."""
        device = self.dist_env.device

        # Generate random input tokens
        tokens = torch.randint(0, self.config.vocab_size, (local_batch_size, seq_len), device=device)

        # Generate labels (shifted input tokens with -100 padding at the end)
        labels = torch.cat(
            [
                tokens[:, 1:],
                torch.full((local_batch_size, 1), -100, device=device, dtype=tokens.dtype),
            ],
            dim=1,
        )

        # Generate position IDs
        position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(tokens.shape[0], -1)

        return {
            "input_ids": tokens,
            "labels": labels,
            "position_ids": position_ids,
        }

    def run_benchmark_loop(self):
        """Run the benchmarking loop with mock data, reusing only _forward_backward_step."""
        for mp in self.model_parts:
            mp.train()

        # Set up benchmarking timestamp
        time.perf_counter()

        # Extract training parameters
        steps = self.cfg.get("steps", 10)
        warmup_steps = self.cfg.get("warmup_steps", 3)
        local_batch_size = self.cfg.step_scheduler.local_batch_size
        seq_len = self.cfg.get("seq_len", 2048)

        # Calculate gradient accumulation steps
        dp_size = self._get_dp_group_size()
        ga_steps = self.cfg.step_scheduler.global_batch_size // (local_batch_size * dp_size)
        assert ga_steps > 0, "Global batch size must be divisible by local batch size * dp_size"

        if self.dist_env.is_main:
            print(f"Starting benchmark with {steps} steps, {warmup_steps} warmup steps")
            print(f"GA steps: {ga_steps}, DP size: {dp_size}, Local batch size: {local_batch_size}")

        # Benchmarking loop
        for i in range(steps):
            if self.dist_env.is_main:
                print(
                    f"Rank {self.dist_env.rank} | Iteration {i} | {ga_steps=} | {dp_size=} | {local_batch_size=} | {self.cfg.step_scheduler.global_batch_size=}"
                )

            # Zero gradients
            for opt in self.optimizer:
                opt.zero_grad()

            # Time the entire iteration
            iter_timer = "iteration_warmup" if i < warmup_steps else "iteration"
            with self.timers(iter_timer, log_level=1):
                loss_buffer = []
                num_label_tokens = 0

                for ga_step_idx in range(ga_steps):
                    torch.cuda.nvtx.range_push(f"iteration_{i}_ga_step_{ga_step_idx}")

                    # Generate mock batch
                    batch = self.generate_mock_batch(local_batch_size, seq_len)

                    # Count label tokens
                    batch_label_tokens = (batch["labels"] != -100).sum().item()
                    num_label_tokens += batch_label_tokens

                    # Time the forward+backward step - reuse parent class method
                    with self.timers(f"forward_backward_{ga_step_idx}", log_level=2):
                        self._forward_backward_step(
                            ga_step_idx,
                            batch,
                            loss_buffer=loss_buffer,
                            num_label_tokens=batch_label_tokens,
                            num_batches=ga_steps,
                            is_train=True,
                        )

                    # Log loss for last stage (similar to original benchmarking script)
                    if (self.pp is None or self.pp.info.has_last_stage) and self._get_dp_rank() == 0:
                        if loss_buffer:
                            loss = torch.mean(torch.stack(loss_buffer)).item()
                            max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                            print(
                                f"Rank {self.dist_env.rank} | Iteration {i} | GA step {ga_step_idx} | "
                                f"Max Memory Allocated: {max_memory_gb:.2f} GB | loss={loss:.4f}"
                            )

                    torch.cuda.nvtx.range_pop()

                # Optimizer step (similar to original benchmarking script)
                with self.timers("optimizer", log_level=2):
                    for opt in self.optimizer:
                        opt.step()
                    logger.debug("Optimizer step")

            # Calculate timing and MFU (similar to original benchmarking script)
            max_iter_time = self.timers._get_global_min_max_time(
                [iter_timer], reset=False, barrier=False, normalizer=1.0
            )[iter_timer][1]

            if self.dist_env.is_main:
                mfu = calculate_mfu(
                    self.tflops,
                    self.dist_env.world_size,
                    max_iter_time,
                    reference_mfu=self.cfg.get("peak_tflops", 989),
                )
                print(f"Max iter time: {max_iter_time:.6f} seconds")
                print(f"\nMFU: {mfu:.6f}%")

            # Log detailed timing (similar to original benchmarking script)
            self.timers.log(
                names=[iter_timer, "optimizer"]
                + [f"forward_backward_{ga_step_idx}" for ga_step_idx in range(ga_steps)],
                rank=0,  # Only log on rank 0
                normalizer=1000.0,  # s
                reset=True,  # Reset timers after logging
                barrier=True,  # Synchronize before collecting times
            )

        self._print_benchmark_summary(steps, warmup_steps)

    def _print_benchmark_summary(self, steps, warmup_steps):
        """Print final benchmarking summary."""
        torch.distributed.barrier()
        if self.dist_env.is_main:
            print(f"\n{'=' * 60}")
            print("Benchmarking Summary")
            print(f"{'=' * 60}")

        # Get timing summaries
        setup_time = self.timers._timers["setup"].active_time() if "setup" in self.timers._timers else 0
        iter_time = self.timers._timers["iteration"].active_time() if "iteration" in self.timers._timers else 0
        warmup_time = (
            self.timers._timers["iteration_warmup"].active_time() if "iteration_warmup" in self.timers._timers else 0
        )

        if self.dist_env.is_main:
            print(f"Total setup time: {setup_time:.2f} seconds")
            print(f"Total warmup time ({warmup_steps} steps): {warmup_time:.2f} seconds")
            print(f"Total iteration time ({steps - warmup_steps} steps): {iter_time:.2f} seconds")

            # Calculate average iteration time
            if steps > warmup_steps:
                avg_iter_time = iter_time / (steps - warmup_steps)
            else:
                avg_iter_time = iter_time / steps

            print(
                f"Average iteration time: {avg_iter_time:.3f} seconds"
                + (
                    f" (excluding first {warmup_steps} warmup iterations)"
                    if steps > warmup_steps
                    else f" (all {steps} iterations)"
                )
            )

            mfu = calculate_mfu(
                self.tflops, self.dist_env.world_size, avg_iter_time, reference_mfu=self.cfg.get("peak_tflops", 989)
            )
            print(
                f"Average MFU: {mfu:.6f}%"
                + (
                    f" (excluding first {warmup_steps} warmup iterations)"
                    if steps > warmup_steps
                    else f" (all {steps} iterations)"
                )
            )
            print(f"{'=' * 60}\n")

    def run_train_validation_loop(self):
        """Override the training loop to run benchmarking instead."""
        self.run_benchmark_loop()


def main(default_config_path="examples/benchmarking/configs/moonlight_16b_torch.yaml"):
    """Main entry point for the benchmarking recipe."""
    # Load configuration from YAML file with default path
    cfg = parse_args_and_load_config(default_config_path)

    # Create and run the benchmarking recipe
    recipe = BenchmarkRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()

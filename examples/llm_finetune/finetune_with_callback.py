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

"""
Example: Using Custom Callbacks with Automodel Training

This example demonstrates how to use callbacks to hook into the training loop
for custom logging, monitoring, or integration with external systems.

Usage:
    # Using default small model (270M - fastest for testing)
    python examples/llm_finetune/finetune_with_callback.py
    
    # Or specify a different model
    python examples/llm_finetune/finetune_with_callback.py \\
        -c examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml
"""

from __future__ import annotations

import logging
from nemo_automodel.components.callbacks import Callback, rank_zero_only
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

logger = logging.getLogger(__name__)


class SimpleLoggingCallback(Callback):
    """
    A basic callback that logs training progress at key milestones.
    
    Note: logger.info() is automatically filtered to rank 0 by RankFilter,
    so no explicit rank checking is needed for simple logging.
    """
    
    def on_train_start(self, recipe, **kwargs):
        # Basic logging works without rank checks (automatically filtered)
        print(f"Logger filters: {logger.filters}")  # Will show [<RankFilter>]
        print(f"Root filters: {logging.getLogger().filters}")  # Also shows RankFilter
        logger.info("This will be filtered to rank 0!")

        logger.info("[SimpleLoggingCallback] 🔥 Training is starting!")
        logger.info(f"[SimpleLoggingCallback]    World size: {recipe.dist_env.world_size} GPUs")
        logger.info(f"[SimpleLoggingCallback]    Total steps: {recipe.step_scheduler.max_steps}")

    def on_train_batch_end(self, recipe, **kwargs):
        step = recipe.step_scheduler.step
        if step % 10 == 0:
            metrics = kwargs['train_log_data'].metrics
            # Logging is automatically filtered to rank 0
            logger.info(
                f"[SimpleLoggingCallback] 🚀 Step {step}/{recipe.step_scheduler.max_steps}: "
                f"Loss = {metrics['loss']:.4f}, LR = {metrics['lr']:.2e}"
            )

    def on_validation_end(self, recipe, **kwargs):
        val_results = kwargs['val_results']
        # val_results is a dict: {"validation": MetricsSample, ...}
        for name, log_data in val_results.items():
            logger.info(
                f"[SimpleLoggingCallback] ✅ Validation '{name}': "
                f"Loss = {log_data.metrics['val_loss']:.4f}"
            )

    def on_save_checkpoint(self, recipe, **kwargs):
        checkpoint_info = kwargs['checkpoint_info']
        logger.info(
            f"[SimpleLoggingCallback] 💾 Checkpoint saved at step {checkpoint_info['step']}, "
            f"epoch {checkpoint_info['epoch']}, path: {checkpoint_info['checkpoint_path']}"
        )

    def on_train_end(self, recipe, **kwargs):
        logger.info(
            f"[SimpleLoggingCallback] 🎉 Training completed successfully! "
            f"Final step: {recipe.step_scheduler.step}"
        )

    def on_exception(self, recipe, **kwargs):
        exception = kwargs['exception']
        logger.error(f"[SimpleLoggingCallback] ❌ Training failed: {exception}")


class MetricsCollectorCallback(Callback):
    """
    Example callback that collects metrics for external reporting.
    
    In a real scenario, this could report to an API, database, or monitoring system.
    
    Note: In distributed training (multiple GPUs), callbacks run on ALL ranks.
    This example shows TWO ways to handle rank filtering:
    
    1. Manual checking: if recipe.dist_env.is_main
    2. Using @rank_zero_only decorator (recommended for cleaner code)
    """
    
    def __init__(self):
        self.training_metrics = []
        self.validation_metrics = []
        self.checkpoints = []
    
    def on_train_batch_end(self, recipe, **kwargs):
        step = recipe.step_scheduler.step
        metrics = kwargs['train_log_data'].metrics
        
        # Collect metrics (happens on all ranks, but that's fine for local state)
        self.training_metrics.append({
            'step': step,
            'loss': metrics['loss'],
            'lr': metrics['lr'],
        })
        
        # Method 1: Manual rank checking
        # In a real use case, only rank 0 should send to external APIs:
        # if recipe.dist_env.is_main and step % 100 == 0:
        #     requests.post('https://api.example.com/metrics', json=metrics)
    
    @rank_zero_only  # Method 2: Using decorator (cleaner!)
    def on_validation_end(self, recipe, **kwargs):
        val_results = kwargs['val_results']
        
        # This only runs on rank 0 thanks to @rank_zero_only
        # val_results is a dict: {"validation": MetricsSample, "squad": MetricsSample, ...}
        for name, log_data in val_results.items():
            self.validation_metrics.append({
                'step': log_data.step,
                'epoch': log_data.epoch,
                'validation_name': name,
                'metrics': log_data.metrics,  # Full metrics dict (val_loss, accuracy, etc.)
            })
        
        logger.info(
            f"[MetricsCollectorCallback] 📊 Collected {len(self.validation_metrics)} validation checkpoints"
        )
    
    @rank_zero_only
    def on_save_checkpoint(self, recipe, **kwargs):
        checkpoint_info = kwargs['checkpoint_info']
        
        # Track checkpoint information for external reporting
        self.checkpoints.append({
            'step': checkpoint_info['step'],
            'epoch': checkpoint_info['epoch'],
            'train_loss': checkpoint_info['train_loss'],
            'val_losses': checkpoint_info['val_losses'],
            'path': checkpoint_info['checkpoint_path'],
        })
        
        logger.info(
            f"[MetricsCollectorCallback] 💾 Tracked checkpoint {len(self.checkpoints)}: "
            f"step={checkpoint_info['step']}, train_loss={checkpoint_info['train_loss']:.4f}"
        )


def main(default_config_path="examples/llm_finetune/gemma/gemma_3_270m_squad_peft.yaml"):
    """
    Main entry point for fine-tuning with custom callbacks.
    
    This example shows how to use multiple callbacks simultaneously.
    
    For faster testing, this uses Gemma 3 270M (the smallest model available).
    You can change the config to use larger models like:
    - examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml
    - examples/llm_finetune/granite/granite_3_3_2b_instruct_squad_peft.yaml
    """
    cfg = parse_args_and_load_config(default_config_path)
    
    # Instantiate multiple callbacks
    logging_callback = SimpleLoggingCallback()
    metrics_callback = MetricsCollectorCallback()
    
    # Pass them to the recipe (they'll be called in order)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(
        cfg, 
        callbacks=[logging_callback, metrics_callback]
    )
    
    recipe.setup()
    
    try:
        recipe.run_train_validation_loop()
    except Exception as e:
        logger.error(f"[Main] Training failed: {e}")
        raise
    
    # After training, you can access collected metrics
    logger.info(
        f"[MetricsCollectorCallback] 🎉 Training complete! "
        f"Collected {len(metrics_callback.training_metrics)} training steps"
    )


if __name__ == "__main__":
    main()

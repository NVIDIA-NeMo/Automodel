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

import functools
from typing import Any, Dict, List, Optional


def rank_zero_only(fn):
    """
    Decorator to ensure a callback method runs only on rank 0 (main process).

    This is useful for operations that should happen once across all GPUs, such as:
    - External API calls
    - File writes
    - Database updates
    - Slack/email notifications

    Usage:
    ```python
    class MyCallback(Callback):
        @rank_zero_only
        def on_train_batch_end(self, recipe, **kwargs):
            # This only runs on rank 0
            requests.post('https://api.example.com/metrics', ...)
    ```

    Note: If not using distributed training (single GPU), this decorator has no effect
    and the function will execute normally.
    """

    @functools.wraps(fn)
    def wrapper(self, recipe, **kwargs):
        # Check if this is rank 0 (main process)
        if hasattr(recipe, "dist_env") and not recipe.dist_env.is_main:
            return None
        return fn(self, recipe, **kwargs)

    return wrapper


class Callback:
    """
    Abstract base class for training callbacks.

    Callbacks provide hooks into the training loop for custom logging, monitoring,
    or other behavior without modifying recipe code directly.

    Available hooks (in execution order):
    - on_train_start: Called once at the start of training (after setup completes)
    - on_train_batch_end: Called after each optimizer step
    - on_validation_end: Called after validation completes
    - on_save_checkpoint: Called when a checkpoint is saved
    - on_train_end: Called when training completes successfully
    - on_exception: Called if training fails with an exception

    Each hook receives the recipe instance and hook-specific kwargs.
    See individual hook methods for detailed arguments.

    Note: in multi-GPU training, callbacks run on ALL ranks. Python logging (logger.info/warn/error)
    is automatically filtered to rank 0. For operations like API calls or file I/O, use the
    @rank_zero_only decorator or check recipe.dist_env.is_main explicitly.

    Usage:
    ```python
    import logging
    from nemo_automodel.components.callbacks import Callback, rank_zero_only
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    logger = logging.getLogger(__name__)

    class MetricsLogger(Callback):
        def on_train_start(self, recipe, **kwargs):
            logger.info(f"Training started on {recipe.dist_env.world_size} GPUs")

        def on_train_batch_end(self, recipe, **kwargs):
            log_data = kwargs['train_log_data']
            if log_data.step % 100 == 0:
                logger.info(f"Step {log_data.step}: Loss = {log_data.metrics['loss']:.4f}")

        @rank_zero_only
        def on_validation_end(self, recipe, **kwargs):
            val_results = kwargs['val_results']
            # Send metrics to external API (only on rank 0)
            for name, log_data in val_results.items():
                requests.post('https://api.example.com/metrics', json={
                    'name': name, 'step': log_data.step, 'loss': log_data.metrics['val_loss']
                })

    cfg = parse_args_and_load_config("config.yaml")
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg, callbacks=[MetricsLogger()])
    recipe.setup()
    recipe.run_train_validation_loop()
    ```
    """

    def on_train_start(self, recipe: Any, **kwargs) -> None:
        """
        Called when the train begins (after setup completes).

        Args:
            recipe: The recipe instance
            **kwargs: No additional kwargs for this hook
        """
        pass

    def on_train_batch_end(self, recipe: Any, **kwargs) -> None:
        """
        Called when the train batch ends (after optimizer step).

        Args:
            recipe: The recipe instance
            **kwargs: Contains `train_log_data` MetricsSample with step, epoch, timestamp, and metrics dict
        """
        pass

    def on_validation_end(self, recipe: Any, **kwargs) -> None:
        """
        Called when the validation loop ends.

        Args:
            recipe: The recipe instance
            **kwargs: Contains `val_results` dict mapping validation names to MetricsSample objects.
                Each MetricsSample has: step, epoch, timestamp, and metrics dict (with `val_loss`, etc.)
        """
        pass

    def on_save_checkpoint(self, recipe: Any, **kwargs) -> None:
        """
        Called when a checkpoint is saved.

        Args:
            recipe: The recipe instance
            **kwargs: Contains `checkpoint_info` dict with:
                - epoch (int): Current epoch
                - step (int): Current training step
                - train_loss (float): Current training loss
                - val_losses (dict): Validation losses (can be empty dict)
                - checkpoint_path (str): Path where checkpoint is saved
                - best_metric_key (str): Key used to determine best checkpoint
        """
        pass

    def on_exception(self, recipe: Any, **kwargs) -> None:
        """
        Called when any exception occurs during training.

        Args:
            recipe: The recipe instance
            **kwargs: Contains `exception` (the Exception instance)
        """
        pass

    def on_train_end(self, recipe: Any, **kwargs) -> None:
        """
        Called when training completes successfully.

        Args:
            recipe: The recipe instance
            **kwargs: No additional kwargs for this hook
        """
        pass


class CallbackRunner:
    """
    Class to run callbacks.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_start(self, recipe: Any, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_train_start(recipe, **kwargs)

    def on_train_batch_end(self, recipe: Any, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(recipe, **kwargs)

    def on_validation_end(self, recipe: Any, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(recipe, **kwargs)

    def on_save_checkpoint(self, recipe: Any, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_save_checkpoint(recipe, **kwargs)

    def on_exception(self, recipe: Any, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_exception(recipe, **kwargs)

    def on_train_end(self, recipe: Any, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_train_end(recipe, **kwargs)

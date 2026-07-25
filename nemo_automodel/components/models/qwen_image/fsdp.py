# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen-Image FSDP2 registration."""

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper


def register_qwen_image_parallel_strategy() -> None:
    """Register full-block Qwen-Image activation checkpointing."""
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
        register_parallel_strategy,
    )

    name = "QwenImageTransformer2DModel"
    if name in PARALLELIZATION_STRATEGIES:
        return

    @register_parallel_strategy(name=name)
    class QwenImageParallelizationStrategy(DefaultParallelizationStrategy):
        def parallelize(self, model, device_mesh, activation_checkpointing=False, **kwargs):
            if activation_checkpointing:
                for index, block in enumerate(model.transformer_blocks):
                    model.transformer_blocks[index] = checkpoint_wrapper(
                        block,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    )
            return super().parallelize(
                model,
                device_mesh,
                activation_checkpointing=False,
                **kwargs,
            )

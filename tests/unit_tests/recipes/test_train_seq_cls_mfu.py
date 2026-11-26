# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
from nemo_automodel.recipes import base_recipe
from nemo_automodel.recipes.llm import train_seq_cls
from nemo_automodel.recipes.llm.train_seq_cls import TrainFinetuneRecipeForSequenceClassification


class _StopAfterMFU(Exception):
    """Raised by the tokenizer build, the step after the MFU calculator in setup()."""


@pytest.mark.parametrize(
    "strategy, sharded, param_dtype, expected",
    [
        # A single process is not sharded: the classifier keeps its storage dtype unless it autocasts.
        (FSDP2Config(), False, torch.float32, 80.0),
        (FSDP2Config(autocast_dtype=torch.bfloat16), False, torch.float32, 2500.0),
        # Sharded FSDP2 computes in the policy's param_dtype, not its reduce_dtype.
        (FSDP2Config(), True, torch.float32, 2500.0),
        (
            FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.bfloat16)),
            True,
            torch.bfloat16,
            80.0,
        ),
        (DDPConfig(), True, torch.float32, 80.0),
        (DDPConfig(autocast_dtype=torch.bfloat16), True, torch.float32, 2500.0),
        (
            FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32), autocast_dtype=torch.bfloat16),
            True,
            torch.float32,
            2500.0,
        ),
    ],
)
def test_classifier_mfu_uses_strategy_compute_dtype(strategy, sharded, param_dtype, expected, monkeypatch):
    # GB200: FP32 80 and BF16 2500 dense TFLOPs/s.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GB200")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    monkeypatch.setattr(base_recipe, "fsdp2_sharding_enabled", lambda device_mesh: sharded)

    model = nn.Linear(16, 2, dtype=param_dtype)
    model.config = SimpleNamespace(model_type="bert")
    mesh = MagicMock(device_type="cpu", mesh_dim_names=("dp", "tp", "cp"))
    mesh.__getitem__.return_value.size.return_value = 1
    distributed_setup = SimpleNamespace(
        mesh_context=SimpleNamespace(device_mesh=mesh, moe_mesh=None, pp_enabled=False),
        strategy_config=strategy,
        pipeline_config=None,
        moe_parallel_config=None,
        activation_checkpointing=False,
    )
    monkeypatch.setattr(
        train_seq_cls, "initialize_distributed", lambda **kwargs: SimpleNamespace(is_main=False, world_size=1)
    )
    monkeypatch.setattr(train_seq_cls, "setup_logging", lambda: None)
    monkeypatch.setattr(train_seq_cls, "apply_cache_compatibility_patches", lambda: None)
    monkeypatch.setattr(train_seq_cls, "StatefulRNG", MagicMock())
    monkeypatch.setattr(
        train_seq_cls, "create_distributed_setup_from_config", lambda cfg, world_size: distributed_setup
    )
    monkeypatch.setattr(train_seq_cls, "build_model", lambda **kwargs: model)
    monkeypatch.setattr(
        train_seq_cls, "shard_optimizers_for_megatron_fsdp", lambda model, optimizer, *a, **kw: optimizer
    )

    def stop(*args, **kwargs):
        raise _StopAfterMFU

    monkeypatch.setattr(train_seq_cls, "_build_tokenizer", stop)

    recipe = TrainFinetuneRecipeForSequenceClassification.__new__(TrainFinetuneRecipeForSequenceClassification)
    recipe.cfg = MagicMock(wandb=None, trackio=None)
    recipe.cfg.get.side_effect = lambda key, default=None: default

    with pytest.raises(_StopAfterMFU):
        recipe.setup()

    assert recipe.mfu_calculator.reference_mfu == expected

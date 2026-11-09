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

"""Parse Flint text recipes through the real typed config and mesh boundaries."""

import copy
from pathlib import Path

import pytest
import torch
import yaml

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.optim.optimizer import AdamWConfig, LRSchedulerConfig
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.recipes._dist_utils import parse_distributed_section
from nemo_automodel.recipes._typed_config import RecipeConfig

_DIRECTORY = Path("examples/llm_finetune/flint")


@pytest.mark.parametrize("name", ["flint_tulu3_sft.yaml", "flint_tulu3_lora.yaml"])
def test_full_and_lora_text_recipes_parse_actual_component_configs(name: str) -> None:
    raw = yaml.safe_load((_DIRECTORY / name).read_text())
    original = copy.deepcopy(raw)
    distributed = parse_distributed_section(raw["distributed"])
    assert isinstance(distributed["strategy_config"], FSDP2Config)
    assert distributed["strategy_config"].mp_policy.reduce_dtype == torch.float32
    assert raw == original
    assert raw["model"]["config"]["router_aux_loss_coef"] == 0.0
    assert raw["model"]["trust_remote_code"] is False
    assert raw["model"]["pretrained_model_name_or_path"] == "RecursiveSI/flint"
    assert raw["dataset"]["path_or_dataset_id"] == "data/frozen_tulu3.jsonl"
    if "peft" in raw:
        assert raw["model"]["dtype"] == "bfloat16"
        assert raw["peft"]["lora_dtype"] == "float32"
        assert "model.layers.*.mlp.experts" in raw["peft"]["target_modules"]
    else:
        # Full SFT holds FP32 masters in FusedAdam, like the long-context recipe.
        assert raw["model"]["dtype"] == "bfloat16"
        assert raw["optimizer"]["master_weight_dtype"] == "float32"
        assert raw["optimizer"]["exp_avg_dtype"] == raw["optimizer"]["exp_avg_sq_dtype"] == "bfloat16"
        pytest.importorskip("transformer_engine.pytorch.optimizers", reason="FusedAdam target needs TE")
    typed = RecipeConfig(ConfigNode(raw))
    assert typed.step_scheduler.max_steps == 100
    if "peft" in raw:
        assert isinstance(typed.optimizer, AdamWConfig)
        assert typed.optimizer.lr > 0
    assert typed.loss_fn.build().chunk_len == 256
    assert typed.checkpoint.v4_compatible is False
    assert typed.checkpoint.cpu_offload is False
    assert typed.step_scheduler.ckpt_every_steps == 50
    assert typed.step_scheduler.save_checkpoint_every_epoch is False
    assert typed.dataloader is not None


@pytest.mark.parametrize("name", ["flint_tulu3_sft.yaml", "flint_tulu3_lora.yaml"])
def test_bounded_control_keeps_full_lr_horizon_and_checkpoint_overlap(name: str) -> None:
    """A clean52-step stop retains the100-step LR/WD objective and overlap saves."""
    raw = yaml.safe_load((_DIRECTORY / name).read_text())
    policies = []
    for maximum in (52, 100):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW(
            [parameter], lr=raw["optimizer"]["lr"], weight_decay=raw["optimizer"]["weight_decay"]
        )
        steps = StepScheduler(
            global_batch_size=4,
            local_batch_size=1,
            dp_size=4,
            dataloader=list(range(32)),
            max_steps=maximum,
            ckpt_every_steps=50 if maximum == 52 else 52,
            save_checkpoint_every_epoch=False,
            preemption_signal=None,
        )
        schedule = LRSchedulerConfig(**raw["lr_scheduler"]).build(optimizer, steps)[0]
        observed = []
        saved = []
        checkpoint50 = None
        for index in range(52):
            steps.step = index
            steps.epoch = index // 32
            if steps.is_ckpt_step:
                saved.append(index + 1)
            schedule.step(1)
            observed.append((optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["weight_decay"]))
            if index == 49:
                checkpoint50 = copy.deepcopy(schedule.state_dict())
        policies.append((observed, saved, checkpoint50, schedule))
    assert policies[0][0] == policies[1][0]
    assert policies[0][1] == [50, 52]
    assert policies[1][1] == [52]
    assert policies[0][2] == policies[1][2]
    policies[1][3].load_state_dict(policies[0][2])
    assert policies[1][3].state_dict() == policies[0][2]


@pytest.mark.parametrize(
    ("name", "attn", "world", "expected"),
    [
        ("flint_tulu3_packed128k_te.yaml", "te", 4, (1, 4, 4, 1)),
        ("flint_tulu3_packed128k_magi_h100.yaml", "magi", 8, (1, 8, 8, 1)),
    ],
)
def test_long_context_recipes_shard_without_pipeline_parallelism(
    name: str, attn: str, world: int, expected: tuple[int, int, int, int]
) -> None:
    raw = yaml.safe_load((_DIRECTORY / name).read_text())
    distributed = parse_distributed_section(raw["distributed"])
    topology = distributed["parallelism_sizes"]
    cp, pp = topology.cp_size, topology.pp_size
    dp = world // (cp * pp * topology.tp_size)
    assert (dp, topology.ep_size, cp, pp) == expected
    # EP reuses the DP x CP ranks: it is not another factor in world size.
    assert (dp * cp) % topology.ep_size == 0
    assert distributed["pipeline_config"] is None
    if attn == "te":
        # TE context parallelism moves the learned sinks with all-to-all, so CP
        # must divide both the 32 query heads and the 4 key/value heads.
        assert 32 % cp == 0 and 4 % cp == 0
        assert raw["step_scheduler"]["local_batch_size"] == raw["dataloader"]["batch_size"] == 2
    else:
        # The Flint Magi path accepts one packed row per step without PP.
        assert raw["step_scheduler"]["local_batch_size"] == raw["dataloader"]["batch_size"] == 1
        assert raw["step_scheduler"]["global_batch_size"] == 2
    assert raw["packed_sequence"]["packed_sequence_size"] == 131072
    assert 131072 <= 262144
    assert raw["model"]["backend"]["attn"] == attn
    assert raw["optimizer"]["master_weight_dtype"] == "float32"
    assert raw["optimizer"]["exp_avg_dtype"] == raw["optimizer"]["exp_avg_sq_dtype"] == "bfloat16"
    assert raw["checkpoint"]["cpu_offload"] is False

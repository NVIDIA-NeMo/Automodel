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

"""Configuration contract for the Qwen3.5-122B packed CP/EP example."""

from pathlib import Path

import yaml

from nemo_automodel.components.distributed.cp_vision_shard import CpVisionShardingConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "examples/vlm_finetune/qwen3_5_moe/qwen3_5_122b_128k_ep8cp16.yaml"


def test_qwen3_5_moe_122b_example_declares_scaling_contract() -> None:
    """The example selects the 122B model and its required CP/EP features."""
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert raw_config["model"]["pretrained_model_name_or_path"] == "Qwen/Qwen3.5-122B-A10B"
    assert raw_config["model"]["attn_implementation"] == "sdpa"
    assert raw_config["model"]["text_config"]["output_hidden_states"] is True
    assert raw_config["model"]["text_config"]["mtp_expert_hf_layout"] == "split"
    assert raw_config["distributed"]["cp_size"] == 16
    assert raw_config["distributed"]["ep_size"] == 8
    assert raw_config["distributed"]["activation_checkpointing"] is True
    assert raw_config["freeze_config"]["freeze_vision_tower"] is False
    assert raw_config["packed_sequence"]["pack_size"] == 131072
    assert raw_config["packed_sequence"]["attn_implementation"] == "sdpa"
    assert raw_config["step_scheduler"]["max_steps"] == 10

    assert raw_config["prewarm"] == {
        "cublas_backward": True,
        "fla_gdn_autotune": True,
        "comm_groups": True,
    }

    optimizer = raw_config["optimizer"]
    assert optimizer["_target_"] == "transformer_engine.pytorch.optimizers.FusedAdam"
    assert optimizer["master_weights"] is True
    assert optimizer["master_weight_dtype"] == "torch.float32"
    assert optimizer["store_param_remainders"] is True
    assert optimizer["exp_avg_dtype"] == "torch.bfloat16"
    assert optimizer["exp_avg_sq_dtype"] == "torch.float32"

    vision_policy = CpVisionShardingConfig(**raw_config["distributed"]["cp_vision_sharding"])
    assert vision_policy == CpVisionShardingConfig(enabled=True, min_tokens=0, cost_alpha="auto")

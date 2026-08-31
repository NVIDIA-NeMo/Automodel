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

"""Contract tests for the HY4-preview full-parameter recipe."""

from pathlib import Path

import yaml

_RECIPE = (
    Path(__file__).resolve().parents[3] / "examples" / "llm_finetune" / "hy_v4" / "hy4_preview_tulu3_4k_cudnn.yaml"
)


def test_hy4_preview_recipe_preserves_validated_memory_and_parallelism_contract():
    """The production recipe must retain the topology used by the full-model run."""
    config = yaml.safe_load(_RECIPE.read_text(encoding="utf-8"))
    distributed = config["distributed"]
    scheduler = config["step_scheduler"]
    backend = config["model"]["backend"]

    assert scheduler["global_batch_size"] == 256
    assert scheduler["local_batch_size"] == 8
    assert scheduler["max_steps"] == 100
    assert distributed["strategy"] == "fsdp2"
    assert distributed["tp_size"] == 1
    assert distributed["cp_size"] == 1
    assert distributed["pp_size"] == 8
    assert distributed["ep_size"] == 32
    assert distributed["activation_checkpointing"] is True
    world_size = config["ci"]["nodes"] * 8
    dp_size = world_size // (distributed["tp_size"] * distributed["cp_size"] * distributed["pp_size"])
    assert dp_size == distributed["ep_size"] == 32
    assert scheduler["global_batch_size"] == scheduler["local_batch_size"] * dp_size
    assert distributed["pipeline"] == {
        "pp_schedule": "interleaved1f1b",
        "pp_microbatch_size": 1,
        "round_virtual_stages_to_pp_multiple": "down",
        "scale_grads_in_schedule": False,
        "patch_inner_model": False,
        "patch_causal_lm_model": False,
        "layers_per_stage": 2,
    }
    assert distributed["moe"]["reshard_after_forward"] is True
    assert distributed["moe"]["wrap_outer_model"] is False
    assert distributed["moe"]["ignore_router_for_ac"] is True
    assert config["prewarm"] == {"comm_groups": True}
    assert config["loss_fn"] == {"_target_": "nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy"}
    assert config["optimizer"]["foreach"] is False
    assert backend["attn"] == "cudnn"
    assert backend["experts"] == "torch_mm"
    assert backend["dispatcher"] == "hybridep"
    assert backend["dispatcher_num_sms"] == 8
    assert backend["rope_fusion"] is False
    assert config["packed_sequence"]["packed_sequence_size"] == 4096
    assert config["dataset"]["path_or_dataset_id"] == "allenai/tulu-3-sft-mixture"
    assert config["dataset"]["split"] == "train[:100000]"
    assert config["wandb"]["enable"] is True
    assert config["wandb"]["mode"] == "online"
    assert config["ci"]["nodes"] == 32
    assert config["ci"]["max_steps"] == 100
    assert config["ci"]["env_vars"]["FINETUNE_ARGS"] == "--wandb.enable=true"
    assert config["ci"]["env_vars"]["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == "8"
    assert config["ci"]["env_vars"]["NUM_OF_STAGES_G2S_COMBINE_API"] == "4"

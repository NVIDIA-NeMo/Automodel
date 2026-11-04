# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration contract for the single-GPU Nemotron 3.5 Bolt LoRA example."""

from pathlib import Path

import yaml
from transformers.core_model_loading import WeightRenaming

from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.config.loader import load_yaml_config

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "examples" / "vlm_finetune" / "nemotron_3_5_bolt" / "nemotron_3_5_bolt_cord_v2_lora.yaml"


def test_nemotron_3_5_bolt_lora_uses_force_hf_without_expert_parallelism() -> None:
    """The example must stay on the single-GPU force-HF BF16 LoRA path."""
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = load_yaml_config(CONFIG_PATH)
    peft = config.peft.instantiate()

    assert isinstance(peft, PeftConfig)
    assert peft.dim == 8
    assert peft.alpha == 32
    assert peft.use_memory_efficient_lora is True
    assert peft.use_triton is True

    model = raw_config["model"]
    assert model["pretrained_model_name_or_path"] == ("nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16")
    assert model["force_hf"] is True
    assert model["trust_remote_code"] is True
    assert model["torch_dtype"] == "torch.bfloat16"
    assert model["attn_implementation"] == "eager"
    assert "revision" not in model
    assert "revision" not in raw_config["processor"]
    assert "backend" not in model
    assert "quantization" not in raw_config

    expected_key_mapping = {
        r"^language_model\.backbone\.": "language_model.model.",
        r"^mlp1\.0\.": "vision_projector.mlp1.norm.",
        r"^mlp1\.1\.": "vision_projector.mlp1.linear1.",
        r"^mlp1\.3\.": "vision_projector.mlp1.linear2.",
    }
    assert model["key_mapping"] == expected_key_mapping

    expected_renames = {
        "language_model.backbone.layers.0.mixer.in_proj.weight": ("language_model.model.layers.0.mixer.in_proj.weight"),
        "mlp1.0.weight": "vision_projector.mlp1.norm.weight",
        "mlp1.1.weight": "vision_projector.mlp1.linear1.weight",
        "mlp1.3.weight": "vision_projector.mlp1.linear2.weight",
    }
    for checkpoint_key, expected_model_key in expected_renames.items():
        renamed_key = checkpoint_key
        for source_pattern, target_pattern in expected_key_mapping.items():
            renamed_key, _ = WeightRenaming(source_pattern, target_pattern).rename_source_key(renamed_key)
        assert renamed_key == expected_model_key

    assert len(peft.target_modules) == 9
    assert all(pattern.startswith("*language_model.") for pattern in peft.target_modules)
    assert not any(".experts." in pattern for pattern in peft.target_modules)
    assert not any(pattern.endswith(".out_proj") for pattern in peft.target_modules)

    distributed = config.distributed
    assert distributed.strategy == "fsdp2"
    assert distributed.dp_size == 1
    assert distributed.tp_size == 1
    assert distributed.cp_size == 1
    assert distributed.pp_size == 1
    assert distributed.ep_size == 1
    assert distributed.activation_checkpointing is False
    assert distributed.reshard_after_forward is True

    optimizer = raw_config["optimizer"]
    assert optimizer["exp_avg_dtype"] == "torch.float32"
    assert optimizer["exp_avg_sq_dtype"] == "torch.float32"

    assert config.step_scheduler.global_batch_size == 1
    assert config.step_scheduler.local_batch_size == 1
    assert config.step_scheduler.max_steps == 100
    assert config.dataset.path_or_dataset == "naver-clova-ix/cord-v2"
    assert config.dataloader.collate_fn.max_length == 4096
    assert config.validation_dataloader.collate_fn.max_length == 4096
    assert config.ci.cluster_tag == "gb300"
    assert config.ci.nodes == 1
    assert config.ci.nproc_per_node == 1
    assert config.ci.time == "00:30:00"
    assert "env_vars" not in raw_config["ci"]

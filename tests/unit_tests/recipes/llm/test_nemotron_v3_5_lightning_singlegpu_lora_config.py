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

"""Configuration contract for the single-GPU Nemotron 3.5 Lightning LoRA example."""

from pathlib import Path

import yaml

from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.components.models.common import BackendConfig

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples" / "llm_finetune" / "nemotron" / "nemotron_nano_v3_5_lightning_singlegpu_lora.yaml"


def test_nemotron_v3_5_lightning_lora_is_strictly_single_gpu() -> None:
    """The example must retain its BF16 LoRA, MTP, memory, and topology contract."""
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config = load_yaml_config(CONFIG_PATH)

    backend = config.model.backend.instantiate()
    peft = config.peft.instantiate()

    assert isinstance(backend, BackendConfig)
    assert backend.attn == "te"
    assert backend.linear == "torch"
    assert backend.rms_norm == "torch_fp32"
    assert backend.experts == "torch_mm"
    assert backend.dispatcher == "torch"

    assert isinstance(peft, PeftConfig)
    assert peft.dim == 8
    assert peft.alpha == 32
    assert peft.use_memory_efficient_lora is True

    assert config.model.pretrained_model_name_or_path == ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    assert config.model.num_nextn_predict_layers == 2
    assert config.model.mtp_use_repeated_layer is True
    assert config.model.mtp_loss_scaling_factor == 0.1
    assert "quantization" not in raw_config

    assert config.step_scheduler.local_batch_size == 1
    assert config.distributed.strategy == "fsdp2"
    assert config.distributed.dp_size == 1
    assert config.distributed.tp_size == 1
    assert config.distributed.cp_size == 1
    assert config.distributed.ep_size == 1
    assert config.distributed.activation_checkpointing is True
    assert config.ci.nproc_per_node == 1

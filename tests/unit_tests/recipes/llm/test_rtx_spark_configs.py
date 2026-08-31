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

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
RTX_SPARK_CONFIGS = (
    "examples/llm_finetune/llama3_1/llama3_1_8b_squad_peft_rtx_spark.yaml",
    "examples/llm_finetune/qwen/qwen3_8b_squad_rtx_spark.yaml",
    "examples/llm_finetune/llama3_3/llama_3_3_70b_instruct_squad_peft_qlora_rtx_spark.yaml",
)


def _load_config(relative_path: str) -> dict:
    config_path = REPO_ROOT / relative_path
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("relative_path", RTX_SPARK_CONFIGS)
def test_rtx_spark_configs_target_single_gpu_gb10(relative_path):
    config = _load_config(relative_path)

    assert Path(relative_path).stem.endswith("_rtx_spark")
    assert config["ci"]["cluster_tag"] == "gb10"
    assert config["ci"]["nproc_per_node"] == 1
    assert config["ci"]["nodes"] == 1
    assert config["distributed"]["strategy"] == "fsdp2"
    assert config["distributed"]["tp_size"] == 1
    assert config["distributed"]["cp_size"] == 1


@pytest.mark.parametrize(
    ("relative_path", "model_name"),
    (
        (RTX_SPARK_CONFIGS[0], "meta-llama/Llama-3.1-8B"),
        (RTX_SPARK_CONFIGS[1], "Qwen/Qwen3-8B"),
    ),
)
def test_rtx_spark_8b_packed_recipes_use_supported_attention(relative_path, model_name):
    config = _load_config(relative_path)

    assert config["model"]["pretrained_model_name_or_path"] == model_name
    assert config["model"]["force_hf"] is True
    assert config["model"]["attn_implementation"] == "flash_attention_2"
    assert config["model"]["disable_mmap"] is True
    assert config["step_scheduler"]["global_batch_size"] == 1
    assert config["step_scheduler"]["local_batch_size"] == 1
    assert config["packed_sequence"]["packed_sequence_size"] == 128
    assert config["peft"]["_target_"] == "nemo_automodel.components._peft.lora.PeftConfig"
    assert config["distributed"]["activation_checkpointing"] is True


def test_rtx_spark_llama_8b_matches_64gb_uma_regression_settings():
    config = _load_config(RTX_SPARK_CONFIGS[0])

    assert config["step_scheduler"]["global_batch_size"] == 1
    assert config["step_scheduler"]["local_batch_size"] == 1
    assert config["packed_sequence"]["packed_sequence_size"] == 128
    assert config["distributed"]["activation_checkpointing"] is True


def test_rtx_spark_llama_70b_preserves_streaming_qlora_settings():
    config = _load_config(RTX_SPARK_CONFIGS[2])

    assert config["model"]["pretrained_model_name_or_path"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert config["quantization"]["load_in_4bit"] is True
    assert config["quantization"]["load_in_8bit"] is False
    assert config["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert config["model"]["disable_mmap"] is True
    assert config["step_scheduler"]["global_batch_size"] == 1
    assert config["step_scheduler"]["local_batch_size"] == 1
    assert config["packed_sequence"]["packed_sequence_size"] == 1024
    assert config["distributed"]["activation_checkpointing"] is True

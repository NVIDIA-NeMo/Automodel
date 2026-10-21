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
from datasets import Dataset

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from tests.ci_tests.utils.generate_ci_tests import generate_pipeline

EXAMPLES = [
    ("examples/llm_finetune/baai/aquila_7b_squad_peft.yaml", "BAAI/Aquila-7B"),
    ("examples/llm_finetune/deepseek_ai/deepseek_llm_7b_chat_squad_peft.yaml", "deepseek-ai/deepseek-llm-7b-chat"),
    ("examples/llm_finetune/eleutherai/gpt_j_6b_squad_peft.yaml", "EleutherAI/gpt-j-6b"),
    ("examples/llm_finetune/eleutherai/gpt_neox_20b_squad_peft.yaml", "EleutherAI/gpt-neox-20b"),
    ("examples/llm_finetune/inceptionai/jais_13b_squad_peft.yaml", "inceptionai/jais-13b"),
    ("examples/llm_finetune/internlm/internlm3_8b_instruct_squad_peft.yaml", "internlm/internlm3-8b-instruct"),
    (
        "examples/llm_finetune/lgai_exaone/exaone_3_0_7_8b_instruct_squad_peft.yaml",
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    ),
    (
        "examples/llm_finetune/nvidia/llama_3_3_nemotron_super_49b_v1_squad_peft.yaml",
        "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    ),
    ("examples/llm_finetune/nvidia/minitron_8b_base_squad_peft.yaml", "nvidia/Minitron-8B-Base"),
    ("examples/llm_finetune/parasail_ai/gritlm_7b_vllm_squad_peft.yaml", "parasail-ai/GritLM-7B-vllm"),
    ("examples/llm_finetune/stabilityai/stablelm_3b_4e1t_squad_peft.yaml", "stabilityai/stablelm-3b-4e1t"),
    ("examples/llm_finetune/tiiuae/falcon_7b_squad_peft.yaml", "tiiuae/falcon-7b"),
    ("examples/llm_finetune/upstage/solar_pro_preview_instruct_squad_peft.yaml", "upstage/solar-pro-preview-instruct"),
]


@pytest.mark.parametrize(("relative_path", "checkpoint"), EXAMPLES)
def test_example_loads_with_consistent_checkpoint_and_batch(relative_path: str, checkpoint: str) -> None:
    config = RecipeConfig(load_yaml_config(relative_path))
    assert config.model.pretrained_model_name_or_path == checkpoint
    assert config.step_scheduler.max_steps == 100
    assert config.checkpoint.enabled
    assert config.optimizer is not None
    assert config.dataloader is not None
    assert config.validation_dataloaders
    raw = yaml.safe_load(Path(relative_path).read_text())
    if "tokenizer" in raw:
        assert raw["tokenizer"]["pretrained_model_name_or_path"] == checkpoint
    schedule = raw["step_scheduler"]
    assert schedule["global_batch_size"] % (schedule["local_batch_size"] * 8) == 0
    assert raw["checkpoint"]["model_save_format"] == "safetensors"
    assert raw["peft"]["dim"] > 0
    assert raw["dataset"]["_target_"] == raw["validation_dataset"]["_target_"]


@pytest.mark.parametrize("scope", ["test", "nightly", "release", "performance"])
def test_unvalidated_examples_do_not_expand_gpu_ci(scope: str) -> None:
    scheduled = set()
    for folder in ["llm_finetune"]:
        pipeline = generate_pipeline(".", scope, folder)
        scheduled.update(
            job["variables"]["CONFIG_PATH"]
            for job in pipeline.values()
            if isinstance(job, dict) and "CONFIG_PATH" in job.get("variables", {})
        )
    assert not {path for path, _ in EXAMPLES}.intersection(scheduled)


@pytest.mark.with_downloads
@pytest.mark.parametrize(
    "relative_path,revision",
    [
        ("examples/llm_finetune/baai/aquila_7b_squad_peft.yaml", "24caff099d1ed30850b0e8e3a40250af73f93346"),
        ("examples/llm_finetune/eleutherai/gpt_j_6b_squad_peft.yaml", "47e169305d2e8376be1d31e765533382721b2cc1"),
        ("examples/llm_finetune/eleutherai/gpt_neox_20b_squad_peft.yaml", "c292233c833e336628618a88a648727eb3dff0a7"),
        ("examples/llm_finetune/nvidia/minitron_8b_base_squad_peft.yaml", "e2c30f2d4c8751f21b33be2b403d10d5f373ffc1"),
        (
            "examples/llm_finetune/stabilityai/stablelm_3b_4e1t_squad_peft.yaml",
            "fa4a6a92fca83c3b4223a3c9bf792887090ebfba",
        ),
        ("examples/llm_finetune/tiiuae/falcon_7b_squad_peft.yaml", "ec89142b67d748a1865ea4451372db8313ada0d8"),
    ],
)
def test_native_recipe_batches_preserve_complete_answer(
    relative_path: str, revision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real checkpoint tokenizers through both configured loaders, without model weights."""
    config = RecipeConfig(load_yaml_config(relative_path))
    tokenizer = NeMoAutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path, revision=revision)
    for answer in ["Kyoto", "Cobalt"]:
        rows = Dataset.from_list(
            [
                {
                    "context": "The observatory is located in Kyoto.",
                    "question": "Where is the observatory located?",
                    "answers": {"text": [answer], "answer_start": [34]},
                }
            ]
        )
        monkeypatch.setattr("nemo_automodel.components.datasets.llm.squad.load_dataset", lambda *args, **kwargs: rows)
        for loader_config in [config.dataloader, *config.validation_dataloaders.values()]:
            loader = loader_config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
            batch = next(iter(loader))
            supervised = batch["labels"][0]
            supervised = supervised[supervised != -100].tolist()
            assert tokenizer.decode(supervised, skip_special_tokens=True).strip() == answer

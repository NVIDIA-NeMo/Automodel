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

from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.recipes._typed_config import RecipeConfig
from tests.ci_tests.utils.generate_ci_tests import generate_pipeline

EXAMPLES = [
    ("examples/retrieval/bi_encoder/llama_embed_nemotron_8b_finetune.yaml", "nvidia/llama-embed-nemotron-8b"),
    ("examples/retrieval/bi_encoder/ministral_3_3b_base_2512_finetune.yaml", "mistralai/Ministral-3-3B-Base-2512"),
    (
        "examples/retrieval/cross_encoder/llama_nemotron_rerank_1b_v2_finetune.yaml",
        "nvidia/llama-nemotron-rerank-1b-v2",
    ),
]


@pytest.mark.parametrize(("relative_path", "checkpoint"), EXAMPLES)
def test_example_loads_with_consistent_checkpoint_and_batch(relative_path: str, checkpoint: str) -> None:
    config = RecipeConfig(load_yaml_config(relative_path))
    assert config.model.pretrained_model_name_or_path == checkpoint
    assert config.step_scheduler.max_steps == 100
    assert config.checkpoint.enabled
    assert config.optimizer is not None
    raw = yaml.safe_load(Path(relative_path).read_text())
    if "tokenizer" in raw:
        assert raw["tokenizer"]["pretrained_model_name_or_path"] == checkpoint
    schedule = raw["step_scheduler"]
    assert schedule["global_batch_size"] % (schedule["local_batch_size"] * 8) == 0
    assert raw["checkpoint"]["model_save_format"] == "safetensors"
    assert raw["model"]["dtype"] == "float32"


@pytest.mark.parametrize("scope", ["test", "nightly", "release", "performance"])
def test_unvalidated_examples_do_not_expand_gpu_ci(scope: str) -> None:
    scheduled = set()
    for folder in ["retrieval_bi_encoder", "retrieval_cross_encoder"]:
        pipeline = generate_pipeline(".", scope, folder)
        scheduled.update(
            job["variables"]["CONFIG_PATH"]
            for job in pipeline.values()
            if isinstance(job, dict) and "CONFIG_PATH" in job.get("variables", {})
        )
    assert not {path for path, _ in EXAMPLES}.intersection(scheduled)

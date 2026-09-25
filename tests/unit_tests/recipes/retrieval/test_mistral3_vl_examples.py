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

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
_PROCESSOR_TARGET = (
    "nemo_automodel.components.models.ministral_bidirectional.processor.Mistral3BiEncoderProcessor.from_pretrained"
)
_EXAMPLES = [
    (
        _REPO_ROOT / "examples/retrieval/bi_encoder/mistral3_vl_embedding.yaml",
        "TrainBiEncoderRecipe",
        "bi_encoder",
        "process_queries_documents_biencoder",
    ),
    (
        _REPO_ROOT / "examples/retrieval/cross_encoder/mistral3_vl_reranker.yaml",
        "TrainCrossEncoderRecipe",
        "cross_encoder",
        "process_queries_documents_crossencoder",
    ),
]


@pytest.mark.parametrize(("config_path", "recipe", "model_type", "collator_method"), _EXAMPLES)
def test_mistral3_vl_example_is_portable(
    config_path: Path,
    recipe: str,
    model_type: str,
    collator_method: str,
) -> None:
    """The release examples use one public base model and no internal run paths."""
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)

    assert config["recipe"] == recipe
    assert config["model"]["pretrained_model_name_or_path"] == _MODEL_ID
    assert config["tokenizer"]["pretrained_model_name_or_path"] == _MODEL_ID
    assert config["tokenizer"]["_target_"] == _PROCESSOR_TARGET
    assert config["dataset"]["model_type"] == model_type
    assert config["dataset"]["use_text_in_document"] is False
    assert config["dataloader"]["collate_fn"]["collator_fn_name"] == collator_method
    assert len(config["dataset"]["data_dir_list"]) == 1
    assert not any(marker in config_text for marker in ("/lustre/", "submit_nemo_run", ".outputs/"))


def test_mistral3_vl_reranker_applies_temperature_once() -> None:
    """The reranker delegates non-unit temperature scaling to the model."""
    config_path = _EXAMPLES[1][0]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["temperature"] == 1.0
    assert config["model"]["temperature"] == 0.02

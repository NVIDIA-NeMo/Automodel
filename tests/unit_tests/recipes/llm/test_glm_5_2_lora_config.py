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

"""Configuration contract for the GLM-5.2 LoRA example."""

from pathlib import Path

from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater
from nemo_automodel.components.models.common import BackendConfig

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples" / "llm_finetune" / "glm" / "glm_5.2_lora.yaml"


def test_glm_5_2_lora_config_contract() -> None:
    """The example must preserve its backend, topology, and sequence-length contract."""
    config = load_yaml_config(CONFIG_PATH)

    backend = config.model.backend.instantiate()

    assert isinstance(backend, BackendConfig)
    assert backend.attn == "tilelang"
    assert backend.experts == "torch_mm"
    assert backend.dispatcher == "hybridep"
    assert config.model.pretrained_model_name_or_path == "zai-org/GLM-5.2"
    assert config.dataset.seq_length == 4096
    assert config.dataset.truncation is True
    assert config.dataset.padding is False
    assert config.validation_dataset.seq_length == 4096
    assert config.validation_dataset.truncation is True
    assert config.validation_dataset.padding is False
    assert config.packed_sequence.packed_sequence_size == 4096
    assert config.distributed.reshard_after_forward is True
    assert (
        config.step_scheduler.global_batch_size % (config.step_scheduler.local_batch_size * config.distributed.ep_size)
        == 0
    )
    assert config.dataloader.collate_fn is packed_sequence_thd_collater
    assert config.validation_dataloader.collate_fn is packed_sequence_thd_collater
    assert config.ci.time == "00:25:00"
    release_overrides = config.ci.release.to_dict()
    assert release_overrides["step_scheduler.max_steps"] == 20
    assert release_overrides["packed_sequence.max_packs"] == 2560

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

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from nemo_automodel._transformers.models.common import BackendConfig
from nemo_automodel._transformers.models.inkling.configuration import InklingConfig
from nemo_automodel._transformers.models.inkling.model import InklingForConditionalGeneration
from nemo_automodel._transformers.registry import ModelRegistry

from .parity_check_inkling import build_tiny_config


def test_inkling_config_registered_with_auto_config():
    assert CONFIG_MAPPING["inkling_mm_model"] is InklingConfig

    cfg = AutoConfig.for_model("inkling_mm_model", architectures=["InklingForConditionalGeneration"])

    assert isinstance(cfg, InklingConfig)
    assert cfg.model_type == "inkling_mm_model"
    assert cfg.text_config.model_type == "inkling_text"


def test_inkling_checkpoint_aliases_and_placeholder_defaults():
    cfg = InklingConfig.from_dict(
        {
            "model_type": "inkling_mm_model",
            "image_token_id": None,
            "audio_token_id": None,
            "text_config": {"hidden_size": 64, "num_hidden_layers": 2},
            "vision_config": {"n_layers": 4, "n_channels": 5, "decoder_dmodel": 64},
            "audio_config": {"n_mel_bins": 8, "mel_vocab_size": 16, "decoder_dmodel": 64},
        }
    )

    assert cfg.image_token_id == 200054
    assert cfg.audio_token_id == 200053
    assert cfg.vision_config.num_hidden_layers == 4
    assert cfg.vision_config.num_channels == 5
    assert cfg.vision_config.text_hidden_size == 64
    assert cfg.audio_config.text_hidden_size == 64


def test_inkling_architecture_instantiates_from_local_config():
    assert ModelRegistry.has_custom_model("InklingForConditionalGeneration")
    backend = BackendConfig(linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    model = InklingForConditionalGeneration.from_config(build_tiny_config(), backend=backend)
    assert isinstance(model, InklingForConditionalGeneration)
    assert model.config.model_type == "inkling_mm_model"

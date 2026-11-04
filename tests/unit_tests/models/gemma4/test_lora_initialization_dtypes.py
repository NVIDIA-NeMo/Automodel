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

"""Dense Gemma4 initialization must preserve the prepared LoRA storage policy."""

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import prepare_frozen_parameters_for_sharding
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration


def test_dense_gemma4_initialization_preserves_fp32_lora_storage():
    text_config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=32,
        enable_moe_block=False,
        layer_types=["full_attention"],
        sliding_window=16,
        torch_dtype="float32",
    )
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=text_config),
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        ),
    ).float()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["*"], dim=4, lora_dtype=torch.float32))

    prepare_frozen_parameters_for_sharding(model, torch.bfloat16)
    assert next(parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()) == torch.bfloat16
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters() if parameter.requires_grad)

    # This is the dtype Checkpointer infers from the first parameter after preparation.
    model.initialize_weights(dtype=torch.bfloat16, buffer_device=torch.device("cpu"))

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.dtype == torch.float32 for parameter in trainable)
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters() if not parameter.requires_grad)

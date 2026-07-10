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

"""Shared DSpark target-identification and tokenization helpers."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig

from nemo_automodel.components.datasets.llm.formatting_utils import _has_chat_template, _resolve_chat_template

GEMMA4_MODEL_TYPES = ("gemma4", "gemma4_unified")
DEEPSEEK_V4_MODEL_TYPE = "deepseek_v4"
GLM_5_2_MODEL_TYPE = "glm_moe_dsa"
MINIMAX_M3_MODEL_TYPES = ("minimax_m3_vl",)


def read_target_model_type(target_path: str, trust_remote_code: bool) -> str:
    """Return the target HF ``model_type`` without instantiating the model when possible."""
    try:
        config_dict, _ = PretrainedConfig.get_config_dict(target_path, trust_remote_code=trust_remote_code)
        model_type = config_dict.get("model_type")
        if model_type:
            return str(model_type)
    except (OSError, ValueError, KeyError):
        pass
    config = AutoConfig.from_pretrained(target_path, trust_remote_code=trust_remote_code)
    return str(getattr(config, "model_type", "") or "")


def apply_target_chat_template(tokenizer, chat_template) -> None:
    """Attach or validate the chat template used to tokenize messages-format data."""
    if chat_template is not None:
        tokenizer.chat_template = _resolve_chat_template(str(chat_template))
        return
    if not _has_chat_template(tokenizer):
        raise ValueError(
            "The target tokenizer has no chat template and --chat-template was not set. "
            "DSpark needs the same template for training and offline precompute."
        )

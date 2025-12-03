# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch.nn as nn


def is_tied_word_embeddings(model: nn.Module) -> bool:
    """
    Check if the model's word embeddings are tied.

    Args:
        model (nn.Module): The model to check.

    Returns:
        bool: True if the model's word embeddings are tied, False otherwise.
    """
    non_tied_lm_head_models = {
        "Qwen3OmniMoeThinkerForConditionalGeneration",  # complicated config structure
    }
    for m in non_tied_lm_head_models:
        if m in type(model[0]).__name__:
            return False
    config = getattr(model, "config", None)
    text_config = getattr(config, "get_text_config", lambda: None)()
    return bool(getattr(text_config, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)))

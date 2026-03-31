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

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_utils import _get_resolved_checkpoint_files, load_state_dict

from nemo_automodel.components.utils.model_utils import resolve_trust_remote_code


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
    model_class_name = type(model).__name__
    for m in non_tied_lm_head_models:
        if m in model_class_name:
            return False
    config = getattr(model, "config", None)
    text_config = getattr(config, "get_text_config", lambda: None)()
    return bool(getattr(text_config, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)))


def _get_checkpoint_tensor_dtypes(
    pretrained_model_name_or_path: str,
    hf_config: Any,
    load_kwargs: Mapping[str, object] | None = None,
) -> dict[str, torch.dtype]:
    """Inspect checkpoint tensors and return their exact dtypes by key.

    This reads checkpoint metadata only by loading tensors on the ``meta``
    device, so it preserves the per-tensor dtype information without
    materializing full checkpoint weights in memory.
    """
    load_kwargs = dict(load_kwargs or {})

    provided_state_dict = load_kwargs.get("state_dict")
    if isinstance(provided_state_dict, Mapping):
        return {name: tensor.dtype for name, tensor in provided_state_dict.items() if isinstance(tensor, torch.Tensor)}

    if load_kwargs.get("gguf_file") is not None:
        return {}

    trust_remote_code = load_kwargs.get(
        "trust_remote_code",
        resolve_trust_remote_code(pretrained_model_name_or_path),
    )
    download_kwargs = {
        key: load_kwargs[key]
        for key in (
            "cache_dir",
            "force_download",
            "proxies",
            "local_files_only",
            "token",
            "revision",
            "subfolder",
            "commit_hash",
        )
        if key in load_kwargs
    }
    checkpoint_files, _ = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        variant=load_kwargs.get("variant"),
        gguf_file=load_kwargs.get("gguf_file"),
        use_safetensors=load_kwargs.get("use_safetensors"),
        user_agent={"file_type": "model", "framework": "pytorch"},
        is_remote_code=bool(trust_remote_code),
        transformers_explicit_filename=getattr(hf_config, "transformers_weights", None),
        download_kwargs=download_kwargs,
    )
    if not checkpoint_files:
        return {}

    checkpoint_dtypes: dict[str, torch.dtype] = {}
    weights_only = bool(load_kwargs.get("weights_only", True))
    for checkpoint_file in checkpoint_files:
        state_dict = load_state_dict(checkpoint_file, map_location="meta", weights_only=weights_only)
        checkpoint_dtypes.update(
            {name: tensor.dtype for name, tensor in state_dict.items() if isinstance(tensor, torch.Tensor)}
        )
    return checkpoint_dtypes

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""HF checkpoint key renames for Gemma4 Unified.

Transformers renames six Gemma4 Unified vision keys while loading a checkpoint, so the
in-memory FQNs differ from the names the original HF checkpoint uses. ``gemma4_unified`` is
HF-native and has no ``state_dict_adapter``, so nothing else reverses those renames; the
checkpointer applies the table below on its save and DCP-resume paths instead.
"""

import torch
from torch import nn

GEMMA4_UNIFIED_MODEL_TYPE = "gemma4_unified"

HF_EXPORT_KEY_RENAMES = {
    "embed_vision.patch_ln1": "vision_embedder.patch_ln1",
    "embed_vision.patch_dense": "vision_embedder.patch_dense",
    "embed_vision.patch_ln2": "vision_embedder.patch_ln2",
    "embed_vision.pos_embedding": "vision_embedder.pos_embedding",
    "embed_vision.pos_norm": "vision_embedder.pos_norm",
    "embed_vision.multimodal_embedder.embedding_projection": "embed_vision.embedding_projection",
}


def maybe_rename_gemma4_unified_keys(
    model_part: nn.Module, state_dict: dict[str, torch.Tensor], to_hf: bool
) -> dict[str, torch.Tensor]:
    """Translate Gemma4 Unified keys between model FQNs and the HF checkpoint layout.

    Without this, saved checkpoints (both the DCP shards and the consolidated safetensors
    merged from them) carry internal names such as
    ``embed_vision.multimodal_embedder.embedding_projection.weight``.

    This is only applied on the save and DCP-resume paths, which read and write checkpoints
    Automodel itself produced. Loading a base HF checkpoint goes through the storage reader's
    ``key_mapping`` (built from the model's ``_checkpoint_conversion_mapping``), which already
    performs the HF-to-model translation; renaming there too would double-transform the keys.

    Args:
        model_part: Model part (already unwrapped from DDP) whose ``config.model_type``
            selects the renaming.
        state_dict: State dict to translate.
        to_hf: If True, rename model FQNs to HF FQNs; otherwise apply the inverse.

    Returns:
        The translated state dict, or ``state_dict`` unchanged for other model types.

    Raises:
        ValueError: If two source keys collide on the same renamed key.
    """
    if getattr(getattr(model_part, "config", None), "model_type", None) != GEMMA4_UNIFIED_MODEL_TYPE:
        return state_dict
    if to_hf:
        renames = HF_EXPORT_KEY_RENAMES
    else:
        renames = {hf_key: model_key for model_key, hf_key in HF_EXPORT_KEY_RENAMES.items()}

    renamed_state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        prefix = "model." if key.startswith("model.") else ""
        unprefixed_key = key[len(prefix) :]
        for source_key, target_key in renames.items():
            if unprefixed_key == source_key or unprefixed_key.startswith(f"{source_key}."):
                unprefixed_key = f"{target_key}{unprefixed_key[len(source_key) :]}"
                break
        renamed_key = f"{prefix}{unprefixed_key}"
        if renamed_key in renamed_state_dict:
            raise ValueError(f"Gemma4 Unified HF export key collision for {renamed_key!r}")
        renamed_state_dict[renamed_key] = tensor
    return renamed_state_dict

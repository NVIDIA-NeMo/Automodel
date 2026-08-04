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

"""State dict adapter for Gemma4 Unified.

Transformers renames six Gemma4 Unified vision keys while loading a checkpoint, so the
in-memory FQNs differ from the names the original HF checkpoint uses::

    checkpoint                              model
    vision_embedder.patch_ln1               embed_vision.patch_ln1
    ...
    embed_vision.embedding_projection       embed_vision.multimodal_embedder.embedding_projection

``gemma4_unified`` is HF-native, so there is no model implementation under
``components/models`` to carry an adapter; ``_init_model`` attaches this one to the model
instance instead. With the adapter attached, the checkpointer's existing ``to_hf`` /
``from_hf`` hooks cover save, base-model init, and DCP resume symmetrically, and the
storage reader's ``key_mapping`` is disabled so the keys are translated exactly once.
"""

import re
from typing import Any

GEMMA4_UNIFIED_MODEL_TYPE = "gemma4_unified"

# model FQN prefix -> HF checkpoint FQN prefix.
HF_EXPORT_KEY_RENAMES = {
    "embed_vision.patch_ln1": "vision_embedder.patch_ln1",
    "embed_vision.patch_dense": "vision_embedder.patch_dense",
    "embed_vision.patch_ln2": "vision_embedder.patch_ln2",
    "embed_vision.pos_embedding": "vision_embedder.pos_embedding",
    "embed_vision.pos_norm": "vision_embedder.pos_norm",
    "embed_vision.multimodal_embedder.embedding_projection": "embed_vision.embedding_projection",
}

_HF_IMPORT_KEY_RENAMES = {hf_key: model_key for model_key, hf_key in HF_EXPORT_KEY_RENAMES.items()}


def _rename_keys(state_dict: dict[str, Any], renames: dict[str, str]) -> dict[str, Any]:
    """Apply prefix ``renames`` to every key, tolerating an optional ``model.`` prefix.

    Args:
        state_dict: State dict whose keys are renamed.
        renames: Source prefix -> target prefix. The first matching entry wins.

    Returns:
        A new state dict with renamed keys.

    Raises:
        ValueError: If two source keys collide on the same renamed key.
    """
    renamed_state_dict: dict[str, Any] = {}
    for key, tensor in state_dict.items():
        prefix = "model." if key.startswith("model.") else ""
        unprefixed_key = key[len(prefix) :]
        for source_key, target_key in renames.items():
            if unprefixed_key == source_key or unprefixed_key.startswith(f"{source_key}."):
                unprefixed_key = f"{target_key}{unprefixed_key[len(source_key) :]}"
                break
        renamed_key = f"{prefix}{unprefixed_key}"
        if renamed_key in renamed_state_dict:
            raise ValueError(f"Gemma4 Unified key collision for {renamed_key!r}")
        renamed_state_dict[renamed_key] = tensor
    return renamed_state_dict


class Gemma4UnifiedStateDictAdapter:
    """Translate Gemma4 Unified vision keys between model FQNs and the HF checkpoint layout."""

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Rename model FQNs to the HF checkpoint FQNs.

        Args:
            state_dict: Model state dict.
            exclude_key_regex: Keys matching this regex are dropped (e.g. ``_extra_state``).
            **kwargs: Ignored; accepted for adapter-interface compatibility.

        Returns:
            State dict keyed by HF checkpoint FQNs.
        """
        if exclude_key_regex is not None:
            state_dict = {k: v for k, v in state_dict.items() if not re.search(exclude_key_regex, k)}
        return _rename_keys(state_dict, HF_EXPORT_KEY_RENAMES)

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Rename HF checkpoint FQNs back to the model FQNs.

        Args:
            hf_state_dict: State dict keyed by HF checkpoint FQNs.
            **kwargs: Ignored; accepted for adapter-interface compatibility.

        Returns:
            State dict keyed by model FQNs.
        """
        return _rename_keys(hf_state_dict, _HF_IMPORT_KEY_RENAMES)

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
"""Weights-only warm start of a DSpark draft.

``recipe_args.draft_weights_path`` starts a new run from existing draft weights:
a released draft or the consolidated export of an earlier run. Only the weights
come from the checkpoint; the optimizer, LR schedule, RNG, and step counter start
fresh. ``checkpoint.restore_from`` keeps its meaning and continues a run instead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from torch import nn

from nemo_automodel.components.checkpoint.checkpointing import load_hf_safetensors_state_dict, load_torch_ckpt

logger = logging.getLogger(__name__)

# Every DSpark draft copies these from the target in ``initialize_embeddings_and_head``.
_TARGET_COPIED_KEYS = frozenset({"embed_tokens.weight", "lm_head.weight"})


def _check_saved_run_settings(path: str, target_layer_ids: Sequence[int], mask_token_id: int) -> None:
    """Refuse an earlier run's export whose tensor-free settings differ from this run's.

    The target feature layers and the noise token own no tensors, so a draft trained
    with others would load cleanly and then train on inputs it never saw. Only
    AutoModel checkpoints record them, in ``dspark_meta.pt`` next to ``model/consolidated``.

    Args:
        path: The configured ``draft_weights_path``.
        target_layer_ids: Target feature layers this run feeds the draft.
        mask_token_id: Noise token this run fills block positions with.
    """
    for parent in list(Path(path).parents)[:3]:
        meta_path = parent / "dspark_meta.pt"
        if meta_path.is_file():
            meta = load_torch_ckpt(str(meta_path), map_location="cpu", weights_only=True)
            break
    else:
        return
    saved_layer_ids = meta.get("target_layer_ids")
    saved_mask_token_id = meta.get("mask_token_id")
    mismatched = {}
    if saved_layer_ids is not None and [int(i) for i in saved_layer_ids] != [int(i) for i in target_layer_ids]:
        mismatched["target_layer_ids"] = {"checkpoint": list(saved_layer_ids), "this run": list(target_layer_ids)}
    if saved_mask_token_id is not None and int(saved_mask_token_id) != int(mask_token_id):
        mismatched["mask_token_id"] = {"checkpoint": int(saved_mask_token_id), "this run": int(mask_token_id)}
    if mismatched:
        raise ValueError(f"draft_weights_path={path!r} was trained with other settings: {mismatched}.")


def load_dspark_draft_weights(
    draft_model: nn.Module,
    path: str,
    *,
    keep_target_embeddings: bool,
    target_layer_ids: Sequence[int],
    mask_token_id: int,
) -> None:
    """Load existing draft weights into a newly built draft, before it is wrapped.

    Args:
        draft_model: Draft after ``initialize_embeddings_and_head``. A draft with a
            ``state_dict_adapter`` reads its released key layout through ``from_hf``,
            and its ``is_draft_checkpoint_key``, if any, selects the tensors to read
            from a target checkpoint that embeds the draft.
        path: A ``.safetensors`` file or a flat or index-sharded directory.
        keep_target_embeddings: Keep the target's embedding and LM head (a frozen
            draft) and ignore any copies in the checkpoint. When false they are
            trainable draft weights and the checkpoint must hold them.
        target_layer_ids: Target feature layers this run feeds the draft.
        mask_token_id: Noise token this run fills block positions with.

    Raises:
        FileNotFoundError: ``path`` holds no safetensors.
        ValueError: The checkpoint's run settings or key set differ from this draft's.
    """
    _check_saved_run_settings(path, target_layer_ids, mask_token_id)
    adapter = getattr(draft_model, "state_dict_adapter", None)
    state_dict = load_hf_safetensors_state_dict(path, key_filter=getattr(adapter, "is_draft_checkpoint_key", None))
    if not state_dict:
        raise FileNotFoundError(f"draft_weights_path={path!r} contains no draft weights in .safetensors files")
    if adapter is not None:
        state_dict = adapter.from_hf(state_dict)
    if keep_target_embeddings:
        state_dict = {key: value for key, value in state_dict.items() if key not in _TARGET_COPIED_KEYS}
    loaded_count = len(state_dict)
    missing, unexpected = draft_model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in missing
        if not key.endswith("_extra_state") and not (keep_target_embeddings and key in _TARGET_COPIED_KEYS)
    ]
    if missing or unexpected:
        raise ValueError(
            f"draft_weights_path={path!r} does not match this draft: {len(missing)} keys missing "
            f"(e.g. {missing[:3]}), {len(unexpected)} unexpected (e.g. {unexpected[:3]})."
        )
    logger.info("Warm-started the DSpark draft from %s (%d tensors).", path, loaded_count)

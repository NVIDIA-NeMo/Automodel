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

"""Multimodal (image+text) data pipeline for DSpark speculative-decoding training.

DSpark's own anchor-sampling / label-gathering logic (``dspark.common``, and every
draft model's ``forward()``) expects a single, *unshifted* ``input_ids`` of length
``T`` plus a ``loss_mask`` of the same length ``T``, and derives its own future-token
targets internally (see ``label_offsets``/``safe_label_indices`` in each draft's
``forward()``). This is different from :func:`~.collate_fns.default_collate_fn`,
which additionally shifts (``labels = labels[:, 1:]``) and truncates (``input_ids =
input_ids[:, :-1]``) for the standard "predict next token" causal-LM loss -- feeding
DSpark that already-shifted, truncated output would silently double-shift / misalign
labels. :func:`dspark_vlm_collate_fn` reuses the same tokenization + label-marking
building blocks as :func:`~.collate_fns.default_collate_fn` but stops before that
shift-and-truncate step.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import DataLoader, DistributedSampler

from nemo_automodel.components.datasets.vlm.collate_fns import (
    HAVE_QWEN_VL_UTILS,
    MISSING_QWEN_VL_UTILS_MSG,
    _ensure_rgb,
    build_labels_from_template,
)
from nemo_automodel.components.datasets.vlm.fake_image import (
    _conversation_has_media,
    inject_fake_image_into_conversation,
    mask_fake_vision_tokens_batch,
)


def dspark_vlm_collate_fn(
    examples: Sequence[Dict[str, Any]],
    processor,
    max_length: int,
    inject_fake_image: bool = True,
) -> Dict[str, torch.Tensor]:
    """Collate multimodal conversations into a DSpark-ready batch.

    Unlike :func:`~.collate_fns.default_collate_fn`, this does **not** shift
    ``labels`` or truncate ``input_ids``/``attention_mask`` by one token: it
    returns a ``loss_mask`` the same length as ``input_ids``, matching what
    DSpark's anchor sampling and every draft's ``forward()`` expect. ``labels``
    itself is dropped -- DSpark re-derives its own future-token targets from
    ``input_ids`` + ``loss_mask``, so carrying a separate ``labels`` tensor
    would be dead weight moved to device every batch for no consumer.

    ``max_length`` is required (unlike ``default_collate_fn``'s optional one):
    DSpark needs a fixed shape across every batch/rank/step for its DFlash
    attention mask and the FSDP-sharded target's forward to stay consistent.

    Every conversation without an image/video gets the same fake-image
    injection ``default_collate_fn`` uses (:func:`~.fake_image
    .inject_fake_image_into_conversation`): MiniMax M3's vision_tower is its
    own FSDP2-sharded unit, so a batch mixing text-only and image-containing
    samples across data-parallel ranks would have some ranks skip the
    vision_tower's all-gather collective while others don't, hanging
    training. The fake image's vision tokens get ``attention_mask = 0``
    (:func:`~.fake_image.mask_fake_vision_tokens_batch`) so they never
    influence the captured hidden states.

    ``inject_fake_image=False`` turns that injection off, for consumers whose
    draft reads the image positions itself. ViSpec is the case in point: it
    locates image spans by ``input_ids == image_token_id``, which a fake image
    satisfies, so the draft would compress the fake vision features and broadcast
    their global vector over every following text position. Zeroing the fake
    tokens in ``attention_mask`` does not prevent that, because the span lookup
    never consults ``attention_mask``.
    """
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    conversations = []
    fake_indices = []
    for i, example in enumerate(examples):
        conversation = example["conversation"]
        if inject_fake_image and not _conversation_has_media(conversation):
            conversation = inject_fake_image_into_conversation(conversation)
            fake_indices.append(i)
        conversations.append(conversation)
    conversations = _ensure_rgb(conversations)

    batch = processor.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": "max_length", "truncation": True, "max_length": max_length},
    )

    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    if "pixel_values_videos" in batch:
        batch["pixel_values_videos"] = batch["pixel_values_videos"].to(torch.bfloat16)

    labels = build_labels_from_template(batch["input_ids"], conversations, processor)
    batch["loss_mask"] = (labels != -100).to(torch.long)
    batch.pop("labels", None)

    if fake_indices:
        mask_fake_vision_tokens_batch(batch, processor, fake_indices)

    return batch


def build_dspark_vlm_dataloader(
    *,
    dataset_cfg,
    processor,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int = 0,
    distributed: bool = False,
    dp_mesh=None,
    inject_fake_image: bool = True,
) -> DataLoader:
    """Build a multimodal DataLoader for DSpark training.

    ``dataset_cfg`` is any config node exposing ``.instantiate()`` (the repo's
    standard ``_target_`` convention, e.g. ``nemo_automodel.components.datasets
    .vlm.datasets.make_medpix_dataset`` or any other function under that module
    returning ``{"conversation": [...]}`` examples) -- the same convention the
    VLM finetune recipe uses for its own ``dataset:`` config block. Deliberately
    simpler than that recipe's ``build_dataloader``: no packing, no pipeline-
    parallel microbatch chunking, no mRoPE position-id generation, since none of
    that applies to DSpark's non-packed, non-pipelined target-capture call
    (MiniMax M3's text decoder defaults to plain sequential position ids
    regardless of whether images are present).

    ``inject_fake_image=False`` forwards to :func:`dspark_vlm_collate_fn` and
    leaves text-only conversations text-only; see that function's docstring for
    when to turn the injection off.
    """
    dataset = dataset_cfg.instantiate()
    collate_fn = functools.partial(
        dspark_vlm_collate_fn,
        processor=processor,
        max_length=max_length,
        inject_fake_image=inject_fake_image,
    )

    sampler = None
    if distributed:
        if dp_mesh is not None:
            num_replicas, rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            num_replicas, rank = None, None
        sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

    # See build_eagle3_dataloader for the forkserver/persistent_workers rationale:
    # the target model is already on CUDA by the time these workers spawn, so
    # `fork` (inheriting a live CUDA context) would abort; `forkserver` avoids it.
    worker_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = True
        if torch.cuda.is_available():
            worker_kwargs["multiprocessing_context"] = "forkserver"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
        **worker_kwargs,
    )


__all__ = ["dspark_vlm_collate_fn", "build_dspark_vlm_dataloader"]

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
"""Dataset adapter and last-turn loss masking for the ``vuhaian/v4_88k`` agentic SFT corpus.

The corpus is turn-exploded: each row is one (prefix, next-action) example, so a
trajectory of N assistant turns contributes N rows whose prefixes are nested. Only
the **final** assistant turn of each row is the intended training target;
supervising every turn would weight early turns by their duplication factor.

The shipped VLM label builder
(:func:`nemo_automodel.components.datasets.vlm.collate_fns.build_labels_from_template`)
supervises every assistant turn, so :func:`last_turn_collate_fn` wraps it and keeps
only the final supervised run.
"""

from typing import Any, Sequence

import torch
from datasets import load_dataset
from transformers import ProcessorMixin

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn

IGNORE_INDEX = -100

_ROLES = frozenset({"system", "user", "assistant"})

# Attribute used to memoize (assistant marker ids, generation-prompt suffix ids) on the
# tokenizer itself, so the cache cannot outlive the object it describes.
_MARKER_ATTR = "_v4_88k_turn_markers"


def make_v4_88k_dataset(
    path_or_dataset: str = "vuhaian/v4_88k",
    split: str = "train",
    **kwargs: Any,
):
    """Load the v4_88k corpus as text-only conversations for the VLM input pipeline.

    Emits the same ``{"conversation": [...]}`` contract as
    :func:`nemo_automodel.components.datasets.vlm.datasets.make_tulu3_dataset`, so the
    standard VLM dataloader consumes it unchanged. Conversations are text-only, so
    batches carry no ``pixel_values`` / vision tensors.

    Deliberately does not use ``_convert_sharegpt_to_conversation``: that helper splits
    user text on literal ``<image>`` / ``<video>`` substrings and drops those segments.
    This is a software-engineering corpus where ``<image>`` can appear verbatim inside
    HTML or Markdown, which would silently corrupt the prompt.

    Args:
        path_or_dataset: HF Hub id, or a path to a local ``.parquet`` file produced by
            ``scripts/prefilter_v4_88k.py``.
        split: HF split expression. Ignored for local Parquet files.
        **kwargs: Ignored. Accepted so recipe-level dataset keys forwarded to the
            dataset target do not raise.

    Returns:
        datasets.Dataset: Rows with a single ``conversation`` column, each a list of
        ``{"role": ..., "content": [{"type": "text", "text": ...}]}`` turns.
    """
    if str(path_or_dataset).endswith(".parquet"):
        dataset = load_dataset("parquet", data_files=str(path_or_dataset), split="train")
    else:
        dataset = load_dataset(path_or_dataset, split=split)

    def _to_conversation(example: dict[str, Any]) -> dict[str, Any]:
        conversation = []
        for message in example["messages"]:
            role = message.get("role")
            if role not in _ROLES:
                raise ValueError(f"Unsupported role {role!r} in {path_or_dataset}")
            turn: dict[str, Any] = {
                "role": role,
                "content": [{"type": "text", "text": message.get("content") or ""}],
            }
            # The upcoming corpus revision moves the THOUGHT prose into
            # `reasoning_content`; the stock Qwen3.6 template renders it as a
            # <think> block on the final turn and drops it from history.
            reasoning = message.get("reasoning_content")
            if reasoning:
                turn["reasoning_content"] = reasoning
            conversation.append(turn)
        return {"conversation": conversation}

    return dataset.map(_to_conversation, remove_columns=dataset.column_names)


def _resolve_markers(tokenizer) -> tuple[list[int], list[list[int]]]:
    """Return the assistant-turn marker and the generation-prompt suffixes that can follow it.

    A suffix is whatever the chat template emits after ``<|im_start|>assistant\\n``
    when asked for a generation prompt -- ``<think>\\n`` for Qwen3.6 with thinking
    enabled, ``<think>\\n\\n</think>\\n\\n`` with thinking disabled. Those tokens are
    supplied by the prompt at inference, so training on them teaches the model to
    re-emit a tag it was already given.

    Both variants are returned because a training render can start with either:

    * Current corpus (no ``reasoning_content``): the template renders every final
      assistant turn with an empty think block, which tokenizes exactly as the
      thinking-disabled suffix. ``\\n\\n`` is a single token, so the thinking-enabled
      ``<think>\\n`` never matches it. Masking the whole block supervises the content
      only, i.e. the model is trained for thinking-disabled inference.
    * ``reasoning_content`` revision: the final turn renders ``<think>\\n`` followed by
      the reasoning, which the thinking-enabled suffix matches, so only the opening
      tag is masked and the reasoning stays supervised.

    Args:
        tokenizer: Tokenizer carrying the chat template.

    Returns:
        ``(assistant_marker_ids, suffixes)`` with ``suffixes`` deduplicated and ordered
        longest first, so the most specific match wins.
    """
    cached = getattr(tokenizer, _MARKER_ATTR, None)
    if cached is not None:
        return cached

    marker = [tokenizer.convert_tokens_to_ids("<|im_start|>")] + tokenizer.encode(
        "assistant\n", add_special_tokens=False
    )

    suffixes: list[list[int]] = []
    for enable_thinking in (True, False):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "u"}],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        ids = list(rendered["input_ids"])
        for start in range(len(ids) - len(marker), -1, -1):
            if ids[start : start + len(marker)] == marker:
                suffix = ids[start + len(marker) :]
                if suffix and suffix not in suffixes:
                    suffixes.append(suffix)
                break
    suffixes.sort(key=len, reverse=True)

    markers = (marker, suffixes)
    try:
        setattr(tokenizer, _MARKER_ATTR, markers)
    except AttributeError:
        # Tokenizer forbids new attributes; recomputing per batch is cheap enough.
        pass
    return markers


def _keep_last_supervised_run(labels: torch.Tensor) -> torch.Tensor:
    """Zero every supervised position except those in each row's final contiguous run.

    Each assistant turn renders as one contiguous supervised span, so keeping the last
    run keeps exactly the final assistant turn. This is the tensor analogue of
    ``nemo_automodel.components.datasets.llm.formatting_utils._mask_labels_to_last_turn``.

    Args:
        labels: ``[batch, sequence]`` label tensor with ``IGNORE_INDEX`` marking
            unsupervised positions.

    Returns:
        Boolean ``[batch, sequence]`` mask selecting the final supervised run per row.
    """
    supervised = labels.ne(IGNORE_INDEX)
    if not bool(supervised.any()):
        return supervised

    batch, seq_len = supervised.shape
    positions = torch.arange(seq_len, device=labels.device).expand(batch, seq_len)

    previous = torch.cat([torch.zeros_like(supervised[:, :1]), supervised[:, :-1]], dim=1)
    run_id = (supervised & ~previous).cumsum(dim=1)

    last_position = torch.where(supervised, positions, torch.full_like(positions, -1)).amax(dim=1)
    last_run = run_id.gather(1, last_position.clamp(min=0).unsqueeze(1))

    keep = supervised & run_id.eq(last_run)
    # Rows with no supervised token at all keep nothing.
    return keep & last_position.unsqueeze(1).ge(0)


def last_turn_collate_fn(
    examples: Sequence[dict[str, Any]],
    processor: ProcessorMixin,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collate VLM conversations, supervising only the final assistant turn.

    Wraps :func:`nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn`
    and rewrites ``labels`` so that only the last contiguous supervised span survives,
    minus the generation-prompt prefix the chat template inserts ahead of assistant
    content.

    Do not pass ``max_length`` here: ``default_collate_fn`` switches to
    ``padding="max_length"`` when it is set, which would pad every sample to that
    length. Enforce the length cap offline with ``scripts/prefilter_v4_88k.py`` instead
    and let this collator pad to the longest sample in the batch.

    Args:
        examples: Conversation samples to collate.
        processor: Multimodal processor used to apply the chat template.
        **kwargs: Forwarded to ``default_collate_fn``.

    Returns:
        The batch mapping from ``default_collate_fn`` with ``labels`` restricted to each
        row's final assistant turn.
    """
    batch = default_collate_fn(examples, processor, **kwargs)

    labels = batch["labels"]
    keep = _keep_last_supervised_run(labels)

    tokenizer = getattr(processor, "tokenizer", processor)
    _, suffixes = _resolve_markers(tokenizer)
    if suffixes:
        input_ids = batch["input_ids"]
        for row in range(keep.shape[0]):
            selected = keep[row].nonzero(as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            start = int(selected[0])
            # default_collate_fn applies the next-token shift (labels[:, 1:] against
            # input_ids[:, :-1]), so the label at position p supervises the token at
            # input_ids position p + 1. The label positions to drop are still
            # [start, start + len(suffix)); the tokens they predict start one later.
            token_start = start + 1
            # Longest first: on the current corpus the span opens with the full empty
            # think block, which only the thinking-disabled suffix covers.
            for suffix in suffixes:
                token_end = token_start + len(suffix)
                if input_ids[row, token_start:token_end].tolist() == suffix:
                    keep[row, start : start + len(suffix)] = False
                    break

    batch["labels"] = labels.masked_fill(~keep, IGNORE_INDEX)
    return batch

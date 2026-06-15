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

"""Shared-prefix rollout folding for multi-turn prefix-tree attention (cp=1).

Folds a group of rollouts that share one prompt prefix (one prompt -> N sampled
completions) into a single deduplicated flat token layout plus a block-sparse
prefix-tree attention mask (:class:`AttnMaskSpec`). The shared prompt is stored
once; every completion attends FULL to the prompt and CAUSAL to itself.

This is the verl RFC #6401 / Automodel #2385 shared-prefix RL training layout,
restricted to the cp=1 path (no context-parallel dispatch). The produced spec is
consumed by the magi custom-model attention backend via
``set_active_attn_spec``; enable it with ``model.backend.attn: magi``.

Branch-point note: the shared prompt's final position is stored once, so it can
predict only one next token. The N completions diverge there, so the
prompt -> first-completion-token transition is left unsupervised (label
``-100`` on the last prompt token); each completion is supervised causally from
its own first token onward. This is inherent to deduplicating the shared prefix.
"""

import random
from dataclasses import dataclass, field

import torch

from nemo_automodel.components.distributed.magi_attn_utils import AttnMaskSpec

CROSS_ENTROPY_IGNORE_IDX = -100


@dataclass
class FoldedRollouts:
    """Deduplicated flat layout + prefix-tree mask for one shared-prefix group.

    Attributes:
        input_ids: flat ``[prompt | completion_0 | completion_1 | ...]`` tokens.
        labels: per-position targets; prompt and the branch-point position are
            ``-100`` (ignored), completion tokens carry their own ids.
        position_ids: prompt positions ``0..P-1``; each completion continues from
            ``P`` (so RoPE sees ``prompt ++ completion``).
        node_lengths: token count of each node, flat-layout order
            ``[len(prompt), len(c_0), len(c_1), ...]``.
        sample_paths: root -> leaf node indices per completion, e.g.
            ``[[0, 1], [0, 2], ...]``.
        spec: the :class:`AttnMaskSpec` built from ``node_lengths``/``sample_paths``.
        sample_token_ranges: per-completion flat ``[start, end)`` ranges of its
            nodes, for reconstructing per-sample outputs.
    """

    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    node_lengths: list[int]
    sample_paths: list[list[int]]
    spec: AttnMaskSpec
    sample_token_ranges: list = field(default_factory=list)


def fold_shared_prefix_rollouts(
    prompt_ids: list[int],
    completions: list[list[int]],
    *,
    completion_labels: list[list[int]] | None = None,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> FoldedRollouts:
    """Fold one shared-prefix rollout group into a deduplicated prefix-tree layout.

    Args:
        prompt_ids: the shared prompt tokens (node 0). May be empty.
        completions: one token-id list per sampled completion (one leaf each).
            Must be non-empty and every completion must be non-empty.
        completion_labels: optional per-completion target ids aligned to
            ``completions`` (use ``ignore_idx`` to mask positions). When omitted,
            every completion token is supervised (labels == tokens).
        ignore_idx: label value for unsupervised positions (default ``-100``).

    Returns:
        A :class:`FoldedRollouts` with the flat tokens, labels, position ids and
        the built :class:`AttnMaskSpec`.

    Raises:
        ValueError: if ``completions`` is empty, any completion is empty, or
            ``completion_labels`` does not match ``completions`` in shape.
    """
    if not completions:
        raise ValueError("completions must be non-empty (need at least one rollout).")
    if any(len(c) == 0 for c in completions):
        raise ValueError("every completion must be non-empty.")
    if completion_labels is not None:
        if len(completion_labels) != len(completions):
            raise ValueError("completion_labels must have one entry per completion.")
        if any(len(lab) != len(tok) for lab, tok in zip(completion_labels, completions)):
            raise ValueError("each completion_labels entry must match its completion length.")

    prompt_len = len(prompt_ids)

    # Flat dedup layout: prompt stored once, completions concatenated after it.
    input_ids = list(prompt_ids)
    for completion in completions:
        input_ids.extend(completion)

    # Labels: prompt masked; each completion supervised (or per caller's labels).
    labels = [ignore_idx] * prompt_len
    for idx, completion in enumerate(completions):
        labels.extend(completion if completion_labels is None else completion_labels[idx])

    # Position ids: prompt 0..P-1; each completion continues from P so RoPE sees
    # the completion as a continuation of the shared prompt.
    position_ids = list(range(prompt_len))
    for completion in completions:
        position_ids.extend(range(prompt_len, prompt_len + len(completion)))

    # Prefix tree: node 0 is the prompt, nodes 1..N are completions; each path is
    # prompt -> completion. A bare group (empty prompt) collapses to varlen blocks.
    node_lengths = [prompt_len] + [len(c) for c in completions]
    if prompt_len > 0:
        sample_paths = [[0, i + 1] for i in range(len(completions))]
    else:
        # No shared prefix: node 0 is empty, each completion is its own causal block.
        sample_paths = [[i + 1] for i in range(len(completions))]

    spec, sample_token_ranges = AttnMaskSpec.prefix_tree(node_lengths, sample_paths)

    return FoldedRollouts(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        node_lengths=node_lengths,
        sample_paths=sample_paths,
        spec=spec,
        sample_token_ranges=sample_token_ranges,
    )


def prefix_tree_collate_fn(batch: list[dict]) -> dict:
    """Collate one shared-prefix rollout group into a model-ready batch (cp=1).

    Folds the group with :func:`fold_shared_prefix_rollouts` and emits the flat
    tokens plus the prefix-tree mask. Only ``local_batch_size == 1`` is supported:
    each group already packs many completions into one flat sequence, and the mask
    is built per group. The returned ``magi_attn_spec`` is picked up by the recipe
    and handed to the magi attention backend via ``set_active_attn_spec``.

    Args:
        batch: a length-1 list holding one rollout group dict with keys
            ``prompt_ids`` and ``completions`` (and optional ``completion_labels``).

    Returns:
        Dict with ``input_ids``, ``labels``, ``position_ids`` (each ``[1, T]``)
        and ``magi_attn_spec`` (the :class:`AttnMaskSpec`).

    Raises:
        ValueError: if ``batch`` does not hold exactly one rollout group.
    """
    if len(batch) != 1:
        raise ValueError(f"prefix_tree_collate_fn supports local_batch_size=1 only, got {len(batch)} groups.")
    group = batch[0]
    folded = fold_shared_prefix_rollouts(
        group["prompt_ids"],
        group["completions"],
        completion_labels=group.get("completion_labels"),
    )
    return {
        "input_ids": torch.tensor([folded.input_ids], dtype=torch.long),
        "labels": torch.tensor([folded.labels], dtype=torch.long),
        "position_ids": torch.tensor([folded.position_ids], dtype=torch.long),
        "magi_attn_spec": folded.spec,
    }


def build_mock_rollout_dataset(
    *,
    num_groups: int = 16,
    completions_per_group: int = 4,
    prompt_len: int = 32,
    completion_len: int = 16,
    vocab_size: int = 1024,
    seed: int = 0,
) -> list[dict]:
    """Build a deterministic mock shared-prefix rollout dataset for smoke runs.

    Each group is one shared prompt with ``completions_per_group`` completions, in
    the ``{"prompt_ids", "completions"}`` schema consumed by
    :func:`prefix_tree_collate_fn`. Token ids are random in ``[2, vocab_size)``;
    this is a pipeline smoke, not a quality dataset.

    Args:
        num_groups: number of rollout groups.
        completions_per_group: completions (leaves) sharing each prompt.
        prompt_len: shared prompt length per group.
        completion_len: length of each completion.
        vocab_size: upper bound (exclusive) for random token ids.
        seed: RNG seed for reproducibility.

    Returns:
        A list of ``{"prompt_ids": list[int], "completions": list[list[int]]}``.
    """
    rng = random.Random(seed)

    def _ids(n: int) -> list[int]:
        return [rng.randint(2, vocab_size - 1) for _ in range(n)]

    return [
        {
            "prompt_ids": _ids(prompt_len),
            "completions": [_ids(completion_len) for _ in range(completions_per_group)],
        }
        for _ in range(num_groups)
    ]

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
completions) into a single deduplicated flat token layout plus the prefix-tree
structure (``node_lengths`` / ``sample_paths``). The shared prompt is stored
once; every completion attends FULL to the prompt and CAUSAL to itself.

This is the verl RFC #6401 / Automodel #2385 shared-prefix RL training layout,
restricted to the cp=1 path (no context-parallel dispatch). The collate carries
the structure on the batch; the magi backend builds the ``AttnMaskSpec`` from it
and activates it (the datasets layer must not import ``components.distributed``).
Enable it with ``model.backend.attn: magi``.

Branch-point note: the shared prompt's final position is stored once, so it can
predict only one next token. The N completions diverge there, so the
prompt -> first-completion-token transition is left unsupervised (label
``-100`` on the last prompt token); each completion is supervised causally from
its own first token onward. This is inherent to deduplicating the shared prefix.
"""

from dataclasses import dataclass

import torch

CROSS_ENTROPY_IGNORE_IDX = -100


@dataclass
class FoldedRollouts:
    """Deduplicated flat layout + prefix-tree structure for one shared-prefix group.

    The attention mask itself is built in the magi backend from ``node_lengths`` /
    ``sample_paths`` (via ``AttnMaskSpec.prefix_tree``); the datasets layer must not
    import ``components.distributed``, so it only carries the structure.

    Attributes:
        input_ids: flat ``[prompt | completion_0 | completion_1 | ...]`` tokens.
        labels: pre-shifted next-token targets (the loss does not shift); the
            prompt and each completion's last token are ``-100`` (ignored).
        position_ids: prompt positions ``0..P-1``; each completion continues from
            ``P`` (so RoPE sees ``prompt ++ completion``).
        node_lengths: token count of each node, flat-layout order
            ``[len(prompt), len(c_0), len(c_1), ...]``.
        sample_paths: root -> leaf node indices per completion, e.g.
            ``[[0, 1], [0, 2], ...]``.
    """

    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    node_lengths: list[int]
    sample_paths: list[list[int]]


def fold_shared_prefix_rollouts(
    prompt_ids: list[int],
    completions: list[list[int]],
    *,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
) -> FoldedRollouts:
    """Fold one shared-prefix rollout group into a deduplicated prefix-tree layout.

    Labels follow this repo's next-token convention: they are pre-shifted (the
    loss does not shift), so position ``p`` carries the id of the token the model
    should predict at ``p``. Within each completion, token ``t`` predicts token
    ``t + 1``; the completion's last token has no in-layout successor and is
    masked. The shared prompt is masked entirely, including its last position:
    that position is the branch point where the N completions diverge, so it
    cannot supervise any single first-completion token (the cost of deduplicating
    the prefix is that each completion's first token is unsupervised).

    Args:
        prompt_ids: the shared prompt tokens (node 0). May be empty.
        completions: one token-id list per sampled completion (one leaf each).
            Must be non-empty and every completion must be non-empty.
        ignore_idx: label value for unsupervised positions (default ``-100``).

    Returns:
        A :class:`FoldedRollouts` with the flat tokens, labels, position ids and
        the built :class:`AttnMaskSpec`.

    Raises:
        ValueError: if ``completions`` is empty or any completion is empty.
    """
    if not completions:
        raise ValueError("completions must be non-empty (need at least one rollout).")
    if any(len(c) == 0 for c in completions):
        raise ValueError("every completion must be non-empty.")

    prompt_len = len(prompt_ids)

    # Single pass builds the deduplicated flat layout: the prompt is stored once
    # (masked, positions 0..P-1) and each completion is appended with positions
    # continuing from P (so RoPE sees prompt ++ completion) and pre-shifted labels
    # (token t predicts t+1; the last token is masked).
    input_ids = list(prompt_ids)
    labels = [ignore_idx] * prompt_len
    position_ids = list(range(prompt_len))
    node_lengths = [prompt_len]
    for completion in completions:
        input_ids.extend(completion)
        labels.extend(completion[1:] + [ignore_idx])
        position_ids.extend(range(prompt_len, prompt_len + len(completion)))
        node_lengths.append(len(completion))

    # Prefix tree: node 0 is the prompt, nodes 1..N are completions; each path is
    # prompt -> completion. A bare group (empty prompt) collapses to varlen blocks:
    # node 0 is empty, so each completion is its own causal block.
    if prompt_len > 0:
        sample_paths = [[0, i + 1] for i in range(len(completions))]
    else:
        sample_paths = [[i + 1] for i in range(len(completions))]

    return FoldedRollouts(
        input_ids=input_ids,
        labels=labels,
        position_ids=position_ids,
        node_lengths=node_lengths,
        sample_paths=sample_paths,
    )


def prefix_tree_collate_fn(batch: list[dict]) -> dict:
    """Collate one shared-prefix rollout group into a model-ready batch (cp=1).

    Folds the group with :func:`fold_shared_prefix_rollouts` and emits the flat
    tokens plus the prefix-tree structure. Only ``local_batch_size == 1`` is
    supported: each group already packs many completions into one flat sequence,
    and the mask is per group. The ``prefix_tree`` entry is popped by the magi
    backend (``MagiState.prepare_batch``), which builds and activates the
    ``AttnMaskSpec`` from it.

    Args:
        batch: a length-1 list holding one rollout group dict with keys
            ``prompt_ids`` and ``completions``.

    Returns:
        Dict with ``input_ids``, ``labels``, ``position_ids`` (each ``[1, T]``)
        and ``prefix_tree`` (``(node_lengths, sample_paths)``).

    Raises:
        ValueError: if ``batch`` does not hold exactly one rollout group.
    """
    if len(batch) != 1:
        raise ValueError(f"prefix_tree_collate_fn supports local_batch_size=1 only, got {len(batch)} groups.")
    group = batch[0]
    folded = fold_shared_prefix_rollouts(group["prompt_ids"], group["completions"])
    return {
        "input_ids": torch.tensor([folded.input_ids], dtype=torch.long),
        "labels": torch.tensor([folded.labels], dtype=torch.long),
        "position_ids": torch.tensor([folded.position_ids], dtype=torch.long),
        "prefix_tree": (folded.node_lengths, folded.sample_paths),
    }

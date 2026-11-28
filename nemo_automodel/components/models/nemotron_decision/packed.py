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

"""Packed (state-blind) decision reads: one sequence holds a state once and several questions, each with its own mask.

Layout of a packed read of questions ``1..n`` that share a state (``json-sep`` prompts)::

    [S][Q1 ... MASK][Q2 ... MASK] ... [Qn ... MASK]

* ``S`` is the rendered prompt up to and including the state-end token; every question's prompt starts with the same
  ``S`` token ids.
* Attention: ``S`` attends only to ``S``; question ``k`` attends to ``S`` and to itself, never to another question.
* Positions: ``S`` is ``0..|S|-1``; every question restarts at ``|S|``.
* Read slots: question ``k`` is read at its own mask token, with the decision head restricted to its option codes.

A question therefore gets the same token ids, positions and attention whatever other questions share its read, so its
answer does not depend on them. With one question, the packed read is that question's prompt plus the mask with
positions ``0..L-1``, but the state does not attend to the question (single reads are fully bidirectional).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

STATE_GROUP = -1
READ_LAYOUT_ATTENTION = "state->state; question->state+self; no question<->question"
READ_LAYOUT_POSITIONS = "state 0..|S|-1; each question restarts at |S|"
READ_LAYOUT = "[state][question 1 + mask][question 2 + mask] ... in request order"
READ_LAYOUT_SLOT = "each question's mask token; softmax over its option-code tokens"
READ_LAYOUT_SINGLE = "packed_read false: one bidirectional sequence per question (prompt + mask), read at its mask"
READ_LAYOUT_STATE_END = "state = the prompt up to and including the state_end_token (right after the state value)"


@dataclass(frozen=True)
class PackedRead:
    """Model inputs of one packed read.

    Attributes:
        input_ids: Token ids, length ``L``.
        position_ids: Positions, length ``L``.
        groups: Attention group of each token, length ``L``: ``-1`` for the state, ``k`` for question ``k``.
        slots: Index of each question's mask token, in question order.
        state_length: Number of state tokens ``|S|``.
    """

    input_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    groups: tuple[int, ...]
    slots: tuple[int, ...]
    state_length: int


def shared_state_length(prompts: Sequence[Sequence[int]], state_end_token_id: int) -> int:
    """``|S|`` of prompts that share one state: tokens up to and including the state-end token.

    Args:
        prompts: Token ids of each question's prompt (without the mask).
        state_end_token_id: The token that closes the state.

    Returns:
        The shared state length.

    Raises:
        ValueError: If a prompt has no state-end token, the prompts end their states at different places or their
            state token ids differ, or a prompt has no question tokens after the state.
    """
    if not prompts:
        raise ValueError("a packed read needs at least one question")
    lengths = set()
    for prompt in prompts:
        if list(prompt).count(state_end_token_id) != 1:
            raise ValueError("every packed prompt needs exactly one state-end token")
        lengths.add(list(prompt).index(state_end_token_id) + 1)
    if len(lengths) != 1:
        raise ValueError("the prompts of a packed read must share one state boundary")
    (state_length,) = lengths
    first = tuple(prompts[0][:state_length])
    if any(tuple(prompt[:state_length]) != first for prompt in prompts):
        raise ValueError("the prompts of a packed read must share the state token ids")
    if any(len(prompt) <= state_length for prompt in prompts):
        raise ValueError("every packed prompt needs question tokens after the state")
    return state_length


def pack_prompts(prompts: Sequence[Sequence[int]], *, state_length: int, mask_token_id: int) -> PackedRead:
    """Pack prompts that share their first ``state_length`` tokens into one read (see the module docstring).

    Args:
        prompts: Token ids of each question's prompt (without the mask), in question order.
        state_length: ``|S|`` from :func:`shared_state_length`.
        mask_token_id: The mask token appended to every question as its read slot.

    Returns:
        The packed read.
    """
    if not prompts:
        raise ValueError("a packed read needs at least one question")
    state = tuple(int(t) for t in prompts[0][:state_length])
    ids = list(state)
    positions = list(range(state_length))
    groups = [STATE_GROUP] * state_length
    slots = []
    for k, prompt in enumerate(prompts):
        if tuple(prompt[:state_length]) != state:
            raise ValueError("the prompts of a packed read must share the state token ids")
        question = [int(t) for t in prompt[state_length:]] + [int(mask_token_id)]
        ids += question
        positions += range(state_length, state_length + len(question))
        groups += [k] * len(question)
        slots.append(len(ids) - 1)
    return PackedRead(tuple(ids), tuple(positions), tuple(groups), tuple(slots), state_length)


def packed_length(state_length: int, prompt_lengths: Sequence[int]) -> int:
    """Length of the packed read of prompts of these lengths (each question adds its tokens after the state and a mask)."""
    return state_length + sum(length - state_length + 1 for length in prompt_lengths)


def split_into_reads(state_length: int, prompt_lengths: Sequence[int], max_tokens: int) -> list[list[int]]:
    """Split questions of one state into packed reads of at most ``max_tokens`` tokens, keeping question order.

    A question's answer does not depend on the other questions of its read, so splitting changes no answer. A question
    whose own read (``prompt + mask``) is longer than ``max_tokens`` is left out, as in single reads.

    Args:
        state_length: ``|S|``.
        prompt_lengths: Prompt length (without the mask) of each question.
        max_tokens: Longest allowed read.

    Returns:
        Question indices of each read; every index whose own read fits appears exactly once.
    """
    reads: list[list[int]] = []
    current: list[int] = []
    length = state_length
    for index, prompt_length in enumerate(prompt_lengths):
        question = prompt_length - state_length + 1
        if prompt_length + 1 > max_tokens:
            continue
        if current and length + question > max_tokens:
            reads.append(current)
            current, length = [], state_length
        current.append(index)
        length += question
    if current:
        reads.append(current)
    return reads


def allowed_attention(groups: torch.Tensor) -> torch.Tensor:
    """Boolean attention matrix of a read: ``[..., L]`` groups to ``[..., L, L]`` (query, key).

    Token ``i`` attends to token ``j`` iff ``j`` is a state token, or ``i`` is a question token and ``j`` is in the
    same question.
    """
    query, key = groups[..., :, None], groups[..., None, :]
    return (key == STATE_GROUP) | ((query == key) & (query != STATE_GROUP))

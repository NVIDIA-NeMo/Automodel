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

"""Key exchange of Inkling2 attention under context parallelism with activation checkpointing."""

import itertools
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from nemo_automodel.components.models.inkling2 import context_parallel
from nemo_automodel.components.models.inkling2.attention import AttentionSources, Inkling2Attention

from .test_reference import tiny_config

pytestmark = pytest.mark.cpu_only

BATCH, SEQUENCE, RANKS = 2, 12, 4
Attend = Callable[[AttentionSources, torch.Tensor, torch.Tensor], torch.Tensor]


def _microbatch(seed: int, split: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hidden states [BATCH, SEQUENCE, hidden], validity and document positions [BATCH, SEQUENCE].

    The first row holds two documents that meet at ``split``; the second row one document and padding.
    """
    hidden = torch.randn(BATCH, SEQUENCE, 16, generator=torch.Generator().manual_seed(seed)).requires_grad_(True)
    valid = torch.ones(BATCH, SEQUENCE, dtype=torch.bool)
    positions = torch.arange(SEQUENCE).repeat(BATCH, 1)
    positions[0, split:] -= split
    valid[1, SEQUENCE - 3 :] = False
    positions[1, SEQUENCE - 3 :] = 0
    return hidden, valid, positions


def _recompute_everything() -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
    """Return activation-checkpoint contexts that save no operation."""
    return create_selective_checkpoint_contexts(lambda ctx, func, *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE)


@pytest.mark.parametrize("local", [True, False])
def test_recomputation_after_another_microbatch_gives_the_gradients_without_checkpointing(
    local: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two pipeline microbatches are in flight: every forward runs before the first backward.

    Two checkpointed layers share the token layout of a microbatch. When the first microbatch is
    recomputed, the layout that the layers kept belongs to the second one.
    """
    config = tiny_config()
    layer = next(index for index in range(config.num_hidden_layers) if (index in config.local_layer_ids) == local)
    torch.manual_seed(0)
    module = Inkling2Attention(config, layer, dtype=torch.float32, backend="eager")
    module.reset_parameters(0.3)
    assert module.is_local == local
    # One CP rank: nothing precedes its shard, and the largest reach over the ranks is its own.
    module.setup_cp_attention(SimpleNamespace(size=lambda: 1, get_local_rank=lambda: 0, get_group=lambda: None))
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op, group: None)
    microbatches = [_microbatch(seed=1, split=5), _microbatch(seed=2, split=8)]
    weight = torch.randn(BATCH, SEQUENCE, 16, generator=torch.Generator().manual_seed(3))
    parameters = list(module.parameters())

    def gradients(attend: Attend) -> list[tuple[torch.Tensor, ...]]:
        losses = []
        for hidden, valid, positions in microbatches:
            first = attend(AttentionSources(hidden, hidden, hidden), valid, positions)
            second = attend(AttentionSources(first, first, first), valid, positions)
            losses.append((second * weight).sum())
        return [
            torch.autograd.grad(loss, [hidden, *parameters])
            for loss, (hidden, _, _) in zip(losses, microbatches, strict=True)
        ]

    def checkpointed(sources: AttentionSources, valid: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return checkpoint(module, sources, valid, positions, use_reentrant=False, context_fn=_recompute_everything)

    expected = gradients(module)
    actual = gradients(checkpointed)
    for got_microbatch, wanted_microbatch in zip(actual, expected, strict=True):
        assert wanted_microbatch[0].abs().max() > 0
        for got, wanted in zip(got_microbatch, wanted_microbatch, strict=True):
            torch.testing.assert_close(got, wanted, rtol=1e-6, atol=1e-7 * wanted.abs().max().clamp_min(1e-6).item())


def test_flex_full_layer_attends_one_key_length_whatever_the_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention is compiled for one key length, so a full layer's keys must not follow the batch.

    The shard's first document starts at another position in every batch. The compiled kernel
    needs CUDA; the eager arithmetic stands in for it and records the keys it is given.
    """
    config = tiny_config()
    layer = next(index for index in range(config.num_hidden_layers) if index not in config.local_layer_ids)
    torch.manual_seed(0)
    module = Inkling2Attention(config, layer, dtype=torch.float32, backend="flex")
    module.reset_parameters(0.3)
    module.setup_cp_attention(SimpleNamespace(size=lambda: 1, get_local_rank=lambda: 0, get_group=lambda: None))
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op, group: None)
    key_lengths = []

    def record(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **layout: torch.Tensor | int) -> torch.Tensor:
        """Record the key length of q [batch, sequence, heads, head_dim] against k, v [batch, key_sequence, ...]."""
        key_lengths.append(k.shape[1])
        return module._eager(q, k, v, **layout)

    assert module.flex is not None
    monkeypatch.setattr(module.flex, "forward", record)
    valid = torch.ones(BATCH, SEQUENCE, dtype=torch.bool)
    for first_position in (0, 9, 30):
        hidden = torch.randn(BATCH, SEQUENCE, 16, generator=torch.Generator().manual_seed(first_position))
        positions = first_position + torch.arange(SEQUENCE).repeat(BATCH, 1)
        assert torch.isfinite(module(AttentionSources(hidden, hidden, hidden), valid, positions)).all()
    assert len(key_lengths) == 3 and len(set(key_lengths)) == 1


def _gather_from_earlier_ranks(sent: list[list[torch.Tensor]], rank: int) -> Callable[..., torch.Tensor]:
    """Return the ``gather_sequence`` of one CP rank when the ranks run one after another in one process.

    Args:
        sent: Per collective of a forward, the shards [batch, count, ...] of the ranks that ran before.
        rank: The rank that runs now; its shard is appended to every collective.

    Returns:
        A gather whose result [batch, ranks * count, ...] holds zeros for the ranks after this one.
    """
    collective = itertools.count()

    def gather(value: torch.Tensor, mesh: SimpleNamespace, *, differentiable: bool) -> torch.Tensor:
        index = next(collective)
        if index == len(sent):
            sent.append([])
        assert len(sent[index]) == rank
        sent[index].append(value)
        later = [torch.zeros_like(value)] * (mesh.size() - rank - 1)
        return torch.cat((*sent[index], *later), dim=1)

    return gather


def test_flex_full_layer_on_four_ranks_matches_one_rank_holding_the_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full flex layer exchanges the shards of every preceding rank, not only of the nearest one.

    Each row holds a document that starts more than one shard before the last rank, so the queries
    there attend keys of several preceding shards. A rank reads only what the ranks before it sent,
    so the ranks run in order in this process. The expected output is the same layer without a CP
    mesh on the whole sequence. The eager arithmetic stands in for the compiled kernel, which needs CUDA.
    """
    config = tiny_config()
    layer = next(index for index in range(config.num_hidden_layers) if index not in config.local_layer_ids)
    torch.manual_seed(0)
    module = Inkling2Attention(config, layer, dtype=torch.float32, backend="flex")
    module.reset_parameters(0.3)
    assert module.flex is not None
    monkeypatch.setattr(module.flex, "forward", module._eager)
    total = RANKS * SEQUENCE
    hidden = torch.randn(BATCH, total, 16, generator=torch.Generator().manual_seed(4))
    valid = torch.ones(BATCH, total, dtype=torch.bool)
    positions = torch.arange(total).repeat(BATCH, 1)
    # Row 0: documents of 5 and 43 tokens. Row 1: documents of 15 and 30 tokens, then padding.
    positions[0, 5:] -= 5
    positions[1, 15:] -= 15
    valid[1, total - 3 :] = False
    positions[1, total - 3 :] = 0
    # The first token of these shards is more than one shard into its document.
    assert positions[0, 2 * SEQUENCE] > SEQUENCE and (positions[:, 3 * SEQUENCE] > SEQUENCE).all()
    with torch.no_grad():
        expected = module(AttentionSources(hidden, hidden, hidden), valid, positions)
        sent: list[list[torch.Tensor]] = []
        for rank in range(RANKS):
            shard = slice(rank * SEQUENCE, (rank + 1) * SEQUENCE)
            mesh = SimpleNamespace(size=lambda: RANKS, get_local_rank=lambda rank=rank: rank, get_group=lambda: None)
            module.setup_cp_attention(mesh)
            monkeypatch.setattr(context_parallel, "gather_sequence", _gather_from_earlier_ranks(sent, rank))
            local = hidden[:, shard]
            actual = module(AttentionSources(local, local, local), valid[:, shard], positions[:, shard])
            assert expected[:, shard].abs().max() > 0
            torch.testing.assert_close(actual, expected[:, shard], rtol=1e-5, atol=1e-6)

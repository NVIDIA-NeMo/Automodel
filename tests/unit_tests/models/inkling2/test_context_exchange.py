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

from collections.abc import Callable
from contextlib import AbstractContextManager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from nemo_automodel.components.models.inkling2.attention import AttentionSources, Inkling2Attention

from .test_reference import tiny_config

pytestmark = pytest.mark.cpu_only

BATCH, SEQUENCE = 2, 12
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

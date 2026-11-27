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

"""Masked tokens under round-robin context parallelism.

Round-robin CP attention is causal only and takes no padding mask, so the batch
handed to the model must not keep a masked token in front of an attended one
(the fake image of a text-only VLM row is the case a loader produces). Expected
orders below are written out by hand; the causal-attention check compares
against the same row with its masked tokens deleted.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel import sharder as cp_sharder
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_batch_identity,
)

# Row 0: two attended tokens, a masked "fake image" (ids 90, 91), two attended tokens, right padding.
# Row 1: attended tokens only, then right padding (already in the order CP needs).
_INPUT_IDS = [[10, 11, 90, 91, 12, 13, 0, 0], [20, 21, 22, 23, 24, 0, 0, 0]]
_ATTENTION_MASK = [[1, 1, 0, 0, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]]
_LABELS = [[-100, -100, -100, -100, 12, 13, -100, -100], [-100, 21, 22, 23, 24, -100, -100, -100]]
# One axis of positions is enough to see that an attended token keeps its own position; masked tokens carry 0.
_POSITIONS = [[0, 1, 0, 0, 2, 3, 0, 0], [0, 1, 2, 3, 4, 0, 0, 0]]

_EXPECTED_INPUT_IDS = [[10, 11, 12, 13, 90, 91, 0, 0], [20, 21, 22, 23, 24, 0, 0, 0]]
_EXPECTED_ATTENTION_MASK = [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]]
_EXPECTED_LABELS = [[-100, -100, 12, 13, -100, -100, -100, -100], [-100, 21, 22, 23, 24, -100, -100, -100]]
_EXPECTED_POSITIONS = [[0, 1, 2, 3, 0, 0, 0, 0], [0, 1, 2, 3, 4, 0, 0, 0]]


class _SubMesh:
    """Stand-in for a device-mesh slice without a process group."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return 0

    def get_group(self) -> None:
        return None


class _DeviceMesh(dict):
    """Stand-in for a device mesh with a ``cp`` and a ``tp`` dimension."""

    def __init__(self, cp_size: int) -> None:
        super().__init__(cp=_SubMesh(cp_size), tp=_SubMesh(1))
        self.mesh_dim_names = ["cp", "tp"]


class _RoundRobinModel:
    """Model whose CP hook selects the round-robin aux-only sharder, as the Qwen VLM hooks do."""

    def prepare_model_inputs_for_cp(
        self, batch: dict[str, torch.Tensor], *, num_chunks: int
    ) -> dict[str, ContextParallelSharder]:
        """Return the sharder only.

        Args:
            batch: Full-sequence batch; ``input_ids`` is [batch, sequence].
            num_chunks: Unused.
        """
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            )
        }


class _OwnSharderModel:
    """Model that pairs the round-robin layout with its own batch sharder, which handles the mask itself."""

    def prepare_model_inputs_for_cp(
        self, batch: dict[str, torch.Tensor], *, num_chunks: int
    ) -> dict[str, ContextParallelSharder]:
        """Return the sharder only.

        Args:
            batch: Full-sequence batch; ``input_ids`` is [batch, sequence].
            num_chunks: Unused.
        """
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_identity,
                local_token_global_indices=round_robin_local_indices,
            )
        }


@pytest.fixture(autouse=True)
def _no_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


def _batch(mrope: bool = False) -> dict[str, torch.Tensor]:
    positions = torch.tensor(_POSITIONS)
    return {
        "input_ids": torch.tensor(_INPUT_IDS),
        "attention_mask": torch.tensor(_ATTENTION_MASK),
        "labels": torch.tensor(_LABELS),
        "position_ids": positions.unsqueeze(0).expand(3, -1, -1).clone() if mrope else positions,
        "mm_token_type_ids": torch.tensor(_INPUT_IDS) // 90,
    }


@pytest.mark.parametrize("mrope", [False, True])
def test_move_masked_tokens_behind_attended(mrope: bool) -> None:
    batch = _batch(mrope)
    loss_mask = torch.tensor(_ATTENTION_MASK, dtype=torch.bool)
    batch["teacher_logits"] = torch.tensor(_INPUT_IDS, dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 3).clone()

    loss_mask = cp_sharder.move_masked_tokens_behind_attended(
        batch, loss_mask=loss_mask, extra_seq_buffers={"teacher_logits": 1}
    )

    assert batch["input_ids"].tolist() == _EXPECTED_INPUT_IDS
    assert batch["attention_mask"].tolist() == _EXPECTED_ATTENTION_MASK
    assert batch["labels"].tolist() == _EXPECTED_LABELS
    assert batch["mm_token_type_ids"].tolist() == [[0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    expected_positions = torch.tensor(_EXPECTED_POSITIONS)
    if mrope:
        expected_positions = expected_positions.unsqueeze(0).expand(3, -1, -1)
    assert torch.equal(batch["position_ids"], expected_positions)
    assert loss_mask.tolist() == [[bool(v) for v in row] for row in _EXPECTED_ATTENTION_MASK]
    assert batch["teacher_logits"][..., 0].tolist() == _EXPECTED_INPUT_IDS


def test_batch_with_masked_tokens_last_is_left_untouched() -> None:
    batch = {key: value[1:] for key, value in _batch().items()}
    before = dict(batch)

    assert cp_sharder.move_masked_tokens_behind_attended(batch) is None

    assert all(batch[key] is before[key] for key in before)


@pytest.mark.parametrize(
    ("model", "cp_size", "expected_input_ids"),
    [
        pytest.param(_RoundRobinModel(), 2, _EXPECTED_INPUT_IDS, id="aux-only sharder of a model hook"),
        pytest.param(None, 2, _EXPECTED_INPUT_IDS, id="generic round-robin sharder"),
        pytest.param(_RoundRobinModel(), 1, _INPUT_IDS, id="no context parallelism"),
        pytest.param(_OwnSharderModel(), 2, _INPUT_IDS, id="model-owned batch sharder"),
    ],
)
def test_cp_sharder_reorders_for_the_sharders_that_drop_the_mask(
    model: _RoundRobinModel | _OwnSharderModel | None, cp_size: int, expected_input_ids: list[list[int]]
) -> None:
    batch = _batch()

    ContextParallelSharder(model, _DeviceMesh(cp_size), batch)

    assert batch["input_ids"].tolist() == expected_input_ids


def test_attended_tokens_cannot_reach_masked_tokens_under_causal_attention() -> None:
    """What the CP ranks compute together: causal attention over the full row, without a padding mask."""
    torch.manual_seed(0)
    embedding = torch.nn.Embedding(100, 16)

    def causal_attention(input_ids: torch.Tensor) -> torch.Tensor:
        """Self-attention with ``is_causal=True`` and no mask.

        Args:
            input_ids: Token ids of shape [batch, sequence].

        Returns:
            Attention output of shape [batch, sequence, hidden].
        """
        hidden = embedding(input_ids).unsqueeze(1)
        return torch.nn.functional.scaled_dot_product_attention(hidden, hidden, hidden, is_causal=True).squeeze(1)

    batch = _batch()
    ContextParallelSharder(_RoundRobinModel(), _DeviceMesh(2), batch)
    attended = batch["attention_mask"][0].bool()
    output = causal_attention(batch["input_ids"])[0, attended]

    # The same row without its masked tokens: ids 10, 11, 12, 13.
    reference = causal_attention(torch.tensor([[10, 11, 12, 13]]))[0]
    torch.testing.assert_close(output, reference)
    # Other content in the masked tokens (other pixels in the fake image) changes nothing for the attended ones.
    other_fake = torch.where(batch["attention_mask"] == 0, 77, batch["input_ids"])
    assert torch.equal(causal_attention(other_fake)[0, attended], output)

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
orders below are written out by hand or built by a per-row partition; the
causal-attention check compares against the same row with its masked tokens
deleted. A caller that pairs model outputs or its own per-token tensors with the
loader's batch needs the order the sharder kept.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel import sharder as cp_sharder
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_batch_identity,
)
from nemo_automodel.components.models.common.mtp import (
    MTPConfig,
    MTPContextParallelInputs,
    prepare_mtp_context_parallel_inputs,
)
from nemo_automodel.recipes.llm import kd as llm_kd
from nemo_automodel.recipes.llm import train_ft
from nemo_automodel.recipes.vlm import kd as vlm_kd

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
# Where each token of the reordered row sat in the loader's row.
_EXPECTED_TOKEN_ORDER = [[0, 1, 4, 5, 2, 3, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]

# One row of 24 tokens with attended and masked tokens interleaved: long enough that a sort that is not
# stable no longer keeps the order inside the two groups.
_LONG_ATTENTION_MASK = [[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]]
_LONG_INPUT_IDS = [list(range(100, 124))]

# Row 0 is padded on the left, as a left-padding tokenizer with ``padding: max_length`` does. Moving its three
# masked tokens behind the five attended ones is not its own inverse, unlike the swap of two pairs in the rows above.
# Row 2 has a masked run of three in the middle and two attended tokens behind it, so its new order is not a
# rotation of the row either.
_LEFT_PADDED_INPUT_IDS = [
    [0, 0, 0, 10, 11, 12, 13, 14],
    [20, 21, 22, 23, 24, 25, 26, 27],
    [30, 90, 91, 92, 31, 32, 0, 0],
]
_LEFT_PADDED_ATTENTION_MASK = [[0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 1, 1, 0, 0]]
_LEFT_PADDED_LABELS = [
    [-100, -100, -100, 11, 12, 13, 14, -100],
    [21, 22, 23, 24, 25, 26, 27, -100],
    [-100, -100, -100, -100, 32, -100, -100, -100],
]


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


class _ContiguousAuxOnlyModel:
    """Model whose CP hook pairs the aux-only sharder with a contiguous layout and its own masked attention."""

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
                local_token_global_indices=contiguous_local_indices,
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
    batch["padding_mask"] = torch.tensor(_ATTENTION_MASK) == 0
    batch["inputs_embeds"] = torch.tensor(_INPUT_IDS, dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 2).clone()

    loss_mask, token_order = cp_sharder.move_masked_tokens_behind_attended(
        batch, loss_mask=loss_mask, extra_seq_buffers={"teacher_logits": 1}
    )

    assert token_order.tolist() == _EXPECTED_TOKEN_ORDER
    assert batch["input_ids"].tolist() == _EXPECTED_INPUT_IDS
    assert batch["padding_mask"].tolist() == [[v == 0 for v in row] for row in _EXPECTED_ATTENTION_MASK]
    assert batch["inputs_embeds"][..., 1].tolist() == _EXPECTED_INPUT_IDS
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

    assert cp_sharder.move_masked_tokens_behind_attended(batch) == (None, None)

    assert all(batch[key] is before[key] for key in before)


def test_batch_with_a_query_key_mask_is_left_untouched() -> None:
    batch = _batch()
    # Causal [batch, query, key] mask: along axis 1 a masked entry is followed by an attended one.
    batch["attention_mask"] = torch.ones(2, 8, 8, dtype=torch.long).tril()
    before = dict(batch)

    assert cp_sharder.move_masked_tokens_behind_attended(batch) == (None, None)

    assert all(batch[key] is before[key] for key in before)


@pytest.mark.parametrize(
    ("model", "cp_size", "expected_input_ids"),
    [
        pytest.param(_RoundRobinModel(), 2, _EXPECTED_INPUT_IDS, id="aux-only sharder of a model hook"),
        pytest.param(None, 2, _EXPECTED_INPUT_IDS, id="generic round-robin sharder"),
        pytest.param(_RoundRobinModel(), 1, _INPUT_IDS, id="no context parallelism"),
        pytest.param(_OwnSharderModel(), 2, _INPUT_IDS, id="model-owned batch sharder"),
        pytest.param(_ContiguousAuxOnlyModel(), 2, _INPUT_IDS, id="aux-only sharder with a contiguous layout"),
    ],
)
def test_cp_sharder_reorders_for_the_sharders_that_drop_the_mask(
    model: _RoundRobinModel | _OwnSharderModel | _ContiguousAuxOnlyModel | None,
    cp_size: int,
    expected_input_ids: list[list[int]],
) -> None:
    batch = _batch()

    sharder = ContextParallelSharder(model, _DeviceMesh(cp_size), batch)

    assert batch["input_ids"].tolist() == expected_input_ids
    if expected_input_ids is _INPUT_IDS:
        assert sharder.token_order is None
    else:
        assert sharder.token_order.tolist() == _EXPECTED_TOKEN_ORDER


@pytest.mark.parametrize("model", [_RoundRobinModel(), None], ids=["aux-only sharder of a model hook", "generic"])
def test_cp_sharder_keeps_the_order_inside_attended_and_masked_tokens(model: _RoundRobinModel | None) -> None:
    mask = _LONG_ATTENTION_MASK[0]
    # Attended positions in order, then masked positions in order.
    partition = [i for i in range(24) if mask[i] == 1] + [i for i in range(24) if mask[i] == 0]
    batch = {
        "input_ids": torch.tensor(_LONG_INPUT_IDS),
        "attention_mask": torch.tensor(_LONG_ATTENTION_MASK),
        "labels": torch.tensor(_LONG_INPUT_IDS) + 1000,
    }

    ContextParallelSharder(model, _DeviceMesh(2), batch)

    assert batch["input_ids"].tolist() == [[100 + i for i in partition]]
    assert batch["labels"].tolist() == [[1100 + i for i in partition]]
    assert batch["attention_mask"].tolist() == [[1] * 14 + [0] * 10]


def test_cp_sharder_reorders_extra_buffers_and_the_loss_mask() -> None:
    batch = _batch()
    batch["teacher_logits"] = torch.tensor(_INPUT_IDS, dtype=torch.float32).unsqueeze(-1).expand(-1, -1, 3).clone()
    sharder = ContextParallelSharder(
        None,
        _DeviceMesh(2),
        batch,
        loss_mask=torch.tensor(_ATTENTION_MASK, dtype=torch.bool),
        extra_seq_buffers={"teacher_logits": 1},
    )
    received = {}

    def record_shard_batch(
        cp_mesh: _SubMesh,
        tp_mesh: _SubMesh,
        batch: dict[str, torch.Tensor],
        *,
        loss_mask: torch.Tensor,
        padding_token_id: int,
    ) -> tuple[type[nullcontext], dict[str, torch.Tensor], None]:
        """Record the loss mask that ``shard`` hands to the batch sharder.

        Args:
            cp_mesh: Unused.
            tp_mesh: Unused.
            batch: Full-sequence batch; ``input_ids`` is [batch, sequence].
            loss_mask: Per-token mask of shape [batch, sequence].
            padding_token_id: Unused.
        """
        received["loss_mask"] = loss_mask
        return nullcontext, batch, None

    sharder.shard_batch = record_shard_batch
    sharder.shard(batch)

    assert batch["teacher_logits"][..., 0].tolist() == _EXPECTED_INPUT_IDS
    assert received["loss_mask"].tolist() == [[bool(v) for v in row] for row in _EXPECTED_ATTENTION_MASK]


def _left_padded_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(_LEFT_PADDED_INPUT_IDS),
        "attention_mask": torch.tensor(_LEFT_PADDED_ATTENTION_MASK),
        "labels": torch.tensor(_LEFT_PADDED_LABELS),
    }


def test_token_order_maps_between_the_loader_and_the_reordered_batch() -> None:
    batch = _left_padded_batch()
    sharder = ContextParallelSharder(None, _DeviceMesh(2), batch)
    assert batch["input_ids"].tolist() == [
        [10, 11, 12, 13, 14, 0, 0, 0],
        _LEFT_PADDED_INPUT_IDS[1],
        [30, 31, 32, 90, 91, 92, 0, 0],
    ]
    # A per-token tensor a recipe built from the loader's batch (here: the next token, as an MTP stream is).
    next_token = torch.roll(torch.tensor(_LEFT_PADDED_INPUT_IDS), shifts=-1, dims=1)

    assert sharder.apply_token_order(next_token).tolist() == [
        [11, 12, 13, 14, 0, 0, 0, 10],
        next_token[1].tolist(),
        [90, 32, 0, 91, 92, 31, 0, 30],
    ]
    # A model output of shape [batch, sequence, vocab] computed on the reordered batch.
    logits = torch.nn.functional.one_hot(batch["input_ids"], 100)
    assert sharder.restore_token_order(logits).argmax(-1).tolist() == _LEFT_PADDED_INPUT_IDS
    multi_axis = batch["input_ids"].unsqueeze(0).expand(3, -1, -1)
    assert sharder.restore_token_order(multi_axis, seq_dim=2)[2].tolist() == _LEFT_PADDED_INPUT_IDS

    untouched = ContextParallelSharder(None, _DeviceMesh(1), _batch())
    assert untouched.apply_token_order(next_token) is next_token
    assert untouched.restore_token_order(logits) is logits


class _TokenIdTeacher(torch.nn.Module):
    """Teacher whose logits name the token it was given at each position."""

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        """Return one-hot logits of shape [batch, sequence, 100] for ``input_ids`` of shape [batch, sequence]."""
        return SimpleNamespace(logits=torch.nn.functional.one_hot(input_ids, 100).float())


class _ReversingSharder(ContextParallelSharder):
    """The real sharder resolution with a stand-in for the rank split.

    A shard is one token longer than the row and in reverse order, so a caller that treats a shard as the full
    row in the batch's order gets other tokens.
    """

    def shard(self, batch: dict[str, torch.Tensor]) -> tuple[type[nullcontext], dict[str, torch.Tensor]]:
        """Append one padding token to every [batch, sequence] tensor and reverse its sequence axis."""
        return nullcontext, {key: torch.nn.functional.pad(value, (0, 1)).flip(1) for key, value in batch.items()}


def _undo_reversing_shard(logits: torch.Tensor, *, device_mesh: _DeviceMesh, sequence_length: int) -> torch.Tensor:
    """Stand-in for ``materialize_teacher_logits``: [batch, sequence + 1, vocab] back to the batch's order."""
    return logits.flip(1)[:, :sequence_length]


class _UnshardedSharder(ContextParallelSharder):
    """The real sharder resolution, with the rank split left out (one process holds the full sequence)."""

    def shard(self, batch: dict[str, torch.Tensor]) -> tuple[type[nullcontext], dict[str, torch.Tensor]]:
        """Return the full-sequence batch unchanged."""
        return nullcontext, batch

    def shard_token_tensor(
        self, tensor: torch.Tensor, seq_dim: int = 1, fill: float | int | None = None
    ) -> torch.Tensor:
        """Return the full-sequence tensor unchanged."""
        return tensor


@pytest.mark.parametrize(
    ("recipe_module", "recipe_cls"),
    [
        (llm_kd, llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction),
        (vlm_kd, vlm_kd.KnowledgeDistillationRecipeForVLM),
    ],
    ids=["llm", "vlm"],
)
def test_separate_teacher_logits_come_back_in_the_loader_order(
    monkeypatch: pytest.MonkeyPatch, recipe_module: ModuleType, recipe_cls: type
) -> None:
    """The student pairs ``teacher_logits[b, i]`` with token ``i`` of its own batch, which is not reordered."""
    monkeypatch.setattr(recipe_module, "ContextParallelSharder", _ReversingSharder)
    monkeypatch.setattr(recipe_module, "materialize_teacher_logits", _undo_reversing_shard)
    recipe = object.__new__(recipe_cls)
    recipe.kd_mesh_bridge = SimpleNamespace(move_to_device=lambda batch: batch)
    recipe.device_mesh = _DeviceMesh(2)
    recipe.teacher_model = _TokenIdTeacher()
    recipe.pp_enabled = False

    logits = recipe._teacher_forward_separate(_left_padded_batch())

    assert logits.argmax(-1).tolist() == _LEFT_PADDED_INPUT_IDS


@pytest.mark.parametrize("microbatch_size", [1, 2])
def test_pipeline_teacher_logits_come_back_in_the_loader_order(
    monkeypatch: pytest.MonkeyPatch, microbatch_size: int
) -> None:
    """The pipeline teacher of the LLM recipe returns its logits one microbatch at a time."""
    teacher = _TokenIdTeacher()
    teacher._teacher_logits_capture = [None]

    def run_schedule(input_ids: torch.Tensor, **kwargs: torch.Tensor | list[torch.Tensor]) -> None:
        """Capture the teacher's logits for each microbatch of ``input_ids`` ([batch, local sequence])."""
        teacher._teacher_logits_capture[0] = [teacher(rows).logits for rows in input_ids.split(microbatch_size)]

    monkeypatch.setattr(llm_kd, "ContextParallelSharder", _ReversingSharder)
    monkeypatch.setattr(llm_kd, "materialize_teacher_logits", _undo_reversing_shard)
    recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.kd_mesh_bridge = SimpleNamespace(move_to_device=lambda batch: batch)
    recipe.device_mesh = _DeviceMesh(2)
    recipe.teacher_model = teacher
    recipe.pp_enabled = True
    recipe.teacher_pp = SimpleNamespace(
        update_seq_len=lambda sequence_length: None,
        info=SimpleNamespace(has_first_stage=True, has_last_stage=True, schedule=SimpleNamespace(eval=run_schedule)),
    )

    logits = recipe._teacher_forward_separate(_left_padded_batch())

    assert logits.argmax(-1).tolist() == _LEFT_PADDED_INPUT_IDS


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


class _MTPModel(torch.nn.Module):
    """Model with one MTP depth that records the ids it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.mtp_config = MTPConfig(num_layers=1, layer_pattern="*")
        self.supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=True)
        self.seen: dict[str, torch.Tensor] = {}

    def prepare_mtp_inputs_for_cp(
        self, batch: dict[str, torch.Tensor], *, ignore_index: int = -100
    ) -> MTPContextParallelInputs:
        """Build the MTP streams from ``input_ids`` and ``labels`` of shape [batch, sequence]."""
        return prepare_mtp_context_parallel_inputs(batch, num_depths=1, ignore_index=ignore_index)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        mtp_per_depth_input_ids: tuple[torch.Tensor, ...],
        mtp_per_depth_position_ids: tuple[torch.Tensor, ...],
        **kwargs: torch.Tensor,
    ) -> SimpleNamespace:
        """Record ``input_ids`` and the first MTP streams, all of shape [batch, sequence]."""
        self.seen = {
            "input_ids": input_ids,
            "mtp_input_ids": mtp_per_depth_input_ids[0],
            "mtp_position_ids": mtp_per_depth_position_ids[0],
        }
        hidden = self.scale * input_ids.float().unsqueeze(-1)
        return SimpleNamespace(
            logits=hidden, mtp_per_depth_h=[hidden], mtp_per_depth_logits=None, mtp_loss_scaling_factor=1.0
        )


def test_llm_recipe_mtp_streams_follow_the_reordered_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LLM recipe builds the MTP streams before it resolves the sharder; they must end up in the batch's order."""
    model = _MTPModel()
    seen_targets = {}

    monkeypatch.setattr(train_ft, "ContextParallelSharder", _UnshardedSharder)
    monkeypatch.setattr(train_ft, "calculate_loss", lambda loss_fn, *, logits, **kwargs: logits.sum() * 0.0)
    monkeypatch.setattr(
        train_ft,
        "calculate_mtp_loss",
        lambda loss_fn, *, mtp_per_depth_h, mtp_per_depth_targets, **kwargs: (
            seen_targets.update(targets=mtp_per_depth_targets[0]) or mtp_per_depth_h[0].sum() * 0.01
        ),
    )
    monkeypatch.setattr(train_ft, "get_final_hidden_states", lambda output: None)
    monkeypatch.setattr(train_ft, "get_sync_ctx", lambda *args, **kwargs: nullcontext())
    recipe = object.__new__(train_ft.TrainFinetuneRecipeForNextTokenPrediction)
    recipe.cfg = SimpleNamespace(mtp=SimpleNamespace(ignore_index=-100, scaling_factor=1.0))
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.device_mesh = _DeviceMesh(2)
    recipe.pp_enabled = False
    recipe.tokenizer = SimpleNamespace(pad_token_id=0)
    recipe.te_quantization = None
    recipe.model_parts = [model]
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.loss_fn = torch.nn.Identity()
    recipe._get_cp_group_size = lambda: 2
    recipe._get_dp_group_size = lambda include_cp=False: 1
    # Row 0 is padded on the left, as a left-padding tokenizer with ``padding: max_length`` does.
    batch = {
        "input_ids": torch.tensor([[0, 0, 10, 11, 12, 13], [20, 21, 22, 23, 24, 25]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 11, 12, 13, -100], [21, 22, 23, 24, 25, -100]]),
    }

    recipe._forward_backward_step(idx=0, batch=batch, loss_buffer=[], num_label_tokens=8, num_batches=1, is_train=True)

    assert model.seen["input_ids"].tolist() == [[10, 11, 12, 13, 0, 0], [20, 21, 22, 23, 24, 25]]
    # At every attended token the MTP stream holds the token after it (0 after the last one) and its label.
    assert model.seen["mtp_input_ids"][0, :4].tolist() == [11, 12, 13, 0]
    assert model.seen["mtp_input_ids"][1].tolist() == [21, 22, 23, 24, 25, 0]
    # The row's first token sits at position 2 of the loader's row; each attended token's stream is one ahead.
    assert model.seen["mtp_position_ids"][0, :3].tolist() == [3, 4, 5]
    assert model.seen["mtp_position_ids"][1, :5].tolist() == [1, 2, 3, 4, 5]
    # The padding in front of row 0 was built with the row's first token as its MTP target. Behind the row it
    # could attend to that token, so it has no target, like right padding.
    assert seen_targets["targets"][0].tolist() == [12, 13, -100, -100, -100, -100]
    assert seen_targets["targets"][1].tolist() == [22, 23, 24, 25, -100, -100]

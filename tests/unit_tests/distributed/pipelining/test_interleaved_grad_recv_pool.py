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


"""Receive lifetime and storage-ownership contracts for interleaved pooling."""

from collections import defaultdict
from types import MethodType

import pytest
import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B, _Action, _ComputationType

from nemo_automodel.components.distributed.pipelining.recv_buffer_pool import (
    _backward_recv_slots,
    _pool_stage_backward_recvs,
)


def make_schedule(virtual: int, microbatches: int, physical: int = 4) -> ScheduleInterleaved1F1B:
    """Use the installed native generator and lowering, without constructing tensors."""
    schedule = object.__new__(ScheduleInterleaved1F1B)
    schedule.n_local_stages = virtual
    schedule.pp_group_size = physical
    schedule._n_microbatches = microbatches
    schedule._num_stages = physical * virtual
    schedule._stages = []
    schedule._defer_pp_recv = False
    schedule.number_of_rounds = max(1, microbatches // physical)
    schedule.microbatches_per_round = microbatches // schedule.number_of_rounds
    schedule.pipeline_order = {rank: schedule._calculate_single_rank_operations(rank) for rank in range(physical)}
    schedule._prepare_schedule_with_comms(schedule.pipeline_order)
    return schedule


@pytest.mark.parametrize("virtual,microbatches", [(2, 8), (2, 16), (3, 16), (4, 16)])
@pytest.mark.parametrize("slack", [0, 2])
def test_native_schedule_never_reuses_a_live_backward_buffer(virtual: int, microbatches: int, slack: int) -> None:
    schedule = make_schedule(virtual, microbatches)
    savings = 0
    for rank, actions in schedule.pipeline_order_with_comms.items():
        for stage in range(rank, schedule._num_stages - 1, schedule.pp_group_size):
            slots, count = _backward_recv_slots(actions, stage, microbatches, slack)
            active: dict[int, int] = {}
            seen: set[int] = set()
            for action in actions:
                if action.stage_index != stage:
                    continue
                microbatch = action.microbatch_index
                if action.computation_type == _ComputationType.RECV_B:
                    assert slots[microbatch] not in active
                    active[slots[microbatch]] = microbatch
                    seen.add(microbatch)
                elif action.computation_type == _ComputationType.FULL_BACKWARD:
                    assert active.pop(slots[microbatch]) == microbatch
            assert not active and seen == set(range(microbatches))
            assert max(slots) < count <= microbatches
            savings += microbatches - count
    assert savings > 0


@pytest.mark.parametrize("defect", ["missing", "duplicate", "consume_first", "split_backward"])
def test_invalid_receive_lifetimes_are_rejected(defect: str) -> None:
    receive = _Action(0, _ComputationType.RECV_B, 0)
    backward = _Action(0, _ComputationType.FULL_BACKWARD, 0)
    actions = {
        "missing": [receive],
        "duplicate": [receive, receive, backward],
        "consume_first": [backward, receive],
        "split_backward": [receive, _Action(0, _ComputationType.BACKWARD_INPUT, 0)],
    }[defect]
    with pytest.raises(ValueError):
        _backward_recv_slots(actions, 0, 1, 0)


def test_too_small_pool_negative_control_detects_overlapping_receives() -> None:
    schedule = make_schedule(4, 16)
    detected = False
    for actions in schedule.pipeline_order_with_comms.values():
        active: dict[int, set[int]] = defaultdict(set)
        for action in actions:
            if action.computation_type == _ComputationType.RECV_B:
                if active[action.stage_index]:
                    detected = True
                active[action.stage_index].add(action.microbatch_index)
            elif action.computation_type == _ComputationType.FULL_BACKWARD:
                active[action.stage_index].discard(action.microbatch_index)
    assert detected, "A one-slot policy must be rejected by this actual schedule"


def test_reinitialization_restores_chunk_count_and_gradients_own_their_storage() -> None:
    # Exercise the instance wrappers with minimal real PipelineStage storage.
    stage = object.__new__(PipelineStage)
    stage.stage_index = 1
    stage.grad_recv_info = {}
    width = 8

    def original_setup(self: PipelineStage, count: int) -> None:
        self.chunks = count
        self.grad_recv_info = {i: (torch.full((width,), float(i)), None) for i in range(count)}

    def original_retrieve(self: PipelineStage, microbatch: int) -> tuple[torch.Tensor | None, ...]:
        return self.grad_recv_info[microbatch]

    stage._setup_backward_recv_info = MethodType(original_setup, stage)
    stage._retrieve_recv_grads = MethodType(original_retrieve, stage)
    _pool_stage_backward_recvs(stage, (0, 1, 0, 1), 2)
    for width in (8, 16):
        stage._setup_backward_recv_info(4)
        assert stage.chunks == 4
        assert len({id(info) for info in stage.grad_recv_info.values()}) == 2
        assert stage.grad_recv_info[0] is stage.grad_recv_info[2]
        received, absent = stage._retrieve_recv_grads(0)
        assert absent is None and received.shape == (width,)
        expected = received.clone()
        # Simulate an asynchronous consumer retaining a view, then slot reuse.
        escaped = received.view_as(received)
        unowned_negative_control = stage.grad_recv_info[0][0].view(width)
        stage.grad_recv_info[2][0].fill_(99)
        torch.testing.assert_close(escaped, expected, rtol=0, atol=0)
        assert not torch.equal(unowned_negative_control, expected)
    with pytest.raises(ValueError, match="microbatch count"):
        stage._setup_backward_recv_info(8)

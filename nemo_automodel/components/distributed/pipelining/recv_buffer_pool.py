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

"""Pooled point-to-point recv buffers for torch pipeline stages.

``torch.distributed.pipelining`` pre-allocates one full-size recv buffer per
microbatch and per direction (activations forward, gradients backward). At
large microbatch counts this dominates pipeline-stage memory: e.g. Kimi-K3 at
2k-token rows x GBS 4096 / mbs 2 / dp32 needs 64 buffer sets per direction,
~45 GiB on middle stages — more than the activations themselves — which is
exactly what pushes that shape out of memory on 64 GB200 nodes.

Under the 1F1B schedule only ``num_stages - stage_index`` microbatches are
ever in flight on a stage: a forward recv buffer is live from irecv-post
until that chunk's backward consumes the stage input (activation-checkpoint
recompute reads it), and a grad recv buffer only until the chunk's backward
returns. This module therefore rebinds the per-chunk recv-info maps onto a
ring of ``K = (num_stages - stage_index) + slack`` real buffer sets.

Gradient accumulation on reused leaf buffers is already impossible upstream:
``stage_backward()`` harvests input grads and sets ``val.grad = None`` per
chunk.

Opt-in via :func:`install_recv_buffer_pool` (wired to
``PipelineConfig.pp_recv_buffer_pool``); it is a no-op that logs a warning
and leaves stock behavior when the torch internals it adapts are not
recognized. Only schedules with bounded in-flight depth are safe — see
:func:`schedule_supports_recv_pool`; schedules with unbounded in-flight
microbatches (e.g. GPipe) would silently corrupt gradients (verified by the
CPU parity harness in the unit tests).

Native interleaved 1F1B uses a separate, instance-local implementation in
:func:`install_interleaved_recv_pool`. Its pools are colored from the actual lowered schedule. Copies give each
microbatch independent input and gradient storage before receive-slot reuse.
"""

import logging
from collections import deque
from types import MethodType

import torch
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B, _Action, _ComputationType

logger = logging.getLogger(__name__)

_INSTALLED = False
# Which torch PipelineStage layout the patch bound to ("per-direction-setup" or
# "prepare-infra"); None until installed. Logged at install time so a run's log
# states which code path it exercised.
_INSTALLED_LAYOUT: str | None = None

# Schedules whose in-flight microbatch count per stage is bounded by
# num_stages - stage_index, making the ring size proof valid. Interleaved /
# looped / zero-bubble schedules and CSV schedules have different (or
# unknown) envelopes and cannot use this ordinary 1F1B implementation.
_SUPPORTED_SCHEDULES = frozenset({"1f1b"})


def schedule_supports_recv_pool(pp_schedule: str | None) -> bool:
    """Return True when the schedule's in-flight depth bound makes pooling safe.

    Args:
        pp_schedule: Schedule name as configured (e.g. ``"1f1b"``, ``"gpipe"``),
            or None (custom CSV schedule).
    """
    return pp_schedule is not None and pp_schedule.lower() in _SUPPORTED_SCHEDULES


def _ring_size(stage, num_microbatches: int, slack: int) -> int:
    """Number of real buffer sets for a stage: in-flight depth plus slack.

    Args:
        stage: torch ``PipelineStage`` (reads ``num_stages`` / ``stage_index``).
        num_microbatches: Microbatches per step (upper bound for the ring).
        slack: Extra buffer sets beyond the 1F1B in-flight depth.
    """
    inflight = max(1, stage.num_stages - stage.stage_index)
    # Floor of 2: even the last-in-flight stage keeps one buffer set receiving
    # while the previous chunk's backward still reads its saved input.
    return max(2, min(num_microbatches, inflight + slack))


def install_recv_buffer_pool(slack: int = 2) -> bool:
    """Monkeypatch pipeline-stage recv-buffer setup with ring pooling.

    Must run before pipeline schedule/stage infra preparation. Installs
    process-wide (class-level) and is idempotent; the first call's ``slack``
    wins.

    Args:
        slack: Extra buffer sets beyond the 1F1B in-flight depth, absorbing
            transient recv-ahead (default 2).

    Returns:
        True if installed (or already installed), False if the torch
        internals do not match any known layout (stock behavior is kept).

    Raises:
        ValueError: if ``slack`` is negative (a ring smaller than the 1F1B
            in-flight depth would silently corrupt gradients).
    """
    global _INSTALLED, _INSTALLED_LAYOUT
    if slack < 0:
        raise ValueError(f"recv_buffer_pool: slack must be >= 0, got {slack}")
    if _INSTALLED:
        return True

    try:
        from torch.distributed.pipelining import stage as stage_mod

        base_cls = stage_mod._PipelineStageBase
        manual_cls = stage_mod.PipelineStage
    except (ImportError, AttributeError) as exc:
        logger.warning("recv_buffer_pool: torch internals mismatch, not installing: %s", exc)
        return False

    def _alias_fwd_ring(self, k, num_microbatches):
        # Forward setup does not touch self.chunks in any known layout; restore
        # it defensively anyway so a layout that does cannot leave it at k.
        self.chunks = num_microbatches
        for chunk_id in range(k, num_microbatches):
            self.args_recv_info[chunk_id] = self.args_recv_info[chunk_id % k]
        logger.debug(
            "recv_buffer_pool: stage %d fwd recv ring %d/%d buffer sets",
            self.stage_index,
            k,
            num_microbatches,
        )

    def _alias_bwd_ring(self, k, num_microbatches):
        # The callee set self.chunks to the value we passed; the schedule uses
        # it to detect the final chunk for grad sync — restore the true count.
        self.chunks = num_microbatches
        for chunk_id in range(k, num_microbatches):
            self.grad_recv_info[chunk_id] = self.grad_recv_info[chunk_id % k]
        logger.debug(
            "recv_buffer_pool: stage %d bwd recv ring %d/%d buffer sets",
            self.stage_index,
            k,
            num_microbatches,
        )

    if hasattr(manual_cls, "_setup_forward_recv_info"):
        # Layout "per-direction-setup" (torch release/2.12 and later): the
        # manual PipelineStage allocates in _setup_forward_recv_info(
        # num_microbatches, has_backward) and the base class in
        # _setup_backward_recv_info(num_microbatches).
        layout = "per-direction-setup"
        orig_fwd = manual_cls._setup_forward_recv_info
        orig_bwd = base_cls._setup_backward_recv_info

        def pooled_setup_forward_recv_info(self, num_microbatches, has_backward):
            k = _ring_size(self, num_microbatches, slack)
            if self.is_first or k >= num_microbatches:
                return orig_fwd(self, num_microbatches, has_backward)
            orig_fwd(self, k, has_backward)
            _alias_fwd_ring(self, k, num_microbatches)

        def pooled_setup_backward_recv_info(self, num_microbatches):
            if not isinstance(self, manual_cls) or self.is_last:
                return orig_bwd(self, num_microbatches)
            k = _ring_size(self, num_microbatches, slack)
            if k >= num_microbatches:
                return orig_bwd(self, num_microbatches)
            orig_bwd(self, k)
            _alias_bwd_ring(self, k, num_microbatches)

        manual_cls._setup_forward_recv_info = pooled_setup_forward_recv_info
        base_cls._setup_backward_recv_info = pooled_setup_backward_recv_info
    elif hasattr(manual_cls, "_prepare_forward_infra"):
        # Layout "prepare-infra" (torch <= 2.11 and the pre-refactor 2.12
        # nightlies, e.g. the NGC 26.06 container): allocation inside
        # _prepare_forward_infra(num_microbatches, args, kwargs) and the base
        # class's _prepare_backward_infra(num_microbatches).
        layout = "prepare-infra"
        orig_fwd = manual_cls._prepare_forward_infra
        orig_bwd = base_cls._prepare_backward_infra

        def pooled_prepare_forward_infra(self, num_microbatches, args, kwargs=None):
            k = _ring_size(self, num_microbatches, slack)
            if self.is_first or k >= num_microbatches:
                return orig_fwd(self, num_microbatches, args, kwargs)
            outputs = orig_fwd(self, k, args, kwargs)
            _alias_fwd_ring(self, k, num_microbatches)
            return outputs

        def pooled_prepare_backward_infra(self, num_microbatches):
            if not isinstance(self, manual_cls) or self.is_last:
                return orig_bwd(self, num_microbatches)
            k = _ring_size(self, num_microbatches, slack)
            if k >= num_microbatches:
                return orig_bwd(self, num_microbatches)
            result = orig_bwd(self, k)
            _alias_bwd_ring(self, k, num_microbatches)
            return result

        manual_cls._prepare_forward_infra = pooled_prepare_forward_infra
        base_cls._prepare_backward_infra = pooled_prepare_backward_infra
    else:
        logger.warning("recv_buffer_pool: no known recv-infra entry points on PipelineStage; not installing")
        return False
    _INSTALLED = True
    _INSTALLED_LAYOUT = layout
    if _is_rank_zero():
        logger.info("recv_buffer_pool: installed (slack=%d, layout=%s)", slack, layout)
    return True


def _is_rank_zero() -> bool:
    """True on rank 0 or when torch.distributed is not initialized (single log line per job)."""
    try:
        import torch.distributed as dist

        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:  # noqa: BLE001 - logging must never fail install
        return True


def _recv_slots(
    actions: list[_Action], stage_index: int, num_microbatches: int, slack: int, *, forward: bool = False
) -> tuple[tuple[int, ...], int]:
    """Color receive lifetimes through the compute action that copies them.

    The copied forward input remains live through backward independently of
    its receive slot. A copied gradient may outlive its backward and send.
    """
    receive_kind = _ComputationType.RECV_F if forward else _ComputationType.RECV_B
    consume_kind = _ComputationType.FORWARD if forward else _ComputationType.FULL_BACKWARD
    if slack < 0:
        raise ValueError("Recv pool slack must be nonnegative")
    events: list[tuple[bool, int]] = []
    pending: set[int] = set()
    received: set[int] = set()
    peak = 0
    for action in actions:
        if action.sub_actions is not None:
            raise ValueError("Backward recv pooling does not support overlapping compute actions")
        if action.stage_index != stage_index:
            continue
        kind = action.computation_type
        if kind in (_ComputationType.BACKWARD_INPUT, _ComputationType.BACKWARD_WEIGHT):
            raise ValueError("Backward recv pooling requires full backward actions")
        if kind not in (receive_kind, consume_kind):
            continue
        microbatch = action.microbatch_index
        if microbatch is None or not 0 <= microbatch < num_microbatches:
            raise ValueError("Invalid microbatch in recv pool schedule")
        is_receive = kind == receive_kind
        events.append((is_receive, microbatch))
        if is_receive:
            if microbatch in received:
                raise ValueError("Duplicate receive in recv pool schedule")
            received.add(microbatch)
            pending.add(microbatch)
            peak = max(peak, len(pending))
        else:
            if microbatch not in pending:
                raise ValueError("Compute action precedes its receive in recv pool schedule")
            pending.remove(microbatch)
    if pending or received != set(range(num_microbatches)):
        raise ValueError("Recv pool schedule must receive and consume every microbatch exactly once")
    count = min(num_microbatches, peak + slack)
    free = deque(range(count))
    live_slots: dict[int, int] = {}
    slots = [-1] * num_microbatches
    for is_receive, microbatch in events:
        if is_receive:
            slot = free.popleft()
            slots[microbatch] = slot
            live_slots[microbatch] = slot
        else:
            free.append(live_slots.pop(microbatch))
    return tuple(slots), count


def _pool_stage_backward_recvs(stage: PipelineStage, slots: tuple[int, ...], count: int) -> None:
    """Install instance-local allocation and ownership rules before stage setup."""
    original_setup = stage._setup_backward_recv_info
    original_retrieve = stage._retrieve_recv_grads

    def setup(self: PipelineStage, num_microbatches: int) -> None:
        if num_microbatches != len(slots):
            raise ValueError("Rebuild the backward recv pool when the microbatch count changes")
        original_setup(count)
        buffers = tuple(self.grad_recv_info[index] for index in range(count))
        self.grad_recv_info = {microbatch: buffers[slot] for microbatch, slot in enumerate(slots)}
        # Native stage logic uses chunks to identify the final gradient sync.
        self.chunks = num_microbatches

    def retrieve(self: PipelineStage, microbatch: int) -> tuple[torch.Tensor | None, ...]:
        # Autograd may retain an alias as an input/parameter gradient, and a
        # backward send can outlive this backward. Give those uses owned storage.
        # NCCL and the copy observe normal PyTorch current-stream dependencies.
        return tuple(gradient.clone() if gradient is not None else None for gradient in original_retrieve(microbatch))

    stage._setup_backward_recv_info = MethodType(setup, stage)
    stage._retrieve_recv_grads = MethodType(retrieve, stage)
    logger.info("interleaved backward recv pool: stage %d uses %d/%d buffer sets", stage.stage_index, count, len(slots))


def _pool_stage_forward_recvs(stage: PipelineStage, slots: tuple[int, ...], count: int) -> None:
    """Copy received inputs into independent leaves for each microbatch."""
    original_setup = stage._setup_forward_recv_info
    original_retrieve = stage._retrieve_recv_activations

    def setup(self: PipelineStage, num_microbatches: int, has_backward: bool) -> None:
        if num_microbatches != len(slots):
            raise ValueError("Rebuild the forward recv pool when the microbatch count changes")
        original_setup(count, has_backward)
        buffers = tuple(self.args_recv_info[index] for index in range(count))
        self.args_recv_info = {microbatch: buffers[slot] for microbatch, slot in enumerate(slots)}
        self.chunks = num_microbatches

    def retrieve(self: PipelineStage, microbatch: int) -> tuple[torch.Tensor, ...]:
        # Checkpoint recomputation and even parameter gradients may retain inputs.
        # Copy to independent leaves: backward must stop at this pipeline boundary,
        # without linking multiple microbatches through the pooled receive tensor.
        return tuple(
            activation.detach().clone().requires_grad_(activation.requires_grad)
            for activation in original_retrieve(microbatch)
        )

    stage._setup_forward_recv_info = MethodType(setup, stage)
    stage._retrieve_recv_activations = MethodType(retrieve, stage)
    logger.info("interleaved forward recv pool: stage %d uses %d/%d buffer sets", stage.stage_index, count, len(slots))


def install_interleaved_recv_pool(schedule: ScheduleInterleaved1F1B, slack: int = 2) -> None:
    """Pool receives using this interleaved schedule's actual actions.

    Call once after schedule construction and before the first step. Received
    activations become independent leaves before forward, while gradients are
    copied before backward. These owned copies protect checkpoint recomputation,
    escaping parameter gradients and asynchronous sends from slot reuse.

    Pooling removes preallocation for every microbatch but retains owned inputs
    until their actual consumers finish. The resulting peak memory depends on
    the schedule and model, including whether outputs alias stage inputs.

    Args:
        schedule: Native interleaved 1F1B schedule with lowered P2P actions.
        slack: Additional slots above the posted-receive lifetime bound.
    """
    if _INSTALLED:
        raise RuntimeError("Interleaved recv pooling cannot share the process-wide ordinary 1F1B pool")
    if type(schedule) is not ScheduleInterleaved1F1B:
        raise ValueError("Recv pooling requires the native interleaved 1F1B schedule")
    # PyTorch's older prepare-infra layout does not expose the per-direction
    # allocation seam used here. Keep native buffers on that third-party API.
    if not all(hasattr(PipelineStage, name) for name in ("_setup_forward_recv_info", "_setup_backward_recv_info")):
        logger.warning("interleaved recv pool requires per-direction stage setup; keeping stock buffers")
        return
    forward_plans: list[tuple[PipelineStage, tuple[int, ...], int]] = []
    backward_plans: list[tuple[PipelineStage, tuple[int, ...], int]] = []
    for stage in schedule._stages:
        if not isinstance(stage, PipelineStage):
            raise TypeError("Recv pooling requires manually constructed PipelineStage instances")
        if stage.grad_recv_info or stage.args_recv_info:
            raise RuntimeError("Install recv pooling before the first pipeline step")
        actions = schedule.pipeline_order_with_comms[stage.group_rank]
        if not stage.is_first:
            slots, count = _recv_slots(actions, stage.stage_index, schedule._n_microbatches, slack, forward=True)
            if count < schedule._n_microbatches:
                forward_plans.append((stage, slots, count))
        if not stage.is_last:
            slots, count = _recv_slots(actions, stage.stage_index, schedule._n_microbatches, slack)
            if count < schedule._n_microbatches:
                backward_plans.append((stage, slots, count))
    for stage, slots, count in forward_plans:
        _pool_stage_forward_recvs(stage, slots, count)
    for stage, slots, count in backward_plans:
        _pool_stage_backward_recvs(stage, slots, count)

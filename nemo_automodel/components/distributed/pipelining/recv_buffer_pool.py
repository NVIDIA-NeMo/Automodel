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

"""Ring-pooled point-to-point recv buffers for torch pipeline stages.

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
"""

import logging

logger = logging.getLogger(__name__)

_INSTALLED = False

# Schedules whose in-flight microbatch count per stage is bounded by
# num_stages - stage_index, making the ring size proof valid. Interleaved /
# looped / zero-bubble schedules and CSV schedules have different (or
# unknown) in-flight envelopes and are intentionally not supported.
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
    """
    global _INSTALLED
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
        for chunk_id in range(k, num_microbatches):
            self.args_recv_info[chunk_id] = self.args_recv_info[chunk_id % k]
        logger.info(
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
        logger.info(
            "recv_buffer_pool: stage %d bwd recv ring %d/%d buffer sets",
            self.stage_index,
            k,
            num_microbatches,
        )

    if hasattr(manual_cls, "_setup_forward_recv_info"):
        # torch >= 2.13: per-direction setup helpers.
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
        # torch 2.12 line (e.g. the NGC 26.06 container): allocation inside
        # _prepare_forward_infra(num_microbatches, args, kwargs) and the base
        # class's _prepare_backward_infra(num_microbatches).
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
    logger.info("recv_buffer_pool: installed (slack=%d)", slack)
    return True

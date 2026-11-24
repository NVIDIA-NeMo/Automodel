# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


"""Collect optimizer-step metrics and finish reports once their tensors are on CPU."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import NotRequired, Protocol, TypedDict

import torch
import torch.distributed as dist

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loggers.step_reporting import DeferredStepReporter


class MetricBatch(TypedDict):
    """The batch fields needed for supervised-token and throughput metrics."""

    labels: torch.Tensor
    input_ids: NotRequired[torch.Tensor]
    dataset_id: NotRequired[torch.Tensor]


class _DataParallelReduce(Protocol):
    def __call__(
        self, tensor: torch.Tensor, /, *, include_cp: bool = False, keep_device: bool = False
    ) -> torch.Tensor: ...


# Recipes pass in the implementations below, so loggers imports no other component.


class _LabelTokenCounter(Protocol):
    """Count supervised entries in integer labels, like loss.utils._count_label_tokens."""

    def __call__(self, labels: torch.Tensor, ignore_index: int, /) -> int: ...


class _TailPaddingCounter(Protocol):
    """Count trailing padding in labels [batch, sequence], like training.utils.count_tail_padding."""

    def __call__(self, labels: torch.Tensor, /) -> int: ...


class _FlopsEstimator(Protocol):
    """Estimate model FLOPs for input IDs [batch, sequence], like AutoMFU."""

    def get_flops(self, input_ids: torch.Tensor, /) -> float | None: ...


class _DomainLabelCounter(Protocol):
    """Name the domains and count supervised labels per domain, like DomainMixture."""

    @property
    def names(self) -> Sequence[str]: ...

    def label_counts(
        self, dataset_ids: torch.Tensor, labels: torch.Tensor, /, *, ignore_index: int
    ) -> torch.Tensor: ...


class _MFUCalculator(Protocol):
    """Convert a step's TFLOPs and seconds into MFU percent, like flops_utils.calculate_mfu."""

    def __call__(
        self, tflops: float, world_size: int, time_seconds: float, /, *, reference_mfu: float | None
    ) -> float: ...


@dataclass(frozen=True)
class StepTokenCounts:
    """Device scalars for supervised labels, tokens excluding tail padding, and packs."""

    labels: torch.Tensor
    tokens: torch.Tensor
    packs: torch.Tensor


@dataclass(frozen=True)
class StepMetricReducer:
    """Reduce optimizer-step scalars on the training device."""

    device: torch.device
    dp_reduce: _DataParallelReduce
    count_label_tokens: _LabelTokenCounter
    count_tail_padding: _TailPaddingCounter
    pp_broadcast: Callable[[torch.Tensor], torch.Tensor] | None = None
    control_group: dist.ProcessGroup | None = None

    def reduce(self, value: torch.Tensor, *, include_cp: bool = False) -> torch.Tensor:
        """Reduce a scalar or a vector of shape [metrics] on the training device.

        Args:
            value: Scalar or vector to sum across data-parallel ranks.
            include_cp: Include context-parallel shards in the sum.

        Returns:
            Reduced tensor, possibly aliasing the input.
        """
        if self.device.type == "cuda" and not value.is_cuda:
            value = value.pin_memory().to(self.device, non_blocking=True)
        return self.dp_reduce(value, include_cp=include_cp, keep_device=True)

    def token_counts(self, batches: Sequence[MetricBatch], *, ignore_index: int, pack_size: int) -> StepTokenCounts:
        """Count labels, occupied token positions, and packs over the accumulation window.

        Args:
            batches: Microbatches with CPU labels of shape [batch, sequence].
            ignore_index: Configured supervised-loss mask value.
            pack_size: Number of tokens per pack, or zero for unpacked batches.

        Returns:
            Three integer scalars reduced across data parallelism on the training device.
        """
        labels = [batch["labels"] for batch in batches]
        counts = torch.tensor(
            [
                sum(self.count_label_tokens(target, ignore_index) for target in labels),
                sum(target.numel() - self.count_tail_padding(target) for target in labels),
                sum(target.shape[0] for target in labels) if pack_size else 0,
            ],
            dtype=torch.long,
            device="cpu",
        )
        return StepTokenCounts(*self.reduce(counts).unbind())

    def domain_counts(
        self, batches: Sequence[MetricBatch], *, mixture: _DomainLabelCounter, ignore_index: int
    ) -> torch.Tensor:
        """Validate domain IDs and sum supervised counts for each domain.

        Args:
            batches: Microbatches with CPU labels [batch, sequence] and domain IDs [batch].
            mixture: Resolves domain IDs and their order in the reduced vector.
            ignore_index: Configured supervised-loss mask value.

        Returns:
            Integer counts of shape [domains] on the training device.
        """
        counts = torch.zeros(len(mixture.names), dtype=torch.long, device="cpu")
        local_error = ""
        # All ranks check the status before any rank enters another device collective.
        try:
            for batch in batches:
                dataset_ids = batch.get("dataset_id")
                if dataset_ids is None:
                    raise ValueError(
                        "domain_mixture requires each training batch to contain dataset_id; "
                        "use a blended dataset that emits one ID per sample"
                    )
                counts += mixture.label_counts(dataset_ids, batch["labels"], ignore_index=ignore_index).cpu()
        except (TypeError, ValueError) as exc:
            local_error = f"{type(exc).__name__}: {exc}"
            counts.zero_()
        failed = torch.tensor(int(bool(local_error)), dtype=torch.int32, device="cpu")
        if self.control_group is not None:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.control_group)
        elif dist.is_initialized():
            failed = self.dp_reduce(failed)
        if failed.item():
            raise ValueError(
                local_error
                or "domain_mixture rejected the dataset_id of another data-parallel rank's batch; "
                "see that rank's log for the specific error"
            )
        return self.reduce(counts)

    def flops(self, batches: Sequence[MetricBatch], calculator: _FlopsEstimator | None) -> torch.Tensor | None:
        """Estimate the optimizer step's total FLOPs from its input shapes.

        Args:
            batches: Microbatches with input IDs of shape [batch, sequence].
            calculator: Model FLOPs estimator, if available.

        Returns:
            Scalar total over data-parallel ranks on the training device, or None for an unsupported input.
        """
        if not batches or calculator is None:
            return None
        total = 0.0
        for batch in batches:
            input_ids = batch.get("input_ids")
            flops = calculator.get_flops(input_ids) if input_ids is not None else None
            if flops is None:
                return None
            total += float(flops)
        # Sum over DP ranks only, like the token count. Every CP rank of a DP replica holds
        # the same unsharded batches (CP shards a copy inside the step), so including CP
        # would count each sequence's FLOPs cp_size times.
        return self.reduce(torch.tensor(total, dtype=torch.float64, device="cpu"))

    def loss(self, losses: Sequence[torch.Tensor], *, normalize_by: torch.Tensor | None) -> torch.Tensor:
        """Reduce reported loss and share the last pipeline stage's result.

        Args:
            losses: Detached scalar losses from local accumulation batches.
            normalize_by: Optional scalar token count on the training device.

        Returns:
            Scalar reported loss on the training device.
        """
        loss = torch.stack(tuple(losses)).sum()
        if self.pp_broadcast is not None:
            # PP stages can contribute losses of different dtypes.
            loss = loss.float()
        loss = self.reduce(loss, include_cp=True)
        if normalize_by is not None:
            loss = loss.float() / normalize_by
        return self.pp_broadcast(loss) if self.pp_broadcast is not None else loss


class TrainingStepReporter(DeferredStepReporter):
    """Finalize training throughput, MFU, pack fill, and domain counts on CPU."""

    def __init__(
        self,
        device: torch.device,
        *,
        world_size: int,
        pack_size: int,
        domain_names: Sequence[str],
        reference_mfu: float | None,
        calculate_mfu: _MFUCalculator,
    ) -> None:
        """Configure metric derivation for the lifetime of a training loop."""
        super().__init__(device)
        self._world_size = world_size
        self._pack_size = pack_size
        self._domain_names = domain_names
        self._reference_mfu = reference_mfu
        self._calculate_mfu = calculate_mfu
        self._throughput_window: tuple[float, int, int] | None = None

    def poll(self, *, wait: bool = False) -> list[tuple[MetricsSample, bool, float]]:
        """Return completed training reports in order, with derived metrics."""
        completed = super().poll(wait=wait)
        self._record_throughput(completed)
        for sample, _, elapsed in completed:
            metrics = sample.metrics
            tps = metrics["num_tokens_per_step"] / elapsed
            metrics.update(tps=tps, tps_per_gpu=tps / self._world_size, step_time_seconds=elapsed, mfu=None)
            if (step_flops := metrics.pop("_step_flops", None)) is not None:
                metrics["mfu"] = self._calculate_mfu(
                    step_flops / 1e12, self._world_size, elapsed, reference_mfu=self._reference_mfu
                )
            if (num_packs := metrics.pop("_num_packs", None)) is not None:
                metrics["pack_fill"] = metrics["num_tokens_per_step"] / (num_packs * self._pack_size)
            if (counts := metrics.pop("_domain_label_counts", None)) is not None:
                counts = counts.tolist()
                total = max(sum(counts), 1)
                for name, count in zip(self._domain_names, counts):
                    metrics[f"domain_mixture/{name}_label_tokens"] = count
                    metrics[f"domain_mixture/{name}_label_fraction"] = count / total
        return completed

    def _record_throughput(self, completed: list[tuple[MetricsSample, bool, float]]) -> None:
        if not completed:
            return
        now = perf_counter()
        if self._throughput_window is None:
            self._throughput_window = (now, 0, 0)
            return
        since, tokens, steps = self._throughput_window
        tokens += sum(sample.metrics["num_tokens_per_step"] for sample, _, _ in completed)
        steps += len(completed)
        if steps >= 20:
            completed[-1][0].metrics["end_to_end_tps"] = tokens / (now - since)
            self._throughput_window = (now, 0, 0)
        else:
            self._throughput_window = (since, tokens, steps)


class ExpertLoad(TypedDict):
    """A layer's expert counts, auxiliary loss, and expert count."""

    expert_load: torch.Tensor
    aux_loss: torch.Tensor | None
    n_experts: int


def snapshot_moe_metrics(sample: MetricsSample, loads: dict[str, ExpertLoad]) -> None:
    """Attach expert counts [experts] and scalar auxiliary losses for readback."""
    for name, data in loads.items():
        sample.metrics[f"_moe_load/{name}"] = data["expert_load"]
        if data["aux_loss"] is not None:
            sample.metrics[f"_moe_aux/{name}"] = data["aux_loss"].reshape(1)


def restore_moe_metrics(sample: MetricsSample) -> dict[str, ExpertLoad]:
    """Remove readback tensors from the sample and return layer loads for logging."""
    loads: dict[str, ExpertLoad] = {}
    for key in tuple(sample.metrics):
        if key.startswith("_moe_load/"):
            name = key.removeprefix("_moe_load/")
            load = sample.metrics.pop(key)
            loads[name] = {
                "expert_load": load,
                "aux_loss": sample.metrics.pop(f"_moe_aux/{name}", None),
                "n_experts": load.numel(),
            }
    return loads


def write_remote_step_metrics(
    *,
    step: int,
    metrics: dict[str, object],
    sinks: Sequence[Callable[..., None]],
    moe_metrics: Callable[[int], dict[str, float]],
) -> None:
    """Send the step sample and compute its MoE payload once for all sinks."""
    if not sinks:
        return
    for log in sinks:
        log(metrics, step=step)
    if loads := moe_metrics(step):
        for log in sinks:
            log(loads, step=step)

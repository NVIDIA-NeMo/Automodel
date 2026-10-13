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

"""OpenTelemetry spans and metrics through the optional ``nemo-lens`` package.

Two parts live here.

Call-site helpers (:func:`span`, :func:`traced`, :func:`set_span_attributes`) let recipes and
components emit spans. ``nemo-lens`` is resolved once, at import, through
:func:`~nemo_automodel.shared.import_utils.safe_import`. When it is not installed, or the span group of
a site is not enabled, the site creates no span and costs one function call and one set lookup.

:class:`TelemetryConfig` is the typed ``telemetry:`` YAML block. Its :meth:`TelemetryConfig.build` is
the only code that installs OpenTelemetry providers, and only recipes call it. Library code must never
initialize telemetry: NeMo-RL imports Automodel components into worker processes whose telemetry it
already owns, and a second initialization there either raises or silences the host's spans. Inside
such a process the component spans below still record, under the host's providers, whenever the host
enables a span group of the same name.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch.distributed as dist

from nemo_automodel.shared.import_utils import UnavailableError, safe_import

if TYPE_CHECKING:
    from nemo.lens import TelemetryHandle
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

_INSTALL_HINT = "nemo-lens is not installed. To enable telemetry, run: uv sync --extra telemetry"

HAVE_NEMO_LENS, _lens = safe_import("nemo.lens", msg=_INSTALL_HINT)
_, _lens_state = safe_import("nemo.lens.state", msg=_INSTALL_HINT)
_, _lens_instruments = safe_import("nemo.lens.instruments", msg=_INSTALL_HINT)
_, _otel_trace = safe_import("opentelemetry.trace", msg=_INSTALL_HINT)
#: ``nemo.lens.semconv``: attribute-name constants. Read it only where a span exists, which implies
#: that nemo-lens is installed.
_, semconv = safe_import("nemo.lens.semconv", msg=_INSTALL_HINT)

# ---------------------------------------------------------------------------
# Span groups
# ---------------------------------------------------------------------------

#: Namespace that Automodel registers its span groups and metrics under.
NAMESPACE = "nemo_automodel"

# Group names are shared with Megatron-LM and NeMo-RL where the phase and its cost match, so a single
# span-group spec selects the same phase in every library of a job.
MODEL_INIT = "model_init"
"""Model construction, weight download and load, and parallelization. Once per model."""
LOAD_CHECKPOINT = "load_checkpoint"
"""Restoring a training checkpoint. Once per resume."""
CHECKPOINT = "checkpoint"
"""Saving a checkpoint, and waiting on an asynchronous save. Once per checkpoint."""
EVALUATE = "evaluate"
"""One pass over a validation dataset."""
FIRST_ITERATION = "first_iteration"
"""The first optimizer step of the process, which pays one-off costs such as compilation."""
STEP = "step"
"""One optimizer step."""
FORWARD_BACKWARD = "forward_backward"
"""The forward and backward passes of one optimizer step, over all its microbatches."""
OPTIMIZER = "optimizer"
"""Gradient clipping, the optimizer update, and the learning-rate scheduler step."""
DATA_LOADING = "data_loading"
"""Waiting on the dataloader for the batches of one optimizer step."""

_DEFAULT_GROUPS = frozenset({MODEL_INIT, LOAD_CHECKPOINT, CHECKPOINT, EVALUATE, FIRST_ITERATION})
_SPAN_GROUP_PRESETS: dict[str, frozenset[str]] = {
    # Spans that occur a bounded number of times per job.
    "default": _DEFAULT_GROUPS,
    # Adds spans that occur on every optimizer step.
    "per_step": _DEFAULT_GROUPS | {STEP, FORWARD_BACKWARD, OPTIMIZER, DATA_LOADING},
}
_SPAN_GROUPS = _SPAN_GROUP_PRESETS["per_step"]

# ---------------------------------------------------------------------------
# Resource attributes that nemo-lens semconv has no constant for
# ---------------------------------------------------------------------------

CONTEXT_PARALLEL_SIZE = "dl.context_parallel.size"
EXPERT_PARALLEL_SIZE = "dl.expert_parallel.size"
RECIPE = "nemo.automodel.recipe"
MODEL = "nemo.automodel.model"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

TRAIN_STEP_DURATION = "automodel.train.step.duration"
"""Histogram, ms: wall time of one optimizer step on this rank."""
TRAIN_THROUGHPUT = "automodel.train.throughput"
"""Gauge, ``{token}/s``: training tokens per second for the whole job, recorded by the main rank."""

# ---------------------------------------------------------------------------
# Call-site helpers
# ---------------------------------------------------------------------------

_NO_SPAN: AbstractContextManager[None] = nullcontext()
_F = TypeVar("_F", bound=Callable[..., Any])


def enabled(group: str) -> bool:
    """Return whether spans of ``group`` are recorded in this process.

    Guard any work done only for telemetry with it, such as building attributes for
    :func:`set_span_attributes`. That includes reading :data:`semconv`, which raises when nemo-lens is
    not installed.

    Args:
        group: Span group name, e.g. :data:`CHECKPOINT`.

    Returns:
        ``True`` when nemo-lens is installed and ``group`` is enabled.
    """
    return HAVE_NEMO_LENS and _lens_state.is_span_group_enabled(group)


def span(group: str, name: str) -> AbstractContextManager[Span | None]:
    """Return a context manager that records its block as a span when ``group`` is enabled.

    The context manager yields the active span, or ``None`` when nemo-lens is not installed or
    ``group`` is not enabled. The block runs either way. Compute attribute values only inside
    ``if span is not None:``, so that a disabled site does no work for them. An exception raised
    in the block is recorded on the span, which ends with ERROR status, and propagates unchanged.

    Args:
        group: Span group that gates the site, e.g. :data:`STEP`.
        name: Span name, e.g. ``"automodel.train.step"``.

    Returns:
        A context manager that yields the span, or ``None``.
    """
    if HAVE_NEMO_LENS and _lens_state.is_span_group_enabled(group):
        return _lens.managed_span(group, name)
    return _NO_SPAN


def _undecorated(func: _F) -> _F:
    return func


def traced(group: str, name: str) -> Callable[[_F], _F]:
    """Decorate a function so that each call is recorded as a span when ``group`` is enabled.

    The group is checked on every call, so a group enabled after import takes effect. Without
    nemo-lens the function is returned unchanged. Do not decorate generator functions: the span
    would cover only the creation of the generator.

    Args:
        group: Span group that gates the function, e.g. :data:`CHECKPOINT`.
        name: Span name, e.g. ``"automodel.checkpoint.save.model"``.

    Returns:
        The decorator.
    """
    if not HAVE_NEMO_LENS:
        return _undecorated
    return _lens.trace_fn(group, name)


def set_span_attributes(group: str, attributes: Mapping[str, str | bool | int | float]) -> None:
    """Set attributes on the span of the enclosing :func:`traced` call when ``group`` is enabled.

    Call it from the decorated function, outside any span the function opens itself, and only
    under ``if enabled(group):`` so that the attributes are not built for a disabled site. When
    ``group`` is not enabled the current span, if any, belongs to an enclosing site, so nothing is
    set.

    Args:
        group: The span group passed to :func:`traced`.
        attributes: Attribute names and scalar values.
    """
    if HAVE_NEMO_LENS and _lens_state.is_span_group_enabled(group):
        _lens.safe_set_span_attributes(_otel_trace.get_current_span(), dict(attributes))


def record_train_step(
    handle: TelemetryHandle,
    *,
    step_duration_ms: float,
    tokens_per_second: float | None,
) -> None:
    """Record the metrics of one optimizer step.

    Args:
        handle: The handle returned by :meth:`TelemetryConfig.build`.
        step_duration_ms: Wall time of the step on this rank, in milliseconds.
        tokens_per_second: Job-wide training throughput. Pass ``None`` on every rank but the main
            one, so the job reports it once.
    """
    _lens_instruments.record_metrics(
        handle.meter,
        NAMESPACE,
        train_step_duration=step_duration_ms,
        train_throughput=tokens_per_second,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

_HANDLE: TelemetryHandle | None = None
_HANDLE_LOCK = threading.Lock()


@dataclass
class TelemetryConfig:
    """OpenTelemetry configuration: the YAML ``telemetry:`` block.

    As for every Automodel config section, the YAML and its CLI overrides are the only source of
    these values. To take one from the environment, interpolate it, for example
    ``enabled: ${NEMO_LENS_ENABLED,false}``. The OTLP endpoint, headers, and protocol come from the
    standard ``OTEL_EXPORTER_OTLP_*`` environment variables, which the OpenTelemetry SDK reads.

    Attributes:
        enabled: Install OpenTelemetry providers on every rank and record spans and metrics.
        exporter: ``"otlp"`` to send to an OTLP endpoint, or ``"console"`` to write JSON lines to
            standard output.
        span_groups: Span groups to record: a preset (``default``, ``per_step`` or ``all``), a group
            name, or a comma-separated list of both.
        service_name: OpenTelemetry ``service.name`` resource attribute.
        run_id: Identifier shared by every rank of the job, exported as ``nemo.run.id``. When unset,
            the Slurm job id is used; without one, rank 0 generates an id and broadcasts it.
        traces_enabled: Export spans.
        metrics_enabled: Export metrics.
    """

    enabled: bool = False
    exporter: str = "otlp"
    span_groups: str = "default"
    service_name: str = "nemo-automodel"
    run_id: str | None = None
    traces_enabled: bool = True
    metrics_enabled: bool = True

    def build(
        self,
        *,
        rank: int,
        world_size: int,
        resource_attributes: Mapping[str, str | bool | int | float],
    ) -> TelemetryHandle | None:
        """Install the OpenTelemetry providers of this process and return the handle.

        Call it once per process, on every rank, after ``torch.distributed`` is initialized. A later
        call returns the first handle. When ``enabled`` is false nothing is installed: OpenTelemetry
        providers can be set only once per process, so a disabled build leaves them to any other
        library. When another library already installed a tracer provider, this logs a warning and
        returns ``None``.

        Args:
            rank: Global rank of this process.
            world_size: Number of processes in the job.
            resource_attributes: Attributes that are constant for the life of the process, such as
                the parallel layout. Rank and world size are added here.

        Returns:
            The handle, or ``None`` when telemetry is disabled or owned by another library.

        Raises:
            UnavailableError: ``enabled`` is true but nemo-lens is not installed.
        """
        global _HANDLE

        if not self.enabled:
            return None
        if not HAVE_NEMO_LENS:
            raise UnavailableError(_INSTALL_HINT)

        with _HANDLE_LOCK:
            if _HANDLE is not None:
                return _HANDLE
            if not isinstance(_otel_trace.get_tracer_provider(), _otel_trace.ProxyTracerProvider):
                logger.warning(
                    "An OpenTelemetry tracer provider is already installed in this process, so "
                    "telemetry.enabled is ignored and spans go to that provider."
                )
                return None

            _register_span_groups_and_metrics()
            config = _lens.NemoLensConfig(
                enabled=True,
                service_name=self.service_name,
                traces_enabled=self.traces_enabled,
                metrics_enabled=self.metrics_enabled,
                span_groups=self.span_groups,
                exporter=self.exporter,
                run_id=self.run_id or _job_run_id(rank=rank, world_size=world_size),
            )
            handle = _lens.setup_telemetry(
                config,
                resource_attributes={
                    **resource_attributes,
                    semconv.NV_DL_RANK: rank,
                    semconv.NV_DL_WORLD_SIZE: world_size,
                },
            )
            atexit.register(handle.shutdown)
            _HANDLE = handle
            logger.info(
                "Telemetry enabled: exporter=%s span_groups=%s run_id=%s",
                self.exporter,
                self.span_groups,
                config.run_id,
            )
            return handle


def _register_span_groups_and_metrics() -> None:
    """Declare Automodel's span groups and metrics to nemo-lens, once per process."""
    if NAMESPACE not in _lens.SpanRegistry.namespaces():
        _lens.SpanRegistry.register(NAMESPACE, _SPAN_GROUPS, presets=_SPAN_GROUP_PRESETS)
    if NAMESPACE not in _lens_instruments.registered_metric_groups():
        _lens_instruments.register_metric_group(
            NAMESPACE,
            [
                _lens_instruments.MetricSpec(
                    key="train_step_duration",
                    name=TRAIN_STEP_DURATION,
                    kind="histogram",
                    unit="ms",
                    description="Wall time of one optimizer step on this rank.",
                ),
                _lens_instruments.MetricSpec(
                    key="train_throughput",
                    name=TRAIN_THROUGHPUT,
                    kind="gauge",
                    unit="{token}/s",
                    description="Training tokens per second for the whole job.",
                ),
            ],
        )


def _job_run_id(*, rank: int, world_size: int) -> str:
    """Return an id shared by every rank: the Slurm job id, else one generated by rank 0.

    Without Slurm, nemo-lens would generate a different id in every process. The broadcast is a
    collective, which is safe because every rank reads ``enabled`` from the same YAML.
    """
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if slurm_job_id:
        return slurm_job_id
    run_id = [uuid.uuid4().hex[:12] if rank == 0 else ""]
    if world_size > 1 and dist.is_initialized():
        dist.broadcast_object_list(run_id, src=0)
    return run_id[0]

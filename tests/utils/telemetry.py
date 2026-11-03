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

"""Fixtures for tests that observe the spans and metrics Automodel emits through nemo-lens.

OpenTelemetry providers, the nemo-lens span-group registry, and Automodel's telemetry handle are
process-global. :func:`reset_telemetry_state` returns all of them to the state of a fresh process,
so a test can install telemetry without leaking it into the next test.
"""

from dataclasses import dataclass
from typing import Any

import pytest


def reset_telemetry_state() -> None:
    """Return OpenTelemetry, nemo-lens, and Automodel telemetry state to that of a fresh process."""
    import nemo.lens.handle as lens_handle
    import opentelemetry.metrics._internal as otel_metrics
    import opentelemetry.trace as otel_trace
    from nemo.lens import SpanRegistry, set_enabled_span_groups
    from nemo.lens.instruments import registered_metric_groups, unregister_metric_group
    from opentelemetry.util._once import Once

    from nemo_automodel.shared import telemetry

    for provider in (otel_trace.get_tracer_provider(), otel_metrics.get_meter_provider()):
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    otel_trace._TRACER_PROVIDER = None
    otel_trace._TRACER_PROVIDER_SET_ONCE = Once()
    otel_metrics._METER_PROVIDER = None
    otel_metrics._METER_PROVIDER_SET_ONCE = Once()
    lens_handle._INITIALIZED = False
    set_enabled_span_groups(frozenset())
    SpanRegistry.clear()
    if telemetry.NAMESPACE in registered_metric_groups():
        unregister_metric_group(telemetry.NAMESPACE)
    telemetry._HANDLE = None


@dataclass
class LensCapture:
    """Telemetry installed for one test, with in-memory span and metric readers."""

    handle: Any
    span_exporter: Any
    metric_reader: Any

    def spans(self) -> list:
        """Return every span finished so far."""
        import opentelemetry.trace as otel_trace

        otel_trace.get_tracer_provider().force_flush()
        return list(self.span_exporter.get_finished_spans())

    def span_named(self, name: str) -> list:
        """Return the finished spans called ``name``."""
        return [span for span in self.spans() if span.name == name]

    def metrics(self) -> dict[str, Any]:
        """Return the collected metrics, keyed by name."""
        data = self.metric_reader.get_metrics_data()
        return {
            metric.name: metric
            for resource_metrics in data.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        }


@pytest.fixture
def lens_capture(monkeypatch):
    """Install nemo-lens with in-memory readers and Automodel's span groups; return a factory.

    Call the factory with a span-group spec, e.g. ``lens_capture("per_step")``. The state is reset
    before and after the test.
    """
    pytest.importorskip("nemo.lens")
    pytest.importorskip("opentelemetry.sdk")
    from nemo.lens import NemoLensConfig, setup_telemetry
    from nemo.lens.semconv import NV_DL_RANK, NV_DL_WORLD_SIZE
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nemo_automodel.shared import telemetry

    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    reset_telemetry_state()

    def install(span_groups: str) -> LensCapture:
        telemetry._register_span_groups_and_metrics()
        span_exporter = InMemorySpanExporter()
        metric_reader = InMemoryMetricReader()
        handle = setup_telemetry(
            NemoLensConfig(enabled=True, exporter="console", span_groups=span_groups, run_id="test-run"),
            resource_attributes={NV_DL_RANK: 0, NV_DL_WORLD_SIZE: 1},
            span_exporter=span_exporter,
            metric_reader=metric_reader,
        )
        return LensCapture(handle=handle, span_exporter=span_exporter, metric_reader=metric_reader)

    yield install
    reset_telemetry_state()

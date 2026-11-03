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

import atexit
import importlib
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.shared import telemetry
from nemo_automodel.shared.import_utils import UnavailableError
from tests.utils.telemetry import lens_capture, reset_telemetry_state  # noqa: F401

_PER_STEP_ONLY = {telemetry.STEP, telemetry.FORWARD_BACKWARD, telemetry.OPTIMIZER, telemetry.DATA_LOADING}


@pytest.fixture
def clean_lens(monkeypatch):
    """Fresh telemetry state, no Slurm or resource-attribute environment, and recorded ``atexit`` hooks.

    The OpenTelemetry SDK registers its own exit hooks when providers are built; they are recorded too.
    """
    pytest.importorskip("nemo.lens")
    pytest.importorskip("opentelemetry.sdk")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("OTEL_RESOURCE_ATTRIBUTES", raising=False)
    exit_hooks = []
    monkeypatch.setattr(atexit, "register", lambda func, *args, **kwargs: exit_hooks.append(func))
    reset_telemetry_state()
    yield exit_hooks
    reset_telemetry_state()


def _attach_memory_exporter():
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


# ---------------------------------------------------------------------------
# TelemetryConfig.build
# ---------------------------------------------------------------------------


def test_disabled_build_leaves_another_library_s_telemetry_alone(clean_lens):
    from nemo.lens import enabled_span_groups, set_enabled_span_groups
    from opentelemetry import trace

    # A host such as NeMo-RL has already enabled its groups in this process.
    set_enabled_span_groups(frozenset({"checkpoint"}))

    assert telemetry.TelemetryConfig().build(rank=0, world_size=1, resource_attributes={}) is None

    assert enabled_span_groups() == frozenset({"checkpoint"})
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
    assert clean_lens == []


def test_enabled_build_without_nemo_lens_raises_with_install_hint(monkeypatch):
    monkeypatch.setattr(telemetry, "HAVE_NEMO_LENS", False)
    with pytest.raises(UnavailableError, match="uv sync --extra telemetry"):
        telemetry.TelemetryConfig(enabled=True).build(rank=0, world_size=1, resource_attributes={})


def test_build_installs_providers_with_rank_identity(clean_lens):
    from nemo.lens import enabled_span_groups

    config = telemetry.TelemetryConfig(enabled=True, exporter="console", span_groups="per_step", run_id="run-1")
    handle = config.build(rank=3, world_size=8, resource_attributes={"service.version": "1.2.3"})
    exporter = _attach_memory_exporter()

    with telemetry.span(telemetry.STEP, "automodel.train.step"):
        pass

    (span,) = exporter.get_finished_spans()
    resource = span.resource.attributes
    assert resource["service.name"] == "nemo-automodel"
    assert resource["service.version"] == "1.2.3"
    assert resource["nemo.run.id"] == "run-1"
    assert resource["nv.dl.rank"] == 3
    assert resource["nv.dl.world_size"] == 8
    assert resource["service.instance.id"] == "run-1-rank3"
    assert enabled_span_groups() == telemetry._SPAN_GROUPS
    # Once per process: a second build returns the first handle and registers no second exit hook.
    assert config.build(rank=3, world_size=8, resource_attributes={}) is handle
    assert clean_lens.count(handle.shutdown) == 1


def test_build_uses_the_slurm_job_id_as_run_id(clean_lens, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setattr(dist, "broadcast_object_list", pytest.fail)
    telemetry.TelemetryConfig(enabled=True, exporter="console").build(rank=0, world_size=4, resource_attributes={})
    exporter = _attach_memory_exporter()
    with telemetry.span(telemetry.CHECKPOINT, "automodel.checkpoint.save"):
        pass
    assert exporter.get_finished_spans()[0].resource.attributes["nemo.run.id"] == "4242"


def test_build_defers_to_a_tracer_provider_another_library_installed(clean_lens, caplog):
    import nemo.lens.handle as lens_handle
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    trace.set_tracer_provider(TracerProvider())
    with caplog.at_level(logging.WARNING):
        handle = telemetry.TelemetryConfig(enabled=True, exporter="console").build(
            rank=0, world_size=1, resource_attributes={}
        )

    assert handle is None
    assert "already installed" in caplog.text
    assert lens_handle._INITIALIZED is False
    assert not telemetry.enabled(telemetry.CHECKPOINT)


def _build_and_report_run_id(rank: int, rendezvous: str, out_dir: str) -> None:
    os.environ.pop("SLURM_JOB_ID", None)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    try:
        from opentelemetry import trace

        telemetry.TelemetryConfig(enabled=True, exporter="console").build(
            rank=rank, world_size=2, resource_attributes={}
        )
        run_id = trace.get_tracer_provider().resource.attributes["nemo.run.id"]
        Path(out_dir, f"rank{rank}").write_text(run_id)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(30, reason="Two fresh Gloo workers each import torch and nemo-lens.")
def test_ranks_without_slurm_share_one_run_id(tmp_path):
    pytest.importorskip("nemo.lens")
    pytest.importorskip("opentelemetry.sdk")
    mp.spawn(_build_and_report_run_id, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)

    run_ids = {(tmp_path / f"rank{rank}").read_text() for rank in range(2)}
    assert len(run_ids) == 1
    assert len(next(iter(run_ids))) == 12


# ---------------------------------------------------------------------------
# Call-site helpers
# ---------------------------------------------------------------------------


def test_span_records_nested_spans_attributes_and_errors(lens_capture):
    capture = lens_capture("per_step")

    with telemetry.span(telemetry.STEP, "automodel.train.step") as step:
        step.set_attribute(telemetry.semconv.DL_ITERATION, 7)
        with telemetry.span(telemetry.FORWARD_BACKWARD, "automodel.train.forward_backward"):
            pass
    with pytest.raises(ValueError, match="boom"):
        with telemetry.span(telemetry.EVALUATE, "automodel.evaluate"):
            raise ValueError("boom")

    (step_span,) = capture.span_named("automodel.train.step")
    (fb_span,) = capture.span_named("automodel.train.forward_backward")
    (eval_span,) = capture.span_named("automodel.evaluate")
    assert step_span.attributes["dl.iteration"] == 7
    assert fb_span.parent.span_id == step_span.context.span_id
    assert eval_span.status.status_code.name == "ERROR"
    assert eval_span.events[0].name == "exception"


def test_disabled_group_records_nothing_and_never_writes_to_the_enclosing_span(lens_capture):
    capture = lens_capture(telemetry.STEP)

    @telemetry.traced(telemetry.CHECKPOINT, "automodel.checkpoint.save")
    def save(step):
        if telemetry.enabled(telemetry.CHECKPOINT):
            telemetry.set_span_attributes(telemetry.CHECKPOINT, {"automodel.checkpoint.step": step})
        telemetry.set_span_attributes(telemetry.CHECKPOINT, {"automodel.checkpoint.unguarded": True})
        return step + 1

    with telemetry.span(telemetry.STEP, "automodel.train.step"):
        assert save(4) == 5
        with telemetry.span(telemetry.CHECKPOINT, "automodel.checkpoint.wait") as wait:
            assert wait is None

    (step_span,) = capture.spans()
    assert step_span.name == "automodel.train.step"
    assert dict(step_span.attributes) == {}


def test_helpers_are_inert_without_nemo_lens(monkeypatch):
    for name in [m for m in sys.modules if m == "nemo.lens" or m.startswith("nemo.lens.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "nemo.lens", None)
    try:
        importlib.reload(telemetry)
        assert telemetry.HAVE_NEMO_LENS is False
        assert telemetry.enabled(telemetry.STEP) is False
        with telemetry.span(telemetry.STEP, "automodel.train.step") as span:
            assert span is None

        def func(value):
            return value

        assert telemetry.traced(telemetry.CHECKPOINT, "automodel.checkpoint.save")(func) is func
        telemetry.set_span_attributes(telemetry.CHECKPOINT, {"automodel.checkpoint.step": 1})
        assert telemetry.TelemetryConfig().build(rank=0, world_size=1, resource_attributes={}) is None
        with pytest.raises(UnavailableError):
            telemetry.TelemetryConfig(enabled=True).build(rank=0, world_size=1, resource_attributes={})
    finally:
        monkeypatch.undo()
        importlib.reload(telemetry)


# ---------------------------------------------------------------------------
# Span groups and metrics
# ---------------------------------------------------------------------------


def test_default_preset_holds_only_bounded_per_job_groups(lens_capture):
    from nemo.lens import SpanRegistry

    lens_capture("default")
    default = SpanRegistry.resolve("default")[0]
    per_step = SpanRegistry.resolve("per_step")[0]
    assert default == {
        telemetry.MODEL_INIT,
        telemetry.LOAD_CHECKPOINT,
        telemetry.CHECKPOINT,
        telemetry.EVALUATE,
        telemetry.FIRST_ITERATION,
    }
    assert per_step == default | _PER_STEP_ONLY
    assert not default & _PER_STEP_ONLY


def test_metric_registry_matches_emitted_names_and_units(lens_capture):
    from nemo.lens.instruments import registered_metric_groups

    lens_capture("default")
    specs = registered_metric_groups()[telemetry.NAMESPACE]
    names = {spec.name for spec in specs}
    assert names == {telemetry.TRAIN_STEP_DURATION, telemetry.TRAIN_THROUGHPUT}
    for spec in specs:
        assert spec.key == spec.name.removeprefix("automodel.").replace(".", "_")
    assert {spec.name: (spec.kind, spec.unit) for spec in specs} == {
        telemetry.TRAIN_STEP_DURATION: ("histogram", "ms"),
        telemetry.TRAIN_THROUGHPUT: ("gauge", "{token}/s"),
    }


def test_record_train_step_emits_both_metrics(lens_capture):
    capture = lens_capture("default")

    telemetry.record_train_step(capture.handle, step_duration_ms=12.5, tokens_per_second=1000.0)
    telemetry.record_train_step(capture.handle, step_duration_ms=7.5, tokens_per_second=None)

    metrics = capture.metrics()
    (duration,) = metrics[telemetry.TRAIN_STEP_DURATION].data.data_points
    (throughput,) = metrics[telemetry.TRAIN_THROUGHPUT].data.data_points
    assert (duration.count, duration.sum) == (2, 20.0)
    assert throughput.value == 1000.0


# ---------------------------------------------------------------------------
# The YAML ``telemetry:`` block
# ---------------------------------------------------------------------------


def test_recipe_config_telemetry_defaults_to_disabled():
    assert RecipeConfig(ConfigNode({})).telemetry == telemetry.TelemetryConfig()
    assert telemetry.TelemetryConfig().enabled is False


def test_recipe_config_telemetry_reads_the_yaml_block(monkeypatch):
    monkeypatch.setenv("AUTOMODEL_TEST_OTEL_ENABLED", "true")
    node = ConfigNode(
        {
            "telemetry": {
                "enabled": "${AUTOMODEL_TEST_OTEL_ENABLED,false}",
                "exporter": "console",
                "span_groups": "per_step,checkpoint",
                "run_id": "job-7",
            }
        }
    )

    config = RecipeConfig(node).telemetry

    assert config == telemetry.TelemetryConfig(
        enabled=True, exporter="console", span_groups="per_step,checkpoint", run_id="job-7"
    )


def test_recipe_config_telemetry_rejects_unknown_keys():
    with pytest.raises(TypeError, match="enabeld"):
        RecipeConfig(ConfigNode({"telemetry": {"enabeld": True}})).telemetry

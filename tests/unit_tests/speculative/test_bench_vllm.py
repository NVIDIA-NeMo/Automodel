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

from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace

import pytest

from nemo_automodel.components.speculative import bench_vllm
from nemo_automodel.components.speculative.bench_common import WorkloadResult, _validate_workload_args
from nemo_automodel.components.speculative.bench_vllm import (
    SpecMetrics,
    _accept_length,
    _acceptance_rate,
    _fetch_spec_metrics,
    _parse_spec_metrics,
    _summarize,
)
from nemo_automodel.components.speculative.regenerate import GenerationConfig

_METRICS_TEXT = """\
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total{engine="0",model_name="Qwen/Qwen3-8B"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 1500.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="Qwen/Qwen3-8B"} 900.0
vllm:num_requests_running{engine="0"} 0.0
"""


# ---------------------------------------------------------------------------
# Prometheus parsing
# ---------------------------------------------------------------------------
def test_parse_spec_metrics_reads_all_counters():
    metrics = _parse_spec_metrics(_METRICS_TEXT)
    assert metrics == SpecMetrics(num_drafts=100.0, num_draft_tokens=1500.0, num_accepted_tokens=900.0)


def test_parse_spec_metrics_sums_label_sets():
    """Counters split across engines (label sets) are summed."""
    two_engines = _METRICS_TEXT + (
        'vllm:spec_decode_num_drafts_total{engine="1",model_name="Qwen/Qwen3-8B"} 50.0\n'
        'vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="Qwen/Qwen3-8B"} 100.0\n'
    )
    metrics = _parse_spec_metrics(two_engines)
    assert metrics.num_drafts == 150.0
    assert metrics.num_accepted_tokens == 1000.0


def test_parse_spec_metrics_without_spec_counters_returns_none():
    """A server without speculative decoding exposes no spec counters."""
    assert _parse_spec_metrics("vllm:num_requests_running 0.0\n") is None


def test_parse_spec_metrics_unlabeled_samples():
    """Samples without a label block (bare `name value`) parse too."""
    text = (
        "vllm:spec_decode_num_drafts_total 10\n"
        "vllm:spec_decode_num_draft_tokens_total 70\n"
        "vllm:spec_decode_num_accepted_tokens_total 30\n"
    )
    assert _parse_spec_metrics(text) == SpecMetrics(10.0, 70.0, 30.0)


def test_parse_spec_metrics_skips_malformed_values():
    text = "vllm:spec_decode_num_drafts_total NaN-ish\nvllm:spec_decode_num_drafts_total 5\n"
    metrics = _parse_spec_metrics(text)
    assert metrics is not None
    assert metrics.num_drafts == 5.0


# ---------------------------------------------------------------------------
# Acceptance math
# ---------------------------------------------------------------------------
def test_delta_subtracts_and_clamps():
    before = SpecMetrics(100.0, 1500.0, 900.0)
    after = SpecMetrics(160.0, 2400.0, 1500.0)
    assert after.delta(before) == SpecMetrics(60.0, 900.0, 600.0)
    # A counter reset (server restart mid-run) clamps at zero instead of going negative.
    assert before.delta(after) == SpecMetrics(0.0, 0.0, 0.0)


def test_accept_length_includes_bonus_token():
    assert _accept_length(SpecMetrics(num_drafts=100, num_draft_tokens=1500, num_accepted_tokens=900)) == 10.0


def test_accept_length_none_cases():
    assert _accept_length(None) is None
    assert _accept_length(SpecMetrics(0, 0, 0)) is None


def test_acceptance_rate():
    assert _acceptance_rate(SpecMetrics(100, 1500, 900)) == pytest.approx(0.6)
    assert _acceptance_rate(None) is None
    assert _acceptance_rate(SpecMetrics(1, 0, 0)) is None


# ---------------------------------------------------------------------------
# Summary assembly
# ---------------------------------------------------------------------------
def _gen_cfg() -> GenerationConfig:
    return GenerationConfig(model="Qwen/Qwen3-8B", max_new_tokens=64, temperature=0.0, top_p=1.0)


def test_summarize_full():
    summary = _summarize(
        gen_cfg=_gen_cfg(),
        spec_result=WorkloadResult(wall_clock_s=2.0, output_tokens=400, completed=8, failed=0),
        metrics_delta=SpecMetrics(num_drafts=50, num_draft_tokens=750, num_accepted_tokens=450),
        baseline_result=WorkloadResult(wall_clock_s=4.0, output_tokens=400, completed=8, failed=0),
    )
    assert summary["accept_length"] == 10.0
    assert summary["acceptance_rate"] == pytest.approx(0.6)
    assert summary["num_drafts"] == 50
    assert summary["output_throughput_tok_s"] == 200.0
    assert summary["baseline_throughput_tok_s"] == 100.0
    assert summary["speedup"] == 2.0


def test_summarize_without_metrics_or_baseline():
    summary = _summarize(
        gen_cfg=_gen_cfg(),
        spec_result=WorkloadResult(wall_clock_s=2.0, output_tokens=400, completed=8, failed=0),
        metrics_delta=None,
        baseline_result=None,
    )
    assert summary["accept_length"] is None
    assert summary["acceptance_rate"] is None
    assert summary["num_drafts"] is None
    assert "speedup" not in summary


# ---------------------------------------------------------------------------
# /metrics fetching (fake aiohttp)
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status, text=""):
        self.status = status
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, get_response):
        self._get_response = get_response
        self.get_calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def get(self, url, *, timeout=None):
        self.get_calls.append(url)
        return self._get_response


def _patch_aiohttp(monkeypatch, session):
    fake = SimpleNamespace(ClientSession=lambda *a, **k: session, ClientTimeout=lambda total=None: None)
    monkeypatch.setattr(bench_vllm, "_import_aiohttp", lambda: fake)


def test_fetch_spec_metrics_hits_metrics_endpoint(monkeypatch):
    session = _FakeSession(_FakeResponse(200, _METRICS_TEXT))
    _patch_aiohttp(monkeypatch, session)
    metrics = asyncio.run(_fetch_spec_metrics("http://localhost:8000/v1", timeout_s=1.0))
    assert session.get_calls == ["http://localhost:8000/metrics"]
    assert metrics.num_drafts == 100.0


def test_fetch_spec_metrics_non_200_returns_none(monkeypatch):
    session = _FakeSession(_FakeResponse(404))
    _patch_aiohttp(monkeypatch, session)
    assert asyncio.run(_fetch_spec_metrics("http://localhost:8000", timeout_s=1.0)) is None


def test_fetch_spec_metrics_transport_error_returns_none(monkeypatch):
    class _BoomSession(_FakeSession):
        def get(self, url, *, timeout=None):
            raise ConnectionError("boom")

    _patch_aiohttp(monkeypatch, _BoomSession(None))
    assert asyncio.run(_fetch_spec_metrics("http://localhost:8000", timeout_s=1.0)) is None


# ---------------------------------------------------------------------------
# Arg validation + orchestration
# ---------------------------------------------------------------------------
def _args(**overrides) -> argparse.Namespace:
    base = dict(
        server="http://localhost:8000",
        model="Qwen/Qwen3-8B",
        input_data="dummy.jsonl",
        baseline_server=None,
        num_prompts=2,
        concurrency=2,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        messages_column="messages",
        split="train",
        dataset_name=None,
        shuffle_seed=None,
        timeout_s=5.0,
        max_retries=0,
        output_json=None,
        log_level="INFO",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_prompts", 0),
        ("concurrency", 0),
        ("max_new_tokens", 0),
        ("max_retries", -1),
        ("timeout_s", 0.0),
    ],
)
def test_validate_args_rejects(field, value):
    with pytest.raises(ValueError, match=field.replace("_", "-")):
        _validate_workload_args(_args(**{field: value}))


def test_run_reports_delta_based_acceptance(monkeypatch, capsys):
    """End-to-end orchestration: before/after snapshots difference into the summary."""
    snapshots = [
        SpecMetrics(num_drafts=100, num_draft_tokens=1500, num_accepted_tokens=900),
        SpecMetrics(num_drafts=150, num_draft_tokens=2250, num_accepted_tokens=1300),
    ]

    async def _fake_fetch(server, *, timeout_s):
        return snapshots.pop(0)

    async def _fake_workload(server, prompts, gen_cfg, **kwargs):
        return WorkloadResult(wall_clock_s=1.0, output_tokens=100, completed=2, failed=0)

    monkeypatch.setattr(bench_vllm, "_fetch_spec_metrics", _fake_fetch)
    monkeypatch.setattr(bench_vllm, "_run_workload", _fake_workload)
    monkeypatch.setattr(bench_vllm, "_load_prompts", lambda args: [[{"role": "user", "content": "hi"}]] * 2)

    rc = asyncio.run(bench_vllm._run(_args()))

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    # Delta: 50 drafts, 750 draft tokens, 400 accepted -> accept_length = 1 + 400/50 = 9.
    assert summary["accept_length"] == 9.0
    assert summary["acceptance_rate"] == pytest.approx(400 / 750)
    assert summary["num_drafts"] == 50


def test_run_without_before_snapshot_uses_cumulative(monkeypatch, capsys):
    """If the first /metrics scrape fails, the after-snapshot is used as-is."""
    snapshots = [None, SpecMetrics(num_drafts=10, num_draft_tokens=150, num_accepted_tokens=50)]

    async def _fake_fetch(server, *, timeout_s):
        return snapshots.pop(0)

    async def _fake_workload(server, prompts, gen_cfg, **kwargs):
        return WorkloadResult(wall_clock_s=1.0, output_tokens=100, completed=2, failed=0)

    monkeypatch.setattr(bench_vllm, "_fetch_spec_metrics", _fake_fetch)
    monkeypatch.setattr(bench_vllm, "_run_workload", _fake_workload)
    monkeypatch.setattr(bench_vllm, "_load_prompts", lambda args: [[{"role": "user", "content": "hi"}]])

    rc = asyncio.run(bench_vllm._run(_args()))

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["accept_length"] == 6.0


def test_run_no_prompts_returns_error(monkeypatch):
    monkeypatch.setattr(bench_vllm, "_load_prompts", lambda args: [])
    rc = asyncio.run(bench_vllm._run(_args()))
    assert rc == 1


def test_run_with_baseline_reports_speedup(monkeypatch, capsys, tmp_path):
    """A baseline server enables speedup; --output-json writes the same payload."""
    calls = []

    async def _fake_fetch(server, *, timeout_s):
        return SpecMetrics(num_drafts=10, num_draft_tokens=150, num_accepted_tokens=50)

    async def _fake_workload(server, prompts, gen_cfg, **kwargs):
        calls.append(server)
        wall = 1.0 if len(calls) == 1 else 2.0
        return WorkloadResult(wall_clock_s=wall, output_tokens=100, completed=1, failed=0)

    monkeypatch.setattr(bench_vllm, "_fetch_spec_metrics", _fake_fetch)
    monkeypatch.setattr(bench_vllm, "_run_workload", _fake_workload)
    monkeypatch.setattr(bench_vllm, "_load_prompts", lambda args: [[{"role": "user", "content": "hi"}]])
    out_path = tmp_path / "metrics.json"

    rc = asyncio.run(bench_vllm._run(_args(baseline_server="http://localhost:8001", output_json=str(out_path))))

    assert rc == 0
    assert calls == ["http://localhost:8000", "http://localhost:8001"]
    summary = json.loads(capsys.readouterr().out)
    assert summary["speedup"] == 2.0
    assert json.loads(out_path.read_text(encoding="utf-8")) == summary


def test_main_parses_and_runs(monkeypatch):
    async def _fake_run(args):
        assert args.server == "http://localhost:8000"
        return 0

    monkeypatch.setattr(bench_vllm, "_run", _fake_run)
    rc = bench_vllm.main(["--server", "http://localhost:8000", "--model", "m", "--input-data", "d.jsonl"])
    assert rc == 0

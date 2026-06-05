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

"""Unit tests for the SGLang EAGLE acceptance/speedup benchmark.

The HTTP path is exercised with a hermetic fake aiohttp session (no network);
the metric parsing mirrors SGLang's own ``bench_serving`` server-info logic and
is tested directly.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from nemo_automodel.components.speculative import bench_sglang
from nemo_automodel.components.speculative.bench_sglang import (
    GenerationConfig,
    WorkloadResult,
    _acceptance_rate,
    _extract_accept_length,
    _extract_num_steps,
    _normalize_server_url,
    _output_throughput,
    _speedup,
    _summarize,
    _unwrap_server_info,
    _validate_args,
)

# ---------------------------------------------------------------------------
# /server_info parsing (mirrors sglang/bench_serving.py)
# ---------------------------------------------------------------------------


def _server_info(accept_length=3.5, num_steps=4, *, wrap_decode=False):
    state = {"avg_spec_accept_length": accept_length, "speculative_num_steps": num_steps}
    info = {"internal_states": [state]}
    return {"decode": [info]} if wrap_decode else info


def test_unwrap_server_info_plain_and_decode_wrapped():
    plain = {"internal_states": [{"avg_spec_accept_length": 2.0}]}
    assert _unwrap_server_info(plain) is plain
    inner = {"internal_states": [{"avg_spec_accept_length": 2.0}]}
    assert _unwrap_server_info({"decode": [inner]}) is inner


def test_unwrap_server_info_handles_bad_shapes():
    assert _unwrap_server_info(None) is None
    assert _unwrap_server_info("nope") is None
    assert _unwrap_server_info({"decode": []}) is None
    assert _unwrap_server_info({"decode": "x"}) is None


def test_extract_accept_length_plain_and_wrapped():
    assert _extract_accept_length(_server_info(3.5)) == 3.5
    assert _extract_accept_length(_server_info(3.5, wrap_decode=True)) == 3.5


def test_extract_accept_length_missing_returns_none():
    assert _extract_accept_length({}) is None
    assert _extract_accept_length({"internal_states": []}) is None
    assert _extract_accept_length({"internal_states": [{}]}) is None
    assert _extract_accept_length({"internal_states": [{"avg_spec_accept_length": "x"}]}) is None
    assert _extract_accept_length(None) is None


def test_extract_num_steps():
    assert _extract_num_steps(_server_info(num_steps=4)) == 4
    assert _extract_num_steps({"internal_states": [{}]}) is None
    assert _extract_num_steps({"internal_states": [{"speculative_num_steps": 0}]}) is None
    assert _extract_num_steps({"internal_states": [{"speculative_num_steps": 2.5}]}) is None


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------


def test_acceptance_rate_formula():
    # (accept_length - 1) / num_steps
    assert _acceptance_rate(3.0, 4) == pytest.approx(0.5)
    assert _acceptance_rate(5.0, 4) == pytest.approx(1.0)


def test_acceptance_rate_clamps_and_handles_missing():
    assert _acceptance_rate(0.5, 4) == 0.0  # accept_length < 1 -> clamp to 0
    assert _acceptance_rate(None, 4) is None
    assert _acceptance_rate(3.0, None) is None
    assert _acceptance_rate(3.0, 0) is None


def test_speedup():
    assert _speedup(200.0, 100.0) == pytest.approx(2.0)
    assert _speedup(None, 100.0) is None
    assert _speedup(200.0, None) is None
    assert _speedup(200.0, 0.0) is None


def test_output_throughput():
    assert _output_throughput(WorkloadResult(wall_clock_s=2.0, output_tokens=100, completed=1, failed=0)) == 50.0
    assert _output_throughput(WorkloadResult(wall_clock_s=0.0, output_tokens=100, completed=1, failed=0)) is None
    assert _output_throughput(WorkloadResult(wall_clock_s=2.0, output_tokens=0, completed=0, failed=1)) is None


def test_normalize_server_url():
    assert _normalize_server_url("http://localhost:30000") == "http://localhost:30000"
    assert _normalize_server_url("http://localhost:30000/") == "http://localhost:30000"
    assert _normalize_server_url("http://localhost:30000/v1") == "http://localhost:30000"
    assert _normalize_server_url("http://localhost:30000/v1/") == "http://localhost:30000"


# ---------------------------------------------------------------------------
# _summarize
# ---------------------------------------------------------------------------


def _gen_cfg():
    return GenerationConfig(model="m", max_new_tokens=8, temperature=0.0, top_p=1.0)


def test_summarize_with_server_info_and_baseline():
    spec = WorkloadResult(wall_clock_s=2.0, output_tokens=200, completed=10, failed=0)
    baseline = WorkloadResult(wall_clock_s=4.0, output_tokens=200, completed=10, failed=0)
    summary = _summarize(
        gen_cfg=_gen_cfg(),
        spec_result=spec,
        server_info=_server_info(3.0, 4),
        num_steps_arg=None,
        baseline_result=baseline,
    )
    assert summary["accept_length"] == 3.0
    assert summary["speculative_num_steps"] == 4
    assert summary["acceptance_rate"] == pytest.approx(0.5)
    assert summary["output_throughput_tok_s"] == pytest.approx(100.0)
    assert summary["baseline_throughput_tok_s"] == pytest.approx(50.0)
    assert summary["speedup"] == pytest.approx(2.0)
    assert summary["num_prompts"] == 10


def test_summarize_without_server_info_uses_num_steps_arg_and_omits_speedup():
    spec = WorkloadResult(wall_clock_s=1.0, output_tokens=50, completed=5, failed=1)
    summary = _summarize(gen_cfg=_gen_cfg(), spec_result=spec, server_info=None, num_steps_arg=4, baseline_result=None)
    assert summary["accept_length"] is None
    assert summary["speculative_num_steps"] == 4  # fell back to the CLI arg
    assert summary["acceptance_rate"] is None  # no accept_length -> no rate
    assert summary["failed"] == 1
    assert "speedup" not in summary
    assert "baseline_throughput_tok_s" not in summary


# ---------------------------------------------------------------------------
# _validate_args
# ---------------------------------------------------------------------------


def _args(**overrides):
    base = dict(num_prompts=8, concurrency=4, max_new_tokens=64, max_retries=3, timeout_s=600.0, num_steps=None)
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides,pattern",
    [
        ({"num_prompts": 0}, "num-prompts"),
        ({"concurrency": 0}, "concurrency"),
        ({"max_new_tokens": 0}, "max-new-tokens"),
        ({"max_retries": -1}, "max-retries"),
        ({"timeout_s": 0}, "timeout-s"),
        ({"num_steps": 0}, "num-steps"),
    ],
)
def test_validate_args_rejects_invalid(overrides, pattern):
    with pytest.raises(ValueError, match=pattern):
        _validate_args(_args(**overrides))


def test_validate_args_accepts_valid():
    _validate_args(_args())  # no raise


# ---------------------------------------------------------------------------
# HTTP path with a hermetic fake aiohttp session
# ---------------------------------------------------------------------------


class _FakeClientResponseError(Exception):
    """Stand-in for ``aiohttp.ClientResponseError`` (carries an HTTP ``status``)."""

    def __init__(self, status):
        self.status = status
        super().__init__(f"HTTP {status}")


class _FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status = status
        self._payload = payload
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    def raise_for_status(self):
        if self.status >= 400:
            raise _FakeClientResponseError(self.status)


class _FakeSession:
    """Async-context-manager session whose post/get pop from queued responses."""

    def __init__(self, post_responses=None, get_response=None):
        self._post_responses = list(post_responses or [])
        self._get_response = get_response
        self.post_calls: list[tuple[str, dict]] = []
        self.get_calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, json=None, timeout=None):  # noqa: A002 -- match aiohttp signature
        self.post_calls.append((url, json))
        return self._post_responses.pop(0)

    def get(self, url, *, timeout=None):
        self.get_calls.append(url)
        return self._get_response


def _fake_aiohttp(session):
    return SimpleNamespace(
        ClientSession=lambda *a, **k: session,
        ClientTimeout=lambda total=None: None,
        ClientResponseError=_FakeClientResponseError,
    )


def _patch_aiohttp(monkeypatch, session):
    monkeypatch.setattr(bench_sglang, "_import_aiohttp", lambda: _fake_aiohttp(session))

    # No real backoff sleeps in retry tests.
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(bench_sglang.asyncio, "sleep", _no_sleep)


def test_chat_completion_returns_completion_tokens(monkeypatch):
    session = _FakeSession(post_responses=[_FakeResponse(200, {"usage": {"completion_tokens": 17}})])
    _patch_aiohttp(monkeypatch, session)
    tokens = asyncio.run(
        bench_sglang._chat_completion(session, "http://x/v1/chat/completions", {}, timeout_s=1.0, max_retries=0)
    )
    assert tokens == 17


def test_chat_completion_missing_usage_returns_zero(monkeypatch):
    session = _FakeSession(post_responses=[_FakeResponse(200, {"choices": []})])
    _patch_aiohttp(monkeypatch, session)
    tokens = asyncio.run(bench_sglang._chat_completion(session, "http://x", {}, timeout_s=1.0, max_retries=0))
    assert tokens == 0


def test_chat_completion_retries_then_succeeds(monkeypatch):
    session = _FakeSession(
        post_responses=[
            _FakeResponse(503, text="overloaded"),
            _FakeResponse(200, {"usage": {"completion_tokens": 5}}),
        ]
    )
    _patch_aiohttp(monkeypatch, session)
    tokens = asyncio.run(bench_sglang._chat_completion(session, "http://x", {}, timeout_s=1.0, max_retries=2))
    assert tokens == 5
    assert len(session.post_calls) == 2


def test_chat_completion_raises_after_max_retries(monkeypatch):
    session = _FakeSession(post_responses=[_FakeResponse(503, text="down"), _FakeResponse(503, text="down")])
    _patch_aiohttp(monkeypatch, session)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        asyncio.run(bench_sglang._chat_completion(session, "http://x", {}, timeout_s=1.0, max_retries=1))


def test_chat_completion_does_not_retry_non_retryable_4xx(monkeypatch):
    # A non-429 4xx (here 404) is a client error that will not succeed on retry.
    # It must surface on the first attempt instead of burning the retry budget;
    # queue extra 404s so a regression that retried would show >1 post call.
    session = _FakeSession(post_responses=[_FakeResponse(404, text="not found")] * 4)
    _patch_aiohttp(monkeypatch, session)
    with pytest.raises(_FakeClientResponseError):
        asyncio.run(bench_sglang._chat_completion(session, "http://x", {}, timeout_s=1.0, max_retries=3))
    assert len(session.post_calls) == 1


def test_run_workload_sums_tokens_and_counts_failures(monkeypatch):
    # 3 prompts: two succeed (10 + 20 tokens), one server-errors past retries.
    session = _FakeSession(
        post_responses=[
            _FakeResponse(200, {"usage": {"completion_tokens": 10}}),
            _FakeResponse(200, {"usage": {"completion_tokens": 20}}),
            _FakeResponse(500, text="boom"),
        ]
    )
    _patch_aiohttp(monkeypatch, session)
    prompts = [[{"role": "user", "content": "a"}]] * 3
    result = asyncio.run(
        bench_sglang._run_workload(
            "http://localhost:30000",
            prompts,
            _gen_cfg(),
            concurrency=1,  # deterministic response ordering
            timeout_s=1.0,
            max_retries=0,
        )
    )
    assert result.output_tokens == 30
    assert result.completed == 2
    assert result.failed == 1
    assert session.post_calls[0][0] == "http://localhost:30000/v1/chat/completions"


def test_fetch_server_info_ok(monkeypatch):
    session = _FakeSession(get_response=_FakeResponse(200, _server_info(3.5, 4)))
    _patch_aiohttp(monkeypatch, session)
    info = asyncio.run(bench_sglang._fetch_server_info("http://localhost:30000/v1", timeout_s=1.0))
    assert _extract_accept_length(info) == 3.5
    assert session.get_calls == ["http://localhost:30000/server_info"]


def test_fetch_server_info_non_200_returns_none(monkeypatch):
    session = _FakeSession(get_response=_FakeResponse(404, text="nope"))
    _patch_aiohttp(monkeypatch, session)
    assert asyncio.run(bench_sglang._fetch_server_info("http://localhost:30000", timeout_s=1.0)) is None


def test_fetch_server_info_transport_error_returns_none(monkeypatch):
    # A connection failure is best-effort: log and return None, never raise.
    class _BoomSession(_FakeSession):
        def get(self, url, *, timeout=None):
            raise RuntimeError("connection refused")

    _patch_aiohttp(monkeypatch, _BoomSession())
    assert asyncio.run(bench_sglang._fetch_server_info("http://x", timeout_s=1.0)) is None


# ---------------------------------------------------------------------------
# Prompt loading, the async driver (_run), and the CLI (main)
# ---------------------------------------------------------------------------


def _run_args(**over):
    base = dict(
        server="http://localhost:30000",
        baseline_server=None,
        model="m",
        input_data="data",
        split="train",
        dataset_name=None,
        shuffle_seed=None,
        messages_column="messages",
        num_prompts=4,
        concurrency=2,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        num_steps=None,
        timeout_s=1.0,
        max_retries=0,
        output_json=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_load_prompts_caps_and_drops_unusable(monkeypatch):
    rows = [{"messages": "a"}, {"messages": "b"}, {"messages": "c"}]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages",
        lambda *a, **k: rows,
    )
    # "b" yields no usable prompt and must be skipped; the cap stops at 2.
    monkeypatch.setattr(
        bench_sglang,
        "_extract_prompt_messages",
        lambda m: None if m == "b" else [{"role": "user", "content": m}],
    )
    prompts = bench_sglang._load_prompts(_run_args(num_prompts=2))
    assert prompts == [[{"role": "user", "content": "a"}], [{"role": "user", "content": "c"}]]


def test_run_happy_path_prints_and_writes_json(monkeypatch, tmp_path):
    monkeypatch.setattr(bench_sglang, "_load_prompts", lambda args: [[{"role": "user", "content": "hi"}]])

    async def _fake_workload(server, prompts, gen_cfg, **kwargs):
        return WorkloadResult(wall_clock_s=2.0, output_tokens=100, completed=1, failed=0)

    async def _fake_info(server, *, timeout_s):
        return _server_info(3.5, 4)

    monkeypatch.setattr(bench_sglang, "_run_workload", _fake_workload)
    monkeypatch.setattr(bench_sglang, "_fetch_server_info", _fake_info)

    out = tmp_path / "metrics.json"
    rc = asyncio.run(bench_sglang._run(_run_args(output_json=str(out))))
    assert rc == 0
    assert json.loads(out.read_text())["accept_length"] == 3.5


def test_run_returns_1_when_no_prompts(monkeypatch):
    monkeypatch.setattr(bench_sglang, "_load_prompts", lambda args: [])
    assert asyncio.run(bench_sglang._run(_run_args())) == 1


def test_run_with_baseline_and_missing_accept_length(monkeypatch):
    monkeypatch.setattr(bench_sglang, "_load_prompts", lambda args: [[{"role": "user", "content": "x"}]])
    servers: list[str] = []

    async def _fake_workload(server, prompts, gen_cfg, **kwargs):
        servers.append(server)
        return WorkloadResult(wall_clock_s=1.0, output_tokens=10, completed=1, failed=0)

    async def _fake_info(server, *, timeout_s):
        return None  # server_info unavailable -> accept_length stays None (warning path)

    monkeypatch.setattr(bench_sglang, "_run_workload", _fake_workload)
    monkeypatch.setattr(bench_sglang, "_fetch_server_info", _fake_info)

    rc = asyncio.run(bench_sglang._run(_run_args(baseline_server="http://localhost:30001")))
    assert rc == 0
    # Both the spec server and the baseline server were benchmarked.
    assert servers == ["http://localhost:30000", "http://localhost:30001"]


def test_main_builds_parser_and_dispatches(monkeypatch):
    seen = {}

    async def _fake_run(args):
        seen["server"] = args.server
        seen["model"] = args.model
        return 0

    monkeypatch.setattr(bench_sglang, "_run", _fake_run)
    rc = bench_sglang.main(["--server", "http://localhost:30000", "--model", "m", "--input-data", "d"])
    assert rc == 0
    assert seen == {"server": "http://localhost:30000", "model": "m"}

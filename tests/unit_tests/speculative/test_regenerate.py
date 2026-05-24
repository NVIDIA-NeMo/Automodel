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

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_automodel.components.speculative.regenerate import (
    GenerationConfig,
    _build_manifest,
    _chat_completion,
    _ensure_manifest_compatible,
    _existing_shard_indices,
    _extract_prompt_messages,
    _process_shard,
    _validate_args,
    _write_shard,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


class _FakeResponse:
    """Stand-in for an aiohttp response so the orchestration tests stay hermetic."""

    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        import json as _json

        return _json.dumps(self._payload)

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"status {self.status}")


class _FakeSession:
    """Captures every POST so tests can assert against the SGLang request shape."""

    def __init__(self, reply_for):
        self._reply_for = reply_for
        self.calls: list[dict] = []

    def post(self, url, *, json, timeout=None):  # noqa: A002 -- match aiohttp signature
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        reply = self._reply_for(json)
        return _FakeResponse({"choices": [{"message": {"role": "assistant", "content": reply}}]})


def test_extract_prompt_drops_trailing_assistant_turn():
    sample = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "stale answer"},
    ]
    prompt = _extract_prompt_messages(sample)
    assert prompt == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    # The original sample must not be mutated -- we return defensive copies.
    assert sample[-1]["role"] == "assistant"


def test_extract_prompt_keeps_intermediate_assistant_turns():
    sample = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2_stale"},
    ]
    prompt = _extract_prompt_messages(sample)
    assert prompt == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


def test_extract_prompt_strips_consecutive_trailing_assistants():
    sample = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "a2"},
    ]
    prompt = _extract_prompt_messages(sample)
    assert prompt == [{"role": "user", "content": "q"}]


def test_extract_prompt_rejects_samples_with_no_user_turn():
    # System-only or all-assistant samples must be skipped, not silently regenerated
    # into vacuous prompts.
    assert _extract_prompt_messages([]) is None
    assert _extract_prompt_messages([{"role": "system", "content": "sys"}]) is None
    assert _extract_prompt_messages([{"role": "assistant", "content": "a"}]) is None


def test_existing_shard_indices_ignores_non_shard_files(tmp_path: Path):
    (tmp_path / "shard-000000.parquet").write_bytes(b"")
    (tmp_path / "shard-000007.parquet").write_bytes(b"")
    # These must NOT count: in-progress tmp, mis-numbered, unrelated files,
    # and the manifest file that lives alongside the shards.
    (tmp_path / "shard-000003.parquet.tmp").write_bytes(b"")
    (tmp_path / "shard-1.parquet").write_bytes(b"")
    (tmp_path / "README.md").write_bytes(b"")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    assert _existing_shard_indices(tmp_path) == {0, 7}


def test_existing_shard_indices_missing_dir_returns_empty(tmp_path: Path):
    assert _existing_shard_indices(tmp_path / "does-not-exist") == set()


def test_resume_manifest_mismatch_raises(tmp_path: Path):
    args = SimpleNamespace(
        input_data="dataset-a",
        output_dir=str(tmp_path),
        target_server="http://localhost:30000/v1",
        model="model-a",
        messages_column="messages",
        split="train",
        dataset_name=None,
        shuffle_seed=None,
        shard_size=1000,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
    )
    manifest = _build_manifest(args)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    args.model = "model-b"
    with pytest.raises(ValueError, match="does not match"):
        _ensure_manifest_compatible(
            tmp_path,
            _build_manifest(args),
            resume=True,
            existing_shards={0},
        )


def test_resume_without_manifest_and_existing_shards_raises(tmp_path: Path):
    (tmp_path / "shard-000000.parquet").write_bytes(b"")
    manifest = {
        "input_data": "dataset-a",
        "target_server": "http://localhost:30000/v1",
        "model": "model-a",
        "messages_column": "messages",
        "split": "train",
        "dataset_name": None,
        "shuffle_seed": None,
        "shard_size": 1000,
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    with pytest.raises(ValueError, match="manifest.json is missing"):
        _ensure_manifest_compatible(tmp_path, manifest, resume=True, existing_shards={0})


def test_fresh_run_with_existing_shards_refuses_to_clobber(tmp_path: Path):
    """Running without --resume against a non-empty output dir must fail loudly."""
    (tmp_path / "shard-000000.parquet").write_bytes(b"")
    args = SimpleNamespace(
        input_data="dataset-a",
        output_dir=str(tmp_path),
        target_server="http://localhost:30000/v1",
        model="model-a",
        messages_column="messages",
        split="train",
        dataset_name=None,
        shuffle_seed=None,
        shard_size=1000,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
    )
    with pytest.raises(ValueError, match="already contains"):
        _ensure_manifest_compatible(
            tmp_path,
            _build_manifest(args),
            resume=False,
            existing_shards={0},
        )


def test_fresh_run_into_empty_dir_writes_manifest(tmp_path: Path):
    """The first invocation must persist a manifest so a later --resume can verify it."""
    args = SimpleNamespace(
        input_data="dataset-a",
        output_dir=str(tmp_path),
        target_server="http://localhost:30000/v1",
        model="model-a",
        messages_column="messages",
        split="train",
        dataset_name=None,
        shuffle_seed=None,
        shard_size=1000,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
    )
    manifest = _build_manifest(args)
    _ensure_manifest_compatible(tmp_path, manifest, resume=False, existing_shards=set())
    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert written == manifest


def test_build_manifest_excludes_self_referential_and_operational_fields():
    """The manifest must not encode output_dir (it lives inside output_dir) or knobs
    that only affect throughput (concurrency / timeout_s / max_retries)."""
    args = SimpleNamespace(
        input_data="dataset-a",
        output_dir="/some/path",
        target_server="http://localhost:30000/v1",
        model="model-a",
        messages_column="messages",
        split="train",
        dataset_name=None,
        shuffle_seed=None,
        shard_size=1000,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        # Operational knobs that intentionally do NOT belong in the manifest.
        concurrency=99,
        timeout_s=123.0,
        max_retries=42,
    )
    manifest = _build_manifest(args)
    assert "output_dir" not in manifest
    for operational in ("concurrency", "timeout_s", "max_retries"):
        assert operational not in manifest


def test_write_shard_is_atomic_and_roundtrips(tmp_path: Path):
    rows = [
        {
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "original_messages": [{"role": "user", "content": "q"}],
        },
    ]
    path = _write_shard(tmp_path, shard_index=4, rows=rows)
    assert path == tmp_path / "shard-000004.parquet"
    assert path.exists()
    # No stale .tmp file should remain after a successful write.
    assert not (tmp_path / "shard-000004.parquet.tmp").exists()

    table = pq.read_table(path)
    out = table.to_pylist()
    assert out == rows


def test_process_shard_sends_prompts_and_collects_assistant_replies():
    prompts = [
        (
            0,
            [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "stale0"}],
            [{"role": "user", "content": "q0"}],
        ),
        (
            1,
            [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "stale1"}],
            [{"role": "user", "content": "q1"}],
        ),
    ]

    def _reply_for(payload: dict) -> str:
        # Echo the user content back so the assertion can pin request<->response.
        return "reply_to_" + payload["messages"][-1]["content"]

    session = _FakeSession(_reply_for)
    gen_cfg = GenerationConfig(model="target-model", max_new_tokens=8, temperature=0.0, top_p=1.0)

    rows = asyncio.run(
        _process_shard(
            session,
            url="http://server/v1/chat/completions",
            shard_samples=prompts,
            gen_cfg=gen_cfg,
            concurrency=4,
            timeout_s=10.0,
            max_retries=0,
        )
    )

    assert len(rows) == 2
    assert rows[0]["messages"][-1] == {"role": "assistant", "content": "reply_to_q0"}
    assert rows[1]["messages"][-1] == {"role": "assistant", "content": "reply_to_q1"}
    # Original is preserved verbatim for traceability.
    assert rows[0]["original_messages"][-1]["content"] == "stale0"
    # The server never sees the stale assistant turn.
    for call in session.calls:
        assert call["json"]["model"] == "target-model"
        assert all(m["role"] != "assistant" for m in call["json"]["messages"])


def test_process_shard_concurrency_is_bounded():
    """Outstanding requests must never exceed --concurrency, even with many prompts."""

    prompts = [(i, [{"role": "user", "content": f"q{i}"}], [{"role": "user", "content": f"q{i}"}]) for i in range(20)]

    in_flight = 0
    peak = 0

    class _CountingSession:
        def post(self, url, *, json, timeout=None):  # noqa: A002
            nonlocal in_flight, peak

            class _Resp:
                async def __aenter__(self_inner):
                    nonlocal in_flight, peak
                    in_flight += 1
                    peak = max(peak, in_flight)
                    await asyncio.sleep(0)  # let scheduler interleave
                    return self_inner

                async def __aexit__(self_inner, exc_type, exc, tb):
                    nonlocal in_flight
                    in_flight -= 1
                    return False

                async def json(self_inner):
                    return {"choices": [{"message": {"content": "ok"}}]}

                async def text(self_inner):
                    return "ok"

                def raise_for_status(self_inner):
                    return None

                status = 200

            return _Resp()

    gen_cfg = GenerationConfig(model="m", max_new_tokens=4, temperature=0.0, top_p=1.0)
    asyncio.run(
        _process_shard(
            _CountingSession(),
            url="http://server/v1/chat/completions",
            shard_samples=prompts,
            gen_cfg=gen_cfg,
            concurrency=3,
            timeout_s=10.0,
            max_retries=0,
        )
    )
    assert peak <= 3, f"peak in-flight {peak} exceeded concurrency=3"


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("concurrency", 0, "concurrency"),
        ("shard_size", 0, "shard-size"),
        ("max_new_tokens", 0, "max-new-tokens"),
        ("max_retries", -1, "max-retries"),
        ("timeout_s", 0.0, "timeout-s"),
    ],
)
def test_validate_args_rejects_invalid_values(field: str, value: int | float, pattern: str):
    args = SimpleNamespace(
        concurrency=4,
        shard_size=1000,
        max_new_tokens=128,
        max_retries=3,
        timeout_s=60.0,
    )
    setattr(args, field, value)
    with pytest.raises(ValueError, match=pattern):
        _validate_args(args)


def _run_args(tmp_path: Path, *, resume: bool, shard_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        input_data="dataset-stub",
        output_dir=str(tmp_path),
        target_server="http://stub:0/v1",
        model="target-model",
        messages_column="messages",
        split="train",
        dataset_name=None,
        shuffle_seed=None,
        shard_size=shard_size,
        concurrency=2,
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        timeout_s=5.0,
        max_retries=0,
        resume=resume,
        log_level="INFO",
    )


def test_run_resume_skips_already_written_shards(tmp_path: Path, monkeypatch):
    """End-to-end: first pass writes shard 0; second pass with --resume must skip
    shard 0 and only request prompts for shard 1, while leaving shard 0 intact."""
    from nemo_automodel.components.speculative import regenerate as regen

    # 4 samples + shard_size=2  =>  exactly two shards (indices 0 and 1).
    fake_dataset = [
        {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"stale{i}"}]}
        for i in range(4)
    ]
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.chat_dataset._load_openai_messages",
        lambda *args, **kwargs: fake_dataset,
    )

    # Per-pass record of the prompts the server received.
    pass1_prompts: list[list[dict]] = []
    pass2_prompts: list[list[dict]] = []

    def _make_session_cls(this_pass_prompts: list[list[dict]]):
        def _reply_for(payload: dict) -> str:
            this_pass_prompts.append([dict(m) for m in payload["messages"]])
            return "regen_for_" + payload["messages"][-1]["content"]

        class _SessionCM:
            def __init__(self_inner, *args, **kwargs):
                self_inner._session = _FakeSession(_reply_for)

            async def __aenter__(self_inner):
                return self_inner._session

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        return _SessionCM

    fake_aiohttp_module = SimpleNamespace(
        ClientSession=_make_session_cls(pass1_prompts),
        ClientTimeout=lambda total=None: ("timeout", total),
    )
    monkeypatch.setattr(regen, "_import_aiohttp", lambda: fake_aiohttp_module)

    # Pass 1: fresh run, both shards get written.
    args1 = _run_args(tmp_path, resume=False, shard_size=2)
    rc = asyncio.run(regen._run(args1))
    assert rc == 0
    assert (tmp_path / "shard-000000.parquet").exists()
    assert (tmp_path / "shard-000001.parquet").exists()
    assert len(pass1_prompts) == 4  # server saw all 4 prompts
    shard0_mtime_before = (tmp_path / "shard-000000.parquet").stat().st_mtime_ns
    shard0_bytes_before = (tmp_path / "shard-000000.parquet").read_bytes()

    # Manually delete shard 1 to simulate a partial first run (crash before
    # the second shard finished). The manifest from pass 1 stays in place.
    (tmp_path / "shard-000001.parquet").unlink()

    # Pass 2: --resume. Only the missing shard should be regenerated.
    fake_aiohttp_module.ClientSession = _make_session_cls(pass2_prompts)
    args2 = _run_args(tmp_path, resume=True, shard_size=2)
    rc = asyncio.run(regen._run(args2))
    assert rc == 0
    assert (tmp_path / "shard-000001.parquet").exists()
    # Server only saw the q2 / q3 prompts on this pass, NOT q0 / q1.
    seen_user_msgs = [msgs[-1]["content"] for msgs in pass2_prompts]
    assert sorted(seen_user_msgs) == ["q2", "q3"], seen_user_msgs
    # Shard 0 must be byte-identical to the first pass.
    assert (tmp_path / "shard-000000.parquet").read_bytes() == shard0_bytes_before
    assert (tmp_path / "shard-000000.parquet").stat().st_mtime_ns == shard0_mtime_before


class _SequentialSession:
    """Returns a pre-defined sequence of _FakeResponse objects, one per POST call."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = iter(responses)
        self.call_count = 0

    def post(self, url, *, json, timeout=None):  # noqa: A002
        self.call_count += 1
        return next(self._responses)


def test_chat_completion_retries_on_5xx_then_succeeds(monkeypatch):
    """_chat_completion must retry on 5xx and return the content on eventual success."""
    import nemo_automodel.components.speculative.regenerate as regen

    sleep_calls: list[float] = []

    async def fast_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(regen.asyncio, "sleep", fast_sleep)

    success_payload = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    session = _SequentialSession(
        [
            _FakeResponse({}, status=500),  # attempt 0 → retry
            _FakeResponse({}, status=429),  # attempt 1 → retry
            _FakeResponse(success_payload, status=200),  # attempt 2 → success
        ]
    )

    result = asyncio.run(_chat_completion(session, "http://stub/completions", {}, timeout_s=1.0, max_retries=3))

    assert result == "ok"
    assert session.call_count == 3
    assert len(sleep_calls) == 2  # slept once after each failed attempt
    assert sleep_calls[0] == 1.0  # 2**0 = 1
    assert sleep_calls[1] == 2.0  # 2**1 = 2


def test_chat_completion_raises_after_max_retries_exhausted(monkeypatch):
    """_chat_completion must raise after max_retries+1 total attempts without success."""
    import nemo_automodel.components.speculative.regenerate as regen

    sleep_calls: list[float] = []

    async def fast_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(regen.asyncio, "sleep", fast_sleep)

    # Always 500 — never succeeds.
    session = _SequentialSession([_FakeResponse({}, status=500)] * 10)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        asyncio.run(_chat_completion(session, "http://stub/completions", {}, timeout_s=1.0, max_retries=2))

    # max_retries=2 → 3 total attempts, 2 sleeps
    assert session.call_count == 3
    assert len(sleep_calls) == 2

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

"""Unit tests for the multi-dataset acceptance-length sweep.

No network / HTTP is exercised here: ``_run_sweep`` is tested by monkeypatching
the engine modules' ``_run_summary`` directly (the same seam ``bench_sweep``
itself calls through), mirroring how ``test_bench_sglang.py`` /
``test_bench_vllm.py`` monkeypatch ``_load_prompts`` / ``_run_workload`` rather
than a live server.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest

from nemo_automodel.components.speculative import bench_sglang, bench_sweep, bench_vllm
from nemo_automodel.components.speculative.bench_sweep import (
    DatasetSpec,
    _aggregate,
    _dataset_args,
    _load_dataset_specs,
    _print_table,
    _run,
    _run_sweep,
    _select_datasets,
    main,
)

# ---------------------------------------------------------------------------
# DatasetSpec
# ---------------------------------------------------------------------------


def test_dataset_spec_requires_exactly_one_column():
    DatasetSpec(name="a", input_data="d", messages_column="messages")  # ok
    DatasetSpec(name="a", input_data="d", prompt_column="text")  # ok
    with pytest.raises(ValueError, match="exactly one"):
        DatasetSpec(name="a", input_data="d")  # neither
    with pytest.raises(ValueError, match="exactly one"):
        DatasetSpec(name="a", input_data="d", messages_column="messages", prompt_column="text")  # both


def test_default_dataset_presets_are_the_eagle_paper_suite():
    names = [s.name for s in bench_sweep.DEFAULT_DATASET_PRESETS]
    assert names == ["mt_bench", "humaneval", "gsm8k", "alpaca"]
    for spec in bench_sweep.DEFAULT_DATASET_PRESETS:
        assert spec.prompt_column is not None
        assert spec.messages_column is None


# ---------------------------------------------------------------------------
# _load_dataset_specs / --datasets-config
# ---------------------------------------------------------------------------


def test_load_dataset_specs_none_returns_defaults():
    assert _load_dataset_specs(None) == list(bench_sweep.DEFAULT_DATASET_PRESETS)


def test_load_dataset_specs_reads_datasets_key(tmp_path):
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(
        "datasets:\n"
        "  - name: custom\n"
        "    input_data: some/dataset\n"
        "    split: test\n"
        "    prompt_column: text\n"
        "    max_new_tokens: 128\n"
    )
    specs = _load_dataset_specs(str(cfg))
    assert specs == [
        DatasetSpec(name="custom", input_data="some/dataset", split="test", prompt_column="text", max_new_tokens=128)
    ]


@pytest.mark.parametrize(
    "content,match",
    [
        # A bare top-level list fails the example-YAML linter; must nest under `datasets:`.
        ("- name: custom\n  input_data: d\n  prompt_column: text\n", "datasets:"),
        ("datasets: []\n", "non-empty"),
        (
            "datasets:\n"
            "  - {name: dup, input_data: a, prompt_column: t}\n"
            "  - {name: dup, input_data: b, prompt_column: t}\n",
            "duplicate dataset name",
        ),
        # An unknown key: DatasetSpec(**entry) would otherwise raise a raw TypeError.
        ("datasets:\n  - {name: a, input_data: d, prompt_col: t}\n", "invalid dataset entry"),
        # A non-mapping entry (bare string instead of a dict).
        ("datasets:\n  - just_a_string\n", "must be a mapping"),
    ],
    ids=["bare-list", "empty", "duplicate-names", "unknown-key", "non-mapping-entry"],
)
def test_load_dataset_specs_rejects_invalid(tmp_path, content, match):
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(content)
    with pytest.raises(ValueError, match=match):
        _load_dataset_specs(str(cfg))


def test_shipped_example_config_loads(monkeypatch):
    """The example YAML under examples/ must itself be a valid --datasets-config."""
    import pathlib

    repo_root = pathlib.Path(bench_sweep.__file__).resolve().parents[3]
    example = repo_root / "examples" / "speculative" / "bench_sweep" / "spec_bench_datasets.yaml"
    specs = _load_dataset_specs(str(example))
    assert [s.name for s in specs] == ["mt_bench", "humaneval", "gsm8k", "alpaca"]


# ---------------------------------------------------------------------------
# _select_datasets
# ---------------------------------------------------------------------------

_SPECS = [
    DatasetSpec(name="a", input_data="d", prompt_column="t"),
    DatasetSpec(name="b", input_data="d", prompt_column="t"),
    DatasetSpec(name="c", input_data="d", prompt_column="t"),
]


def test_select_datasets_none_returns_all():
    assert _select_datasets(_SPECS, None) == _SPECS


def test_select_datasets_subset_preserves_order():
    assert [s.name for s in _select_datasets(_SPECS, ["c", "a"])] == ["a", "c"]


def test_select_datasets_rejects_unknown_name():
    with pytest.raises(ValueError, match="not in the sweep list"):
        _select_datasets(_SPECS, ["a", "zzz"])


# ---------------------------------------------------------------------------
# _dataset_args
# ---------------------------------------------------------------------------


def test_dataset_args_overrides_only_dataset_fields():
    base = argparse.Namespace(
        server="http://x",
        model="m",
        input_data="base",
        split="train",
        dataset_name=None,
        messages_column="messages",
        prompt_column=None,
        max_new_tokens=256,
        num_prompts=64,
    )
    spec = DatasetSpec(name="gsm8k", input_data="openai/gsm8k", dataset_name="main", split="test", prompt_column="q")
    args = _dataset_args(base, spec)
    assert args.input_data == "openai/gsm8k"
    assert args.split == "test"
    assert args.dataset_name == "main"
    assert args.prompt_column == "q"
    assert args.messages_column is None
    # Unrelated fields pass through unchanged.
    assert args.server == "http://x"
    assert args.num_prompts == 64
    assert args.max_new_tokens == 256  # spec has no override -> base value kept


def test_dataset_args_applies_per_dataset_max_new_tokens_override():
    base = argparse.Namespace(
        max_new_tokens=256,
        input_data="x",
        split="train",
        dataset_name=None,
        messages_column="messages",
        prompt_column=None,
    )
    spec = DatasetSpec(name="humaneval", input_data="d", prompt_column="p", max_new_tokens=512)
    assert _dataset_args(base, spec).max_new_tokens == 512


def test_dataset_args_does_not_mutate_base():
    base = argparse.Namespace(
        input_data="base",
        split="train",
        dataset_name=None,
        messages_column="messages",
        prompt_column=None,
        max_new_tokens=256,
    )
    _dataset_args(base, DatasetSpec(name="a", input_data="other", prompt_column="p"))
    assert base.input_data == "base"


# ---------------------------------------------------------------------------
# _run_sweep: per-dataset isolation
# ---------------------------------------------------------------------------


def _sweep_args(**overrides):
    base = dict(
        engine="sglang",
        server="http://localhost:30000",
        model="m",
        baseline_server=None,
        num_prompts=4,
        concurrency=2,
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        num_steps=None,
        shuffle_seed=None,
        timeout_s=1.0,
        max_retries=0,
        output_json=None,
        datasets_config=None,
        datasets=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_sweep_collects_one_summary_per_dataset(monkeypatch):
    specs = [
        DatasetSpec(name="a", input_data="d", prompt_column="t"),
        DatasetSpec(name="b", input_data="d", prompt_column="t"),
    ]

    async def _fake_summary(args):
        return {
            "accept_length": 3.0,
            "acceptance_rate": 0.5,
            "output_throughput_tok_s": 100.0,
            "completed": 4,
            "failed": 0,
        }

    monkeypatch.setattr(bench_sglang, "_run_summary", _fake_summary)
    results = asyncio.run(_run_sweep(_sweep_args(), specs))
    assert [r["dataset"] for r in results] == ["a", "b"]
    assert all("error" not in r for r in results)
    assert results[0]["accept_length"] == 3.0


def test_run_sweep_one_dataset_raising_does_not_abort_others(monkeypatch):
    specs = [
        DatasetSpec(name="bad", input_data="d", prompt_column="t"),
        DatasetSpec(name="good", input_data="d", prompt_column="t"),
    ]
    call_count = {"n": 0}

    async def _flaky_summary(args):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("dataset not found")
        return {"accept_length": 2.0, "completed": 1, "failed": 0}

    monkeypatch.setattr(bench_sglang, "_run_summary", _flaky_summary)
    results = asyncio.run(_run_sweep(_sweep_args(), specs))
    assert results[0] == {"dataset": "bad", "error": "dataset not found"}
    assert results[1]["dataset"] == "good"
    assert "error" not in results[1]


def test_run_sweep_no_prompts_recorded_as_error(monkeypatch):
    specs = [DatasetSpec(name="empty", input_data="d", prompt_column="t")]

    async def _fake_summary(args):
        return None

    monkeypatch.setattr(bench_sglang, "_run_summary", _fake_summary)
    results = asyncio.run(_run_sweep(_sweep_args(), specs))
    assert results == [{"dataset": "empty", "error": "no usable prompts loaded"}]


def test_run_sweep_dispatches_to_selected_engine(monkeypatch):
    specs = [DatasetSpec(name="a", input_data="d", prompt_column="t")]
    calls = []

    async def _sglang_summary(args):
        calls.append("sglang")
        return {"accept_length": 1.0, "completed": 1, "failed": 0}

    async def _vllm_summary(args):
        calls.append("vllm")
        return {"accept_length": 2.0, "completed": 1, "failed": 0}

    monkeypatch.setattr(bench_sglang, "_run_summary", _sglang_summary)
    monkeypatch.setattr(bench_vllm, "_run_summary", _vllm_summary)

    asyncio.run(_run_sweep(_sweep_args(engine="vllm"), specs))
    assert calls == ["vllm"]


def test_run_sweep_warns_once_for_sglang_multi_dataset(monkeypatch, caplog):
    """SGLang's accept_length is server-cumulative; sweeping >1 dataset against one
    server contaminates every dataset after the first, so this must be flagged."""
    specs = [
        DatasetSpec(name="a", input_data="d", prompt_column="t"),
        DatasetSpec(name="b", input_data="d", prompt_column="t"),
    ]

    async def _fake_summary(args):
        return {"accept_length": 1.0, "completed": 1, "failed": 0}

    monkeypatch.setattr(bench_sglang, "_run_summary", _fake_summary)
    with caplog.at_level("WARNING"):
        asyncio.run(_run_sweep(_sweep_args(engine="sglang"), specs))
    assert any("cumulative" in r.message for r in caplog.records)


def test_run_sweep_no_warning_for_vllm_multi_dataset(monkeypatch, caplog):
    """vLLM diffs its counters per dataset, so the sglang-only caveat does not apply."""
    specs = [
        DatasetSpec(name="a", input_data="d", prompt_column="t"),
        DatasetSpec(name="b", input_data="d", prompt_column="t"),
    ]

    async def _fake_summary(args):
        return {"accept_length": 1.0, "completed": 1, "failed": 0}

    monkeypatch.setattr(bench_vllm, "_run_summary", _fake_summary)
    with caplog.at_level("WARNING"):
        asyncio.run(_run_sweep(_sweep_args(engine="vllm"), specs))
    assert not any("cumulative" in r.message for r in caplog.records)


def test_run_sweep_no_warning_for_single_dataset(monkeypatch, caplog):
    specs = [DatasetSpec(name="a", input_data="d", prompt_column="t")]

    async def _fake_summary(args):
        return {"accept_length": 1.0, "completed": 1, "failed": 0}

    monkeypatch.setattr(bench_sglang, "_run_summary", _fake_summary)
    with caplog.at_level("WARNING"):
        asyncio.run(_run_sweep(_sweep_args(engine="sglang"), specs))
    assert not any("cumulative" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# _run: shared-arg validation fails fast, once, instead of being caught per dataset
# ---------------------------------------------------------------------------


def test_run_raises_immediately_on_bad_shared_arg_instead_of_per_dataset_errors(monkeypatch):
    """A bad shared flag (e.g. num_prompts=0) must surface once, not as N duplicate
    per-dataset error rows -- it is a single misconfiguration, not N failures."""
    calls = []

    async def _should_not_be_called(args):
        calls.append(args)
        return {"accept_length": 1.0, "completed": 1, "failed": 0}

    monkeypatch.setattr(bench_sglang, "_run_summary", _should_not_be_called)
    with pytest.raises(ValueError, match="num-prompts"):
        asyncio.run(_run(_sweep_args(num_prompts=0)))
    assert calls == []  # the loop never ran


def test_run_validates_with_the_selected_engines_validator():
    """sglang's validator additionally checks num_steps; vllm's does not."""
    with pytest.raises(ValueError, match="num-steps"):
        asyncio.run(_run(_sweep_args(engine="sglang", num_steps=0)))


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------


def test_aggregate_completed_weighted_mean():
    results = [
        {
            "dataset": "a",
            "accept_length": 2.0,
            "acceptance_rate": 0.4,
            "output_throughput_tok_s": 100.0,
            "completed": 10,
        },
        {
            "dataset": "b",
            "accept_length": 4.0,
            "acceptance_rate": 0.8,
            "output_throughput_tok_s": 200.0,
            "completed": 30,
        },
    ]
    agg = _aggregate(results)
    # weighted mean: (2*10 + 4*30) / 40 = 3.5
    assert agg["accept_length"] == pytest.approx(3.5)
    assert agg["acceptance_rate"] == pytest.approx((0.4 * 10 + 0.8 * 30) / 40)
    assert agg["total_completed"] == 40
    assert agg["num_datasets"] == 2
    assert agg["num_datasets_ok"] == 2


def test_aggregate_excludes_error_rows():
    results = [
        {"dataset": "a", "accept_length": 2.0, "completed": 10},
        {"dataset": "b", "error": "boom"},
    ]
    agg = _aggregate(results)
    assert agg["accept_length"] == 2.0
    assert agg["num_datasets"] == 2
    assert agg["num_datasets_ok"] == 1
    assert agg["total_completed"] == 10  # only the ok row's completed count


def test_aggregate_all_errors_returns_none_metrics():
    results = [{"dataset": "a", "error": "x"}, {"dataset": "b", "error": "y"}]
    agg = _aggregate(results)
    assert agg["num_datasets_ok"] == 0
    assert agg["accept_length"] is None
    assert agg["total_completed"] == 0


def test_aggregate_skips_none_valued_metrics_without_crashing():
    results = [{"dataset": "a", "accept_length": None, "completed": 5}]
    agg = _aggregate(results)
    assert agg["accept_length"] is None


# ---------------------------------------------------------------------------
# _print_table (smoke: must not crash on error rows / None metrics)
# ---------------------------------------------------------------------------


def test_print_table_handles_errors_and_missing_metrics(capsys):
    results = [
        {
            "dataset": "a",
            "accept_length": 3.0,
            "acceptance_rate": 0.5,
            "output_throughput_tok_s": 100.0,
            "completed": 10,
            "speedup": 1.2,
        },
        {"dataset": "b", "error": "connection refused"},
    ]
    _print_table(results, _aggregate(results))
    out = capsys.readouterr().out
    assert "a" in out
    assert "ERROR" in out and "connection refused" in out
    assert "aggregate" in out


# ---------------------------------------------------------------------------
# _run / main end-to-end (mocked engine + JSON output)
# ---------------------------------------------------------------------------


def test_run_writes_output_json_and_returns_0(monkeypatch, tmp_path, capsys):
    async def _fake_summary(args):
        return {
            "accept_length": 3.0,
            "acceptance_rate": 0.5,
            "output_throughput_tok_s": 100.0,
            "completed": 4,
            "failed": 0,
        }

    monkeypatch.setattr(bench_sglang, "_run_summary", _fake_summary)
    out = tmp_path / "sweep.json"
    args = _sweep_args(datasets=["mt_bench"], output_json=str(out))
    rc = asyncio.run(_run(args))
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["results"][0]["dataset"] == "mt_bench"
    assert payload["aggregate"]["num_datasets_ok"] == 1
    assert "mt_bench" in capsys.readouterr().out


def test_run_returns_1_when_every_dataset_fails(monkeypatch):
    async def _fake_summary(args):
        return None

    monkeypatch.setattr(bench_sglang, "_run_summary", _fake_summary)
    rc = asyncio.run(_run(_sweep_args(datasets=["mt_bench"])))
    assert rc == 1


def test_run_rejects_unknown_dataset_name():
    with pytest.raises(ValueError, match="not in the sweep list"):
        asyncio.run(_run(_sweep_args(datasets=["not_a_real_dataset"])))


def test_main_builds_parser_and_dispatches(monkeypatch):
    seen = {}

    async def _fake_run(args):
        seen["engine"] = args.engine
        seen["server"] = args.server
        return 0

    monkeypatch.setattr(bench_sweep, "_run", _fake_run)
    rc = main(["--engine", "vllm", "--server", "http://localhost:8000", "--model", "m"])
    assert rc == 0
    assert seen == {"engine": "vllm", "server": "http://localhost:8000"}


# ---------------------------------------------------------------------------
# prompt_context_column (Alpaca-style secondary field) + all-requests-failed row
# ---------------------------------------------------------------------------


def test_dataset_spec_context_column_requires_prompt_column():
    # Valid alongside prompt_column.
    DatasetSpec(name="alpaca", input_data="d", prompt_column="instruction", prompt_context_column="input")
    # Invalid with messages_column (no raw text field to append to).
    with pytest.raises(ValueError, match="prompt_context_column"):
        DatasetSpec(name="a", input_data="d", messages_column="messages", prompt_context_column="input")


def test_default_presets_mt_bench_column_and_alpaca_context():
    presets = {s.name: s for s in bench_sweep.DEFAULT_DATASET_PRESETS}
    # MT-Bench prompts live under `prompt` on HuggingFaceH4/mt_bench_prompts, not `turns`.
    assert presets["mt_bench"].prompt_column == "prompt"
    # Alpaca appends its `input` task-context field.
    assert presets["alpaca"].prompt_column == "instruction"
    assert presets["alpaca"].prompt_context_column == "input"


def test_dataset_args_passes_prompt_context_column():
    base = argparse.Namespace(
        input_data="base",
        split="train",
        dataset_name=None,
        messages_column=None,
        prompt_column=None,
        max_new_tokens=256,
    )
    spec = DatasetSpec(
        name="alpaca", input_data="tatsu-lab/alpaca", prompt_column="instruction", prompt_context_column="input"
    )
    args = _dataset_args(base, spec)
    assert args.prompt_column == "instruction"
    assert args.prompt_context_column == "input"


def test_run_sweep_all_requests_failed_recorded_as_error(monkeypatch):
    # An engine whose every request failed returns a summary with completed=0
    # rather than raising; it must be an error row, not counted as a success.
    specs = [DatasetSpec(name="down", input_data="d", prompt_column="t")]

    async def _all_failed_summary(args):
        return {"accept_length": None, "completed": 0, "failed": 4}

    monkeypatch.setattr(bench_sglang, "_run_summary", _all_failed_summary)
    results = asyncio.run(_run_sweep(_sweep_args(), specs))
    assert results == [{"dataset": "down", "error": "all requests failed (0 completed)"}]

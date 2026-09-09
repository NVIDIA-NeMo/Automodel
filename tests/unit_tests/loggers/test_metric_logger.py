# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import threading

import nemo_automodel.components.loggers.metric_logger as metric_logger_mod
from nemo_automodel.components.loggers.metric_logger import MetricLogger, MetricLoggerDist


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_metric_logger_basic_jsonl(tmp_path):
    logfile = tmp_path / "metrics.jsonl"
    logger = MetricLogger(str(logfile), flush=True, append=False)

    logger.log(metric_logger_mod.MetricsSample(step=1, epoch=0, metrics={"loss": 1.23}))
    logger.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"accuracy": 0.9}))
    logger.close()

    assert logfile.exists()
    rows = _read_jsonl(logfile)
    assert len(rows) == 2

    # Every row should be valid JSON and include a timestamp
    assert "timestamp" in rows[0]
    assert rows[0]["loss"] == 1.23
    assert rows[0]["step"] == 1

    assert "timestamp" in rows[1]
    assert rows[1]["accuracy"] == 0.9
    assert rows[1]["epoch"] == 0


def test_append_vs_write_modes(tmp_path):
    logfile = tmp_path / "metrics.jsonl"

    # First run: write mode (truncate)
    logger = MetricLogger(str(logfile), flush=True, append=False)
    logger.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"a": 1}))
    logger.close()

    # Second run: append mode
    logger2 = MetricLogger(str(logfile), flush=True, append=True)
    logger2.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"b": 2}))
    logger2.close()

    rows = _read_jsonl(logfile)
    assert [row.get("a") for row in rows] == [1, None]
    assert [row.get("b") for row in rows] == [None, 2]


def test_thread_safe_logging(tmp_path):
    logfile = tmp_path / "metrics.jsonl"
    logger = MetricLogger(str(logfile), flush=True, append=False)

    num_threads = 10
    per_thread = 5

    def worker(tid: int):
        for i in range(per_thread):
            logger.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"thread": tid, "i": i}))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logger.close()

    rows = _read_jsonl(logfile)
    assert len(rows) == num_threads * per_thread

    # Verify that each expected (thread, i) pair appears exactly once
    seen = {(row["thread"], row["i"]) for row in rows}
    expected = {(t, i) for t in range(num_threads) for i in range(per_thread)}
    assert seen == expected


def test_flush_fsync_behavior(tmp_path, monkeypatch):
    logfile = tmp_path / "metrics.jsonl"
    # Use buffer_size=1 to force a save (and thus fsync) on each log call
    logger = MetricLogger(str(logfile), flush=True, append=False, buffer_size=1)

    calls = []

    def _fake_fsync(fd):
        calls.append(fd)

    monkeypatch.setattr(metric_logger_mod.os, "fsync", _fake_fsync)

    logger.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"x": 1}))
    logger.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"y": 2}))
    logger.close()

    # fsync should be called once per log when flush=True and buffer_size=1
    assert len(calls) == 2


def test_no_fsync_when_flush_false(tmp_path, monkeypatch):
    logfile = tmp_path / "metrics.jsonl"
    logger = MetricLogger(str(logfile), flush=False, append=False)

    called = False

    def _fake_fsync(fd):
        nonlocal called
        called = True

    monkeypatch.setattr(metric_logger_mod.os, "fsync", _fake_fsync)

    logger.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"a": 1}))
    logger.close()

    assert called is False


def test_metric_logger_dist_rank0_logs(tmp_path, monkeypatch):
    logfile = tmp_path / "metrics.jsonl"

    # Pretend distributed is initialized and we are rank 0
    monkeypatch.setattr(metric_logger_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(metric_logger_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(metric_logger_mod.dist, "get_world_size", lambda: 2)

    logger = MetricLoggerDist(str(logfile), flush=True, append=False)
    logger.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"k": 1}))
    logger.close()

    rows = _read_jsonl(logfile)
    assert len(rows) == 1
    assert rows[0]["k"] == 1


def test_metric_logger_dist_nonzero_noop(tmp_path, monkeypatch):
    logfile = tmp_path / "metrics.jsonl"

    # Pretend distributed is initialized and we are rank 1 (non-zero)
    monkeypatch.setattr(metric_logger_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(metric_logger_mod.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(metric_logger_mod.dist, "get_world_size", lambda: 2)

    logger = MetricLoggerDist(str(logfile), flush=True, append=False)
    # After __init__, .log is replaced with no-op on nonzero ranks
    logger.log(metric_logger_mod.MetricsSample(step=0, epoch=0, metrics={"should_not_write": True}))
    logger.close()

    # File may exist due to opener in base __init__, but it should be empty
    if logfile.exists():
        rows = _read_jsonl(logfile)
        assert rows == []
    else:
        # If file was not created, that's also acceptable
        assert True


def test_flush_method_is_callable_and_not_shadowed_by_the_flag(tmp_path):
    """The ``flush=`` constructor flag and the ``flush()`` method must coexist.

    Storing the flag as ``self.flush`` replaced the method with a bool on every instance,
    so the first caller -- the recipe, at its first checkpoint -- got
    ``TypeError: 'bool' object is not callable``.
    """
    logger = MetricLogger(str(tmp_path / "metrics.jsonl"), flush=True, append=False)
    try:
        assert callable(logger.flush)
    finally:
        logger.close()


def test_flush_persists_buffered_records_without_closing(tmp_path):
    """flush() is what lets a caller align durability with an external event.

    buffer_size is far larger than the number of records, so nothing reaches the file on
    record count alone; only the explicit flush can put them there, and the file must stay
    open for further writes afterwards.
    """
    logfile = tmp_path / "metrics.jsonl"
    logger = MetricLogger(str(logfile), append=False, buffer_size=1000)
    try:
        logger.log(metric_logger_mod.MetricsSample(step=1, epoch=0, metrics={"loss": 1.0}))
        logger.log(metric_logger_mod.MetricsSample(step=2, epoch=0, metrics={"loss": 2.0}))
        assert _read_jsonl(logfile) == [], "buffer_size not reached, so nothing is durable yet"

        logger.flush()
        rows = _read_jsonl(logfile)
        assert [r["step"] for r in rows] == [1, 2]

        # still usable: the flush drained the buffer, it did not close the file
        logger.log(metric_logger_mod.MetricsSample(step=3, epoch=0, metrics={"loss": 3.0}))
        logger.flush()
        assert [r["step"] for r in _read_jsonl(logfile)] == [1, 2, 3]
    finally:
        logger.close()


def test_flush_on_an_empty_buffer_is_a_no_op(tmp_path):
    """The recipe flushes at every checkpoint, including ones with nothing buffered."""
    logfile = tmp_path / "metrics.jsonl"
    logger = MetricLogger(str(logfile), append=False, buffer_size=1000)
    try:
        logger.flush()
        logger.flush()
        assert logfile.read_text() == ""
    finally:
        logger.close()

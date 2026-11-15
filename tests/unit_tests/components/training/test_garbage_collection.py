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

import gc
from unittest.mock import MagicMock, patch

import pytest

from nemo_automodel.components.training.garbage_collection import GarbageCollection


@pytest.fixture(autouse=True)
def restore_gc_state():
    """Ensure garbage collection state is restored after each test."""
    was_enabled = gc.isenabled()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()


def test_init_disables_automatic_gc_and_collects(monkeypatch):
    disable_mock = MagicMock()
    collect_mock = MagicMock()
    monkeypatch.setattr("nemo_automodel.components.training.garbage_collection.gc.disable", disable_mock)
    monkeypatch.setattr("nemo_automodel.components.training.garbage_collection.gc.collect", collect_mock)

    GarbageCollection(gc_every_steps=10)

    disable_mock.assert_called_once()
    collect_mock.assert_called_once_with(1)


def test_init_defaults():
    with patch.object(GarbageCollection, "collect") as mock_collect, patch("gc.disable") as mock_disable:
        collector = GarbageCollection()
        assert collector.gc_every_steps == 1000
        mock_disable.assert_called_once()
        mock_collect.assert_called_once_with("Initial GC collection")


def test_init_custom_positive_steps():
    with patch.object(GarbageCollection, "collect") as mock_collect, patch("gc.disable") as mock_disable:
        collector = GarbageCollection(gc_every_steps=50)
        assert collector.gc_every_steps == 50
        mock_disable.assert_called_once()
        mock_collect.assert_called_once_with("Initial GC collection")


@pytest.mark.parametrize("invalid_steps", [0, -1, -100])
def test_gc_every_steps_must_be_positive(invalid_steps: int):
    with pytest.raises(AssertionError, match="gc_every_steps must be a positive integer"):
        GarbageCollection(gc_every_steps=invalid_steps)


def test_run_collects_periodically(monkeypatch):
    collect_mock = MagicMock()
    monkeypatch.setattr(
        "nemo_automodel.components.training.garbage_collection.GarbageCollection.collect",
        collect_mock,
    )

    gc_tool = GarbageCollection(gc_every_steps=3)
    gc_tool.run(1)
    gc_tool.run(2)
    gc_tool.run(3)
    gc_tool.run(6)

    assert collect_mock.call_count == 3
    collect_mock.assert_any_call("Initial GC collection")
    collect_mock.assert_any_call("Performing periodic GC collection")


def test_run_triggers_on_multiples():
    with patch.object(GarbageCollection, "collect") as mock_collect, patch("gc.disable"):
        collector = GarbageCollection(gc_every_steps=10)
        mock_collect.reset_mock()

        # step_count <= 1: should not trigger
        collector.run(0)
        collector.run(1)
        mock_collect.assert_not_called()

        # step_count > 1 and not a multiple: should not trigger
        collector.run(5)
        collector.run(9)
        collector.run(11)
        mock_collect.assert_not_called()

        # step_count > 1 and exact multiple: should trigger
        collector.run(10)
        mock_collect.assert_called_once_with("Performing periodic GC collection")

        mock_collect.reset_mock()
        collector.run(20)
        mock_collect.assert_called_once_with("Performing periodic GC collection")


def test_collect_invokes_gc_and_logs(caplog):
    with patch("gc.collect") as mock_gc_collect, caplog.at_level("INFO"):
        GarbageCollection.collect("Test reason", generation=2)
        mock_gc_collect.assert_called_once_with(2)
        assert any("[GC] Test reason took" in record.message for record in caplog.records)

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

import gc
from unittest.mock import patch

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


class TestGarbageCollection:
    def test_init_defaults(self):
        with patch.object(GarbageCollection, "collect") as mock_collect, patch("gc.disable") as mock_disable:
            collector = GarbageCollection()
            assert collector.gc_every_steps == 1000
            mock_disable.assert_called_once()
            mock_collect.assert_called_once_with("Initial GC collection")

    def test_init_custom_positive_steps(self):
        with patch.object(GarbageCollection, "collect") as mock_collect, patch("gc.disable") as mock_disable:
            collector = GarbageCollection(gc_every_steps=50)
            assert collector.gc_every_steps == 50
            mock_disable.assert_called_once()
            mock_collect.assert_called_once_with("Initial GC collection")

    @pytest.mark.parametrize("invalid_steps", [0, -1, -100])
    def test_init_invalid_steps_raises(self, invalid_steps: int):
        with pytest.raises(AssertionError, match="gc_every_steps must be a positive integer"):
            GarbageCollection(gc_every_steps=invalid_steps)

    def test_run_triggers_on_multiples(self):
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

    def test_collect_invokes_gc_and_logs(self, caplog):
        with patch("gc.collect") as mock_gc_collect, caplog.at_level("INFO"):
            GarbageCollection.collect("Test reason", generation=2)
            mock_gc_collect.assert_called_once_with(2)
            assert any("[GC] Test reason took" in record.message for record in caplog.records)

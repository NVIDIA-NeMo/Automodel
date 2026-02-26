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

from unittest.mock import MagicMock

import pytest

from nemo_automodel.components.training.garbage_collection import GarbageCollection


def test_init_disables_automatic_gc_and_collects(monkeypatch):
    disable_mock = MagicMock()
    collect_mock = MagicMock()
    monkeypatch.setattr("nemo_automodel.components.training.garbage_collection.gc.disable", disable_mock)
    monkeypatch.setattr("nemo_automodel.components.training.garbage_collection.gc.collect", collect_mock)

    GarbageCollection(gc_every_steps=10)

    disable_mock.assert_called_once()
    collect_mock.assert_called_once_with(1)


def test_gc_every_steps_must_be_positive():
    with pytest.raises(AssertionError, match="gc_every_steps must be a positive integer"):
        GarbageCollection(gc_every_steps=0)


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

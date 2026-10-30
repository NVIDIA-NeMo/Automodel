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

"""Spans around the blocking waits of an asynchronous checkpoint save."""

from concurrent.futures import Future
from types import SimpleNamespace

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from tests.utils.telemetry import lens_capture  # noqa: F401


def _done() -> Future:
    future = Future()
    future.set_result(None)
    return future


def _checkpointer() -> Checkpointer:
    checkpointer = Checkpointer.__new__(Checkpointer)
    checkpointer._model_ctx = SimpleNamespace(staging_active=False, future=None, stager=None)
    checkpointer._optim_ctx = SimpleNamespace(staging_active=False, future=None, stager=None)
    checkpointer._join_deferred_consolidation = lambda: None
    return checkpointer


def test_waits_emit_no_span_when_nothing_is_pending(lens_capture):
    capture = lens_capture("checkpoint")
    checkpointer = _checkpointer()

    # Called before every optimizer step; with no save in flight it must stay span-free.
    checkpointer.maybe_wait_for_staging()
    checkpointer.async_wait()

    assert capture.spans() == []


def test_waits_emit_one_span_per_pending_future(lens_capture):
    capture = lens_capture("checkpoint")
    checkpointer = _checkpointer()
    for context in (checkpointer._model_ctx, checkpointer._optim_ctx):
        context.staging_active = True
        context.future = SimpleNamespace(staging_completion=_done(), upload_completion=_done())

    checkpointer.maybe_wait_for_staging()
    checkpointer.async_wait()

    assert [span.name for span in capture.spans()] == [
        "automodel.checkpoint.wait.staging.model",
        "automodel.checkpoint.wait.staging.optimizer",
        "automodel.checkpoint.wait.upload.model",
        "automodel.checkpoint.wait.upload.optimizer",
    ]
    assert checkpointer._model_ctx.future is None and checkpointer._optim_ctx.future is None

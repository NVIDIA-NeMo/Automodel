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

from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from nemo_automodel.components.checkpoint import _torch_backports


def test_async_checkpoint_patch_waits_for_daemon_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    original_called = threading.Event()
    patched_returned = threading.Event()
    save_future: Future[None] = Future()
    returned_futures: list[Future[None]] = []

    class FakeExecutor:
        @staticmethod
        def _execute_save_impl() -> None:
            return None

        def execute_save(self) -> Future[None]:
            original_called.set()
            return save_future

    async_process_executor = SimpleNamespace(
        _CHECKPOINT_PROCESS=None,
        _ProcessBasedAsyncCheckpointExecutor=FakeExecutor,
    )
    monkeypatch.setattr(
        _torch_backports.importlib,
        "import_module",
        lambda _: async_process_executor,
    )
    _torch_backports.apply_async_checkpoint_patch()

    def call_patched_execute_save() -> None:
        returned_futures.append(FakeExecutor().execute_save())
        patched_returned.set()

    caller = threading.Thread(target=call_patched_execute_save)
    caller.start()
    try:
        assert original_called.wait(timeout=1)
        assert not patched_returned.wait(timeout=0.05)
    finally:
        async_process_executor._CHECKPOINT_PROCESS = object()
        caller.join(timeout=1)

    assert patched_returned.wait(timeout=1)
    assert returned_futures == [save_future]


def test_async_checkpoint_patch_propagates_initialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    initialization_error = RuntimeError("daemon initialization failed")
    save_future: Future[None] = Future()
    save_future.set_exception(initialization_error)

    class FakeExecutor:
        @staticmethod
        def _execute_save_impl() -> None:
            return None

        def execute_save(self) -> Future[None]:
            return save_future

    async_process_executor = SimpleNamespace(
        _CHECKPOINT_PROCESS=None,
        _ProcessBasedAsyncCheckpointExecutor=FakeExecutor,
    )
    monkeypatch.setattr(
        _torch_backports.importlib,
        "import_module",
        lambda _: async_process_executor,
    )
    _torch_backports.apply_async_checkpoint_patch()

    with pytest.raises(RuntimeError, match="daemon initialization failed"):
        FakeExecutor().execute_save()


def test_async_checkpoint_patch_accepts_successful_completion_after_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AssigningFuture(Future[None]):
        def done(self) -> bool:
            async_process_executor._CHECKPOINT_PROCESS = object()
            if not super().done():
                self.set_result(None)
            return True

    save_future: Future[None] = AssigningFuture()

    class FakeExecutor:
        @staticmethod
        def _execute_save_impl() -> None:
            return None

        def execute_save(self) -> Future[None]:
            return save_future

    async_process_executor = SimpleNamespace(
        _CHECKPOINT_PROCESS=None,
        _ProcessBasedAsyncCheckpointExecutor=FakeExecutor,
    )
    monkeypatch.setattr(
        _torch_backports.importlib,
        "import_module",
        lambda _: async_process_executor,
    )
    _torch_backports.apply_async_checkpoint_patch()

    assert FakeExecutor().execute_save() is save_future
    assert async_process_executor._CHECKPOINT_PROCESS is not None

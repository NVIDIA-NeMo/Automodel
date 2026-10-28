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

"""Bounded CPU workers and a parent-owned store for distributed loader tests."""

import sys
import time
from collections.abc import Callable, Iterator
from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaConfig


@pytest.fixture
def distributed_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    # Free-threaded Python currently needs the GIL for local Gloo startup.
    monkeypatch.setenv("PYTHON_GIL", "1")
    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    # Keep the store alive until all workers exit, including file:// rendezvous
    # tests whose Zephon MTP workers still need a shared aggregation endpoint.
    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=timedelta(seconds=30))
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(store.port))
    yield f"torch://127.0.0.1:{store.port}"


def run_ranks(worker: Callable[..., None], args: tuple[object, ...], *, ranks: int = 2, timeout: int = 90) -> None:
    """Join every worker or terminate the whole group after a failure/timeout."""
    context = mp.spawn(worker, args=args, nprocs=ranks, join=False)
    try:
        deadline = time.monotonic() + timeout
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{ranks}-rank Zephon integration exceeded {timeout} seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


def tiny_llama_config() -> LlamaConfig:
    """Two decoder layers allow real TP and two-stage PP tests on the CPU."""
    return LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        attn_implementation="sdpa",
        use_cache=False,
    )

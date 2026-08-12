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

"""Scheduled process-group coverage for Hugging Face checkpoint reload synchronization."""

import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist
import torch.multiprocessing as mp

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _finish_hf_reload_sync,
    _prepare_hf_reload_sync,
)


def _run_hf_reload_sync_rank(rank: int, init_path: str, checkpoint_dir: str) -> None:
    os.environ["TORCHELASTIC_RUN_ID"] = "checkpoint-robustness-hf-sync-test"
    os.environ["HF_RELOAD_TIMEOUT_SECONDS"] = "15"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=3),
    )
    try:
        cfg = SimpleNamespace(checkpoint=SimpleNamespace(checkpoint_dir=checkpoint_dir))
        sync_paths = _prepare_hf_reload_sync(cfg)
        if rank == 0:
            time.sleep(4)
        _finish_hf_reload_sync(sync_paths)
    finally:
        dist.destroy_process_group()


def test_hf_reload_wait_does_not_start_collective_during_rank0_work(tmp_path: Path) -> None:
    mp.spawn(
        _run_hf_reload_sync_rank,
        args=(str(tmp_path / "dist-init"), str(tmp_path / "checkpoints")),
        nprocs=2,
        join=True,
    )

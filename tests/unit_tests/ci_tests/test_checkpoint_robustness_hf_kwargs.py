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

import os
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _finish_hf_reload_sync,
    _hf_source_load_kwargs,
    _prepare_hf_reload_sync,
    _wait_for_hf_reload_rank0,
)


def _run_hf_reload_sync_rank(rank, init_path, checkpoint_dir):
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


@pytest.mark.parametrize(
    ("model_type", "expected_attn_implementation"),
    [("nemotron_h", "eager"), ("nemotron_flash", "flash_attention_2")],
)
def test_remote_code_attention_implementation(model_type, expected_attn_implementation):
    with patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(model_type=model_type),
    ) as from_pretrained:
        hf_kwargs = _hf_source_load_kwargs(
            {"revision": "model-revision", "token": "model-token"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["attn_implementation"] == expected_attn_implementation
    from_pretrained.assert_called_once_with(
        "model-path",
        trust_remote_code=True,
        revision="model-revision",
        token="model-token",
    )


def test_explicit_attention_implementation_is_preserved():
    with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("must not probe config")):
        hf_kwargs = _hf_source_load_kwargs(
            {"attn_implementation": "eager"},
            pretrained_model_name_or_path="model-path",
            source_dtype=torch.bfloat16,
            trust_remote_code=True,
            experts_implementation=None,
            device=torch.device("cpu"),
            hf_device_map_auto=False,
        )

    assert hf_kwargs["attn_implementation"] == "eager"


def test_hf_reload_wait_returns_after_rank0_marker(tmp_path):
    done_path = tmp_path / "done"
    done_path.write_text("ok\n")

    _wait_for_hf_reload_rank0(done_path)


def test_hf_reload_wait_has_separate_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_RELOAD_TIMEOUT_SECONDS", "0")

    with pytest.raises(TimeoutError, match="rank 0 vanilla-HF reload"):
        _wait_for_hf_reload_rank0(tmp_path / "done")


def test_hf_reload_wait_does_not_start_collective_during_rank0_work(tmp_path):
    mp.spawn(
        _run_hf_reload_sync_rank,
        args=(str(tmp_path / "dist-init"), str(tmp_path / "checkpoints")),
        nprocs=2,
        join=True,
    )

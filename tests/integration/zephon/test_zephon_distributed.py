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

"""Bounded two-rank CPU checkpoint replay and synchronized finite completion."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from transformers import PreTrainedTokenizerFast

from tests.utils.zephon_distributed import distributed_environment as distributed_environment
from tests.utils.zephon_distributed import run_ranks
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


def _dcp_worker(rank, directory, source_config, tokenizer_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    config = replace(
        source_config,
        batch_size=1,
        zephon=replace(
            source_config.zephon,
            checkpoint=replace(
                source_config.zephon.checkpoint,
                aggregate_dir=f"{directory}/aggregate",
                run_id="dcp-test",
                canonical_replicas=2,
            ),
        ),
    )
    loader = config.build(tokenizer=tokenizer, dp_rank=rank, dp_world_size=2)
    restored = config.build(tokenizer=tokenizer, dp_rank=rank, dp_world_size=2)
    try:
        first = next(loader)
        # This test's source gives each document a unique repeated word token.
        document_ids = set(first["input_ids"].flatten().tolist()) - {tokenizer.eos_token_id}
        peers = [None, None]
        dist.all_gather_object(peers, document_ids)
        assert all(peers) and peers[0].isdisjoint(peers[1])
        dcp.save({"dataloader": loader}, checkpoint_id=f"{directory}/checkpoint")
        expected = [next(loader) for _ in range(3)]
        # DCP calls checkpoint() on the fresh destination to discover its keys.
        dcp.load({"dataloader": restored}, checkpoint_id=f"{directory}/checkpoint")
        for wanted in expected:
            torch.testing.assert_close(next(restored), wanted)
        Path(directory, f"rank-{rank}.ok").write_text("replayed")
    finally:
        loader.close()
        restored.close()
        dist.destroy_process_group()


def test_two_rank_dcp_into_fresh_loaders(config, tokenizer, tmp_path, distributed_environment):
    Path(config.zephon.sources[0].path, "text.jsonl").write_text(
        "".join(json.dumps({"text": f"w{i} w{i}"}) + "\n" for i in range(36))
    )
    run_ranks(_dcp_worker, (str(tmp_path), config, tokenizer.name_or_path))
    assert [Path(tmp_path, f"rank-{rank}.ok").read_text() for rank in range(2)] == ["replayed", "replayed"]


def _finite_worker(rank, directory, source_config, tokenizer_path):
    from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig

    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/finite-rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    config = replace(
        source_config,
        batch_size=1,
        zephon=replace(
            source_config.zephon,
            checkpoint=replace(source_config.zephon.checkpoint, run_id="finite-test", canonical_replicas=2),
            runtime=replace(source_config.zephon.runtime, mtp_mode=False),
        ),
    )
    loader = config.build(tokenizer=tokenizer, dp_rank=rank, dp_world_size=2)
    try:
        scheduler = StepSchedulerConfig(global_batch_size=4, max_steps=100, preemption_signal=None).build(loader, 2, 1)
        steps = 0
        for epoch in scheduler.epochs:
            scheduler.set_epoch(epoch)
            for batches in scheduler:
                assert len(batches) == 2
                steps += 1
        assert scheduler.stream_exhausted
        assert steps > 0
        counts = [None, None]
        dist.all_gather_object(counts, steps)
        assert counts[0] == counts[1]
        dcp.save({"dataloader": loader, "scheduler": scheduler}, checkpoint_id=f"{directory}/finite-checkpoint")
        assert scheduler.state_dict()["step"] == steps
        Path(directory, f"finite-{rank}.ok").write_text(str(steps))
    finally:
        loader.close()
        dist.destroy_process_group()


def test_two_rank_real_pipeline_finite_completion(config, tokenizer, tmp_path, distributed_environment):
    run_ranks(_finite_worker, (str(tmp_path), config, tokenizer.name_or_path))
    assert Path(tmp_path, "finite-0.ok").read_text() == Path(tmp_path, "finite-1.ok").read_text()

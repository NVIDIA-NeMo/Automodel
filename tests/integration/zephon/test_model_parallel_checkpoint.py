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

"""All-rank data/checkpoint coordination for a DP=1, TP=2, PP=2 CPU mesh.

This tests the data topology and checkpoint protocol. Model execution and PP
microbatch slicing have separate tests; no model is simulated here.
"""

import hashlib
import json
import random
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from transformers import PreTrainedTokenizerFast

from tests.utils.zephon_distributed import distributed_environment as distributed_environment
from tests.utils.zephon_distributed import run_ranks
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


class _CheckpointWitness:
    """Record actual DCP calls while delegating the opaque loader state unchanged."""

    def __init__(self, loader):
        self.loader = loader
        self.checkpoint_calls = 0
        self.restore_calls = 0
        self.digest = None

    def state_dict(self):
        self.checkpoint_calls += 1
        state = self.loader.state_dict()
        self.digest = hashlib.sha256(state["zephon"]).hexdigest()
        return state

    def load_state_dict(self, state):
        self.restore_calls += 1
        self.loader.load_state_dict(state)


def _assert_peer_batches(batch):
    peers = [None] * 4
    dist.all_gather_object(peers, batch)
    assert batch["input_ids"].shape == batch["labels"].shape == (2, 7)
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2]] * 2
    assert batch["_packed_seq_ids"].tolist() == [[1, 1, 1, 1, 2, 2, 2]] * 2
    assert batch["_packed_seq_ids"].shape == (2, 7)
    assert batch["_packed_seq_ids"].tolist() == [[1, 1, 1, 1, 2, 2, 2]] * 2
    assert "attention_mask" not in batch
    for peer in peers:
        torch.testing.assert_close(peer, batch, atol=0, rtol=0)


def _model_parallel_worker(rank, directory, source_config, tokenizer_path, aggregate_store):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    loader = restored = None
    try:
        mesh = init_device_mesh("cpu", (1, 2, 2), mesh_dim_names=("dp", "tp", "pp"))
        dp = mesh["dp"]
        assert dp.size() == 1 and dp.get_local_rank() == 0
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        config = replace(
            source_config,
            zephon=replace(
                source_config.zephon,
                checkpoint=replace(
                    source_config.zephon.checkpoint,
                    aggregate_store=aggregate_store,
                    run_id="tp-pp-checkpoint",
                    canonical_replicas=1,
                ),
                runtime=replace(source_config.zephon.runtime, mtp_mode=True),
            ),
        )
        # Ambient global-rank RNG state must not alter Zephon's DP-owned stream.
        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        loader = config.build(
            tokenizer=tokenizer,
            dp_rank=dp.get_local_rank(),
            dp_world_size=dp.size(),
            pp_enabled=True,
            supports_seq_lens=False,
            cp_size=1,
        )
        _assert_peer_batches(next(loader))

        saved = _CheckpointWitness(loader)
        dcp.save({"dataloader": saved}, checkpoint_id=f"{directory}/checkpoint")
        witnesses = [None] * 4
        dist.all_gather_object(witnesses, (saved.checkpoint_calls, saved.digest))
        assert all(calls > 0 for calls, _ in witnesses)
        assert len({digest for _, digest in witnesses}) == 1

        expected = []
        for _ in range(3):
            batch = next(loader)
            _assert_peer_batches(batch)
            expected.append(batch)
        loader.close()

        restored = config.build(tokenizer=tokenizer, dp_rank=dp.get_local_rank(), dp_world_size=dp.size())
        loaded = _CheckpointWitness(restored)
        dcp.load({"dataloader": loaded}, checkpoint_id=f"{directory}/checkpoint")
        assert loaded.checkpoint_calls > 0 and loaded.restore_calls == 1
        for wanted in expected:
            actual = next(restored)
            _assert_peer_batches(actual)
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
        Path(directory, f"rank-{rank}.json").write_text(
            json.dumps({"coordinate": mesh.get_coordinate(), "replayed": len(expected)})
        )
    finally:
        if loader is not None:
            loader.close()
        if restored is not None:
            restored.close()
        dist.destroy_process_group()


def test_tp_pp_peers_checkpoint_and_replay_on_every_rank(config, tokenizer, tmp_path, distributed_environment):
    run_ranks(
        _model_parallel_worker,
        (str(tmp_path), config, tokenizer.name_or_path, distributed_environment),
        ranks=4,
        timeout=120,
    )
    results = [json.loads(Path(tmp_path, f"rank-{rank}.json").read_text()) for rank in range(4)]
    assert {tuple(result["coordinate"]) for result in results} == {
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    }
    assert all(result["replayed"] == 3 for result in results)

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

"""Exercise the CLM distributed encoding to single-rank training transition."""

import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from nemo_automodel.components.models.clm.heads import EmbeddingTable
from nemo_automodel.recipes.decision.train_clm_heads import TrainCLMHeadsRecipe


class TinyTokenizer:
    pad_token = "pad"

    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.tensor([[len(text)] for text in texts])}


class TinyEncoder(nn.Module):
    def encode(self, batch):
        values = batch["input_ids"].float()
        return torch.cat((values, values + 1), dim=1)


def _worker(rank, rendezvous, output, scenario):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=3)
    )
    try:
        recipe = TrainCLMHeadsRecipe.__new__(TrainCLMHeadsRecipe)
        recipe.dist_env = SimpleNamespace(is_main=rank == 0, rank=rank, world_size=2)
        recipe.device = torch.device("cpu")
        recipe.cfg = SimpleNamespace(get={}.get, encoder=SimpleNamespace(instantiate=TinyEncoder))
        output = Path(output)
        if scenario == "training":
            recipe.seed = 7
            recipe.mode = "eval"
            recipe.output_dir = output
            recipe.splits = {"test": []}

            def report(heads, split):
                # Exceed the group's collective timeout during successful rank-zero work.
                time.sleep(4)
                (output / "reported").touch()

            recipe._report = report
            with patch("nemo_automodel.recipes.decision.train_clm_heads.load_clm_heads", return_value=nn.Identity()):
                recipe.run_train_validation_loop()
            assert not dist.is_initialized()
        elif scenario == "mixed_visibility":
            cache = output / f"rank{rank}.pt"
            if rank == 0:
                EmbeddingTable(["text"], torch.ones(1, 2)).save(cache)
            recipe._cache_path = lambda texts: cache
            with pytest.raises(ValueError, match="cache.*visible.*all ranks"):
                recipe._embed_all(["text"])
        else:
            recipe._cache_path = lambda texts: output / "shared.pt"
            # First use leaves rank 1 with an empty shard; the second loads the shared cache.
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()) as tokenizer:
                table = recipe._embed_all(["text"])
                torch.testing.assert_close(table.embeddings, torch.tensor([[4.0, 5.0]]))
                assert tokenizer.call_count == 1
                table = recipe._embed_all(["text"])
                torch.testing.assert_close(table.embeddings, torch.tensor([[4.0, 5.0]]))
                assert tokenizer.call_count == 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["training", "mixed_visibility", "cache"])
def test_clm_distributed_phase_boundaries(tmp_path, scenario):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), scenario), nprocs=2, join=True)
    if scenario == "training":
        assert (tmp_path / "reported").exists()

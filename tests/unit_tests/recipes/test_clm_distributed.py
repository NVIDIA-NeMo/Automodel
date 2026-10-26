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
from contextlib import nullcontext
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
    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode lengths into deterministic test features.

        Args:
            batch: Mapping with input_ids of shape [batch, 1], containing text lengths.

        Returns:
            Float32 Tensor of shape [batch, 2], with rows [length, length + 1].
        """
        values = batch["input_ids"].float()
        return torch.cat((values, values + 1), dim=1)


def _worker(rank, rendezvous, output, scenario):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        recipe = TrainCLMHeadsRecipe.__new__(TrainCLMHeadsRecipe)
        recipe.dist_env = SimpleNamespace(is_main=rank == 0, rank=rank, world_size=2)
        recipe.device = torch.device("cpu")

        def build_encoder(**kwargs):
            # These omitted YAML settings must be resolved identically for cache identity and execution.
            assert kwargs["pooling"] == "last"
            assert kwargs["l2_normalize"] is True
            assert kwargs["is_causal"] is True
            assert kwargs["dtype"] == "bfloat16"
            assert kwargs["attn_implementation"] == "sdpa"
            return TinyEncoder()

        recipe.cfg = SimpleNamespace(
            get={"encoder.pretrained_model_name_or_path": "org/encoder", "encoder.revision": "a" * 40}.get,
            encoder=SimpleNamespace(instantiate=build_encoder),
        )
        output = Path(output)
        if scenario == "training":
            recipe.seed = 7
            recipe.mode = "eval"
            recipe.output_dir = output
            recipe.splits = {"test": []}

            def report(heads, split):
                # Exceed the group's collective timeout during successful rank-zero work.
                time.sleep(31)
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
        elif scenario == "cache_failures":
            recipe._cache_path = lambda texts: output / f"private{rank}" / "cache.pt"
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()):
                with pytest.raises(RuntimeError, match="assemble embedding cache"):
                    recipe._embed_all(["text"])
                recipe._cache_path = lambda texts: output / "write-failure.pt"
                failure = (
                    patch("torch.save", side_effect=OSError("injected disk error")) if rank == 1 else nullcontext()
                )
                with failure, pytest.raises(RuntimeError, match="write embedding cache shards"):
                    recipe._embed_all(["text"])
            with pytest.raises(ValueError, match="at least one text"):
                recipe._embed_all([])
            assert not (output / "write-failure.pt").exists()
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


@pytest.mark.parametrize("scenario", ["training", "mixed_visibility", "cache", "cache_failures"])
def test_clm_distributed_phase_boundaries(tmp_path, scenario):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), scenario), nprocs=2, join=True)
    if scenario == "training":
        assert (tmp_path / "reported").exists()

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

"""Real CPU collectives for uneven multimodal mining partitions."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.retrieval.mine_hard_negatives import MineHardNegativesRecipe


class _Encoder:
    def encode_queries(self, queries, *, batch_size):
        assert queries, "An empty rank must not call the encoder"
        return np.array([[float(query), float(query) + 1] for query in queries], dtype=np.float32)

    def encode_documents(self, documents, *, batch_size):
        assert 0 < len(documents) <= batch_size
        return self.encode_queries([document["text"] for document in documents], batch_size=batch_size)


class _Documents:
    def __init__(self):
        self.calls = []
        self.offset = 0

    def get_document_by_id(self, document_id):
        self.calls.append(document_id)
        return {"text": str(int(document_id) + self.offset), "image": None}


def _mining_worker(rank: int, rendezvous: str, scratch: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        recipe = MineHardNegativesRecipe(ConfigNode({}))
        recipe.dist_env = SimpleNamespace(rank=rank, world_size=2, is_main=rank == 0, device=torch.device("cpu"))
        recipe.multimodal_encoder = _Encoder()
        recipe.query_embedding_batch_size = 1
        recipe.document_embedding_batch_size = 1
        recipe.corpus_chunk_size = 2
        recipe.cache_embeddings_dir = scratch
        recipe.idx_to_doc = {index: str(index) for index in range(3)}
        recipe.documents_dataset = _Documents()
        for offset in (0, 10):
            recipe.questions = [str(offset)]
            recipe.documents_dataset.offset = offset
            queries = recipe._encode_queries_sharded()
            documents = recipe._encode_all_documents()
            if rank == 0:
                np.testing.assert_array_equal(queries, [[offset, offset + 1]])
                np.testing.assert_array_equal(documents, [[offset + index, offset + index + 1] for index in range(3)])
                assert np.isfinite(documents).all()
            else:
                assert queries.shape == documents.shape == (0, 0)
            # The second pass deliberately reuses scratch with changed inputs.
            # Every rank must finish reading before the next pass replaces files.
            dist.barrier()
        assert recipe.documents_dataset.calls == (["0", "2", "0", "2"] if rank == 0 else ["1", "1"])
        assert not (Path(scratch) / "query_shards").exists()
        assert not (Path(scratch) / "corpus_chunks").exists()
    finally:
        dist.destroy_process_group()


def test_real_two_rank_mining_handles_empty_queries_and_tail_chunk(tmp_path):
    mp.spawn(_mining_worker, args=((tmp_path / "gloo").as_uri(), str(tmp_path / "scratch")), nprocs=2, join=True)

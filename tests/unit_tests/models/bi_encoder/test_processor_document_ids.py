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

"""Real CPU processor and distributed masking regression; no model downloads."""

import json
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PixtralImageProcessor
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.llm.retrieval_collator import ProcessorMethodCollator
from nemo_automodel.components.datasets.llm.retrieval_dataset import make_retrieval_dataset
from nemo_automodel.components.models.common.inbatch_neg_utils import (
    dist_gather_tensor,
    mask_gathered_passages_same_doc_as_positive,
)
from nemo_automodel.components.models.ministral_bidirectional.processor import Mistral3BiEncoderProcessor
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe
from nemo_automodel.shared.retrieval_ids import document_id_to_int64


def processor():
    backend = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["[IMG]", "[IMG_BREAK]", "[IMG_END]"],
        model_max_length=256,
        padding_side="right",
    )
    return Mistral3BiEncoderProcessor(
        tokenizer=tokenizer,
        image_processor=PixtralImageProcessor(size={"longest_edge": 56}),
        patch_size=14,
        q_max_length=64,
        p_max_length=256,
    )


def examples(visual):
    image = Image.new("RGB", (56, 56), "white") if visual else ""
    return [
        {
            "question": "user need A",
            "doc_text": ["shared source", "different A"],
            "doc_image": [image, ""],
            "doc_id": ["shared", "negative-a"],
        },
        {
            "question": "user need B",
            "doc_text": ["shared source", "different B"],
            "doc_image": [image, ""],
            "doc_id": ["shared", "negative-b"],
        },
    ]


@pytest.mark.parametrize("visual", [False, True])
@pytest.mark.parametrize("backend", ["pt", "np"])
def test_real_processor_preserves_order_identity_and_backend(visual, backend):
    result = processor().process_queries_documents_biencoder(examples(visual), return_tensors=backend)
    ids = result["passage_doc_ids"]
    assert ids.tolist() == [7139269065447014814, 4739242844499128429, 7139269065447014814, 5682275166721363359]
    assert ids.dtype == (torch.int64 if backend == "pt" else np.int64)
    assert result["d_input_ids"].shape[0] == len(ids) == 4
    assert result["passage_modality"].tolist() == ([2, 0, 2, 0] if visual else [0, 0, 0, 0])
    assert (result["d_pixel_values"] is not None) == visual
    assert "d_passage_doc_ids" not in result


def test_records_without_ids_remain_supported():
    features = examples(False)
    for feature in features:
        del feature["doc_id"]
    result = ProcessorMethodCollator(processor(), "process_queries_documents_biencoder")(features)
    assert "passage_doc_ids" not in result
    assert result["d_input_ids"].shape[0] == 4


def test_inline_jsonl_loader_omits_absent_ids_before_processor_collation(tmp_path):
    source = tmp_path / "inline.jsonl"
    source.write_text(json.dumps({"query": "user need", "pos_doc": "positive", "neg_doc": ["negative"]}) + "\n")
    dataset = make_retrieval_dataset(data_dir_list=str(source), data_type="train", n_passages=2)
    feature = dataset[0]
    assert feature["doc_id"] == ["", ""]

    batch = ProcessorMethodCollator(processor(), "process_queries_documents_biencoder")([feature])
    assert "passage_doc_ids" not in batch
    assert batch["d_input_ids"].shape[0] == 2


def test_portable_processor_identity_matches_shared_text_collator_encoding():
    identifiers = ["ascii", "文档", "médicament", "a/b/page:1"] + [f"source-{index}" for index in range(20)]
    features = [
        {
            "question": "user need",
            "doc_text": ["source"] * len(identifiers),
            "doc_image": [""] * len(identifiers),
            "doc_id": identifiers,
        }
    ]
    result = processor().process_queries_documents_biencoder(features)
    assert result["passage_doc_ids"].tolist() == [document_id_to_int64(value) for value in identifiers]


@pytest.mark.parametrize(
    "bad_ids", [None, [], ["only-one"], ["", "negative"], [1, 2], "not-a-list", ["a", "b", "extra"]]
)
def test_incomplete_or_misaligned_ids_fail_clearly(bad_ids):
    features = examples(False)
    features[1]["doc_id"] = bad_ids
    with pytest.raises(ValueError, match="one aligned string per candidate"):
        processor().process_queries_documents_biencoder(features)


def test_mixed_id_policy_is_not_silently_disabled():
    features = examples(False)
    del features[1]["doc_id"]
    with pytest.raises(ValueError, match="same present or absent ID policy"):
        processor().process_queries_documents_biencoder(features)


def test_missing_and_all_empty_id_groups_both_mean_absent():
    features = examples(False)
    features[0]["doc_id"] = ["", ""]
    del features[1]["doc_id"]
    result = processor().process_queries_documents_biencoder(features)
    assert "passage_doc_ids" not in result


def test_mixed_present_and_all_empty_id_groups_fail_clearly():
    features = examples(False)
    features[1]["doc_id"] = ["", ""]
    with pytest.raises(ValueError, match="same present or absent ID policy"):
        processor().process_queries_documents_biencoder(features)


def distributed_mask_worker(rank, world_size, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=45)
    )
    try:
        features = examples(True)[rank : rank + 1]
        batch = ProcessorMethodCollator(processor(), "process_queries_documents_biencoder")(features)
        ids = dist_gather_tensor(batch["passage_doc_ids"].contiguous())
        logits = torch.tensor([[10.0, 0.0] * world_size], requires_grad=True)
        scores = logits * 1.0
        mask_gathered_passages_same_doc_as_positive(scores, ids, 2, rank, 1)
        target = torch.tensor([rank * 2])
        loss = torch.nn.functional.cross_entropy(scores, target)
        reference = torch.tensor([[10.0] + [0.0] * world_size], requires_grad=True)
        expected = torch.nn.functional.cross_entropy(reference, torch.tensor([0]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        expected.backward()
        assert torch.isfinite(logits.grad).all()
        torch.testing.assert_close(logits.grad[0, rank * 2], reference.grad[0, 0])
        torch.testing.assert_close(logits.grad[0, 1::2], reference.grad[0, 1:])
        for other_rank in range(world_size):
            if other_rank != rank:
                assert logits.grad[0, other_rank * 2] == 0
                assert scores[0, other_rank * 2] == torch.finfo(scores.dtype).min
        assert scores[0, rank * 2] == 10
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.runtime_budget(25, reason="spawns real Gloo ranks and reimports PyTorch in each process")
def test_native_ids_enable_duplicate_masking_across_real_processes(tmp_path, world_size):
    mp.spawn(distributed_mask_worker, args=(world_size, (tmp_path / "gloo").as_uri()), nprocs=world_size, join=True)


class _PrecomputedEncoder(torch.nn.Module):
    pooling = "avg"
    l2_normalize = False
    do_distributed_inbatch_negative = True

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return precomputed embeddings.

        Args:
            inputs: Mapping with ``embeddings`` of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, hidden].
        """
        return inputs["embeddings"]


def mixed_rank_id_policy_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        recipe = TrainBiEncoderRecipe(ConfigNode({}))
        recipe.model_parts = [_PrecomputedEncoder()]
        recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
        recipe.distributed_config = SimpleNamespace()
        recipe.train_n_passages = 2
        batch = {
            "q_embeddings": torch.tensor([[1.0, 0.0]]),
            "d_embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }
        if rank == 0:
            batch["passage_doc_ids"] = torch.tensor([1, 2])
        with pytest.raises(ValueError, match="Every rank must use the same present or absent passage_doc_ids policy"):
            recipe._forward_backward_step(0, batch, loss_buffer=[], num_batches=1, is_train=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(25, reason="spawns real Gloo ranks to verify mixed ID policy fails symmetrically")
def test_mixed_rank_id_policy_fails_without_hanging(tmp_path):
    mp.spawn(mixed_rank_id_policy_worker, args=((tmp_path / "gloo_mixed").as_uri(),), nprocs=2, join=True)

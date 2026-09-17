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

from datetime import timedelta

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

from nemo_automodel.components.datasets.llm.retrieval_collator import ProcessorMethodCollator
from nemo_automodel.components.models.common.inbatch_neg_utils import (
    dist_gather_tensor,
    mask_gathered_passages_same_doc_as_positive,
)
from nemo_automodel.components.models.ministral_bidirectional.processor import Mistral3BiEncoderProcessor
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
    with pytest.raises(ValueError, match="one nonempty string per candidate"):
        processor().process_queries_documents_biencoder(features)


def test_mixed_id_policy_is_not_silently_disabled():
    features = examples(False)
    del features[1]["doc_id"]
    with pytest.raises(ValueError, match="every example"):
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
def test_native_ids_enable_duplicate_masking_across_real_processes(tmp_path, world_size):
    mp.spawn(distributed_mask_worker, args=(world_size, (tmp_path / "gloo").as_uri()), nprocs=world_size, join=True)

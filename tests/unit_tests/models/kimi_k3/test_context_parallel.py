# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.kimi_k3.cp import (
    KimiPackedContext,
    _document_causal_block_mask,
    shard_batch_for_kimi_cp,
)
from nemo_automodel.components.models.kimi_k3.model import (
    KimiDeltaAttention,
    KimiK3ForCausalLM,
    KimiMLAAttention,
)


class _FakeCPMesh:
    def size(self):
        return 2

    def get_local_rank(self):
        return 1


def test_prepare_model_inputs_returns_current_cp_sharder_api():
    model = object.__new__(KimiK3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.cp_mesh = _FakeCPMesh()
    mla = object.__new__(KimiMLAAttention)
    torch.nn.Module.__init__(mla)
    mla._cp_mesh = None
    kda = object.__new__(KimiDeltaAttention)
    torch.nn.Module.__init__(kda)
    kda._cp_mesh = None
    model.add_module("mla", mla)
    model.add_module("kda", kda)

    prepared = model.prepare_model_inputs_for_cp({}, num_chunks=1)

    assert set(prepared) == {"cp_sharder"}
    assert isinstance(prepared["cp_sharder"], ContextParallelSharder)
    assert prepared["cp_sharder"].shard_batch is shard_batch_for_kimi_cp
    assert mla._cp_mesh is model.cp_mesh
    assert kda._cp_mesh is model.cp_mesh


def test_kimi_cp_sharder_keeps_contiguous_tokens_and_global_document_map():
    batch = {
        "input_ids": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    _, local_batch, layout = shard_batch_for_kimi_cp(
        _FakeCPMesh(),
        None,
        batch,
        padding_token_id=99,
    )

    assert local_batch["input_ids"].tolist() == [[4, 5, 6, 7]]
    assert local_batch["labels"].tolist() == [[4, 5, 6, 7]]
    assert local_batch["position_ids"].tolist() == [[4, 5, 6, 7]]
    assert local_batch["padding_mask"].tolist() == [[False, False, True, True]]
    assert "attention_mask" not in local_batch
    assert local_batch["kimi_packed_doc_ids"].tolist() == [[1, 1, 1, 1, 1, 1, 0, 0]]
    assert local_batch["kimi_packed_seq_start"] == 4
    assert local_batch["kimi_packed_cp_size"] == 2
    assert layout.original_seq_len == 8
    assert layout.padded_seq_len == 8


def test_kimi_cp_sharder_moves_masked_holes_behind_valid_tokens():
    """A masked span inside a document must not split it into separate CP segments."""
    batch = {
        "input_ids": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(10, 18).unsqueeze(0),
        "attention_mask": torch.tensor([[1, 1, 0, 0, 1, 1, 1, 0]]),
    }
    loss_mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0, 0]])

    _, local_batch, _ = shard_batch_for_kimi_cp(_FakeCPMesh(), None, batch, loss_mask=loss_mask)

    # Valid tokens 0, 1, 4, 5, 6 keep their order; masked tokens 2, 3, 7 trail them.
    assert local_batch["kimi_packed_doc_ids"].tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]
    assert local_batch["input_ids"].tolist() == [[6, 2, 3, 7]]
    assert local_batch["labels"].tolist() == [[16, 12, 13, 17]]
    assert local_batch["loss_mask"].tolist() == [[0, 0, 0, 0]]
    assert local_batch["padding_mask"].tolist() == [[False, True, True, True]]


def test_kimi_cp_sharder_consumes_dataset_packing_metadata():
    batch = {
        "input_ids": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attention_mask": torch.ones(1, 1, 8, 8, dtype=torch.bool),
        "_packed_seq_ids": torch.tensor([[1, 1, 1, 2, 2, 2, 0, 0]]),
        "packed_token_indices": torch.arange(6),
        "cu_seqlens": torch.tensor([0, 3, 6], dtype=torch.int32),
        "max_seqlen": 3,
    }

    _, local_batch, _ = shard_batch_for_kimi_cp(_FakeCPMesh(), None, batch)

    assert local_batch["kimi_packed_doc_ids"].tolist() == [[1, 1, 1, 2, 2, 2, 0, 0]]
    assert not {
        "attention_mask",
        "_packed_seq_ids",
        "packed_token_indices",
        "cu_seqlens",
        "max_seqlen",
    } & set(local_batch)


def test_pipeline_microbatches_chunk_kimi_document_map():
    batch = {
        "input_ids": torch.arange(16).reshape(2, 8),
        "labels": torch.arange(16).reshape(2, 8),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ]
        ),
    }
    _, local_batch, _ = shard_batch_for_kimi_cp(_FakeCPMesh(), None, batch)

    _, microbatches = split_args_kwargs_into_chunks((), local_batch, chunks=2)

    assert len(microbatches) == 2
    assert [microbatch["input_ids"].shape for microbatch in microbatches] == [(1, 4), (1, 4)]
    assert [microbatch["kimi_packed_doc_ids"].shape for microbatch in microbatches] == [(1, 8), (1, 8)]
    assert microbatches[0]["kimi_packed_doc_ids"].tolist() == [[1, 1, 1, 1, 1, 1, 0, 0]]
    assert microbatches[1]["kimi_packed_doc_ids"].tolist() == [[1, 1, 1, 1, 0, 0, 0, 0]]


def test_packed_context_ignores_masked_gaps_when_checking_document_runs():
    """A masked hole inside a document is not a repeated document; interleaving still is."""
    context = KimiPackedContext(torch.tensor([[1, 1, 0, 1, 1, 2, 2]]))
    assert context.has_multiple_documents
    with pytest.raises(ValueError, match="one contiguous run"):
        KimiPackedContext(torch.tensor([[1, 2, 1]]))
    with pytest.raises(ValueError, match="one contiguous run"):
        KimiPackedContext(torch.tensor([[1, 1, 2, 2, 0, 1]]))


def test_indexed_mask_with_masked_gap_trains_through_the_model():
    """The direct indexed-mask forward must accept a masked hole inside a document."""
    from tests.unit_tests.models.kimi_k3.test_multimodal import _tiny_vlm

    torch.manual_seed(0)
    model = _tiny_vlm()
    input_ids = torch.tensor([[5, 6, 7, 8, 9, 10, 11]])
    attention_mask = torch.tensor([[1, 1, 0, 1, 1, 2, 2]])
    labels = torch.where(attention_mask > 0, input_ids, torch.full_like(input_ids, -100))

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    valid = labels != -100
    loss = torch.nn.functional.cross_entropy(logits[valid], labels[valid])
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for name, p in model.named_parameters() if not name.startswith("vision_tower.")]
    assert any(grad is not None for grad in grads)
    assert all(grad is None or torch.isfinite(grad).all() for grad in grads)
    # Document 2 cannot read document 1: perturbing document 1 leaves its logits unchanged.
    with torch.no_grad():
        perturbed_ids = input_ids.clone()
        perturbed_ids[0, 0] = 12
        perturbed = model(input_ids=perturbed_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(perturbed[0, 5:], logits[0, 5:].detach(), rtol=0, atol=0)
    assert not torch.allclose(perturbed[0, 3:5], logits[0, 3:5].detach())


def test_mla_masks_remain_owned_by_their_microbatch():
    """Interleaving document layouts must retain each microbatch's original mask."""
    first = KimiPackedContext(torch.tensor([[1, 1, 1, 1, 1, 2, 2, 0]]), seq_start=4, cp_size=2)
    second = KimiPackedContext(torch.tensor([[1, 1, 2, 2, 2, 2, 0, 0]]), seq_start=4, cp_size=2)
    first_mask = first.mla_block_mask
    second_mask = second.mla_block_mask
    assert first.mla_block_mask is first_mask
    assert second_mask is not first_mask
    q = torch.arange(4)[:, None]
    k = torch.arange(8)[None, :]
    for context, mask in ((first, first_mask), (second, second_mask)):
        query_docs = context.doc_ids[0, 4:, None]
        key_docs = context.doc_ids[0, None, :]
        expected = (k <= q + 4) & (query_docs == key_docs) & (key_docs > 0)
        expected = torch.where(query_docs == 0, k == 0, expected)
        actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), q, k)
        torch.testing.assert_close(actual, expected)


def test_interleaved_mla_masks_preserve_selective_checkpoint_operations():
    """A second microbatch must not add metadata operations during recomputation."""
    from functools import partial

    from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

    first = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 0]])
    second = torch.tensor([[1, 1, 2, 2, 2, 2, 0, 0]])
    q = torch.arange(4)[:, None]
    k = torch.arange(8)[None, :]
    zero = torch.tensor(0)

    def build(documents):
        """Build this test's trailing-shard attention mask.

        Args:
            documents: Tensor of shape [batch, global_sequence], with global_sequence=8.

        Returns:
            BlockMask for [batch, local_sequence=4] queries against
            [batch, global_sequence=8] keys.
        """
        return _document_causal_block_mask(documents[:, 4:], documents, q_global_start=4)

    def policy(context, operation, *args, **kwargs):
        if operation in (torch.ops.aten.arange.default, torch.ops.aten.arange.start, torch.ops.aten.arange.start_step):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def region(value):
        """Reduce the checkpointed query values.

        Args:
            value: Tensor of shape [local_sequence], with local_sequence=4.

        Returns:
            Scalar tensor of shape [] containing the weighted sine reduction.
        """
        mask = build(first)
        weights = mask.mask_mod(zero, zero, q, k).float().sum(-1)
        return (value.sin() * weights).sum()

    build(first)
    value = torch.linspace(-0.7, 0.9, 4, requires_grad=True)
    loss = checkpoint(
        region, value, use_reentrant=False, context_fn=partial(create_selective_checkpoint_contexts, policy)
    )
    build(second)
    loss.backward()
    # Local positions 4,5,6,7 can read respectively 5,1,2,1 keys.
    expected = value.detach().cos() * torch.tensor([5.0, 1.0, 2.0, 1.0])
    torch.testing.assert_close(value.grad, expected)

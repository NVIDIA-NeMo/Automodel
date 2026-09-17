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

"""Packed forwards must equal independent documents, including gradients and CP."""

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset
from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater
from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.models.deepseek_v41.cp import gather_sequence, shard_cp_batch
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.models.deepseek_v41.packing import packed_layout
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.deepseek_v41.test_engram import _tokenizer
from tests.unit_tests.models.deepseek_v41.test_model import _backend, _tiny_config


def _model(*, full_schedule=False, owner_group=None):
    config = _tiny_config()
    text = config.text_config
    text.initializer_range = 0.08
    text.sliding_window = 4
    text.engram_layer_ids = [1, 3]
    text.engram_num_embeddings = [113, 253]
    text.engram_vocab_size = 11
    text.engram_compressed_vocab_size = 6
    text.engram_max_ngram_size = 4
    text.engram_n_heads = 2
    text.engram_head_dim = 4
    text.engram_pad_token_id = 2
    if full_schedule:
        text.num_hidden_layers = 40
        text.compress_ratios = [0, 0] + [2] * 18 + [1] * 20
        text.kv_source_layer_ids = [2, 8, 14, 20]
        text.index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
        text.candidate_source_layer_id = 20
        text.candidate_block_size = 8
        text.engram_layer_ids = [1, 14]
    model = DeepseekV41ForCausalLM(config, backend=_backend(), tokenizer=_tokenizer(), engram_process_group=owner_group)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model


def _documents():
    return [
        torch.tensor([[7]]),
        torch.tensor([[1, 4, 7]]),
        torch.tensor([[7, 4, 3, 8, 7, 5, 6, 1, 2]]),
        torch.tensor([[3, 7, 4, 5, 8, 2, 1, 7, 4, 3, 5, 1, 7, 4, 2, 8, 5]]),
    ]


@pytest.mark.parametrize("full_schedule", [False, True])
def test_packed_matches_independent_documents_and_optimizer(full_schedule):
    torch.set_num_threads(1)
    torch.manual_seed(928)
    model = _model(full_schedule=full_schedule)
    reference = copy.deepcopy(model)
    documents = _documents()
    ids = torch.cat(documents, dim=1)
    lengths = torch.tensor([[doc.shape[1] for doc in documents]])
    result = model(ids, seq_lens=lengths, labels=ids, output_hidden_states=True)
    expected_outputs = [reference(doc, output_hidden_states=True) for doc in documents]
    expected_logits = torch.cat([out.logits for out in expected_outputs], dim=1)
    torch.testing.assert_close(result.logits, expected_logits, atol=3e-6, rtol=3e-5)
    for layer, hidden in enumerate(result.hidden_states):
        expected = torch.cat([out.hidden_states[layer] for out in expected_outputs], dim=1)
        torch.testing.assert_close(hidden, expected, atol=3e-6, rtol=3e-5)
    expected_loss = sum(
        F.cross_entropy(out.logits[:, :-1].flatten(0, 1), doc[:, 1:].flatten(), reduction="sum")
        for doc, out in zip(documents, expected_outputs, strict=True)
        if doc.shape[1] > 1
    ) / sum(doc.shape[1] - 1 for doc in documents)
    torch.testing.assert_close(result.loss, expected_loss, atol=2e-6, rtol=2e-6)
    result.loss.backward()
    expected_loss.backward()
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(
                actual.grad, expected.grad, atol=3e-6, rtol=5e-4, msg=lambda detail: f"{name}: {detail}"
            )
    actual_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
    torch.testing.assert_close(actual_norm, expected_norm, atol=3e-6, rtol=3e-5)
    for instance in (model, reference):
        torch.optim.SGD(instance.parameters(), lr=0.01).step()
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5, msg=lambda detail: f"{name}: {detail}")


def test_packed_document_isolation_and_engram_hashes():
    torch.set_num_threads(1)
    torch.manual_seed(641)
    model = _model(full_schedule=True)
    documents = _documents()
    ids = torch.cat(documents, dim=1)
    lengths = torch.tensor([[doc.shape[1] for doc in documents]])
    baseline = model(ids, seq_lens=lengths).logits
    modified = ids.clone()
    modified[:, :4] = 6
    actual = model(modified, seq_lens=lengths).logits
    torch.testing.assert_close(actual[:, 4:], baseline[:, 4:], atol=2e-6, rtol=2e-5)
    embedded = []

    def retain_embedding(module, args, output):
        """Retain output [batch, aligned_sequence, hidden]; args contains integer input_ids [batch, aligned_sequence]."""
        output.retain_grad()
        embedded.append(output)

    handle = model.model.embed_tokens.register_forward_hook(retain_embedding)
    try:
        output = model(ids, seq_lens=lengths).logits
        torch.manual_seed(991)
        output[:, 4:13].backward(torch.randn_like(output[:, 4:13]))
    finally:
        handle.remove()
    layout = packed_layout(
        lengths, seq_lens_padded=None, input_shape=tuple(ids.shape), alignment=model._packed_alignment
    )
    grad = layout.restore(embedded[0].grad)
    assert torch.count_nonzero(grad[:, :4]) == 0
    assert torch.count_nonzero(grad[:, 13:]) == 0
    assert torch.count_nonzero(grad[:, 4:13]) > 0
    packed_ids = layout.pack(ids)
    hashes = model.model.engram_hash(packed_ids, token_mask=layout.sequence_ids > 0, sequence_ids=layout.sequence_ids)
    expected_hashes = torch.cat([model.model.engram_hash(doc) for doc in documents], dim=1)
    torch.testing.assert_close(layout.restore(hashes), expected_hashes, atol=0, rtol=0)


def test_real_packer_collator_keeps_shifted_targets_and_token_coordinates():
    documents = _documents()
    examples = [{"input_ids": doc[0].tolist(), "labels": doc[0, 1:].tolist() + [-100]} for doc in documents]
    packs = pack_dataset(copy.deepcopy(examples), split=None, packed_sequence_size=32, padding_idx=2, cp_size=1)
    batch = packed_sequence_thd_collater([packs[0]])
    original = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    _, aligned, layout = shard_cp_batch(None, None, batch, pad_multiple=2, packed_alignment=8)
    expected_ids = torch.cat(documents, dim=1)
    expected_labels = torch.tensor([sum([item["labels"] for item in examples], [])])
    positions = layout.input_token_stream_positions[:, : expected_ids.shape[1]]
    torch.testing.assert_close(aligned["input_ids"].gather(1, positions), expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(aligned["labels"].gather(1, positions), expected_labels, atol=0, rtol=0)
    assert (aligned["labels"] != -100).sum() == (original["labels"] != -100).sum()
    for index, length in enumerate([1, 3, 9, 17], 1):
        selected = aligned["packed_seq_ids"] == index
        torch.testing.assert_close(aligned["position_ids"][selected], torch.arange(length), atol=0, rtol=0)


def _cp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=150)
    )
    try:
        owners = [dist.new_group([owner]) for owner in range(2)]
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("cp",))
        for mostly_padding in (False, True):
            for ac in (False, True):
                torch.manual_seed(392)
                model = _model(owner_group=owners[rank])
                reference = _model(owner_group=owners[rank])
                reference.load_state_dict(model.state_dict())
                owner_mesh = DeviceMesh.from_group(owners[rank], "cpu", mesh_dim_names=("dp_shard_cp",))
                model._nemo_prepare_model_owned_dtensors(owner_mesh)
                reference._nemo_prepare_model_owned_dtensors(owner_mesh)
                if ac:
                    apply_ac(model)
                documents = [torch.tensor([[7, 1, 4]])] if mostly_padding else _documents()
                ids = torch.cat(documents, dim=1)
                lengths = torch.tensor([[doc.shape[1] for doc in documents]])
                # Loss is already shifted per document, as in the production recipe.
                targets = torch.cat([F.pad(doc[:, 1:], (0, 1), value=-100) for doc in documents], dim=1)
                if mostly_padding:
                    ids = F.pad(ids, (0, 32 - ids.shape[1]), value=2)
                    targets = F.pad(targets, (0, 32 - targets.shape[1]), value=-100)
                batch = {"input_ids": ids, "labels": targets, "seq_lens": lengths, "qkv_format": "thd"}
                _, local, layout = shard_cp_batch(
                    mesh, None, batch, pad_multiple=2, packed_alignment=model._packed_alignment
                )
                labels = local.pop("labels")
                local_logits = model(**local).logits
                expected_logits = reference(ids, seq_lens=lengths).logits
                full_logits = gather_sequence(local_logits.detach(), dist.group.WORLD)
                original_positions = layout.input_token_stream_positions
                restored = full_logits.gather(
                    1, original_positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, full_logits.shape[-1])
                )
                restored = restored.masked_fill((original_positions < 0).unsqueeze(-1), 0)
                torch.testing.assert_close(restored, expected_logits, atol=3e-6, rtol=3e-5)
                expected_loss = F.cross_entropy(expected_logits.flatten(0, 1), targets.flatten(), reduction="sum")
                actual_loss = F.cross_entropy(local_logits.flatten(0, 1), labels.flatten(), reduction="sum")
                actual_loss.backward()
                expected_loss.backward()
                for (name, actual), (_, expected) in zip(
                    model.named_parameters(), reference.named_parameters(), strict=True
                ):
                    if expected.grad is None:
                        assert actual.grad is None, name
                    else:
                        assert actual.grad is not None, name
                        actual_grad = actual.grad.to_local() if isinstance(actual.grad, DTensor) else actual.grad
                        expected_grad = (
                            expected.grad.to_local() if isinstance(expected.grad, DTensor) else expected.grad
                        )
                        dist.all_reduce(actual_grad)
                        torch.testing.assert_close(
                            actual_grad, expected_grad, atol=2e-5, rtol=5e-4, msg=lambda detail: f"{name}: {detail}"
                        )
                dist.all_reduce(actual_loss)
                torch.testing.assert_close(actual_loss, expected_loss, atol=1e-5, rtol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    45,
    hard_timeout=60,
    reason="two spawned workers import the model stack and compile packed CP forward/backward with checkpointing",
)
def test_packed_cp2_forward_backward_with_activation_checkpointing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Other test modules disable compilation at collection time. Spawned workers
    # need the default compiler state for the fullgraph expert activation.
    with monkeypatch.context() as worker_env:
        worker_env.delenv("TORCH_COMPILE_DISABLE", raising=False)
        mp.spawn(_cp_worker, args=(str(tmp_path / "packed-gloo"),), nprocs=2, join=True)


@pytest.mark.parametrize("lengths,spans", [([3], [2]), ([33], [33]), ([-2], [-2]), ([0], [4])])
def test_invalid_packed_metadata_fails_before_attention(lengths, spans):
    with pytest.raises(ValueError):
        packed_layout(torch.tensor([lengths]), seq_lens_padded=torch.tensor([spans]), input_shape=(1, 32), alignment=8)


def test_document_alignment_counts_inside_fixed_pack_budget():
    examples = [
        {"input_ids": [1, 4, 7], "labels": [4, 7, -100]},
        {"input_ids": [7, 3, 5, 2, 8], "labels": [3, 5, 2, 8, -100]},
        {"input_ids": [2], "labels": [-100]},
    ]
    packs = pack_dataset(
        copy.deepcopy(examples), split=None, packed_sequence_size=8, padding_idx=2, pad_to_multiple_of=2
    )
    assert len(packs) == 2
    assert all(len(pack["input_ids"]) == 8 for pack in packs)
    assert list(packs[0]["seq_lens"]) == [3]
    assert list(packs[1]["seq_lens"]) == [5, 1]
    assert list(packs[1]["input_ids"]) == [7, 3, 5, 2, 8, 2, 2, 2]
    assert sum(int((torch.as_tensor(pack["labels"]) != -100).sum()) for pack in packs) == 6


def test_packed_empty_supervision_has_zero_loss_and_zero_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(524)
    model = _model()
    ids = torch.tensor([[7, 1, 4]])
    output = model(ids, seq_lens=torch.tensor([[1, 2]]), labels=torch.full_like(ids, -100))
    assert output.loss.item() == 0
    output.loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("padding_token_id", [None, 2])
def test_context_parallel_sharder_prepares_packed_cp1_without_a_mesh(padding_token_id):
    torch.set_num_threads(1)
    model = _model()
    # Native model-owned THD dispatch is the same capability forwarded by the AutoModel wrapper.
    from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate

    attach_capabilities_and_validate(model, None)
    documents = _documents()
    ids = torch.cat(documents, dim=1)
    labels = torch.cat([F.pad(doc[:, 1:], (0, 1), value=-100) for doc in documents], dim=1)
    batch = dict(input_ids=ids, labels=labels, seq_lens=torch.tensor([[1, 3, 9, 17]]), qkv_format="thd")
    sharder = ContextParallelSharder(model, None, batch, padding_token_id=padding_token_id)
    _, prepared = sharder.shard(batch)
    assert "seq_lens" not in prepared and "packed_seq_ids" in prepared
    assert prepared["input_ids"].shape == (1, 34)
    torch.testing.assert_close(
        sharder.gather_token_tensor(prepared["labels"], seq_dim=1, trim=True, fill=-100), labels, atol=0, rtol=0
    )

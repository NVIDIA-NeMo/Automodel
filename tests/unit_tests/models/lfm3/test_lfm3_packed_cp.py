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

"""LFM3 on packed documents and on context-parallel shards, against each document run alone."""

import copy
import queue
import threading

import pytest
import torch

from nemo_automodel.components.distributed.activation_checkpointing import apply_full_layer_checkpointing_to_layers
from nemo_automodel.components.models.lfm3 import context_parallel
from nemo_automodel.components.models.lfm3 import model as lfm3_model
from nemo_automodel.components.models.lfm3.config import Lfm3Config
from nemo_automodel.components.models.lfm3.context_parallel import _halo_sources, shard_documents
from nemo_automodel.components.models.lfm3.model import Lfm3ForCausalLM

# Document lengths of the two packed rows; each row ends with padding.
ROWS = ((37, 5, 60, 3, 21), (9, 70, 1, 46))
SEQUENCE = 136
WINDOW = 8
TOLERANCE = {"atol": 2e-5, "rtol": 1e-4}


def _model(attn_implementation="sdpa"):
    torch.manual_seed(0)
    config = Lfm3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=7,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=WINDOW,
        conv_bias=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=["conv", "conv", "full_attention", "conv", "sliding_attention", "full_attention", "conv"],
    )
    with torch.device("cpu"):
        model = Lfm3ForCausalLM(config).float().eval()
    model.set_attn_implementation(attn_implementation)
    return model


def _thd_batch(rows=ROWS):
    """Build a THD batch: token Tensors of shape [rows, SEQUENCE] whose documents have the given lengths.

    Each row ends with padding (token 0, label -100). ``seq_lens`` and ``seq_lens_padded`` have shape
    [rows, 5], padded with -1000; the last padded span absorbs the row's padding.
    """
    generator = torch.Generator().manual_seed(1)
    ids = torch.zeros(len(rows), SEQUENCE, dtype=torch.long)
    positions = torch.zeros_like(ids)
    for row, lengths in enumerate(rows):
        ids[row, : sum(lengths)] = torch.randint(3, 97, (sum(lengths),), generator=generator)
        spans = [*lengths[:-1], SEQUENCE - sum(lengths[:-1])]
        positions[row] = torch.cat([torch.arange(span) for span in spans])
    return {
        "input_ids": ids,
        "labels": ids.masked_fill(ids == 0, -100),
        "position_ids": positions,
        "seq_lens": torch.tensor([[*lengths, *[-1000] * (5 - len(lengths))] for lengths in rows]),
        "seq_lens_padded": torch.tensor(
            [[*lengths[:-1], SEQUENCE - sum(lengths[:-1]), *[-1000] * (5 - len(lengths))] for lengths in rows]
        ),
        "qkv_format": "thd",
    }


def _packed_logits(model, batch, cp_mesh=None):
    """Shard the batch for this rank and run the model; returns logits [rows, local sequence, vocab] and labels."""
    _, batch, _ = shard_documents(cp_mesh, None, batch)
    labels = batch.pop("labels")
    return model(**batch).logits, labels


def _documents(batch, rows=ROWS):
    """Yield (row, start, end) of every document."""
    for row, lengths in enumerate(rows):
        start = 0
        for length in lengths:
            yield row, start, start + length
            start += length


class _Mesh:
    """One CP rank's view of a mesh whose ranks are threads of this process."""

    def __init__(self, size, rank, group=None):
        self._size, self._rank, self._group = size, rank, group

    def size(self):
        return self._size

    def get_local_rank(self):
        return self._rank

    def get_group(self):
        return self._group


class _ThreadGroup:
    """Stands in for the CP process group: a queue for every ordered pair of ranks."""

    def __init__(self, size):
        self.wires = {(source, target): queue.Queue() for source in range(size) for target in range(size)}
        self.rank = threading.local()


def _transfer_between_threads(group, sends, receives):
    """Stand-in for the point-to-point transfer: pieces of shape [batch, channels, tokens, ...] move through queues."""
    for rank, piece in sends:
        group.wires[group.rank.value, rank].put(piece.detach().clone())
    for rank, piece in receives:
        piece.copy_(group.wires[rank, group.rank.value].get(timeout=60))


def _run_ranks(model, batch, size):
    """Run every CP rank in its own thread and return per-rank logits and summed parameter gradients."""
    group = _ThreadGroup(size)
    logits, gradients, errors = [None] * size, [None] * size, []

    def run(rank):
        try:
            group.rank.value = rank
            replica = copy.deepcopy(model)
            local_logits, labels = _packed_logits(replica, dict(batch), _Mesh(size, rank, group))
            torch.nn.functional.cross_entropy(local_logits.flatten(0, 1), labels.flatten(), reduction="sum").backward()
            logits[rank] = local_logits.detach()
            gradients[rank] = {name: parameter.grad for name, parameter in replica.named_parameters()}
        except BaseException as error:  # noqa: BLE001 - reported by the test thread
            errors.append(error)

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    return torch.cat(logits, dim=1), {name: sum(gradient[name] for gradient in gradients) for name in gradients[0]}


@pytest.fixture
def thread_ranks(monkeypatch):
    monkeypatch.setattr(context_parallel, "_transfer", _transfer_between_threads)
    monkeypatch.setattr(context_parallel.dist, "get_rank", lambda group: group.rank.value)


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_packed_rows_match_documents_run_alone(attn_implementation):
    model, batch = _model(attn_implementation), _thd_batch()
    with torch.no_grad():
        logits, _ = _packed_logits(model, dict(batch))
        for row, start, end in _documents(batch):
            alone = model(input_ids=batch["input_ids"][row : row + 1, start:end]).logits[0]
            torch.testing.assert_close(logits[row, start:end], alone, **TOLERANCE)


def _convolve_row(conv, gated, documents):
    """The convolution without document boundaries."""
    return conv(gated)[..., : gated.shape[-1]]


def _attend_row(documents, key, value, sliding_window, attn_implementation):
    """Causal (and windowed) attention inputs without document boundaries."""
    position = torch.arange(key.shape[2])
    allowed = position <= position[:, None]
    if sliding_window is not None:
        allowed &= position > position[:, None] - sliding_window
    return key, value, allowed[None, None], {}


@pytest.mark.parametrize(
    ("handler", "without_boundaries"),
    [("convolve_documents", _convolve_row), ("document_attention_inputs", _attend_row)],
)
def test_dropping_the_boundaries_of_one_layer_type_leaks_into_later_documents(monkeypatch, handler, without_boundaries):
    model, batch = _model(), _thd_batch()
    monkeypatch.setattr(lfm3_model, handler, without_boundaries)
    with torch.no_grad():
        logits, _ = _packed_logits(model, dict(batch))
        errors = [
            (logits[row, start:end] - model(input_ids=batch["input_ids"][row : row + 1, start:end]).logits[0])
            .abs()
            .max()
            for row, start, end in _documents(batch)
        ]
    # The first document of each row has nothing before it; every later one reads its predecessors.
    first = [index for index, (_, start, _) in enumerate(_documents(batch)) if start == 0]
    assert all(errors[index] < 1e-4 for index in first)
    assert all(error > 1e-2 for index, error in enumerate(errors) if index not in first)


def test_changing_one_document_leaves_every_other_document_unchanged():
    model, batch = _model(), _thd_batch()
    changed = {**batch, "input_ids": batch["input_ids"].clone()}
    changed["input_ids"][0, 42:102] = (changed["input_ids"][0, 42:102] + 1) % 94 + 3  # the third document of row 0
    with torch.no_grad():
        before, _ = _packed_logits(model, dict(batch))
        after, _ = _packed_logits(model, changed)
    for row, start, end in _documents(batch):
        if (row, start) == (0, 42):
            assert not torch.equal(before[row, start:end], after[row, start:end])
        else:
            assert torch.equal(before[row, start:end], after[row, start:end])


@pytest.mark.parametrize("size", [2, 3, 4, 8])
@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_context_parallel_shards_match_one_process(thread_ranks, size, attn_implementation):
    # The 60-token document spans up to five shards; the padding tail fills the last one.
    model, batch = _model(attn_implementation).train(), _thd_batch(ROWS[:1])
    expected, labels = _packed_logits(model, dict(batch))
    torch.nn.functional.cross_entropy(expected.flatten(0, 1), labels.flatten(), reduction="sum").backward()

    logits, gradients = _run_ranks(model, batch, size)

    torch.testing.assert_close(logits[:, :SEQUENCE], expected, **TOLERANCE)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(gradients[name], parameter.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("size", [1, 2, 4])
def test_checkpointed_layers_give_the_gradients_of_plain_layers(thread_ranks, size):
    model, batch = _model().train(), _thd_batch(ROWS[:1])
    expected, labels = _packed_logits(model, dict(batch))
    torch.nn.functional.cross_entropy(expected.flatten(0, 1), labels.flatten(), reduction="sum").backward()
    checkpointed = copy.deepcopy(model)
    checkpointed.zero_grad()
    # What `distributed.activation_checkpointing: true` does for this model: every layer, convolutions included.
    assert Lfm3ForCausalLM.parallelizer._use_full_layer_activation_checkpointing(checkpointed)
    apply_full_layer_checkpointing_to_layers(checkpointed, list(checkpointed.model.layers))

    logits, gradients = _run_ranks(checkpointed, batch, size)

    torch.testing.assert_close(logits[:, :SEQUENCE], expected, **TOLERANCE)
    plain = {name.replace("._checkpoint_wrapped_module", ""): gradient for name, gradient in gradients.items()}
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(plain[name], parameter.grad, atol=1e-4, rtol=1e-3)


def test_unpacked_row_on_context_parallel_shards_matches_the_plain_forward(thread_ranks):
    model = _model()
    ids = _thd_batch(ROWS[:1])["input_ids"]
    real = sum(ROWS[0])
    with torch.no_grad():
        expected = model(input_ids=ids[:, :real]).logits
    batch = {"input_ids": ids, "labels": ids.masked_fill(ids == 0, -100), "attention_mask": (ids != 0).long()}

    logits, _ = _run_ranks(model.train(), batch, 4)

    torch.testing.assert_close(logits[:, :real], expected, **TOLERANCE)


def test_context_parallel_shards_differ_from_one_process_without_the_halo(thread_ranks, monkeypatch):
    model, batch = _model(), _thd_batch(ROWS[:1])
    with torch.no_grad():
        expected, _ = _packed_logits(model, dict(batch))
    sources = context_parallel._halo_sources
    monkeypatch.setattr(context_parallel, "_halo_sources", lambda lengths, *rest: sources((0,) * len(lengths), *rest))

    logits, _ = _run_ranks(model.train(), batch, 4)

    # Shard 0 has no halo; shards 1 and 2 start inside the 60-token document.
    torch.testing.assert_close(logits[:, :34], expected[:, :34], **TOLERANCE)
    assert (logits[:, 34:102] - expected[:, 34:102]).abs().max() > 1e-2


def test_shards_keep_their_tokens_and_describe_their_documents():
    batch = _thd_batch(ROWS[:1])
    for rank, (boundaries, longest) in enumerate([((0, 34), 34), ((0, 3, 8, 34), 26), ((0, 34), 34), ((0, 3, 34), 31)]):
        _, shard, layout = shard_documents(_Mesh(4, rank), None, dict(batch))
        assert torch.equal(shard["input_ids"], batch["input_ids"][:, rank * 34 : (rank + 1) * 34])
        assert torch.equal(shard["position_ids"], batch["position_ids"][:, rank * 34 : (rank + 1) * 34])
        assert shard["cu_seq_lens_q"].tolist() == list(boundaries) and shard["cu_seq_lens_q"].dtype == torch.int32
        assert shard["max_length_q"] == longest
        assert shard["halo"].lengths == (0, 34, 26, 0)
        assert (layout.original_seq_len, layout.padded_seq_len) == (SEQUENCE, SEQUENCE)
        assert not {"qkv_format", "seq_lens", "seq_lens_padded", "padding_mask"} & set(shard)


def test_batch_packed_for_flash_attention_keeps_its_boundaries_on_shards():
    thd = _thd_batch(ROWS[:1])
    boundaries = [0, *torch.tensor(ROWS[0][:-1]).cumsum(0).tolist(), SEQUENCE]
    flash = {
        "input_ids": thd["input_ids"],
        "labels": thd["labels"],
        "position_ids": thd["position_ids"],
        "cu_seq_lens_q": torch.tensor(boundaries, dtype=torch.int32),
        "max_length_q": 60,
    }
    for rank in range(4):
        _, expected, _ = shard_documents(_Mesh(4, rank), None, dict(thd))
        _, shard, _ = shard_documents(_Mesh(4, rank), None, dict(flash))
        assert torch.equal(shard["cu_seq_lens_q"], expected["cu_seq_lens_q"])
        assert (shard["max_length_q"], shard["halo"].lengths) == (expected["max_length_q"], expected["halo"].lengths)


def test_halo_comes_from_every_earlier_shard_a_document_spans():
    lengths = (0, 34, 26, 0, 136)
    assert _halo_sources(lengths, 34, rank=0, limit=None) == []
    assert _halo_sources(lengths, 34, rank=1, limit=None) == [(0, 34)]
    assert _halo_sources(lengths, 34, rank=2, limit=None) == [(1, 26)]
    assert _halo_sources(lengths, 34, rank=2, limit=6) == [(1, 6)]
    assert _halo_sources(lengths, 34, rank=4, limit=None) == [(0, 34), (1, 34), (2, 34), (3, 34)]
    assert _halo_sources(lengths, 34, rank=4, limit=40) == [(2, 6), (3, 34)]


def test_unsupported_packed_inputs_are_rejected():
    model = _model()
    with pytest.raises(ValueError, match="FlashAttention backend"):
        model.validate_packed_attention_backend("sdpa")
    model.validate_packed_attention_backend("flash_attention_4")
    with pytest.raises(ValueError, match="through the sharder"):
        model(**{name: value for name, value in _thd_batch().items() if name != "labels"})
    with pytest.raises(ValueError, match="one row per rank"):
        shard_documents(_Mesh(2, 0), None, _thd_batch())
    short = _thd_batch(ROWS[:1])
    short["seq_lens_padded"][0, 4] -= 1
    with pytest.raises(ValueError, match="add up to the sequence length"):
        shard_documents(None, None, short)


def test_packed_inputs_reject_a_supplied_cache_and_create_none():
    model = _model()
    _, packed, _ = shard_documents(None, None, _thd_batch(ROWS[:1]))
    packed.pop("labels")
    with torch.no_grad():
        cache = model(input_ids=packed["input_ids"][:, :8], use_cache=True).past_key_values
        assert cache is not None and cache.get_seq_length() == 8
        with pytest.raises(ValueError, match="past_key_values"):
            model(**packed, past_key_values=cache)
        assert cache.get_seq_length() == 8
        assert model(**packed, use_cache=True).past_key_values is None


def _document_id_batch():
    """Row 0 of ``ROWS`` as a batch packed for FlashAttention consumers: ``_packed_seq_ids`` with 1-based document indices, 0 at padding."""
    thd = _thd_batch(ROWS[:1])
    ids = torch.zeros(1, SEQUENCE, dtype=torch.long)
    for index, (_, start, end) in enumerate(_documents(thd, ROWS[:1]), start=1):
        ids[0, start:end] = index
    return {
        "input_ids": thd["input_ids"],
        "labels": thd["labels"],
        "position_ids": thd["position_ids"],
        "_packed_seq_ids": ids,
    }


def test_document_ids_without_flash_offsets_keep_their_boundaries_on_shards(thread_ranks):
    thd, batch = _thd_batch(ROWS[:1]), _document_id_batch()
    # Unlike the THD spans, the document ids keep the padding tail (tokens 126 to 136) as a segment of its own.
    local_boundaries = [[0, 34], [0, 3, 8, 34], [0, 34], [0, 3, 24, 34]]
    for rank in range(4):
        _, expected, _ = shard_documents(_Mesh(4, rank), None, dict(thd))
        _, shard, _ = shard_documents(_Mesh(4, rank), None, dict(batch))
        assert shard["cu_seq_lens_q"].tolist() == local_boundaries[rank]
        assert shard["max_length_q"] == max(b - a for a, b in zip(local_boundaries[rank], local_boundaries[rank][1:]))
        assert shard["halo"].lengths == expected["halo"].lengths
        assert "_packed_seq_ids" not in shard
    # Two documents and a padding tail in one row of eight tokens, over two shards of four.
    small = {"input_ids": torch.arange(3, 11)[None], "labels": torch.arange(3, 11)[None]}
    for rank in range(2):
        _, shard, _ = shard_documents(
            _Mesh(2, rank), None, {**small, "_packed_seq_ids": torch.tensor([[1, 1, 2, 2, 2, 2, 0, 0]])}
        )
        assert shard["cu_seq_lens_q"].tolist() == [0, 2, 4] and shard["halo"].lengths == (0, 2)

    model = _model().train()
    expected, labels = _packed_logits(model, dict(thd))
    torch.nn.functional.cross_entropy(expected.flatten(0, 1), labels.flatten(), reduction="sum").backward()

    logits, gradients = _run_ranks(model, batch, 4)

    # Real tokens see the same documents either way; only the unsupervised padding tail is segmented differently.
    torch.testing.assert_close(logits[:, : sum(ROWS[0])], expected[:, : sum(ROWS[0])], **TOLERANCE)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(gradients[name], parameter.grad, atol=1e-4, rtol=1e-3)


def test_left_padded_row_on_context_parallel_shards_matches_the_row_run_alone(thread_ranks):
    # Three masked tokens, then 61 real ones that cross the shard boundary at token 32; conv_bias is on.
    model = _model()
    generator = torch.Generator().manual_seed(2)
    real = torch.randint(3, 97, (1, 61), generator=generator)
    ids = torch.cat([torch.zeros(1, 3, dtype=torch.long), real], dim=1)
    batch = {
        "input_ids": ids,
        "labels": ids.masked_fill(ids == 0, -100),
        "attention_mask": (ids != 0).long(),
        "position_ids": torch.cat([torch.arange(3), torch.arange(61)])[None],
    }
    reference = copy.deepcopy(model).train()
    expected = reference(input_ids=real).logits
    torch.nn.functional.cross_entropy(expected.flatten(0, 1), real.flatten(), reduction="sum").backward()

    logits, gradients = _run_ranks(model.train(), batch, 2)
    changed = {**batch, "input_ids": ids.clone()}
    changed["input_ids"][0, :3] = torch.tensor([5, 6, 7])
    other, _ = _run_ranks(model, changed, 2)

    torch.testing.assert_close(logits[:, 3:], expected, **TOLERANCE)
    assert torch.equal(other[:, 3:], logits[:, 3:])
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(gradients[name], parameter.grad, atol=1e-4, rtol=1e-3)


def test_one_rank_fsdp2_checkpointing_disables_the_cache_and_keeps_plain_gradients():
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from nemo_automodel.components.distributed.config import FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext
    from nemo_automodel.components.distributed.model_parallelizer import _parallelize_unsharded_fsdp2

    model = _model().train()
    assert model.config.use_cache
    ids = _thd_batch(ROWS[:1])["input_ids"][:, : sum(ROWS[0])]
    plain = copy.deepcopy(model)
    expected = plain(input_ids=ids, use_cache=False).logits
    expected.square().mean().backward()

    # One rank: the parallelizer wraps whole layers without sharding; float32 compute needs no device mesh.
    strategy = FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32))
    _parallelize_unsharded_fsdp2(
        model, MeshContext(strategy_config=strategy, activation_checkpointing=True), parallelizer=model.parallelizer
    )
    output = model(input_ids=ids)
    output.logits.square().mean().backward()

    assert output.past_key_values is None and not model.config.use_cache
    torch.testing.assert_close(output.logits, expected, **TOLERANCE)
    gradients = {name.replace("._checkpoint_wrapped_module", ""): p.grad for name, p in model.named_parameters()}
    for name, parameter in plain.named_parameters():
        assert torch.isfinite(gradients[name]).all()
        torch.testing.assert_close(gradients[name], parameter.grad, atol=1e-5, rtol=1e-4)

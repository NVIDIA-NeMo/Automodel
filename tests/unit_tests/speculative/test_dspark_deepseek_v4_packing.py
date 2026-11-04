# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sequence packing for the DeepSeek V4 and V4.1 DSpark targets and drafts.

Packing must not change what the draft learns from a document: the frozen target's
features for each packed document equal its features when the document runs alone,
and a draft block anchored in a document produces the same outputs whatever else
shares its row.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4 import dspark_target as v4_dspark_target
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM
from nemo_automodel.components.models.deepseek_v41 import dspark_target as v41_dspark_target
from nemo_automodel.components.models.deepseek_v41 import quantization as v41_quantization
from nemo_automodel.components.models.deepseek_v41.dspark import DeepseekV41DSparkModel
from nemo_automodel.components.speculative.dspark import Qwen3DSparkModel
from nemo_automodel.components.speculative.dspark.config import build_deepseek_v4_draft_config
from nemo_automodel.components.speculative.dspark.draft_deepseek_v4 import DeepseekV4DSparkModel
from nemo_automodel.components.speculative.dspark.draft_kimi_k3 import KimiK3DSparkModel
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.recipes.llm.train_dspark import _validate_draft_packing
from tests.unit_tests.models.deepseek_v4.test_dsv4_model_smoke import _tiny_config as _v4_tiny_config
from tests.unit_tests.models.deepseek_v41.test_model import _build_model as _build_v41_model
from tests.unit_tests.models.deepseek_v41.test_model import _tiny_config as _v41_tiny_config
from tests.unit_tests.speculative.test_dspark_deepseek_v41 import _args as _v41_draft_args
from tests.unit_tests.speculative.test_dspark_deepseek_v41 import _target_config as _v41_draft_target_config
from tests.unit_tests.speculative.test_dspark_draft_deepseek_v4 import _tiny_model_args as _v4_draft_args
from tests.unit_tests.speculative.test_dspark_draft_deepseek_v4 import _tiny_target_config as _v4_draft_target_config

# Document lengths per packed row. They are not multiples of the compression ratios,
# so a target without alignment would pool groups across document boundaries.
ROW_DOCUMENTS = ([7, 12, 5], [10, 14])
# Packed-vs-alone parity gate; the negative controls must miss the expected features by 10 times it.
PARITY_TOL = 2e-5


def _unrounded_v41_caches(monkeypatch) -> None:
    """Run V4.1 without its FP8/FP4 cache rounding.

    The rounding stays inside a token, so it cannot mix documents, but a value on a rounding
    threshold turns the packed-vs-alone reduction-order noise into a full FP8 step on some stacks.
    """
    monkeypatch.setattr(v41_quantization, "_quantize", lambda values, block_size, format: values)


def _pack(row_documents, vocab_size: int, generator: torch.Generator):
    """Pack random documents into equal-width rows.

    Returns:
        Per-row lists of document token tensors of shape [1, length], and the packed
        input_ids, position_ids, doc_remaining of shape [batch, sequence] and
        zero-padded seq_lens of shape [batch, max_documents].
    """
    documents = [[torch.randint(1, vocab_size, (1, n), generator=generator) for n in row] for row in row_documents]
    input_ids = torch.cat([torch.cat(row, dim=1) for row in documents], dim=0)
    position_ids = torch.stack([torch.cat([torch.arange(n) for n in row]) for row in row_documents])
    doc_remaining = torch.stack([torch.cat([torch.arange(n - 1, -1, -1) for n in row]) for row in row_documents])
    max_documents = max(len(row) for row in row_documents)
    seq_lens = torch.tensor([row + [0] * (max_documents - len(row)) for row in row_documents])
    return documents, input_ids, position_ids, doc_remaining, seq_lens


def _v4_target() -> DeepseekV4ForCausalLM:
    config = _v4_tiny_config(
        # A shared expert allocates a CUDA stream in MoE.forward.
        n_shared_experts=0,
        # Overlapping CSA (ratio 4) and a second ratio give a 4-token alignment.
        compress_ratios=[0, 4, 2, 0],
        sliding_window=4,
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(config, backend=backend)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def _v41_target():
    config = _v41_tiny_config()
    # Ratios 4 and 2 give a 4-token alignment.
    config.text_config.compress_ratios = [0, 0, 4, 4, 2, 2]
    config.text_config.sliding_window = 4
    torch.manual_seed(0)
    return _build_v41_model(config).eval()


def _packed_and_alone_features(wrapper: HFDSparkTargetModel, vocab_size: int):
    """Target outputs for the packed ``ROW_DOCUMENTS`` rows and for their documents run alone.

    Returns:
        The packed input_ids, position_ids and seq_lens, the packed batch, and per row the alone
        runs' features and last hidden states concatenated to [1, sequence, ...].
    """
    documents, input_ids, position_ids, doc_remaining, seq_lens = _pack(
        ROW_DOCUMENTS, vocab_size=vocab_size, generator=torch.Generator().manual_seed(1)
    )
    ones = torch.ones_like(input_ids)
    with torch.no_grad():
        packed = wrapper.generate_batch(
            input_ids, ones, ones, position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining
        )
        alone = [
            [wrapper.generate_batch(doc, torch.ones_like(doc), torch.ones_like(doc)) for doc in row]
            for row in documents
        ]
    expected = [
        (
            torch.cat([batch.target_hidden_states for batch in row], dim=1),
            torch.cat([batch.target_last_hidden_states for batch in row], dim=1),
        )
        for row in alone
    ]
    return input_ids, position_ids, seq_lens, packed, expected


@pytest.mark.parametrize("hybridep", [False, True], ids=["local", "hybridep"])
@pytest.mark.parametrize(
    ("build_target", "packed_forward"),
    [
        (_v4_target, v4_dspark_target.dspark_packed_forward),
        (_v41_target, v41_dspark_target.dspark_packed_forward),
    ],
    ids=["deepseek_v4", "deepseek_v41"],
)
def test_packed_target_features_match_each_document_alone(build_target, packed_forward, hybridep, monkeypatch):
    if build_target is _v41_target:
        _unrounded_v41_caches(monkeypatch)
    model = build_target()
    # Both rows align to 28 tokens (4-token alignment): 8 + 12 + 8 and 12 + 16.
    aligned_width = 28
    if hybridep:
        # HybridEP pads every rank to the longest aligned row; pretend another rank's
        # row is 16 tokens longer.
        model.backend = dataclasses.replace(model.backend, dispatcher="hybridep")
        monkeypatch.setattr(v41_dspark_target.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(v41_dspark_target.dist, "all_reduce", lambda tensor, op: tensor.add_(16))
        aligned_width += 16
    wrapper = HFDSparkTargetModel(model, target_layer_ids=[1, 3], packed_forward=packed_forward)
    vocab_size = getattr(model.config, "text_config", model.config).vocab_size
    input_ids, position_ids, seq_lens, packed, expected = _packed_and_alone_features(wrapper, vocab_size)

    assert packed.target_hidden_states.shape[:2] == input_ids.shape
    for row, (features, last) in enumerate(expected):
        torch.testing.assert_close(
            packed.target_hidden_states[row : row + 1], features, atol=PARITY_TOL, rtol=PARITY_TOL
        )
        torch.testing.assert_close(
            packed.target_last_hidden_states[row : row + 1], last, atol=PARITY_TOL, rtol=PARITY_TOL
        )
    assert packed.seq_lens is seq_lens and packed.position_ids is position_ids
    with torch.no_grad():
        layout = packed_forward(model, input_ids, seq_lens=seq_lens)
        # Negative control: without the packing metadata a row runs as one sequence, sharing
        # positions, attention and compressed groups across its documents. HybridEP does not reach it.
        ones = torch.ones_like(input_ids)
        unpacked = None if hybridep else wrapper.generate_batch(input_ids, ones, ones)
    assert layout.sequence_ids.shape == (len(ROW_DOCUMENTS), aligned_width)
    if unpacked is not None:
        for row, (features, _) in enumerate(expected):
            assert not torch.allclose(
                unpacked.target_hidden_states[row : row + 1], features, atol=10 * PARITY_TOL, rtol=0
            )


def test_v41_packed_target_parity_catches_documents_off_the_compression_grid(monkeypatch):
    """Packing that keeps documents isolated but skips the compression alignment fails the parity gate."""
    _unrounded_v41_caches(monkeypatch)
    model = _v41_target()
    model._packed_alignment = 1  # documents no longer start on a compressed-group boundary
    wrapper = HFDSparkTargetModel(
        model, target_layer_ids=[1, 3], packed_forward=v41_dspark_target.dspark_packed_forward
    )
    _, _, _, packed, expected = _packed_and_alone_features(wrapper, model.config.text_config.vocab_size)
    for row, (features, _) in enumerate(expected):
        assert not torch.allclose(packed.target_hidden_states[row : row + 1], features, atol=10 * PARITY_TOL, rtol=0)


def test_aligned_packed_layout_skips_absent_documents_and_keeps_the_row_width():
    seq_lens = torch.tensor([[7, 12, 5], [10, 14, -1000]])
    layout = v41_dspark_target.aligned_packed_layout(
        seq_lens, input_shape=(2, 24), alignment=4, sync_across_ranks=False
    )
    assert layout.sequence_ids.shape == (2, 28)
    assert layout.sequence_ids[1].tolist() == [1] * 10 + [0] * 2 + [2] * 14 + [0] * 2
    layout = v41_dspark_target.aligned_packed_layout(
        seq_lens, input_shape=(2, 32), alignment=4, sync_across_ranks=False
    )
    assert layout.sequence_ids.shape == (2, 32)


def test_aligned_packed_layout_syncs_the_aligned_width_across_ranks(monkeypatch):
    seq_lens = torch.tensor([[7, 12, 5], [10, 14, -1000]])
    reduced = []

    def all_reduce(tensor, op):
        reduced.append(int(tensor))
        tensor.add_(4)  # Another rank's aligned row is one alignment unit longer.

    monkeypatch.setattr(v41_dspark_target.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(v41_dspark_target.dist, "all_reduce", all_reduce)
    layout = v41_dspark_target.aligned_packed_layout(seq_lens, input_shape=(2, 24), alignment=4, sync_across_ranks=True)
    # The absent -1000 document adds no span, so this rank contributes 28 tokens.
    assert reduced == [28]
    assert layout.sequence_ids.shape == (2, 32)


def _v4_draft() -> DeepseekV4DSparkModel:
    args = _v4_draft_args()
    args["num_anchors"] = 16
    torch.manual_seed(2)
    draft = DeepseekV4DSparkModel(build_deepseek_v4_draft_config(_v4_draft_target_config(), args))
    draft = draft.to(torch.float32).eval()
    hidden, vocab = draft.config.hidden_size, draft.config.vocab_size
    draft.initialize_embeddings_and_head(
        embed_tokens=nn.Embedding(vocab, hidden), lm_head=nn.Linear(hidden, vocab, bias=False), freeze=True
    )
    return draft


def _v41_draft() -> DeepseekV41DSparkModel:
    torch.manual_seed(2)
    draft = _v41_draft_target_config().build_dspark_draft(_v41_draft_args(num_anchors=16)).eval()
    hidden, vocab = draft.config.hidden_size, draft.config.vocab_size
    draft.initialize_embeddings_and_head(embed_tokens=nn.Embedding(vocab, hidden), lm_head=nn.Linear(hidden, vocab))
    return draft


@pytest.mark.parametrize("build_draft", [_v4_draft, _v41_draft], ids=["deepseek_v4", "deepseek_v41"])
def test_packed_draft_blocks_match_the_document_alone(build_draft, monkeypatch):
    """Blocks anchored in a document match the unpacked run of that document alone.

    The row packs an unsupervised document, the document under test, and a supervised
    document. Without per-document context, positions, and supervision, the tested
    document's blocks would read the first document or train on the last one. num_anchors
    exceeds the candidate count, so both runs sample every candidate anchor.
    """
    if build_draft is _v41_draft:
        _unrounded_v41_caches(monkeypatch)
    draft = build_draft()
    lengths = ROW_DOCUMENTS[0]
    start, end = lengths[0], lengths[0] + lengths[1]
    generator = torch.Generator().manual_seed(3)
    _, input_ids, position_ids, doc_remaining, seq_lens = _pack(
        [lengths], vocab_size=draft.config.vocab_size - 1, generator=generator
    )
    features = torch.randn(1, sum(lengths), 2 * draft.config.hidden_size, generator=generator)
    last_hidden = torch.randn(1, sum(lengths), draft.config.hidden_size, generator=generator)
    loss_mask = torch.ones(1, sum(lengths))
    loss_mask[:, :start] = 0

    with torch.no_grad():
        alone = draft(
            input_ids[:, start:end], features[:, start:end], loss_mask[:, start:end], last_hidden[:, start:end]
        )
        packed = draft(
            input_ids,
            features,
            loss_mask,
            last_hidden,
            position_ids=position_ids,
            seq_lens=seq_lens,
            doc_remaining=doc_remaining,
        )

    # The tested document yields the first anchors of the packed row, in the same order.
    kept = int(alone.block_keep_mask.sum())
    assert kept == lengths[1] - 1
    assert int(packed.block_keep_mask.sum()) == kept + lengths[2] - 1
    eval_mask = alone.eval_mask[:, :kept]
    assert eval_mask.any()
    torch.testing.assert_close(packed.eval_mask[:, :kept], eval_mask)
    torch.testing.assert_close(packed.target_ids[:, :kept][eval_mask], alone.target_ids[:, :kept][eval_mask])
    for field in ("draft_logits", "confidence_pred", "aligned_target_logits"):
        torch.testing.assert_close(
            getattr(packed, field)[:, :kept][eval_mask],
            getattr(alone, field)[:, :kept][eval_mask],
            atol=PARITY_TOL,
            rtol=PARITY_TOL,
        )


def test_only_drafts_with_packing_metadata_accept_packing():
    for draft_class in (Qwen3DSparkModel, DeepseekV4DSparkModel, DeepseekV41DSparkModel):
        assert draft_class.supports_sequence_packing is True
    _validate_draft_packing(SimpleNamespace(supports_sequence_packing=True))
    # Kimi K3 inherits the V4 draft forward, but its target does not isolate packed documents.
    assert KimiK3DSparkModel.supports_sequence_packing is False
    with pytest.raises(NotImplementedError, match="not supported by the Linear DSpark draft"):
        _validate_draft_packing(nn.Linear(2, 2))

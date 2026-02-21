# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Functional tests for pad-token / EOS-token overlap handling.

When pad_token_id == eos_token_id (e.g. Qwen2.5), a naïve
``labels == pad_token_id`` comparison masks **all** EOS tokens, including
real end-of-sequence markers, causing significant PPL degradation.

These tests load real tokenizers (via ``NeMoAutoTokenizer``, which enforces
``add_eos_token=True``) from four model families and verify that
``_get_right_trailing_pad_mask`` correctly preserves the first (real) EOS in a
trailing run while masking only the subsequent (padding) positions.

``TestSquadCollateEndToEnd`` drives the full SFT pipeline
(squad dataset → default_collater) with each tokenizer and verifies that
labels are padded with -100 (never with pad_token_id), even when the two
IDs collide.
"""

from __future__ import annotations

import pytest
import torch
from datasets import Dataset
from pathlib import Path
import os

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
import nemo_automodel.components.datasets.llm.squad as squad_module
from nemo_automodel.components.datasets.llm.formatting_utils import (
    _get_right_trailing_pad_mask,
)
from nemo_automodel.components.datasets.utils import default_collater


#  Helpers

def _effective_pad_id(tok) -> int:
    """Return the pad token id the GPTDataset would actually use.

    Mirrors the fallback chain in GPTDataset.__init__ and _add_pad_token:
    if pad_token_id is not set, eos_token_id is used instead.
    """
    if tok.pad_token_id is not None:
        return tok.pad_token_id
    return tok.eos_token_id


def _simulate_getitem_masking(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-implement the masking block from GPTDataset.__getitem__."""
    seq_len = tokens.shape[0]
    loss_mask = torch.ones(seq_len, dtype=torch.float)
    position_ids = torch.arange(seq_len, dtype=torch.long)
    tokens_pad_mask = _get_right_trailing_pad_mask(tokens, pad_token_id, eos_token_id)
    labels_pad_mask = _get_right_trailing_pad_mask(labels, pad_token_id, eos_token_id)

    loss_mask[labels_pad_mask] = 0.0
    tokens[tokens_pad_mask] = 0
    labels[labels_pad_mask] = 0
    position_ids[tokens_pad_mask] = 0
    return tokens, labels, loss_mask, position_ids


def _build_shifted_pair(text_ids: list[int]):
    """Convert a full token list into (tokens, labels) via the standard shift."""
    t = torch.tensor(text_ids, dtype=torch.long)
    return t[:-1].clone(), t[1:].clone()



#  Parametrised fixture - one entry per tokenizer family

_TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "/home/TestData/automodel")
_TOKENIZER_BASE = Path(_TEST_DATA_DIR) / "tokenizers"

TOKENIZER_MODELS = [
    pytest.param(str(_TOKENIZER_BASE / "Qwen2.5-0.5B"), id="qwen2.5"),
    pytest.param(str(_TOKENIZER_BASE / "Llama-3.2-1B"), id="llama3.2"),
    pytest.param(str(_TOKENIZER_BASE / "gemma-3-1b-pt"), id="gemma3"),
    pytest.param(str(_TOKENIZER_BASE / "Mixtral-8x7B-v0.1"), id="mixtral"),
]


@pytest.fixture(params=TOKENIZER_MODELS)
def tok(request):
    path = Path(request.param)
    if not path.exists():
        pytest.skip(f"Missing tokenizer data: {path}")
    return NeMoAutoTokenizer.from_pretrained(request.param, trust_remote_code=True)


#  Functional tests with real tokenizers (GPT pretrain path)

class TestPadEosOverlapFunctional:
    """Verify correct EOS / padding handling for Qwen, Llama, Gemma3, Mixtral."""

    def test_pad_eos_relationship_detected(self, tok):
        """Sanity-check: report the pad/eos relationship for each tokenizer."""
        pad_id = _effective_pad_id(tok)
        eos_id = tok.eos_token_id
        overlap = pad_id == eos_id
        assert isinstance(overlap, bool)

    def test_single_doc_trailing_padding(self, tok):
        """[content … EOS  PAD PAD PAD] - first EOS in labels must keep loss=1."""
        eos_id = tok.eos_token_id
        pad_id = _effective_pad_id(tok)

        content = tok.encode("The quick brown fox jumps.", add_special_tokens=False)
        assert len(content) >= 2, "need at least 2 content tokens for a meaningful test"
        n_pad = 4
        text = content + [eos_id] + [pad_id] * n_pad + [pad_id]
        tokens, labels = _build_shifted_pair(text)
        tokens_out, labels_out, loss_mask, position_ids = _simulate_getitem_masking(
            tokens.clone(), labels.clone(), pad_id, eos_id,
        )

        eos_label_idx = len(content) - 1
        assert loss_mask[eos_label_idx].item() == 1.0, (
            f"Real EOS at label index {eos_label_idx} must NOT be masked"
        )
        for i in range(len(content)):
            assert loss_mask[i].item() == 1.0, f"Content label at {i} must not be masked"
        for i in range(len(content), len(loss_mask)):
            assert loss_mask[i].item() == 0.0, f"Padding label at {i} must be masked"

        eos_token_idx = len(content)
        assert tokens_out[eos_token_idx].item() == eos_id, (
            "Real EOS in input tokens must not be replaced with 0"
        )
        for i in range(eos_token_idx + 1, len(tokens_out)):
            assert tokens_out[i].item() == 0, f"Padding token at {i} must be 0"

        for i in range(eos_token_idx + 1, len(position_ids)):
            assert position_ids[i].item() == 0, (
                f"Padding position_id at {i} must be 0 (got {position_ids[i].item()})"
            )
        for i in range(eos_token_idx + 1):
            assert position_ids[i].item() == i, (
                f"Content position_id at {i} must equal {i}"
            )

    def test_multi_doc_packed_with_padding(self, tok):
        """[doc1 … EOS doc2 … EOS  PAD PAD] - inter-doc EOS tokens preserved."""
        eos_id = tok.eos_token_id
        pad_id = _effective_pad_id(tok)

        doc1 = tok.encode("First document.", add_special_tokens=False)
        doc2 = tok.encode("Second document.", add_special_tokens=False)
        n_pad = 3
        text = doc1 + [eos_id] + doc2 + [eos_id] + [pad_id] * n_pad + [pad_id]
        tokens, labels = _build_shifted_pair(text)
        tokens_out, labels_out, loss_mask, position_ids = _simulate_getitem_masking(
            tokens.clone(), labels.clone(), pad_id, eos_id,
        )

        content_len = len(doc1) + 1 + len(doc2) + 1
        for i in range(content_len - 1):
            assert loss_mask[i].item() == 1.0, (
                f"Content/EOS label at index {i} must not be masked"
            )
        for i in range(content_len - 1, len(loss_mask)):
            assert loss_mask[i].item() == 0.0, (
                f"Padding label at index {i} must be masked"
            )

        assert tokens_out[len(doc1)].item() == eos_id, (
            "Inter-document EOS in tokens must not be replaced"
        )
        assert tokens_out[len(doc1) + 1 + len(doc2)].item() == eos_id, (
            "Doc2 EOS in tokens must not be replaced"
        )

        last_content_token_idx = content_len - 1
        for i in range(last_content_token_idx + 1):
            assert position_ids[i].item() == i, (
                f"Content position_id at {i} must equal {i}"
            )
        for i in range(last_content_token_idx + 1, len(position_ids)):
            assert position_ids[i].item() == 0, (
                f"Padding position_id at {i} must be 0"
            )

    def test_no_padding(self, tok):
        """Sequence fills the length exactly - nothing should be masked."""
        eos_id = tok.eos_token_id
        pad_id = _effective_pad_id(tok)

        content = tok.encode("A complete sequence ending with EOS.", add_special_tokens=False)
        text = content + [eos_id]
        tokens, labels = _build_shifted_pair(text)
        _, _, loss_mask, position_ids = _simulate_getitem_masking(
            tokens.clone(), labels.clone(), pad_id, eos_id,
        )

        assert loss_mask.sum().item() == len(loss_mask), (
            "With no padding, every label must contribute to the loss"
        )
        expected_pos = list(range(len(position_ids)))
        assert position_ids.tolist() == expected_pos, (
            "With no padding, position_ids must be sequential"
        )

    def test_regression_old_vs_new_behaviour(self, tok):
        """Show the old (buggy) behaviour masks the real EOS; new one doesn't.

        Only runs for tokenizers where pad_token_id == eos_token_id.
        """
        eos_id = tok.eos_token_id
        pad_id = _effective_pad_id(tok)

        if pad_id != eos_id:
            pytest.skip("Regression test only relevant when pad_token_id == eos_token_id")

        content = tok.encode("Short text.", add_special_tokens=False)
        text = content + [eos_id] + [pad_id] * 3 + [pad_id]
        _, labels = _build_shifted_pair(text)

        eos_label_idx = len(content) - 1

        old_mask = torch.ones(len(labels), dtype=torch.float)
        old_mask[labels == pad_id] = 0.0
        assert old_mask[eos_label_idx].item() == 0.0, (
            "Old behaviour should (incorrectly) mask the real EOS - "
            "confirming the bug this patch fixes"
        )

        new_mask = torch.ones(len(labels), dtype=torch.float)
        new_mask[_get_right_trailing_pad_mask(labels, pad_id, eos_id)] = 0.0
        assert new_mask[eos_label_idx].item() == 1.0, (
            "New behaviour must preserve the real EOS in the loss"
        )

    def test_heavy_padding_ratio(self, tok):
        """Short content with very long padding tail - boundary must hold."""
        eos_id = tok.eos_token_id
        pad_id = _effective_pad_id(tok)

        content = tok.encode("Hi.", add_special_tokens=False)
        n_pad = 50
        text = content + [eos_id] + [pad_id] * n_pad + [pad_id]
        tokens, labels = _build_shifted_pair(text)
        tokens_out, _, loss_mask, position_ids = _simulate_getitem_masking(
            tokens.clone(), labels.clone(), pad_id, eos_id,
        )

        assert loss_mask[: len(content)].sum().item() == len(content)
        assert loss_mask[len(content) :].sum().item() == 0.0
        assert tokens_out[len(content)].item() == eos_id

        eos_token_idx = len(content)
        for i in range(eos_token_idx + 1, len(position_ids)):
            assert position_ids[i].item() == 0, (
                f"Padding position_id at {i} must be 0"
            )


#  End-to-end: squad dataset → default_collater

_SQUAD_ROWS = {
    "id": ["0", "1", "2"],
    "title": ["t0", "t1", "t2"],
    "context": [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        "Python is a high-level, general-purpose programming language created by Guido van Rossum.",
        "The Sun is the star at the centre of the Solar System.",
    ],
    "question": [
        "Where is the Eiffel Tower located?",
        "Who created Python?",
        "What is the Sun?",
    ],
    "answers": [
        {"text": ["Paris, France"], "answer_start": [69]},
        {"text": ["Guido van Rossum"], "answer_start": [67]},
        {"text": ["the star at the centre of the Solar System"], "answer_start": [11]},
    ],
}


class TestSquadCollateEndToEnd:
    """Full SFT pipeline: squad dataset → default_collater → batch verification.

    Uses NeMoAutoTokenizer which enforces add_eos_token=True, matching the
    real training pipeline.  The SFT path pads labels with -100 and tracks
    attention_mask independently, so it is inherently safe against the
    pad==eos collision.  These tests confirm that invariant holds end-to-end.
    """

    @pytest.fixture(autouse=True)
    def _patch_load_dataset(self, monkeypatch):
        mock_ds = Dataset.from_dict(_SQUAD_ROWS)

        def _fake_load(name, split=None, **kw):
            if isinstance(split, str) and "[" in split:
                upper = int(split.split("[")[1].split(":")[1].rstrip("]"))
                return mock_ds.select(range(min(upper, len(mock_ds))))
            return mock_ds

        monkeypatch.setattr(squad_module, "load_dataset", _fake_load)

    def test_dataset_examples_before_collation(self, tok):
        """Each example must have matching-length keys and -100 masked prompts."""
        ds = squad_module.make_squad_dataset(tok, split="train")
        assert len(ds) == len(_SQUAD_ROWS["id"])

        eos_id = tok.eos_token_id

        for i in range(len(ds)):
            ex = ds[i]
            assert {"input_ids", "labels", "attention_mask"} <= ex.keys()
            n = len(ex["input_ids"])
            assert len(ex["labels"]) == n
            assert len(ex["attention_mask"]) == n

            assert ex["labels"][0] == -100

            supervised = [l for l in ex["labels"] if l != -100]
            assert eos_id in supervised, (
                f"Example {i}: EOS ({eos_id}) missing from supervised labels"
            )

    def test_collated_batch_structure(self, tok):
        """Collated batch has correct keys, shapes, and no leftover metadata."""
        ds = squad_module.make_squad_dataset(tok, split="train")
        batch = [ds[i] for i in range(len(ds))]
        collated = default_collater(batch)

        assert "input_ids" in collated
        assert "labels" in collated
        assert "attention_mask" in collated
        assert "___PAD_TOKEN_IDS___" not in collated

        bs, seq_len = collated["input_ids"].shape
        assert bs == len(_SQUAD_ROWS["id"])
        assert collated["labels"].shape == (bs, seq_len)
        assert collated["attention_mask"].shape == (bs, seq_len)

    def test_labels_padded_with_ignore_index(self, tok):
        """Label padding MUST be -100, never pad_token_id - the key SFT safety invariant."""
        pad_id = _effective_pad_id(tok)
        eos_id = tok.eos_token_id

        ds = squad_module.make_squad_dataset(tok, split="train")
        batch = [ds[i] for i in range(len(ds))]
        collated = default_collater(batch)

        labels = collated["labels"]
        attention_mask = collated["attention_mask"]

        for b in range(labels.shape[0]):
            pad_positions = attention_mask[b] == 0
            if not pad_positions.any():
                continue

            pad_labels = labels[b][pad_positions]
            assert (pad_labels == -100).all(), (
                f"[row {b}] Collater must pad labels with -100. "
                f"Found {pad_labels.unique().tolist()} at padding positions. "
                f"pad_token_id={pad_id}, eos_token_id={eos_id}, overlap={pad_id == eos_id}"
            )

    def test_real_eos_in_supervised_labels(self, tok):
        """The real EOS token must survive in the supervised region of every row."""
        eos_id = tok.eos_token_id

        ds = squad_module.make_squad_dataset(tok, split="train")
        batch = [ds[i] for i in range(len(ds))]
        collated = default_collater(batch)

        labels = collated["labels"]

        for b in range(labels.shape[0]):
            content_labels = labels[b][labels[b] != -100]
            assert (content_labels == eos_id).any(), (
                f"[row {b}] EOS ({eos_id}) must appear in supervised labels"
            )
            content_mask = labels[b] != -100
            last_idx = content_mask.nonzero(as_tuple=True)[0][-1].item()
            assert labels[b][last_idx].item() == eos_id, (
                f"[row {b}] Last supervised label should be EOS ({eos_id}), "
                f"got {labels[b][last_idx].item()}"
            )

    def test_attention_mask_right_padded(self, tok):
        """Attention mask must be contiguous 1s followed by contiguous 0s."""
        ds = squad_module.make_squad_dataset(tok, split="train")
        batch = [ds[i] for i in range(len(ds))]
        collated = default_collater(batch)

        attention_mask = collated["attention_mask"]

        for b in range(attention_mask.shape[0]):
            mask = attention_mask[b].tolist()
            seen_zero = False
            for j, v in enumerate(mask):
                if v == 0:
                    seen_zero = True
                elif seen_zero:
                    pytest.fail(
                        f"[row {b}] Attention mask has 1 after 0 at position {j} "
                        "(not right-padded)"
                    )

    def test_pad_eos_overlap_input_vs_label_padding(self, tok):
        """When pad==eos, input_ids padding IS the eos token, but labels use -100.

        This is the critical invariant that makes the SFT path safe: the model
        sees eos tokens at padding positions in the input, but the loss function
        ignores them because labels are -100 there.
        """
        pad_id = _effective_pad_id(tok)
        eos_id = tok.eos_token_id
        if pad_id != eos_id:
            pytest.skip("Only relevant when pad_token_id == eos_token_id")

        ds = squad_module.make_squad_dataset(tok, split="train")
        batch = [ds[i] for i in range(len(ds))]
        collated = default_collater(batch)

        input_ids = collated["input_ids"]
        labels = collated["labels"]
        attention_mask = collated["attention_mask"]

        for b in range(input_ids.shape[0]):
            pad_positions = attention_mask[b] == 0
            if not pad_positions.any():
                continue

            assert (input_ids[b][pad_positions] == pad_id).all(), (
                f"[row {b}] input_ids padding should use pad_token_id ({pad_id})"
            )
            assert (labels[b][pad_positions] == -100).all(), (
                f"[row {b}] labels at padding positions must be -100, not eos_token_id"
            )
            assert not (labels[b][pad_positions] == eos_id).any(), (
                f"[row {b}] labels must never use eos_token_id ({eos_id}) as padding"
            )

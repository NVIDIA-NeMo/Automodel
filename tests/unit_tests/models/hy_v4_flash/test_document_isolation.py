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

"""Adversarial document-isolation tests of the packed path (single rank, fp32, PyTorch attention).

Nothing computed for a packed document may depend on another document of the pack. The oracle is always the same
document run alone and unpacked, never the packing code. Every leak path has a negative control: a test-only variant
with that boundary handling removed must make the same check fail.
"""

import dataclasses
from collections.abc import Callable

import pytest
import torch

import nemo_automodel.components.models.hy_v4_flash.model as model_module
from nemo_automodel.components.models.hy_v4_flash.model import HY4FlashBatch
from tests.unit_tests.models.hy_v4_flash.tiny import (
    KPOOL,
    TOLERANCE,
    TOPK,
    WINDOW,
    build_model,
    build_plain_model,
    pack,
    random_documents,
)


def run_packed(model, documents, padding=0, **overrides) -> list[torch.Tensor]:
    """Logits of every document of the pack, each of shape [document_tokens, vocab]."""
    batch = {**pack(documents, padding), **overrides}
    logits = model(**batch).logits[0]
    return list(logits[: sum(map(len, documents))].split([len(d) for d in documents]))


def run_alone(model, document) -> torch.Tensor:
    """Logits of one unpacked document, of shape [document_tokens, vocab] (the independent oracle)."""
    return model(document[None]).logits[0]


def assert_documents_match_alone(model, documents, padding=0, **overrides) -> None:
    with torch.no_grad():
        packed = run_packed(model, documents, padding, **overrides)
        for index, document in enumerate(documents):
            torch.testing.assert_close(
                packed[index],
                run_alone(model, document),
                msg=lambda m, index=index: f"document {index}: {m}",
                **TOLERANCE,
            )


def test_layout_exercises_selection_and_windows() -> None:
    # The oracle comparison only means something if sparse selection and the window actually drop keys.
    model = build_model()
    documents = random_documents()
    start = torch.cat([torch.full((len(d),), s) for d, s in zip(documents, _starts(documents))])
    position = torch.arange(start.shape[0])
    dropped = {"indexed_attention": 0, "sliding_attention": 0}
    for layer in model.model.layers:
        attn = layer.self_attn
        original = attn._attend

        def spy(query, kv, plan, sinks, layout, attn=attn, original=original):
            dropped[attn.attention_type] += int((plan.lengths < position - start + 1).sum())
            used = torch.arange(plan.indices.shape[1]) < plan.lengths[:, None]
            inside = (plan.indices >= start[:, None]) & (plan.indices <= position[:, None])
            assert bool((inside | ~used).all())  # every key of every query lies inside the query's own document
            return original(query, kv, plan, sinks, layout)

        attn._attend = spy
    with torch.no_grad():
        model(**pack(documents))
    assert dropped["indexed_attention"] > 0 and dropped["sliding_attention"] > 0


def _starts(documents: list[torch.Tensor]) -> list[int]:
    starts, total = [], 0
    for document in documents:
        starts.append(total)
        total += len(document)
    return starts


@pytest.mark.parametrize("padding", [0, 7])
def test_every_packed_document_equals_the_document_run_alone(padding: int) -> None:
    assert_documents_match_alone(build_model(), random_documents(), padding)


def test_duplicate_document_attack() -> None:
    # The same document four times back to back, next to a near-copy: any leak makes a later copy differ.
    model = build_model()
    document = random_documents((39,))[0]
    near_copy = document.clone()
    near_copy[-1] = (near_copy[-1] + 1) % 270 + 3
    documents = [document, document, near_copy, document, document[:38], document]
    with torch.no_grad():
        packed = run_packed(model, documents)
        alone = run_alone(model, document)
    for index in (0, 1, 3, 5):
        torch.testing.assert_close(packed[index], alone, **TOLERANCE)
    torch.testing.assert_close(packed[2][:-1], alone[:-1], **TOLERANCE)  # the near-copy only differs at its last token
    assert not torch.allclose(packed[2][-1], alone[-1], **TOLERANCE)


def test_victim_is_unchanged_when_every_other_document_is_replaced_by_adversarial_content() -> None:
    # Neighbours are copies of the victim (identical index keys, so they would win its top-k if they were visible),
    # shifted copies (a pooled group straddling the boundary would mix them in) and unrelated text.
    model = build_model()
    victim = random_documents((43,), seed=5)[0]
    benign = [d for d in random_documents((22, 9, 31, 6), seed=6)]
    hostile = [victim.clone(), victim[1:].clone(), victim[:-3].clone(), victim.flip(0)]
    with torch.no_grad():
        alone = run_alone(model, victim)
        for offset in range(KPOOL):  # victim start at every offset mod the pooling ratio
            lead = random_documents((8 + offset,), seed=7)
            for others in (benign, hostile):
                packed = run_packed(model, [*lead, others[0], others[1], victim, others[2], others[3]])
                torch.testing.assert_close(packed[3], alone, **TOLERANCE)


def test_gradient_isolation_and_additivity() -> None:
    model = build_model()
    documents = random_documents((23, 3, 37, 1, 18), seed=9)
    captured = {}

    def keep_embedding_gradient(module: torch.nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        captured["embeddings"] = output

    hook = model.model.embed_tokens.register_forward_hook(keep_embedding_gradient)
    victim = 2
    logits = run_packed(model, documents)
    logits[victim].square().sum().backward()
    hook.remove()
    grad = captured["embeddings"].grad[0]
    spans = list(zip(_starts(documents), map(len, documents)))
    for index, (start, length) in enumerate(spans):
        if index == victim:
            assert grad[start : start + length].abs().sum() > 0
        else:  # exactly zero, not small: no path from another document to the victim's loss
            assert torch.count_nonzero(grad[start : start + length]) == 0

    def gradients(run: Callable[[], None]) -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        run()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    packed = gradients(lambda: sum(x.square().sum() for x in run_packed(model, documents)).backward())
    total = None
    for document in documents:
        single = gradients(lambda document=document: run_alone(model, document).square().sum().backward())
        total = single if total is None else {n: total[n] + g for n, g in single.items()}
    assert packed.keys() == total.keys()
    for name, value in total.items():
        torch.testing.assert_close(packed[name], value, rtol=2e-3, atol=2e-4, msg=lambda m, name=name: f"{name}: {m}")


# ------------------------------------------------------------------------------------------------- negative controls
def _leaky_window(monkeypatch: pytest.MonkeyPatch) -> None:
    original = model_module.causal_prefix_plan
    monkeypatch.setattr(
        model_module,
        "causal_prefix_plan",
        lambda position, doc_start, width: original(position, doc_start * (width != WINDOW), width),
    )


def _leaky_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    original = model_module.causal_prefix_plan
    monkeypatch.setattr(
        model_module,
        "causal_prefix_plan",
        lambda position, doc_start, width: original(position, doc_start * (width != TOPK), width),
    )


def _leaky_indexer_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    # The indexer treats the whole pack as one document: groups are pack-aligned and may come from other documents.
    original = model_module.HY4FlashModel.prepare_batch

    def prepare(self: model_module.HY4FlashModel, *args: torch.Tensor, **kwargs: torch.Tensor | None) -> HY4FlashBatch:
        batch = original(self, *args, **kwargs)
        layout = dataclasses.replace(batch.layout, doc_ranges=((0, batch.layout.local_end),))
        return dataclasses.replace(batch, layout=layout)

    monkeypatch.setattr(model_module.HY4FlashModel, "prepare_batch", prepare)


def _leaky_engram(monkeypatch: pytest.MonkeyPatch) -> None:
    original = model_module.build_ngram_tokens
    monkeypatch.setattr(
        model_module,
        "build_ngram_tokens",
        lambda ids, positions, size, pad: original(ids, torch.full_like(positions, size), size, pad),
    )


@pytest.mark.parametrize("leak", [_leaky_window, _leaky_dense, _leaky_indexer_groups, _leaky_engram])
def test_removing_one_boundary_mechanism_is_detected(
    monkeypatch: pytest.MonkeyPatch, leak: Callable[[pytest.MonkeyPatch], None]
) -> None:
    model = build_model()
    documents = random_documents()
    assert_documents_match_alone(model, documents)  # the unbroken model passes on this exact input
    leak(monkeypatch)
    with pytest.raises(AssertionError):
        assert_documents_match_alone(model, documents)


def test_positions_that_do_not_restart_per_document_are_detected() -> None:
    model = build_model()
    documents = random_documents()
    running = torch.arange(sum(map(len, documents)))[None]
    with pytest.raises(AssertionError):
        assert_documents_match_alone(model, documents, position_ids=running)


def test_packed_documents_equal_separate_forwards() -> None:
    """Packing must not let documents see each other: packed logits equal each document run alone."""
    model = build_plain_model()
    generator = torch.Generator().manual_seed(2)
    lengths = [21, 12, 9]
    docs = [torch.randint(3, 280, (1, n), generator=generator) for n in lengths]
    with torch.no_grad():
        alone = torch.cat([model(d).logits for d in docs], dim=1)
        packed = model(
            torch.cat(docs, dim=1),
            seq_lens=torch.tensor([lengths]),
            qkv_format="thd",
        ).logits
    torch.testing.assert_close(packed, alone, **TOLERANCE)


def test_padded_batch_rows_equal_separate_forwards() -> None:
    model = build_plain_model()
    generator = torch.Generator().manual_seed(3)
    a, b = torch.randint(3, 280, (1, 20), generator=generator), torch.randint(3, 280, (1, 13), generator=generator)
    batch = torch.full((2, 20), 2)
    batch[0], batch[1, :13] = a[0], b[0]
    mask = torch.zeros(2, 20, dtype=torch.long)
    mask[0], mask[1, :13] = 1, 1
    with torch.no_grad():
        out = model(batch, attention_mask=mask).logits
        torch.testing.assert_close(out[0], model(a).logits[0], **TOLERANCE)
        torch.testing.assert_close(out[1, :13], model(b).logits[0], **TOLERANCE)

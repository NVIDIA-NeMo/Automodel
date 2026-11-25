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

import pytest
import torch

import nemo_automodel.components.models.hy_v4_flash.model as model_module
from nemo_automodel.components.models.hy_v4_flash.model import HY4FlashForCausalLM

from .tiny import CPU_BACKEND, tiny_config

# index_topk 16 with 4-token groups (the released ratio): three recalled groups plus the query's own group.
KPOOL, TOPK, WINDOW = 4, 16, 6
# Starts at every offset mod 4, documents of 1, 2 and 3 tokens, documents shorter and longer than window and top-k.
DOC_LENGTHS = (1, 2, 3, 21, 22, 23, 37, 5, 41)
TOLERANCE = dict(rtol=1e-4, atol=2e-5)  # fp32: same arithmetic, only the GEMM batch shapes (summation order) differ


def build_model() -> HY4FlashForCausalLM:
    torch.manual_seed(0)
    config = tiny_config(
        vision={"num_hidden_layers": 0},
        index_topk=TOPK,
        compressed_indexer_compress_ratio=KPOOL,
        sliding_window=WINDOW,
    )
    model = HY4FlashForCausalLM(config, backend=CPU_BACKEND).float()
    model.initialize_weights(dtype=torch.float32)
    return model


def random_documents(lengths=DOC_LENGTHS, seed=1) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(3, 280, (length,), generator=generator) for length in lengths]


def pack(documents: list[torch.Tensor], padding: int = 0) -> dict[str, torch.Tensor]:
    """One packed row: token ids, document ids (0 = padding) and document-local positions, each of shape [1, tokens]."""
    ids = torch.cat([*documents, torch.full((padding,), 2)])
    doc = torch.cat([*(torch.full((len(d),), i + 1) for i, d in enumerate(documents)), torch.zeros(padding).long()])
    positions = torch.cat([*(torch.arange(len(d)) for d in documents), torch.zeros(padding).long()])
    return dict(
        input_ids=ids[None], packed_seq_ids=doc[None], position_ids=positions[None], attention_mask=doc[None] > 0
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


def test_layout_exercises_selection_and_windows():
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


def _starts(documents):
    starts, total = [], 0
    for document in documents:
        starts.append(total)
        total += len(document)
    return starts


@pytest.mark.parametrize("padding", [0, 7])
def test_every_packed_document_equals_the_document_run_alone(padding):
    assert_documents_match_alone(build_model(), random_documents(), padding)


def test_duplicate_document_attack():
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


def test_victim_is_unchanged_when_every_other_document_is_replaced_by_adversarial_content():
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


def test_gradient_isolation_and_additivity():
    model = build_model()
    documents = random_documents((23, 3, 37, 1, 18), seed=9)
    captured = {}

    def keep_embedding_gradient(module, args, output):
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

    def gradients(run):
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
def _leaky_window(monkeypatch):
    original = model_module.causal_prefix_plan
    monkeypatch.setattr(
        model_module,
        "causal_prefix_plan",
        lambda position, doc_start, width: original(position, doc_start * (width != WINDOW), width),
    )


def _leaky_dense(monkeypatch):
    original = model_module.causal_prefix_plan
    monkeypatch.setattr(
        model_module,
        "causal_prefix_plan",
        lambda position, doc_start, width: original(position, doc_start * (width != TOPK), width),
    )


def _leaky_indexer_groups(monkeypatch):
    # The indexer treats the whole pack as one document: groups are pack-aligned and may come from other documents.
    original = model_module.HY4FlashModel.prepare_batch

    def prepare(self, *args, **kwargs):
        batch = original(self, *args, **kwargs)
        layout = dataclasses.replace(batch.layout, doc_ranges=((0, batch.layout.local_end),))
        return dataclasses.replace(batch, layout=layout)

    monkeypatch.setattr(model_module.HY4FlashModel, "prepare_batch", prepare)


def _leaky_engram(monkeypatch):
    original = model_module.build_ngram_tokens
    monkeypatch.setattr(
        model_module,
        "build_ngram_tokens",
        lambda ids, positions, size, pad: original(ids, torch.full_like(positions, size), size, pad),
    )


@pytest.mark.parametrize("leak", [_leaky_window, _leaky_dense, _leaky_indexer_groups, _leaky_engram])
def test_removing_one_boundary_mechanism_is_detected(monkeypatch, leak):
    model = build_model()
    documents = random_documents()
    assert_documents_match_alone(model, documents)  # the unbroken model passes on this exact input
    leak(monkeypatch)
    with pytest.raises(AssertionError):
        assert_documents_match_alone(model, documents)


def test_positions_that_do_not_restart_per_document_are_detected():
    model = build_model()
    documents = random_documents()
    running = torch.arange(sum(map(len, documents)))[None]
    with pytest.raises(AssertionError):
        assert_documents_match_alone(model, documents, position_ids=running)


@pytest.mark.parametrize("lengths", [(5, 9, 3), (5, 41, 3)])
def test_pooling_modules_are_entered_whether_or_not_a_document_needs_selection(lengths):
    """FSDP turns the first call of a sharded unit into a collective, so every rank must issue the same calls.

    The fp32 group-pooling modules only do work for documents longer than ``index_topk``. A rank whose pack holds no
    such document must still enter them, at the same place, or its collective sequence differs from a rank that does
    (a deadlock under FSDP). Both packs must therefore enter each module in every sparse layer.
    """
    model = build_model()
    assert (max(lengths) > TOPK) == (lengths == (5, 41, 3))
    sparse_layers = [layer for layer in model.model.layers if layer.self_attn.compressed_indexer is not None]
    assert sparse_layers
    entered = []
    for index, layer in enumerate(sparse_layers):
        indexer = layer.self_attn.compressed_indexer
        for name in ("key_value_projection", "key_gate_projection", "intra_group_ape", "key_norm"):
            getattr(indexer, name).register_forward_pre_hook(
                lambda module, args, index=index, name=name: entered.append((index, name))
            )
    with torch.no_grad():
        model(**pack(random_documents(lengths)))
    first_entries = list(dict.fromkeys(entered))
    assert first_entries == [
        (index, name)
        for index in range(len(sparse_layers))
        for name in ("key_value_projection", "key_gate_projection", "intra_group_ape", "key_norm")
    ]


def test_engram_hash_multipliers_follow_a_lookup_table_loaded_after_initialisation():
    """Recipe order: weights are initialised first, the checkpoint (with its frozen vocabulary map) is loaded second.

    The hash multipliers are bounded by the number of distinct compressed ids of the loaded map. Multipliers drawn
    at initialisation, for the placeholder identity map, send every n-gram to a different table row than the
    checkpoint was trained with; they must be redrawn when the map arrives, whichever way it is loaded.
    """
    from nemo_automodel.components.models.hy_v4_flash.engram import draw_hash_multipliers

    def engram_layers(model):
        return [layer.engram_layer for layer in model.model.layers if layer.engram_layer is not None]

    def run(model):
        with torch.no_grad():
            return model(**pack(random_documents((9, 12)))).logits

    source = build_model()
    for engram in engram_layers(source):  # a many-to-few map, like the released one (fewer compressed ids than tokens)
        engram.lookup_table.copy_(torch.arange(engram.lookup_table.numel(), dtype=torch.int32) % 97)
    expected_logits = run(source)
    state = source.state_dict()
    for in_place in (True, False):
        target = build_model()  # ``initialize_weights`` has drawn multipliers for the identity map
        run(target)
        target.load_state_dict(state, assign=not in_place)
        logits = run(target)
        for engram in engram_layers(target):
            compressed = int(engram.lookup_table.max()) + 1
            assert compressed == 97
            expected = draw_hash_multipliers(engram.seed, engram.layer_id, engram.max_ngram_size, compressed)
            torch.testing.assert_close(engram.hash_multipliers, expected)
        torch.testing.assert_close(logits, expected_logits)


def test_engram_gate_is_one_half_at_a_logit_of_exactly_zero():
    """The reference gates with ``sqrt(clamp(|x|, 1e-6)) * sign(x)``; ``sign(0) = 0`` gives ``sigmoid(0) = 0.5``."""
    import torch.nn.functional as F

    from nemo_automodel.components.models.hy_v4_flash.engram import build_ngram_tokens

    model = build_model()
    engram = next(layer.engram_layer for layer in model.model.layers if layer.engram_layer is not None)
    with torch.no_grad():
        for projection in engram.key_projs:  # zero keys: every gate logit is exactly 0
            projection.weight.zero_()
            projection.bias.zero_()
        ids = random_documents((11,))[0][None]
        ngram = build_ngram_tokens(ids, torch.arange(ids.shape[1])[None], engram.max_ngram_size, engram.pad_id)
        streams = torch.randn(1, ids.shape[1], engram.hc_mult, engram.hidden_size)
        out = engram(streams, ngram, torch.ones_like(ids, dtype=torch.bool))
        # the gated signal, computed independently of the gate
        emb = engram.multi_head_embedding.embedding(engram.hash_ngrams(ngram))
        emb = F.rms_norm(emb, (engram.head_dim,)) * torch.stack([n.weight for n in engram.lookup_per_head_norms])
        main = engram.unified_pre_gamma * F.rms_norm(engram.value_proj(emb.flatten(-2)), (engram.hidden_size,))
        scales = torch.stack(list(engram.unified_per_hc_scales))
        expected = 0.5 * main.unsqueeze(2) + main.unsqueeze(2) * scales
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-7)


def _moe_oracle(mlp, x, limit, *, clamp_routed=True, clamp_shared=False):
    """Latent MoE as the reference writes it: clamped SwiGLU in the routed experts, plain SwiGLU in the shared expert.

    Args:
        mlp: The layer's MoE module (only its weights are read).
        x: Tensor of shape [tokens, hidden].
        limit: ``swiglu_limit``.
        clamp_routed: Clamp the routed experts' gate (one-sided) and up (two-sided) pre-activations.
        clamp_shared: Clamp the shared expert the same way (the reference does not).

    Returns:
        Tuple of the output [tokens, hidden] and the largest routed and shared pre-activation magnitudes.
    """
    import torch.nn.functional as F

    def swiglu(gate, up, clamp):
        if clamp:
            gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        return F.silu(gate) * up

    scores = torch.sigmoid(x @ mlp.gate.weight.T)
    top = (scores + mlp.gate.e_score_correction_bias).topk(mlp.gate.topk, dim=-1).indices
    weights = scores.gather(1, top)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * mlp.gate.route_scale
    latent = x @ mlp.fc1_latent_proj.weight.T
    routed = torch.zeros_like(latent)
    routed_peak = 0.0
    for slot in range(top.shape[1]):
        gate_up = torch.einsum("ti,tio->to", latent, mlp.experts.gate_and_up_projs[top[:, slot]])
        gate, up = gate_up.chunk(2, dim=-1)
        routed_peak = max(routed_peak, float(gate.max()), float(up.abs().max()))
        hidden = swiglu(gate, up, clamp_routed)
        routed += torch.einsum("ti,tio->to", hidden, mlp.experts.down_projs[top[:, slot]]) * weights[:, slot, None]
    shared = mlp.shared_experts
    gate, up = shared.gate_proj(x), shared.up_proj(x)
    out = routed @ mlp.fc2_latent_proj.weight.T + shared.down_proj(swiglu(gate, up, clamp_shared))
    return out, routed_peak, max(float(gate.max()), float(up.abs().max()))


def test_swiglu_clamp_applies_to_routed_experts_and_not_to_the_shared_expert():
    """The one MoE detail where Hy4-flash departs from the shared MoE: upstream clamps only the routed experts.

    Random tiny models never reach the limit, so the input is scaled until both kinds of pre-activation exceed it.
    The two wrong variants (shared expert clamped as the stock shared MoE does; routed experts not clamped) must
    not match.
    """
    model = build_model()
    mlp = next(layer.mlp for layer in model.model.layers if type(layer.mlp).__name__ == "MoE")
    limit = model.config.text_config.swiglu_limit
    torch.manual_seed(3)
    x = torch.randn(40, model.config.text_config.hidden_size) * 400.0
    with torch.no_grad():
        mlp.experts.gate_and_up_projs.mul_(30.0)  # the latent projection shrinks the routed input; scale it back up
        ours = mlp(x[None])[0]
        expected, routed_peak, shared_peak = _moe_oracle(mlp, x, limit)
        shared_clamped, _, _ = _moe_oracle(mlp, x, limit, clamp_shared=True)
        routed_unclamped, _, _ = _moe_oracle(mlp, x, limit, clamp_routed=False)
    assert routed_peak > limit and shared_peak > limit  # the clamp is actually exercised on both paths
    torch.testing.assert_close(ours, expected, rtol=1e-4, atol=1e-4)
    assert not torch.allclose(ours, shared_clamped, rtol=1e-2, atol=1e-2)
    assert not torch.allclose(ours, routed_unclamped, rtol=1e-2, atol=1e-2)

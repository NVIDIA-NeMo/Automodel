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

"""Expansion under sequence packing.

Packing concatenates several sequences into one row and relies on ``position_ids`` -- which
restart at zero per sequence -- to keep attention from crossing the boundaries. Transformers
derives the flash-attention ``cu_seq_lens`` from those ids and passes them to the layer as
keyword arguments.

That is the whole interaction surface with expansion, and the design's answer to it is to
carry the two streams as a tuple rather than stacked on the batch axis, so each stream is
handed the layer's *unmodified* arguments. Batch stacking would have broken here: packing
flattens the batch axis into a token stream, so a stacked second stream would be read as
more tokens of the same pack.

These tests therefore check two things: that the two passes really do receive identical
packing metadata, and that function preservation survives a packed layout.

The real flash-attention varlen kernel is not exercised here -- it needs Ampere or newer and
is an optional dependency. What is exercised is the part this repository owns: which
arguments reach each stream.
"""

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from nemo_automodel.components.expansion import (
    ExpansionConfig,
    apply_expansion,
    expansion_parameters,
)

VOCAB, HIDDEN, LAYERS, SEQ = 64, 32, 4, 12
EXPANDED_LAYERS = [1, 2]
#: Three sequences of 4 tokens packed into one row of 12.
PACKED_POSITION_IDS = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])


def _build(layers: list[int] | None = None, perturb: float = 0.0) -> LlamaForCausalLM:
    """A tiny Llama, optionally expanded and perturbed."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=SEQ * 2,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(config).eval()
    if layers is not None:
        apply_expansion(model, ExpansionConfig(enabled=True, layers=layers))
    if perturb:
        generator = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for _, param in expansion_parameters(model):
                param.add_(torch.randn(param.shape, generator=generator) * perturb)
    return model


@pytest.fixture
def packed_batch() -> dict:
    """A single row holding three packed sequences, with per-sequence position ids."""
    torch.manual_seed(1)
    return {
        "input_ids": torch.randint(0, VOCAB, (1, SEQ)),
        "position_ids": PACKED_POSITION_IDS,
        "use_cache": False,
    }


def _logits(model: LlamaForCausalLM, batch: dict) -> torch.Tensor:
    with torch.no_grad():
        return model(**batch).logits


def test_function_preservation_holds_under_a_packed_layout(packed_batch):
    """A packed batch must not disturb the guarantee the unpacked one gives."""
    assert torch.equal(_logits(_build(layers=EXPANDED_LAYERS), packed_batch), _logits(_build(), packed_batch))


def test_both_streams_receive_identical_packing_metadata(packed_batch):
    """Stream B has to see the same sequence boundaries stream A saw.

    If the two passes disagreed about where sequences begin, stream B's attention would
    cross a boundary stream A respected, and the lateral term would be added to tokens that
    attended to different context. Zero-initialized output projections would hide it, so
    the expansion weights are perturbed and the arguments are inspected directly.
    """
    model = _build(layers=EXPANDED_LAYERS, perturb=0.05)
    layer = model.model.layers[EXPANDED_LAYERS[0]]
    calls = []

    attention = layer.self_attn
    original = type(attention).forward

    def spy(self, hidden_states, *args, **kwargs):
        """Args: hidden_states ``[batch, sequence, hidden]``. Returns: what the real forward returns."""
        # Patching the class catches every layer's attention, so record only the one under
        # test; the other layers run in skip mode and would drown the two calls that matter.
        if self is attention:
            calls.append(
                {
                    "position_embeddings": kwargs.get("position_embeddings"),
                    "attention_mask": kwargs.get("attention_mask"),
                    "position_ids": kwargs.get("position_ids"),
                    "shape": tuple(hidden_states.shape),
                }
            )
        return original(self, hidden_states, *args, **kwargs)

    type(attention).forward = spy
    try:
        _logits(model, packed_batch)
    finally:
        type(attention).forward = original

    assert len(calls) == 2, f"an expanded layer must run once per stream, saw {len(calls)} calls"
    stream_a, stream_b = calls
    assert stream_a["shape"] == stream_b["shape"]
    for key in ("attention_mask", "position_ids"):
        a, b = stream_a[key], stream_b[key]
        assert (a is None and b is None) or torch.equal(a, b), f"{key} differed between the streams"
    for a, b in zip(stream_a["position_embeddings"] or (), stream_b["position_embeddings"] or ()):
        assert torch.equal(a, b), "rotary embeddings differed between the streams"


def test_packing_metadata_reaches_the_layer_unchanged(packed_batch):
    """The carrier must not rewrite the ids packing depends on."""
    model = _build(layers=EXPANDED_LAYERS, perturb=0.05)
    layer = model.model.layers[EXPANDED_LAYERS[0]]
    seen = []

    original = type(layer.self_attn).forward

    def spy(self, hidden_states, *args, **kwargs):
        """Args: hidden_states ``[batch, sequence, hidden]``. Returns: what the real forward returns."""
        seen.append(kwargs.get("position_ids"))  # every layer, expanded or not
        return original(self, hidden_states, *args, **kwargs)

    type(layer.self_attn).forward = spy
    try:
        _logits(model, packed_batch)
    finally:
        type(layer.self_attn).forward = original

    assert seen, "the attention module was never called"
    for position_ids in seen:
        if position_ids is not None:
            assert torch.equal(position_ids, PACKED_POSITION_IDS)

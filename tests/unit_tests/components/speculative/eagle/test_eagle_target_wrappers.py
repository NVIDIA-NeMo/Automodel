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

"""Unit tests for EAGLE target-model wrappers' custom-impl path.

Covers the branch exercised when ``target_force_hf=False`` and the loaded
target model is the AutoModel custom implementation (e.g. the custom
``Qwen3MoeForCausalLM`` under ``components/models/qwen3_moe``) rather than
the stock HuggingFace class. Verifies:

1. ``_get_transformer_layers`` normalizes both ``nn.ModuleDict`` (custom
   impl, str-keyed) and ``nn.ModuleList`` (HF) into an ordered layer list.
2. ``generate_batch`` does not forward ``output_hidden_states`` /
   ``output_attentions`` / ``use_cache`` to a custom backbone whose
   ``forward`` does not declare them.
3. ``generate_batch`` handles a bare-tensor return value (custom backbone)
   as well as a HF dataclass return with a ``.logits`` attribute.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel
from nemo_automodel.components.speculative.eagle.target_v12 import HFEagleTargetModel

_FORBIDDEN_HF_FLAGS = {"output_hidden_states", "output_attentions", "use_cache"}


class _FakeAutoModelBackbone(nn.Module):
    """Mimics an AutoModel custom-impl backbone: ``ModuleDict`` layers and
    a ``**attn_kwargs`` catch-all in ``forward``."""

    def __init__(self, num_layers: int, hidden: int, embed: nn.Embedding) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({str(i): nn.Linear(hidden, hidden) for i in range(num_layers)})
        self._embed = embed

    def forward(self, input_ids, attention_mask=None, position_ids=None, **attn_kwargs):
        leaked = _FORBIDDEN_HF_FLAGS & set(attn_kwargs)
        if leaked:
            raise AssertionError(f"HF flag leaked to custom backbone: {leaked}")
        h = self._embed(input_ids)
        for layer in self.layers.values():
            h = layer(h)
        return h


class _FakeAutoModelCausalLM(nn.Module):
    """Outer wrapper matching the AutoModel custom-impl surface used by
    EAGLE-3 (``self.model.model.layers`` is a ``ModuleDict``) and EAGLE-1/2
    (``self.model.model`` exposes a bare-tensor forward)."""

    def __init__(self, num_layers: int = 4, hidden: int = 16, vocab: int = 32) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": num_layers})
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.model = _FakeAutoModelBackbone(num_layers, hidden, self.embed_tokens)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids, attention_mask=None, **attn_kwargs):
        h = self.model(input_ids, attention_mask=attention_mask, **attn_kwargs)
        return self.lm_head(h)


class _FakeHFBackbone(nn.Module):
    """Mimics a HuggingFace backbone: ``ModuleList`` layers, explicit HF
    flags in ``forward`` signature, and a tuple-style return."""

    def __init__(self, num_layers: int, hidden: int, embed: nn.Embedding) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_layers)])
        self._embed = embed

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        h = self._embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return (h,)


class _FakeHFCausalLM(nn.Module):
    """HF causal-LM stand-in returning ``CausalLMOutput`` with ``.logits``."""

    def __init__(self, num_layers: int = 4, hidden: int = 16, vocab: int = 32) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": num_layers})
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.model = _FakeHFBackbone(num_layers, hidden, self.embed_tokens)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        h = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )[0]
        return CausalLMOutput(logits=self.lm_head(h))


def _tiny_batch(batch: int = 2, seq: int = 8, vocab: int = 32):
    input_ids = torch.randint(0, vocab, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)
    loss = torch.ones(batch, seq, dtype=torch.long)
    return input_ids, attn, loss


def test_eagle3_get_layers_normalizes_moduledict() -> None:
    fake = _FakeAutoModelCausalLM(num_layers=4)
    wrapper = HFEagle3TargetModel(fake, aux_layer_ids=[0, 1, 3])
    layers = wrapper._get_transformer_layers()
    assert len(layers) == 4
    for i, layer in enumerate(layers):
        assert layer is fake.model.layers[str(i)]


def test_eagle3_get_layers_normalizes_modulelist() -> None:
    fake = _FakeHFCausalLM(num_layers=4)
    wrapper = HFEagle3TargetModel(fake, aux_layer_ids=[0, 1, 3])
    layers = wrapper._get_transformer_layers()
    assert len(layers) == 4
    for i, layer in enumerate(layers):
        assert layer is fake.model.layers[i]


def test_eagle3_drops_hf_flags_for_custom_backbone() -> None:
    fake = _FakeAutoModelCausalLM(num_layers=4)
    wrapper = HFEagle3TargetModel(fake, aux_layer_ids=[0, 1, 3])
    input_ids, attn, loss = _tiny_batch()
    batch = wrapper.generate_batch(input_ids, attn, loss)
    assert batch.logits.shape == (2, 8, 32)


def test_eagle3_handles_bare_tensor_return() -> None:
    fake = _FakeAutoModelCausalLM(num_layers=4)
    wrapper = HFEagle3TargetModel(fake, aux_layer_ids=[0, 1, 3])
    input_ids, attn, loss = _tiny_batch()
    batch = wrapper.generate_batch(input_ids, attn, loss)
    assert isinstance(batch.logits, torch.Tensor)


def test_eagle3_handles_hf_dataclass_return() -> None:
    fake = _FakeHFCausalLM(num_layers=4)
    wrapper = HFEagle3TargetModel(fake, aux_layer_ids=[0, 1, 3])
    input_ids, attn, loss = _tiny_batch()
    batch = wrapper.generate_batch(input_ids, attn, loss)
    assert batch.logits.shape == (2, 8, 32)


def test_eagle1_drops_hf_flags_for_custom_backbone() -> None:
    fake = _FakeAutoModelCausalLM(num_layers=2)
    wrapper = HFEagleTargetModel(fake)
    input_ids, attn, loss = _tiny_batch()
    batch = wrapper.generate_batch(input_ids, attn, loss)
    assert batch.target_logits.shape == (2, 8, 32)


def test_eagle1_handles_bare_tensor_return() -> None:
    fake = _FakeAutoModelCausalLM(num_layers=2)
    wrapper = HFEagleTargetModel(fake)
    input_ids, attn, loss = _tiny_batch()
    batch = wrapper.generate_batch(input_ids, attn, loss)
    assert isinstance(batch.input_hidden_states, torch.Tensor)
    assert isinstance(batch.target_hidden_states, torch.Tensor)


def test_eagle1_handles_hf_dataclass_return() -> None:
    fake = _FakeHFCausalLM(num_layers=2)
    wrapper = HFEagleTargetModel(fake)
    input_ids, attn, loss = _tiny_batch()
    batch = wrapper.generate_batch(input_ids, attn, loss)
    assert batch.target_logits.shape == (2, 8, 32)

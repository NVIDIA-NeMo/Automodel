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

"""Smoke tests for the DeepSeek V4.1 model: structure on CPU, forward / backward on CUDA."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v41.model import document_relative_positions
from tests.unit_tests.models.deepseek_v41.conftest import build_tiny_model, requires_cuda, tiny_config


class TestStructure:
    def test_module_layout_follows_csa2_modes(self):
        model = build_tiny_model(tiny_config())
        layers = model.model.layers
        assert layers["0"].self_attn.compressor is None and layers["0"].self_attn.indexer is None
        assert layers["2"].self_attn.compressor is not None and layers["2"].self_attn.indexer.owns_k
        assert layers["3"].self_attn.compressor is None and layers["3"].self_attn.indexer is None
        assert layers["5"].self_attn.compressor is None and not layers["5"].self_attn.indexer.owns_k
        assert layers["1"].engram is not None and layers["0"].engram is None
        assert not hasattr(model.model, "hc_head")
        assert model.lm_head.weight.dtype == torch.float32
        assert model.model.engram_hasher is not None and model.model.engram_hasher.has_token_map

    def test_engram_disabled_removes_modules(self):
        model = build_tiny_model(tiny_config(engram_enabled=False))
        assert model.model.engram_hasher is None
        assert all(block.engram is None for block in model.model.layers.values())

    def test_hf_state_dict_adapter_attached(self):
        model = build_tiny_model(tiny_config(), enable_hf_state_dict_adapter=True)
        assert model.state_dict_adapter is not None

    def test_tied_embeddings_rejected(self):
        with pytest.raises(Exception):
            build_tiny_model(tiny_config(tie_word_embeddings=True))

    def test_document_relative_positions(self):
        seq_ids = torch.tensor([[1, 1, 1, 2, 2, 0, 0]])
        assert document_relative_positions(seq_ids).tolist() == [[0, 1, 2, 0, 1, 0, 1]]

    def test_initialize_weights_on_cpu(self):
        model = build_tiny_model(tiny_config())
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), name


@requires_cuda
class TestForwardBackward:
    @staticmethod
    def _model_and_inputs(seq_len=13, batch=2, **config_overrides):
        model = build_tiny_model(tiny_config(**config_overrides)).cuda()
        torch.manual_seed(0)
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len), device="cuda")
        return model, input_ids

    def test_forward_shape_and_backward(self):
        model, input_ids = self._model_and_inputs()
        model.train()
        out = model(input_ids, output_hidden_states=True)
        assert out.logits.shape == (2, 13, model.config.vocab_size)
        assert out.hidden_states.shape == (2, 13, model.config.hidden_size)
        assert torch.isfinite(out.logits).all()
        out.logits.float().sum().backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
        assert grads["model.layers.2.self_attn.compressor.wkv.weight"] is not None
        assert grads["model.layers.0.attn_hc.fn"] is not None
        assert grads["model.layers.1.engram.wkv.weight"] is not None
        assert grads["model.embed_tokens.weight"].abs().sum() > 0

    def test_causality_across_all_layers(self):
        model, input_ids = self._model_and_inputs(seq_len=12)
        with torch.no_grad():
            base = model(input_ids).logits
            perturbed = input_ids.clone()
            perturbed[:, -1] = (perturbed[:, -1] + 1) % model.config.vocab_size
            other = model(perturbed).logits
        assert torch.allclose(base[:, :-1], other[:, :-1], atol=1e-4, rtol=1e-4)

    def test_packed_thd_matches_separate_documents(self):
        # Compressed groups are aligned to the packed sequence (reference semantics), so
        # documents whose lengths are multiples of every compress ratio pack losslessly.
        model, _ = self._model_and_inputs()
        torch.manual_seed(1)
        doc_a = torch.randint(0, model.config.vocab_size, (1, 10), device="cuda")
        doc_b = torch.randint(0, model.config.vocab_size, (1, 6), device="cuda")
        with torch.no_grad():
            logits_a = model(doc_a).logits
            logits_b = model(doc_b).logits
            packed = torch.cat([doc_a, doc_b], dim=1).squeeze(0)
            packed_out = model(
                packed,
                qkv_format="thd",
                seq_lens=torch.tensor([10, 6], device="cuda"),
                seq_lens_padded=torch.tensor([10, 6], device="cuda"),
            ).logits
        assert packed_out.shape == (1, 16, model.config.vocab_size)
        assert torch.allclose(packed_out[:, :10], logits_a, atol=1e-3, rtol=1e-3)
        assert torch.allclose(packed_out[:, 10:], logits_b, atol=1e-3, rtol=1e-3)

    def test_right_padding_does_not_change_valid_logits(self):
        model, input_ids = self._model_and_inputs(seq_len=10, batch=1)
        padded = torch.cat([input_ids, torch.full((1, 4), 2, device="cuda")], dim=1)
        attention_mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 4)], dim=1).cuda()
        with torch.no_grad():
            base = model(input_ids).logits
            with_pad = model(padded, attention_mask=attention_mask).logits
        assert torch.allclose(with_pad[:, :10], base, atol=1e-3, rtol=1e-3)

    def test_fake_quant_toggle_changes_outputs_only_slightly(self):
        model, input_ids = self._model_and_inputs(seq_len=11, batch=1)
        with torch.no_grad():
            quant = model(input_ids).logits
            for module in model.modules():
                if hasattr(module, "fake_quant"):
                    module.fake_quant = False
            plain = model(input_ids).logits
        assert not torch.equal(quant, plain)
        assert torch.allclose(quant, plain, atol=0.5, rtol=0.1)

    def test_repeated_forwards_are_stateless(self):
        model, input_ids = self._model_and_inputs(seq_len=6, batch=1)
        with torch.no_grad():
            first = model(input_ids).logits
            model(torch.flip(input_ids, dims=[1]))
            second = model(input_ids).logits
        assert torch.equal(first, second)

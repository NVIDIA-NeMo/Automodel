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

"""Smoke tests for the DeepSeek V4.1 model: structure and forward / backward on CPU."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v41.model import (
    document_relative_positions,
)
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.deepseek_v41.conftest import build_tiny_model, tiny_config


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


class TestForwardBackward:
    @staticmethod
    def _model_and_inputs(seq_len=13, batch=2, **config_overrides):
        model = build_tiny_model(tiny_config(**config_overrides))
        torch.manual_seed(0)
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len), device="cpu")
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
        # Aligned documents must retain the reference single-document behavior.
        model, _ = self._model_and_inputs(index_n_heads=16)
        torch.manual_seed(1)
        doc_a = torch.randint(0, model.config.vocab_size, (1, 10), device="cpu")
        doc_b = torch.randint(0, model.config.vocab_size, (1, 6), device="cpu")
        with torch.no_grad():
            logits_a = model(doc_a).logits
            logits_b = model(doc_b).logits
            packed = torch.cat([doc_a, doc_b], dim=1).squeeze(0)
            packed_out = model(
                packed,
                qkv_format="thd",
                seq_lens=torch.tensor([10, 6], device="cpu"),
                seq_lens_padded=torch.tensor([10, 6], device="cpu"),
            ).logits
        assert packed_out.shape == (1, 16, model.config.vocab_size)
        assert torch.allclose(packed_out[:, :10], logits_a, atol=1e-3, rtol=1e-3)
        assert torch.allclose(packed_out[:, 10:], logits_b, atol=1e-3, rtol=1e-3)

    def test_right_padding_does_not_change_valid_logits(self):
        model, input_ids = self._model_and_inputs(seq_len=10, batch=1)
        padded = torch.cat([input_ids, torch.full((1, 4), 2, device="cpu")], dim=1)
        attention_mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 4)], dim=1)
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


class TestTrainingRegressions:
    @pytest.mark.parametrize("fake_quant", [False, True])
    @pytest.mark.parametrize("ignore_router", [False, True])
    @pytest.mark.parametrize("packed", [False, True])
    def test_checkpoint_matches_eager_gradients(self, fake_quant, ignore_router, packed):
        config = tiny_config(
            kv_cache_fake_quant=fake_quant,
            engram_trainable=True,
            num_hidden_layers=8,
            compress_ratios=[0, 0, 2, 2, 2, 2, 1, 1],
            kv_source_layer_ids=[2, 4, 6],
            index_source_layer_ids=[2, 4, 6, 7],
            candidate_source_layer_id=6,
        )
        eager = build_tiny_model(config).train()
        checked = build_tiny_model(config).train()
        checked_parameters = dict(checked.named_parameters())
        apply_ac(checked, ignore_router=ignore_router)
        tokens = torch.tensor([[5, 6, 7, 11, 12, 13, 14, 15]])
        kwargs = (
            dict(qkv_format="thd", seq_lens=torch.tensor([3, 5]), seq_lens_padded=torch.tensor([3, 5]))
            if packed
            else {}
        )
        eager_outputs = [eager(tokens, **kwargs).logits, eager(tokens.flip(1), **kwargs).logits]
        checked_outputs = [checked(tokens, **kwargs).logits, checked(tokens.flip(1), **kwargs).logits]
        for actual, expected in zip(checked_outputs, eager_outputs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        sum(out.square().mean() for out in eager_outputs).backward()
        sum(out.square().mean() for out in checked_outputs).backward()
        expected_grads = {name: p.grad for name, p in eager.named_parameters()}
        actual_grads = {name: p.grad for name, p in checked_parameters.items()}
        assert actual_grads.keys() == expected_grads.keys()
        for name, expected in expected_grads.items():
            actual = actual_grads[name]
            if expected is None:
                assert actual is None, name
            else:
                assert actual is not None, name
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7, msg=name)
        assert expected_grads["model.layers.2.self_attn.compressor.wkv.weight"].abs().sum() > 0
        assert expected_grads["model.layers.4.self_attn.compressor.wkv.weight"].abs().sum() > 0
        assert expected_grads["model.layers.1.engram.embed.weight"].abs().sum() > 0

    @pytest.mark.parametrize("lengths", [(3, 5), (1, 3), (5, 2), (4, 6), (11, 17), (5, 19, 13)])
    def test_unaligned_packing_matches_independent_documents(self, lengths):
        # Two random index heads frequently produce tied zero ReLU scores. Use
        # more heads to compare packing rather than unspecified Top-K tie order.
        model = build_tiny_model(tiny_config(kv_cache_fake_quant=False, index_n_heads=16))
        documents = [torch.arange(5 + 20 * i, 5 + 20 * i + length).unsqueeze(0) for i, length in enumerate(lengths)]
        with torch.no_grad():
            expected = torch.cat([model(doc).logits for doc in documents], dim=1)
            actual = model(
                torch.cat(documents, dim=1).squeeze(0),
                qkv_format="thd",
                seq_lens=torch.tensor(lengths),
                seq_lens_padded=torch.tensor(lengths),
            ).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("fake_quant", [False, True])
    def test_unaligned_packing_is_causal(self, fake_quant):
        model = build_tiny_model(tiny_config(kv_cache_fake_quant=fake_quant))
        tokens = torch.tensor([5, 6, 7, 11, 12, 13, 14, 15])
        kwargs = dict(
            qkv_format="thd",
            seq_lens=torch.tensor([3, 5]),
            seq_lens_padded=torch.tensor([3, 5]),
        )
        with torch.no_grad():
            expected = model(tokens, **kwargs).logits
            changed = tokens.clone()
            changed[5] = 99
            actual = model(changed, **kwargs).logits
        # MoE token regrouping can change floating-point reduction order.
        torch.testing.assert_close(actual[:, :5], expected[:, :5], rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("pad", [1, 3])
    def test_left_padding_preserves_logits(self, pad):
        model = build_tiny_model(tiny_config(kv_cache_fake_quant=False))
        tokens = torch.tensor([[5, 6, 7, 8, 9]])
        padded = torch.nn.functional.pad(tokens, (pad, 0), value=2)
        mask = torch.nn.functional.pad(torch.ones_like(tokens), (pad, 0))
        with torch.no_grad():
            expected = model(tokens).logits
            actual = model(padded, attention_mask=mask).logits[:, pad:]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_future_hidden_states_have_zero_gradient(self):
        model = build_tiny_model(tiny_config(engram_enabled=False, kv_cache_fake_quant=False))
        embeddings = torch.randn(1, 8, model.config.hidden_size, requires_grad=True)
        hidden = model.model(
            inputs_embeds=embeddings,
            qkv_format="thd",
            seq_lens=torch.tensor([3, 5]),
            seq_lens_padded=torch.tensor([3, 5]),
        )
        (gradient,) = torch.autograd.grad(hidden[:, 4].square().sum(), embeddings)
        assert torch.count_nonzero(gradient[:, 5:]) == 0
        assert torch.count_nonzero(gradient[:, :3]) == 0
        assert gradient[:, 3:5].abs().sum() > 0

    def test_packed_parameter_gradients_match_independent_documents(self):
        config = tiny_config(kv_cache_fake_quant=False, index_n_heads=16, engram_trainable=True)
        packed_model = build_tiny_model(config).train()
        separate_model = build_tiny_model(config).train()
        first = torch.tensor([[5, 6, 7]])
        second = torch.tensor([[11, 12, 13, 14, 15]])
        packed_output = packed_model(
            torch.cat([first, second], dim=1),
            qkv_format="thd",
            seq_lens=torch.tensor([3, 5]),
            seq_lens_padded=torch.tensor([3, 5]),
        ).logits
        separate_output = torch.cat([separate_model(first).logits, separate_model(second).logits], dim=1)
        packed_output.square().mean().backward()
        separate_output.square().mean().backward()
        separate_parameters = dict(separate_model.named_parameters())
        for name, parameter in packed_model.named_parameters():
            expected = separate_parameters[name].grad
            if expected is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6, msg=name)

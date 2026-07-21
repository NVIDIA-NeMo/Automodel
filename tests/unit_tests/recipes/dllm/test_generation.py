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

"""Tests for the example dLLM generation entry point."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "dllm_generate"
sys.path.insert(0, str(EXAMPLE_DIR))

from generate import (  # noqa: E402
    SAMPLERS,
    IDLMSampler,
    LLaDA2Sampler,
    encode_generation_prompts,
    generate_llada2,
    main,
)


class _FakeLLaDA2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[901, 902]], device=self.anchor.device)


class _FakeTokenizer:
    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids, *, skip_special_tokens):
        ids = token_ids.tolist()
        self.decode_calls.append((ids, skip_special_tokens))
        return f"decoded:{ids}"


class _FakeChatTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"input_ids": [[101, 102], [201, 202]]}


def test_llada2_sampler_is_registered_with_native_generation_defaults():
    assert SAMPLERS["llada2"] is LLaDA2Sampler
    config = LLaDA2Sampler.default_config
    assert config.steps == 32
    assert config.block_size == 32
    assert config.threshold == 0.5


def test_chat_prompt_encoding_requests_dictionary_output():
    tokenizer = _FakeChatTokenizer()

    inputs = encode_generation_prompts(tokenizer, ["first", "second"], raw=False)

    assert inputs == [[101, 102], [201, 202]]
    messages, kwargs = tokenizer.calls[0]
    assert messages == [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]
    assert kwargs == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_tensors": None,
        "return_dict": True,
    }


def test_generate_llada2_maps_config_and_decodes_generated_only_tokens():
    model = _FakeLLaDA2()
    tokenizer = _FakeTokenizer()
    config = replace(
        LLaDA2Sampler.default_config,
        steps=17,
        max_new_tokens=64,
        block_size=16,
        temperature=0.2,
        threshold=0.7,
    )

    responses = generate_llada2(model, tokenizer, [[11, 12, 13], [21]], config, mask_id=156895, eos_id=156892)

    assert responses == ["decoded:[901, 902]", "decoded:[901, 902]"]
    assert tokenizer.decode_calls == [([901, 902], True), ([901, 902], True)]
    assert len(model.generate_calls) == 2
    assert model.generate_calls[0]["inputs"].tolist() == [[11, 12, 13]]
    assert model.generate_calls[1]["inputs"].tolist() == [[21]]
    for call in model.generate_calls:
        assert {key: value for key, value in call.items() if key != "inputs"} == {
            "temperature": 0.2,
            "block_length": 16,
            "steps": 17,
            "gen_length": 64,
            "eos_early_stop": True,
            "threshold": 0.7,
            "editing_threshold": 0.0,
            "max_post_steps": 16,
            "eos_id": 156892,
            "mask_id": 156895,
        }


@pytest.mark.parametrize(("mask_id", "eos_id"), [(None, 156892), (156895, None)])
def test_generate_llada2_requires_special_token_ids(mask_id, eos_id):
    with pytest.raises(ValueError, match="mask and EOS token IDs"):
        generate_llada2(_FakeLLaDA2(), _FakeTokenizer(), [[1]], LLaDA2Sampler.default_config, mask_id, eos_id)


class _FakeShiftModel(torch.nn.Module):
    """Causal LM stub whose row ``i`` predicts a position-determined token.

    Row ``i`` votes for token id ``i % (vocab-1)`` (always a real token, never
    ``mask_id = vocab-1``) with strictly-decreasing confidence in ``i`` so ties
    never arise. The prediction depends only on the position, so a shifted
    (``logit_shift=1``) decode fills mask ``p`` with ``(p-1) % (vocab-1)`` while
    an unshifted decode fills it with ``p % (vocab-1)`` — isolating the shift.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Args: input_ids: Tensor of shape [batch, sequence]."""
        B, T = input_ids.shape
        V = self.vocab_size
        # Non-target logits at 0; target logit stays positive (always the argmax)
        # but decreases with position so softmax confidence is strictly ordered
        # (earlier positions win topk first) with resolvable, non-tied gaps.
        logits = torch.zeros((B, T, V), device=input_ids.device)
        for i in range(T):
            logits[:, i, i % (V - 1)] = 5.0 - 0.5 * i
        return type("_Out", (), {"logits": logits})()


def test_idlm_sampler_is_registered_with_shift_and_strided_defaults():
    assert SAMPLERS["idlm"] is IDLMSampler
    assert IDLMSampler.logit_shift == 1
    config = IDLMSampler.default_config
    assert config.block_size == 4  # paper stride N=4
    assert config.use_kv_cache is False  # shifted decode is full-forward only
    assert config.remasking == "low_confidence"


def test_idlm_shift_rejects_kv_cache():
    sampler = IDLMSampler(_FakeShiftModel(vocab_size=8), mask_id=7, eos_id=0)
    with pytest.raises(ValueError, match="use_kv_cache=False"):
        sampler.sample([[1, 2]], use_kv_cache=True)


def test_idlm_shift_reads_the_preceding_position():
    """The shifted decode fills mask ``p`` from the logit at ``p-1``.

    With the position-only stub, a correct ``logit_shift=1`` decode yields
    ``(p-1) % (vocab-1)`` at each generated position; the unshifted decode yields
    ``p % (vocab-1)``. Asserting both proves the shift is load-bearing.
    """
    vocab, mask_id, eos_id = 16, 15, 0
    model = _FakeShiftModel(vocab_size=vocab)
    prompt = [1, 2, 3]
    kwargs = dict(block_size=1, max_new_tokens=4, steps=4, threshold=None, eos_token_id=None)

    out = IDLMSampler(model, mask_id=mask_id, eos_id=eos_id).sample([prompt], **kwargs)
    assert out[0, :3].tolist() == prompt  # prompt preserved
    assert (out == mask_id).sum().item() == 0  # every mask resolved
    assert out[0, 3:].tolist() == [(p - 1) % (vocab - 1) for p in range(3, 7)]

    unshifted = IDLMSampler(model, mask_id=mask_id, eos_id=eos_id)
    unshifted.logit_shift = 0
    out0 = unshifted.sample([prompt], **kwargs)
    assert out0[0, 3:].tolist() == [p % (vocab - 1) for p in range(3, 7)]
    assert not torch.equal(out0, out)


def test_llada2_infill_is_rejected_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", "--checkpoint", "unused", "--prompt", "hello", "--sampler", "llada2", "--infill"],
    )

    with pytest.raises(SystemExit, match="2"):
        main()

    assert "--infill is not supported by the LLaDA2 generation path" in capsys.readouterr().err

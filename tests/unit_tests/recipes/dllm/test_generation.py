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
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

EXAMPLE_DIR = Path(__file__).resolve().parents[4] / "examples" / "dllm_generate"
sys.path.insert(0, str(EXAMPLE_DIR))

from generate import (  # noqa: E402
    SAMPLERS,
    LLaDA2Sampler,
    LLaDASampler,
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


def test_llada2_infill_is_rejected_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", "--checkpoint", "unused", "--prompt", "hello", "--sampler", "llada2", "--infill"],
    )

    with pytest.raises(SystemExit, match="2"):
        main()

    assert "--infill is not supported by the LLaDA2 generation path" in capsys.readouterr().err


class _FakeDenoiser(torch.nn.Module):
    """Rigged denoiser: always predicts token 7, with confidence strictly
    increasing by position. Under multi-block decoding, the highest-confidence
    masked positions therefore always sit in FUTURE blocks — the exact setup
    where selecting over the full sequence (instead of the current block)
    wastes transfer slots and strands mask tokens."""

    def __init__(self, vocab_size: int = 16):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, x, attention_mask=None):
        B, L = x.shape
        logits = torch.zeros(B, L, self.vocab_size)
        logits[:, :, 7] = 5.0 + 0.1 * torch.arange(L, dtype=torch.float32).unsqueeze(0)
        return types.SimpleNamespace(logits=logits)


def test_multi_block_sampling_unmasks_every_scheduled_position():
    """Regression: with block_size < max_new_tokens, out-of-window positions
    must not win top-k transfer slots — every block's schedule must fully
    unmask its own window, leaving zero mask tokens in the output."""
    mask_id = 9
    sampler = LLaDASampler(
        _FakeDenoiser(),
        mask_id=mask_id,
        # A real EOS id (any int distinct from mask_id=9 and the denoiser's
        # prediction 7): sample() fills the canvas with eos_id, so None would
        # break torch.full. 0 keeps the assertions strict — a stranded position
        # would read back as 0, not 7.
        eos_id=0,
        steps=4,
        max_new_tokens=8,
        block_size=4,
        temperature=0.0,
        use_kv_cache=False,
        eos_token_id=None,
    )

    out = sampler.sample([[1, 2]])

    assert (out == mask_id).sum().item() == 0, "residual mask tokens: block schedules were underfilled"
    assert (out[0, 2:] == 7).all(), "every generated position should hold the denoiser's prediction"


def test_nemotron_infill_is_rejected_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", "--checkpoint", "unused", "--prompt", "hello", "--sampler", "nemotron", "--infill"],
    )

    with pytest.raises(SystemExit, match="2"):
        main()

    assert "--infill is not supported by the Nemotron generation path" in capsys.readouterr().err

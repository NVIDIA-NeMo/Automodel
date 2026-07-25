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
    DiffusionGemmaSampler,
    LLaDA2Sampler,
    encode_generation_prompts,
    generate_gemma,
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


class _FakeGemma(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        prompt = kwargs["input_ids"]
        generated = torch.tensor([[901, 902]], device=prompt.device)
        return types.SimpleNamespace(sequences=torch.cat([prompt, generated], dim=1))


def test_gemma_sampler_is_registered_with_hf_generation_defaults():
    assert SAMPLERS["gemma"] is DiffusionGemmaSampler
    config = DiffusionGemmaSampler.default_config
    assert config.max_new_tokens == 256
    assert config.steps == 48


def test_generate_gemma_decodes_generated_only_tokens():
    model = _FakeGemma()
    tokenizer = _FakeTokenizer()
    config = replace(DiffusionGemmaSampler.default_config, max_new_tokens=64, steps=12)

    responses = generate_gemma(model, tokenizer, [[11, 12, 13], [21]], config, eos_id=1)

    assert responses == ["decoded:[901, 902]", "decoded:[901, 902]"]
    assert tokenizer.decode_calls == [([901, 902], True), ([901, 902], True)]
    assert len(model.generate_calls) == 2
    assert model.generate_calls[0]["input_ids"].tolist() == [[11, 12, 13]]
    assert model.generate_calls[1]["input_ids"].tolist() == [[21]]
    for call in model.generate_calls:
        assert {key: value for key, value in call.items() if key != "input_ids"} == {
            "max_new_tokens": 64,
            "max_denoising_steps": 12,
            "eos_token_id": 1,
            "pad_token_id": 1,  # falls back to eos when the tokenizer has no pad token
        }


def test_generate_gemma_requires_eos_token_id():
    with pytest.raises(ValueError, match="EOS token ID"):
        generate_gemma(_FakeGemma(), _FakeTokenizer(), [[1]], DiffusionGemmaSampler.default_config, eos_id=None)


def test_gemma_infill_is_rejected_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate.py", "--checkpoint", "unused", "--prompt", "hello", "--sampler", "gemma", "--infill"],
    )

    with pytest.raises(SystemExit, match="2"):
        main()

    assert "--infill is not supported by the DiffusionGemma generation path" in capsys.readouterr().err


class _TinyProj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


def test_merge_adapter_matches_manual_lora_math(tmp_path):
    peft = pytest.importorskip("peft")
    from utils import merge_adapter

    torch.manual_seed(0)
    model = _TinyProj()
    base_weight = model.proj.weight.detach().clone()

    lora_config = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], init_lora_weights=False)
    peft_model = peft.get_peft_model(model, lora_config)
    lora_layer = peft_model.base_model.model.proj
    lora_a = lora_layer.lora_A["default"].weight.detach().clone()
    lora_b = lora_layer.lora_B["default"].weight.detach().clone()
    peft_model.save_pretrained(tmp_path)  # adapter_config.json + adapter_model.safetensors

    fresh = _TinyProj()
    with torch.no_grad():
        fresh.proj.weight.copy_(base_weight)
    merged = merge_adapter(fresh, str(tmp_path))

    expected = base_weight + (4 / 2) * (lora_b @ lora_a)  # W + (alpha/r) * B @ A
    assert torch.allclose(merged.proj.weight, expected, atol=1e-5)
    assert not isinstance(merged, peft.PeftModel), "adapters must be merged and the PEFT wrapper dropped"


def test_cli_exposes_adapter_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["generate.py", "--help"])
    with pytest.raises(SystemExit, match="0"):
        main()
    assert "--adapter" in capsys.readouterr().out


def test_translate_adapter_reparents_gemma_module_paths(tmp_path):
    pytest.importorskip("safetensors")
    import json

    from safetensors.torch import load_file, save_file
    from utils import GEMMA_ADAPTER_KEY_MAP, translate_adapter

    src = tmp_path / "adapter"
    src.mkdir()
    save_file(
        {"base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight": torch.zeros(2, 2)},
        str(src / "adapter_model.safetensors"),
    )
    (src / "adapter_config.json").write_text(
        json.dumps({"target_modules": ["model.layers.3.self_attn.q_proj"], "r": 2})
    )

    out = Path(translate_adapter(str(src), GEMMA_ADAPTER_KEY_MAP))

    translated = load_file(str(out / "adapter_model.safetensors"))
    assert list(translated) == ["base_model.model.model.decoder.layers.3.self_attn.q_proj.lora_A.weight"]
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["target_modules"] == ["model.decoder.layers.3.self_attn.q_proj"]
    assert cfg["r"] == 2  # non-key fields untouched

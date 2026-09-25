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

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
)

from tests.functional_tests.checkpoint_robustness import test_checkpoint_vllm_deploy as harness
from tests.functional_tests.checkpoint_robustness.test_checkpoint_vllm_deploy import _resolve_args


def test_resolve_args_enables_vllm_expert_parallel_from_recipe(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
model:
  pretrained_model_name_or_path: test/model
ci:
  vllm_smoke_test: true
  vllm_enable_expert_parallel: true
"""
    )

    args = _resolve_args(
        {
            "config_path": str(config_path),
            "deploy_mode": "peft",
            "adapter_path": "/tmp/adapter",
        }
    )

    assert args["enable_expert_parallel"] is True


def test_resolve_args_disables_vllm_expert_parallel_by_default(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
model:
  pretrained_model_name_or_path: test/model
ci:
  vllm_smoke_test: true
"""
    )

    args = _resolve_args(
        {
            "config_path": str(config_path),
            "deploy_mode": "peft",
            "adapter_path": "/tmp/adapter",
        }
    )

    assert args["enable_expert_parallel"] is False


def test_resolve_args_uses_checkpoint_robustness_tokenizer(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
model:
  pretrained_model_name_or_path: test/model
ci:
  checkpoint_robustness:
    tokenizer_name: test/tokenizer
"""
    )

    args = _resolve_args(
        {
            "config_path": str(config_path),
            "deploy_mode": "peft",
            "adapter_path": "/tmp/adapter",
        }
    )

    assert args["tokenizer"] == "test/tokenizer"


def test_resolve_args_prefers_cli_tokenizer_over_recipe(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
model:
  pretrained_model_name_or_path: test/model
ci:
  checkpoint_robustness:
    tokenizer_name: test/recipe-tokenizer
"""
    )

    args = _resolve_args(
        {
            "config_path": str(config_path),
            "deploy_mode": "peft",
            "adapter_path": "/tmp/adapter",
            "tokenizer": "test/cli-tokenizer",
        }
    )

    assert args["tokenizer"] == "test/cli-tokenizer"


@pytest.mark.parametrize("do_sample", [False, None, True])
def test_merged_adapter_saves_generation_config(tmp_path, monkeypatch, do_sample):
    model = GPT2LMHeadModel(GPT2Config(vocab_size=16, n_embd=8, n_layer=1, n_head=1, bos_token_id=1, eos_token_id=2))
    model.generation_config = GenerationConfig(
        do_sample=do_sample,
        top_p=0.95,
        bos_token_id=1,
        eos_token_id=[2, 3],
        pad_token_id=0,
        repetition_penalty=1.2,
        max_length=128,
    )
    peft_model = Mock()
    peft_model.merge_and_unload.return_value = model
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        "model:\n  pretrained_model_name_or_path: test/model\n"
        "ci:\n  checkpoint_robustness:\n    vllm_merge_lora: true\n"
    )
    monkeypatch.setattr(
        harness,
        "_custom_args",
        {"config_path": str(config_path), "deploy_mode": "peft", "adapter_path": str(adapter_path)},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", Mock(return_value=model))
    monkeypatch.setattr(PeftModel, "from_pretrained", Mock(return_value=peft_model))
    tokenizer = Mock(return_value=BatchEncoding({"input_ids": torch.tensor([[1]])}))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    # Stub inference and adapter loading; HF model/config serialization remains real.
    peft_model.generate.return_value = torch.tensor([[1, 4, 5, 6, 7, 8]])
    llm = Mock()
    llm.generate.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4, 5, 6, 7, 8])]) for _ in harness.PROMPTS
    ]
    llm_factory = Mock(return_value=llm)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=llm_factory, SamplingParams=Mock()))

    harness.test_vllm_greedy_matches_hf()

    merged_dir = llm_factory.call_args.kwargs["model"]
    reloaded = GPT2LMHeadModel.from_pretrained(merged_dir)
    saved_config = reloaded.generation_config
    saved_config.validate(strict=True)
    assert saved_config.do_sample is do_sample
    assert saved_config.top_p == (0.95 if do_sample else 1.0)
    assert saved_config.bos_token_id == 1
    assert saved_config.eos_token_id == [2, 3]
    assert saved_config.pad_token_id == 0
    assert saved_config.repetition_penalty == 1.2
    assert saved_config.max_length == 128
    assert peft_model.generate.call_args.kwargs["do_sample"] is False

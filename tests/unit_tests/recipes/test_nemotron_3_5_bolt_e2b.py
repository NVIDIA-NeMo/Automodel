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

"""Bolt E2B recipes and prompt behavior must retain the checkpoint's contracts."""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.components.datasets.vlm.collate_fns import build_labels_from_template
from nemo_automodel.recipes._dist_utils import parse_distributed_section

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[3]
RECIPES = ROOT / "examples/vlm_finetune/nemotron_3_5_bolt"
TEMPLATE = RECIPES / "nemotron_3_5_bolt_e2b_chat_template.jinja"


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerFast:
    """Use an offline tokenizer so prompt and mask checks require no Hub access."""
    words = [
        "[UNK]",
        "<pad>",
        "<bos>",
        "<eos>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<channel|>",
        "<|think|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "model",
        "user",
        "system",
        "thought",
        "Question",
        "Answer",
        "Policy",
        "u",
        "XSENTINELMARKERX",
    ]
    engine = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
    engine.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=engine,
        unk_token="[UNK]",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=[word for word in words if word.startswith("<")],
        chat_template=TEMPLATE.read_text(),
        padding_side="right",
    )


@pytest.mark.parametrize("peft", [False, True])
def test_e2b_recipe_builds(peft: bool) -> None:
    """Resolve the real loader and typed precision/PEFT configs without downloading weights."""
    suffix = "_peft" if peft else ""
    config = load_yaml_config(RECIPES / f"nemotron_3_5_bolt_e2b{suffix}.yaml")
    assert config.model.pretrained_model_name_or_path == "nvidia/NVIDIA-Nemotron-3.5-Bolt-E2B-BF16"
    assert config.model.dtype == ("bfloat16" if peft else "float32")
    assert config.model.attn_implementation == ("sdpa" if peft else "eager")
    assert config.ci.nproc_per_node == (2 if peft else 8)
    assert config.optimizer.lr == (2e-4 if peft else 1e-5)
    assert config.dataset.path_or_dataset == "mmoukouba/MedPix-VQA"
    assert config.model.text_config.use_cache is False
    assert config.distributed.activation_checkpointing is True
    strategy = parse_distributed_section(config.distributed.raw_config)["strategy_config"]
    assert strategy.mp_policy.param_dtype == torch.bfloat16
    assert strategy.mp_policy.reduce_dtype == torch.float32
    for section in (config.dataset, config.validation_dataset):
        assert ROOT / section.chat_template == TEMPLATE
    assert config.dataloader.collate_fn.max_length == 2048
    assert config.validation_dataloader.collate_fn.max_length == 2048
    if peft:
        adapter = config.peft.instantiate()
        assert isinstance(adapter, PeftConfig)
        assert adapter.dim == 16
        assert all("language_model.layers." in name for name in adapter.target_modules)
    # The processor has no independent checkpoint ID: the VLM recipe uses the overridden model path.
    assert "pretrained_model_name_or_path" not in config.processor
    assert config.ci.nemo_ci_schedule == "skipped"


@pytest.mark.parametrize("thinking", [False, True])
def test_e2b_thinking_prompt(tokenizer: PreTrainedTokenizerFast, thinking: bool) -> None:
    """Thinking is opt-in; disabled thinking must not inject a synthetic thought suffix."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    assert ("<|think|>" in prompt) is thinking
    assert prompt.endswith("<|turn>model\n")
    assert "<|channel>thought" not in prompt


@pytest.mark.parametrize("thinking", [False, True])
def test_e2b_tool_response_continues_mixed_assistant_turn(
    tokenizer: PreTrainedTokenizerFast,
    thinking: bool,
) -> None:
    """Text preceding a tool call must not terminate the assistant after the response."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "Answer",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": {"city": "Austin"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    assert "Answer<|tool_call>" in prompt
    assert prompt.endswith("<tool_response|>" + ("<|channel>thought\n" if thinking else ""))
    assert prompt.count("<|turn>model\n") == 1


def test_e2b_conversational_prompt(tokenizer: PreTrainedTokenizerFast) -> None:
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": "Policy"}, {"role": "user", "content": "Question"}],
        tokenize=False,
        conversational_mode=True,
    )
    assert "Policy" in prompt
    assert "spoken conversation" in prompt
    assert prompt.count("<|turn>system") == 1


def test_e2b_labels_mask_prompt_and_padding(tokenizer: PreTrainedTokenizerFast) -> None:
    """Assistant answer and turn terminator are supervised; user/system/padding are ignored."""
    conversations = [
        [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}],
    ]
    ids = tokenizer.apply_chat_template(conversations, tokenize=True, padding=True, return_tensors="pt")["input_ids"]
    labels = build_labels_from_template(ids, conversations, tokenizer)
    for row in labels:
        assert row[row != -100].tolist() == tokenizer.encode("Answer<turn|>", add_special_tokens=False)
    assert (labels[ids == tokenizer.pad_token_id] == -100).all()

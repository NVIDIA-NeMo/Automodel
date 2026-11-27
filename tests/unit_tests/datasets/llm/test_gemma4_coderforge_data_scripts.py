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

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[4]
RECIPE_DIR = REPO_ROOT / "examples/long_context_validation/gemma4_31B"
DATA_DIR = RECIPE_DIR / "data"
RECIPE_YAML = RECIPE_DIR / "gemma4_31b_base_coderforge_cp8_64k_1e5_800steps.yaml"


def _load_script(name):
    """Load a CoderForge data script (the example directory is not a package)."""
    spec = importlib.util.spec_from_file_location(f"gemma4_coderforge_{name}", DATA_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prefilter():
    return _load_script("prefilter_dataset")


def test_clean_message_keeps_json_string_arguments_and_fills_empty_ones(prefilter):
    # Arguments stay JSON strings (what the JSONL cache and ChatDataset carry); NeMoAutoTokenizer
    # decodes JSON objects for the Gemma4 template, which raises on any other string, so a call
    # without arguments must become "{}" rather than "".
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}},
            {"function": {"name": "think", "arguments": None}},
            {"function": {"name": "finish", "arguments": ""}},
            {"function": {"name": "edit", "arguments": {"path": "a.py"}}},
        ],
        "function_call": None,
    }
    out = prefilter.clean_message(msg)
    assert out["content"] == ""
    assert [call["function"] for call in out["tool_calls"]] == [
        {"name": "bash", "arguments": '{"cmd": "ls"}'},
        {"name": "think", "arguments": "{}"},
        {"name": "finish", "arguments": "{}"},
        {"name": "edit", "arguments": '{"path": "a.py"}'},
    ]
    assert [call["id"] for call in out["tool_calls"]] == ["a", "call_1", "call_2", "call_3"]


def test_analyzed_cache_path_keys_on_template_contents(prefilter, tmp_path):
    # n_tokens depends on the rendering, so editing the template in place must miss the old cache,
    # while the same contents keep hitting it.
    template = tmp_path / "chat_template.jinja"
    template.write_text("v1")

    def path():
        return prefilter.analyzed_cache_path(str(tmp_path), "org/ds", "train", "/models/gemma", str(template))

    first = path()
    assert path() == first
    template.write_text("v2")
    assert path() != first
    # A template that is not a file (inline Jinja, or None for the checkpoint's own) keys on its value.
    assert prefilter.analyzed_cache_path(str(tmp_path), "org/ds", "train", "/models/gemma", None).endswith(
        "_analyzed.jsonl"
    )


def test_validate_accepts_either_gemma4_stop_token():
    # CoderForge trajectories end on a `finish` tool call, so the final supervised token is the
    # <|tool_response> stop token (50) rather than <turn|> (106); both are generation stop tokens.
    validate_data = _load_script("validate_data")
    ends_on_tool_call = {"input_ids": [2, 7, 8, 50], "labels": [-100, 7, 8, 50]}
    ends_on_turn = {"input_ids": [2, 7, 106, 9], "labels": [-100, 7, 106, 9]}
    truncated = {"input_ids": [2, 7, 8, 9], "labels": [-100, 7, 8, 9]}
    # Cut inside a masked tool result: the supervision ends on the previous call's 50, not at the end.
    truncated_in_tool_result = {"input_ids": [2, 7, 50, 30, 31, 32], "labels": [-100, 7, 50, -100, -100, -100]}

    _, passed, failed, _, n = validate_data.validate(
        [ends_on_tool_call, ends_on_turn, truncated, truncated_in_tool_result],
        pad_token_id=0,
        stop_token_ids=[106, 50],
        terminator_window=2,
        num_samples=0,
    )
    assert n == 4
    assert passed["final_turn_terminated"] == 2 and failed["final_turn_terminated"] == 2
    assert passed["stop_token_in_supervised"] == 3 and failed["stop_token_in_supervised"] == 1


def _byte_level_gemma4_tokenizer():
    """Byte-level tokenizer that knows the Gemma4 control tokens the template emits."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>")
    tok.add_tokens(
        ["<|turn>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>']
        + ["<|tool>", "<tool|>", "<|channel>", "<channel|>"],
        special_tokens=True,
    )
    return tok


def test_recipe_dataset_reads_a_tool_calling_row(tmp_path, monkeypatch):
    # Build the dataset the way FinetuneRecipeForVLM does: the recipe drops `dataset.tokenizer` and
    # hands ChatDataset the processor's tokenizer, so the JSON-string arguments that the prefilter
    # writes must be decoded on that path too, or the Gemma4 template raises on every row.
    from transformers import ProcessorMixin

    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.components.datasets.vlm import loader as vlm_loader
    from nemo_automodel.recipes._typed_config import RecipeConfig

    class _Processor(ProcessorMixin):
        attributes = []

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    cfg = yaml.safe_load(RECIPE_YAML.read_text())
    assert cfg["recipe"] == "FinetuneRecipeForVLM"
    row = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "Look first:",
                "tool_calls": [
                    {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c0", "content": "a.py"},
            {"role": "assistant", "content": "Done."},
        ],
        "tools": [],
    }
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(row) + "\n")
    cfg["dataset"]["path_or_dataset_id"] = str(data)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "gemma4"}')
    tok = _byte_level_gemma4_tokenizer()
    monkeypatch.setattr(vlm_loader.AutoProcessor, "from_pretrained", lambda *args, **kwargs: _Processor(tok))
    monkeypatch.chdir(REPO_ROOT)  # the YAML's chat_template path is relative to the repo root

    dataset, _ = RecipeConfig(ConfigNode(cfg)).vlm_dataloader._build_source(
        pretrained_model_name_or_path=str(model_dir), dp_rank=0, dp_world_size=1, dataset_build_context=None
    )
    labels = dataset[0]["labels"]
    assert tok.decode([t for t in labels if t != -100]) == (
        'Look first:<|tool_call>call:bash{cmd:<|"|>ls<|"|>}<tool_call|><|tool_response>Done.<turn|>\n'
    )

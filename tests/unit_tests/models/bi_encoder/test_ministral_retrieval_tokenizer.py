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

"""Tests for the retrieval-specific Ministral tokenizer policy."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from nemo_automodel import NeMoAutoTokenizer


def test_loader_uses_tokenizers_backend_directly_with_retrieval_policy(monkeypatch):
    expected = SimpleNamespace(pad_token_id=7)
    calls = []

    def fake_from_pretrained(model_id, *args, **kwargs):
        calls.append((model_id, args, kwargs))
        return expected

    monkeypatch.setattr(TokenizersBackend, "from_pretrained", fake_from_pretrained)

    actual = NeMoAutoTokenizer.from_pretrained(
        "mistralai/Ministral-3-3B-Instruct-2512",
        tokenizer_backend="tokenizers",
        fix_mistral_regex=True,
        split_special_tokens=True,
        add_bos_token=True,
        add_eos_token=False,
    )

    assert actual is expected
    assert calls == [
        (
            "mistralai/Ministral-3-3B-Instruct-2512",
            (),
            {
                "fix_mistral_regex": True,
                "split_special_tokens": True,
                "add_bos_token": True,
                "add_eos_token": False,
            },
        )
    ]


def test_facade_rejects_unknown_tokenizer_backend():
    with pytest.raises(ValueError, match="tokenizer_backend must be one of"):
        NeMoAutoTokenizer.from_pretrained("unused", tokenizer_backend="unknown")


def test_loaded_tokenizer_policy_survives_save_and_auto_reload(tmp_path):
    source_dir = tmp_path / "source"
    saved_dir = tmp_path / "saved"
    backend = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<s>": 1, "</s>": 2, "[INST]": 3, "hello": 4, "[": 5, "INST": 6, "]": 7},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    source_tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["[INST]"],
    )
    source_tokenizer.save_pretrained(source_dir)
    (source_dir / "tekken.json").write_text("{}")

    tokenizer = NeMoAutoTokenizer.from_pretrained(
        source_dir,
        tokenizer_backend="tokenizers",
        fix_mistral_regex=True,
        split_special_tokens=True,
        add_bos_token=True,
        add_eos_token=False,
    )
    runtime_text_ids = tokenizer.encode("hello")
    runtime_literal_ids = tokenizer.encode("[INST]", add_special_tokens=False)
    tokenizer.save_pretrained(saved_dir)
    reloaded = AutoTokenizer.from_pretrained(saved_dir)

    saved_config = json.loads((saved_dir / "tokenizer_config.json").read_text())

    assert isinstance(tokenizer, TokenizersBackend)
    assert isinstance(reloaded, TokenizersBackend)
    assert runtime_text_ids == [1, 4]
    assert runtime_literal_ids == [5, 6, 7]
    assert reloaded.encode("hello") == runtime_text_ids
    assert reloaded.encode("[INST]", add_special_tokens=False) == runtime_literal_ids
    assert saved_config["split_special_tokens"] is True
    assert "auto_map" not in saved_config
    assert not (saved_dir / "tekken.json").exists()
    assert not list(saved_dir.glob("*.py"))


def test_all_maintained_ministral_retrieval_recipes_use_tokenizers_backend_policy():
    recipe_dir = Path(__file__).parents[4] / "examples" / "retrieval" / "bi_encoder"
    recipe_paths = sorted(path for path in recipe_dir.glob("*ministral*.yaml") if path.stat().st_size)

    assert recipe_paths
    for recipe_path in recipe_paths:
        tokenizer_config = yaml.safe_load(recipe_path.read_text())["tokenizer"]
        if tokenizer_config["_target_"] == (
            "nemo_automodel.components.models.ministral_bidirectional.processor."
            "Mistral3BiEncoderProcessor.from_pretrained"
        ):
            continue

        assert tokenizer_config["_target_"] == "nemo_automodel.NeMoAutoTokenizer.from_pretrained"
        assert tokenizer_config["tokenizer_backend"] == "tokenizers"
        assert tokenizer_config["fix_mistral_regex"] is True
        assert tokenizer_config["split_special_tokens"] is True
        assert tokenizer_config["add_bos_token"] is True
        assert tokenizer_config["add_eos_token"] is False

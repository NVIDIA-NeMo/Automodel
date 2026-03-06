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

"""Functional tests for save_pretrained v4 compatibility.

Verifies that loading tokenizer assets with NeMoAutoTokenizer (on transformers
v5) and saving them back produces a ``tokenizer_config.json`` that is
consumable by downstream tools still on transformers v4.

Tokenizer assets should be pulled **raw** from the HF Hub (e.g. via
``huggingface-cli download``) so the source files are unmodified.

To set up test data::

    huggingface-cli download nvidia/NVIDIA-Nemotron-Nano-9B-v2 \\
        --include "tokenizer*" "special_tokens_map.json" \\
        --local-dir $TEST_DATA_DIR/tokenizers/NVIDIA-Nemotron-Nano-9B-v2
"""

import json
import os
from pathlib import Path

import pytest

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
    _PRESERVED_SPECIAL_TOKEN_KEYS,
)

_TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "/home/TestData/automodel")
_TOKENIZER_BASE = Path(_TEST_DATA_DIR) / "tokenizers"
NEMOTRON_NANO_PATH = _TOKENIZER_BASE / "NVIDIA-Nemotron-Nano-9B-v2"


@pytest.fixture
def nemotron_nano_path():
    assert NEMOTRON_NANO_PATH.exists(), (
        f"Tokenizer assets not found at {NEMOTRON_NANO_PATH}. "
        "Download them with: huggingface-cli download nvidia/NVIDIA-Nemotron-Nano-9B-v2 "
        '--include "tokenizer*" "special_tokens_map.json" '
        f"--local-dir {NEMOTRON_NANO_PATH}"
    )
    return str(NEMOTRON_NANO_PATH)


@pytest.fixture
def original_config(nemotron_nano_path):
    """The unmodified tokenizer_config.json straight from the hub."""
    with open(os.path.join(nemotron_nano_path, "tokenizer_config.json")) as f:
        return json.load(f)


class TestSavePretrainedV4Compat:
    """Load real tokenizer assets, save them, and verify v4 compatibility."""

    def test_tokenizer_class_preserved(self, nemotron_nano_path, original_config, tmp_path):
        """tokenizer_class must match the original, not TokenizersBackend."""
        tok = NeMoAutoTokenizer.from_pretrained(nemotron_nano_path)
        tok.save_pretrained(str(tmp_path))

        with open(tmp_path / "tokenizer_config.json") as f:
            saved = json.load(f)

        original_cls = original_config.get("tokenizer_class")
        assert saved["tokenizer_class"] == original_cls, (
            f"tokenizer_class changed from {original_cls!r} to {saved['tokenizer_class']!r}"
        )

    def test_special_tokens_preserved(self, nemotron_nano_path, original_config, tmp_path):
        """All special token fields from the original config must survive
        a load/save round-trip unchanged."""
        tok = NeMoAutoTokenizer.from_pretrained(nemotron_nano_path)
        tok.save_pretrained(str(tmp_path))

        with open(tmp_path / "tokenizer_config.json") as f:
            saved = json.load(f)

        for key in _PRESERVED_SPECIAL_TOKEN_KEYS:
            if key in original_config:
                assert saved.get(key) == original_config[key], (
                    f"{key} changed from {original_config[key]!r} to {saved.get(key)!r}"
                )
            else:
                assert key not in saved, (
                    f"{key} was injected into saved config with value {saved[key]!r} but was not in the original"
                )

    def test_no_runtime_overrides_leak_into_saved_config(self, nemotron_nano_path, original_config, tmp_path):
        """The wrapper forces add_bos_token=True and add_eos_token=True at
        runtime.  These must NOT appear in the saved config unless the original
        already had them."""
        tok = NeMoAutoTokenizer.from_pretrained(nemotron_nano_path)

        assert tok.add_bos_token is True, "Wrapper should force add_bos_token=True at runtime"
        assert tok.add_eos_token is True, "Wrapper should force add_eos_token=True at runtime"

        tok.save_pretrained(str(tmp_path))

        with open(tmp_path / "tokenizer_config.json") as f:
            saved = json.load(f)

        if "add_bos_token" not in original_config:
            assert "add_bos_token" not in saved, "add_bos_token leaked into saved config"
        else:
            assert saved["add_bos_token"] == original_config["add_bos_token"]

        if "add_eos_token" not in original_config:
            assert "add_eos_token" not in saved, "add_eos_token leaked into saved config"
        else:
            assert saved["add_eos_token"] == original_config["add_eos_token"]

    def test_encode_decode_roundtrip_after_reload(self, nemotron_nano_path, tmp_path):
        """Save and reload: the re-loaded tokenizer must produce identical
        token IDs for the same input text."""
        tok_original = NeMoAutoTokenizer.from_pretrained(nemotron_nano_path)

        tok_original.save_pretrained(str(tmp_path))
        tok_reloaded = NeMoAutoTokenizer.from_pretrained(str(tmp_path))

        text = "The quick brown fox jumps over the lazy dog."
        ids_original = tok_original.encode(text)
        ids_reloaded = tok_reloaded.encode(text)
        assert ids_original == ids_reloaded, "Token IDs differ after save/reload round-trip"

        decoded = tok_reloaded.decode(ids_reloaded, skip_special_tokens=True)
        assert text in decoded

    def test_saved_config_is_valid_json(self, nemotron_nano_path, tmp_path):
        """Basic sanity: the saved tokenizer_config.json must be parseable."""
        tok = NeMoAutoTokenizer.from_pretrained(nemotron_nano_path)
        tok.save_pretrained(str(tmp_path))

        with open(tmp_path / "tokenizer_config.json") as f:
            config = json.load(f)

        assert isinstance(config, dict)
        assert "tokenizer_class" in config

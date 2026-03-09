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

"""Functional test for Ministral tokenizer save/load round-trip.

Verifies that the Ministral tokenizer (MistralCommonBackend) is correctly:
  1. Recognized by ``is_tokenizer()`` in the checkpoint save path
  2. Saved with all required files (tekken.json, tokenizer_config.json, etc.)
  3. Reloadable from the saved directory

This test requires ``mistral-common`` to be installed and the Ministral model
assets to be available locally or in the HF cache.

To set up test data::

    huggingface-cli download mistralai/Ministral-3B-Instruct-2412 \
        --include "tokenizer*" "tekken.json" "config.json" "special_tokens_map.json" \
        --local-dir $TEST_DATA_DIR/tokenizers/Ministral-3B-Instruct-2412

Can also be run standalone::

    python tests/functional_tests/data/llm/test_ministral_tokenizer_save.py \
        [--model-path PATH_OR_HF_ID]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "/home/TestData/automodel")
_TOKENIZER_BASE = Path(_TEST_DATA_DIR) / "tokenizers"
MINISTRAL_PATH = _TOKENIZER_BASE / "Ministral-3B-Instruct-2412"
MINISTRAL_HF_ID = "mistralai/Ministral-3B-Instruct-2412"

EXPECTED_TOKENIZER_FILES = {"tekken.json", "tokenizer_config.json"}


def _resolve_model_path(model_path: str | None = None) -> str:
    """Resolve a usable model path for the Ministral tokenizer."""
    if model_path:
        return model_path
    if MINISTRAL_PATH.exists():
        return str(MINISTRAL_PATH)
    return MINISTRAL_HF_ID


def _load_tokenizer_via_auto(model_path: str):
    from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

    return NeMoAutoTokenizer.from_pretrained(model_path)


def _load_tokenizer_via_registry(model_path: str):
    from nemo_automodel._transformers.tokenization.tokenization_mistral_common import (
        MistralCommonBackend,
    )

    return MistralCommonBackend.from_pretrained(model_path)


def _check_is_tokenizer(tokenizer) -> bool:
    from nemo_automodel.recipes.base_recipe import is_tokenizer

    return is_tokenizer(tokenizer)


def _check_saved_files(save_dir: str) -> dict:
    """Return diagnostics about files in save_dir."""
    files = sorted(os.listdir(save_dir))
    tokenizer_files = [f for f in files if _is_tokenizer_like(f)]
    return {
        "all_files": files,
        "tokenizer_files": tokenizer_files,
        "has_tekken": "tekken.json" in files,
        "has_tokenizer_config": "tokenizer_config.json" in files,
        "has_special_tokens_map": "special_tokens_map.json" in files,
    }


def _is_tokenizer_like(filename: str) -> bool:
    import fnmatch

    patterns = (
        "tokenizer*",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.*",
        "merges.txt",
        "spiece.model",
        "tekken.json",
    )
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------


@pytest.fixture
def model_path():
    path = _resolve_model_path()
    if not os.path.isdir(path) and path == MINISTRAL_HF_ID:
        pytest.skip("Ministral tokenizer assets not found locally; set TEST_DATA_DIR or download them.")
    return path


class TestMinistralTokenizerSave:
    """Verify Ministral tokenizer save produces required files."""

    def test_is_tokenizer_recognizes_mistral_common_backend(self, model_path):
        """is_tokenizer() must return True for MistralCommonBackend instances."""
        tok = _load_tokenizer_via_auto(model_path)
        assert _check_is_tokenizer(tok), (
            f"is_tokenizer() returned False for {type(tok).__name__}. MRO: {[c.__name__ for c in type(tok).__mro__]}"
        )

    def test_save_pretrained_produces_tokenizer_files(self, model_path, tmp_path):
        """save_pretrained must produce tekken.json and tokenizer_config.json."""
        tok = _load_tokenizer_via_auto(model_path)
        tok.save_pretrained(str(tmp_path))

        info = _check_saved_files(str(tmp_path))
        assert info["tokenizer_files"], f"No tokenizer files found after save_pretrained. Files: {info['all_files']}"
        for expected in EXPECTED_TOKENIZER_FILES:
            assert expected in info["all_files"], f"Missing {expected} in saved directory. Files: {info['all_files']}"

    def test_tokenizer_config_json_generated(self, model_path, tmp_path):
        """tokenizer_config.json must be generated even when absent from source."""
        tok = _load_tokenizer_via_auto(model_path)
        tok.save_pretrained(str(tmp_path))

        config_path = tmp_path / "tokenizer_config.json"
        assert config_path.exists(), f"tokenizer_config.json not generated. Files: {sorted(os.listdir(str(tmp_path)))}"
        with open(config_path) as f:
            config = json.load(f)
        assert "tokenizer_class" in config

    def test_encode_decode_roundtrip_direct(self, model_path, tmp_path):
        """Token IDs must be identical when reloading via MistralCommonBackend directly."""
        tok = _load_tokenizer_via_auto(model_path)
        tok.save_pretrained(str(tmp_path))

        tok_reloaded = _load_tokenizer_via_registry(str(tmp_path))
        text = "The quick brown fox jumps over the lazy dog."
        ids_orig = tok.encode(text)
        ids_reload = tok_reloaded.encode(text)
        assert ids_orig == ids_reload, (
            f"Token IDs differ after round-trip.\n  original: {ids_orig}\n  reloaded: {ids_reload}"
        )

    def test_ministral3_in_tokenizer_registry(self):
        """ministral3 model_type must map to MistralCommonBackend in the registry."""
        from nemo_automodel._transformers.tokenization.registry import TokenizerRegistry

        cls = TokenizerRegistry.get_custom_tokenizer_cls("ministral3")
        assert cls is not None, (
            "ministral3 is not registered in TokenizerRegistry. "
            "Output models with model_type='ministral3' will not use MistralCommonBackend."
        )


# ---------------------------------------------------------------------------
# Standalone runner with diagnostics
# ---------------------------------------------------------------------------


def _run_diagnostics(model_path: str):
    """Run all checks and print detailed diagnostics."""
    print(f"{'=' * 70}")
    print("Ministral Tokenizer Save Diagnostics")
    print(f"{'=' * 70}")
    print(f"Model path: {model_path}")
    print()

    # 1. Check registry
    print("[1/6] Checking tokenizer registry...")
    from nemo_automodel._transformers.tokenization.registry import TokenizerRegistry

    for mt in ("mistral", "mistral3", "ministral3", "pixtral"):
        cls = TokenizerRegistry.get_custom_tokenizer_cls(mt)
        status = cls.__name__ if cls else "NOT REGISTERED"
        print(f"  model_type={mt!r:15s} -> {status}")
    print()

    # 2. Check model_type detection
    print("[2/6] Detecting model_type from config...")
    from nemo_automodel._transformers.auto_tokenizer import _get_model_type

    model_type = _get_model_type(model_path)
    print(f"  model_type = {model_type!r}")
    print()

    # 3. Load tokenizer
    print("[3/6] Loading tokenizer via NeMoAutoTokenizer...")
    tok = _load_tokenizer_via_auto(model_path)
    tok_type = type(tok)
    print(f"  type       = {tok_type.__name__}")
    print(f"  module     = {tok_type.__module__}")
    print(f"  MRO        = {[c.__name__ for c in tok_type.__mro__]}")
    print()

    # 4. Check is_tokenizer
    print("[4/6] Checking is_tokenizer()...")
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    is_ptb = isinstance(tok, PreTrainedTokenizerBase)
    is_pm = isinstance(tok, ProcessorMixin)
    is_tok = _check_is_tokenizer(tok)
    print(f"  isinstance(tok, PreTrainedTokenizerBase) = {is_ptb}")
    print(f"  isinstance(tok, ProcessorMixin)           = {is_pm}")
    print(f"  is_tokenizer(tok)                         = {is_tok}")
    if not is_tok:
        print("  *** BUG: is_tokenizer() returns False — tokenizer will NOT be saved during checkpointing! ***")
    print()

    # 5. Test save_pretrained
    print("[5/7] Testing save_pretrained...")
    save_ok = False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tok.save_pretrained(tmp)
            info = _check_saved_files(tmp)
            print(f"  All files:       {info['all_files']}")
            print(f"  Tokenizer files: {info['tokenizer_files']}")
            print(f"  has tekken.json:           {info['has_tekken']}")
            print(f"  has tokenizer_config.json: {info['has_tokenizer_config']}")
            if not info["tokenizer_files"]:
                print("  *** BUG: No tokenizer files produced by save_pretrained! ***")
            elif not info["has_tokenizer_config"]:
                print("  *** WARNING: tokenizer_config.json missing — downstream tools may fail ***")
            else:
                save_ok = True
        except Exception as e:
            print(f"  *** ERROR during save_pretrained: {e} ***")
            import traceback

            traceback.print_exc()
    print()

    # 6. Test reload via MistralCommonBackend (simulates checkpoint load with config.json routing)
    print("[6/7] Testing save/reload round-trip (MistralCommonBackend direct)...")
    roundtrip_direct_ok = False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tok.save_pretrained(tmp)
            tok2 = _load_tokenizer_via_registry(tmp)
            text = "The quick brown fox jumps over the lazy dog."
            ids1 = tok.encode(text)
            ids2 = tok2.encode(text)
            roundtrip_direct_ok = ids1 == ids2
            print(f"  Round-trip IDs match: {roundtrip_direct_ok}")
            if not roundtrip_direct_ok:
                print(f"  original: {ids1}")
                print(f"  reloaded: {ids2}")
        except Exception as e:
            print(f"  *** ERROR during round-trip: {e} ***")
            import traceback

            traceback.print_exc()
    print()

    # 7. Test reload via NeMoAutoTokenizer (needs config.json for routing)
    print("[7/7] Testing save/reload round-trip (NeMoAutoTokenizer with config.json)...")
    roundtrip_auto_ok = False
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tok.save_pretrained(tmp)
            config_json = os.path.join(tmp, "config.json")
            if not os.path.exists(config_json):
                with open(config_json, "w") as f:
                    json.dump({"model_type": model_type or "mistral3"}, f)
                print(f"  (injected config.json with model_type={model_type or 'mistral3'!r})")
            tok2 = _load_tokenizer_via_auto(tmp)
            text = "The quick brown fox jumps over the lazy dog."
            ids1 = tok.encode(text)
            ids2 = tok2.encode(text)
            roundtrip_auto_ok = ids1 == ids2
            print(f"  Round-trip IDs match: {roundtrip_auto_ok}")
            if not roundtrip_auto_ok:
                print(f"  original: {ids1}")
                print(f"  reloaded: {ids2}")
        except Exception as e:
            print(f"  *** ERROR during round-trip: {e} ***")
            import traceback

            traceback.print_exc()

    print()
    print(f"{'=' * 70}")
    all_ok = is_tok and save_ok and roundtrip_direct_ok and roundtrip_auto_ok
    print(f"is_tokenizer()          : {'PASS' if is_tok else 'FAIL'}")
    print(f"save_pretrained()       : {'PASS' if save_ok else 'FAIL'}")
    print(f"round-trip (direct)     : {'PASS' if roundtrip_direct_ok else 'FAIL'}")
    print(f"round-trip (auto+config): {'PASS' if roundtrip_auto_ok else 'FAIL'}")
    print(f"Overall                 : {'PASS' if all_ok else 'FAIL'}")
    print(f"{'=' * 70}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ministral tokenizer save diagnostics")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path or HF model ID for the Ministral model (default: auto-detect)",
    )
    args = parser.parse_args()
    model_path = _resolve_model_path(args.model_path)
    sys.exit(_run_diagnostics(model_path))

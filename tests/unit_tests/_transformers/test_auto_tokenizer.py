# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from unittest.mock import patch

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
    _add_token,
    _read_tokenizer_class,
    _read_tokenizer_config,
    _remap_system_role,
    _restore_tokenizer_config,
    _try_convert_tiktoken_to_native,
)


class _StubHFTokenizer:
    def __init__(self, bos_id=101, eos_id=102):
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.add_bos_token = True
        self.add_eos_token = True

    def __call__(self, *args, **kwargs):
        return BatchEncoding(
            {
                "input_ids": [[5, 6]],
                "attention_mask": [[1, 1]],
                "assistant_masks": [[0, 1]],
            }
        )

    def encode(self, *args, **kwargs):
        return [5, 6]


class _TikTokenLikeTokenizer:
    """Stub that mimics a TikToken tokenizer without add_bos/eos_token attributes."""

    def __init__(self, bos_id=101, eos_id=102):
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        # Deliberately does NOT have add_bos_token / add_eos_token

    def __call__(self, *args, **kwargs):
        return BatchEncoding(
            {
                "input_ids": [[5, 6]],
                "attention_mask": [[1, 1]],
            }
        )

    def encode(self, *args, **kwargs):
        return [5, 6]


class _StubConfig:
    model_type = "stub"


class TestNeMoAutoTokenizerFromPretrained:
    def test_patched_adds_bos_eos(self):
        stub = _StubHFTokenizer()
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            out = tok(["x"])
            assert isinstance(out, BatchEncoding)
            assert out["input_ids"] == [[stub.bos_token_id, 5, 6, stub.eos_token_id]]
            assert out["attention_mask"] == [[1, 1, 1, 1]]
            assert out["assistant_masks"] == [[1, 0, 1, 1]]

            out = tok(["x"], add_special_tokens=False)
            assert isinstance(out, BatchEncoding)
            assert out["input_ids"] == [[5, 6]]
            assert out["attention_mask"] == [[1, 1]]
            assert out["assistant_masks"] == [[0, 1]]

            enc = tok.encode("x")
            assert enc == [stub.bos_token_id, 5, 6, stub.eos_token_id]

            enc = tok.encode("x", add_special_tokens=False)
            assert enc == [5, 6]

    def test_cls_sep_pattern_fixed_when_tokens_missing(self):
        """Transformers >=5.0 defaults special_tokens_pattern to 'cls_sep'.
        When cls/sep token IDs are None (e.g. Moonlight TikTokenTokenizer),
        the wrapper should reset it to 'none' to avoid None in input_ids."""
        stub = _StubHFTokenizer()
        # Simulate a tokenizer that got the default "cls_sep" pattern but has no CLS/SEP
        stub.special_tokens_pattern = "cls_sep"
        stub.cls_token_id = None
        stub.sep_token_id = None
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok.special_tokens_pattern == "none"

    def test_cls_sep_pattern_preserved_when_tokens_present(self):
        """When cls/sep token IDs are properly defined, the pattern should stay."""
        stub = _StubHFTokenizer()
        stub.special_tokens_pattern = "cls_sep"
        stub.cls_token_id = 200
        stub.sep_token_id = 201
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok.special_tokens_pattern == "cls_sep"

    def test_retry_on_layer_types_mismatch(self):
        """When AutoTokenizer fails because the underlying config has
        layer_types longer than num_hidden_layers (e.g. stepfun-ai/Step-3.5-Flash),
        the wrapper should relax the validator globally and retry."""
        from huggingface_hub.errors import StrictDataclassClassValidationError

        stub = _StubHFTokenizer()
        calls = {"n": 0}

        def fake_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                cause = ValueError("`num_hidden_layers` (45) must be equal to the number of layer types (48).")
                raise StrictDataclassClassValidationError(validator="validate_layer_type", cause=cause)
            return stub

        with (
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_from_pretrained),
            patch(
                "nemo_automodel._transformers.v4_patches.layer_types.relax_layer_types_validator",
                return_value=True,
            ) as mock_relax,
        ):
            tok = NeMoAutoTokenizer.from_pretrained("stepfun-ai/Step-3.5-Flash", trust_remote_code=True)
            assert tok is not None
            assert calls["n"] == 2
            mock_relax.assert_called_once()

    def test_unrelated_value_error_is_not_retried(self):
        """Unrelated ValueErrors should propagate without triggering the fix path."""
        with (
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                side_effect=ValueError("totally unrelated tokenizer failure"),
            ),
            patch("nemo_automodel._transformers.v4_patches.layer_types.relax_layer_types_validator") as mock_relax,
        ):
            with pytest.raises(ValueError, match="totally unrelated"):
                NeMoAutoTokenizer.from_pretrained("dummy/model")
            mock_relax.assert_not_called()

    def test_force_hf_passthrough(self):
        stub = _StubHFTokenizer()
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=stub):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model", force_hf=True)
            # Should be the original stub and unmodified outputs
            out = tok(["x"])
            assert out["input_ids"] == [[5, 6]]
            assert out["attention_mask"] == [[1, 1]]
            assert tok.encode("x") == [5, 6]

    def test_add_bos_token_falls_back_on_value_error(self):
        """When tokenizer.add_bos_token setter raises ValueError (e.g. transformers v5
        read-only property), fall back to setting _add_bos_token directly."""

        class _StrictBosTokenizer(_StubHFTokenizer):
            """Tokenizer whose add_bos_token property raises on set."""

            bos_token = "<s>"
            eos_token = "</s>"

            def __init__(self):
                # Skip parent __init__ which would trigger the strict setter
                self.bos_token_id = 101
                self.eos_token_id = 102
                self.add_eos_token = True

            @property
            def add_bos_token(self):
                return getattr(self, "_add_bos_token", False)

            @add_bos_token.setter
            def add_bos_token(self, value):
                raise ValueError("read-only in this tokenizer version")

        stub = _StrictBosTokenizer()
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok._add_bos_token is True

    def test_add_eos_token_falls_back_on_value_error(self):
        """When tokenizer.add_eos_token setter raises ValueError,
        fall back to setting _add_eos_token directly."""

        class _StrictEosTokenizer(_StubHFTokenizer):
            """Tokenizer whose add_eos_token property raises on set."""

            bos_token = "<s>"
            eos_token = "</s>"

            def __init__(self):
                # Skip parent __init__ which would trigger the strict setter
                self.bos_token_id = 101
                self.eos_token_id = 102
                self.add_bos_token = True

            @property
            def add_eos_token(self):
                return getattr(self, "_add_eos_token", False)

            @add_eos_token.setter
            def add_eos_token(self, value):
                raise ValueError("read-only in this tokenizer version")

        stub = _StrictEosTokenizer()
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok._add_eos_token is True


class TestAddTokenHelper:
    def test_input_ids_single_sequence_no_duplicates(self):
        enc = BatchEncoding({"input_ids": [5, 6]})
        # prepend bos
        _add_token(enc, 101, 0, "input_ids")
        out = enc
        assert out["input_ids"] == [101, 5, 6]
        # append eos
        _add_token(out, 102, -1, "input_ids")
        assert out["input_ids"] == [101, 5, 6, 102]
        # no duplicate prepend
        _add_token(out, 101, 0, "input_ids")
        assert out["input_ids"] == [101, 5, 6, 102]
        # no duplicate append
        _add_token(out, 102, -1, "input_ids")
        assert out["input_ids"] == [101, 5, 6, 102]

    def test_masks_batched_always_extend(self):
        enc = BatchEncoding({"attention_mask": [[1, 1], [1]]})
        # always add on prepend
        _add_token(enc, 1, 0, "attention_mask")
        out = enc
        assert out["attention_mask"] == [[1, 1, 1], [1, 1]]
        # always add on append
        _add_token(out, 1, -1, "attention_mask")
        assert out["attention_mask"] == [[1, 1, 1, 1], [1, 1, 1]]

    def test_empty_sequences(self):
        # input_ids empty
        enc_ids = BatchEncoding({"input_ids": []})
        _add_token(enc_ids, 101, 0, "input_ids")
        out_ids = enc_ids
        assert out_ids["input_ids"] == [101]
        _add_token(out_ids, 102, -1, "input_ids")
        assert out_ids["input_ids"] == [101, 102]
        # masks empty batched
        enc_mask = BatchEncoding({"assistant_masks": [[]]})
        _add_token(enc_mask, 1, 0, "assistant_masks")
        out_mask = enc_mask
        assert out_mask["assistant_masks"] == [[1]]
        _add_token(out_mask, 1, -1, "assistant_masks")
        assert out_mask["assistant_masks"] == [[1, 1]]

    def test_invalid_position_raises(self):
        enc = BatchEncoding({"input_ids": [5, 6]})
        with pytest.raises(ValueError):
            _add_token(enc, 999, 1, "input_ids")

    def test_invalid_sequence_type_raises(self):
        enc = BatchEncoding({"input_ids": "not-a-list"})
        with pytest.raises(ValueError):
            _add_token(enc, 101, 0, "input_ids")


class TestRemapSystemRole:
    """Unit tests for _remap_system_role."""

    def test_merges_system_into_first_user(self):
        conv = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _remap_system_role(conv)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "You are helpful.\nHello"
        assert result[1] == {"role": "assistant", "content": "Hi!"}

    def test_system_removed_from_output(self):
        conv = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Summarize"},
        ]
        result = _remap_system_role(conv)
        assert all(m.get("role") != "system" for m in result)

    def test_no_user_message_creates_one(self):
        conv = [
            {"role": "system", "content": "System prompt only."},
            {"role": "assistant", "content": "OK"},
        ]
        result = _remap_system_role(conv)
        assert result[0] == {"role": "user", "content": "System prompt only."}
        assert result[1] == {"role": "assistant", "content": "OK"}

    def test_multiple_system_messages_raises(self):
        conv = [
            {"role": "system", "content": "First"},
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Second"},
        ]
        with pytest.raises(ValueError, match="System role appeared in multiple messages"):
            _remap_system_role(conv)

    def test_empty_system_content(self):
        conv = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Question"},
        ]
        result = _remap_system_role(conv)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "\nQuestion"

    def test_only_merges_first_user(self):
        conv = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second user"},
        ]
        result = _remap_system_role(conv)
        assert len(result) == 3
        assert result[0]["content"] == "Sys\nFirst user"
        assert result[2]["content"] == "Second user"

    def test_preserves_extra_keys_in_messages(self):
        conv = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hi", "name": "alice"},
        ]
        result = _remap_system_role(conv)
        assert result[0]["name"] == "alice"

    def test_system_without_content_key(self):
        conv = [
            {"role": "system"},
            {"role": "user", "content": "Hello"},
        ]
        result = _remap_system_role(conv)
        assert result[0]["content"] == "\nHello"


class TestReadTokenizerClass:
    """Unit tests for _read_tokenizer_class."""

    def test_reads_from_local_directory(self, tmp_path):
        config = {"tokenizer_class": "PreTrainedTokenizerFast", "model_type": "llama"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
        assert _read_tokenizer_class(str(tmp_path)) == "PreTrainedTokenizerFast"

    def test_returns_none_when_key_missing(self, tmp_path):
        config = {"model_type": "llama"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
        assert _read_tokenizer_class(str(tmp_path)) is None

    def test_returns_none_for_nonexistent_directory(self, tmp_path):
        assert _read_tokenizer_class(str(tmp_path / "does_not_exist")) is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text("{bad json")
        assert _read_tokenizer_class(str(tmp_path)) is None


class TestSavePretrainedPreservesTokenizerClass:
    """Verify that save_pretrained writes the original tokenizer_class, not the
    transformers-v5 runtime class name (e.g. TokenizersBackend)."""

    def _make_stub_tokenizer(self, original_tokenizer_class, base_class_name="TokenizersBackend"):
        """Create a minimal stub that mimics the attributes set by from_pretrained."""
        stub = _StubHFTokenizer()
        stub.bos_token = "<s>"
        stub.eos_token = "</s>"

        base_class = type(base_class_name, (), {})
        stub._base_class = base_class
        stub._original_tokenizer_class = original_tokenizer_class
        return stub

    def test_preserves_original_tokenizer_class_on_save(self, tmp_path):
        """When the source config had 'PreTrainedTokenizerFast', save_pretrained
        should write that same value even if the runtime class is different."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        src_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 131072,
        }
        (src_dir / "tokenizer_config.json").write_text(json.dumps(src_config))

        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from:
            stub = _StubHFTokenizer()
            stub.bos_token = "<s>"
            stub.eos_token = "</s>"
            mock_from.return_value = stub

            with patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()):
                tok = NeMoAutoTokenizer.from_pretrained(str(src_dir))

        assert tok._original_tokenizer_class == "PreTrainedTokenizerFast"

    def test_save_pretrained_writes_original_class_name(self, tmp_path):
        """End-to-end: save_pretrained should produce a tokenizer_config.json
        whose tokenizer_class matches the original, not the v5 runtime class."""

        save_dir = tmp_path / "saved"
        save_dir.mkdir()

        saved_config = {}

        class _FakeSaveBase:
            """Fake base class that records the class name used during save."""

            def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
                saved_config["tokenizer_class"] = type(self).__name__
                config_path = os.path.join(save_directory, "tokenizer_config.json")
                with open(config_path, "w") as f:
                    json.dump(saved_config, f)

        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
            NeMoAutoTokenizerWithBosEosEnforced,
        )

        tok = _StubHFTokenizer()
        tok.bos_token = "<s>"
        tok.eos_token = "</s>"
        tok._base_class = _FakeSaveBase
        tok._original_tokenizer_class = "PreTrainedTokenizerFast"
        tok._original_tokenizer_config = {"tokenizer_class": "PreTrainedTokenizerFast"}

        wrapper_cls = type(
            NeMoAutoTokenizerWithBosEosEnforced.__name__,
            (NeMoAutoTokenizerWithBosEosEnforced, _FakeSaveBase),
            {},
        )
        tok.__class__ = wrapper_cls

        tok.save_pretrained(str(save_dir))

        with open(save_dir / "tokenizer_config.json") as f:
            result = json.load(f)
        assert result["tokenizer_class"] == "PreTrainedTokenizerFast"

    def test_save_pretrained_falls_back_to_base_class_name(self, tmp_path):
        """When _original_tokenizer_class is None, the base class name is used."""
        save_dir = tmp_path / "saved"
        save_dir.mkdir()

        saved_config = {}

        class _FakeSaveBase:
            def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
                saved_config["tokenizer_class"] = type(self).__name__
                config_path = os.path.join(save_directory, "tokenizer_config.json")
                with open(config_path, "w") as f:
                    json.dump(saved_config, f)

        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
            NeMoAutoTokenizerWithBosEosEnforced,
        )

        tok = _StubHFTokenizer()
        tok.bos_token = "<s>"
        tok.eos_token = "</s>"
        tok._base_class = _FakeSaveBase
        tok._original_tokenizer_class = None
        tok._original_tokenizer_config = None

        wrapper_cls = type(
            NeMoAutoTokenizerWithBosEosEnforced.__name__,
            (NeMoAutoTokenizerWithBosEosEnforced, _FakeSaveBase),
            {},
        )
        tok.__class__ = wrapper_cls

        tok.save_pretrained(str(save_dir))

        with open(save_dir / "tokenizer_config.json") as f:
            result = json.load(f)
        assert result["tokenizer_class"] == "_FakeSaveBase"

    def test_save_pretrained_restores_original_class_after_save(self, tmp_path):
        """The __class__ must be restored even if save_pretrained raises."""
        save_dir = tmp_path / "saved"
        save_dir.mkdir()

        class _ExplodingSaveBase:
            def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
                raise RuntimeError("boom")

        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
            NeMoAutoTokenizerWithBosEosEnforced,
        )

        tok = _StubHFTokenizer()
        tok.bos_token = "<s>"
        tok.eos_token = "</s>"
        tok._base_class = _ExplodingSaveBase
        tok._original_tokenizer_class = "PreTrainedTokenizerFast"
        tok._original_tokenizer_config = {"tokenizer_class": "PreTrainedTokenizerFast"}

        wrapper_cls = type(
            NeMoAutoTokenizerWithBosEosEnforced.__name__,
            (NeMoAutoTokenizerWithBosEosEnforced, _ExplodingSaveBase),
            {},
        )
        tok.__class__ = wrapper_cls
        original_cls = tok.__class__

        with pytest.raises(RuntimeError, match="boom"):
            tok.save_pretrained(str(save_dir))

        assert tok.__class__ is original_cls


class TestReadTokenizerConfig:
    """Unit tests for _read_tokenizer_config."""

    def test_reads_full_config_from_local_directory(self, tmp_path):
        config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "model_max_length": 131072,
        }
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
        result = _read_tokenizer_config(str(tmp_path))
        assert result == config

    def test_returns_none_for_nonexistent_directory(self, tmp_path):
        assert _read_tokenizer_config(str(tmp_path / "does_not_exist")) is None

    def test_returns_none_on_malformed_json(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text("{bad json")
        assert _read_tokenizer_config(str(tmp_path)) is None

    def test_read_tokenizer_class_delegates_to_config(self, tmp_path):
        """_read_tokenizer_class should return the same value as reading from
        _read_tokenizer_config directly."""
        config = {"tokenizer_class": "LlamaTokenizerFast", "bos_token": "<s>"}
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
        assert _read_tokenizer_class(str(tmp_path)) == "LlamaTokenizerFast"


class TestRestoreTokenizerConfig:
    """Unit tests for _restore_tokenizer_config."""

    def test_replaces_config_with_original(self, tmp_path):
        """The saved config should be replaced wholesale by the original."""
        saved = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "add_bos_token": True,
            "add_eos_token": True,
            "bos_token": {"content": "<s>", "lstrip": False},
            "backend": "tokenizers",
        }
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(saved))

        original = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "model_max_length": 131072,
        }
        _restore_tokenizer_config(str(tmp_path), original)

        with open(tmp_path / "tokenizer_config.json") as f:
            result = json.load(f)
        assert result == original

    def test_preserves_v5_tokenizer_class_when_original_lacks_it(self, tmp_path):
        """When the original config has no tokenizer_class, the v5-written
        value should be carried over."""
        saved = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": {"content": "<s>", "lstrip": False},
        }
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(saved))

        original = {
            "bos_token": "<s>",
            "model_max_length": 4096,
        }
        _restore_tokenizer_config(str(tmp_path), original)

        with open(tmp_path / "tokenizer_config.json") as f:
            result = json.load(f)
        assert result["tokenizer_class"] == "PreTrainedTokenizerFast"
        assert result["bos_token"] == "<s>"
        assert result["model_max_length"] == 4096

    def test_restores_original_special_token_format(self, tmp_path):
        """If v5 serialized bos_token as an object but the original had a
        plain string, the original format should be restored."""
        saved = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": {"content": "<s>", "lstrip": False, "rstrip": False},
            "eos_token": {"content": "</s>", "lstrip": False, "rstrip": False},
        }
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(saved))

        original = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        _restore_tokenizer_config(str(tmp_path), original)

        with open(tmp_path / "tokenizer_config.json") as f:
            result = json.load(f)
        assert result["bos_token"] == "<s>"
        assert result["eos_token"] == "</s>"

    def test_no_op_when_config_matches(self, tmp_path):
        """When saved config already matches the original, the file should not
        be rewritten (content stays identical)."""
        config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        config_path = tmp_path / "tokenizer_config.json"
        config_path.write_text(json.dumps(config))
        mtime_before = config_path.stat().st_mtime_ns

        _restore_tokenizer_config(str(tmp_path), config)

        mtime_after = config_path.stat().st_mtime_ns
        assert mtime_before == mtime_after

    def test_removes_keys_not_in_original(self, tmp_path):
        """Keys present in the saved config but absent from the original
        should be removed after restoration."""
        saved = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "additional_special_tokens": ["<extra_1>", "<extra_2>"],
            "add_bos_token": True,
        }
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(saved))

        original = {"tokenizer_class": "PreTrainedTokenizerFast"}
        _restore_tokenizer_config(str(tmp_path), original)

        with open(tmp_path / "tokenizer_config.json") as f:
            result = json.load(f)
        assert "additional_special_tokens" not in result
        assert "add_bos_token" not in result

    def test_no_crash_when_config_file_missing(self, tmp_path):
        """If tokenizer_config.json does not exist, the function should be a
        silent no-op."""
        _restore_tokenizer_config(str(tmp_path), {"bos_token": "<s>"})


class TestSavePretrainedPreservesSpecialTokens:
    """Verify that save_pretrained restores original special token fields
    in tokenizer_config.json after the base class writes it."""

    def test_full_round_trip_preserves_special_tokens(self, tmp_path):
        """Load from a config with specific special token values, save,
        and verify the saved config matches the original tokens."""
        save_dir = tmp_path / "saved"
        save_dir.mkdir()

        original_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "<SPECIAL_12>",
            "unk_token": "<unk>",
        }

        class _FakeSaveBase:
            def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
                config_path = os.path.join(save_directory, "tokenizer_config.json")
                with open(config_path, "w") as f:
                    json.dump(
                        {
                            "tokenizer_class": type(self).__name__,
                            "add_bos_token": True,
                            "add_eos_token": True,
                            "bos_token": {"content": "<s>", "lstrip": False},
                            "eos_token": {"content": "<SPECIAL_12>", "lstrip": False},
                            "unk_token": {"content": "<unk>", "lstrip": False},
                        },
                        f,
                    )

        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import (
            NeMoAutoTokenizerWithBosEosEnforced,
        )

        tok = _StubHFTokenizer()
        tok.bos_token = "<s>"
        tok.eos_token = "<SPECIAL_12>"
        tok._base_class = _FakeSaveBase
        tok._original_tokenizer_class = "PreTrainedTokenizerFast"
        tok._original_tokenizer_config = original_config

        wrapper_cls = type(
            NeMoAutoTokenizerWithBosEosEnforced.__name__,
            (NeMoAutoTokenizerWithBosEosEnforced, _FakeSaveBase),
            {},
        )
        tok.__class__ = wrapper_cls

        tok.save_pretrained(str(save_dir))

        with open(save_dir / "tokenizer_config.json") as f:
            result = json.load(f)

        assert result["bos_token"] == "<s>"
        assert result["eos_token"] == "<SPECIAL_12>"
        assert result["unk_token"] == "<unk>"
        assert "add_bos_token" not in result
        assert "add_eos_token" not in result

    def test_from_pretrained_stores_original_config(self, tmp_path):
        """from_pretrained should store the full original tokenizer_config
        on the tokenizer instance."""
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        src_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<s>",
            "eos_token": "<SPECIAL_12>",
            "unk_token": "<unk>",
            "model_max_length": 131072,
        }
        (src_dir / "tokenizer_config.json").write_text(json.dumps(src_config))

        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from:
            stub = _StubHFTokenizer()
            stub.bos_token = "<s>"
            stub.eos_token = "<SPECIAL_12>"
            mock_from.return_value = stub

            with patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()):
                tok = NeMoAutoTokenizer.from_pretrained(str(src_dir))

        assert tok._original_tokenizer_config == src_config
        assert tok._original_tokenizer_class == "PreTrainedTokenizerFast"


class TestTikTokenLikeTokenizerGuards:
    """Tests for the hasattr guards that prevent forcing add_bos/eos_token
    on tokenizers (like TikToken) that don't natively declare them."""

    def test_from_pretrained_skips_bos_eos_when_not_declared(self):
        """When a tokenizer lacks add_bos_token / add_eos_token attributes,
        from_pretrained should not create them."""
        stub = _TikTokenLikeTokenizer()
        assert not hasattr(stub, "add_bos_token")
        assert not hasattr(stub, "add_eos_token")
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert not hasattr(tok, "add_bos_token")
            assert not hasattr(tok, "add_eos_token")

    def test_from_pretrained_still_sets_bos_eos_when_declared(self):
        """When a tokenizer declares add_bos_token / add_eos_token,
        from_pretrained should set them as usual."""
        stub = _StubHFTokenizer()
        stub.bos_token = "<s>"
        stub.eos_token = "</s>"
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            assert tok.add_bos_token is True
            assert tok.add_eos_token is True

    def test_call_no_bos_eos_when_attributes_missing(self):
        """__call__ should not add BOS/EOS when add_bos_token / add_eos_token
        attributes are absent (getattr defaults to False)."""
        stub = _TikTokenLikeTokenizer()
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            out = tok(["x"])
            assert isinstance(out, BatchEncoding)
            # Should NOT have BOS/EOS prepended/appended
            assert out["input_ids"] == [[5, 6]]
            assert out["attention_mask"] == [[1, 1]]

    def test_encode_no_bos_eos_when_attributes_missing(self):
        """encode() should not add BOS/EOS when add_bos_token / add_eos_token
        attributes are absent (getattr defaults to False)."""
        stub = _TikTokenLikeTokenizer()
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=stub),
            patch("transformers.AutoConfig.from_pretrained", return_value=_StubConfig()),
        ):
            tok = NeMoAutoTokenizer.from_pretrained("dummy/model")
            enc = tok.encode("x")
            assert enc == [5, 6]


class TestTryConvertTikTokenToNative:
    """Tests for _try_convert_tiktoken_to_native."""

    def test_returns_fast_tokenizer_unchanged(self):
        """Fast tokenizers should be returned as-is."""
        stub = _StubHFTokenizer()
        stub.is_fast = True
        result = _try_convert_tiktoken_to_native(stub)
        assert result is stub

    def test_returns_non_tiktoken_slow_tokenizer_unchanged(self):
        """Slow tokenizers without vocab_file/special_tokens should be returned as-is."""
        stub = _StubHFTokenizer()
        stub.is_fast = False
        result = _try_convert_tiktoken_to_native(stub)
        assert result is stub

    def test_returns_original_on_conversion_failure(self):
        """If TikTokenConverter raises, the original tokenizer should be returned."""
        stub = _StubHFTokenizer()
        stub.is_fast = False
        stub.vocab_file = "/nonexistent/path.model"
        stub.special_tokens = {"<s>": 0}
        result = _try_convert_tiktoken_to_native(stub)
        assert result is stub

    def test_converts_tiktoken_tokenizer(self):
        """A tokenizer with vocab_file and special_tokens should be converted to fast."""
        stub = _StubHFTokenizer()
        stub.is_fast = False
        stub.vocab_file = "/fake/path.model"
        stub.special_tokens = {"<s>": 0, "</s>": 1}
        stub.pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+"
        stub.bos_token = "<s>"
        stub.eos_token = "</s>"
        stub.chat_template = "dummy template"

        mock_fast_backend = object()
        mock_fast_tokenizer = _StubHFTokenizer()
        mock_fast_tokenizer.is_fast = True

        with (
            patch("transformers.convert_slow_tokenizer.TikTokenConverter") as mock_converter_cls,
            patch(
                "transformers.tokenization_utils_tokenizers.TokenizersBackend",
                return_value=mock_fast_tokenizer,
            ) as mock_backend_cls,
        ):
            mock_converter_cls.return_value.converted.return_value = mock_fast_backend
            result = _try_convert_tiktoken_to_native(stub)

        assert result is mock_fast_tokenizer
        mock_converter_cls.assert_called_once_with(
            vocab_file="/fake/path.model",
            pattern=stub.pat_str,
            extra_special_tokens=stub.special_tokens,
        )
        mock_backend_cls.assert_called_once_with(
            tokenizer_object=mock_fast_backend,
            bos_token="<s>",
            eos_token="</s>",
            unk_token=None,
            pad_token=None,
        )
        assert result.chat_template == "dummy template"

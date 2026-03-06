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

import fnmatch
import json
import logging
import os
import shutil

from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)


_PRESERVED_SPECIAL_TOKEN_KEYS = frozenset(
    {
        "add_bos_token",
        "add_eos_token",
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
        "special_tokens_pattern",
    }
)


def _read_tokenizer_config(pretrained_model_name_or_path, **kwargs):
    """Read the full ``tokenizer_config.json`` as a dict.

    Works for local directories and HF Hub model IDs (cache lookup only).
    Returns ``None`` when the file cannot be read.
    """
    config_path = os.path.join(str(pretrained_model_name_or_path), "tokenizer_config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            return None
    try:
        from huggingface_hub import hf_hub_download

        hub_kwargs = {k: kwargs[k] for k in ("cache_dir", "revision", "token") if k in kwargs}
        resolved = hf_hub_download(
            repo_id=str(pretrained_model_name_or_path),
            filename="tokenizer_config.json",
            local_files_only=True,
            **hub_kwargs,
        )
        with open(resolved) as f:
            return json.load(f)
    except Exception:
        return None


def _read_tokenizer_class(pretrained_model_name_or_path, **kwargs):
    """Read the ``tokenizer_class`` value from an existing ``tokenizer_config.json``.

    Works for local directories and HF Hub model IDs (cache lookup only).
    Returns ``None`` when the value cannot be determined.
    """
    config = _read_tokenizer_config(pretrained_model_name_or_path, **kwargs)
    if config is None:
        return None
    return config.get("tokenizer_class")


def _resolve_source_dir(pretrained_model_name_or_path, **kwargs):
    """Return the local directory that contains the original tokenizer files.

    For local paths this is the path itself.  For Hub model IDs we look up
    the HF cache (``local_files_only``) so we never trigger a download.
    Returns ``None`` when the directory cannot be determined.
    """
    path = str(pretrained_model_name_or_path)
    if os.path.isdir(path):
        return path
    try:
        from huggingface_hub import hf_hub_download

        hub_kwargs = {k: kwargs[k] for k in ("cache_dir", "revision", "token") if k in kwargs}
        resolved = hf_hub_download(
            repo_id=path,
            filename="tokenizer_config.json",
            local_files_only=True,
            **hub_kwargs,
        )
        return os.path.dirname(resolved)
    except Exception:
        return None


# File patterns that should be preserved from the original source directory
# during ``save_pretrained``.  If a matching file exists in the source, it is
# copied verbatim into the save directory so that downstream v4 consumers see
# exactly the same assets the original model shipped.
#
# NOTE: ``config.json`` is the *model* config, not a tokenizer file.  It is
# included here as a workaround for a transformers v5 behaviour where
# ``AutoTokenizer.from_pretrained`` uses ``config.json`` → ``model_type`` to
# select the tokenizer backend.  For SentencePiece-based models (Llama, Phi-3,
# Mistral, …) the backend initialisation sets up a whitespace normalizer
# (Prepend("▁") + Replace(" ", "▁")).  Without ``config.json`` the loader
# falls back to a generic backend that **ignores** the normalizer already
# present in ``tokenizer.json``, producing different token IDs for whitespace
# inputs.  Copying ``config.json`` is the pragmatic fix until upstream
# transformers honours the normalizer stored in the tokenizer file.
_TOKENIZER_FILE_PATTERNS = (
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.*",
    "merges.txt",
    "spiece.model",
    "config.json",
)


def _is_tokenizer_file(filename):
    """Return True if *filename* matches a known tokenizer file pattern."""
    return any(fnmatch.fnmatch(filename, pat) for pat in _TOKENIZER_FILE_PATTERNS)


def _restore_original_assets(save_directory, source_dir, original_config):
    """Restore original tokenizer assets that were present in the source.

    For every tokenizer-related file found in *source_dir*, copy it into
    *save_directory* (overwriting the v5-written version if one exists).
    Files that were **not** present in the source are left untouched, so
    anything ``save_pretrained`` generated is kept as a fallback.

    This ensures byte-for-byte v4 compatibility for all files the original
    model shipped, while still benefiting from v5-generated files (e.g.
    ``chat_template.jinja``) that the original may not have had.
    """
    save_directory = str(save_directory)
    source_dir = str(source_dir)

    saved_config_path = os.path.join(save_directory, "tokenizer_config.json")
    v5_tokenizer_class = None
    if os.path.isfile(saved_config_path):
        try:
            with open(saved_config_path) as f:
                v5_tokenizer_class = json.load(f).get("tokenizer_class")
        except Exception:
            pass

    for fname in os.listdir(source_dir):
        src = os.path.join(source_dir, fname)
        if not os.path.isfile(src) or not _is_tokenizer_file(fname):
            continue
        shutil.copy2(src, os.path.join(save_directory, fname))

    _ensure_tokenizer_class(save_directory, original_config, v5_tokenizer_class)


def _ensure_tokenizer_class(save_directory, original_config, v5_tokenizer_class):
    """Make sure ``tokenizer_config.json`` contains a ``tokenizer_class`` field.

    Some older HF models (GPT-2, BERT, etc.) ship without ``tokenizer_class``
    in their original config.  After restoring the original file we inject the
    value that the v5 save produced so v4 consumers can identify the class.
    """
    config_path = os.path.join(str(save_directory), "tokenizer_config.json")
    if not os.path.isfile(config_path):
        if original_config:
            _restore_tokenizer_config(save_directory, original_config)
        return
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        if "tokenizer_class" in cfg:
            return
        tokenizer_class = v5_tokenizer_class
        if tokenizer_class:
            cfg["tokenizer_class"] = tokenizer_class
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
                f.write("\n")
    except Exception:
        pass


def _restore_tokenizer_config(save_directory, original_config):
    """Replace the saved ``tokenizer_config.json`` with the original version.

    Transformers v5 may alter many fields during ``save_pretrained`` — special
    tokens, ``added_tokens_decoder``, ``chat_template``, backend markers, etc.
    Restoring the original config wholesale is the safest way to guarantee that
    downstream v4 consumers see the exact same configuration.

    The only field carried over from the v5-written config is
    ``tokenizer_class`` when it was not present in the original, so the saved
    config always advertises a loadable class name.
    """
    config_path = os.path.join(str(save_directory), "tokenizer_config.json")
    if not os.path.isfile(config_path):
        return
    try:
        with open(config_path) as f:
            saved_config = json.load(f)

        restored = dict(original_config)

        if "tokenizer_class" not in restored and "tokenizer_class" in saved_config:
            restored["tokenizer_class"] = saved_config["tokenizer_class"]

        if restored != saved_config:
            with open(config_path, "w") as f:
                json.dump(restored, f, indent=2, ensure_ascii=False)
                f.write("\n")
    except Exception:
        pass


def _remap_system_role(conversation):
    """Merge a single system message into the first user message.

    If the template's Jinja raises ``TemplateError("System role not supported")``,
    this helper folds the system content into the first ``user`` turn so the
    template can render without error.

    Raises ``ValueError`` when more than one system message is present (ambiguous).
    """
    system_msgs = [m for m in conversation if isinstance(m, dict) and m.get("role") == "system"]
    if len(system_msgs) > 1:
        raise ValueError("System role appeared in multiple messages. Only a single system message is supported.")

    system_content = system_msgs[0].get("content", "")
    remapped = []
    merged = False
    for m in conversation:
        if isinstance(m, dict) and m.get("role") == "system":
            continue
        if not merged and isinstance(m, dict) and m.get("role") == "user":
            remapped.append({**m, "content": f"{system_content}\n{m['content']}"})
            merged = True
        else:
            remapped.append(m)
    if not merged:
        remapped.insert(0, {"role": "user", "content": system_content})
    return remapped


class NeMoAutoTokenizerWithBosEosEnforced(AutoTokenizer):
    """
    A wrapper around HuggingFace's AutoTokenizer that ensures consistent BOS/EOS token handling.

    There are pre-existing issues with some tokenizers (e.g. GPT2Tokenizer) where the BOS/EOS tokens
    are not added automatically. This wrapper ensures they are always added when requested.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, add_bos_token=True, add_eos_token=True, **kwargs):
        """
        Load the HF tokenizer class via AutoTokenizer and (optionally) wrap it to add BOS/EOS.

        Args:
            pretrained_model_name_or_path: Model identifier or path
            add_bos_token: Whether to add BOS token (default: True)
            add_eos_token: Whether to add EOS token (default: True)
        """
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Transformers >=5.0.0 defaults special_tokens_pattern to "cls_sep", which inserts
        # cls_token_id / sep_token_id into input_ids via build_inputs_with_special_tokens.
        # Moonlight's TikTokenTokenizer doesn't define CLS/SEP, so those IDs are None,
        # resulting in None values in input_ids and a downstream ValueError in pad().
        # Fix: when the pattern is "cls_sep" but the required tokens are missing, fall
        # back to "none" so build_inputs_with_special_tokens passes through unchanged.
        if getattr(tokenizer, "special_tokens_pattern", None) == "cls_sep" and (
            getattr(tokenizer, "cls_token_id", None) is None or getattr(tokenizer, "sep_token_id", None) is None
        ):
            tokenizer.special_tokens_pattern = "none"

        if add_bos_token and getattr(tokenizer, "bos_token", None) is not None:
            try:
                tokenizer.add_bos_token = add_bos_token
            except ValueError:
                tokenizer._add_bos_token = add_bos_token
        if add_eos_token and getattr(tokenizer, "eos_token", None) is not None:
            try:
                tokenizer.add_eos_token = add_eos_token
            except ValueError:
                tokenizer._add_eos_token = add_eos_token
        # Keep the wrapper class name at runtime, but remember the original HF tokenizer class
        # so we can save an HF-compatible `tokenizer_class` in `save_pretrained()`.
        base_tokenizer_cls = type(tokenizer)
        tokenizer._base_class = base_tokenizer_cls
        tokenizer._original_tokenizer_config = _read_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        tokenizer._original_tokenizer_class = (tokenizer._original_tokenizer_config or {}).get("tokenizer_class")

        tokenizer._source_dir = _resolve_source_dir(pretrained_model_name_or_path, **kwargs)

        tokenizer.__class__ = type(cls.__name__, (cls, base_tokenizer_cls), {})
        return tokenizer

    def apply_chat_template(self, conversation, *args, **kwargs):
        has_system = any(isinstance(m, dict) and m.get("role") == "system" for m in conversation)
        if not has_system:
            return super().apply_chat_template(conversation, *args, **kwargs)

        system_msgs = [m for m in conversation if isinstance(m, dict) and m.get("role") == "system"]
        if len(system_msgs) > 1:
            raise ValueError("System role appeared in multiple messages. Only a single system message is supported.")

        try:
            return super().apply_chat_template(conversation, *args, **kwargs)
        except TemplateError as e:
            if "System role not supported" not in str(e):
                raise
            logger.debug("Chat template does not support system role; merging into first user message.")
            remapped = _remap_system_role(conversation)
            return super().apply_chat_template(remapped, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        tokenized = super().__call__(*args, **kwargs)
        if not kwargs.get("add_special_tokens", True):
            return tokenized
        if isinstance(tokenized, BatchEncoding):
            _tokenized_keys = {"input_ids", "attention_mask", "assistant_masks"}
            add_bos_ids = self.add_bos_token and (getattr(self, "bos_token_id", None) is not None)
            add_eos_ids = self.add_eos_token and (getattr(self, "eos_token_id", None) is not None)
            if not "input_ids" in tokenized:
                return tokenized
            if add_bos_ids:
                add_bos_ids = _add_token(tokenized, self.bos_token_id, 0, "input_ids")
            if add_eos_ids:
                add_eos_ids = _add_token(tokenized, self.eos_token_id, -1, "input_ids")

            for key in {"attention_mask", "assistant_masks"}:
                if key not in tokenized:
                    continue
                if add_bos_ids:
                    _add_token(tokenized, 1, 0, key)
                if add_eos_ids:
                    _add_token(tokenized, 1, -1, key)
        return tokenized

    def encode(self, *args, **kwargs):
        encoded = super().encode(*args, **kwargs)
        if not kwargs.get("add_special_tokens", True):
            return encoded
        if self.add_bos_token:
            if encoded and (getattr(self, "bos_token_id", None) is not None) and encoded[0] != self.bos_token_id:
                encoded = [self.bos_token_id] + encoded
        if self.add_eos_token:
            if encoded and (getattr(self, "eos_token_id", None) is not None) and encoded[-1] != self.eos_token_id:
                encoded = encoded + [self.eos_token_id]
        return encoded

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        # HF writes ``tokenizer_class`` using ``self.__class__.__name__``.
        # In transformers v5 the runtime class is ``TokenizersBackend``, but
        # downstream v4 consumers still need the original class name (e.g.
        # ``PreTrainedTokenizerFast``).  We temporarily swap ``self.__class__``
        # to a dynamic subclass whose ``__name__`` matches the original value.
        base_class = getattr(self, "_base_class", None)
        if not base_class:
            return super().save_pretrained(save_directory, push_to_hub=push_to_hub, **kwargs)

        original_tokenizer_class = getattr(self, "_original_tokenizer_class", None)
        save_cls_name = original_tokenizer_class or base_class.__name__

        if save_cls_name != base_class.__name__:
            save_class = type(save_cls_name, (base_class,), {})
        else:
            save_class = base_class

        original_cls = self.__class__
        try:
            self.__class__ = save_class
            result = save_class.save_pretrained(self, save_directory, push_to_hub=push_to_hub, **kwargs)
        finally:
            self.__class__ = original_cls

        source_dir = getattr(self, "_source_dir", None)
        original_config = getattr(self, "_original_tokenizer_config", None)

        if source_dir:
            _restore_original_assets(save_directory, source_dir, original_config)
        elif original_config:
            _restore_tokenizer_config(save_directory, original_config)

        return result


def _add_token(tokenized, value, position, key):
    def _extend_single(sequence, val, pos, always_add):
        if pos == 0:
            if always_add or not sequence or sequence[0] != val:
                return [val] + sequence, True
            return sequence, False
        if pos == -1:
            if always_add or not sequence or sequence[-1] != val:
                return sequence + [val], True
            return sequence, False
        raise ValueError(f"Invalid position: {pos}")

    sequences = tokenized[key]
    always_add = key != "input_ids"
    if isinstance(sequences, list) and sequences and isinstance(sequences[0], list):
        ans = [_extend_single(seq, value, position, always_add) for seq in sequences]
        tokenized[key] = list(map(lambda x: x[0], ans))
        return any(map(lambda x: x[1], ans))
    elif isinstance(sequences, list):
        ans = _extend_single(sequences, value, position, always_add)
        tokenized[key] = ans[0]
        return ans[1]
    else:
        raise ValueError(f"Invalid sequence type: {type(sequences)}")
    return False

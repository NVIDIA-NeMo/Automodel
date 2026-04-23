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

from nemo_automodel.components.distributed.utils import FirstRankPerNode

try:
    from huggingface_hub.errors import StrictDataclassClassValidationError
except ImportError:
    StrictDataclassClassValidationError = ValueError

logger = logging.getLogger(__name__)


def _ensure_pad_token_id(tokenizer, model_name: str) -> None:
    """Default pad_token_id to 0 when the tokenizer does not define one."""
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = 0
        logger.warning(
            "Tokenizer '%s' has pad_token_id=None; defaulting to 0. "
            "This can cause incorrect MoE auxiliary loss calculations if valid tokens "
            "share token ID 0. Set pad_token_id explicitly in your tokenizer config to avoid this.",
            model_name,
        )


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
# NOTE: ``config.json`` (the *model* config) is intentionally excluded.
# While transformers v5 uses ``model_type`` from ``config.json`` to select the
# tokenizer backend and SentencePiece normalizer, copying it here would be
# fragile — the user may have modified the model, making the original
# ``config.json`` stale or incorrect.  The caller is responsible for ensuring
# ``config.json`` is present alongside the saved tokenizer if needed.
_TOKENIZER_FILE_PATTERNS = (
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.*",
    "merges.txt",
    "spiece.model",
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


def _try_convert_tiktoken_to_native(tokenizer):
    """Convert a TikToken-based tokenizer to a native HF tokenizer.

    This enables char_to_token() for {% generation %} mask computation
    without needing a heuristic fallback for char_to_token().
    Returns the original tokenizer unchanged if conversion is not applicable.
    """
    if getattr(tokenizer, "is_fast", True):
        return tokenizer
    if not (hasattr(tokenizer, "vocab_file") and hasattr(tokenizer, "special_tokens")):
        return tokenizer

    try:
        from transformers.convert_slow_tokenizer import TikTokenConverter
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        fast_backend = TikTokenConverter(
            vocab_file=tokenizer.vocab_file,
            pattern=getattr(tokenizer, "pat_str", None),
            extra_special_tokens=tokenizer.special_tokens,
        ).converted()

        fast = TokenizersBackend(
            tokenizer_object=fast_backend,
            bos_token=getattr(tokenizer, "bos_token", None),
            eos_token=getattr(tokenizer, "eos_token", None),
            unk_token=getattr(tokenizer, "unk_token", None),
            pad_token=getattr(tokenizer, "pad_token", None),
        )

        # Carry over chat template and any other custom attributes.
        if hasattr(tokenizer, "chat_template"):
            fast.chat_template = tokenizer.chat_template

        logger.info("Converted TikToken tokenizer to fast backend for char_to_token() support.")
        return fast
    except Exception:
        logger.debug("TikToken-to-fast conversion failed, keeping original tokenizer.", exc_info=True)
        return tokenizer


def _prefetch_tokenizer_assets(pretrained_model_name_or_path, **kwargs):
    """Best-effort pre-fetch of tokenizer assets into the HF cache.

    When transformers' cached_file lookup fails silently (e.g. transient Hub
    outage or rate-limit), PreTrainedTokenizerFast.__init__ raises an opaque
    "Couldn't instantiate the backend tokenizer" error. Running snapshot_download
    first populates the cache so the subsequent AutoTokenizer.from_pretrained
    call finds tokenizer.json locally. No-op for local paths.
    """
    path = str(pretrained_model_name_or_path)
    if os.path.isdir(path):
        return
    try:
        from huggingface_hub import snapshot_download

        hub_kwargs = {k: kwargs[k] for k in ("cache_dir", "revision", "token") if k in kwargs}
        snapshot_download(
            path,
            allow_patterns=[
                "tokenizer*",
                "vocab*",
                "merges*",
                "spiece.model",
                "tiktoken.model",
                "special_tokens_map.json",
                "added_tokens.json",
                "chat_template*",
            ],
            **hub_kwargs,
        )
    except Exception as e:
        logger.debug("Tokenizer asset pre-fetch failed (non-fatal): %s", e)


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
        # Serialize concurrent downloads across ranks + pre-fetch tokenizer assets.
        #
        # transformers 5.5 silently swallows list_repo_files failures (try/except in
        # tokenization_utils_base.py:_from_pretrained) and cached_file returns None
        # when called with _raise_exceptions_for_missing_entries=False. The combo
        # produces an opaque "Couldn't instantiate the backend tokenizer" ValueError
        # from PreTrainedTokenizerFast.__init__ — seen on CI nodes where HF Hub
        # isn't reliably reachable and the tokenizer isn't in the shared cache.
        #
        # We pre-populate the cache on rank 0 via snapshot_download with tokenizer-
        # only patterns so subsequent cached_file lookups hit the cache regardless
        # of later list_repo_files behavior. FirstRankPerNode also guards the
        # remote-code copy race on shared filesystems (see #1840).
        with FirstRankPerNode():
            _prefetch_tokenizer_assets(pretrained_model_name_or_path, **kwargs)
            try:
                tokenizer = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            except (ValueError, StrictDataclassClassValidationError) as e:
                err = str(e)
                if "num_hidden_layers" in err and ("layer_types" in err or "layer types" in err):
                    # AutoTokenizer.from_pretrained internally calls AutoConfig.from_pretrained,
                    # so configs whose layer_types length differs from num_hidden_layers (e.g.
                    # stepfun-ai/Step-3.5-Flash) trip validate_layer_type before the tokenizer
                    # is built. The tokenizer itself doesn't depend on layer_types, so relax
                    # the validator globally and retry.
                    from nemo_automodel._transformers.v4_patches.layer_types import relax_layer_types_validator

                    relax_layer_types_validator()
                    tokenizer = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                elif "Couldn't instantiate the backend tokenizer" in err and kwargs.get("use_fast") is not False:
                    # transformers 5.5's PreTrainedTokenizerFast.__init__ can fail when no
                    # tokenizer.json is present and no slow-to-fast conversion path is
                    # viable. Fall back to the slow tokenizer; _try_convert_tiktoken_to_native
                    # below will still promote it to fast when vocab_file + pat_str are set.
                    logger.warning(
                        "Fast tokenizer init failed for %s (%s); retrying with use_fast=False.",
                        pretrained_model_name_or_path,
                        type(e).__name__,
                    )
                    tokenizer = super().from_pretrained(
                        pretrained_model_name_or_path, *args, **{**kwargs, "use_fast": False}
                    )
                else:
                    raise

        # Convert TikToken-based tokenizers to fast (Rust-backed) tokenizers so that
        # char_to_token() works natively for {% generation %} mask computation.
        tokenizer = _try_convert_tiktoken_to_native(tokenizer)

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

        # Only set add_bos/eos_token if the tokenizer already declares the attribute.
        # Forcing them on tokenizers that don't natively support them (e.g. TikToken)
        # causes spurious BOS/EOS insertion around every text segment.
        if add_bos_token and getattr(tokenizer, "bos_token", None) is not None:
            if hasattr(tokenizer, "add_bos_token"):
                try:
                    tokenizer.add_bos_token = add_bos_token
                except ValueError:
                    tokenizer._add_bos_token = add_bos_token
        if add_eos_token and getattr(tokenizer, "eos_token", None) is not None:
            if hasattr(tokenizer, "add_eos_token"):
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

        _ensure_pad_token_id(tokenizer, pretrained_model_name_or_path)

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
            add_bos_ids = getattr(self, "add_bos_token", False) and (getattr(self, "bos_token_id", None) is not None)
            add_eos_ids = getattr(self, "add_eos_token", False) and (getattr(self, "eos_token_id", None) is not None)
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
        if getattr(self, "add_bos_token", False):
            if encoded and (getattr(self, "bos_token_id", None) is not None) and encoded[0] != self.bos_token_id:
                encoded = [self.bos_token_id] + encoded
        if getattr(self, "add_eos_token", False):
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

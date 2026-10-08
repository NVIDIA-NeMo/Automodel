# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Optional gigatoken tokenizer backend (issue #3177).

`gigatoken <https://github.com/marcelroed/gigatoken>`_ is a Rust BPE tokenizer
~1000x faster than the HuggingFace ``tokenizers`` library. Its ``.as_hf()``
compatibility mode returns an HF-style tokenizer (``__call__``, ``encode``,
``decode``, special tokens) whose token IDs are identical to the source HF
tokenizer, so it can accelerate the plain-text tokenization path.

This backend is optional and additive: it is a no-op when ``gigatoken`` is not
installed, and it is opt-in via
``NeMoAutoTokenizer.from_pretrained(..., use_gigatoken=True)``. Note that gigatoken
only supports BPE tokenizers (not SentencePiece or WordPiece), and its ``.as_hf()``
does not implement ``apply_chat_template`` or ``pad``; :class:`GigatokenTokenizer`
delegates those to the wrapped HF tokenizer.
"""

import logging
from typing import Any

from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

HAVE_GIGATOKEN, gigatoken = safe_import("gigatoken")


def is_available() -> bool:
    """Return whether the optional ``gigatoken`` package is importable."""
    return HAVE_GIGATOKEN


def build_gigatoken_tokenizer(hf_tokenizer: Any) -> Any | None:
    """Wrap a HuggingFace BPE tokenizer with the gigatoken fast backend.

    Args:
        hf_tokenizer: A loaded HuggingFace tokenizer (BPE) to accelerate.

    Returns:
        An HF-compatible tokenizer backed by gigatoken (``gt.Tokenizer(...).as_hf()``),
        or ``None`` if gigatoken is unavailable, so the caller can fall back to
        ``hf_tokenizer``.
    """
    if not HAVE_GIGATOKEN:
        return None
    return gigatoken.Tokenizer(hf_tokenizer).as_hf()


class GigatokenTokenizer:
    """Delegate to an HF tokenizer, but encode text with the faster gigatoken backend.

    ``encode`` and ``__call__`` run through gigatoken (identical token IDs, much
    faster); every other attribute -- ``apply_chat_template``, ``pad``, ``decode``,
    special tokens, config -- falls through to the wrapped HF tokenizer via
    ``__getattr__``, so chat/SFT flows keep working unchanged.

    When the wrapped tokenizer has opted into BOS/EOS enforcement
    (``add_bos_token``/``add_eos_token``), ``encode``/``__call__`` delegate to the HF
    tokenizer instead, since gigatoken is built from the raw HF tokenizer and has no
    knowledge of that Python-level post-processing.
    """

    def __init__(self, hf_tokenizer: Any, gigatoken_tokenizer: Any) -> None:
        """Store the HF tokenizer and its gigatoken-backed fast encoder.

        Args:
            hf_tokenizer: The wrapped HuggingFace tokenizer (source of chat templates,
                padding, special tokens, and config).
            gigatoken_tokenizer: The gigatoken ``.as_hf()`` fast encoder.
        """
        self._hf = hf_tokenizer
        self._gt = gigatoken_tokenizer

    def _enforces_special_tokens(self, kwargs: dict) -> bool:
        """Return True when the HF tokenizer would add BOS/EOS for this call."""
        if not kwargs.get("add_special_tokens", True):
            return False
        return bool(getattr(self._hf, "add_bos_token", False) or getattr(self._hf, "add_eos_token", False))

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        """Encode text to token IDs using gigatoken (or HF when BOS/EOS is enforced)."""
        if self._enforces_special_tokens(kwargs):
            return self._hf.encode(*args, **kwargs)
        return self._gt.encode(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Tokenize text using gigatoken (or HF when BOS/EOS is enforced)."""
        if self._enforces_special_tokens(kwargs):
            return self._hf(*args, **kwargs)
        return self._gt(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward any non-overridden attribute to the wrapped HF tokenizer."""
        # _hf/_gt are set in __init__; guard against recursion before then.
        if name in ("_hf", "_gt"):
            raise AttributeError(name)
        return getattr(self._hf, name)

    def __len__(self) -> int:
        """Forward ``len(tokenizer)`` to the HF tokenizer (dunders skip __getattr__)."""
        return len(self._hf)

    def __contains__(self, item: Any) -> bool:
        """Forward ``token in tokenizer`` to the HF tokenizer."""
        return item in self._hf

    def __getitem__(self, key: Any) -> Any:
        """Forward ``tokenizer[key]`` to the HF tokenizer."""
        return self._hf[key]


def maybe_wrap_with_gigatoken(hf_tokenizer: Any) -> Any:
    """Wrap an HF tokenizer with gigatoken when possible, else return it unchanged.

    Falls back to ``hf_tokenizer`` (with a warning) when gigatoken is not installed or
    the tokenizer is not a supported BPE tokenizer.

    Args:
        hf_tokenizer: A loaded HuggingFace tokenizer to accelerate.

    Returns:
        A :class:`GigatokenTokenizer` wrapping ``hf_tokenizer``, or ``hf_tokenizer``
        unchanged when gigatoken is unavailable or cannot wrap it.
    """
    if not HAVE_GIGATOKEN:
        logger.warning("use_gigatoken=True but gigatoken is not installed; using the HF tokenizer.")
        return hf_tokenizer
    try:
        gt = build_gigatoken_tokenizer(hf_tokenizer)
    except Exception as e:
        logger.warning("gigatoken could not wrap this tokenizer (%s); using the HF tokenizer.", e)
        return hf_tokenizer
    return GigatokenTokenizer(hf_tokenizer, gt) if gt is not None else hf_tokenizer

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
installed and is not yet wired into :class:`NeMoAutoTokenizer` dispatch. Note that
gigatoken only supports BPE tokenizers (not SentencePiece or WordPiece), and its
``.as_hf()`` does not implement ``apply_chat_template``.
"""

from typing import Any

from nemo_automodel.shared.import_utils import safe_import

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

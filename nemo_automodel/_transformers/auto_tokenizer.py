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

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


class NeMoAutoTokenizer:
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *args, force_hf=False, add_bos_token=True, add_eos_token=True, **kwargs
    ):
        """
        Load the HF tokenizer class via AutoTokenizer and (optionally) wrap it to add BOS/EOS.

        There are pre-existing issues with some tokenizers (e.g. GPT2Tokenizer) where the BOS/EOS tokens are not added
        """
        hf_tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if force_hf:
            return hf_tok

        return cls(hf_tok, add_bos_token=add_bos_token, add_eos_token=add_eos_token)

    def __init__(self, base_tokenizer, *, add_bos_token: bool, add_eos_token: bool):
        self._base = base_tokenizer
        self._add_bos = bool(add_bos_token)
        self._add_eos = bool(add_eos_token)

    @property
    def add_bos_token(self):
        return self._add_bos

    @property
    def add_eos_token(self):
        return self._add_eos

    def __getattr__(self, name):
        # Delegate everything else to the underlying tokenizer
        return getattr(self._base, name)

    def __call__(self, *args, **kwargs):
        tokenized = self._base(*args, **kwargs)
        if not kwargs.get("add_special_tokens", True):
            return tokenized
        if isinstance(tokenized, BatchEncoding):
            _tokenized_keys = {"input_ids", "attention_mask", "assistant_masks"}
            add_bos_ids = self._add_bos and (getattr(self, "bos_token_id", None) is not None)
            add_eos_ids = self._add_eos and (getattr(self, "eos_token_id", None) is not None)
            for key in _tokenized_keys:
                if key not in tokenized:
                    continue
                if key == "input_ids":
                    if add_bos_ids:
                        _add_token(tokenized, self.bos_token_id, 0, key)
                    if add_eos_ids:
                        _add_token(tokenized, self.eos_token_id, -1, key)
                else:
                    if add_bos_ids:
                        _add_token(tokenized, 1, 0, key)
                    if add_eos_ids:
                        _add_token(tokenized, 1, -1, key)
        return tokenized

    def encode(self, *args, **kwargs):
        encoded = self._base.encode(*args, **kwargs)
        if not kwargs.get("add_special_tokens", True):
            return encoded
        if self._add_bos:
            if encoded and (getattr(self, "bos_token_id", None) is not None) and encoded[0] != self.bos_token_id:
                encoded = [self.bos_token_id] + encoded
        if self._add_eos:
            if encoded and (getattr(self, "eos_token_id", None) is not None) and encoded[-1] != self.eos_token_id:
                encoded = encoded + [self.eos_token_id]
        return encoded


def _add_token(tokenized, value, position, key):
    def _extend_single(sequence, val, pos, always_add):
        if pos == 0:
            if always_add or not sequence or sequence[0] != val:
                return [val] + sequence
            return sequence
        if pos == -1:
            if always_add or not sequence or sequence[-1] != val:
                return sequence + [val]
            return sequence
        raise ValueError(f"Invalid position: {pos}")

    sequences = tokenized[key]
    always_add = key != "input_ids"
    if isinstance(sequences, list) and sequences and isinstance(sequences[0], list):
        tokenized[key] = [_extend_single(seq, value, position, always_add) for seq in sequences]
    elif isinstance(sequences, list):
        tokenized[key] = _extend_single(sequences, value, position, always_add)
    else:
        raise ValueError(f"Invalid sequence type: {type(sequences)}")
    return tokenized
